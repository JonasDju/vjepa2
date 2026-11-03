# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient accumulation in ``app/vjepa_2_1/train.py`` (``optimization.grad_accum_steps``).

The training loop lives inline in ``main``; these tests drive the helpers it is built from, in the order the loop
calls them. Per micro-step, the forward and ``(loss / K).backward()`` run inside ``grad_sync(modules,
micro_step_syncs(...))``. Once per optimizer step, ``optimizer_step`` and then ``ema_update`` run.

- Two CPU (gloo) ranks with K micro-batches each must give what one process gives on the whole global batch:
  the same gradient, grad norm (so clipping sees the global-batch gradient), online and EMA weights. The test
  models mirror the real DDP wrappers: a ``static_graph=True`` encoder and a ``find_unused_parameters=True``
  predictor with a parameter that is never used.
- ``no_sync`` on the first backward of a ``static_graph=True`` DDP fails. That is why ``micro_step_syncs`` syncs
  every micro-step of the first optimizer step. If that test starts failing after a torch upgrade, the rule may
  no longer be needed.
- The EMA runs once per optimizer step, the fp16 ``GradScaler`` path matches fp32, and ``grad_accum_steps`` is
  validated.
"""

import copy
import datetime
import json
import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_2_1.train import (
    check_grad_accum_steps,
    ema_update,
    grad_sync,
    micro_step_syncs,
    optimizer_step,
)

WORLD_SIZE = 2
K = 4  # micro-steps per optimizer step
MICRO_BATCH = 3
N_STEPS = 3
MOMENTUM = 0.9
CLIP = 0.05  # small enough that the clipping is active


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 8))

    def forward(self, x):
        return self.net(x)


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4)
        self.unused = nn.Linear(8, 4)  # like the predictor's parameters a video-only run never uses

    def forward(self, z):
        return self.proj(z)


def _models():
    torch.manual_seed(0)
    encoder, predictor = ToyEncoder().double(), ToyPredictor().double()
    target = copy.deepcopy(encoder)
    for p in target.parameters():
        p.requires_grad = False
    return encoder, predictor, target


def _loss(encoder, predictor, target, x):
    with torch.no_grad():
        h = target(x)[:, :4]
    return ((predictor(encoder(x)) - h) ** 2).mean()


def _global_batch(step):
    """The step's global batch, ``WORLD_SIZE * K * MICRO_BATCH`` samples."""
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randn(WORLD_SIZE * K * MICRO_BATCH, 6, generator=g, dtype=torch.float64)


def _micro_batch(x, rank, micro_itr):
    i = (rank * K + micro_itr) * MICRO_BATCH
    return x[i : i + MICRO_BATCH]


def _grads(*modules):
    return [p.grad.detach().clone() for m in modules for p in m.parameters() if p.grad is not None]


def _flat(*modules):
    return torch.cat([p.detach().flatten() for m in modules for p in m.parameters()])


def _optimizer(encoder, predictor):
    return torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2)


def _init_group(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=60))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _equivalence_worker(rank, port):
    _init_group(rank, port)
    try:
        # the reference: one process, the whole global batch per step
        ref_enc, ref_pred, ref_target = _models()
        ref_opt = _optimizer(ref_enc, ref_pred)

        enc, pred, target = _models()
        opt = _optimizer(enc, pred)
        enc = DistributedDataParallel(enc, static_graph=True)
        pred = DistributedDataParallel(pred, static_graph=False, find_unused_parameters=True)

        first_step = True
        for step in range(N_STEPS):
            x = _global_batch(step)

            _loss(ref_enc, ref_pred, ref_target, x).backward()
            ref_grads = _grads(ref_enc, ref_pred)
            ref_finite, ref_clipped, ref_norm = optimizer_step(
                ref_opt, None, list(ref_enc.parameters()) + list(ref_pred.parameters()), CLIP
            )
            ema_update(ref_enc, ref_target, MOMENTUM)

            for micro_itr in range(K):
                with grad_sync([enc, pred], micro_step_syncs(micro_itr, K, first_step)):
                    (_loss(enc, pred, target, _micro_batch(x, rank, micro_itr)) / K).backward()
            first_step = False
            grads = _grads(enc.module, pred.module)
            finite, clipped, norm = optimizer_step(
                opt, None, list(enc.parameters()) + list(pred.parameters()), CLIP
            )
            ema_update(enc, target, MOMENTUM)

            assert len(grads) == len(ref_grads), (len(grads), len(ref_grads))
            for g, rg in zip(grads, ref_grads):
                torch.testing.assert_close(g, rg, rtol=1e-10, atol=1e-12)
            assert finite and ref_finite and clipped and ref_clipped, (finite, ref_finite, clipped, ref_clipped)
            assert abs(norm - ref_norm) <= 1e-10 * ref_norm, (step, norm, ref_norm)
            torch.testing.assert_close(_flat(enc.module, pred.module), _flat(ref_enc, ref_pred), rtol=1e-9, atol=1e-11)
            torch.testing.assert_close(_flat(target), _flat(ref_target), rtol=1e-9, atol=1e-11)

            # the ranks hold identical online and target weights
            mine = _flat(enc.module, pred.module, target)
            gathered = [torch.zeros_like(mine) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered, mine)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _static_graph_trap_worker(rank, port, out_dir):
    """``no_sync`` on the very first backward of a ``static_graph=True`` DDP (no first-step rule)."""
    _init_group(rank, port)
    result = "ok"
    try:
        enc, _, _ = _models()
        enc = DistributedDataParallel(enc, static_graph=True)
        x = _global_batch(0)
        try:
            for micro_itr in range(K):
                with grad_sync([enc], micro_itr == K - 1):
                    (enc(_micro_batch(x, rank, micro_itr)).pow(2).mean() / K).backward()
            g = torch.cat([p.grad.flatten() for p in enc.parameters()])
            gathered = [torch.zeros_like(g) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered, g)
            if not torch.equal(gathered[0], gathered[1]):
                result = "diverged"
        except RuntimeError as e:
            result = f"raised: {str(e).splitlines()[0]}"
    finally:
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
            json.dump(result, f)
    # After the failed backward the reducer is broken, and destroy_process_group() can hang waiting for the other
    # rank (a race; it hung a full-suite run). The result is on disk, so leave without any teardown.
    os._exit(0)


class DistributedAccumulationTest(unittest.TestCase):
    def test_matches_the_global_batch(self):
        mp.spawn(_equivalence_worker, args=(_free_port(),), nprocs=WORLD_SIZE, join=True)

    def test_no_sync_on_the_first_static_graph_backward_fails(self):
        with tempfile.TemporaryDirectory() as out_dir:
            mp.spawn(_static_graph_trap_worker, args=(_free_port(), out_dir), nprocs=WORLD_SIZE, join=True)
            results = []
            for rank in range(WORLD_SIZE):
                with open(os.path.join(out_dir, f"rank{rank}.json")) as f:
                    results.append(json.load(f))
        self.assertTrue(
            any(r != "ok" for r in results),
            "no_sync on the first static-graph backward now works (torch upgrade?): "
            f"micro_step_syncs' first-step rule may be unnecessary. Results: {results}",
        )


class MicroStepSyncsTest(unittest.TestCase):
    def test_only_the_last_micro_step_syncs(self):
        self.assertEqual([micro_step_syncs(i, 4, False) for i in range(4)], [False, False, False, True])

    def test_the_first_step_syncs_every_micro_step(self):
        self.assertEqual([micro_step_syncs(i, 4, True) for i in range(4)], [True] * 4)

    def test_no_accumulation_always_syncs(self):
        self.assertTrue(micro_step_syncs(0, 1, False))

    def test_grad_sync_without_no_sync_is_a_no_op(self):
        with grad_sync([object()], True):  # never touches the modules when syncing
            pass


class EmaOncePerStepTest(unittest.TestCase):
    def test_target_moves_once_towards_the_stepped_weights(self):
        enc, pred, target = _models()
        opt = torch.optim.SGD(list(enc.parameters()) + list(pred.parameters()), lr=0.1)
        target_before = _flat(target)
        x = _global_batch(0)
        for micro_itr in range(K):
            (_loss(enc, pred, target, _micro_batch(x, 0, micro_itr)) / K).backward()
        self.assertTrue(torch.equal(_flat(target), target_before))  # micro-steps never touch the target
        optimizer_step(opt, None, list(enc.parameters()) + list(pred.parameters()), None)
        ema_update(enc, target, MOMENTUM)
        torch.testing.assert_close(_flat(target), MOMENTUM * target_before + (1 - MOMENTUM) * _flat(enc))

    def test_optimizer_step_clears_the_accumulated_grads(self):
        enc, pred, target = _models()
        opt = torch.optim.SGD(list(enc.parameters()) + list(pred.parameters()), lr=0.1)
        _loss(enc, pred, target, _global_batch(0)).backward()
        optimizer_step(opt, None, list(enc.parameters()) + list(pred.parameters()), None)
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in enc.parameters()))


class GradScalerAccumulationTest(unittest.TestCase):
    def _run(self, scaler):
        enc, pred, target = _models()
        params = list(enc.parameters()) + list(pred.parameters())
        opt = torch.optim.SGD(params, lr=0.1)
        x = _global_batch(0)
        for micro_itr in range(K):
            loss = _loss(enc, pred, target, _micro_batch(x, 0, micro_itr)) / K
            (scaler.scale(loss) if scaler is not None else loss).backward()
        _, _, norm = optimizer_step(opt, scaler, params, None)
        return norm, _flat(enc, pred)

    def test_scaled_accumulation_matches_unscaled(self):
        try:
            scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10)
        except Exception as e:  # pragma: no cover - depends on the torch build
            self.skipTest(f"no CPU GradScaler: {e}")
        ref_norm, ref_params = self._run(None)
        norm, params = self._run(scaler)
        self.assertAlmostEqual(norm, ref_norm, places=12)
        torch.testing.assert_close(params, ref_params, rtol=1e-12, atol=1e-14)
        self.assertEqual(scaler.get_scale(), 2.0**10)  # updated once, no inf found


class CheckGradAccumStepsTest(unittest.TestCase):
    def test_accepts_positive_ints(self):
        self.assertEqual(check_grad_accum_steps(1), 1)
        self.assertEqual(check_grad_accum_steps(16), 16)

    def test_rejects_the_rest(self):
        for bad in (0, -1, 1.5, 2.0, "4", None, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_grad_accum_steps(bad)


if __name__ == "__main__":
    unittest.main()
