# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Non-finite-gradient guard for ``app/vjepa_2_1/train.py``.

The guard lives inline in the (deeply nested) ``train_step`` closure, so it cannot be
imported. We test the one extractable piece (``compute_grad_norm``'s non-finite
detection) and a faithful inline replica of the guard block: on a non-finite gradient
the optimizer step and the target-EMA update are both skipped, ``optimizer.zero_grad()``
still runs, and the EMA momentum schedule still advances. A separate replica covers the
consecutive-skip counter that aborts the run after ``MAX_CONSECUTIVE_NONFINITE`` skips
in a row.

Also covers the other extractable step-region pieces: ``clip_gradients``
(``optimization.gradient_clipping``, applied between the norm probe and the optimizer step),
``check_optimization_config`` (``gradient_clipping`` / ``betas`` validation),
``apply_optimizer_hparams`` (the configured betas/eps win over a resumed optimizer state) and
``qk_temperature_summary`` (the epoch-end temperature log line).
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from app.vjepa_2_1.models.utils.modules import Attention
from app.vjepa_2_1.train import (
    MAX_CONSECUTIVE_NONFINITE,
    check_optimization_config,
    clip_gradients,
    compute_grad_norm,
    qk_temperature_summary,
)
from app.vjepa_2_1.utils import apply_optimizer_hparams


def _ema_schedule(m=0.99, n=1000):
    return (m for _ in range(n))


def _apply_guarded_step(online, target, optimizer, momentum_scheduler, poison=None, max_norm=None):
    """Replica of the train.py step region: backward, grad-norm probe, guard, clip, EMA.

    Mirrors ``train.py``:
        loss.backward()
        grad_norm = compute_grad_norm(enc_params + pred_params)
        grads_finite = bool(np.isfinite(grad_norm))
        if grads_finite:
            clip_gradients(enc_params + pred_params, gradient_clipping, grad_norm)
            optimizer.step()
        optimizer.zero_grad()
        m = next(momentum_scheduler)            # advances every iter
        if run_step and grads_finite: <EMA update>

    Returns ``(grads_finite, step_applied, grad_norm)``.
    """
    x = torch.randn(8, 4)
    loss = (online(x) ** 2).mean()
    loss.backward()

    if poison is not None:
        dict(online.named_parameters())[poison].grad[0] = float("inf")

    grad_norm = compute_grad_norm(list(online.parameters()))
    grads_finite = bool(np.isfinite(grad_norm))

    if grads_finite:
        clip_gradients(list(online.parameters()), max_norm, grad_norm)
        optimizer.step()
    optimizer.zero_grad()

    m = next(momentum_scheduler)  # advance every iter, aligned to the step count
    step_applied = grads_finite
    if step_applied:
        with torch.no_grad():
            for p_q, p_k in zip(online.parameters(), target.parameters()):
                p_k.mul_(m).add_(p_q, alpha=1 - m)
    return grads_finite, step_applied, grad_norm


class ComputeGradNormTest(unittest.TestCase):
    def test_finite_for_normal_grads(self):
        mod = nn.Linear(4, 4)
        (mod(torch.randn(5, 4)) ** 2).mean().backward()
        gn = compute_grad_norm(list(mod.parameters()))
        self.assertTrue(np.isfinite(gn))
        self.assertGreater(gn, 0.0)

    def test_nan_when_no_grads(self):
        mod = nn.Linear(4, 4)
        self.assertTrue(np.isnan(compute_grad_norm(list(mod.parameters()))))

    def test_propagates_non_finite(self):
        mod = nn.Linear(4, 4)
        (mod(torch.randn(5, 4)) ** 2).mean().backward()
        mod.weight.grad[0, 0] = float("inf")
        self.assertFalse(np.isfinite(compute_grad_norm(list(mod.parameters()))))

        mod.weight.grad[0, 0] = float("nan")
        self.assertFalse(np.isfinite(compute_grad_norm(list(mod.parameters()))))

    def test_never_rescales_grads(self):
        mod = nn.Linear(4, 4)
        (mod(torch.randn(5, 4)) ** 2).mean().backward()
        before = [p.grad.clone() for p in mod.parameters()]
        compute_grad_norm(list(mod.parameters()))
        for b, p in zip(before, mod.parameters()):
            self.assertTrue(torch.equal(b, p.grad))


class NonFiniteGradGuardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.online = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        self.target = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.AdamW(self.online.parameters(), lr=1e-3)
        self.sched = _ema_schedule()

    def _snapshot(self, module):
        return [p.detach().clone() for p in module.parameters()]

    def _changed(self, before, module):
        return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))

    def test_finite_grad_steps_and_updates_ema(self):
        on0, tg0 = self._snapshot(self.online), self._snapshot(self.target)
        grads_finite, step_applied, gnorm = _apply_guarded_step(
            self.online, self.target, self.opt, self.sched
        )
        self.assertTrue(grads_finite)
        self.assertTrue(step_applied)
        self.assertTrue(np.isfinite(gnorm))
        self.assertTrue(self._changed(on0, self.online), "online weights should move")
        self.assertTrue(self._changed(tg0, self.target), "target EMA should move")

    def test_nonfinite_grad_skips_step_ema_and_advances_schedule(self):
        on0, tg0 = self._snapshot(self.online), self._snapshot(self.target)
        grads_finite, step_applied, gnorm = _apply_guarded_step(
            self.online, self.target, self.opt, self.sched, poison="0.weight"
        )
        self.assertFalse(grads_finite)
        self.assertFalse(step_applied)
        self.assertFalse(np.isfinite(gnorm))
        # (a) optimizer step skipped -> online weights untouched
        self.assertFalse(self._changed(on0, self.online))
        # (b) EMA update skipped -> target untouched
        self.assertFalse(self._changed(tg0, self.target))
        # (c) zero_grad ran -> grads cleared
        self.assertTrue(
            all(
                p.grad is None or torch.count_nonzero(p.grad) == 0
                for p in self.online.parameters()
            )
        )
        # (d) EMA schedule still advanced (one value consumed from a fresh 1000-gen)
        self.assertEqual(sum(1 for _ in self.sched), 999)

    def test_recovery_after_skip(self):
        _apply_guarded_step(self.online, self.target, self.opt, self.sched, poison="0.weight")
        on0 = self._snapshot(self.online)
        grads_finite, step_applied, _ = _apply_guarded_step(
            self.online, self.target, self.opt, self.sched
        )
        self.assertTrue(step_applied)
        self.assertTrue(self._changed(on0, self.online))


def _run_with_consecutive_guard(grads_finite_seq):
    """Replica of train.py's consecutive-skip counter:

        if grads_finite:
            consecutive_nonfinite = 0
        else:
            consecutive_nonfinite += 1
            if consecutive_nonfinite > MAX_CONSECUTIVE_NONFINITE:
                raise RuntimeError(...)

    ``grads_finite_seq`` is an iterable of bools (one per iteration). Returns the number
    of iterations processed before returning normally; raises ``RuntimeError`` if the
    guard trips.
    """
    consecutive_nonfinite = 0
    for i, grads_finite in enumerate(grads_finite_seq):
        if grads_finite:
            consecutive_nonfinite = 0
        else:
            consecutive_nonfinite += 1
            if consecutive_nonfinite > MAX_CONSECUTIVE_NONFINITE:
                raise RuntimeError(
                    f"{consecutive_nonfinite} consecutive non-finite-gradient "
                    f"iterations (last at itr {i}); aborting."
                )
    return i + 1


class ConsecutiveNonFiniteAbortTest(unittest.TestCase):
    def test_survives_exactly_the_limit_in_a_row(self):
        # MAX in a row is tolerated; the (MAX+1)-th consecutive skip aborts.
        n = _run_with_consecutive_guard([False] * MAX_CONSECUTIVE_NONFINITE)
        self.assertEqual(n, MAX_CONSECUTIVE_NONFINITE)

    def test_aborts_after_limit_exceeded(self):
        with self.assertRaises(RuntimeError):
            _run_with_consecutive_guard([False] * (MAX_CONSECUTIVE_NONFINITE + 1))

    def test_finite_step_resets_the_counter(self):
        # skip MAX times, recover once, then skip MAX more -> never trips.
        seq = (
            [False] * MAX_CONSECUTIVE_NONFINITE
            + [True]
            + [False] * MAX_CONSECUTIVE_NONFINITE
        )
        n = _run_with_consecutive_guard(seq)
        self.assertEqual(n, len(seq))

    def test_isolated_skips_never_trip(self):
        seq = [True, False, True, False, False, True] * 50
        self.assertEqual(_run_with_consecutive_guard(seq), len(seq))


def _grads_of(mod):
    return [p.grad.clone() for p in mod.parameters()]


class ClipGradientsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mod = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        (self.mod(torch.randn(8, 4)) ** 2).sum().backward()
        self.params = list(self.mod.parameters())
        self.norm = compute_grad_norm(self.params)

    def test_disabled_is_a_no_op(self):
        before = _grads_of(self.mod)
        self.assertFalse(clip_gradients(self.params, None, self.norm))
        for b, p in zip(before, self.params):
            self.assertTrue(torch.equal(b, p.grad))

    def test_below_max_is_a_no_op(self):
        before = _grads_of(self.mod)
        self.assertFalse(clip_gradients(self.params, 2 * self.norm, self.norm))
        for b, p in zip(before, self.params):
            self.assertTrue(torch.equal(b, p.grad))

    def test_above_max_rescales_to_max(self):
        max_norm = self.norm / 4
        before = _grads_of(self.mod)
        self.assertTrue(clip_gradients(self.params, max_norm, self.norm))
        self.assertAlmostEqual(compute_grad_norm(self.params), max_norm, places=4)
        for b, p in zip(before, self.params):  # direction kept
            self.assertTrue(torch.allclose(p.grad, b * (max_norm / self.norm), rtol=1e-4))

    def test_skips_params_without_grad(self):
        extra = nn.Parameter(torch.zeros(3))  # grad is None
        self.assertTrue(clip_gradients(self.params + [extra], self.norm / 2, self.norm))
        self.assertIsNone(extra.grad)

    def test_clip_happens_before_the_step(self):
        """SGD(lr=1): the parameter update is exactly -grad, so its norm is the clipped norm."""
        torch.manual_seed(0)
        online = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        target = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        before = [p.detach().clone() for p in online.parameters()]
        max_norm = 1e-3
        grads_finite, _, gnorm = _apply_guarded_step(online, target, opt, _ema_schedule(), max_norm=max_norm)
        self.assertTrue(grads_finite)
        self.assertGreater(gnorm, max_norm)  # logged norm is the pre-clip one
        delta = torch.cat([(p.detach() - b).flatten() for b, p in zip(before, online.parameters())])
        self.assertAlmostEqual(float(delta.norm()), max_norm, places=6)

    def test_nonfinite_step_is_still_skipped(self):
        torch.manual_seed(0)
        online = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        target = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 4))
        opt = torch.optim.SGD(online.parameters(), lr=1.0)
        before = [p.detach().clone() for p in online.parameters()]
        grads_finite, step_applied, _ = _apply_guarded_step(
            online, target, opt, _ema_schedule(), poison="0.weight", max_norm=1.0
        )
        self.assertFalse(grads_finite or step_applied)
        for b, p in zip(before, online.parameters()):
            self.assertTrue(torch.equal(b, p))


class CheckOptimizationConfigTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(check_optimization_config(None, (0.9, 0.999)), (None, (0.9, 0.999)))

    def test_normalises_yaml_values(self):
        self.assertEqual(check_optimization_config(1, [0.9, 0.95]), (1.0, (0.9, 0.95)))

    def test_rejects_bad_values(self):
        bad = ((0, (0.9, 0.95)), (-1.0, (0.9, 0.95)), (None, (0.9,)), (None, (0.9, 1.0)), (None, (-0.1, 0.9)))
        for clip, betas in bad:
            with self.subTest(clip=clip, betas=betas), self.assertRaises(ValueError):
                check_optimization_config(clip, betas)


class ApplyOptimizerHparamsTest(unittest.TestCase):
    def test_config_wins_over_resumed_state(self):
        mod = nn.Linear(4, 4)
        old = torch.optim.AdamW(
            [{"params": [mod.weight]}, {"params": [mod.bias], "weight_decay": 0}], betas=(0.9, 0.999), eps=1e-8
        )
        new = torch.optim.AdamW(
            [{"params": [mod.weight]}, {"params": [mod.bias], "weight_decay": 0}], betas=(0.9, 0.95), eps=1e-6
        )
        new.load_state_dict(old.state_dict())
        self.assertEqual(new.param_groups[0]["betas"], (0.9, 0.999))  # the trap this guards against

        apply_optimizer_hparams(new, (0.9, 0.95), 1e-6)
        for g in new.param_groups:
            self.assertEqual(g["betas"], (0.9, 0.95))
            self.assertEqual(g["eps"], 1e-6)

        (mod(torch.randn(3, 4)) ** 2).mean().backward()
        new.step()  # still a working optimizer


class QKTemperatureSummaryTest(unittest.TestCase):
    def test_none_without_temperatures(self):
        self.assertIsNone(qk_temperature_summary(Attention(16, num_heads=2, qk_norm="rms")))

    def test_reports_max_and_heads_at_cap(self):
        class Blocks(nn.Module):
            def __init__(self):
                super().__init__()
                kw = dict(num_heads=2, qk_norm="rms", qk_norm_affine=False, qk_temperature_max=4.0)
                self.blocks = nn.ModuleList([Attention(16, **kw) for _ in range(2)])

        m = Blocks()
        with torch.no_grad():
            m.blocks[1].qk_temperature.log_temperature.copy_(torch.tensor([10.0, 0.0]))
        self.assertEqual(qk_temperature_summary(m), "0:1.00/0 1:4.00/1")


if __name__ == "__main__":
    unittest.main()
