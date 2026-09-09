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
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from app.vjepa_2_1.train import MAX_CONSECUTIVE_NONFINITE, compute_grad_norm


def _ema_schedule(m=0.99, n=1000):
    return (m for _ in range(n))


def _apply_guarded_step(online, target, optimizer, momentum_scheduler, poison=None):
    """Replica of the train.py step region: backward, grad-norm probe, guard, EMA.

    Mirrors ``train.py``:
        loss.backward()
        grad_norm = compute_grad_norm(enc_params + pred_params)
        grads_finite = bool(np.isfinite(grad_norm))
        if grads_finite: optimizer.step()
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


if __name__ == "__main__":
    unittest.main()
