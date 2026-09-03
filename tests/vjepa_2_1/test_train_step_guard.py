# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Non-finite-gradient guard for ``app/vjepa_2_1/train.py``.

No GPU on the dev box, and the guard lives inline in the (deeply nested) ``train_step``
closure, so we cannot import it. Instead we test the two extractable pieces
(``compute_grad_norm`` non-finite detection, ``_grad_layer_report`` localisation) and a
faithful inline replica of the guard block: on a non-finite gradient the optimizer step
and the target-EMA update are both skipped, ``optimizer.zero_grad()`` still runs, and the
EMA schedule still advances.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from app.vjepa_2_1.train import (
    _grad_layer_report,
    _module_l2_distance,
    compute_grad_norm,
)


def _ema_schedule(m=0.99, n=1000):
    return (m for _ in range(n))


def _apply_guarded_step(online, target, optimizer, momentum_scheduler, poison=None):
    """Replica of the train.py step region: backward, grad-norm probe, guard, EMA.

    Returns ``(grads_finite, step_applied, grad_norm, layer_report)``."""
    x = torch.randn(8, 4)
    loss = (online(x) ** 2).mean()
    loss.backward()

    if poison is not None:
        dict(online.named_parameters())[poison].grad[0] = float("inf")

    enc_params = list(online.parameters())
    grad_norm = compute_grad_norm(enc_params)
    grads_finite = bool(np.isfinite(grad_norm))
    layer_report = None
    if not grads_finite:
        layer_report = _grad_layer_report(online.named_parameters())

    if grads_finite:
        optimizer.step()
    optimizer.zero_grad()

    m = next(momentum_scheduler)  # advance every iter, aligned to the step count
    step_applied = grads_finite
    if step_applied:
        with torch.no_grad():
            for p_q, p_k in zip(online.parameters(), target.parameters()):
                p_k.mul_(m).add_(p_q, alpha=1 - m)
    return grads_finite, step_applied, grad_norm, layer_report


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
        grads_finite, step_applied, gnorm, rep = _apply_guarded_step(
            self.online, self.target, self.opt, self.sched
        )
        self.assertTrue(grads_finite)
        self.assertTrue(step_applied)
        self.assertTrue(np.isfinite(gnorm))
        self.assertIsNone(rep)
        self.assertTrue(self._changed(on0, self.online), "online weights should move")
        self.assertTrue(self._changed(tg0, self.target), "target EMA should move")

    def test_nonfinite_grad_skips_step_ema_and_advances_schedule(self):
        on0, tg0 = self._snapshot(self.online), self._snapshot(self.target)
        poison = "0.weight"  # first Linear
        grads_finite, step_applied, gnorm, rep = _apply_guarded_step(
            self.online, self.target, self.opt, self.sched, poison=poison
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
            all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in self.online.parameters())
        )
        # (d) EMA schedule still advanced (one value consumed)
        # a fresh 1000-length generator: 999 remain
        self.assertEqual(sum(1 for _ in self.sched), 999)
        # (e) the poisoned layer is localised, non-finite entries counted, listed first
        self.assertIsNotNone(rep)
        self.assertEqual(rep[0][0], poison)
        self.assertGreater(rep[0][2], 0)

    def test_recovery_after_skip(self):
        # skip once, then a clean iteration must step again
        _apply_guarded_step(self.online, self.target, self.opt, self.sched, poison="0.weight")
        on0 = self._snapshot(self.online)
        grads_finite, step_applied, _, _ = _apply_guarded_step(
            self.online, self.target, self.opt, self.sched
        )
        self.assertTrue(step_applied)
        self.assertTrue(self._changed(on0, self.online))


class GradLayerReportTest(unittest.TestCase):
    def test_orders_nonfinite_first_then_topk_by_norm(self):
        torch.manual_seed(1)
        mod = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3))
        (mod(torch.randn(5, 3)) ** 2).mean().backward()
        # scale one grad up, poison another
        params = dict(mod.named_parameters())
        params["1.weight"].grad *= 1e4
        params["2.bias"].grad[0] = float("nan")

        rep = _grad_layer_report(mod.named_parameters(), topk=2)
        names = [r[0] for r in rep]
        self.assertEqual(names[0], "2.bias")  # non-finite first
        self.assertEqual(rep[0][2], 1)
        self.assertIn("1.weight", names[1:])  # largest finite grad in the topk
        self.assertLessEqual(len(rep), 1 + 2)

    def test_skips_params_without_grad(self):
        mod = nn.Linear(3, 3)
        rep = _grad_layer_report(mod.named_parameters())
        self.assertEqual(rep, [])


class ModuleDistanceSanityTest(unittest.TestCase):
    def test_zero_when_identical(self):
        a = nn.Linear(4, 4)
        b = nn.Linear(4, 4)
        b.load_state_dict(a.state_dict())
        self.assertAlmostEqual(_module_l2_distance(a, b), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
