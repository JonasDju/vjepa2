# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bit-identity of the predictor's token reordering (``predictor.reorder_tokens``).

``VisionTransformerPredictor.forward`` sorts context + mask tokens into input-grid order
before its blocks and un-sorts them afterwards. Upstream did that with a per-sample Python
loop, ``torch.stack([x[i, row] for i, row in enumerate(order)])``; under torch.compile with
dynamic token counts that unrolled stack crashed Inductor (``CantSplit: 18432*s3 + 18432*s89
not divisible by 48*s3 + 48*s89`` on the first predictor recompile, job 4502603). It is now a
single ``torch.gather``.

The default V-JEPA 2.1 recipe must stay exactly what it was (it is the reference for the
comparison against the other pretraining methods), so everything here compares with
``torch.equal`` -- bitwise, no tolerance:
- ``reorder_tokens`` == the upstream loop (kept below as ``loop_reorder``) on mask indices and
  on float32 / bfloat16 / float16 tokens, forward and backward.
- The full predictor, run once as is and once with ``reorder_tokens`` patched back to the loop:
  identical ``x_pred`` / ``x_context``, input gradient and every parameter gradient -- for both
  ``return_all_tokens`` branches, the upstream q/k setup and the MI config's bounded-QK setup,
  float32 and bfloat16 autocast, both mask tokens. Always one mask tensor per call, which is
  what ``PredictorMultiSeqWrapper`` passes; the predictor's list-of-masks path is inconsistent
  upstream (``B = len(x) // len(masks_x)`` *and* ``x.repeat(len(masks_x))``) and unused.
"""

import unittest
from unittest import mock

import torch

import app.vjepa_2_1.models.predictor as vit_pred
from app.vjepa_2_1.models.predictor import reorder_tokens

# tiny MI-like grid: 8 slices / tubelet 2 x (64 / 16)^2 = 4 x 4 x 4 = 64 tokens
CROP, PATCH, FRAMES, TUBELET = 64, 16, 8, 2
NUM_PATCHES = (FRAMES // TUBELET) * (CROP // PATCH) ** 2
EMBED_DIM = 32  # encoder width; the predictor input is 4 hierarchical levels of it
PRED_DIM, PRED_HEADS = 64, 2  # head_dim 32, as in the real predictor (384 / 12)
BATCH = 4


def loop_reorder(x, order):
    """The upstream implementation, verbatim, for both call sites (masks and tokens)."""
    if x.dim() == 3:
        return torch.stack([x[i, row, :] for i, row in enumerate(order)], dim=0)
    return torch.stack([x[i, row] for i, row in enumerate(order)], dim=0)


def random_permutations(rows, n, generator):
    return torch.stack([torch.randperm(n, generator=generator) for _ in range(rows)])


def make_masks(batch, n_ctxt, n_pred, generator):
    """Disjoint context / target index sets per sample, sorted like ``MaskCollator``'s output."""
    perm = random_permutations(batch, NUM_PATCHES, generator)
    masks_x = perm[:, :n_ctxt].sort(dim=1).values
    masks_y = perm[:, n_ctxt : n_ctxt + n_pred].sort(dim=1).values
    return masks_x, masks_y


def make_predictor(return_all_tokens, qk_kwargs):
    torch.manual_seed(0)
    return vit_pred.vit_predictor(
        img_size=CROP,
        patch_size=PATCH,
        num_frames=FRAMES,
        tubelet_size=TUBELET,
        embed_dim=EMBED_DIM,
        predictor_embed_dim=PRED_DIM,
        depth=4,
        num_heads=PRED_HEADS,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=False,  # nonzero mask tokens -> a misplaced token changes the output
        use_rope=True,
        use_sdpa=True,
        return_all_tokens=return_all_tokens,
        interpolate_rope=True,
        modality_embedding=True,
        img_temporal_dim_size=1,
        **qk_kwargs,
    )


QK_SETUPS = {
    "upstream": {},
    "mi_bounded_qk": {"qk_norm": "rms", "qk_norm_affine": False, "qk_temperature_max": 4.0},
}


class ReorderTokensTest(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(0)

    def test_mask_indices(self):
        masks = random_permutations(BATCH, NUM_PATCHES, self.g)
        order = torch.argsort(masks, dim=1)
        out = reorder_tokens(masks, order)
        self.assertTrue(torch.equal(out, loop_reorder(masks, order)))
        self.assertEqual(out.dtype, masks.dtype)
        # sorting a permutation of arange(N) by its argsort gives arange(N) in every row
        self.assertTrue(torch.equal(out, torch.arange(NUM_PATCHES).expand(BATCH, -1)))

    def test_tokens_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                order = random_permutations(BATCH, NUM_PATCHES, self.g)
                x = torch.randn(BATCH, NUM_PATCHES, PRED_DIM, generator=self.g).to(dtype)
                upstream = torch.randn(BATCH, NUM_PATCHES, PRED_DIM, generator=self.g).to(dtype)
                outs, grads = [], []
                for fn in (reorder_tokens, loop_reorder):
                    xi = x.clone().requires_grad_(True)
                    out = fn(xi, order)
                    out.backward(upstream)
                    outs.append(out.detach())
                    grads.append(xi.grad)
                self.assertEqual(outs[0].dtype, dtype)
                self.assertTrue(torch.equal(outs[0], outs[1]))
                self.assertTrue(torch.equal(grads[0], grads[1]))

    def test_reverse_order_restores_input(self):
        """argsort-then-reverse-argsort (what the predictor does around its blocks) is the identity."""
        x = torch.randn(BATCH, NUM_PATCHES, PRED_DIM, generator=self.g)
        order = random_permutations(BATCH, NUM_PATCHES, self.g)
        back = reorder_tokens(reorder_tokens(x, order), torch.argsort(order, dim=1))
        self.assertTrue(torch.equal(back, x))


class PredictorBitIdentityTest(unittest.TestCase):
    """Whole ``VisionTransformerPredictor`` forward + backward: gather vs. the upstream loop."""

    def run_predictor(self, predictor, x, masks_x, masks_y, mask_index, autocast, reorder):
        predictor.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        with mock.patch.object(vit_pred, "reorder_tokens", reorder):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                x_pred, x_context = predictor(x, masks_x, masks_y, mod="video", mask_index=mask_index)
                # fixed pseudo-random weights, so every output element feeds a distinct gradient
                w_gen = torch.Generator().manual_seed(1)
                loss = (x_pred.float() * torch.randn(x_pred.shape, generator=w_gen)).sum()
                if x_context is not None:
                    loss = loss + (x_context.float() * torch.randn(x_context.shape, generator=w_gen)).sum()
        loss.backward()
        outputs = [x_pred.detach(), None if x_context is None else x_context.detach(), x.grad]
        grads = {n: p.grad for n, p in predictor.named_parameters() if p.grad is not None}
        return outputs, grads

    def assert_bit_identical(self, a, b):
        (outs_a, grads_a), (outs_b, grads_b) = a, b
        for name, oa, ob in zip(("x_pred", "x_context", "x.grad"), outs_a, outs_b):
            if oa is None:
                self.assertIsNone(ob, name)
                continue
            self.assertTrue(torch.equal(oa, ob), f"{name} differs")
        self.assertEqual(grads_a.keys(), grads_b.keys())
        for name in grads_a:
            self.assertTrue(torch.equal(grads_a[name], grads_b[name]), f"grad of {name} differs")

    def test_gather_matches_upstream_loop(self):
        g = torch.Generator().manual_seed(0)
        # (n_ctxt, n_pred, mask_index): the two mask configs' mask tokens, and lengths that don't
        # cover the whole grid in total, like MaskCollator's min-trimmed masks
        mask_setups = [(37, 21, 0), (23, 30, 1)]
        for qk_name, qk_kwargs in QK_SETUPS.items():
            for return_all_tokens in (True, False):
                predictor = make_predictor(return_all_tokens, qk_kwargs)
                for autocast in (False, True):
                    for n_ctxt, n_pred, mask_index in mask_setups:
                        with self.subTest(
                            qk=qk_name, return_all_tokens=return_all_tokens, autocast=autocast, n_ctxt=n_ctxt
                        ):
                            # one mask tensor per call, as PredictorMultiSeqWrapper passes it
                            masks_x, masks_y = make_masks(BATCH, n_ctxt, n_pred, g)
                            x = torch.randn(BATCH, n_ctxt, 4 * EMBED_DIM, generator=g)
                            args = (predictor, x, masks_x, masks_y, mask_index, autocast)

                            gather = self.run_predictor(*args, reorder_tokens)
                            gather_again = self.run_predictor(*args, reorder_tokens)
                            loop = self.run_predictor(*args, loop_reorder)
                            # control: the predictor itself is deterministic, so a mismatch below is real
                            self.assert_bit_identical(gather, gather_again)
                            self.assert_bit_identical(gather, loop)

    def test_patch_reaches_forward(self):
        """Guard for the test itself: forward must look ``reorder_tokens`` up at call time."""
        predictor = make_predictor(True, {})
        g = torch.Generator().manual_seed(0)
        masks_x, masks_y = make_masks(BATCH, 30, 20, g)
        calls = []

        def spy(x, order):
            calls.append(x.dim())
            return loop_reorder(x, order)

        with mock.patch.object(vit_pred, "reorder_tokens", spy):
            predictor(torch.randn(BATCH, 30, 4 * EMBED_DIM), masks_x, masks_y)
        self.assertEqual(calls, [2, 3, 3])  # masks, tokens before the blocks, tokens after


if __name__ == "__main__":
    unittest.main()
