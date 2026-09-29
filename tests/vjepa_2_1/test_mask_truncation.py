# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``random_truncation`` in ``src/masks/multiseq_multiblock3d.py``.

``_MaskGenerator`` cuts every sample's context / target indices to the batch minimum. The upstream
prefix cut always drops the last depth positions; ``random_truncation: true`` keeps a random subset.
"""

import unittest

import torch

from src.masks.multiseq_multiblock3d import MaskCollator, _MaskGenerator

FRAMES, TUBELET, CROP, PATCH = 24, 2, 128, 16
DEPTH, GRID = FRAMES // TUBELET, CROP // PATCH  # 12 depth positions, 8 x 8 in-plane
TOKENS_PER_DEPTH = GRID * GRID

# The shipped MI config's long-range tube mask: the most truncation-heavy of the two
LONG_TUBE = {
    "aspect_ratio": [0.75, 1.5],
    "num_blocks": 2,
    "spatial_scale": [0.7, 0.7],
    "temporal_scale": [1.0, 1.0],
}


def _generator(random_truncation, **overrides):
    cfg = {**LONG_TUBE, **overrides}
    return _MaskGenerator(
        crop_size=CROP,
        num_frames=FRAMES,
        spatial_patch_size=PATCH,
        temporal_patch_size=TUBELET,
        spatial_pred_mask_scale=cfg["spatial_scale"],
        temporal_pred_mask_scale=cfg["temporal_scale"],
        aspect_ratio=cfg["aspect_ratio"],
        npred=cfg["num_blocks"],
        random_truncation=random_truncation,
    )


def _context_per_depth(random_truncation, batches=40, batch_size=16, seed=0):
    """Fraction of all context tokens that falls into each depth position."""
    torch.manual_seed(seed)
    generator = _generator(random_truncation)
    counts = torch.zeros(DEPTH)
    for _ in range(batches):
        masks_enc, _ = generator(batch_size)
        counts += torch.bincount((masks_enc // TOKENS_PER_DEPTH).flatten(), minlength=DEPTH)
    return counts / counts.sum()


class TruncateTest(unittest.TestCase):
    def test_off_keeps_the_prefix(self):
        indices = torch.arange(10, 30)
        self.assertTrue(torch.equal(_generator(False)._truncate(indices, 5), indices[:5]))

    def test_on_keeps_a_sorted_random_subset(self):
        torch.manual_seed(0)
        indices = torch.arange(100, 400, 3)
        generator = _generator(True)
        kept = [generator._truncate(indices, 40) for _ in range(20)]
        for k in kept:
            self.assertEqual(len(k), 40)
            self.assertEqual(k.dtype, indices.dtype)
            self.assertTrue(torch.equal(k, k.sort().values))
            self.assertEqual(len(k.unique()), 40)
            self.assertTrue(torch.isin(k, indices).all())
        self.assertGreater(len({tuple(k.tolist()) for k in kept}), 1, "every call kept the same subset")
        # the tail of the list is reachable, unlike with the prefix cut
        self.assertTrue(any(k.max() > indices[39] for k in kept))

    def test_on_leaves_short_lists_untouched(self):
        indices = torch.arange(7)
        self.assertTrue(torch.equal(_generator(True)._truncate(indices, 7), indices))
        self.assertTrue(torch.equal(_generator(True)._truncate(indices, 9), indices))


class MaskGeneratorTest(unittest.TestCase):
    def test_same_shapes_and_disjoint_masks(self):
        # The random subset is drawn after every block mask of the batch, so with the same seed both
        # settings sample the same masks and cut them to the same batch minimum.
        outputs = {}
        for random_truncation in (False, True):
            torch.manual_seed(3)
            outputs[random_truncation] = _generator(random_truncation)(8)
        (enc_off, pred_off), (enc_on, pred_on) = outputs[False], outputs[True]
        self.assertEqual(enc_on.shape, enc_off.shape)
        self.assertEqual(pred_on.shape, pred_off.shape)
        for enc, pred in zip(enc_on, pred_on):
            self.assertTrue(torch.equal(enc, enc.sort().values))
            self.assertTrue(torch.equal(pred, pred.sort().values))
            self.assertFalse(torch.isin(enc, pred).any(), "a token is both context and target")

    def test_prefix_cut_starves_the_last_depths(self):
        # Documents the upstream behaviour the switch exists for.
        share = _context_per_depth(random_truncation=False)
        self.assertLess(share[-3:].sum(), 0.25 * share[:3].sum())

    def test_random_cut_is_depth_balanced(self):
        # Tube masks cover every depth position alike, so an unbiased cut keeps the context uniform.
        share = _context_per_depth(random_truncation=True)
        self.assertLess((share - 1 / DEPTH).abs().max().item(), 0.02)


class MaskCollatorTest(unittest.TestCase):
    def test_config_key_reaches_the_generators(self):
        for cfg, expected in ((LONG_TUBE, False), ({**LONG_TUBE, "random_truncation": True}, True)):
            collator = MaskCollator(
                cfgs_mask=[cfg], dataset_fpcs=[FRAMES], crop_size=CROP, patch_size=PATCH, tubelet_size=TUBELET
            )
            self.assertIs(collator.mask_generators[FRAMES][0].random_truncation, expected)


if __name__ == "__main__":
    unittest.main()
