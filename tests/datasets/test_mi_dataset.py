# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Parity + integration checks for the KneeNo-backed MIDataset wrapper.

The raw volume loading now lives in ``kneeno``; this verifies the V-JEPA wrapper
reproduces the old ``(buffer, label, clip_indices)`` sample and still flows
through ``MaskCollator``.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kneeno import UnlabeledKneeMRIDataset
from src.datasets.mi_dataset import MIDataset, make_MIDataset
from src.masks.multiseq_multiblock3d import MaskCollator

H, W = 12, 10
SPEC = {
    "c0": {"cor": 4, "sag": 6},
    "c1": {"cor": 4, "tra": 8},
    "c2": {"sag": 6},
}
MASK_CFGS = [
    {"aspect_ratio": [0.75, 1.5], "spatial_scale": [0.15, 0.15], "temporal_scale": [1.0, 1.0], "num_blocks": 8},
    {"aspect_ratio": [0.75, 1.5], "spatial_scale": [0.7, 0.7], "temporal_scale": [1.0, 1.0], "num_blocks": 2},
]


def make_dataset(root):
    rng = np.random.default_rng(0)
    meta = {}
    for case_id, series in SPEC.items():
        meta[case_id] = {}
        for name, n in series.items():
            d = Path(root) / case_id / name
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(d / f"{i:03d}.jpeg")
            meta[case_id][name] = {"n_images": n, "resolution": [H, W]}
    p = Path(root) / "metadata.json"
    p.write_text(json.dumps(meta))
    return str(p)


class MIDatasetWrapperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.meta = make_dataset(self.root)

    def test_sample_matches_raw_loader(self):
        core = UnlabeledKneeMRIDataset(self.root, self.meta)
        wrapped = MIDataset(self.root, self.meta, transform=None)

        self.assertEqual(len(wrapped), len(core))
        self.assertEqual(len(wrapped), 5)
        for i in range(len(wrapped)):
            buffer, label, clip_indices = wrapped[i]
            self.assertEqual(label, 0)
            self.assertEqual(len(buffer), 1)
            self.assertTrue(np.array_equal(buffer[0], core[i].numpy()))
            self.assertEqual(buffer[0].dtype, np.uint8)
            self.assertEqual(len(clip_indices), 1)
            self.assertTrue(
                np.array_equal(clip_indices[0], np.arange(core.effective_depth(i)))
            )

    def test_transform_is_applied_to_raw_volume(self):
        core = UnlabeledKneeMRIDataset(self.root, self.meta)
        calls = []

        def fake_transform(vol):
            calls.append(vol)
            return torch.as_tensor(vol).permute(3, 0, 1, 2).float()

        wrapped = MIDataset(self.root, self.meta, transform=fake_transform)
        buffer, _, _ = wrapped[2]
        self.assertTrue(np.array_equal(calls[0], core[2].numpy()))
        self.assertTrue(
            torch.equal(
                buffer[0],
                torch.as_tensor(core[2].numpy()).permute(3, 0, 1, 2).float(),
            )
        )

    def test_flows_through_mask_collator(self):
        collator = MaskCollator(
            cfgs_mask=MASK_CFGS,
            dataset_fpcs=[4, 6, 8],
            crop_size=64,
            patch_size=16,
            tubelet_size=2,
        )
        _, loader, sampler = make_MIDataset(
            data_root=self.root,
            data_meta=self.meta,
            batch_size=2,
            transform=None,
            collator=collator,
            num_workers=0,
            pin_mem=False,
            persistent_workers=False,
        )
        sampler.set_epoch(0)

        batches = list(loader)
        self.assertTrue(batches, "sampler produced no batches")
        for fpc_collations in batches:
            self.assertEqual(len(fpc_collations), 1)  # single-depth batches
            collated_batch, masks_enc, masks_pred = fpc_collations[0]
            clips = collated_batch[0][0]  # buffer list -> first (only) entry, collated
            self.assertEqual(clips.shape[0], 2)  # batch_size
            self.assertEqual(len(masks_enc), len(MASK_CFGS))
            self.assertEqual(len(masks_pred), len(MASK_CFGS))


if __name__ == "__main__":
    unittest.main()
