# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the KneeNo <-> V-JEPA 2.1 bridge.

Builds a tiny real ``VisionTransformer`` (not a mock) so this exercises the
actual patch-embedding / forward path, not just the adapter's own logic.
``depth=12`` is required by a quirk of ``VisionTransformer.__init__``
(``hierarchical_layers`` is only populated for depth in {12, 24, 40, 48}).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.utils import init_video_model
from app.vjepa_2_1.wrappers import MultiSeqWrapper
from kneeno import ClassificationEvaluator, LabeledKneeMRIDataset
from src.datasets.kneeno_adapter import VJepa21Adapter

H, W = 20, 24
SPEC = {f"c{i}": {"cor": 4, "sag": 6} for i in range(10)}

EMBED_DIM = 16
CROP_SIZE = 32
PATCH_SIZE = 8
TUBELET_SIZE = 2
NUM_FRAMES = 4


def make_dataset(root, spec=SPEC, h=H, w=W):
    rng = np.random.default_rng(0)
    meta = {}
    for case_id, series in spec.items():
        meta[case_id] = {}
        for name, n in series.items():
            d = Path(root) / case_id / name
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(d / f"{i:03d}.jpeg")
            meta[case_id][name] = {"n_images": n, "resolution": [h, w]}
    p = Path(root) / "metadata.json"
    p.write_text(json.dumps(meta))
    return str(p)


def make_tiny_encoder():
    backbone = video_vit.VisionTransformer(
        img_size=CROP_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        in_chans=1,
        embed_dim=EMBED_DIM,
        depth=12,
        num_heads=2,
        use_rope=True,
        modality_embedding=True,
    )
    model = MultiSeqWrapper(backbone)
    model.eval()
    return model


class VJepa21AdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = VJepa21Adapter(embed_dim=EMBED_DIM, crop_size=CROP_SIZE, normalize=((0.5,), (0.5,)))
        rng = np.random.default_rng(0)
        self.volume = torch.from_numpy(
            rng.integers(0, 256, size=(NUM_FRAMES, H, W, 1), dtype=np.uint8)
        )

    def test_prepare_input_shape_and_dtype(self):
        prepped = self.adapter.prepare_input(self.volume)
        self.assertEqual(tuple(prepped.shape), (1, NUM_FRAMES, CROP_SIZE, CROP_SIZE))
        self.assertEqual(prepped.dtype, torch.float32)

    def test_collate_stacks_batch_axis(self):
        prepped = [self.adapter.prepare_input(self.volume) for _ in range(3)]
        batch = self.adapter.collate(prepped)
        self.assertEqual(tuple(batch.shape), (3, 1, NUM_FRAMES, CROP_SIZE, CROP_SIZE))

    def test_forward_features_against_real_encoder(self):
        model = make_tiny_encoder()
        batch = self.adapter.collate([self.adapter.prepare_input(self.volume) for _ in range(2)])
        with torch.no_grad():
            out = self.adapter.forward_features(model, batch)

        self.assertIsNone(out["cls"])
        n_temporal = NUM_FRAMES // TUBELET_SIZE
        n_spatial = CROP_SIZE // PATCH_SIZE
        expected_n = n_temporal * n_spatial * n_spatial
        self.assertEqual(tuple(out["patches"].shape), (2, expected_n, EMBED_DIM))

    def test_has_no_cls_token(self):
        self.assertFalse(self.adapter.has_cls_token)


class InitVideoModelCompatibilityTest(unittest.TestCase):
    """The adapter must work against train.py's actual encoder factory
    (init_video_model), not just a hand-built VisionTransformer."""

    def test_encoder_from_init_video_model_is_adapter_compatible(self):
        encoder, _ = init_video_model(
            device=torch.device("cpu"),
            patch_size=PATCH_SIZE,
            max_num_frames=NUM_FRAMES,
            tubelet_size=TUBELET_SIZE,
            in_chans=1,
            model_name="vit_tiny",
            crop_size=CROP_SIZE,
            pred_depth=12,  # pred_depth must be in {12,24,40,48}, same quirk as the encoder
            pred_embed_dim=32,
            use_rope=True,
            modality_embedding=True,
        )
        encoder.eval()
        adapter = VJepa21Adapter(embed_dim=encoder.embed_dim, crop_size=CROP_SIZE, normalize=((0.5,), (0.5,)))

        rng = np.random.default_rng(0)
        volume = torch.from_numpy(rng.integers(0, 256, size=(NUM_FRAMES, H, W, 1), dtype=np.uint8))
        batch = adapter.collate([adapter.prepare_input(volume) for _ in range(2)])

        with torch.no_grad():
            out = adapter.forward_features(encoder, batch)

        self.assertIsNone(out["cls"])
        self.assertEqual(out["patches"].shape[0], 2)
        self.assertEqual(out["patches"].shape[2], encoder.embed_dim)


class ClassificationEvaluatorAgainstVJepaEncoderTest(unittest.TestCase):
    """End-to-end: real (tiny) encoder + real adapter through KneeNo's evaluator."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.meta = make_dataset(self._tmp.name)
        self.dataset = LabeledKneeMRIDataset(
            self._tmp.name, self.meta, num_classes=3, series_depth=NUM_FRAMES, resample_mode="nearest"
        )
        self.model = make_tiny_encoder()
        self.adapter = VJepa21Adapter(embed_dim=EMBED_DIM, crop_size=CROP_SIZE, normalize=((0.5,), (0.5,)))

    def _config(self, **overrides):
        config = {
            "seed": 3,
            "split": {"test_fraction": 0.3},
            "data": {"series_depth": NUM_FRAMES, "batch_size": 4, "num_workers": 0},
            "logging": {"tensorboard_dir": None},
            "knn": {"k": [3]},
            "linear_pool": {"epochs": 1, "batch_size": 4},
            "attentive_pool": {"epochs": 1, "batch_size": 4, "num_heads": 4},
        }
        config.update(overrides)
        return config

    def test_knn_and_pooled_heads_run(self):
        evaluator = ClassificationEvaluator(config=self._config(), adapter=self.adapter, dataset=self.dataset)
        for task in ("knn", "linear_pool", "attentive_pool"):
            metrics = evaluator.evaluate(self.model, tasks=[task], epoch=0)
            self.assertTrue(any(k.startswith(task) for k in metrics))

    def test_linear_task_rejected_without_cls_token(self):
        evaluator = ClassificationEvaluator(config=self._config(), adapter=self.adapter, dataset=self.dataset)
        with self.assertRaises(ValueError):
            evaluator.evaluate(self.model, tasks=["linear"], epoch=0)


if __name__ == "__main__":
    unittest.main()
