# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end CPU check for the standalone post-training evaluation script
(``app/vjepa_2_1/eval_classification.py``): builds a tiny checkpoint the way
``train.py::save_checkpoint`` would, then drives the script's own ``main()``
through ``sys.argv`` -- exercising config parsing, encoder reconstruction,
checkpoint loading, and the standalone (``log_every_head_epoch=True``) call
into ``ClassificationEvaluator`` in one pass.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from app.vjepa_2_1.utils import init_video_model
from tests.vjepa_2_1.test_kneeno_adapter import CROP_SIZE, NUM_FRAMES, PATCH_SIZE, TUBELET_SIZE

H, W = 20, 24
SPEC = {f"c{i}": {"cor": 4, "sag": 6} for i in range(10)}


class EvalClassificationCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.data_root = self.root / "data"
        self._make_dataset()
        self._make_checkpoint()
        self._make_config()

    def _make_dataset(self):
        rng = np.random.default_rng(0)
        meta = {}
        for case_id, series in SPEC.items():
            meta[case_id] = {}
            for name, n in series.items():
                d = self.data_root / case_id / name
                d.mkdir(parents=True)
                for i in range(n):
                    arr = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
                    Image.fromarray(arr, mode="L").save(d / f"{i:03d}.jpeg")
                meta[case_id][name] = {"n_images": n, "resolution": [H, W]}
        self.meta_path = self.root / "metadata.json"
        self.meta_path.write_text(json.dumps(meta))

    def _make_checkpoint(self):
        encoder, predictor = init_video_model(
            device=torch.device("cpu"),
            patch_size=PATCH_SIZE,
            max_num_frames=NUM_FRAMES,
            tubelet_size=TUBELET_SIZE,
            in_chans=1,
            model_name="vit_tiny",
            crop_size=CROP_SIZE,
            pred_depth=12,
            pred_embed_dim=32,
            use_rope=True,
            modality_embedding=True,
        )
        self.checkpoint_path = self.root / "latest.pth.tar"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "target_encoder": encoder.state_dict(),
                "epoch": 3,
                "opt": {},
                "scaler": None,
                "batch_size": 2,
                "world_size": 1,
                "lr": 1e-4,
            },
            self.checkpoint_path,
        )

    def _make_config(self):
        self.tb_dir = self.root / "tb"
        config = {
            "data": {
                "dataset_type": "MIDataset",
                "data_root": str(self.data_root),
                "data_meta": str(self.meta_path),
                "series_depth": NUM_FRAMES,
                "resample_mode": "nearest",
                "patch_size": PATCH_SIZE,
                "tubelet_size": TUBELET_SIZE,
                "crop_size": CROP_SIZE,
            },
            "model": {
                "model_name": "vit_tiny",
                "pred_depth": 12,
                "pred_embed_dim": 32,
                "use_rope": True,
                "modality_embedding": True,
            },
            "eval": {
                "seed": 1,
                "split": {"test_fraction": 0.3},
                "data": {
                    "data_root": str(self.data_root),
                    "label_meta": str(self.meta_path),
                    "num_classes": 3,
                    "series_depth": NUM_FRAMES,
                    "batch_size": 4,
                    "num_workers": 0,
                },
                "logging": {"tensorboard_dir": str(self.tb_dir)},
                "knn": {"k": [3]},
                "linear_pool": {"epochs": 2, "batch_size": 4},
                "attentive_pool": {"epochs": 2, "batch_size": 4, "num_heads": 4},
            },
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.dump(config))

    def test_cli_runs_and_logs_one_point_per_head_epoch(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        from app.vjepa_2_1 import eval_classification

        argv = [
            "eval_classification.py",
            "--fname",
            str(self.config_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--tasks",
            "knn",
            "linear_pool",
            "attentive_pool",
            "--device",
            "cpu",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            eval_classification.main()
        finally:
            sys.argv = old_argv

        ea = EventAccumulator(str(self.tb_dir))
        ea.Reload()
        tags = ea.Tags()["scalars"]
        self.assertTrue(any(t.startswith("eval/knn") for t in tags))
        self.assertTrue(any(t.startswith("eval/linear_pool") for t in tags))
        self.assertTrue(any(t.startswith("eval/attentive_pool") for t in tags))
        for tag in tags:
            events = ea.Scalars(tag)
            expected = 1 if tag.startswith("eval/knn") else 2  # linear_pool/attentive_pool ran 2 head epochs
            self.assertEqual(len(events), expected, f"{tag}: expected {expected} points, got {len(events)}")


if __name__ == "__main__":
    unittest.main()
