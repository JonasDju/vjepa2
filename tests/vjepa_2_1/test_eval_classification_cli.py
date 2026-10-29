# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end CPU check for the standalone post-training evaluation script
(``app/vjepa_2_1/eval_classification.py``): builds a tiny checkpoint the way
``train.py::save_checkpoint`` would, then drives the script's own ``main()``
through ``sys.argv`` -- exercising config parsing, encoder reconstruction,
checkpoint loading, and the standalone call into ``ClassificationEvaluator`` in one pass.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from kneeno import LabeledExternalKneeMRIDataset
from kneeno.evaluation.splits import write_manifest_for_metadata

from app.vjepa_2_1.utils import init_video_model
from tests.vjepa_2_1.labeled_fixture import (
    EXTERNAL_SEQUENCES,
    INTERNAL_SEQUENCES,
    full_exam_spec,
    make_internal_labeled_dataset,
    make_labeled_dataset,
)
from tests.vjepa_2_1.test_kneeno_adapter import CROP_SIZE, NUM_FRAMES, PATCH_SIZE, TUBELET_SIZE

H, W = 20, 24
# eval_classification.py's defaults for the q/k options (the bounded-QK training recipe): the checkpoints below
# are built with them and the configs omit both keys, so the tests exercise the defaults
QK_NORM_AFFINE = False
QK_TEMPERATURE_MAX = 4.0
# labeled (evaluation) data: the sequences have different depths -> resampled to NUM_FRAMES.
# Internal: 7 exams in the global split "train" (the train part) and 3 in "val" (the "internal" test set).
SPEC = full_exam_spec(10, INTERNAL_SEQUENCES, prefix="i")
VAL_CASES = {f"i{i}": "val" for i in range(7, 10)}
# External: 4 exams, all in "test" (the "external" test set); case ids distinct from the internal ones.
EXTERNAL_SPEC = full_exam_spec(4, EXTERNAL_SEQUENCES, prefix="e")


class EvalClassificationCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._make_checkpoint()

    def _make_datasets(self):
        """Labeled data for the eval block (the script never loads pretraining volumes).

        The script lets ``ClassificationEvaluator`` build its own dataset from ``eval.data`` (see ``_make_config``):
        the internal JPEG slices (train part + ``internal`` test set) and the external NIfTIs (``external`` test set).
        """
        self.data_root = {source: str(self.root / f"labeled_{source}") for source in ("internal", "external")}
        self.label_meta = {
            "internal": make_internal_labeled_dataset(
                self.data_root["internal"], SPEC, h=H, w=W, case_splits=VAL_CASES
            ),
            "external": make_labeled_dataset(self.data_root["external"], EXTERNAL_SPEC, h=H, w=W),
        }
        self.manifest = write_manifest_for_metadata(
            self.label_meta["internal"], self.root / "split_manifest.json", val_fraction=0.3
        )

    def _make_checkpoint(self, in_chans=1, qk_norm="rms"):
        encoder, predictor = init_video_model(
            device=torch.device("cpu"),
            patch_size=PATCH_SIZE,
            max_num_frames=NUM_FRAMES,
            tubelet_size=TUBELET_SIZE,
            in_chans=in_chans,
            model_name="vit_tiny",
            crop_size=CROP_SIZE,
            pred_depth=12,
            pred_embed_dim=32,
            use_rope=True,
            modality_embedding=True,
            qk_norm=qk_norm,
            qk_norm_affine=QK_NORM_AFFINE,
            # the temperature needs a q/k norm; official checkpoints (qk_norm "none") have neither
            qk_temperature_max=None if qk_norm == "none" else QK_TEMPERATURE_MAX,
        )
        # train.py saves the DDP-wrapped modules, so every key carries a "module." prefix
        ddp_state = {f"module.{k}": v for k, v in encoder.state_dict().items()}
        self.checkpoint_path = self.root / "latest.pth.tar"
        torch.save(
            {
                "encoder": ddp_state,
                "predictor": {f"module.{k}": v for k, v in predictor.state_dict().items()},
                "target_encoder": ddp_state,
                "epoch": 3,
                "opt": {},
                "scaler": None,
                "batch_size": 2,
                "world_size": 1,
                "lr": 1e-4,
            },
            self.checkpoint_path,
        )

    def _make_official_checkpoint(self):
        """Mirrors the layout of the official V-JEPA 2.1 checkpoints (e.g. vjepa2_1_vitb_dist_vitG_384.pt):
        RGB patch embedding, no qk_norm, and the EMA encoder stored under "ema_encoder" (no "target_encoder").
        The online encoder differs from the EMA one, so loading the wrong one is detectable."""
        self._make_checkpoint(in_chans=3, qk_norm="none")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        ema_state = checkpoint.pop("target_encoder")
        checkpoint["encoder"] = {k: v + 1.0 for k, v in ema_state.items()}
        checkpoint["ema_encoder"] = ema_state
        torch.save(checkpoint, self.checkpoint_path)
        return ema_state

    def _make_config(self, series_depth, n_channels=None, qk_norm="rms", test_sets=("internal",)):
        # data: only holds the encoder's architecture; its depth is eval.data.series_depth
        self.test_sets = list(test_sets)
        self.tb_dir = self.root / f"tb_{'_'.join(test_sets)}"
        # results.json and params.yaml go to its parent
        self.out_dir = self.root / f"eval_{'_'.join(test_sets)}"
        config = {
            "data": {
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
                "qk_norm": qk_norm,
            },
            "eval": {
                "seed": 1,
                "split": {"manifest": str(self.manifest), "level": 1.0, "rep": 0},
                "data": {
                    "data_root": self.data_root,
                    "label_meta": self.label_meta,
                    "test_sets": self.test_sets,
                    "series_depth": series_depth,
                    "batch_size": 4,
                    "num_workers": 0,
                },
                "logging": {"tensorboard_dir": str(self.tb_dir), "per_label_dir": str(self.out_dir / "per_label")},
                "freq": {"knn": 1, "linear_pool": 1, "attentive_pool": 1},
                "knn": {"k": [3]},
                "linear_pool": {"max_epochs": 2, "batch_size": 4},
                "attentive_pool": {"max_epochs": 2, "batch_size": 4, "num_heads": 4},
            },
        }
        if n_channels is not None:
            config["model"]["n_channels"] = n_channels
        if qk_norm == "none":
            # the default qk_temperature_max (4.0) requires a q/k norm, so a config for a checkpoint without one
            # (e.g. the official ones) must disable it explicitly
            config["model"]["qk_temperature_max"] = None
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.dump(config))

    def test_cli_runs_and_logs_one_test_point_per_task(self):
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        self._run_and_check_tensorboard()

    def test_test_sets_select_the_datasets_and_the_adapters_preprocessing(self):
        # The fixtures' layouts differ (JPEG slices vs NIfTI), so a run only succeeds if the evaluator built the
        # datasets eval.data.test_sets names. The adapter learns each volume's source from the dataset's items:
        # only external volumes are quantized (num_workers is 0, so the spy sees every call).
        self._make_datasets()
        for test_sets in (["internal"], ["internal", "external"]):
            with self.subTest(test_sets=test_sets):
                self._make_config(series_depth=NUM_FRAMES, test_sets=test_sets)
                with mock.patch.object(
                    LabeledExternalKneeMRIDataset, "to_uint8", wraps=LabeledExternalKneeMRIDataset.to_uint8
                ) as to_uint8:
                    self._run_and_check_tensorboard()
                self.assertEqual(to_uint8.called, "external" in test_sets)

    def test_writes_results_json_and_refuses_to_overwrite_it(self):
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        self._run_main()
        results = json.loads((self.out_dir / "results.json").read_text())
        for task in ("knn", "linear_pool", "attentive_pool"):
            self.assertTrue(any(k.startswith(f"internal/{task}/") for k in results), task)
        self.assertIsInstance(results["evaluation_val/linear_pool/val_auroc"], float)
        self.assertEqual(yaml.safe_load((self.out_dir / "params.yaml").read_text())["eval"]["freq"]["knn"], 1)
        # results.json alone is enough to refuse
        (self.out_dir / "params.yaml").unlink()
        shutil.rmtree(self.out_dir / "per_label")
        shutil.rmtree(self.tb_dir)
        with self.assertRaisesRegex(ValueError, "already exist.*results.json"):
            self._run_main()

    def test_results_go_next_to_the_checkpoint_without_per_label_dir(self):
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        config = yaml.safe_load(self.config_path.read_text())
        config["eval"]["logging"]["per_label_dir"] = None
        config["eval"]["freq"] = {"knn": 1}
        self.config_path.write_text(yaml.dump(config))
        self._run_main()
        results = json.loads((self.checkpoint_path.parent / "results.json").read_text())
        self.assertTrue(any(k.startswith("internal/knn/") for k in results), results)
        self.assertFalse(any(self.root.rglob("params.yaml")))

    def test_split_error_leaves_no_params(self):
        # The evaluator builds the dataset and the split; a config they reject leaves nothing that blocks the rerun.
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        config = yaml.safe_load(self.config_path.read_text())
        config["eval"]["split"]["rep"] = 99
        self.config_path.write_text(yaml.dump(config))
        with self.assertRaisesRegex(ValueError, "no split with level"):
            self._run_main()
        self.assertFalse(self.out_dir.exists())
        self.assertFalse(self.tb_dir.exists())

    def test_no_enabled_task_is_rejected_up_front(self):
        # The tasks are those with a non-null eval.freq > 0; KneeNo's defaults are all null.
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        config = yaml.safe_load(self.config_path.read_text())
        config["eval"]["freq"] = {"knn": 0, "linear_pool": None}
        self.config_path.write_text(yaml.dump(config))
        with self.assertRaisesRegex(ValueError, "No task is enabled"):
            self._run_main()

    def test_native_depth_is_rejected_up_front(self):
        # V-JEPA 2.1 has no native-depth mode: the encoder is sized for eval.data.series_depth and
        # VJepa21Adapter keeps whatever depth it gets, so it must be > 0.
        self._make_datasets()
        for series_depth in (0, -1, None):
            with self.subTest(series_depth=series_depth):
                self._make_config(series_depth=series_depth)
                with self.assertRaisesRegex(ValueError, "series_depth must be a positive number"):
                    self._run_main()

    def test_loads_every_weight_of_a_ddp_checkpoint(self):
        # Regression: the "module." prefix of train.py's DDP checkpoints and the config's qk_norm
        # were both dropped once, and load_state_dict(strict=False) skipped the unmatched weights
        # silently -- the script evaluated a randomly initialised encoder. With the default
        # qk_norm_affine: false the q/k norms have no weights; the per-head temperature (which
        # only exists with a q/k norm) shows that the q/k options reached the model.
        from app.vjepa_2_1.eval_classification import build_encoder, load_frozen_encoder

        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        config = yaml.safe_load(self.config_path.read_text())
        encoder = build_encoder(config["data"], config["model"], 1, NUM_FRAMES, torch.device("cpu"))
        encoder = load_frozen_encoder(str(self.checkpoint_path), encoder, "target_encoder")

        saved = torch.load(self.checkpoint_path, map_location="cpu")["target_encoder"]
        loaded = encoder.state_dict()
        self.assertEqual({f"module.{k}" for k in loaded}, set(saved))
        self.assertTrue(any(".qk_temperature." in k for k in loaded))
        for k, v in loaded.items():
            self.assertTrue(torch.equal(v, saved[f"module.{k}"]), k)

    def test_cli_runs_on_official_rgb_checkpoint(self):
        # n_channels: 3 -> RGB encoder, volumes repeated to 3 channels; target -> the official "ema_encoder"
        self._make_official_checkpoint()
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES, n_channels=3, qk_norm="none")
        self._run_and_check_tensorboard()

    def test_target_encoder_falls_back_to_official_ema_encoder(self):
        from app.vjepa_2_1.eval_classification import (
            ENCODER_STATE_DICT_KEYS,
            build_encoder,
            load_frozen_encoder,
        )

        ema_state = self._make_official_checkpoint()
        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES, n_channels=3, qk_norm="none")
        config = yaml.safe_load(self.config_path.read_text())
        encoder = build_encoder(config["data"], config["model"], 3, NUM_FRAMES, torch.device("cpu"))
        encoder = load_frozen_encoder(str(self.checkpoint_path), encoder, ENCODER_STATE_DICT_KEYS["target"])
        for k, v in encoder.state_dict().items():
            self.assertTrue(torch.equal(v, ema_state[f"module.{k}"]), k)

    def test_missing_encoder_key_raises(self):
        from app.vjepa_2_1.eval_classification import build_encoder, load_frozen_encoder

        self._make_datasets()
        self._make_config(series_depth=NUM_FRAMES)
        config = yaml.safe_load(self.config_path.read_text())
        encoder = build_encoder(config["data"], config["model"], 1, NUM_FRAMES, torch.device("cpu"))
        with self.assertRaises(KeyError):
            load_frozen_encoder(str(self.checkpoint_path), encoder, ("ema_encoder",))

    def _run_main(self):
        from app.vjepa_2_1 import eval_classification

        argv = [
            "eval_classification.py",
            "--fname",
            str(self.config_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--device",
            "cpu",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            eval_classification.main()
        finally:
            sys.argv = old_argv

    def _run_and_check_tensorboard(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self._run_main()
        ea = EventAccumulator(str(self.tb_dir))
        ea.Reload()
        tags = ea.Tags()["scalars"]
        for test_set in self.test_sets:
            for task in ("knn", "linear_pool", "attentive_pool"):
                self.assertTrue(any(t.startswith(f"eval/{test_set}/{task}/") for t in tags), (test_set, task))
        self.assertIn("eval/evaluation_val/linear_pool/val_auroc", tags)
        for tag in tags:
            # Only the test metrics are logged, once per task, however many head epochs ran.
            self.assertEqual([e.step for e in ea.Scalars(tag)], [0], tag)


if __name__ == "__main__":
    unittest.main()
