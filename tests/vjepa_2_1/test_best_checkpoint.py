# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Best-model selection in ``app/vjepa_2_1/train.py`` (``eval.model_selection``).

After each in-training evaluation, ``update_best_model`` scores the metrics with a ``kneeno.BestModelTracker`` and,
on a new best, saves a full checkpoint to ``<folder>/best.pth.tar``. Every checkpoint carries the tracker's state
(``best_model``), and ``load_checkpoint`` restores it on resume, but not on a cooldown's fresh start from
``anneal_ckpt``.
"""

import math
import os
import tempfile
import unittest

import torch
import torch.nn as nn
from kneeno import MODEL_SELECTION_KEYS, BestModelTracker

from app.vjepa_2_1.train import update_best_model, write_checkpoint
from app.vjepa_2_1.utils import load_checkpoint

KNN, LINEAR_POOL = MODEL_SELECTION_KEYS


def _metrics(score):
    return {KNN: score, LINEAR_POOL: score}


def _models():
    torch.manual_seed(0)
    encoder, predictor = nn.Linear(3, 3), nn.Linear(3, 3)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()))
    return encoder, predictor, opt


def _save_dict(encoder, predictor, opt, epoch, tracker=None):
    save_dict = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": encoder.state_dict(),
        "opt": opt.state_dict(),
        "scaler": None,
        "epoch": epoch,
    }
    if tracker is not None:
        save_dict["best_model"] = tracker.state_dict()
    return save_dict


class UpdateBestModelTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "best.pth.tar")
        self.tracker = BestModelTracker()
        self.saves = []

    def tearDown(self):
        self._tmp.cleanup()

    def _save_checkpoint(self, epoch, path):
        # what train.py's closure writes: the tracker state at the time of the save
        self.saves.append(epoch)
        return write_checkpoint({"epoch": epoch, "best_model": self.tracker.state_dict()}, path)

    def _update(self, score, epoch):
        return update_best_model(self.tracker, _metrics(score), epoch, self._save_checkpoint, self.path)

    def test_saves_only_on_improvement_and_overwrites(self):
        self.assertTrue(self._update(0.6, 10))
        self.assertFalse(self._update(0.55, 20))
        self.assertFalse(self._update(0.6, 30))  # a tie is no improvement
        self.assertTrue(self._update(0.7, 40))
        self.assertEqual(self.saves, [10, 40])
        saved = torch.load(self.path, weights_only=False)
        self.assertEqual(saved["epoch"], 40)
        self.assertEqual(saved["best_model"], {"best_score": 0.7, "best_epoch": 40})
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_no_score_no_save(self):
        self.assertFalse(update_best_model(self.tracker, {KNN: 0.9}, 10, self._save_checkpoint, self.path))
        self.assertFalse(self._update(math.nan, 20))
        self.assertEqual(self.saves, [])
        self.assertFalse(os.path.exists(self.path))

    def test_failed_save_keeps_the_previous_best(self):
        self._update(0.6, 10)
        self.assertFalse(update_best_model(self.tracker, _metrics(0.7), 20, lambda epoch, path: False, self.path))
        self.assertEqual(self.tracker.state_dict(), {"best_score": 0.6, "best_epoch": 10})
        self.assertTrue(self._update(0.7, 30))  # the next evaluation tries again

    def test_write_checkpoint_failure_returns_false_and_keeps_the_file(self):
        write_checkpoint({"epoch": 1}, self.path)
        missing_dir = os.path.join(self._tmp.name, "missing", "best.pth.tar")
        self.assertFalse(write_checkpoint({"epoch": 2}, missing_dir))
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 1)


class LoadCheckpointTrackerTest(unittest.TestCase):
    def _round_trip(self, tracker, **kwargs):
        encoder, predictor, opt = _models()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "latest.pth.tar")
            write_checkpoint(_save_dict(encoder, predictor, opt, 7, tracker), path)
            restored = BestModelTracker()
            *_, epoch = load_checkpoint(
                path, encoder, predictor, encoder, opt, None, best_model_tracker=restored, **kwargs
            )
        return restored, epoch

    def _tracker(self):
        tracker = BestModelTracker()
        tracker.update(_metrics(0.65), 5)
        return tracker

    def test_resume_restores_the_best(self):
        restored, epoch = self._round_trip(self._tracker())
        self.assertEqual(epoch, 7)
        self.assertEqual(restored.state_dict(), {"best_score": 0.65, "best_epoch": 5})
        self.assertFalse(restored.update(_metrics(0.6), 8))

    def test_anneal_start_ignores_the_stored_best(self):
        restored, _ = self._round_trip(self._tracker(), is_anneal=True)
        self.assertEqual(restored.state_dict(), BestModelTracker().state_dict())

    def test_old_checkpoint_without_the_key_loads_fresh(self):
        with self.assertLogs(level="INFO") as logs:
            restored, _ = self._round_trip(None)
        self.assertEqual(restored.state_dict(), BestModelTracker().state_dict())
        log = "\n".join(logs.output)
        self.assertIn("no model selection state", log)
        self.assertNotIn("restored model selection state", log)

    def test_resume_logs_the_restored_state(self):
        with self.assertLogs(level="INFO") as logs:
            self._round_trip(self._tracker())
        self.assertIn("restored model selection state", "\n".join(logs.output))

    def test_without_tracker_argument(self):
        encoder, predictor, opt = _models()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "latest.pth.tar")
            write_checkpoint(_save_dict(encoder, predictor, opt, 7, self._tracker()), path)
            *_, epoch = load_checkpoint(path, encoder, predictor, encoder, opt, None)
        self.assertEqual(epoch, 7)


if __name__ == "__main__":
    unittest.main()
