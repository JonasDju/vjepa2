"""``app/pretrain_job_id.py``: ``PRETRAIN_JOB_ID`` defaults to ``SLURM_JOB_ID``, and a given one must name a run to
resume; the MI training configs are named after it."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from app.pretrain_job_id import resolve_pretrain_job_id

REPO = Path(__file__).parents[2]
MI_CONFIGS = sorted((REPO / "configs" / "train_2_1").glob("*/*-MI-*.yaml"))


class ResolvePretrainJobIdTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.params = {"folder": str(self.tmp / "runs" / "${PRETRAIN_JOB_ID}.vitb")}

    def resolve(self, env):
        """Runs ``resolve_pretrain_job_id`` in ``env`` alone; returns the PRETRAIN_JOB_ID it leaves and its output."""
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(out):
            resolve_pretrain_job_id(self.params)
            return os.environ["PRETRAIN_JOB_ID"], out.getvalue()

    def test_defaults_to_slurm_job_id(self):
        job_id, out = self.resolve({"SLURM_JOB_ID": "777"})
        self.assertEqual(job_id, "777")
        self.assertIn("PRETRAIN_JOB_ID=777 (this job's SLURM_JOB_ID)", out)
        self.assertIn("a new run", out)

    def test_given_id_with_a_checkpoint_resumes(self):
        folder = self.tmp / "runs" / "42.vitb"
        folder.mkdir(parents=True)
        (folder / "latest.pth.tar").touch()
        job_id, out = self.resolve({"SLURM_JOB_ID": "777", "PRETRAIN_JOB_ID": "42"})
        self.assertEqual(job_id, "42")
        self.assertIn("PRETRAIN_JOB_ID=42 (given)", out)
        self.assertIn("latest.pth.tar to resume from", out)

    def test_given_id_without_a_checkpoint_raises_in_slurm(self):
        # A mistyped ID, and a folder that exists without a checkpoint: neither may start a new run.
        for make_folder in (False, True):
            with self.subTest(make_folder=make_folder):
                if make_folder:
                    (self.tmp / "runs" / "42.vitb").mkdir(parents=True)
                with self.assertRaisesRegex(FileNotFoundError, "PRETRAIN_JOB_ID=42 names a run"):
                    self.resolve({"SLURM_JOB_ID": "777", "PRETRAIN_JOB_ID": "42"})

    def test_given_id_outside_slurm_may_be_new(self):
        job_id, out = self.resolve({"PRETRAIN_JOB_ID": "local"})
        self.assertEqual(job_id, "local")
        self.assertIn("a new run", out)

    def test_no_id_raises(self):
        with self.assertRaisesRegex(ValueError, "Neither PRETRAIN_JOB_ID nor SLURM_JOB_ID is set"):
            self.resolve({})
        with self.assertRaisesRegex(ValueError, "PRETRAIN_JOB_ID is set but empty"):
            self.resolve({"SLURM_JOB_ID": "777", "PRETRAIN_JOB_ID": ""})


class MIConfigsTest(unittest.TestCase):
    def test_named_after_pretrain_job_id(self):
        """Never ${SLURM_JOB_ID}, which changes with every job: resuming would need the config edited."""
        self.assertTrue(MI_CONFIGS)
        for path in MI_CONFIGS:
            with self.subTest(path=path.name):
                text = path.read_text()
                self.assertNotIn("SLURM_JOB_ID", text)
                config = yaml.safe_load(text)
                self.assertIn("${PRETRAIN_JOB_ID}", config["folder"])
                for key in ("tensorboard_dir", "per_label_dir"):
                    self.assertTrue(config["eval"]["logging"][key].startswith(config["folder"] + "/"))


if __name__ == "__main__":
    unittest.main()
