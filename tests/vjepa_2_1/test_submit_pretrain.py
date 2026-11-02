"""The pretraining SLURM script (``app/vjepa_2_1/submit_pretrain.sh``): runs it with ``srun``, ``rm`` and the venv's
python stubbed, and checks where it extracts the data, how it launches ``app.main`` (and that it deletes nothing)
and what it refuses up front.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "app" / "vjepa_2_1" / "submit_pretrain.sh"
DEFAULT_CONFIG = "configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml"


class SubmitPretrainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = self.root / "repo"
        self.calls = self.root / "calls.txt"
        self.beeond = self.root / "beeond"
        bin_dir, venv_bin, kneeno = self.root / "bin", self.repo / ".venv" / "bin", self.root / "KneeNo" / "data"
        for directory in (bin_dir, venv_bin, kneeno, self.beeond, (self.repo / DEFAULT_CONFIG).parent):
            directory.mkdir(parents=True)
        (kneeno / "prepare_data.py").touch()
        (self.repo / DEFAULT_CONFIG).touch()
        (self.repo / "configs" / "other.yaml").touch()
        stubs = {
            "srun": f'#!/bin/bash\necho "srun" >> {self.calls}\nexec "$@"\n',
            "rm": f'#!/bin/bash\necho "rm $*" >> {self.calls}\n',
        }
        for name, text in stubs.items():
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        (venv_bin / "python").write_text(f'#!/bin/bash\necho "python $*" >> {self.calls}\n')
        (venv_bin / "python").chmod(0o755)
        self.env = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_SUBMIT_DIR": str(self.repo),
            "SLURM_JOB_ID": "123",
            "SLURM_NNODES": "1",
            "SLURM_NTASKS": "1",
            "KNEENO_DIR": str(self.root / "KneeNo"),
            "BEEOND": str(self.beeond),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }

    def _run(self, **env):
        self.calls.unlink(missing_ok=True)
        run_env = {k: v for k, v in {**self.env, **env}.items() if v is not None}
        result = subprocess.run(["bash", str(SCRIPT)], env=run_env, capture_output=True, text=True)
        return result, self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_prepares_and_trains(self):
        result, calls = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        # One extraction (BeeOND is one file system for the whole job), then training; no cleanup (BeeOND deletes the
        # data when the job ends)
        self.assertEqual(calls[0], "srun")
        prepare, train = calls[1:]
        self.assertIn("prepare_data.py", prepare)
        self.assertIn(f"--data-root {self.beeond}/kneeno_data", prepare)
        self.assertNotIn("--msk-tar-dir", prepare)
        self.assertEqual(
            train,
            f"python -m app.main --fname {self.repo / DEFAULT_CONFIG} --devices cuda:0 cuda:1 cuda:2 cuda:3",
        )

    def test_devices_follow_cuda_visible_devices(self):
        result, calls = self._run(CUDA_VISIBLE_DEVICES="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(call.endswith("--devices cuda:2") for call in calls), calls)

    def test_config_and_msk_tar_dir(self):
        msk = self.root / "msk_tars"
        msk.mkdir()
        result, calls = self._run(CONFIG="configs/other.yaml", MSK_TAR_DIR=str(msk))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(f"--msk-tar-dir {msk}" in call for call in calls), calls)
        self.assertTrue(any(f"--fname {self.repo / 'configs' / 'other.yaml'}" in call for call in calls), calls)

    def test_refuses_before_extracting(self):
        cases = {
            "no beeond": ({"BEEOND": None}, "BEEOND is not a directory"),
            "missing beeond dir": ({"BEEOND": str(self.root / "nope")}, "BEEOND is not a directory"),
            "missing config": ({"CONFIG": "configs/nope.yaml"}, "Config not found"),
            "missing kneeno": ({"KNEENO_DIR": str(self.root / "nope")}, "KneeNo not found"),
            "missing msk dir": ({"MSK_TAR_DIR": str(self.root / "nope")}, "MSK_TAR_DIR is not a directory"),
            "two nodes": ({"SLURM_NNODES": "2"}, "one node in one task"),
            "four tasks": ({"SLURM_NTASKS": "4"}, "one node in one task"),
            "no gpus": ({"CUDA_VISIBLE_DEVICES": None}, "No GPUs visible"),
        }
        for case, (env, message) in cases.items():
            with self.subTest(case):
                result, calls = self._run(**env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
