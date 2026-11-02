"""``PRETRAIN_JOB_ID``: the ID of a training run, which the MI configs name their output folders after.

The configs' ``folder`` and ``eval.logging`` paths contain ``${PRETRAIN_JOB_ID}``. The launchers (``app/main.py``,
and ``app/main_distributed.py`` on the compute node) set it before the config is expanded:

* unset (the default): this job's ``SLURM_JOB_ID``, so every submission is a new run, and a job SLURM re-queues
  continues its own run;
* set: an earlier run to resume, by the ID of its first job. RWTH's sbatch does not pass a
  ``PRETRAIN_JOB_ID=... sbatch`` prefix to the job, so ``sbatch --export=ALL,PRETRAIN_JOB_ID=<id> ...``. Inside a
  SLURM job its ``<folder>/latest.pth.tar`` must exist, so a mistyped ID fails instead of starting a new run. Outside
  SLURM it must be set, and names a new run as well as an old one.

Resuming itself is ``train.py``'s: with ``meta.load_checkpoint: true`` (or, for a cooldown,
``optimization.resume_anneal: true``) it continues from ``<folder>/latest.pth.tar`` if that exists. The standalone
eval does not use this variable: its config names the checkpoint through ``CHECKPOINT``.
"""

import os
from pathlib import Path

CHECKPOINT_NAME = "latest.pth.tar"


def set_pretrain_job_id() -> bool:
    """Sets ``PRETRAIN_JOB_ID`` to this job's ``SLURM_JOB_ID`` unless it is set; returns whether it was set.

    Must run before the config is expanded: ``expand_env_vars`` leaves an unset variable as it is, which would
    create a literal ``${PRETRAIN_JOB_ID}`` folder. Every process of a job sees the same ``SLURM_JOB_ID``.
    """
    given = os.environ.get("PRETRAIN_JOB_ID")
    if given is not None:
        if not given:
            raise ValueError("PRETRAIN_JOB_ID is set but empty; unset it to start a new run.")
        return True
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise ValueError(
            "Neither PRETRAIN_JOB_ID nor SLURM_JOB_ID is set, and the config's paths may be named after "
            "${PRETRAIN_JOB_ID}. Outside SLURM, set PRETRAIN_JOB_ID yourself."
        )
    os.environ["PRETRAIN_JOB_ID"] = slurm_job_id
    return False


def check_folder(folder: Path, job_id_given: bool) -> None:
    """Raises if a ``PRETRAIN_JOB_ID`` given inside a SLURM job names no run to resume, i.e. ``folder`` has no
    ``latest.pth.tar`` (a mistyped ID would otherwise start a new run); prints which run this is."""
    job_id = os.environ["PRETRAIN_JOB_ID"]
    checkpoint = Path(folder) / CHECKPOINT_NAME
    has_checkpoint = checkpoint.is_file()
    if job_id_given and os.environ.get("SLURM_JOB_ID") and not has_checkpoint:
        raise FileNotFoundError(
            f"PRETRAIN_JOB_ID={job_id} names a run to resume, but '{checkpoint}' does not exist. Check the ID, or "
            "leave PRETRAIN_JOB_ID unset to start a new run."
        )
    source = "given" if job_id_given else "this job's SLURM_JOB_ID"
    state = f"has a {CHECKPOINT_NAME} to resume from" if has_checkpoint else "a new run"
    print(f"PRETRAIN_JOB_ID={job_id} ({source}); folder '{folder}': {state}.", flush=True)


def resolve_pretrain_job_id(params: dict) -> None:
    """``set_pretrain_job_id`` then ``check_folder`` on the (unexpanded) config ``params``' ``folder``."""
    from kneeno.config import expand_env_vars

    job_id_given = set_pretrain_job_id()
    check_folder(Path(expand_env_vars(params["folder"])), job_id_given)
