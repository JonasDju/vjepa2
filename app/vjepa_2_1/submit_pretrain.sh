#!/bin/bash
# V-JEPA 2.1 pretraining on the KneeNo data (app/main.py) on one node. Submit from the repo root:
#     sbatch app/vjepa_2_1/submit_pretrain.sh
# Every submission is a new run, named after its SLURM job ID (the config's ${PRETRAIN_JOB_ID}). To resume a run, pass
# the ID of its first job; its <folder>/latest.pth.tar must exist (app/pretrain_job_id.py):
#     sbatch --export=ALL,PRETRAIN_JOB_ID=4839286 app/vjepa_2_1/submit_pretrain.sh
# Pass env vars (PRETRAIN_JOB_ID, CONFIG, REPO_DIR, KNEENO_DIR, MSK_TAR_DIR) with --export: RWTH's sbatch submits with
# --export=NONE, so a `VAR=... sbatch ...` prefix never reaches the job.
#
# app/main.py starts one training process per GPU of the job (--devices from CUDA_VISIBLE_DEVICES), so the job has a
# single task and node; more GPUs: --gres=gpu:N. The config's nodes / tasks_per_node / cpus_per_task are only read by
# app/main_distributed.py (submitit), not here. The data goes to $TMP, the job's folder on the node's local SSD
# (deleted when the job ends). Not $BEEOND: that BeeGFS is metadata-bound and starved the loader on millions of small
# JPEGs. KneeNo's prepare_data.py must give the extracted files fresh timestamps (FreshTimesTarFile): the cluster's
# nightly cleaner deletes $TMP files whose atime and mtime are both old, which tar-restored ones are. The job logs to
# slurm_logs/pretrain.<job id>.log in the submit directory (slurm_logs/ is in the repo: SLURM does not create it).
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --job-name=vjepa21-pretrain
#SBATCH --output=slurm_logs/pretrain.%j.log

set -euo pipefail

# --- Locations (override through --export) ---------------------------------------------------------------
# The output folder and everything else about the run are in the config.
# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
CONFIG="${CONFIG:-configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml}"  # relative to REPO_DIR
# The MSK tars (KneeNo/data/msk/preprocess_msk.py's out_dir), extracted to $TMP/kneeno_data/msk. Set it iff the
# config's data.data_root.msk is set (pretraining fails at startup if that folder is missing).
MSK_TAR_DIR="${MSK_TAR_DIR:-}"

# Fail here rather than after the data extraction
[[ "$CONFIG" == /* ]] || CONFIG="$REPO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Config not found at $CONFIG (set CONFIG)" >&2; exit 1; }
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }
[[ -z "$MSK_TAR_DIR" || -d "$MSK_TAR_DIR" ]] || { echo "MSK_TAR_DIR is not a directory: $MSK_TAR_DIR" >&2; exit 1; }
[[ -n "${TMP:-}" && -d "$TMP" ]] || { echo "TMP is not a directory (the job's node-local scratch): ${TMP:-unset}" >&2; exit 1; }
if [[ "${SLURM_NNODES:-1}" != 1 || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "app/main.py runs on one node in one task (it starts the per-GPU processes itself); got" \
         "${SLURM_NNODES:-1} nodes, ${SLURM_NTASKS:-1} tasks" >&2
    exit 1
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "No GPUs visible (CUDA_VISIBLE_DEVICES is empty)" >&2; exit 1; }
# One cuda:<id> per GPU SLURM gave the job; app/main.py sets each process's CUDA_VISIBLE_DEVICES to its <id>.
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
DEVICES=("${GPUS[@]/#/cuda:}")
MSK_ARGS=()
[[ -z "$MSK_TAR_DIR" ]] || MSK_ARGS=(--msk-tar-dir "$MSK_TAR_DIR")

cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"
echo "vjepa2 commit: $(git -C "$REPO_DIR" describe --always --dirty 2>/dev/null || echo unknown)," \
     "KneeNo commit: $(git -C "$KNEENO_DIR" describe --always --dirty 2>/dev/null || echo unknown)"

# --- Data -> $TMP/kneeno_data/internal (the JPEG slices for pretraining and the in-training eval alike), and
# .../msk with MSK_TAR_DIR; each skipped if already populated. No cleanup: $TMP is deleted when the job ends.
srun "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --internal-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
     ${MSK_ARGS[@]+"${MSK_ARGS[@]}"} --pool-size 16 --data-root "$TMP/kneeno_data"

# --- Training ---------------------------------------------------------------------------------------------
"$PYTHON" -m app.main --fname "$CONFIG" --devices "${DEVICES[@]}"