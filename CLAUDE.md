# CLAUDE.md

## What this repo is

A fork of Meta/FAIR's official **V-JEPA 2** codebase, adapted to **pretrain a V-JEPA 2.1
encoder on a medical-imaging dataset** (knee MRI series). Upstream code for V-JEPA 2,
V-JEPA 2-AC, V-JEPA 2.1 and all evals is otherwise unchanged.

**Scope rule:** only the `app: vjepa_2_1` path is in scope for this project
(`app/vjepa_2_1/`, `src/datasets/`, `src/masks/`, `src/models/`). Do **not** modify
`app/vjepa/`, `app/vjepa_droid/`, or `evals/` unless explicitly asked.

## Environment

- Python **3.12**, dependencies managed with **`uv`** (`pyproject.toml` + `uv.lock`).
- Run Python as **`.venv/bin/python`** — the system Python has no torch.
- **`kneeno`** (the sibling `KneeNo/` repo) is an editable path dependency
  (`pyproject.toml` `dependencies` + `[tool.uv.sources]`); `uv sync` installs it. It owns
  the model-agnostic knee-MRI volume loader; `src/datasets/mi_dataset.py` is a thin
  wrapper over it. KneeNo depends on neither model repo.
- The dev box has an **old NVIDIA driver (CUDA 11050)** → `torch.cuda` is unavailable
  here. Anything requiring a GPU or a process group must run on the cluster. Logic is
  verified on CPU with small synthetic fixtures instead (see Testing).
- Code style: **black** / **isort** with `line-length = 119`; `.flake8` is enforced.

## Running training

Local / single GPU (also inside an sbatch script):
```bash
.venv/bin/python -m app.main \
  --fname configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml \
  --devices cuda:0
```
- `--devices` = one forked process per listed device. **Its default is 8 GPUs**, so
  omitting it makes `app.main` try to build an 8-rank NCCL group and fail with
  `Default process group has not been initialized`. Always pass `--devices cuda:0` for
  single-GPU runs. `--debugmode true` runs in-process (no forking).
- `cuda:0` means "device index 0 *after* the code overwrites `CUDA_VISIBLE_DEVICES`".
  On a cgroup-isolated cluster (the common case) your allocated GPU is always re-indexed
  to 0, so `cuda:0` is correct regardless of the physical index.
- `nodes` / `tasks_per_node` in the YAML are **only** read by `app/main_distributed.py`
  (the `submitit` launcher run from a login node), not by `app/main.py`.
- The progress line (every `log_freq` = 10 iterations) includes `[grad: …]`: the global L2
  norm of the encoder + predictor gradients of that iteration, after the DDP all-reduce and
  before the optimizer step and before any clipping (so it is the pre-clip norm); `nan` when
  the step was skipped. The same value is written for *every* iteration to the column
  (`grad-norm`) of `<folder>/log_r<rank>.csv`.

## Stability options (added 2026-09-28 after the block-0 QK-gain runaway)

The shipped MI config enables all of them. The two q/k options default to the bounded-QK recipe in
the config readers (`train.py`, `eval_classification.py`, since 2026-09-29); everything else -- and the
model-level defaults of `init_video_model` / the modules -- reproduces the old behaviour exactly (same
state_dict keys, same step). The old q/k behaviour needs `qk_norm_affine: true` + `qk_temperature_max: null`;
a config with `qk_norm: none` (the default, and the upstream configs) must set `qk_temperature_max: null`.
- `model.qk_norm_affine` (config default `false`): `false` builds the q/k norm without its learnable
  gain (and without the LayerNorm bias). The unbounded gain is what ran away.
- `model.qk_temperature_max` (config default `4.0`): learnable per-head temperature
  (`QKTemperature` in `app/vjepa_2_1/models/utils/modules.py`), stored as `log t`, init 1,
  clamped in forward to `<= max` and multiplied into q. |logit| <= `max * sqrt(head_dim)`.
  Requires `qk_norm != none` and `qk_norm_affine: false` (raises otherwise). The epoch-end log
  line `[epoch N qk-temperature] block:max/heads-at-cap` shows whether heads are pinned at the cap.
- `optimization.gradient_clipping` (default `null`): clips the global grad norm
  (`train.py::clip_gradients`) after `unscale_`/all-reduce and before the step. The epoch summary
  counts `clipped=`.
- `optimization.betas` / `eps` (defaults `[0.9, 0.999]` / `1e-8`). Re-applied after
  `load_checkpoint` (`apply_optimizer_hparams`), because `opt.load_state_dict` restores the
  checkpoint's values.
- `load_checkpoint` loads `strict=False`, but it **raises** if a `q_norm.`/`k_norm.`/`qk_temperature.`
  key is missing or unexpected. A run cannot resume across a change of the qk options; start a
  new `folder:`. `eval_classification.py` forwards both model keys, so its eval config must match
  the checkpoint. Its defaults are the bounded-QK recipe, like train.py's -- an eval config for a checkpoint
  without q/k norm (e.g. the official ones, `qk_norm: none`) must set `qk_temperature_max: null`
  (`tests/vjepa_2_1/test_eval_classification_cli.py` covers both cases).
- Tests: `tests/vjepa_2_1/test_qk_norm.py` (model side, checkpoint guard),
  `tests/vjepa_2_1/test_train_step_guard.py` (clipping, validation, betas on resume).

## Mask truncation (`mask[*].random_truncation`, added 2026-09-29)

`_MaskGenerator.__call__` (`src/masks/multiseq_multiblock3d.py`) cuts every sample's context and target
indices to the batch minimum so they stack. The indices are sorted depth-major, so the upstream prefix
cut (`random_truncation: false`, the default) always drops the **last depth positions**. With 24-slice
volumes at batch 48, the long-range tube mask drew its context almost only from the first ~8 slices, and
the last ~8 slices were rarely context or target in either mask. `random_truncation: true` keeps a
uniformly random subset per sample (global torch RNG, like the block positions), re-sorted. It is set on
both masks of the shipped MI config. Shapes, and the block masks for a given seed, are unchanged. It holds
no state, so a resumed run simply switches behaviour. Tests: `tests/vjepa_2_1/test_mask_truncation.py`.
Background and 3D-mask (`temporal_scale < 1`) suggestions:
`MA/baseline_comparison/ANALYSIS_16.8.vitb.256px.24f_4gpu.md`, addendum.

## The MIDataset adaptation

New dataset type `MIDataset`: each imaging series is a folder of grayscale JPEG slices
treated as a video (slices = frames), one training item per series.

**Where the code lives:** the model-agnostic core (metadata parsing, series
enumeration, volume loading, depth resampling) lives in the `kneeno` package
(`KneeNo/kneeno/`) so the DINOv2 side can reuse it. `src/datasets/mi_dataset.py` only
wraps `kneeno.UnlabeledKneeMRIDataset` to add the V-JEPA sample format and transform, and
keeps `make_MIDataset` plus `check_series_depth`. KneeNo's depth-bucket sampler,
`effective_depth` and `get_series_depths` were removed (2026-09-28) together with this repo's
native-depth mode.

**Data layout**
```
data_root/<case_id>/<series_name>/<NNN>.jpeg      # NNN = 000 .. n_images-1
data_meta  -> metadata.json: { case_id: { series_name: {"dimensions": [H, W, n_images], ...} } }
```
`series_name` ∈ {`cor`, `sag`, `st1`, `tra`} but not every case has all four — iterate
whatever keys exist. `metadata.json` at the repo root is a ~19 MB reference sample
(untracked, keep it that way): **only ever read a few lines of it**.

**Key files**
- `KneeNo/kneeno/dataset.py` — `KneeNoDataset` (base), `UnlabeledKneeMRIDataset` (returns a
  bare channel-first `(1,D,H,W)` tensor), `load_metadata()`.
- `src/datasets/mi_dataset.py` — `MIDataset` (wrapper), `make_MIDataset`. Item shape:
  `([tensor (1,D,H,W)], 0, [arange(D)])` (`D` = depth = slice count, treated as
  video frames). True 1-channel (`in_chans=1`); kneeno returns `(1,D,H,W)` and
  `MIDataset` flips it back to channel-last `(D,H,W,1)` for the shared
  `VideoTransform`. Plain `DistributedSampler`.
- `src/datasets/data_manager.py` — `init_data()` gets a `midataset` branch and the
  `data_root` / `data_meta` / `series_depth` / `resample_mode` kwargs.
- `app/vjepa_2_1/train.py` — all MI wiring guarded by `is_mi_dataset`: `dataset_fpcs =
  [series_depth]`, `in_chans=1`, `normalize=((0.5,),(0.5,))`,
  `meta.enc_static_graph` flag on the encoder DDP wrap. Also fixed a pre-existing bug
  (missing `vit_base -> embed_dim_encoder = 768` case).
- `app/vjepa_2_1/utils.py` — `init_video_model()` gets an `in_chans` param.
- `configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml` — the MI config
  (`img_data:` / `img_mask:` blocks removed — no ImageNet co-training).

**Config keys added to `data:`**
- `series_depth` (**required**, `> 0`): resample every volume to that many slices along
  depth. `dataset_fpcs = [series_depth]` and the encoder's `max_num_frames` follow from it.
  `<= 0` / missing raises a `ValueError` at startup (`check_series_depth`), and so does a
  non-positive `eval.data.series_depth`: V-JEPA 2.1 has no native-depth mode. (KneeNo still
  supports native depth, but only for consumers that resize depth downstream, which neither
  `MaskCollator` nor `VJepa21Adapter` does.)
- `resample_mode`: `"nearest"` = nearest-slice pick
  via `np.linspace` (slices dropped/duplicated, uint8); `"interpolate"` = trilinear
  `F.interpolate` along depth, `align_corners=True` (float32 output).

**Why the collator constraint matters:** `src/masks/multiseq_multiblock3d.py::MaskCollator`
buckets each batch by exact frame count and **silently drops** samples whose depth isn't
in `dataset_fpcs`. With every volume resampled to `series_depth` and `dataset_fpcs =
[series_depth]` that cannot happen — keep it that way if you ever touch either.

## KneeNo classification evaluation

Frozen-encoder, multi-label image/series classification (k-NN, linear, linear-pool,
attentive-pool) lives in `KneeNo/kneeno/evaluation/` — shared with the DINOv2 side, see
`KneeNo/README.md` for the full task/config description. This repo only supplies the
bridge and the wiring.

**Where the code lives**
- `src/datasets/kneeno_adapter.py` — `VJepa21Adapter(EncoderAdapter)`. `prepare_input` is
  the deterministic (non-augmenting) counterpart of `app/vjepa_2_1/transforms.py`'s
  `VideoTransform` (resize + center crop + normalize, no random crop/flip/erase).
  `forward_features` unwraps DDP and calls the `MultiSeqWrapper`-wrapped encoder directly.
  `has_cls_token = False` always — `VisionTransformer.cls_token` is unconditionally `None`
  (`app/vjepa_2_1/models/vision_transformer.py`), so the `linear` task is unavailable for
  this model; use `linear_pool` instead. Selecting `linear` anyway raises `ValueError`.
- `app/vjepa_2_1/train.py` — builds one `ClassificationEvaluator` (guarded by the
  presence of an `eval:` config block) after the encoder/target_encoder are constructed
  and checkpoint-loaded, then calls `evaluator.evaluate(...)` at the end of every epoch
  with whatever tasks `kneeno.evaluation.tasks_due(epoch, evaluator.config["freq"])` returns
  due — so e.g. `knn` (freq 1) and `linear_pool` (freq 5) each log exactly one TensorBoard
  point (their test metrics) per *embedding-model* epoch, keeping their curves aligned despite
  `linear_pool` running several internal head-training epochs (early-stopped on val). Each trained head
  also logs its best epoch's `<task>/val_auroc`: compare epochs by that, not by the test curves. The
  pretraining configs set `eval.data.abort_on_io_error: false` (an exam that fails to load is replaced by a
  random one rather than killing the run).
  Called on every rank; `ClassificationEvaluator` only does work on rank 0 internally
  (barrier + broadcast), so this is safe under DDP without any rank-guarding here.
  `eval.encoder` selects `target_encoder` (EMA, default) or `encoder` (online).
  The waiting ranks sit in a NCCL barrier for the whole evaluation, which is why
  `src/utils/distributed.py` creates the process group with a 180-min timeout
  (`PROCESS_GROUP_TIMEOUT`; NCCL's default of 10 min would abort multi-GPU runs).
- `app/vjepa_2_1/eval_classification.py` — standalone post-training script. Rebuilds an
  encoder via `init_video_model` (must mirror the pretraining architecture exactly, or the
  checkpoint's `state_dict` keys/shapes won't match: `model:`, plus `data:`'s `patch_size` /
  `tubelet_size` / `crop_size`, the only keys it reads from `data:`; the encoder's depth is
  `eval.data.series_depth`, so `n_channels` defaults to 1), loads the requested
  encoder's weights from a `save_checkpoint`-produced `.pth.tar` (stripping the DDP `module.`
  prefix, `strict=True`), freezes it, and calls `evaluate(..., epoch=0)` once. Like the
  in-training hook it logs one point per task, the test metrics of each head's best val epoch;
  `eval.verbose` puts the per-head-epoch val metrics in the log. Its config
  (`configs/eval_2_1/vitb-384/medical_imaging.yaml`) sets `eval.data.abort_on_io_error: true`: an exam
  that fails to load raises instead of being replaced by one of another split. Run as:
  ```bash
  .venv/bin/python -m app.vjepa_2_1.eval_classification \
    --fname configs/eval_2_1/vitb-384/medical_imaging.yaml \
    --checkpoint /path/to/latest.pth.tar --tasks knn linear_pool attentive_pool
  ```
  Don't pass a pretraining config: its `abort_on_io_error: false` lets a broken exam be replaced by one of
  another split.
  **Before the strict-loading fix this script evaluated a randomly initialised encoder** on
  every real checkpoint: it neither stripped `module.` nor forwarded `model.qk_norm`, and
  `load_state_dict(strict=False)` skipped all 182 unmatched keys silently. Any standalone
  results produced before that fix are random-encoder numbers. It also writes to the config's
  `eval.logging` dirs — point those somewhere else than the training run's own dirs.

**Config**: an `eval:` block at the YAML top level (sibling of `data:`/`model:`/etc.),
consumed by both the in-training hook and the standalone script. Since 2026-10-12 it uses
KneeNo's train/val/test protocol: `split.{train,val,test}_fraction`, `training.early_stop_on` /
`early_stop_patience`, `verbose`, and `max_epochs` (not `epochs`) per head task; an old-style
`epochs` or test-only `split` is rejected at load. See
`configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml` for a real example, and
`KneeNo/kneeno/evaluation/config.py::DEFAULT_EVAL_CONFIG` for every key/default. **Trap
to watch for:** config merging is a deep-merge over the defaults, so disabling a task
that has a nonzero *default* frequency (`linear` defaults to 5) requires an *explicit*
`freq: {linear: null, ...}` in the YAML — omitting the key silently leaves the default
enabled. This matters here specifically because `linear` is unavailable for this model
and selecting it raises.

**Status:** `eval.data.dataset_type` picks the labeled dataset the evaluator builds
(`KneeNo/kneeno/evaluation/dataset.py`; layouts/schemas in `KneeNo/README.md`). It has no default (unset raises).
`internal` (what every shipped config sets) is `LabeledInternalKneeMRIDataset`: the JPEG slices
pretraining reads (`eval.data.data_root` = `.../kneeno_data/internal`, `eval.data.label_meta` =
`BigKneeTar/metadata_labeled.json`). `external` is `LabeledExternalKneeMRIDataset`: one reoriented
`.nii.gz` per series under `.../kneeno_data/external` (`prepare_data.py --external-tar`), metadata
from `data/external/extract_unlabeled_meta.py` + `data/merge_labels_into_unlabeled_meta.py`. Its volumes
come back **unscaled, in the file's own dtype** (e.g. int16 in the thousands);
`VJepa21Adapter.prepare_input` rescales them with `LabeledExternalKneeMRIDataset.to_uint8` first, so
`external` works like `internal`. It decides per volume from `source` (`"internal"` / `"external"`, from the
dataset's `(volumes, labels, source)` items, passed by KneeNo), not from the config: the adapter has no
`dataset_type` argument any more, and an unknown `source` raises. `mixed` (train/val on internal, test on every
external exam; `eval.data.data_root` / `label_meta` as `{internal: ..., external: ...}` mappings) therefore works
too. Until 2026-10-12 `train.py` read the eval `dataset_type` into the same variable as the pretraining
`data.dataset_type`, so `init_data` got `"internal"` and failed (since 7cc546e, 2026-10-05). Evaluation is
exam-level: one sample = the four sequence volumes of one exam (see `MA/CLAUDE.md`), and every
`eval.*.batch_size` / `eval.data.subset` counts exams, so the encoder sees 4x `batch_size` volumes. `series_depth` under `eval.data` should match
`data.series_depth` (both must be `> 0`; checked at startup by `train.py`). The standalone
script has no pretraining `data.series_depth`: it builds the encoder with `eval.data.series_depth`
and checks only that one. Set it to the pretraining depth. `SimpleITK` must be
installed in this venv (`uv lock`/`uv sync` after the KneeNo change; locally `uv pip install simpleitk`).

## Env vars in configs

`kneeno.config.expand_env_vars(obj)` (in the `KneeNo/` repo, imported as
`from kneeno.config import expand_env_vars`) recursively expands `$VAR` / `${VAR}` / `~`
in every string value of the parsed YAML (undefined vars left as-is, idempotent). Called
in `app/main.py`, `app/main_distributed.py` (parse-time + again in `Trainer.__call__` on
the compute node), and at the top of `app/vjepa_2_1/train.py::main`. So config paths like
`data_root: $TMP/kneeno_data/internal` and `folder: $WORK/...` work, and job-local vars
(`$TMPDIR`, `$SLURM_*`) resolve on the node, not the submitting shell.

It lives in KneeNo (not this repo) so all three consuming repos share one
implementation — `kneeno.evaluation.config.load_eval_config` uses it too. This repo
no longer has its own `src/utils/config.py`.

## Testing

No GPU here, so verify with small synthetic fixtures on CPU. Committed:
`PYTHONPATH=. .venv/bin/python -m unittest discover -s tests/datasets` — covers the
`MIDataset` wrapper parity (sample == raw loader, transform applied) and a full
`make_MIDataset` + `MaskCollator` flow (`tests/datasets/test_mi_dataset.py`).
`PYTHONPATH=. .venv/bin/python -m unittest discover -s tests/vjepa_2_1` — the KneeNo
evaluation bridge: `VJepa21Adapter` against both a hand-built `VisionTransformer` and the
real `init_video_model` factory (`test_kneeno_adapter.py`), and a full
`app/vjepa_2_1/eval_classification.py` CLI run against a synthetic checkpoint + labeled
dataset, asserting every task logs exactly one TensorBoard point (its test metrics)
(`test_eval_classification_cli.py`, which also checks that a non-positive or missing
`eval.data.series_depth` is rejected up front). The
labeled fixture is `tests/vjepa_2_1/labeled_fixture.py`: `make_labeled_dataset` (tiny NIfTIs named
`sag`/`sagt1`/`cor`/`ax` + `{"label_names", "cases"}` JSON, for `LabeledExternalKneeMRIDataset`; needs
SimpleITK) and
`make_internal_labeled_dataset` (JPEG slices + `{"label_names", "cases"}` JSON, for
`LabeledInternalKneeMRIDataset` — the dataset `ClassificationEvaluator` builds itself from
`eval.data`, so the CLI test must use this one). The model-agnostic core is tested in
`KneeNo/tests/test_dataset.py`, `test_labeled_dataset.py` and `test_evaluation.py`
(`python -m unittest discover -s tests` from `KneeNo/`): both
resample modes, shape/dtype, and all four classification tasks end to end against a
synthetic dataset/encoder. Not yet covered anywhere: a full `app.main` training run with
the `eval:` hook wired in (the epoch-end call in `train.py` is exercised only by code
review + the pieces it calls, not by driving the whole training loop — the existing mask
configs are not sized for a fast CPU smoke run) and a full
`init_video_model(in_chans=1)` forward+backward mirroring `train_step`.

## Gotchas

- When committing new work, `git add` the **new files** too — a project-added module was
  once missed by a `git add -u`-style commit and broke the cluster checkout with
  `ModuleNotFoundError`.
- `src/` has no `__init__.py` anywhere — it relies on namespace packages.
- **`compile_model: true` (torch 2.13)**: full compilation needs `app/vjepa_2_1/compile_fixes.py`
  (`compile_models`, the one compile path used by train.py). Without it, the predictor's first dynamic-shape
  recompile (iteration 1) crashes with Inductor `CantSplit: 18432*sA + 18432*sB not divisible by
  48*sA + 48*sB`: Inductor fuses the residual adds into the next block's `norm1` and can't prove that
  divisibility over the `n_ctxt + n_pred` token axis. The fix is an exact `sympy.cancel` fallback in
  `SizeVarAllocator.statically_known_multiple_of` (`tests/vjepa_2_1/test_compile_fixes.py`; one test documents
  the bug -- if it starts failing after a torch upgrade, the patch can go). The predictor's token reorder is a
  bit-identical `torch.gather` (`predictor.reorder_tokens`, `tests/vjepa_2_1/test_predictor_reorder.py`).
  Cluster: budget 0.9 -> 0.71 s/it, 54 GB (budget 1.0 OOMs); a CUDA equivalence test (compiled vs eager
  against fp64 / chunked-fp32 references) passed in fp32 and bf16 at batch 48.
  History: compiled runs before `fa00bbd` only "worked" because the deprecated
  `torch.backends.cuda.sdp_kernel()` made Dynamo skip the transformer blocks (they ran eager).
  Branches kept for debugging: `debug-predictor-compile` (the CUDA equivalence test
  `tests/vjepa_2_1/test_compile_equivalence_cuda.py`, the `VJEPA_DEBUG_CANTSPLIT` kernel logger
  `app/vjepa_2_1/compile_debug.py`), `debug-sdpa-graph-break` (the old graph-break behaviour; note
  `torch.compiler.disable` placements do NOT reproduce it).
- `metadata.json` and `slurm_log.txt` at the repo root are local artifacts; keep them
  untracked.
