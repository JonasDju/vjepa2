# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA-only: ``compile_model: true`` must not change what a V-JEPA 2.1 training step computes.

Run on a GPU node (skipped without CUDA), from the repo root:

    PYTHONPATH=. .venv/bin/python -m unittest -v tests.vjepa_2_1.test_compile_equivalence_cuda

Environment (all optional):
    VJEPA_COMPILE_TEST_CONFIG      training YAML (default configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml);
                                   model / data / mask / loss / meta.dtype / activation_memory_budget come from it
    VJEPA_COMPILE_TEST_CHECKPOINT  a save_checkpoint .pth.tar trained with this config's model keys (recommended:
                                   realistic activations); otherwise random init (see Setup.build_models)
    VJEPA_COMPILE_TEST_STEPS       training steps compared per check (default 3, min 2)
    VJEPA_COMPILE_TEST_STRICT_BATCH  batch size of the fp32 check (default 4)
    VJEPA_COMPILE_TEST_STRICT_REF_CHUNK  samples per chunk of its float64 reference (default 1: float64 attention
                                   has no memory-efficient kernel, ~30 GB per sample at 256 px x 24 slices)
    VJEPA_COMPILE_TEST_REF_CHUNK   samples per chunk of the fp32 reference of the training-setting check (default 8)

What is compared. The same batch (same volumes, same masks) goes through the uncompiled ("eager") and the
compiled model, built and compiled exactly as train.py does it (``init_video_model`` with the config's
model keys, ``compile_fixes.compile_models``). Per step: the target encoder's output ``h``, the predictor's
``z_pred`` / ``z_context``, ``loss`` / ``loss_pred`` / ``loss_context`` and the gradient of every encoder
and predictor parameter. Steps use different mask shapes (the config's mask generators, stepped like the
collator), so step 2 is the dynamic-shape recompile that used to crash (asserted: the compiled model does
recompile).

Why not bit-identical: Inductor fuses ops and keeps intermediates in fp32 registers, so rounding differs
from eager by design; under bf16 autocast both are only accurate to ~0.4 %. So each comparison uses a
higher-precision reference computed without torch.compile, and requires

    err(compiled, reference) <= FACTOR * err(eager, reference) + FLOOR          (relative L2 errors)

i.e. compiling may not make any output, loss or gradient measurably less accurate than the eager model
already is (FACTOR 2 for outputs / losses / the whole gradient vector, 5 per parameter tensor -- see FACTOR).
Wrong indexing, a dropped/duplicated term or a broken recomputation shows up as an error of order 1, far
above that. Two checks:

1. ``test_1_fp32_strict`` -- float32 with TF32 off at a small batch vs. a float64 reference (chunked like
   the one below): the pure "same math" check, tight floor.
2. ``test_2_training_setting`` -- the config's dtype (bfloat16 autocast), the config's batch size and
   ``activation_memory_budget``, TF32 as in training. The reference is the eager model in float32 (TF32
   off), run in chunks of VJEPA_COMPILE_TEST_REF_CHUNK samples with losses / gradients averaged: every loss
   term is a mean over batch x tokens and samples never interact, so that is exactly the full-batch math at
   a fraction of the memory.

The loss mirrors ``train_step`` in app/vjepa_2_1/train.py (forward_target / forward_context / loss_fn,
context loss weighted by ``lambda_value_vid`` -- the warmup's final value, so the context-loss gradients
are exercised). No DDP (single process; train.py sets ``optimize_ddp = False``, so DDP does not change the
compiled graphs). Takes roughly 20-40 min on an H100, mostly compilation.
"""

import copy
import functools
import gc
import os
import time
import traceback
import unittest

import torch
import torch.nn.functional as F
import yaml

from app.vjepa_2_1.compile_fixes import compile_models
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.utils import init_video_model, normalize_nested
from kneeno.config import expand_env_vars
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.tensors import trunc_normal_

DEFAULT_CONFIG = "configs/train_2_1/vitb16/pretrain-MI-256px-24f.yaml"
EMBED_DIMS = {
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant_xformers": 1408,
    "vit_gigantic_xformers": 1664,
}
# err(compiled) <= FACTOR[kind] * err(eager) + FLOOR[check]. Outputs / losses and the global gradient (all
# parameters as one vector) are large aggregates whose rounding error is stable: compiled / eager stayed within
# 1.03x in dry runs. Single parameter tensors can be tiny sums with heavy cancellation (e.g. the 12-element
# qk_temperature.log_temperature gradients): under bf16 both errors there are essentially random and their
# ratio reached 2.3x, hence 5x. A real defect gives O(1) errors either way.
FACTOR = {"output": 2.0, "grad_global": 2.0, "grad": 5.0}
FLOOR = {"strict": 1e-5, "training": 1e-3}


def env_int(name, default):
    return int(os.environ.get(name, default))


# --------------------------------------------------------------------------- setup (mirrors train.py)


def load_config(path=None):
    with open(path or os.environ.get("VJEPA_COMPILE_TEST_CONFIG", DEFAULT_CONFIG)) as f:
        return expand_env_vars(yaml.safe_load(f))


class Setup:
    """Everything train.py derives from the config that the forward / loss / compile path needs."""

    def __init__(self, cfg):
        m, d, loss, meta = cfg["model"], cfg["data"], cfg["loss"], cfg["meta"]
        self.cfg = cfg
        self.model_name = m["model_name"]
        self.embed_dim = EMBED_DIMS[self.model_name]
        self.levels_predictor = m.get("levels_predictor", 4)
        self.normalize_predictor = m.get("normalize_predictor", False)
        self.img_temporal_dim_size = m.get("img_temporal_dim_size", None)
        self.use_activation_checkpointing = m.get("use_activation_checkpointing", True)
        self.activation_memory_budget = m.get("activation_memory_budget", 0.5)
        self.lambda_value = m.get("lambda_value_vid", 0.0)
        self.series_depth = d["series_depth"]
        self.batch_size = d["batch_size"]
        self.crop_size = d.get("crop_size", 224)
        self.patch_size = d["patch_size"]
        self.tubelet_size = d["tubelet_size"]
        self.grid_size = self.crop_size // self.patch_size
        self.cfgs_mask = cfg["mask"]
        self.loss_exp = loss["loss_exp"]
        self.predict_all = loss.get("predict_all", True)
        self.shift_by_n = loss.get("shift_by_n")
        self.weight_distance_loss = loss.get("weight_distance_loss", False)
        self.offset_context_loss = loss.get("offset_context_loss", False)
        self.has_cls_first = m.get("has_cls_first", False)
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(meta["dtype"].lower(), torch.float32)

    def build_models(self, device):
        m = self.cfg["model"]
        torch.manual_seed(0)
        encoder, predictor = init_video_model(
            uniform_power=m.get("uniform_power", False),
            use_mask_tokens=m.get("use_mask_tokens", False),
            num_mask_tokens=len(self.cfgs_mask),  # len(cfgs_mask) * len(dataset_fpcs), one fpc for MI data
            zero_init_mask_tokens=m.get("zero_init_mask_tokens", True),
            device=device,
            patch_size=self.patch_size,
            max_num_frames=self.series_depth,
            tubelet_size=self.tubelet_size,
            in_chans=1,
            model_name=self.model_name,
            crop_size=self.crop_size,
            pred_depth=m.get("pred_depth"),
            pred_num_heads=m.get("pred_num_heads", None),
            pred_embed_dim=m.get("pred_embed_dim"),
            is_causal=m.get("is_causal", False),
            pred_is_causal=m.get("pred_is_causal", False),
            use_sdpa=self.cfg["meta"].get("use_sdpa", False),
            use_silu=m.get("use_silu", False),
            use_pred_silu=m.get("use_pred_silu", False),
            wide_silu=m.get("wide_silu", True),
            use_rope=m.get("use_rope", False),
            use_activation_checkpointing=self.use_activation_checkpointing,
            return_all_tokens=self.predict_all,
            chop_last_n_tokens=self.shift_by_n,
            init_type=m.get("init_type", "default"),
            img_temporal_dim_size=self.img_temporal_dim_size,
            n_registers=m.get("n_registers", 0),
            n_registers_predictor=m.get("n_registers_predictor", 0),
            has_cls_first=self.has_cls_first,
            interpolate_rope=m.get("interpolate_rope", False),
            modality_embedding=m.get("modality_embedding", False),
            qk_norm=m.get("qk_norm", "none"),
            qk_norm_affine=m.get("qk_norm_affine", True),
            qk_temperature_max=m.get("qk_temperature_max", None),
        )
        target_encoder = copy.deepcopy(encoder)
        ckpt = os.environ.get("VJEPA_COMPILE_TEST_CHECKPOINT")
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            for module, key in ((encoder, "encoder"), (predictor, "predictor"), (target_encoder, "target_encoder")):
                sd = {k.removeprefix("module."): v for k, v in state[key].items()}
                module.load_state_dict(sd, strict=True)
            print(f"weights: {ckpt} (epoch {state.get('epoch')})", flush=True)
        else:
            # At init the mask tokens are exactly zero and the modality embeddings ~1e-6, so every mask token
            # reaches the predictor's first LayerNorm as an almost constant vector. That normalisation is
            # ill-conditioned: float32 vs float64 then differs by ~15 % in z_pred and ~35 % in every gradient
            # for eager and compiled alike, which would leave the comparison without power. Give them normal
            # values instead (trained weights have them anyway); with this all errors are ~1e-6.
            pb = predictor.backbone
            with torch.no_grad():
                for t in list(getattr(pb, "mask_tokens", None) or []) + [
                    getattr(pb, name) for name in ("video_mod_embed", "img_mod_embed") if hasattr(pb, name)
                ]:
                    trunc_normal_(t, std=pb.init_std)
            print("weights: random init (mask tokens / modality embeddings re-drawn, see build_models)", flush=True)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return encoder, target_encoder, predictor

    def make_steps(self, n_steps, batch_size):
        """n_steps batches (CPU): random normalised volumes + masks from the config's mask generators, stepped
        like MaskCollator (different block sizes -> different token counts per step)."""
        collator = MaskCollator(
            cfgs_mask=self.cfgs_mask,
            dataset_fpcs=[self.series_depth],
            crop_size=self.crop_size,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        generators = collator.mask_generators[self.series_depth]
        g = torch.Generator().manual_seed(1234)
        torch.manual_seed(1234)  # block positions come from the global RNG
        steps = []
        for _ in range(n_steps):
            pairs = [gen(batch_size) for gen in generators]
            masks_enc = [[p[0] for p in pairs]]
            masks_pred = [[p[1] for p in pairs]]
            clips = [torch.randn(batch_size, 1, self.series_depth, self.crop_size, self.crop_size, generator=g)]
            steps.append((clips, masks_enc, masks_pred))
        return steps


# --------------------------------------------------------------------------- one training step (train_step)


def training_step(
    s, encoder, target_encoder, predictor, clips, masks_enc, masks_pred, autocast_dtype, distance_weights=None
):
    """Forward + loss + backward exactly like train.py's train_step; returns (outputs, grads) on the CPU.

    ``distance_weights``: precomputed ``compute_mask_distance`` result for exactly these masks (used by the
    chunked references; the function squeezes away the batch dimension at batch size 1).
    """
    embed_dim = s.embed_dim

    def forward_target(c):
        with torch.no_grad():
            h = target_encoder(c, gram_mode=False, training_mode=True)
            new_h = []
            for hi in h:
                if s.levels_predictor > 1:
                    hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                    hi_1 = F.layer_norm(hi[:, :, embed_dim : embed_dim * 2], (embed_dim,))
                    hi_2 = F.layer_norm(hi[:, :, embed_dim * 2 : embed_dim * 3], (embed_dim,))
                    hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                    new_h.append(torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2))
                else:
                    new_h.append(F.layer_norm(hi, (hi.size(-1),)))
            return new_h

    def forward_context(c):
        modality = "video"
        if s.img_temporal_dim_size is not None and c[0].shape[2] == s.img_temporal_dim_size:
            modality = "image"
        z = encoder(c, masks_enc, gram_mode=False, training_mode=True)
        z_pred, z_context = predictor(z, masks_enc, masks_pred, mod=modality)
        if s.normalize_predictor:
            z_pred = normalize_nested(z_pred, embed_dim)
            if s.predict_all:
                z_context = normalize_nested(z_context, embed_dim)
        return z_pred, z_context

    def loss_fn(z, h, masks_to_apply, cls_loss, d_weights):
        if cls_loss:
            h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
            h = [apply_masks(hi[:, 1:], mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
            loss, n = 0, 0
            for zi, hi, hi_cls in zip(z, h, h_cls):
                for zij, hij in zip(zi, hi):
                    h_term = torch.cat([hi_cls, hij], dim=1)
                    loss += torch.mean(torch.abs(zij - h_term) ** s.loss_exp) / s.loss_exp
                    n += 1
            return loss / n
        h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
        loss, n = 0, 0
        if d_weights is not None:
            for zi, hi, d_i in zip(z, h, d_weights):
                for zij, hij, d_ij in zip(zi, hi, d_i):
                    loss_n = torch.abs(zij - hij) ** s.loss_exp * (1 / d_ij.unsqueeze(2))
                    loss += torch.mean(loss_n) / s.loss_exp
                    n += 1
        else:
            for zi, hi in zip(z, h):
                for zij, hij in zip(zi, hi):
                    loss += torch.mean(torch.abs(zij - hij) ** s.loss_exp) / s.loss_exp
                    n += 1
        return loss / n

    for module in (encoder, predictor):
        module.zero_grad(set_to_none=True)
    device_type = clips[0].device.type
    enabled = autocast_dtype is not None
    with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype or torch.float32, enabled=enabled):
        h = forward_target(clips)
        z_pred, z_context = forward_context(clips)
        loss_pred = loss_fn(z_pred, h, masks_pred, cls_loss=s.has_cls_first, d_weights=None)
        loss = loss_pred
        loss_context = None
        if s.predict_all:
            if distance_weights is None:
                distance_weights = compute_mask_distance(masks_pred, masks_enc, s.grid_size, s.offset_context_loss)
            d_weights = distance_weights if s.weight_distance_loss else None
            loss_context = loss_fn(z_context, h, masks_enc, cls_loss=False, d_weights=d_weights)
            loss = loss + loss_context * s.lambda_value
    loss.backward()

    outputs = {"loss": loss, "loss_pred": loss_pred}
    if loss_context is not None:
        outputs["loss_context"] = loss_context
    for i, hi in enumerate(h):
        outputs[f"h[{i}]"] = hi
    for name, nested in (("z_pred", z_pred), ("z_context", z_context)):
        if nested is None:
            continue
        for i, zi in enumerate(nested):
            for j, zij in enumerate(zi):
                if zij is not None:
                    outputs[f"{name}[{i}][{j}]"] = zij
    def keep(t):  # a float64 reference stays float64; everything else is stored as float32
        return t.detach().cpu().to(torch.float64 if t.dtype == torch.float64 else torch.float32)

    outputs = {k: keep(v) for k, v in outputs.items()}
    grads = {}
    for prefix, module in (("encoder", encoder), ("predictor", predictor)):
        for n, p in module.named_parameters():
            if p.grad is not None:
                grads[f"{prefix}.{n}"] = keep(p.grad)
        module.zero_grad(set_to_none=True)
    return outputs, grads


def to_device(step, device, batch_slice=slice(None), dtype=None):
    clips, masks_enc, masks_pred = step
    clips = [c[batch_slice].to(device, dtype=dtype or c.dtype) for c in clips]
    masks_enc = [[m[batch_slice].to(device) for m in ms] for ms in masks_enc]
    masks_pred = [[m[batch_slice].to(device) for m in ms] for ms in masks_pred]
    return clips, masks_enc, masks_pred


def run_steps(s, models, steps, device, autocast_dtype=None, chunk=None, input_dtype=None):
    """Run every step through ``models``; with ``chunk``, in batch chunks averaged back to the full batch."""
    results = []
    for step in steps:
        B = step[0][0].shape[0]
        size = chunk or B
        assert B % size == 0, f"batch {B} not divisible by chunk {size}"
        acc_out, acc_grad = None, None
        for start in range(0, B, size):
            batch = slice(start, start + size)
            inputs = to_device(step, device, batch, input_dtype)
            # distance weights depend only on the masks: compute them for the full batch and slice
            _, full_enc, full_pred = to_device(step, device)
            dist = compute_mask_distance(full_pred, full_enc, s.grid_size, s.offset_context_loss)
            dist = [[d[batch] for d in row] for row in dist]
            out, grads = training_step(s, *models, *inputs, autocast_dtype, distance_weights=dist)
            if acc_out is None:
                acc_out = {k: [v] for k, v in out.items()}
                acc_grad = {k: v.double() / (B // size) for k, v in grads.items()}
            else:
                for k, v in out.items():
                    acc_out[k].append(v)
                assert grads.keys() == acc_grad.keys()
                for k, v in grads.items():
                    acc_grad[k] += v.double() / (B // size)
        # losses are means over equal-sized chunks -> average; per-sample tensors -> concatenate along batch
        out = {
            k: (torch.stack(v).double().mean(0) if v[0].dim() == 0 else torch.cat(v).double())
            for k, v in acc_out.items()
        }
        results.append((out, acc_grad))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------- comparison


def rel_err(a, ref):
    a, ref = a.double(), ref.double()
    num, den = (a - ref).norm().item(), ref.norm().item()
    if den == 0:
        return 0.0 if num == 0 else float("inf")
    return num / den


def flat_grads(grads, names):
    return torch.cat([grads[n].double().flatten() for n in names])


def compare(reference, eager, compiled, floor):
    """Returns (failures, rows) over all steps / tensors; a row = (step, kind, name, err_eager, err_compiled)."""
    failures, rows = [], []
    for step, ((ref_o, ref_g), (e_o, e_g), (c_o, c_g)) in enumerate(zip(reference, eager, compiled)):
        kinds = [("output", ref_o, e_o, c_o), ("grad", ref_g, e_g, c_g)]
        if ref_g.keys() == e_g.keys() == c_g.keys():
            names = sorted(ref_g)
            kinds.append(
                ("grad_global", *({"all parameters": flat_grads(g, names)} for g in (ref_g, e_g, c_g)))
            )
        for kind, ref_d, e_d, c_d in kinds:
            if not (ref_d.keys() == e_d.keys() == c_d.keys()):
                failures.append(f"step {step}: {kind} keys differ: {sorted(set(ref_d) ^ set(c_d))[:5]}")
                continue
            for name in ref_d:
                if c_d[name].shape != ref_d[name].shape:
                    shapes = f"{tuple(c_d[name].shape)} vs {tuple(ref_d[name].shape)}"
                    failures.append(f"step {step} {kind} {name}: shape {shapes}")
                    continue
                ee, ce = rel_err(e_d[name], ref_d[name]), rel_err(c_d[name], ref_d[name])
                rows.append((step, kind, name, ee, ce))
                if not ce <= FACTOR[kind] * ee + floor:
                    failures.append(
                        f"step {step} {kind} {name}: rel err compiled {ce:.3e} > "
                        f"{FACTOR[kind]} x eager {ee:.3e} + {floor:.0e}"
                    )
    return failures, rows


def summarize(title, rows):
    lines = [f"\n=== {title}: relative L2 error vs. reference (eager | compiled) ==="]
    for kind in ("output", "grad_global", "grad"):
        sub = [r for r in rows if r[1] == kind]
        if not sub:
            continue
        worst = max(sub, key=lambda r: r[4] / max(r[3], 1e-12))
        lines.append(
            f"  {kind:6s}: {len(sub)} comparisons, max eager {max(r[3] for r in sub):.2e}, "
            f"max compiled {max(r[4] for r in sub):.2e}; worst ratio step {worst[0]} {worst[2]} "
            f"({worst[3]:.2e} | {worst[4]:.2e})"
        )
    for r in rows:
        if r[1] in ("output", "grad_global"):
            lines.append(f"    step {r[0]} {r[2]:18s} {r[3]:.2e} | {r[4]:.2e}")
    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- the checks


def release_on_error(test):
    """unittest keeps a failed test's traceback -- and with it every frame's locals (model copies, activations,
    GPU tensors) -- until the run ends, so one failing check would starve the next one of GPU memory. Report the
    traceback as text instead and clear the frames."""

    @functools.wraps(test)
    def wrapper(self):
        try:
            return test(self)
        except AssertionError:
            raise  # comparison failures hold only CPU results
        except Exception as e:
            text = "".join(traceback.format_exception(e))
            traceback.clear_frames(e.__traceback__)
            del e
            free_cuda()
            raise RuntimeError(f"check aborted:\n{text}") from None

    return wrapper


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase(name, fn):
    """Run one phase with its time and peak GPU memory printed; frees the cache afterwards."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = fn()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  {name}: {time.time() - t0:.0f}s, peak GPU memory {peak:.1f} GiB", flush=True)
    free_cuda()
    return result


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-only (torch.compile on the training GPUs)")
class CompileEquivalenceCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0")
        cls.s = Setup(load_config())
        cls.n_steps = max(2, env_int("VJEPA_COMPILE_TEST_STEPS", 3))
        cls.base = cls.s.build_models(cls.device)
        torch.backends.cudnn.benchmark = True  # as in train.py

    def setUp(self):
        torch._dynamo.reset()
        self._tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        self._budget = torch._functorch.config.activation_memory_budget

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self._tf32
        torch._functorch.config.activation_memory_budget = self._budget
        torch._dynamo.reset()
        free_cuda()

    def compiled_copy(self, models, activation_memory_budget):
        copies = [copy.deepcopy(m) for m in models]
        compile_models(*copies, activation_memory_budget=activation_memory_budget)
        return copies

    def run_compiled(self, models, steps, **kw):
        """Run the compiled copy step by step; asserts the dynamic-shape recompile actually happened."""
        from torch._dynamo.utils import counters

        results, graphs = [], []
        for step in steps:
            t0 = time.time()
            results += run_steps(self.s, models, [step], self.device, **kw)
            graphs.append(counters["stats"]["unique_graphs"])
            elapsed = time.time() - t0
            print(f"  compiled step {len(results) - 1}: {elapsed:.0f}s, unique graphs {graphs[-1]}", flush=True)
        self.assertGreater(graphs[-1], graphs[0], "the compiled model never recompiled for new mask shapes")
        return results

    @release_on_error
    def test_1_fp32_strict(self):
        s, dev = self.s, self.device
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        steps = s.make_steps(self.n_steps, env_int("VJEPA_COMPILE_TEST_STRICT_BATCH", 4))
        ref_chunk = env_int("VJEPA_COMPILE_TEST_STRICT_REF_CHUNK", 1)
        ref_models = [copy.deepcopy(m).double() for m in self.base]
        reference = phase(
            "reference (float64)",
            lambda: run_steps(s, ref_models, steps, dev, input_dtype=torch.float64, chunk=ref_chunk),
        )
        del ref_models
        eager = phase("eager (float32)", lambda: run_steps(s, self.base, steps, dev))
        budget = None if s.use_activation_checkpointing else s.activation_memory_budget
        compiled = phase("compiled (float32)", lambda: self.run_compiled(self.compiled_copy(self.base, budget), steps))
        failures, rows = compare(reference, eager, compiled, FLOOR["strict"])
        summarize("fp32 strict (reference: eager float64)", rows)
        self.assertFalse(failures, "\n".join(failures[:30]))

    @release_on_error
    def test_2_training_setting(self):
        s, dev = self.s, self.device
        steps = s.make_steps(self.n_steps, s.batch_size)
        autocast = s.dtype if s.dtype != torch.float32 else None
        # reference: eager float32, TF32 off, chunked (exactly the full-batch math, see module docstring)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        ref_chunk = env_int("VJEPA_COMPILE_TEST_REF_CHUNK", 8)
        reference = phase("reference (float32)", lambda: run_steps(s, self.base, steps, dev, chunk=ref_chunk))
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self._tf32  # training defaults
        eager = phase(f"eager ({s.dtype})", lambda: run_steps(s, self.base, steps, dev, autocast_dtype=autocast))
        budget = None if s.use_activation_checkpointing else s.activation_memory_budget
        compiled = phase(
            f"compiled ({s.dtype})",
            lambda: self.run_compiled(self.compiled_copy(self.base, budget), steps, autocast_dtype=autocast),
        )
        failures, rows = compare(reference, eager, compiled, FLOOR["training"])
        title = f"training setting: {s.dtype}, batch {s.batch_size}, budget {budget} (reference: eager float32)"
        summarize(title, rows)
        self.assertFalse(failures, "\n".join(failures[:30]))


if __name__ == "__main__":
    unittest.main()
