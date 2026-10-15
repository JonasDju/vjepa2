# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the configurable QK-normalization + the configurable context-loss
lambda warmup window.

No GPU on the dev box, so everything runs on a tiny synthetic ViT. ``depth`` here is
the number of transformer blocks (a model-architecture hyperparameter, unrelated to
``data.series_depth``): ``VisionTransformer.__init__`` only assigns
``self.hierarchical_layers`` for ``depth`` in {12, 24, 40, 48} and crashes otherwise,
so the fixture uses 12 (the smallest valid value) rather than something faster.

``model.qk_norm`` is a string, one of:
- ``"none"`` -> ``nn.Identity`` on Q and K (upstream behaviour, no new params)
- ``"layer"`` -> ``nn.LayerNorm(head_dim)`` (weight + bias)
- ``"rms"``   -> ``nn.RMSNorm(head_dim)`` (weight only)

Coverage:
- ``get_norm_layer`` maps the strings (case-insensitive) and rejects the rest.
- ``qk_norm="none"`` (or omitted) == today: ``q_norm``/``k_norm`` are ``nn.Identity``,
  the forward is numerically identical, and no parameters are added.
- ``qk_norm="layer"`` / ``"rms"``: the right norm module on Q and K in both the RoPE
  and the plain attention path; forward + backward finite; learnable params exist.
- ``init_video_model(qk_norm=...)`` wires the norm into the predictor too.
- A ``qk_norm="layer"`` model loads a ``qk_norm="none"`` checkpoint (``strict=False``)
  with only the ``*_norm.*`` keys missing; the reverse only reports them as unexpected.
- ``Lambda_LinearWarmupHold`` honours a custom ``start_iter``/``end_iter``.

Bounded QK logits (``model.qk_norm_affine`` / ``model.qk_temperature_max``):
- the defaults (``True`` / ``None``) build exactly today's model: same state_dict keys, same forward.
- ``qk_norm_affine=False`` drops the norm's gain (and LayerNorm bias); ``_init_weights`` copes.
- ``qk_temperature_max`` adds one ``log_temperature`` per head (init ``t = 1``, so the forward
  equals the non-affine model without it), clamps it to the max, bounds every logit by
  ``t_max * sqrt(head_dim)``, gets gradients, and is rejected in combinations that leave the
  logits unbounded.
- ``init_video_model`` wires both into the predictor too.
- ``load_checkpoint`` refuses a checkpoint whose q/k-norm parameters don't match the model.
"""

import os
import tempfile

import math
import unittest

import torch
import torch.nn as nn

import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.models.utils.modules import (
    Attention,
    Lambda_LinearWarmupHold,
    QKTemperature,
    get_norm_layer,
)
from app.vjepa_2_1.utils import init_opt, init_video_model, load_checkpoint

EMBED_DIM = 16
NUM_HEADS = 2
HEAD_DIM = EMBED_DIM // NUM_HEADS
CROP_SIZE = 32
PATCH_SIZE = 8
TUBELET_SIZE = 2
NUM_FRAMES = 4
DEPTH = 12

# per-block extra parameters each mode adds to one attention module (q_norm + k_norm)
EXTRA_PARAMS_PER_BLOCK = {
    "none": 0,
    "layer": 2 * (2 * HEAD_DIM),  # weight + bias, x2 (q and k)
    "rms": 2 * HEAD_DIM,  # weight only, x2 (q and k)
}


def _make_vit(use_rope, **kw):
    return video_vit.VisionTransformer(
        img_size=CROP_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        in_chans=1,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        use_rope=use_rope,
        modality_embedding=True,
        **kw,
    )


def _clip(bs=2):
    g = torch.Generator().manual_seed(0)
    return torch.randn(bs, 1, NUM_FRAMES, CROP_SIZE, CROP_SIZE, generator=g)


class GetNormLayerTest(unittest.TestCase):
    def test_maps_known_strings_case_insensitive(self):
        self.assertIs(get_norm_layer("none"), nn.Identity)
        self.assertIs(get_norm_layer("layer"), nn.LayerNorm)
        self.assertIs(get_norm_layer("rms"), nn.RMSNorm)
        self.assertIs(get_norm_layer("LAYER"), nn.LayerNorm)
        self.assertIs(get_norm_layer("Rms"), nn.RMSNorm)

    def test_rejects_unknown(self):
        for bad in ("layernorm", "rmsnorm", "", "l2", "true"):
            with self.assertRaises(ValueError):
                get_norm_layer(bad)


class QKNormNoneIsIdentityTest(unittest.TestCase):
    def test_default_is_none_and_identity(self):
        torch.manual_seed(0)
        a = _make_vit(use_rope=True)  # qk_norm omitted -> "none"
        torch.manual_seed(0)
        b = _make_vit(use_rope=True, qk_norm="none")

        self.assertIsInstance(a.blocks[0].attn.q_norm, nn.Identity)
        self.assertIsInstance(a.blocks[0].attn.k_norm, nn.Identity)

        a.eval()
        b.eval()
        x = _clip()
        with torch.no_grad():
            self.assertTrue(torch.equal(a(x), b(x)))

    def test_no_extra_parameters_when_none(self):
        base = sum(p.numel() for p in _make_vit(use_rope=True, qk_norm="none").parameters())
        for mode in ("layer", "rms"):
            n = sum(p.numel() for p in _make_vit(use_rope=True, qk_norm=mode).parameters())
            self.assertEqual(
                n - base,
                DEPTH * EXTRA_PARAMS_PER_BLOCK[mode],
                f"unexpected param delta for qk_norm={mode!r}",
            )


class QKNormEnabledTest(unittest.TestCase):
    def test_rope_path_module_type_and_finite_forward_backward(self):
        for mode, cls in (("layer", nn.LayerNorm), ("rms", nn.RMSNorm)):
            with self.subTest(qk_norm=mode):
                torch.manual_seed(0)
                m = _make_vit(use_rope=True, qk_norm=mode)
                qn = m.blocks[0].attn.q_norm
                self.assertIsInstance(qn, cls)
                self.assertEqual(tuple(qn.weight.shape), (HEAD_DIM,))

                x = _clip()
                out = m(x)
                self.assertTrue(torch.isfinite(out).all())
                self.assertEqual(out.shape[0], x.shape[0])

                out.pow(2).mean().backward()
                self.assertTrue(torch.isfinite(m.blocks[0].attn.q_norm.weight.grad).all())
                self.assertTrue(torch.isfinite(m.blocks[0].attn.k_norm.weight.grad).all())

    def test_plain_attention_path(self):
        off = Attention(EMBED_DIM, num_heads=NUM_HEADS)  # qk_norm omitted -> "none"
        self.assertIsInstance(off.q_norm, nn.Identity)
        self.assertIsInstance(off.k_norm, nn.Identity)

        for mode, cls in (("layer", nn.LayerNorm), ("rms", nn.RMSNorm)):
            with self.subTest(qk_norm=mode):
                torch.manual_seed(0)
                on = Attention(EMBED_DIM, num_heads=NUM_HEADS, qk_norm=mode)
                self.assertIsInstance(on.q_norm, cls)
                self.assertEqual(tuple(on.q_norm.weight.shape), (HEAD_DIM,))

                x = torch.randn(2, 7, EMBED_DIM)
                y = on(x)
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(y.shape, x.shape)
                y.pow(2).mean().backward()
                self.assertTrue(torch.isfinite(on.q_norm.weight.grad).all())


class InitVideoModelPredictorTest(unittest.TestCase):
    def _build(self, qk_norm):
        torch.manual_seed(0)
        return init_video_model(
            device=torch.device("cpu"),
            patch_size=PATCH_SIZE,
            max_num_frames=NUM_FRAMES,
            tubelet_size=TUBELET_SIZE,
            in_chans=1,
            model_name="vit_tiny",
            crop_size=CROP_SIZE,
            pred_depth=DEPTH,
            pred_embed_dim=24,  # divisible by vit_tiny's num_heads (3)
            use_rope=True,
            modality_embedding=True,
            qk_norm=qk_norm,
        )

    def test_encoder_and_predictor_get_qk_norm(self):
        enc, pred = self._build(qk_norm="layer")
        self.assertIsInstance(enc.backbone.blocks[0].attn.q_norm, nn.LayerNorm)
        self.assertIsInstance(
            pred.backbone.predictor_blocks[0].attn.q_norm, nn.LayerNorm
        )
        for name, p in pred.named_parameters():
            self.assertTrue(torch.isfinite(p).all(), name)

    def test_off_by_default(self):
        enc, pred = self._build(qk_norm="none")
        self.assertIsInstance(enc.backbone.blocks[0].attn.q_norm, nn.Identity)
        self.assertIsInstance(
            pred.backbone.predictor_blocks[0].attn.q_norm, nn.Identity
        )

    def test_encoder_forward_backward_finite(self):
        enc, _ = self._build(qk_norm="rms")
        enc.train()
        out = enc([_clip()])  # MultiSeqWrapper no-mask path -> list of tensors
        loss = sum(o.pow(2).mean() for o in out)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        g = enc.backbone.blocks[0].attn.q_norm.weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())


class CheckpointCompatTest(unittest.TestCase):
    def test_qk_norm_model_loads_none_checkpoint_strict_false(self):
        torch.manual_seed(0)
        none_model = _make_vit(use_rope=True, qk_norm="none")
        torch.manual_seed(0)
        layer_model = _make_vit(use_rope=True, qk_norm="layer")

        msg = layer_model.load_state_dict(none_model.state_dict(), strict=False)
        self.assertEqual(list(msg.unexpected_keys), [])
        self.assertTrue(msg.missing_keys)
        self.assertTrue(
            all(".q_norm." in k or ".k_norm." in k for k in msg.missing_keys)
        )

    def test_none_model_loads_qk_norm_checkpoint_strict_false(self):
        torch.manual_seed(0)
        none_model = _make_vit(use_rope=True, qk_norm="none")
        torch.manual_seed(0)
        rms_model = _make_vit(use_rope=True, qk_norm="rms")

        msg = none_model.load_state_dict(rms_model.state_dict(), strict=False)
        self.assertEqual(list(msg.missing_keys), [])
        self.assertTrue(msg.unexpected_keys)
        self.assertTrue(
            all(".q_norm." in k or ".k_norm." in k for k in msg.unexpected_keys)
        )


T_MAX = 4.0


def _plain_attn(**kw):
    """The non-RoPE path, driven via ``Attention`` directly: a ``use_rope=False``
    ``VisionTransformer`` has no ``pos_embed`` and cannot run a forward pass."""
    torch.manual_seed(0)
    return Attention(EMBED_DIM, num_heads=NUM_HEADS, **kw)


class QKNormAffineDefaultTest(unittest.TestCase):
    def test_defaults_build_todays_model(self):
        for mode in ("none", "layer", "rms"):
            with self.subTest(qk_norm=mode):
                torch.manual_seed(0)
                a = _make_vit(use_rope=True, qk_norm=mode)
                torch.manual_seed(0)
                b = _make_vit(use_rope=True, qk_norm=mode, qk_norm_affine=True, qk_temperature_max=None)
                self.assertEqual(list(a.state_dict()), list(b.state_dict()))
                self.assertIsInstance(b.blocks[0].attn.qk_temperature, nn.Identity)
                a.eval()
                b.eval()
                x = _clip()
                with torch.no_grad():
                    self.assertTrue(torch.equal(a(x), b(x)))

                pa = _plain_attn(qk_norm=mode)
                pb = _plain_attn(qk_norm=mode, qk_norm_affine=True, qk_temperature_max=None)
                self.assertEqual(list(pa.state_dict()), list(pb.state_dict()))
                x = torch.randn(2, 7, EMBED_DIM)
                with torch.no_grad():
                    self.assertTrue(torch.equal(pa(x), pb(x)))


class QKNormNonAffineTest(unittest.TestCase):
    def test_no_norm_params_and_init_does_not_crash(self):
        base = sum(p.numel() for p in _make_vit(use_rope=True, qk_norm="none").parameters())
        for mode, cls in (("layer", nn.LayerNorm), ("rms", nn.RMSNorm)):
            with self.subTest(qk_norm=mode):
                m = _make_vit(use_rope=True, qk_norm=mode, qk_norm_affine=False)  # runs _init_weights
                qn = m.blocks[0].attn.q_norm
                self.assertIsInstance(qn, cls)
                self.assertIsNone(qn.weight)
                self.assertEqual(sum(p.numel() for p in m.parameters()), base)
                self.assertTrue(torch.isfinite(m(_clip())).all())

    def test_normalises_q_to_sqrt_head_dim(self):
        for mode in ("layer", "rms"):
            with self.subTest(qk_norm=mode):
                attn = Attention(EMBED_DIM, num_heads=NUM_HEADS, qk_norm=mode, qk_norm_affine=False)
                q = 100 * torch.randn(2, NUM_HEADS, 7, HEAD_DIM)
                norms = attn.q_norm(q).norm(dim=-1)
                self.assertTrue(torch.allclose(norms, torch.full_like(norms, math.sqrt(HEAD_DIM)), rtol=1e-3))


class QKTemperatureTest(unittest.TestCase):
    def _vit(self, mode="rms", use_rope=True, **kw):
        return _make_vit(use_rope=use_rope, qk_norm=mode, qk_norm_affine=False, **kw)

    def test_one_param_per_head_per_block(self):
        base = sum(p.numel() for p in self._vit().parameters())
        n = sum(p.numel() for p in self._vit(qk_temperature_max=T_MAX).parameters())
        self.assertEqual(n - base, DEPTH * NUM_HEADS)
        temp = self._vit(qk_temperature_max=T_MAX).blocks[0].attn.qk_temperature
        self.assertIsInstance(temp, QKTemperature)
        self.assertEqual(tuple(temp.log_temperature.shape), (NUM_HEADS,))

    def test_init_is_identity(self):
        """t = 1 at init, so the forward equals the non-affine model without a temperature."""
        for mode in ("layer", "rms"):
            with self.subTest(qk_norm=mode):
                torch.manual_seed(0)
                a = self._vit(mode)
                torch.manual_seed(0)
                b = self._vit(mode, qk_temperature_max=T_MAX)
                self.assertTrue(torch.equal(b.blocks[0].attn.qk_temperature.temperature(), torch.ones(NUM_HEADS)))
                b.load_state_dict(a.state_dict(), strict=False)
                a.eval()
                b.eval()
                x = _clip()
                with torch.no_grad():
                    self.assertTrue(torch.allclose(a(x), b(x), atol=1e-6))

                pa = _plain_attn(qk_norm=mode, qk_norm_affine=False)
                pb = _plain_attn(qk_norm=mode, qk_norm_affine=False, qk_temperature_max=T_MAX)
                x = torch.randn(2, 7, EMBED_DIM)
                with torch.no_grad():
                    self.assertTrue(torch.allclose(pa(x), pb(x), atol=1e-6))

    def test_clamp_bounds_the_logits(self):
        for mode in ("layer", "rms"):
            with self.subTest(qk_norm=mode):
                attn = Attention(
                    EMBED_DIM, num_heads=NUM_HEADS, qk_norm=mode, qk_norm_affine=False, qk_temperature_max=T_MAX
                )
                with torch.no_grad():
                    attn.qk_temperature.log_temperature.fill_(10.0)  # far above log(T_MAX)
                self.assertTrue(torch.allclose(attn.qk_temperature.temperature(), torch.full((NUM_HEADS,), T_MAX)))

                q = 100 * torch.randn(2, NUM_HEADS, 7, HEAD_DIM)
                logits = attn.qk_temperature(attn.q_norm(q)) @ attn.k_norm(q).transpose(-2, -1) * attn.scale
                cap = T_MAX * math.sqrt(HEAD_DIM)
                self.assertLessEqual(float(logits.abs().max()), cap * (1 + 1e-4))
                self.assertGreater(float(logits.abs().max()), 0.99 * cap)  # q . q hits the cap

    def test_gets_gradients_below_cap_and_none_at_cap(self):
        m = self._vit(qk_temperature_max=T_MAX)
        m(_clip()).pow(2).mean().backward()
        g = m.blocks[0].attn.qk_temperature.log_temperature.grad
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue((g != 0).any())

        attn = Attention(EMBED_DIM, num_heads=NUM_HEADS, qk_norm="rms", qk_norm_affine=False, qk_temperature_max=T_MAX)
        with torch.no_grad():
            attn.qk_temperature.log_temperature.fill_(10.0)
        attn(torch.randn(2, 7, EMBED_DIM)).pow(2).mean().backward()
        self.assertTrue(torch.equal(attn.qk_temperature.log_temperature.grad, torch.zeros(NUM_HEADS)))

    def test_in_no_weight_decay_group(self):
        m = self._vit(qk_temperature_max=T_MAX)
        optimizer, _, _, _ = init_opt(
            is_anneal=False,
            encoder=m,
            predictor=nn.Linear(2, 2),
            iterations_per_epoch=1,
            start_lr=1e-4,
            ref_lr=1e-4,
            warmup=0,
            num_epochs=1,
        )
        temps = {id(mod.log_temperature) for mod in m.modules() if isinstance(mod, QKTemperature)}
        self.assertEqual(len(temps), DEPTH)
        no_wd = {id(p) for g in optimizer.param_groups if g.get("WD_exclude") for p in g["params"]}
        self.assertTrue(temps <= no_wd)

    def test_rejects_unbounded_combinations(self):
        bad = (
            dict(qk_norm="none", qk_norm_affine=False, qk_temperature_max=T_MAX),
            dict(qk_norm="rms", qk_norm_affine=True, qk_temperature_max=T_MAX),
            dict(qk_norm="layer", qk_norm_affine=True, qk_temperature_max=T_MAX),
            dict(qk_norm="rms", qk_norm_affine=False, qk_temperature_max=0.5),
        )
        for kw in bad:
            for use_rope in (True, False):
                with self.subTest(use_rope=use_rope, **kw):
                    with self.assertRaises(ValueError):
                        _make_vit(use_rope=use_rope, **kw)


class InitVideoModelQKTemperatureTest(unittest.TestCase):
    def test_encoder_and_predictor_get_both_options(self):
        torch.manual_seed(0)
        enc, pred = init_video_model(
            device=torch.device("cpu"),
            patch_size=PATCH_SIZE,
            max_num_frames=NUM_FRAMES,
            tubelet_size=TUBELET_SIZE,
            in_chans=1,
            model_name="vit_tiny",
            crop_size=CROP_SIZE,
            pred_depth=DEPTH,
            pred_embed_dim=24,
            use_rope=True,
            modality_embedding=True,
            qk_norm="rms",
            qk_norm_affine=False,
            qk_temperature_max=T_MAX,
        )
        for attn in (enc.backbone.blocks[0].attn, pred.backbone.predictor_blocks[0].attn):
            self.assertIsNone(attn.q_norm.weight)
            self.assertIsInstance(attn.qk_temperature, QKTemperature)
            self.assertEqual(attn.qk_temperature.log_temperature.numel(), attn.num_heads)

        out = enc([_clip()])
        sum(o.pow(2).mean() for o in out).backward()
        g = enc.backbone.blocks[0].attn.qk_temperature.log_temperature.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())


class LoadCheckpointQKGuardTest(unittest.TestCase):
    """``load_checkpoint`` loads ``strict=False``, but must not silently drop q/k-norm params."""

    def _save(self, encoder, path):
        predictor = nn.Linear(2, 2)
        opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()))
        torch.save(
            {
                "epoch": 3,
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "target_encoder": encoder.state_dict(),
                "opt": opt.state_dict(),
                "scaler": None,
            },
            path,
        )

    def _load(self, path, encoder):
        predictor = nn.Linear(2, 2)
        opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()))
        return load_checkpoint(path, encoder, predictor, encoder, opt, None)

    def test_matching_architecture_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "latest.pth.tar")
            kw = dict(qk_norm="rms", qk_norm_affine=False, qk_temperature_max=T_MAX)
            src = _make_vit(use_rope=True, **kw)
            with torch.no_grad():
                src.blocks[0].attn.qk_temperature.log_temperature.fill_(0.5)
            self._save(src, path)
            dst = _make_vit(use_rope=True, **kw)
            *_, epoch = self._load(path, dst)
            self.assertEqual(epoch, 3)
            log_t = dst.blocks[0].attn.qk_temperature.log_temperature
            self.assertTrue(torch.equal(log_t, torch.full((NUM_HEADS,), 0.5)))

    def test_mismatched_qk_norm_raises(self):
        affine = dict(qk_norm="rms")
        bounded = dict(qk_norm="rms", qk_norm_affine=False, qk_temperature_max=T_MAX)
        for src_kw, dst_kw in ((affine, bounded), (bounded, affine), (dict(qk_norm="none"), affine)):
            with self.subTest(src=src_kw, dst=dst_kw), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "latest.pth.tar")
                self._save(_make_vit(use_rope=True, **src_kw), path)
                with self.assertRaisesRegex(ValueError, "q/k norm"):
                    self._load(path, _make_vit(use_rope=True, **dst_kw))


class LambdaWarmupWindowTest(unittest.TestCase):
    def test_custom_start_end_iter(self):
        s = Lambda_LinearWarmupHold(lambda_value=0.5, start_iter=100, end_iter=300)
        self.assertEqual(s.value(99), 0.0)
        self.assertEqual(s.value(100), 0.0)
        self.assertAlmostEqual(s.value(200), 0.25)
        self.assertEqual(s.value(300), 0.5)
        self.assertEqual(s.value(999), 0.5)

    def test_defaults_unchanged(self):
        s = Lambda_LinearWarmupHold(lambda_value=0.5)
        self.assertEqual(s.start, 15000)
        self.assertEqual(s.end, 30000)

    def test_bad_window_raises(self):
        with self.assertRaises(AssertionError):
            Lambda_LinearWarmupHold(lambda_value=0.5, start_iter=300, end_iter=100)


if __name__ == "__main__":
    unittest.main()
