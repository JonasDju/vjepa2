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
"""

import unittest

import torch
import torch.nn as nn

import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.models.utils.modules import (
    Attention,
    Lambda_LinearWarmupHold,
    get_norm_layer,
)
from app.vjepa_2_1.utils import init_video_model

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
