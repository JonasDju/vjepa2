# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the config-toggled ViT-22B QK-normalization + the configurable
context-loss lambda warmup window.

No GPU on the dev box, so everything runs on a tiny synthetic ViT (the fixture
shape mirrors the archived ``test_kneeno_adapter.py``: ``depth`` must be in
{12, 24, 40, 48} for ``VisionTransformer``'s ``hierarchical_layers`` quirk).

Coverage:
- ``use_qk_norm=False`` (or omitted) == today: ``q_norm``/``k_norm`` are
  ``nn.Identity`` and the forward is numerically identical.
- ``use_qk_norm=True``: LayerNorm(head_dim) on Q and K in both the RoPE and the
  plain attention path; forward + backward finite; learnable params exist.
- ``init_video_model(use_qk_norm=True)`` wires the norm into the predictor too.
- A ``use_qk_norm=True`` model loads a ``use_qk_norm=False`` checkpoint
  (``strict=False``) with only the ``*_norm.*`` keys missing.
- ``Lambda_LinearWarmupHold`` honours a custom ``start_iter``/``end_iter``.
"""

import unittest

import torch
import torch.nn as nn

import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.models.utils.modules import Attention, Lambda_LinearWarmupHold
from app.vjepa_2_1.utils import init_video_model

EMBED_DIM = 16
NUM_HEADS = 2
HEAD_DIM = EMBED_DIM // NUM_HEADS
CROP_SIZE = 32
PATCH_SIZE = 8
TUBELET_SIZE = 2
NUM_FRAMES = 4
DEPTH = 12


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


class QKNormOffIsIdentityTest(unittest.TestCase):
    def test_rope_off_is_identity_module_and_numerically_equal(self):
        torch.manual_seed(0)
        a = _make_vit(use_rope=True)
        torch.manual_seed(0)
        b = _make_vit(use_rope=True, use_qk_norm=False)

        self.assertIsInstance(a.blocks[0].attn.q_norm, nn.Identity)
        self.assertIsInstance(a.blocks[0].attn.k_norm, nn.Identity)

        a.eval()
        b.eval()
        x = _clip()
        with torch.no_grad():
            self.assertTrue(torch.equal(a(x), b(x)))

    def test_no_extra_parameters_when_off(self):
        torch.manual_seed(0)
        off = _make_vit(use_rope=True, use_qk_norm=False)
        torch.manual_seed(0)
        on = _make_vit(use_rope=True, use_qk_norm=True)
        self.assertEqual(
            sum(p.numel() for p in on.parameters())
            - sum(p.numel() for p in off.parameters()),
            # q_norm + k_norm, each LayerNorm(head_dim) = 2*head_dim params, per block
            DEPTH * 2 * (2 * HEAD_DIM),
        )


class QKNormOnTest(unittest.TestCase):
    def test_rope_path_layernorm_and_finite_forward_backward(self):
        torch.manual_seed(0)
        m = _make_vit(use_rope=True, use_qk_norm=True)
        qn = m.blocks[0].attn.q_norm
        self.assertIsInstance(qn, nn.LayerNorm)
        self.assertEqual(tuple(qn.weight.shape), (HEAD_DIM,))

        x = _clip()
        out = m(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape[0], x.shape[0])

        out.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(m.blocks[0].attn.q_norm.weight.grad).all())
        self.assertTrue(torch.isfinite(m.blocks[0].attn.k_norm.weight.grad).all())

    def test_plain_attention_path(self):
        torch.manual_seed(0)
        off = Attention(EMBED_DIM, num_heads=NUM_HEADS)
        self.assertIsInstance(off.q_norm, nn.Identity)
        self.assertIsInstance(off.k_norm, nn.Identity)

        torch.manual_seed(0)
        on = Attention(EMBED_DIM, num_heads=NUM_HEADS, use_qk_norm=True)
        self.assertIsInstance(on.q_norm, nn.LayerNorm)
        self.assertEqual(tuple(on.q_norm.weight.shape), (HEAD_DIM,))

        x = torch.randn(2, 7, EMBED_DIM)
        y = on(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(y.shape, x.shape)
        y.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(on.q_norm.weight.grad).all())


class InitVideoModelPredictorTest(unittest.TestCase):
    def _build(self, use_qk_norm):
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
            use_qk_norm=use_qk_norm,
        )

    def test_predictor_and_encoder_get_qk_norm(self):
        enc, pred = self._build(use_qk_norm=True)
        self.assertIsInstance(
            enc.backbone.blocks[0].attn.q_norm, nn.LayerNorm
        )
        self.assertIsInstance(
            pred.backbone.predictor_blocks[0].attn.q_norm, nn.LayerNorm
        )
        for name, p in pred.named_parameters():
            self.assertTrue(torch.isfinite(p).all(), name)

    def test_off_by_default(self):
        enc, pred = self._build(use_qk_norm=False)
        self.assertIsInstance(enc.backbone.blocks[0].attn.q_norm, nn.Identity)
        self.assertIsInstance(
            pred.backbone.predictor_blocks[0].attn.q_norm, nn.Identity
        )

    def test_encoder_forward_backward_finite(self):
        enc, _ = self._build(use_qk_norm=True)
        enc.train()
        out = enc([_clip()])  # MultiSeqWrapper no-mask path -> list of tensors
        loss = sum(o.pow(2).mean() for o in out)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        g = enc.backbone.blocks[0].attn.q_norm.weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())


class CheckpointBackCompatTest(unittest.TestCase):
    def test_qk_norm_model_loads_pre_qk_norm_state_dict(self):
        torch.manual_seed(0)
        old = _make_vit(use_rope=True, use_qk_norm=False)
        torch.manual_seed(0)
        new = _make_vit(use_rope=True, use_qk_norm=True)

        msg = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(list(msg.unexpected_keys), [])
        self.assertTrue(msg.missing_keys)
        self.assertTrue(all(".q_norm." in k or ".k_norm." in k for k in msg.missing_keys))


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
