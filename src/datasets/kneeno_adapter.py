# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bridges the V-JEPA 2.1 encoder to ``kneeno.evaluation``'s unified feature format.

``VJepa21Adapter`` is the ``kneeno.evaluation.adapter.EncoderAdapter``
implementation this repo hands to ``kneeno.evaluation.ClassificationEvaluator``.
``prepare_input`` is the deterministic (non-augmenting) counterpart of
``app/vjepa_2_1/transforms.py::VideoTransform``. Evaluation must not apply
random resized crop / flip / erasing, only the same resize + normalize.
"""

from logging import getLogger

import torch
import torch.nn.functional as F
from kneeno.evaluation.adapter import EncoderAdapter

logger = getLogger()


class VJepa21Adapter(EncoderAdapter):
    """``EncoderAdapter`` for the V-JEPA 2.1 ``MultiSeqWrapper``-wrapped encoder.

    V-JEPA 2.1 has no cls token (``VisionTransformer.cls_token`` is unconditionally
    ``None``, see ``app/vjepa_2_1/models/vision_transformer.py``) so
    ``has_cls_token`` is always False here, and the KneeNo ``linear`` task is
    unavailable for this model (use ``linear_pool`` instead).
    """

    has_cls_token = False

    def __init__(self, embed_dim, crop_size=256, normalize=((0.5,), (0.5,))):
        self._embed_dim = embed_dim
        self.crop_size = crop_size
        mean, std = normalize
        # ((0.5,), (0.5,)) is the MI normalization set in app/vjepa_2_1/train.py;
        # scaled by 255 to match uint8 pixel range, as VideoTransform does.
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1) * 255.0
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1) * 255.0

    @property
    def embed_dim(self):
        return self._embed_dim

    def prepare_input(self, volume):
        """``(T, H, W, 1)`` raw volume -> ``(1, T, crop_size, crop_size)`` float32.

        Deterministic resize (shorter side -> ``crop_size``) + center crop, then
        normalize -- the eval-time counterpart of ``VideoTransform``'s random
        resized crop. Depth (``T``) is left untouched here; any depth
        resampling is the labeled dataset's job (``series_depth`` /
        ``resample_mode``, matching ``UnlabeledKneeMRIDataset``), so evaluation
        preprocessing mirrors whatever the pretraining run used.
        """
        if not torch.is_tensor(volume):
            volume = torch.as_tensor(volume)
        buffer = volume.float().permute(3, 0, 1, 2)  # (T, H, W, 1) -> (C, T, H, W), C=1

        c, t, h, w = buffer.shape
        if h <= w:
            new_h = self.crop_size
            new_w = max(self.crop_size, int(-(-w * self.crop_size // h)))  # ceil
        else:
            new_h = max(self.crop_size, int(-(-h * self.crop_size // w)))  # ceil
            new_w = self.crop_size

        frames = buffer.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

        top = (new_h - self.crop_size) // 2
        left = (new_w - self.crop_size) // 2
        frames = frames[:, :, top : top + self.crop_size, left : left + self.crop_size]

        buffer = frames.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        buffer = (buffer - self.mean) / self.std
        return buffer

    def forward_features(self, model, batch):
        """``batch``: ``(B, 1, T, crop_size, crop_size)`` (the default ``collate``
        stacks ``prepare_input`` outputs along a new batch axis, which is exactly
        what ``MultiSeqWrapper`` expects for a single fpc bucket).

        Unwraps DDP (if wrapped) before calling the encoder directly, so a frozen
        forward pass under ``torch.no_grad()`` does not trip DDP's backward-pass
        bookkeeping.
        """
        encoder = model.module if hasattr(model, "module") else model
        patches = encoder([batch])[0]  # MultiSeqWrapper(x=[batch]) -> [ (B, N, D) ]
        return {"cls": None, "patches": patches}
