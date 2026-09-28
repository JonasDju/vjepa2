# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 2.1 adapter around the model-agnostic knee-MRI dataset in KneeNo.

Volume loading, metadata parsing and depth resampling live in the ``kneeno``
package so the DINOv2 side can reuse them. This module
only adds the V-JEPA sample format ``(buffer, label, clip_indices)`` and applies
the V-JEPA video transform.
"""

from logging import getLogger

import numpy as np
import torch

from kneeno.dataset import UnlabeledKneeMRIDataset

logger = getLogger(__name__)


class MIDataset(torch.utils.data.Dataset):
    """Wrap ``kneeno.UnlabeledKneeMRIDataset`` to yield V-JEPA training samples.

    Item shape mirrors ``VideoDataset``: ``([tensor (1, D, H, W)], 0, [arange(D)])``
    (``D`` = depth, i.e. the number of slices -- treated as video frames here).

    ``series_depth`` must be positive: every volume is resampled to that many slices, because
    ``MaskCollator`` only handles the frame counts in ``dataset_fpcs`` (``[series_depth]``) and
    a batch has to stack.
    """

    def __init__(
        self,
        data_root,
        data_meta,
        series_depth,
        transform=None,
        resample_mode="nearest",
        min_series_len=2,
        max_series_len=None,
    ):
        check_series_depth(series_depth)
        self._core = UnlabeledKneeMRIDataset(
            data_root=data_root,
            data_meta=data_meta,
            series_depth=series_depth,
            resample_mode=resample_mode,
            min_series_len=min_series_len,
            max_series_len=max_series_len,
        )
        self.transform = transform

    def __len__(self):
        return len(self._core)

    def __getitem__(self, index):
        # kneeno now returns channel-first (1, D, H, W); flip it back to the
        # channel-last (D, H, W, 1) layout that VideoTransform (shared with the
        # genuine video / ImageNet paths, must not be modified) expects. This
        # makes the pre-transform array byte-for-byte what the old loader passed
        # in -- do not "simplify" this flip away.
        vol = self._core[index].numpy()[0, ..., None]  # (1, D, H, W) -> (D, H, W, 1)
        depth = vol.shape[0]
        buffer = vol
        if self.transform is not None:
            buffer = self.transform(buffer)  # (1, D, H, W) float tensor
        buffer = [buffer]
        label = 0
        clip_indices = [np.arange(depth, dtype=np.int64)]
        return buffer, label, clip_indices


def check_series_depth(series_depth, key="data.series_depth"):
    """Raise unless ``series_depth`` is a positive int (V-JEPA 2.1 has no native-depth mode)."""
    if isinstance(series_depth, bool) or not isinstance(series_depth, int) or series_depth <= 0:
        raise ValueError(
            f"{key} must be a positive number of slices for V-JEPA 2.1, got {series_depth!r}; "
            "native per-series depth (<= 0) is not supported"
        )


def make_MIDataset(
    data_root,
    data_meta,
    batch_size,
    series_depth,
    transform=None,
    resample_mode="nearest",
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    min_series_len=2,
    max_series_len=None,
):
    dataset = MIDataset(
        data_root=data_root,
        data_meta=data_meta,
        transform=transform,
        series_depth=series_depth,
        resample_mode=resample_mode,
        min_series_len=min_series_len,
        max_series_len=max_series_len,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("MIDataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler
