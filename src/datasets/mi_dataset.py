# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 2.1 adapter around the model-agnostic knee-MRI dataset in KneeNo.

Volume loading, metadata parsing, depth resampling and the depth-bucket sampler
live in the ``kneeno`` package so the DINOv2 side can reuse them. This module
only adds the V-JEPA sample format ``(buffer, label, clip_indices)`` and applies
the V-JEPA video transform.

``get_series_depths`` and ``DistributedDepthBucketSampler`` are re-exported here
so existing imports (``app/vjepa_2_1/train.py``, ``src/datasets/data_manager.py``)
keep working unchanged.
"""

from logging import getLogger

import numpy as np
import torch

from kneeno.dataset import UnlabeledKneeMRIDataset, get_series_depths  # noqa: F401
from kneeno.sampler import DistributedDepthBucketSampler  # noqa: F401

logger = getLogger()


class MIDataset(torch.utils.data.Dataset):
    """Wrap ``kneeno.UnlabeledKneeMRIDataset`` to yield V-JEPA training samples.

    Item shape mirrors ``VideoDataset``: ``([tensor (1, T, H, W)], 0, [arange(T)])``.
    """

    def __init__(
        self,
        data_root,
        data_meta,
        transform=None,
        series_depth=0,
        resample_mode="nearest",
        min_series_len=2,
        max_series_len=None,
    ):
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

    def effective_depth(self, index):
        return self._core.effective_depth(index)

    def __getitem__(self, index):
        # (T, H, W, 1) ndarray -- the same array the old loader passed to the transform.
        vol = self._core[index].numpy()
        depth = vol.shape[0]
        buffer = vol
        if self.transform is not None:
            buffer = self.transform(buffer)  # (1, T, H, W) float tensor
        buffer = [buffer]
        label = 0
        clip_indices = [np.arange(depth, dtype=np.int64)]
        return buffer, label, clip_indices


def make_MIDataset(
    data_root,
    data_meta,
    batch_size,
    transform=None,
    series_depth=0,
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

    use_persistent = (num_workers > 0) and persistent_workers

    if series_depth and series_depth > 0:
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
            persistent_workers=use_persistent,
        )
    else:
        dist_sampler = DistributedDepthBucketSampler(
            depths_per_index=[dataset.effective_depth(i) for i in range(len(dataset))],
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            batch_sampler=dist_sampler,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=use_persistent,
        )

    logger.info("MIDataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler
