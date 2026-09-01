# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import warnings
from collections import defaultdict
from logging import getLogger

import numpy as np
import torch
from PIL import Image

_GLOBAL_SEED = 0
logger = getLogger()


def _load_metadata(data_meta):
    with open(data_meta, "r") as f:
        return json.load(f)


def _iter_series(metadata, min_len=2, max_len=None):
    """Yield (case_id, series_name, n_images) for every series passing the length filter."""
    for case_id, series_dict in metadata.items():
        for series_name, info in series_dict.items():
            n_images = int(info["n_images"])
            if n_images < min_len:
                continue
            if max_len is not None and n_images > max_len:
                continue
            yield case_id, series_name, n_images


def get_series_depths(data_meta, min_len=2, max_len=None):
    """Return the sorted list of unique series lengths (n_images) in the metadata file.

    Used to populate ``dataset_fpcs`` so that the mask collator builds one mask
    generator per possible volume depth.
    """
    metadata = _load_metadata(data_meta)
    depths = {n for _, _, n in _iter_series(metadata, min_len=min_len, max_len=max_len)}
    if not depths:
        raise ValueError(f"No series found in {data_meta} with length in [{min_len}, {max_len}]")
    return sorted(depths)


class MIDataset(torch.utils.data.Dataset):
    """Medical-imaging dataset.

    Each item is one imaging series (a folder of JPEG slices) loaded as a single
    grayscale 3D volume of shape ``(L, H, W, 1)`` and treated as a video clip.
    No slice sampling is performed: every slice of the series is loaded.
    """

    def __init__(
        self,
        data_root,
        data_meta,
        transform=None,
        series_depth=0,
        min_series_len=2,
        max_series_len=None,
    ):
        self.data_root = data_root
        self.transform = transform
        # series_depth <= 0 -> keep every slice; > 0 -> resample volume to this depth
        self.series_depth = series_depth if (series_depth and series_depth > 0) else 0

        metadata = _load_metadata(data_meta)
        self.samples = list(
            _iter_series(metadata, min_len=min_series_len, max_len=max_series_len)
        )
        if len(self.samples) == 0:
            raise ValueError(f"No usable series found under {data_root} (meta: {data_meta})")

        self.depths = sorted({self.effective_depth(i) for i in range(len(self.samples))})
        logger.info(
            "MIDataset created: %d series, series_depth=%s, depths=%s"
            % (len(self.samples), self.series_depth, self.depths)
        )

    def effective_depth(self, index):
        if self.series_depth > 0:
            return self.series_depth
        return self.samples[index][2]

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, index):
        case_id, series_name, n_images = self.samples[index]
        series_dir = os.path.join(self.data_root, case_id, series_name)

        slices = []
        for i in range(n_images):
            fpath = os.path.join(series_dir, f"{i:03d}.jpeg")
            with Image.open(fpath) as img:
                slices.append(np.asarray(img.convert("L")))  # (H, W) uint8

        vol = np.stack(slices, axis=0)  # (L, H, W)

        if self.series_depth > 0 and vol.shape[0] != self.series_depth:
            idx = np.linspace(0, vol.shape[0] - 1, self.series_depth).round().astype(int)
            vol = vol[idx]

        vol = vol[..., None]  # (L, H, W, 1)
        return vol

    def __getitem__(self, index):
        # Keep trying until a series loads successfully (mirrors VideoDataset).
        for _ in range(10):
            try:
                vol = self._load_volume(index)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Failed to load MI series {self.samples[index]}: {e}")
                index = random.randint(0, len(self.samples) - 1)
                continue

            depth = vol.shape[0]
            buffer = vol
            if self.transform is not None:
                buffer = self.transform(buffer)  # (1, T, H, W) float tensor
            buffer = [buffer]
            label = 0
            clip_indices = [np.arange(depth, dtype=np.int64)]
            return buffer, label, clip_indices

        raise RuntimeError("MIDataset: exceeded max retries while loading a series")


class DistributedDepthBucketSampler(torch.utils.data.Sampler):
    """Batch sampler that groups series of equal depth into the same batch.

    Every yielded value is a list of dataset indices (a full batch) whose series
    all share the same number of slices, so ``default_collate`` inside the mask
    collator can stack them without padding. Batches are sharded across
    distributed replicas; the global batch list is identical on every rank
    (seed depends only on the epoch), so ``batches[rank::num_replicas]`` gives a
    disjoint, equal-sized partition.
    """

    def __init__(
        self,
        depths_per_index,
        batch_size,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=0,
    ):
        self.batch_size = batch_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.depth_to_indices = defaultdict(list)
        for idx, depth in enumerate(depths_per_index):
            self.depth_to_indices[int(depth)].append(idx)

    def _chunk_count(self, n):
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _build(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for depth in sorted(self.depth_to_indices):
            idxs = list(self.depth_to_indices[depth])
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i : i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        batches = batches[:usable]
        return batches[self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(self._build())

    def __len__(self):
        total = sum(
            self._chunk_count(len(idxs)) for idxs in self.depth_to_indices.values()
        )
        return total // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_MIDataset(
    data_root,
    data_meta,
    batch_size,
    transform=None,
    series_depth=0,
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
