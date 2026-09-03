#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone post-pretraining KneeNo classification evaluation.

Loads a frozen encoder from a V-JEPA 2.1 pretraining checkpoint (as written by
``app/vjepa_2_1/train.py::save_checkpoint``) and hands it to
``kneeno.evaluation.ClassificationEvaluator``, logging the *whole* head
fine-tuning curve to TensorBoard (``log_every_head_epoch=True``) -- unlike the
in-training hook in ``train.py``, which logs one point per embedding-model
epoch, here the fine-tuning procedure itself is what we care about.

Usage::

    .venv/bin/python -m app.vjepa_2_1.eval_classification \\
        --fname configs/train_2_1/vitb16/pretrain-MI-256px-16f.yaml \\
        --checkpoint /path/to/latest.pth.tar \\
        --tasks knn linear_pool attentive_pool
"""

import argparse
import pprint

import torch
import yaml
from app.vjepa_2_1.utils import init_video_model
from kneeno.evaluation import ClassificationEvaluator
from src.datasets.kneeno_adapter import VJepa21Adapter
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.config import expand_env_vars
from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)

# eval.encoder / --encoder value -> key in the checkpoint dict written by save_checkpoint()
ENCODER_STATE_DICT_KEYS = {"target": "target_encoder", "online": "encoder"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fname", type=str, required=True, help="pretraining config (model + data + eval blocks)")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a .pth.tar checkpoint")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=list(ENCODER_STATE_DICT_KEYS),
        default=None,
        help="which encoder to evaluate; defaults to the config's eval.encoder (else 'target', i.e. the EMA encoder)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        choices=["knn", "linear", "linear_pool", "attentive_pool"],
        help="subset of tasks to run; defaults to all four (note: 'linear' is unavailable for "
        "V-JEPA 2.1 -- it has no cls token, see VJepa21Adapter)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="e.g. 'cpu' or 'cuda:0'")
    return parser.parse_args()


def build_encoder(cfgs_data, cfgs_model, in_chans, max_num_frames, device):
    """Rebuild an encoder with the exact architecture the checkpoint was trained
    with (parameter shapes/keys must match), mirroring the init_video_model
    call in app/vjepa_2_1/train.py. The predictor this also returns is unused
    and discarded -- init_video_model always builds both."""
    encoder, _ = init_video_model(
        device=device,
        patch_size=cfgs_data["patch_size"],
        max_num_frames=max_num_frames,
        tubelet_size=cfgs_data["tubelet_size"],
        in_chans=in_chans,
        model_name=cfgs_model["model_name"],
        crop_size=cfgs_data.get("crop_size", 224),
        pred_depth=cfgs_model.get("pred_depth", 12),
        pred_embed_dim=cfgs_model.get("pred_embed_dim", 384),
        uniform_power=cfgs_model.get("uniform_power", False),
        is_causal=cfgs_model.get("is_causal", False),
        use_sdpa=True,
        use_silu=cfgs_model.get("use_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        use_rope=cfgs_model.get("use_rope", False),
        init_type=cfgs_model.get("init_type", "default"),
        img_temporal_dim_size=cfgs_model.get("img_temporal_dim_size", None),
        n_registers=cfgs_model.get("n_registers", 0),
        has_cls_first=cfgs_model.get("has_cls_first", False),
        interpolate_rope=cfgs_model.get("interpolate_rope", False),
        modality_embedding=cfgs_model.get("modality_embedding", False),
    )
    return encoder


def load_frozen_encoder(checkpoint_path, encoder, state_dict_key):
    """Load one encoder's weights from a pretraining checkpoint, freeze it."""
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    pretrained_dict = checkpoint[state_dict_key]
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained {state_dict_key!r} from epoch {checkpoint.get('epoch')} with msg: {msg}")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


def main():
    args = parse_args()
    with open(args.fname, "r") as f:
        config = expand_env_vars(yaml.load(f, Loader=yaml.FullLoader))
    logger.info("loaded config:\n%s", pprint.pformat(config))

    cfgs_eval = config.get("eval")
    if cfgs_eval is None:
        raise ValueError(f"{args.fname} has no 'eval:' block -- nothing to evaluate against")

    cfgs_model = config["model"]
    cfgs_data = config["data"]
    is_mi_dataset = cfgs_data.get("dataset_type", "videodataset").lower() == "midataset"
    in_chans = 1 if is_mi_dataset else 3

    series_depth = cfgs_data.get("series_depth", 0)
    if series_depth and series_depth > 0:
        max_num_frames = series_depth
    else:
        from kneeno import get_series_depths

        max_num_frames = max(get_series_depths(cfgs_data["data_meta"]))

    device = torch.device(args.device)

    encoder = build_encoder(cfgs_data, cfgs_model, in_chans, max_num_frames, device)
    encoder_choice = args.encoder or cfgs_eval.get("encoder", "target")
    encoder = load_frozen_encoder(args.checkpoint, encoder, ENCODER_STATE_DICT_KEYS[encoder_choice])
    encoder.to(device)

    adapter = VJepa21Adapter(
        embed_dim=encoder.embed_dim,
        crop_size=cfgs_data.get("crop_size", 224),
        normalize=((0.5,), (0.5,)) if is_mi_dataset else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )

    evaluator = ClassificationEvaluator(config=cfgs_eval, adapter=adapter, device=device)
    try:
        metrics = evaluator.evaluate(encoder, tasks=args.tasks, epoch=0, log_every_head_epoch=True)
    finally:
        # Flush+close the TensorBoard writer so buffered scalars are not lost
        # if the process exits right after (SummaryWriter's flush_secs default
        # is 120s -- do not rely on it here).
        evaluator.tb.close()

    logger.info("final metrics: %s", metrics)
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
