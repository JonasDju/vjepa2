"""Synthetic labeled datasets in the real KneeNo layouts.

* :func:`make_labeled_dataset` -- NIfTI volumes + ``merge_labels_into_unlabeled_meta.py`` JSON, read by
  ``LabeledExternalKneeMRIDataset``.
* :func:`make_internal_labeled_dataset` -- JPEG slices + ``merge_labels_into_unlabeled_meta.py`` JSON, read
  by ``LabeledInternalKneeMRIDataset`` (the train part of the ``KneeNoEvalDataset`` that
  ``ClassificationEvaluator`` builds itself from ``eval.data``).
* :func:`make_eval_dataset` -- a ``KneeNoEvalDataset`` over the NIfTI fixture, for injecting into the evaluator.

Both write the exams grouped by global split: ``{"label_names": [...], "splits": {split: {case_id: ...}}}``.

Kept local to this repo (KneeNo's own test fixtures are not importable from here); needs SimpleITK
and Pillow, which ``kneeno`` pulls in as dependencies.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from kneeno.evaluation.dataset import KneeNoEvalDataset, LabeledExternalKneeMRIDataset
from PIL import Image

NUM_CLASSES = 3

# Orientation each series' NIfTI is stored in (matches LabeledExternalKneeMRIDataset.TARGET_ORIENTATION's sources).
SOURCE_ORIENTATION = {
    "sag": "ASL",
    "sagt1": "ASL",
    "cor": "LSA",
    "ax": "LAS",
}

# The labeled datasets only keep exams with all four sequences (their ``SEQUENCES``, in this order).
EXTERNAL_SEQUENCES = ("sag", "sagt1", "cor", "ax")
INTERNAL_SEQUENCES = ("sag", "st1", "cor", "tra")


def full_exam_spec(n_exams, sequences, depths=(6, 5, 4, 7), prefix="c"):
    """``{case_id: {series_name: depth}}`` for ``n_exams`` complete exams (case ids ``<prefix>0``, ...)."""
    return {f"{prefix}{i}": dict(zip(sequences, depths)) for i in range(n_exams)}


def _write_metadata(root, cases, num_classes, split, case_splits):
    """Write ``root/metadata.json``: every case in ``split``, unless ``case_splits`` (``{case_id: split}``)
    names another one."""
    splits = {}
    for case_id, case in cases.items():
        splits.setdefault((case_splits or {}).get(case_id, split), {})[case_id] = case
    meta = {"label_names": [f"label_{i}" for i in range(num_classes)], "splits": splits}
    meta_path = Path(root) / "metadata.json"
    meta_path.write_text(json.dumps(meta))
    return str(meta_path)


def make_labeled_dataset(root, spec, h, w, num_classes=NUM_CLASSES, seed=0, split="test", case_splits=None):
    """Write ``root/<case_id>/<series>.nii.gz`` and ``root/metadata.json``; return the metadata path.

    The metadata is ``{"label_names": [...], "splits": {split: {case_id: {series: {"dimensions": [H, W, D], ...},
    ..., "labels": [...]}}}}``, as ``merge_labels_into_unlabeled_meta.py`` writes it. Every case is in the
    global split ``split`` (the real external metadata only has ``test``) unless ``case_splits`` names another.

    :param spec: ``{case_id: {series_name: depth}}`` with series names from ``SOURCE_ORIENTATION``.
    """
    rng = np.random.default_rng(seed)
    cases = {}
    for case_id, series in spec.items():
        cases[case_id] = {}
        for name, depth in series.items():
            array = (rng.random((depth, h, w)) * 1000 + 1).astype(np.float32)
            image = sitk.GetImageFromArray(array)
            image.SetDirection(sitk.DICOMOrientImageFilter().GetDirectionCosinesFromOrientation(SOURCE_ORIENTATION[name]))
            path = Path(root) / case_id / f"{name}.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(image, str(path))
            cases[case_id][name] = {"dimensions": [h, w, depth], "data_resolution": [0.3, 0.3, 3.0]}
        cases[case_id]["labels"] = [int(v) for v in rng.integers(0, 2, size=num_classes)]
    return _write_metadata(root, cases, num_classes, split, case_splits)


def make_internal_labeled_dataset(
    root, spec, h, w, num_classes=NUM_CLASSES, seed=0, split="train", case_splits=None
):
    """Write ``root/<case_id>/<series>/<NNN>.jpeg`` slices and ``root/metadata.json``; return the metadata path.

    The metadata is ``{"label_names": [...], "splits": {split: {case_id: {series: {"dimensions": [H, W, D]},
    ..., "labels": [...]}}}}``, as ``merge_labels_into_unlabeled_meta.py`` writes it. Every case is in the
    global split ``split`` unless ``case_splits`` (``{case_id: split}``) names another (e.g. ``"val"``, the
    ``internal`` test set).

    :param spec: ``{case_id: {series_name: depth}}``; the dataset only keeps exams with every one of
        ``INTERNAL_SEQUENCES``.
    """
    rng = np.random.default_rng(seed)
    cases = {}
    for case_id, series in spec.items():
        cases[case_id] = {}
        for name, depth in series.items():
            series_dir = Path(root) / case_id / name
            series_dir.mkdir(parents=True, exist_ok=True)
            for i in range(depth):
                Image.fromarray(rng.integers(0, 256, size=(h, w), dtype=np.uint8), mode="L").save(
                    series_dir / f"{i:03d}.jpeg"
                )
            cases[case_id][name] = {"dimensions": [h, w, depth]}
        cases[case_id]["labels"] = [int(v) for v in rng.integers(0, 2, size=num_classes)]
    return _write_metadata(root, cases, num_classes, split, case_splits)


def make_eval_dataset(root, spec, num_test, h, w, **dataset_kwargs):
    """Write the NIfTI fixture and return a ``KneeNoEvalDataset`` over it, to inject into the evaluator.

    The last ``num_test`` cases of ``spec`` are the global split ``test``, read as the test set ``"external"``;
    the others are ``train``, the train part (both read by ``LabeledExternalKneeMRIDataset``: what matters here
    is the split, not the reader). ``dataset_kwargs`` go to both parts.
    """
    case_ids = list(spec)
    case_splits = {case_id: "test" for case_id in case_ids[len(case_ids) - num_test :]}
    meta = make_labeled_dataset(root, spec, h=h, w=w, split="train", case_splits=case_splits)
    train = LabeledExternalKneeMRIDataset(root, meta, split="train", **dataset_kwargs)
    test = LabeledExternalKneeMRIDataset(root, meta, split="test", **dataset_kwargs)
    return KneeNoEvalDataset(train, {"external": test})
