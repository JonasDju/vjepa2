"""Synthetic labeled datasets in the real KneeNo layouts.

* :func:`make_labeled_dataset` -- NIfTI volumes + ``extract_labeled_meta.py`` JSON, read by
  ``LabeledExternalKneeMRIDataset``.
* :func:`make_internal_labeled_dataset` -- JPEG slices + ``merge_clinical_labels_into_unlabeled_meta.py``
  JSON, read by ``LabeledInternalKneeMRIDataset`` (the dataset ``ClassificationEvaluator`` builds
  itself from ``eval.data``).

Kept local to this repo (KneeNo's own test fixtures are not importable from here); needs SimpleITK
and Pillow, which ``kneeno`` pulls in as dependencies.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

NUM_CLASSES = 3

# Orientation each series' NIfTI is stored in (matches LabeledExternalKneeMRIDataset.TARGET_ORIENTATION's sources).
SOURCE_ORIENTATION = {"CORONAL_PROTON": "LSA", "SAGITTAL_PROTON": "ASL"}


def make_labeled_dataset(root, spec, h, w, num_classes=NUM_CLASSES, seed=0):
    """Write ``root/<uid>/<series>.nii.gz`` and ``root/metadata.json``; return the metadata path.

    :param spec: ``{uid: {series_name: depth}}`` with series names from ``SOURCE_ORIENTATION``.
    """
    rng = np.random.default_rng(seed)
    meta = {}
    for uid, series in spec.items():
        meta[uid] = {}
        for name, depth in series.items():
            array = (rng.random((depth, h, w)) * 1000 + 1).astype(np.float32)
            image = sitk.GetImageFromArray(array)
            image.SetDirection(sitk.DICOMOrientImageFilter().GetDirectionCosinesFromOrientation(SOURCE_ORIENTATION[name]))
            path = Path(root) / uid / f"{name}.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(image, str(path))
            meta[uid][name] = {"dimensions": [h, w, depth], "data_resolution": [0.3, 0.3, 3.0]}
        meta[uid]["labels"] = [int(v) for v in rng.integers(0, 2, size=num_classes)]
    meta_path = Path(root) / "metadata.json"
    meta_path.write_text(json.dumps(meta))
    return str(meta_path)


def make_internal_labeled_dataset(root, spec, h, w, num_classes=NUM_CLASSES, seed=0):
    """Write ``root/<case_id>/<series>/<NNN>.jpeg`` slices and ``root/metadata.json``; return the metadata path.

    The metadata is ``{"label_names": [...], "cases": {case_id: {series: {"dimensions": [H, W, D]},
    ..., "labels": [...]}}}``, as ``merge_clinical_labels_into_unlabeled_meta.py`` writes it.

    :param spec: ``{case_id: {series_name: depth}}``; any series names work.
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
    meta = {"label_names": [f"label_{i}" for i in range(num_classes)], "cases": cases}
    meta_path = Path(root) / "metadata.json"
    meta_path.write_text(json.dumps(meta))
    return str(meta_path)
