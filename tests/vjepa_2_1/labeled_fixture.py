"""Synthetic labeled dataset in the real KneeNo layout (NIfTI volumes + ``extract_labeled_meta.py`` JSON).

Kept local to this repo (KneeNo's own test fixtures are not importable from here); needs SimpleITK,
which ``kneeno`` pulls in as a dependency.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

NUM_CLASSES = 3

# Orientation each series' NIfTI is stored in (matches LabeledKneeMRIDataset.TARGET_ORIENTATION's sources).
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
