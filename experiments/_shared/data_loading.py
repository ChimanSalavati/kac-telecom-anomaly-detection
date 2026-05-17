"""Common data-loading helpers shared by the public-benchmark notebooks.

These helpers wrap the NPZ + Chronos-residual conventions used by every
KAC and baseline notebook so that no notebook ever hard-codes a path.

Conventions
-----------
Each public dataset cache directory contains:

* ``<DATASET>_train.npz``, ``<DATASET>_val.npz``, ``<DATASET>_test.npz``
  with arrays ``X`` (windows x time x KPI), ``y`` (binary labels),
  optional ``descriptions`` (text summaries), and ``feature_cols``.
* ``features_cache/residuals_{train,val,test}.npy`` produced by
  ``scripts/compute_chronos_residuals.py``.

The Production benchmark uses a slightly different schema and is not
shipped in this repository (see ``docs/data_availability.md``); the
identical code path is still defined in
:mod:`experiments._shared.kac_ablation` and can be re-pointed at an
internal data root if you have authorised access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .notebook_helpers import dataset_cache_dir


_DATASET_FILENAMES = {
    "telecomts": ("TelecomTS_train.npz", "TelecomTS_val.npz", "TelecomTS_test.npz"),
    "spotlight": ("SpotLight_train.npz", "SpotLight_val.npz", "SpotLight_test.npz"),
}


def public_dataset_root(dataset: str, override: Optional[Path] = None) -> Path:
    """Resolve the cache directory of a public dataset.

    ``override`` lets users point to a non-default location (e.g. a
    shared scratch volume) without editing the notebooks.
    """
    if override is not None:
        return Path(override)
    return dataset_cache_dir(dataset)


def load_public_split(dataset: str, split: str, root: Optional[Path] = None):
    """Return the raw NPZ for one split of a public dataset."""
    key = dataset.strip().lower()
    if key not in _DATASET_FILENAMES:
        raise ValueError(f"Unknown public dataset {dataset!r}.")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Split must be train/val/test, got {split!r}.")
    fname = _DATASET_FILENAMES[key][["train", "val", "test"].index(split)]
    return np.load(public_dataset_root(key, root) / fname, allow_pickle=True)


def residual_cache(dataset: str, root: Optional[Path] = None) -> Path:
    """Return the ``features_cache/`` directory for a public dataset."""
    base = public_dataset_root(dataset, root) / "features_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_residuals(dataset: str, split: str, root: Optional[Path] = None) -> np.ndarray:
    """Load the precomputed Chronos-2 residual tensor for one split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Split must be train/val/test, got {split!r}.")
    return np.load(residual_cache(dataset, root) / f"residuals_{split}.npy").astype(
        np.float32
    )
