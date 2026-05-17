#!/usr/bin/env python3
"""Download and cache the public TelecomTS corpus.

TelecomTS is published on HuggingFace Datasets as ``AliMaatouk/TelecomTS``
(Feng et al., arXiv:2510.06063). This script fetches the dataset and
materialises the three NPZ files the KAC notebooks expect:

* ``TelecomTS_train.npz``
* ``TelecomTS_val.npz``
* ``TelecomTS_test.npz``

Each NPZ contains:

* ``X``               -- ``(N, T, K)`` KPI windows
* ``y``               -- ``(N,)`` binary labels
* ``descriptions``    -- ``(N,)`` short text summaries
* ``feature_cols``    -- list of KPI names

Files land in ``experiments/_shared/cache/telecomts/``. Re-running the
script is a no-op if the splits already exist.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "experiments" / "_shared" / "cache" / "telecomts"
HF_DATASET_ID = "AliMaatouk/TelecomTS"


def _require_dependencies():
    try:
        import datasets  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print(
            "Missing dependency:",
            exc,
            "\nInstall the project requirements first: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    _require_dependencies()
    import numpy as np
    from datasets import load_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_files = {
        "train": CACHE_DIR / "TelecomTS_train.npz",
        "val":   CACHE_DIR / "TelecomTS_val.npz",
        "test":  CACHE_DIR / "TelecomTS_test.npz",
    }
    if all(p.exists() for p in out_files.values()):
        print(f"TelecomTS already cached at {CACHE_DIR}; nothing to do.")
        return

    print(f"Fetching {HF_DATASET_ID} from HuggingFace ...")
    ds = load_dataset(HF_DATASET_ID)

    split_map = {"train": "train", "val": "validation", "test": "test"}
    for short, hf_name in split_map.items():
        if hf_name not in ds:
            raise RuntimeError(
                f"Expected split {hf_name!r} in the HuggingFace dataset; "
                f"found {list(ds.keys())!r}. The dataset layout may have "
                "changed since the paper was written."
            )
        split = ds[hf_name]
        X = np.asarray(split["X"], dtype=np.float32)
        y = np.asarray(split["y"], dtype=np.int64)
        descriptions = np.asarray(
            split.get("descriptions", [""] * len(split)), dtype=object,
        )
        feature_cols = np.asarray(
            split.features.get("feature_cols", []) or ds[hf_name].column_names,
            dtype=object,
        )
        out = out_files[short]
        np.savez_compressed(
            out, X=X, y=y, descriptions=descriptions, feature_cols=feature_cols,
        )
        print(f"  {short}: {X.shape} -> {out}")

    print(f"\nTelecomTS ready at {CACHE_DIR}")
    print("Next: python scripts/compute_chronos_residuals.py --dataset telecomts")


if __name__ == "__main__":
    main()
