#!/usr/bin/env python3
"""Build the imbalanced TelecomTS stress split from the cached balanced splits.

The paper evaluates TelecomTS under two scenarios: the benchmark's balanced
split, and a **severely imbalanced stress split** (natural anomaly rate
~3.9%; paper Table 2, "Imbalanced test set"). This script derives the
imbalanced split deterministically from the balanced NPZs produced by
``scripts/download_telecomts.py``:

* keep **all** normal (y=0) windows, and
* **subsample** anomalous (y=1) windows so the resulting anomaly rate equals
  ``--rate`` (default 0.039),

using a fixed RNG seed for reproducibility. It writes, next to the balanced
splits, the files ``main.py`` expects for ``--scenario imbalanced``:

* ``TelecomTS_imbalanced_train.npz``
* ``TelecomTS_imbalanced_val.npz``
* ``TelecomTS_imbalanced_test.npz``

Each output preserves the balanced NPZ schema (``X``, ``y``, ``descriptions``,
``feature_cols``).

Usage::

    python scripts/download_telecomts.py            # produces the balanced splits
    python scripts/build_telecomts_imbalanced.py    # -> imbalanced stress splits
    python scripts/build_telecomts_imbalanced.py --rate 0.039 --seed 42 \
        --splits train val test

Note: this reconstructs an imbalanced split at the requested anomaly rate; the
exact window membership is seed-dependent and is not guaranteed to be
bit-identical to the authors' internal run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "experiments" / "_shared" / "cache" / "telecomts"


def _imbalance_split(src: Path, dst: Path, rate: float, seed: int) -> tuple[int, int]:
    import numpy as np

    d = np.load(src, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"]).astype(np.int64)
    n = len(y)
    descriptions = (
        np.asarray(d["descriptions"], dtype=object)
        if "descriptions" in d.files
        else np.asarray([""] * n, dtype=object)
    )
    feature_cols = (
        d["feature_cols"] if "feature_cols" in d.files else np.asarray([], dtype=object)
    )

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"{src.name} has a single class (pos={len(pos)}, neg={len(neg)}).")

    # Keep all negatives; choose #positives so pos/(pos+neg) == rate.
    target_pos = int(round(rate * len(neg) / (1.0 - rate)))
    target_pos = max(1, min(target_pos, len(pos)))

    rng = np.random.default_rng(seed)
    keep_pos = rng.choice(pos, size=target_pos, replace=False)
    idx = np.concatenate([neg, keep_pos])
    idx.sort()  # preserve original temporal order

    np.savez_compressed(
        dst, X=X[idx], y=y[idx], descriptions=descriptions[idx], feature_cols=feature_cols,
    )
    return len(idx), int(y[idx].sum())


def main(argv: list[str] | None = None) -> int:
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("Install requirements first: pip install -r requirements.txt", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=0.039,
                    help="Target anomaly rate of the imbalanced split (default 0.039).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for positive subsampling.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--cache-dir", default=str(CACHE_DIR),
                    help="Directory holding the balanced TelecomTS_{split}.npz files.")
    args = ap.parse_args(argv)

    cache = Path(args.cache_dir)
    missing = [s for s in args.splits if not (cache / f"TelecomTS_{s}.npz").exists()]
    if missing:
        print(
            f"Balanced split(s) {missing} not found in {cache}. "
            "Run scripts/download_telecomts.py first.",
            file=sys.stderr,
        )
        return 2

    for split in args.splits:
        src = cache / f"TelecomTS_{split}.npz"
        dst = cache / f"TelecomTS_imbalanced_{split}.npz"
        n, n_pos = _imbalance_split(src, dst, rate=args.rate, seed=args.seed)
        print(f"  {split}: {n} windows, {n_pos} anomalies ({n_pos / n:.3%}) -> {dst.name}")

    print(f"\nImbalanced TelecomTS ready at {cache}")
    print("Next: python main.py kac --dataset telecomts --scenario imbalanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
