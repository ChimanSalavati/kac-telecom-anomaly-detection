#!/usr/bin/env python3
"""Compute the Chronos-2 residual cache for a public dataset.

Each KAC and KAC-ablation notebook consumes
``<dataset>/features_cache/residuals_{train,val,test}.npy`` produced by
this script. The cache is deterministic given the pinned Chronos-2
release in ``requirements.txt``.

Usage::

    python scripts/compute_chronos_residuals.py --dataset telecomts
    python scripts/compute_chronos_residuals.py --dataset spotlight

The script appends ``experiments/`` to ``sys.path`` so it can reuse the
shared helpers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _shared.data_loading import (  # noqa: E402  (sys.path tweak above)
    load_public_split,
    public_dataset_root,
    residual_cache,
)


CONTEXT_LENGTH_DEFAULT = 20
QUANTILE_LEVELS = [0.05, 0.5, 0.95]
EPS = 1e-8
FEAT_CHUNK = 50


def _load_pipeline():
    try:
        from chronos import ChronosPipeline  # type: ignore
    except ImportError as exc:
        print(
            "Chronos-2 is not installed:",
            exc,
            "\nInstall it via `pip install chronos-forecasting` (the package "
            "name shipped by the authors).",
            file=sys.stderr,
        )
        sys.exit(1)
    return ChronosPipeline.from_pretrained("amazon/chronos-2-base")


def _process_window(pipeline, X_win, context_length, quantile_levels):
    import numpy as np
    T, F = X_win.shape
    n_preds = T - context_length
    actuals = X_win[context_length:]
    all_resid = np.zeros((n_preds, F * 5), dtype=np.float32)
    for f_start in range(0, F, FEAT_CHUNK):
        f_end = min(f_start + FEAT_CHUNK, F)
        f_size = f_end - f_start
        contexts = np.stack([
            X_win[t - context_length:t, f_start:f_end].T
            for t in range(context_length, T)
        ])
        quantiles, _ = pipeline.predict_quantiles(
            contexts, prediction_length=1, quantile_levels=quantile_levels,
        )
        if not isinstance(quantiles, np.ndarray):
            quantiles = np.array(quantiles)
        q05 = quantiles[..., 0]
        q50 = quantiles[..., 1]
        q95 = quantiles[..., 2]
        a = actuals[:, f_start:f_end]
        width = q95 - q05 + EPS
        z = (a - q50) / width
        block = np.stack(
            [q50.squeeze(-1), a - q50.squeeze(-1), z.squeeze(-1),
             width.squeeze(-1), (a > q95).astype(np.float32) - (a < q05).astype(np.float32).squeeze(-1)],
            axis=-1,
        )
        all_resid[:, f_start * 5:f_end * 5] = block.reshape(n_preds, f_size * 5)
    return all_resid


def _process_split(pipeline, dataset, split, context_length):
    import numpy as np
    npz = load_public_split(dataset, split)
    X = np.asarray(npz["X"], dtype=np.float32)
    N, T, F = X.shape
    n_preds = T - context_length
    out = np.zeros((N, n_preds, F * 5), dtype=np.float32)
    start = time.time()
    for i in range(N):
        out[i] = _process_window(pipeline, X[i], context_length, QUANTILE_LEVELS)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(
                f"  {dataset}/{split}: {i + 1}/{N} windows  "
                f"({elapsed / (i + 1):.2f}s/win)"
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["telecomts", "spotlight"], required=True)
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH_DEFAULT,
        help="Chronos-2 context length. Defaults to 20 to match the paper.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    args = parser.parse_args()

    root = public_dataset_root(args.dataset)
    cache = residual_cache(args.dataset)
    print(f"Using dataset cache: {root}")
    print(f"Writing residuals to: {cache}")

    pipeline = _load_pipeline()
    import numpy as np
    for split in args.splits:
        out_path = cache / f"residuals_{split}.npy"
        if out_path.exists():
            print(f"  {split}: already cached at {out_path}")
            continue
        resid = _process_split(pipeline, args.dataset, split, args.context_length)
        np.save(out_path, resid)
        print(f"  {split}: wrote {resid.shape} to {out_path}")


if __name__ == "__main__":
    main()
