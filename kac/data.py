"""Data access for the centralized runners.

Two code paths:

* :func:`load_dataset` -- load the real public NPZ splits (TelecomTS / SpotLight)
  and, when present, the precomputed Chronos-2 residual cache. File names come
  from :meth:`ExperimentConfig.split_filenames`; the directory comes from
  ``cfg.data_root`` or the canonical ``experiments/_shared/cache/<dataset>``.
* :func:`make_synthetic` -- generate tiny, deterministic tensors with the right
  shapes so every runner can be exercised end-to-end in ``--smoke`` mode without
  any downloaded data, GPU, or network access.

Both return the same dict schema so runners are agnostic to the source.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import ExperimentConfig

# Make ``experiments/`` importable (``import experiments._shared. ...``) no matter
# what the current working directory is.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Real public splits
# ---------------------------------------------------------------------------

def _cache_dir(cfg: ExperimentConfig) -> Path:
    if cfg.data_root:
        return Path(cfg.data_root)
    return REPO_ROOT / "experiments" / "_shared" / "cache" / cfg.dataset


def load_dataset(cfg: ExperimentConfig) -> Dict[str, object]:
    """Load the real splits for ``cfg.dataset`` / ``cfg.scenario``.

    Returns a dict with ``X_{train,val,test}`` (N, T, K) float32 arrays,
    ``y_{train,val,test}`` int64 label vectors, ``texts_{train,val,test}``
    lists of summary strings (empty strings if absent), ``kpi_names``, and --
    when ``features_cache/residuals_*.npy`` exists -- ``R_{train,val,test}``.
    """
    cache = _cache_dir(cfg)
    names = cfg.split_filenames()
    out: Dict[str, object] = {}
    kpi_names = None
    for split in ("train", "val", "test"):
        path = cache / names[split]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split {path}. Run scripts/download_{cfg.dataset}.py first "
                f"(or pass --data-root). ProdTrace-SA is not redistributed."
            )
        npz = np.load(path, allow_pickle=True)
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64)
        texts = (
            [str(t) for t in npz["descriptions"]]
            if "descriptions" in npz.files
            else ["" for _ in range(len(y))]
        )
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        out[f"texts_{split}"] = texts
        if kpi_names is None:
            kpi_names = (
                [str(c) for c in npz["feature_cols"]]
                if "feature_cols" in npz.files
                else [f"KPI_{i}" for i in range(X.shape[-1])]
            )
    out["kpi_names"] = kpi_names

    res_dir = cache / "features_cache"
    if (res_dir / "residuals_train.npy").exists():
        for split in ("train", "val", "test"):
            out[f"R_{split}"] = np.load(res_dir / f"residuals_{split}.npy").astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Tiny synthetic data for smoke runs
# ---------------------------------------------------------------------------

def make_synthetic(
    cfg: ExperimentConfig,
    *,
    n_train: int = 24,
    n_eval: int = 12,
    K: int = 6,
    T: int = 16,
    feats_per_kpi: int = 5,
) -> Dict[str, object]:
    """Deterministic tiny dataset with the real schema, for ``--smoke`` runs.

    Anomalous windows get an additive mean shift so that even a 1-epoch model
    can separate the two classes well enough for the artifact-writing code to
    exercise the full metric path.
    """
    rng = np.random.default_rng(0)
    C = min(cfg.chronos_context, T - 4)
    T_r = T - C

    def _make(n: int):
        y = np.array([i % 2 for i in range(n)], dtype=np.int64)
        X = rng.normal(0.0, 1.0, size=(n, T, K)).astype(np.float32)
        X[y == 1] += 1.5  # mean shift makes anomalies learnable
        R = rng.normal(0.0, 1.0, size=(n, T_r, feats_per_kpi * K)).astype(np.float32)
        R[y == 1] += 1.2
        # raw interval-width feature (index 3 of every 5) used for uncertainty weights
        W = np.abs(rng.normal(1.0, 0.2, size=(n, T_r, K))).astype(np.float32)
        texts = [
            ("kpi shows elevated level and rising trend" if lab else "kpi nominal stable")
            for lab in y
        ]
        return X, y, R, W, texts

    out: Dict[str, object] = {"kpi_names": [f"KPI_{i}" for i in range(K)]}
    for split, n in (("train", n_train), ("val", n_eval), ("test", n_eval)):
        X, y, R, W, texts = _make(n)
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        out[f"R_{split}"] = R
        out[f"W_{split}"] = W
        out[f"texts_{split}"] = texts
    return out


def get_data(cfg: ExperimentConfig) -> Dict[str, object]:
    """Dispatch to synthetic data in smoke mode, else load the real splits."""
    return make_synthetic(cfg) if cfg.smoke else load_dataset(cfg)
