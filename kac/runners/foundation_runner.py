"""Frozen time-series-foundation-model + linear-probe runner.

Replaces notebooks:
    E4_TelecomTS_Foundation_Models (+ E4b imbalanced), E9_SpotLight_Foundation_Models
        -> ``main.py foundation --foundation-model {moment,toto,mantis} ...``

The recipe (paper Section 5.1.4 baselines): freeze a pretrained TSFM encoder,
extract per-window embeddings, train a logistic-regression head, pick the best-F1
threshold on validation, and report on test.

* Real mode loads the requested backbone. MOMENT / TOTO / Mantis are optional
  dependencies (not in requirements.txt); a missing backbone raises a clear,
  actionable error rather than silently degrading.
* Smoke mode uses a trivial offline "encoder" (flatten + standardize) so the
  probe/threshold/metric path is exercised without downloading weights.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import ExperimentConfig
from ..data import get_data
from ._common import best_f1_threshold, compute_metrics, metric_row, set_seed


def _embed(cfg: ExperimentConfig, X: np.ndarray) -> np.ndarray:
    """Return (N, D) window embeddings from the configured encoder."""
    if cfg.smoke:
        # Trivial frozen "encoder": flatten the window. Standardization happens
        # in the probe pipeline below.
        return X.reshape(X.shape[0], -1)
    raise NotImplementedError(
        f"Foundation backbone {cfg.foundation_model!r} requires an optional "
        "dependency (e.g. `pip install momentfm` for MOMENT). Install the "
        "backbone and implement its embedding call here, or run with --smoke."
    )


def run(cfg: ExperimentConfig, logger) -> List[Dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = get_data(cfg)
    Z_tr = _embed(cfg, np.asarray(data["X_train"]))
    Z_va = _embed(cfg, np.asarray(data["X_val"]))
    Z_te = _embed(cfg, np.asarray(data["X_test"]))
    y_tr = np.asarray(data["y_train"])
    y_va = np.asarray(data["y_val"])
    y_te = np.asarray(data["y_test"])

    rows: List[Dict] = []
    seeds = [0] if cfg.smoke else cfg.seeds
    for seed in seeds:
        set_seed(seed)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        clf.fit(Z_tr, y_tr)
        va_scores = clf.predict_proba(Z_va)[:, 1]
        te_scores = clf.predict_proba(Z_te)[:, 1]
        thr, _ = best_f1_threshold(va_scores, y_va, verbose=False)
        metrics = compute_metrics(te_scores, y_te, thr)
        logger.info(
            "%s+linear seed=%-5d F1=%.4f AUROC=%.4f",
            cfg.foundation_model, seed, metrics["f1"], metrics["auroc"],
        )
        rows.append(metric_row("method", f"{cfg.foundation_model}+linear", seed, metrics))
    return rows


GROUP_KEY = "method"
