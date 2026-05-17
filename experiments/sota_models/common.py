"""Shared utilities for SOTA AD baselines: scoring, threshold selection, metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def reduce_window_score(point_scores: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Reduce per-(window, time) anomaly scores to per-window scores.

    Args:
        point_scores: (N, T) array of per-step anomaly scores.
        mode: ``mean`` (default) or ``max``.
    """
    if point_scores.ndim != 2:
        raise ValueError(f"expected (N,T), got {point_scores.shape}")
    if mode == "mean":
        return point_scores.mean(axis=1)
    if mode == "max":
        return point_scores.max(axis=1)
    raise ValueError(f"unknown reduction mode: {mode!r}")


def best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_grid: int = 201,
    min_predicted_positive: float = 0.02,
    max_predicted_positive: float = 0.98,
    verbose: bool = True,
):
    """Pick the threshold on (scores, labels) that maximizes window-level F1.

    The grid search excludes thresholds whose predicted-positive fraction falls
    outside ``[min_predicted_positive, max_predicted_positive]`` so that the
    "predict everything as anomaly" or "predict nothing as anomaly" degenerate
    optima are filtered out. If every grid threshold is degenerate (which
    happens when the score distributions for normal/anomalous windows fully
    overlap, i.e. the model failed to learn), the function falls back to the
    median score and emits a warning.

    Returns ``(best_threshold, best_f1)``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return 0.5, 0.0
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if lo == hi:
        return lo, float(f1_score(labels, np.zeros_like(labels), zero_division=0))

    grid = np.linspace(lo, hi, n_grid)
    n = float(scores.size)
    best_t, best_f1 = None, -1.0
    for t in grid:
        pred = (scores >= t).astype(np.int64)
        frac_pos = pred.mean()
        if frac_pos < min_predicted_positive or frac_pos > max_predicted_positive:
            continue
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    if best_t is None:
        # Score distribution is uninformative -- fall back to the median so we
        # do not silently report the degenerate "predict-all-positive" F1.
        med = float(np.median(scores))
        pred = (scores >= med).astype(np.int64)
        best_f1 = float(f1_score(labels, pred, zero_division=0))
        best_t = med
        if verbose:
            n_pos = int(labels.sum())
            print(
                f"  [best_f1_threshold] WARNING: no non-degenerate threshold found "
                f"(scores in [{lo:.4g}, {hi:.4g}], {n_pos}/{int(n)} positives). "
                f"Falling back to median={med:.4g} (F1={best_f1:.3f}). "
                f"This usually means the model failed to learn -- check data "
                f"normalization, training epochs, and AUROC."
            )
    return best_t, best_f1


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Window-level metrics from a continuous score and a fixed threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = (scores >= threshold).astype(np.int64)
    out = {
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "threshold": float(threshold),
    }
    # AUROC / AP only defined when both classes present
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["ap"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out


def summarize_seeds(rows: list[dict]) -> dict:
    """Mean +/- std summary of per-seed metric dicts."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    summary = {}
    for col in df.columns:
        if col == "seed":
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if vals.size == 0:
            summary[f"{col}_mean"] = float("nan")
            summary[f"{col}_std"] = float("nan")
        else:
            summary[f"{col}_mean"] = float(np.mean(vals))
            summary[f"{col}_std"] = float(np.std(vals))
    return summary


def select_normal_only(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the subset of windows whose label is 0 (normal)."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    mask = (y == 0)
    return X[mask]
