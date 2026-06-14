"""Rebuild the SpotLight (MobiCom 2024) NPZ splits from the public release.

``scripts/download_spotlight.py`` downloads the upstream release and calls
:func:`build_splits` here to turn the raw per-run telemetry (``radio.csv`` +
``platform.csv``) into the windowed NPZ splits the KAC pipeline consumes:
``SpotLight_{train,val,test}.npz``, each with the same schema as the TelecomTS
splits (``X`` ``(N, T, K)`` float32, ``y`` ``(N,)`` int64, ``descriptions``,
``feature_cols``).

This implements the paper's stated protocol:

* **Windowing.** Non-overlapping windows of length ``T = 64`` per run.
* **Labeling.** A window is anomalous if *any* constituent timestep is
  anomalous.
* **Splitting.** By run (to avoid window-level leakage across splits),
  deterministic given ``seed``.
* **Descriptions.** Label-free, train-split-only saliency: KPIs are ranked by
  variance over the *training* windows, and each window is summarized with a
  short deterministic template over the top KPIs. (The paper's SpotLight gain
  comes mainly from KPI-level contrastive alignment, not the text content, so
  template summaries are sufficient for the public path.)

Schema robustness
-----------------
The exact column names in the upstream release can vary. This module
auto-detects the per-timestep label column (``label``/``anomaly``/``is_anomaly``/
``attack``/``y``) and treats the remaining numeric columns as KPI features
(``radio_*`` / ``platform_*`` prefixes avoid collisions). If detection fails it
raises an actionable error listing the columns it found, so you can adjust the
constants below to match your copy of the release. It therefore reconstructs the
splits per the paper's protocol; exact window membership depends on the release
contents and ``seed`` and is not guaranteed bit-identical to the authors' run.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

WINDOW_T = 64
DEFAULT_SPLIT = (0.70, 0.15, 0.15)  # train / val / test, by run
DEFAULT_SEED = 42
DESC_TOP_K = 3  # KPIs named in each window summary

# Candidate names (lower-cased) for the per-timestep anomaly label column.
_LABEL_NAMES = {"label", "anomaly", "is_anomaly", "anomalous", "attack", "y", "target"}
# Columns excluded from the KPI feature set even if numeric.
_NON_FEATURE = {"time", "timestamp", "ts", "index", "idx", "run", "ue", "seq", "step"}


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _detect_label(df) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in _LABEL_NAMES:
            return c
    return None


def _numeric_features(df, prefix: str, label_col: Optional[str]) -> "Tuple[list, object]":
    import pandas as pd  # noqa: F401

    cols = []
    for c in df.columns:
        name = str(c).strip().lower()
        if c == label_col or name in _NON_FEATURE:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols.append(c)
    feats = {f"{prefix}_{c}": df[c].to_numpy(dtype=np.float32) for c in cols}
    return list(feats.keys()), feats


def _find_runs(raw_dir: Path, variant: str) -> List[Path]:
    """Locate per-run directories that contain both radio.csv and platform.csv."""
    runs = sorted(
        {p.parent for p in raw_dir.rglob("radio.csv")}
        & {p.parent for p in raw_dir.rglob("platform.csv")}
    )
    if variant == "paper5ue_single":
        # The paper uses the single-UE subset. Prefer runs whose path encodes it;
        # fall back to all runs (with a note) if the naming differs in your copy.
        filtered = [r for r in runs if "single" in r.as_posix().lower()
                    or "5ue" in r.as_posix().lower()]
        if filtered:
            runs = filtered
        else:
            print("[spotlight_preprocess] variant 'paper5ue_single' requested but no "
                  "run path matched 'single'/'5ue'; using all discovered runs.")
    return runs


def _windows_for_run(run_dir: Path) -> "Optional[Tuple[np.ndarray, np.ndarray, list]]":
    radio = _read_csv(run_dir / "radio.csv")
    platform = _read_csv(run_dir / "platform.csv")

    label_col = _detect_label(radio) or _detect_label(platform)
    if label_col is None:
        raise ValueError(
            f"No anomaly-label column found in {run_dir}. Looked for {sorted(_LABEL_NAMES)}; "
            f"radio.csv has {list(radio.columns)} and platform.csv has {list(platform.columns)}. "
            "Adjust _LABEL_NAMES in spotlight_preprocess.py to match the release."
        )

    n = min(len(radio), len(platform))
    if n < WINDOW_T:
        return None
    radio, platform = radio.iloc[:n], platform.iloc[:n]

    r_names, r_feats = _numeric_features(radio, "radio", label_col)
    p_names, p_feats = _numeric_features(platform, "platform", label_col)
    names = r_names + p_names
    X = np.column_stack([*r_feats.values(), *p_feats.values()]).astype(np.float32)

    lab_src = radio if label_col in radio.columns else platform
    y_t = (lab_src[label_col].to_numpy() > 0).astype(np.int64)

    n_win = n // WINDOW_T
    Xw = X[: n_win * WINDOW_T].reshape(n_win, WINDOW_T, X.shape[1])
    yw = y_t[: n_win * WINDOW_T].reshape(n_win, WINDOW_T).max(axis=1).astype(np.int64)
    return Xw, yw, names


def _describe(window: np.ndarray, feature_cols: List[str], top_idx: np.ndarray) -> str:
    """Deterministic, label-free template summary over the top salient KPIs."""
    parts = []
    for k in top_idx[:DESC_TOP_K]:
        series = window[:, k]
        level = float(series.mean())
        trend = "rising" if series[-1] > series[0] else "falling"
        parts.append(f"{feature_cols[k]} level {level:.2f} {trend}")
    return "; ".join(parts)


def build_splits(
    raw_dir: Path,
    out_dir: Path,
    variant: str = "paper5ue_single",
    seed: int = DEFAULT_SEED,
) -> None:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _find_runs(raw_dir, variant)
    if not runs:
        raise FileNotFoundError(
            f"No SpotLight runs (radio.csv + platform.csv) found under {raw_dir}. "
            "Check the extracted release layout."
        )

    per_run = []  # (run_dir, Xw, yw, names)
    common: Optional[set] = None
    for r in runs:
        res = _windows_for_run(r)
        if res is None:
            continue
        Xw, yw, names = res
        per_run.append((r, Xw, yw, names))
        common = set(names) if common is None else (common & set(names))
    if not per_run or not common:
        raise ValueError("No usable runs / no shared KPI columns across runs.")

    feature_cols = sorted(common)
    col_index = {n: i for i, n in enumerate(feature_cols)}

    # Re-index every run's windows onto the shared, sorted KPI column set.
    def _reindex(Xw: np.ndarray, names: List[str]) -> np.ndarray:
        out = np.zeros((Xw.shape[0], WINDOW_T, len(feature_cols)), dtype=np.float32)
        for src_i, name in enumerate(names):
            if name in col_index:
                out[:, :, col_index[name]] = Xw[:, :, src_i]
        return out

    # Deterministic split by run.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(per_run))
    n_tr = int(DEFAULT_SPLIT[0] * len(per_run))
    n_va = int(DEFAULT_SPLIT[1] * len(per_run))
    assign = {}
    for rank, ridx in enumerate(order):
        assign[ridx] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")

    buckets = {"train": [], "val": [], "test": []}
    for ridx, (_, Xw, yw, names) in enumerate(per_run):
        buckets[assign[ridx]].append((_reindex(Xw, names), yw))

    # Train-split-only KPI saliency (variance) for the description template.
    train_X = np.concatenate([x for x, _ in buckets["train"]], axis=0) if buckets["train"] else None
    if train_X is None or len(train_X) == 0:
        raise ValueError("Empty training split; not enough runs to split.")
    saliency = train_X.reshape(-1, len(feature_cols)).var(axis=0)
    top_idx = np.argsort(saliency)[::-1]

    for split, items in buckets.items():
        if not items:
            raise ValueError(f"Empty {split} split; need more runs.")
        X = np.concatenate([x for x, _ in items], axis=0).astype(np.float32)
        y = np.concatenate([yy for _, yy in items], axis=0).astype(np.int64)
        descriptions = np.asarray(
            [_describe(X[i], feature_cols, top_idx) for i in range(len(X))], dtype=object,
        )
        out = out_dir / f"SpotLight_{split}.npz"
        np.savez_compressed(
            out, X=X, y=y, descriptions=descriptions,
            feature_cols=np.asarray(feature_cols, dtype=object),
        )
        print(f"  {split}: X={X.shape} anomalies={int(y.sum())}/{len(y)} -> {out.name}")
