"""Pre-production shadow-mode harness for KAC.

Mirrors the paper's "Shadow Integration Status": the pipeline passes cached KPI
windows to KAC, cached residuals/summaries are loaded, and the resulting anomaly
scores are **logged for offline comparison** against an internal detector. KAC
does **not** promote incidents, trigger alerts, or initiate mitigation -- this
harness only scores and appends to a shadow log.

It is intentionally infrastructure-agnostic: in Nokia's setup the windows arrive
via NATS and cached features come from MinIO behind a Kong route; here the same
loop reads cached tensors from disk (or generates tiny synthetic windows with
``--smoke``) and writes a JSONL shadow log under ``logs/``.

Examples
--------
Offline plumbing run (no data/network)::

    python -m deployment.shadow_runner --smoke

Score real cached windows and compare against a reference detector's scores::

    python -m deployment.shadow_runner \
        --windows path/to/windows.npz --reference path/to/internal_scores.npy
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .kac_service import build_inference_model, git_sha, score_batch


def _synthetic_windows(n: int = 8, K: int = 6, T_r: int = 12, L: int = 8):
    rng = np.random.default_rng(0)
    y = np.array([i % 2 for i in range(n)], dtype=int)
    residuals = rng.normal(0, 1, size=(n, T_r, K * 5)).astype(np.float32)
    residuals[y == 1] += 1.2
    input_ids = rng.integers(0, 1000, size=(n, L)).tolist()
    attention_mask = np.ones((n, L), dtype=int).tolist()
    return input_ids, attention_mask, residuals.tolist(), y, K


def _load_windows(path: Path):
    npz = np.load(path, allow_pickle=True)
    residuals = np.asarray(npz["residuals"], dtype=np.float32)
    input_ids = np.asarray(npz["input_ids"], dtype=int).tolist()
    attention_mask = np.asarray(npz["attention_mask"], dtype=int).tolist()
    y = np.asarray(npz["y"], dtype=int) if "y" in npz.files else None
    K = int(residuals.shape[-1] // 5)
    return input_ids, attention_mask, residuals.tolist(), y, K


def run_shadow(
    windows: Optional[str] = None,
    reference: Optional[str] = None,
    out_log: str = "logs/shadow_scores.jsonl",
    smoke: bool = False,
    batch_size: int = 16,
) -> Dict[str, object]:
    if smoke or windows is None:
        input_ids, attention_mask, residuals, y, K = _synthetic_windows()
        smoke = True
    else:
        input_ids, attention_mask, residuals, y, K = _load_windows(Path(windows))

    feats_per_kpi = len(residuals[0][0]) // K
    model = build_inference_model(n_kpis=K, feats_per_kpi=feats_per_kpi, smoke=smoke)
    sha = git_sha()

    ref = np.load(reference) if reference else None

    out_path = Path(out_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(residuals)
    all_scores: List[float] = []
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            res = score_batch(
                model,
                input_ids[start:end],
                attention_mask[start:end],
                residuals[start:end],
            )
            for j, prob in enumerate(res["probs"]):
                idx = start + j
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "window_id": idx,
                    "kac_score": float(prob),
                    "model_sha": sha,
                    "promoted": False,  # shadow mode never promotes incidents
                }
                if y is not None:
                    rec["label"] = int(y[idx])
                if ref is not None and idx < len(ref):
                    rec["reference_score"] = float(ref[idx])
                f.write(json.dumps(rec) + "\n")
                all_scores.append(float(prob))

    summary = {
        "windows_scored": n,
        "shadow_log": str(out_path),
        "model_sha": sha,
        "mean_score": float(np.mean(all_scores)) if all_scores else float("nan"),
        "promoted_incidents": 0,
    }
    if y is not None:
        try:
            from sklearn.metrics import roc_auc_score

            summary["offline_auroc_vs_labels"] = float(roc_auc_score(y, all_scores))
        except Exception:
            pass
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="KAC pre-production shadow-mode scorer.")
    ap.add_argument("--windows", default=None, help="NPZ with input_ids/attention_mask/residuals[/y].")
    ap.add_argument("--reference", default=None, help="Optional .npy of an internal detector's scores.")
    ap.add_argument("--out-log", default="logs/shadow_scores.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="Tiny synthetic windows, offline.")
    args = ap.parse_args(argv)
    run_shadow(
        windows=args.windows, reference=args.reference, out_log=args.out_log,
        smoke=args.smoke, batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
