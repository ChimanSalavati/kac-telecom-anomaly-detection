"""Centralized run directories, logging, and artifact writers.

Every experiment writes to the same place so results are easy to find, diff,
and publish:

    artifacts/<run_id>/
        config.json          # the exact resolved ExperimentConfig
        metrics.csv          # one row per (method/variant, seed)
        summary.csv          # mean +/- std across seeds
        *.pt / *.pdf / ...   # optional checkpoints / figures
    logs/<run_id>.log        # full stdout/stderr transcript of the run

``run_id`` is produced by :meth:`kac.config.ExperimentConfig.run_id`.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from .config import ExperimentConfig


def run_dir(cfg: ExperimentConfig) -> Path:
    """Create and return ``artifacts/<run_id>/`` for this config."""
    d = Path(cfg.output_root) / cfg.run_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(cfg: ExperimentConfig) -> logging.Logger:
    """Configure a logger that tees to console and ``logs/<run_id>.log``."""
    Path(cfg.log_root).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_root) / f"{cfg.run_id()}.log"

    logger = logging.getLogger(f"kac.{cfg.run_id()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("run_id=%s  log=%s", cfg.run_id(), log_path)
    return logger


def save_config(cfg: ExperimentConfig) -> Path:
    """Persist the resolved config next to the artifacts for provenance."""
    p = run_dir(cfg) / "config.json"
    p.write_text(cfg.to_json(), encoding="utf-8")
    return p


def write_metrics(cfg: ExperimentConfig, rows: Sequence[Dict], name: str = "metrics.csv") -> Path:
    """Write per-(method/variant, seed) metric rows to ``artifacts/<run_id>/``."""
    p = run_dir(cfg) / name
    rows = list(rows)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def summarize(rows: Sequence[Dict], group_key: str) -> List[Dict]:
    """Mean/std over seeds, grouped by ``group_key`` (e.g. method or variant)."""
    import numpy as np

    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(group_key, "all")), []).append(r)

    metric_cols = ("precision", "recall", "f1", "auroc", "ap")
    out: List[Dict] = []
    for g, items in groups.items():
        srow: Dict[str, object] = {group_key: g, "n_seeds": len(items)}
        for col in metric_cols:
            vals = [float(it[col]) for it in items if col in it and it[col] == it[col]]
            if vals:
                srow[f"{col}_mean"] = float(np.mean(vals))
                srow[f"{col}_std"] = float(np.std(vals))
        out.append(srow)
    return out


def write_summary(cfg: ExperimentConfig, rows: Sequence[Dict], group_key: str) -> Path:
    """Compute and write the across-seed summary table."""
    return write_metrics(cfg, summarize(rows, group_key), name="summary.csv")


def save_json(cfg: ExperimentConfig, obj: object, name: str) -> Path:
    p = run_dir(cfg) / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
