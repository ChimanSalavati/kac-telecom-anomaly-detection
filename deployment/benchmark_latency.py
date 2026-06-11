"""Reproduce the KAC operational-cost table (paper Table "deployment_cost").

Measures, on CPU, the KAC scoring path on cached residual/summary inputs
(DistilBERT text encoding + lightweight fusion/classification heads; excludes
offline Chronos-2 extraction and summary generation, exactly as stated in the
paper). Emits:

* the parameter breakdown (frozen DistilBERT / last-2 blocks / KAC fusion+heads
  / learnable queries / total / trainable), and
* median, p95, and throughput for batch sizes 1 / 16 / 64.

Writes ``artifacts/deployment/latency.csv`` and ``params.csv`` and prints a
table. Use ``--smoke`` for a fast offline check (tiny random-init backbone).

Note: absolute latencies depend on the host CPU; the paper's numbers were
measured on an Apple M1 (CPU-only). This script regenerates the *methodology*
and the parameter breakdown deterministically; rerun it on your hardware to get
machine-specific latencies.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .kac_service import build_inference_model, param_breakdown, score_batch


def _make_inputs(batch: int, K: int, T_r: int, L: int):
    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, 1000, size=(batch, L)).tolist()
    attention_mask = np.ones((batch, L), dtype=int).tolist()
    residuals = rng.normal(0, 1, size=(batch, T_r, K * 5)).astype(np.float32).tolist()
    return input_ids, attention_mask, residuals


def benchmark(
    n_kpis: int = 64,
    t_r: int = 24,
    max_len: int = 128,
    batches: Optional[List[int]] = None,
    iters: int = 50,
    warmup: int = 5,
    smoke: bool = False,
    out_dir: str = "artifacts/deployment",
) -> Dict[str, object]:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if batches is None:
        batches = [1, 16, 64]
    if smoke:
        n_kpis, t_r, max_len = 6, 12, 8
        batches = [1, 4]
        iters, warmup = 5, 2

    model = build_inference_model(n_kpis=n_kpis, feats_per_kpi=5, smoke=smoke)
    params = param_breakdown(model)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for b in batches:
        ids, mask, resid = _make_inputs(b, n_kpis, t_r, max_len)
        for _ in range(warmup):
            score_batch(model, ids, mask, resid)
        times_ms: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            score_batch(model, ids, mask, resid)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        times_ms.sort()
        med = statistics.median(times_ms)
        p95 = times_ms[min(len(times_ms) - 1, int(0.95 * len(times_ms)))]
        tput = (b / med) * 1000.0 if med > 0 else float("nan")
        rows.append({
            "batch": b,
            "median_ms": round(med, 2),
            "p95_ms": round(p95, 2),
            "throughput_w_per_s": round(tput, 1),
        })

    with open(out / "latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "median_ms", "p95_ms", "throughput_w_per_s"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "params.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component", "params"])
        for k in ("distilbert_frozen", "distilbert_last2", "kac_fusion_heads",
                  "learnable_queries", "total", "trainable"):
            w.writerow([k, params[k]])

    print("=== KAC scorer parameter breakdown ===")
    for k in ("distilbert_frozen", "distilbert_last2", "kac_fusion_heads",
              "learnable_queries", "total", "trainable"):
        print(f"  {k:<20} {params[k]:>12,}")
    print(f"  KAC-specific share of params: {params['kac_specific_pct']}%")
    print("\n=== CPU latency (cached residual/summary inputs) ===")
    print(f"  {'batch':>6} {'median(ms)':>12} {'p95(ms)':>10} {'tput(w/s)':>12}")
    for r in rows:
        print(f"  {r['batch']:>6} {r['median_ms']:>12} {r['p95_ms']:>10} {r['throughput_w_per_s']:>12}")
    print(f"\nWrote {out/'latency.csv'} and {out/'params.csv'}")
    return {"params": params, "latency": rows, "out_dir": str(out)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="KAC CPU latency / parameter benchmark.")
    ap.add_argument("--n-kpis", type=int, default=64)
    ap.add_argument("--t-r", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default="artifacts/deployment")
    args = ap.parse_args(argv)
    benchmark(
        n_kpis=args.n_kpis, t_r=args.t_r, max_len=args.max_len, batches=args.batches,
        iters=args.iters, warmup=args.warmup, smoke=args.smoke, out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
