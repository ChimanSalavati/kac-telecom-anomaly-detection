#!/usr/bin/env python3
"""KAC CPU/GPU latency on **ProdTrace-SA tensor shapes** (no Nokia NPZ required).

Uses ``K=64``, ``T_r=24``, ``L=128`` (paper Table~\\ref{tab:dataset_stats}) and the
same ``build_model`` as training. Random residuals + placeholder summaries.

Run::

    cd kac-telecom-anomaly-detection
    python experiments/revision_pipeline/runner_labtrace_shape_bench.py

Set ``HF_HUB_OFFLINE=1`` if DistilBERT is already cached.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]  # experiments/
sys.path.insert(0, str(_ROOT / "_shared"))
import kac_ablation as KA  # noqa: E402

BATCH_SIZES = [1, 16, 64]
K, T_R = 64, 24
FEATS = 5


def measure(model, ids, mask, resid, *, warmup: int = 5, n: int = 30) -> tuple[float, float]:
    use_cuda = torch.cuda.is_available()
    model.eval()
    device = "cuda" if use_cuda else "cpu"
    model.to(device)
    ids = ids.to(device)
    mask = mask.to(device)
    resid = resid.to(device)
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(ids, mask, resid)[0]
            sync()
        ts = []
        for _ in range(n):
            sync()
            t0 = time.perf_counter()
            _ = model(ids, mask, resid)[0]
            sync()
            ts.append(time.perf_counter() - t0)
    arr = np.array(ts) * 1000.0
    return float(np.median(arr)), float(np.percentile(arr, 95))


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(KA.TEXT_MODEL_NAME, local_files_only=True)
    enc = tok(
        ["ProdTrace-SA shaped smoke window. "] * max(BATCH_SIZES),
        padding="max_length",
        truncation=True,
        max_length=KA.MAX_LEN,
        return_tensors="pt",
    )
    R = torch.randn(max(BATCH_SIZES), T_R, K * FEATS, dtype=torch.float32)
    model = KA.build_model(K, FEATS)
    print(f"ProdTrace-SA shape bench: K={K}, T_r={T_R}, resid_dim={K*FEATS}")
    for B in BATCH_SIZES:
        ids = enc["input_ids"][:B]
        mask = enc["attention_mask"][:B]
        resid = R[:B]
        med, p95 = measure(model, ids, mask, resid)
        tput = B / (med / 1000.0)
        print(f"  batch={B:2d}  median_ms={med:7.2f}  p95_ms={p95:7.2f}  thruput_ws={tput:6.1f}")


if __name__ == "__main__":
    main()
