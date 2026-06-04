"""Qualitative analysis: load saved KAC checkpoint, run forward passes,
and extract per-KPI attention weights and uncertainty weights for case
studies (Workstream J).

Run from the dataset directory::

    cd evaluation_ver2/TelecomTS
    python ../../ICDM_2026_Applied_Track/paper/revision_pipeline/runner_qualitative.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation_ver2"))
import kac_ablation as KA

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    cfg = KA.DATASET_CONFIGS["TelecomTS"]
    data = KA.load_dataset(cfg)
    tok = AutoTokenizer.from_pretrained(KA.TEXT_MODEL_NAME)
    enc = tok(data["texts_test"], padding="max_length", truncation=True,
              max_length=KA.MAX_LEN, return_tensors="pt")
    R = torch.tensor(data["R_test"], dtype=torch.float32)
    W = torch.tensor(data["W_test"], dtype=torch.float32)
    y = data["y_test"]

    model = KA.build_model(data["n_kpis"], data["feats_per_kpi"])
    ckpt_path = Path("results") / "kac_uncertainty_best.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # Forward all test windows in batches
    all_probs = []
    all_attn = []
    with torch.no_grad():
        for s in range(0, len(y), 16):
            ids = enc["input_ids"][s:s + 16]
            mask = enc["attention_mask"][s:s + 16]
            res = R[s:s + 16]
            logits, _, _, _, attn = model(ids, mask, res, use_text=True)
            all_probs.append(torch.sigmoid(logits).numpy())
            all_attn.append(attn.numpy())
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    attn = np.concatenate(all_attn, axis=0)  # [N, K, L]

    # Pick examples
    tp_idx = next(i for i in range(len(y)) if y[i] == 1 and preds[i] == 1
                  and probs[i] > 0.95)
    tn_idx = next(i for i in range(len(y)) if y[i] == 0 and preds[i] == 0
                  and probs[i] < 0.05)
    fp_candidates = [i for i in range(len(y)) if y[i] == 0 and preds[i] == 1]
    fn_candidates = [i for i in range(len(y)) if y[i] == 1 and preds[i] == 0]

    cases = [("True positive", tp_idx),
             ("True negative", tn_idx)]
    if fp_candidates:
        cases.append(("False positive", fp_candidates[0]))
    if fn_candidates:
        cases.append(("False negative", fn_candidates[0]))

    fig, axs = plt.subplots(len(cases), 2, figsize=(11, 2.5 * len(cases)))
    if len(cases) == 1:
        axs = axs.reshape(1, -1)
    for row, (label, idx) in enumerate(cases):
        # KPI-token attention heatmap
        a = attn[idx]  # [K, L]
        # Truncate to non-padding
        attn_mask = enc["attention_mask"][idx].numpy().astype(bool)
        L = int(attn_mask.sum())
        a = a[:, :L]
        im = axs[row, 0].imshow(a, aspect="auto", cmap="viridis")
        axs[row, 0].set_xlabel("Token position")
        axs[row, 0].set_ylabel("KPI index")
        axs[row, 0].set_title(f"{label}: prob={probs[idx]:.3f} (true={int(y[idx])})")
        # Per-KPI uncertainty weight (inverse mean Chronos width)
        w = W[idx].numpy()  # [T_r, K]
        u = 1.0 / (w.mean(axis=0) + 1e-6)
        u = u / u.mean()
        axs[row, 1].bar(range(len(u)), u)
        axs[row, 1].set_xlabel("KPI index")
        axs[row, 1].set_ylabel("Uncertainty weight $\\omega_k$")
        axs[row, 1].set_title("Per-KPI uncertainty weights")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures_new" / "case_studies.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
