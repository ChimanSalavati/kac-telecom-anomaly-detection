"""Runner for the five missing baselines (Workstream C of the revision plan).

This script reuses the dataset loaders, residual cache, text masking, and
training loop from ``evaluation_ver2/kac_ablation.py`` and adds five new
component-toggle variants:

  Bx-text-only      : text branch only (KPI residual zeroed out).
  Bx-residual-cnn   : residual branch only with a 1D CNN head and BCE.
  Bx-naive-fusion   : concat(text mean, residual mean) -> MLP -> BCE.
  Bx-mindts-style   : MindTS-style fine-grained text-residual alignment.
  Bx-tsclip-style   : CLIP-style window-level text/residual contrastive.

Each variant is trained for 5 seeds (42, 123, 456, 789, 1337) on each
dataset with the same splits as the headline KAC runs.

Output CSV: ``evaluation_ver2/<dataset>/results/missing_baselines.csv``.

Run from the dataset directory (so relative paths to ``data/features_cache``
match the headline notebooks)::

    cd evaluation_ver2/TelecomTS
    python ../../ICDM_2026_Applied_Track/paper/revision_pipeline/runner_missing_baselines.py \
        --dataset TelecomTS --variants text-only residual-cnn naive-fusion \
                                       mindts-style tsclip-style \
        --seeds 42 123 456 789 1337
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

# Re-use the kac_ablation infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation_ver2"))
import kac_ablation as KA


@dataclass
class BaselineSpec:
    code: str
    description: str
    use_text: bool
    use_residual: bool
    fusion: str   # "text-only", "residual-cnn", "naive", "mindts", "tsclip"


BASELINES: Dict[str, BaselineSpec] = {
    "text-only":     BaselineSpec("B1", "DistilBERT(text) -> MLP",
                                   True, False, "text-only"),
    "residual-cnn":  BaselineSpec("B2", "Residual -> Conv1D -> MLP",
                                   False, True, "residual-cnn"),
    "naive-fusion":  BaselineSpec("B3", "concat(text_mean, residual_mean) -> MLP",
                                   True, True, "naive"),
    "mindts-style":  BaselineSpec("B4", "Token-level text-residual alignment + recon",
                                   True, True, "mindts"),
    "tsclip-style":  BaselineSpec("B5", "CLIP-style window contrastive(text<->residual)",
                                   True, True, "tsclip"),
}


# ----- Models ---------------------------------------------------------------

class TextOnlyHead(nn.Module):
    def __init__(self, text_encoder, hidden: int = KA.HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.proj = nn.Linear(KA.TEXT_DIM, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, ids, mask, x_resid=None):
        h = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.proj(pooled)).squeeze(-1)


class ResidualCNN(nn.Module):
    def __init__(self, resid_dim: int, hidden: int = KA.HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(resid_dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, ids, mask, x_resid):
        x = self.proj(x_resid)
        x = self.norm(x + self.conv(x.transpose(1, 2)).transpose(1, 2))
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class NaiveFusion(nn.Module):
    def __init__(self, text_encoder, resid_dim: int,
                 hidden: int = KA.HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_proj = nn.Linear(KA.TEXT_DIM, hidden)
        self.resid_proj = nn.Linear(resid_dim, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, ids, mask, x_resid):
        h = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        text_pool = (h * m).sum(1) / m.sum(1).clamp(min=1)
        text_pool = self.text_proj(text_pool)
        r = self.resid_proj(x_resid).mean(dim=1)
        return self.head(torch.cat([text_pool, r], dim=-1)).squeeze(-1)


class MindTSStyle(nn.Module):
    """Token-level text/residual cross-attention + reconstruction loss.

    A small re-implementation in the spirit of MindTS [Hu et al. 2026]:
    the text tokens attend over residual timesteps; the residual is
    reconstructed from the text-conditioned representation; the BCE loss
    is paired with a small MSE reconstruction term.
    """

    def __init__(self, text_encoder, resid_dim: int,
                 hidden: int = KA.HIDDEN, dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_proj = nn.Linear(KA.TEXT_DIM, hidden)
        self.resid_proj = nn.Linear(resid_dim, hidden)
        self.cross = nn.MultiheadAttention(hidden, num_heads, dropout=dropout,
                                            batch_first=True)
        self.recon = nn.Linear(hidden, resid_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, ids, mask, x_resid):
        h = self.text_proj(self.text_encoder(
            input_ids=ids, attention_mask=mask).last_hidden_state)
        r = self.resid_proj(x_resid)
        attn_out, _ = self.cross(query=r, key=h, value=h,
                                  key_padding_mask=(mask == 0))
        recon = self.recon(attn_out)
        pooled = attn_out.mean(dim=1)
        return self.head(pooled).squeeze(-1), recon


class TSCLIPStyle(nn.Module):
    """Window-level CLIP-style alignment between text and residual.

    A linear classifier head is added on top of the residual embedding;
    the CLIP loss is auxiliary.
    """

    def __init__(self, text_encoder, resid_dim: int,
                 hidden: int = KA.HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_proj = nn.Linear(KA.TEXT_DIM, hidden)
        self.resid_proj = nn.Linear(resid_dim, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, ids, mask, x_resid):
        h = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        text_pool = (h * m).sum(1) / m.sum(1).clamp(min=1)
        text_pool = self.text_proj(text_pool)
        r_pool = self.resid_proj(x_resid).mean(dim=1)
        z_t = F.normalize(text_pool, dim=-1)
        z_r = F.normalize(r_pool, dim=-1)
        logit = self.head(r_pool).squeeze(-1)
        return logit, z_t, z_r


def build(spec: BaselineSpec, n_kpis: int, feats_per_kpi: int):
    text_encoder = KA.build_model(n_kpis, feats_per_kpi).text_encoder
    resid_dim = n_kpis * feats_per_kpi
    if spec.fusion == "text-only":
        return TextOnlyHead(text_encoder)
    if spec.fusion == "residual-cnn":
        return ResidualCNN(resid_dim)
    if spec.fusion == "naive":
        return NaiveFusion(text_encoder, resid_dim)
    if spec.fusion == "mindts":
        return MindTSStyle(text_encoder, resid_dim)
    if spec.fusion == "tsclip":
        return TSCLIPStyle(text_encoder, resid_dim)
    raise ValueError(spec.fusion)


def train_one(model, train_loader, val_loader, test_loader, y_train,
              spec: BaselineSpec, *, lr_head: float, seed: int = 42,
              epochs: int = KA.EPOCHS, patience: int = KA.PATIENCE):
    KA.set_seed(seed)
    model = model.to(KA.DEVICE)
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    pw = torch.tensor([neg / pos], dtype=torch.float32, device=KA.DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    enc_params = ([p for p in getattr(model, "text_encoder", nn.Module()).parameters()
                   if p.requires_grad]
                  if spec.use_text else [])
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("text_encoder")]
    groups = [{"params": other_params, "lr": lr_head, "weight_decay": 1e-2}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": 2e-5,
                          "weight_decay": 1e-2})
    opt = torch.optim.AdamW(groups)

    best_f1, best_state, pat = -1.0, None, 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for b in train_loader:
            ids = b["input_ids"].to(KA.DEVICE)
            msk = b["attention_mask"].to(KA.DEVICE)
            res = b["resid"].to(KA.DEVICE)
            y = b["label"].to(KA.DEVICE)
            opt.zero_grad()
            if spec.fusion == "mindts":
                logits, recon = model(ids, msk, res)
                loss = bce(logits, y) + 0.1 * F.mse_loss(recon, res)
            elif spec.fusion == "tsclip":
                logits, zt, zr = model(ids, msk, res)
                logits_clip = zt @ zr.T / KA.TEMPERATURE
                labels = torch.arange(zt.size(0), device=zt.device)
                clip_loss = (
                    F.cross_entropy(logits_clip, labels)
                    + F.cross_entropy(logits_clip.T, labels)
                ) / 2
                loss = bce(logits, y) + 0.5 * clip_loss
            else:
                logits = model(ids, msk, res)
                loss = bce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # Validation
        with torch.no_grad():
            metrics = _eval(model, val_loader, spec)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
    elapsed = time.perf_counter() - t0
    if best_state:
        model.load_state_dict(best_state)
    test_metrics = _eval(model, test_loader, spec)
    test_metrics.update({"variant_code": spec.code,
                         "variant_name": spec.description,
                         "seed": seed, "wall_seconds": round(elapsed, 1)})
    return test_metrics


def _eval(model, loader, spec: BaselineSpec, threshold: float = 0.5):
    model.eval()
    all_logits, all_labels = [], []
    for b in loader:
        ids = b["input_ids"].to(KA.DEVICE)
        msk = b["attention_mask"].to(KA.DEVICE)
        res = b["resid"].to(KA.DEVICE)
        y = b["label"].to(KA.DEVICE)
        out = model(ids, msk, res)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        all_logits.append(logits.cpu()); all_labels.append(y.cpu())
    al = torch.cat(all_logits).numpy()
    yl = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-al))
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yl, preds, average="binary",
                                                   zero_division=0)
    try:
        auroc = roc_auc_score(yl, probs)
    except Exception:
        auroc = float("nan")
    try:
        ap = average_precision_score(yl, probs)
    except Exception:
        ap = float("nan")
    return {"precision": p, "recall": r, "f1": f1,
            "acc": accuracy_score(yl, preds),
            "auroc": auroc, "ap": ap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(KA.DATASET_CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--variants", nargs="+", default=list(BASELINES.keys()),
                    choices=list(BASELINES.keys()))
    args = ap.parse_args()

    cfg = KA.DATASET_CONFIGS[args.dataset]
    data = KA.load_dataset(cfg)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(KA.TEXT_MODEL_NAME)

    def make_loader(texts, R, y, W, shuffle):
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=KA.MAX_LEN, return_tensors="pt")
        ds = KA.ContrastiveDataset(enc["input_ids"], enc["attention_mask"], R, y, W)
        return DataLoader(ds, batch_size=KA.BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(data["texts_train"], data["R_train"],
                                data["y_train"], data["W_train"], True)
    val_loader = make_loader(data["texts_val"], data["R_val"],
                              data["y_val"], data["W_val"], False)
    test_loader = make_loader(data["texts_test"], data["R_test"],
                               data["y_test"], data["W_test"], False)

    out_path = (Path("results") / "missing_baselines.csv")
    out_path.parent.mkdir(exist_ok=True)
    write_header = not out_path.exists()

    fields = ["dataset", "variant_code", "variant_name", "seed",
              "precision", "recall", "f1", "acc", "auroc", "ap",
              "wall_seconds"]
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()
    for v in args.variants:
        spec = BASELINES[v]
        for s in args.seeds:
            print(f"==> {args.dataset} {spec.code} ({v}) seed={s}")
            model = build(spec, data["n_kpis"], data["feats_per_kpi"])
            res = train_one(model, train_loader, val_loader, test_loader,
                            data["y_train"], spec,
                            lr_head=cfg.lr_head, seed=s)
            res = {"dataset": args.dataset, **res}
            writer.writerow(res); f.flush()
    f.close()


if __name__ == "__main__":
    main()
