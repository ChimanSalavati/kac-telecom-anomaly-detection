"""
KAC component ablation (TelecomTS / SpotLight / Production).

Runs four variants of KAC on the same final splits used in the main
benchmark tables of the IEEE ICDM 2026 paper:

  V1  Residual-only            : no text branch, no KPI contrastive, no uncertainty
  V2  +Text fusion             : full text branch, no KPI contrastive
  V3  +KPI contrastive         : full text + KPI contrastive with UNIFORM weights
  V4  +Uncertainty (full KAC)  : full text + KPI contrastive with UNCERTAINTY weights

Each variant is trained with five seeds (42, 123, 456, 789, 1337).
Test-set metrics (precision, recall, F1, AUROC, AP) are written to
``results/kac_ablation.csv`` and summarised in
``results/kac_ablation_summary.csv`` for inclusion in the paper.

Usage
-----
The script resolves NPZ inputs and the Chronos-2 residual cache relative
to the current working directory. Run it from the experiment folder so
the layout matches the public-data scripts in ``scripts/``::

    cd experiments/E11_KAC_ablation_TelecomTS
    python -m experiments._shared.kac_ablation --dataset TelecomTS

    cd experiments/E12_KAC_ablation_SpotLight
    python -m experiments._shared.kac_ablation --dataset SpotLight

The ``Production`` configuration is kept for parity with the paper's
Table 6 column but is intentionally not runnable from a clean checkout
because the Production data is Nokia-internal and not redistributed.
See ``docs/data_availability.md`` for the full statement.

Notes
-----
* Reuses the exact model definition, loss functions and training loop
  from the headline KAC notebooks so V4 is identical to the headline
  KAC numbers reported in Tables 2 and 4.
* Variant V1 (Residual-only) uses the same model class but with the
  text encoder bypassed (zero text embeddings + zero attention mask),
  so the comparison isolates which components are active rather than
  architectural differences.
* For dataset-specific hyperparameters (``alpha_supcon``, ``beta_kpi``,
  ``lr_head``, Chronos context ``C``) the values match Table 2 of the
  paper.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# 0. Globals
# ---------------------------------------------------------------------------

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
TEXT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
TEXT_DIM = 768
HIDDEN = 64
PROJ_DIM = 128
NUM_HEADS = 4
DROPOUT = 0.1
TEMPERATURE = 0.07
WIDTH_IDX = 3  # index of the q95-q05 prediction-interval-width feature in the 5-feat-per-KPI residual layout
BATCH_SIZE = 16
EPOCHS = 80
PATIENCE = 15

ANOMALY_WORDS = [
    r"\banomaly\b", r"\banomalous\b", r"\babnormal\b", r"\bdegraded\b",
    r"\bdegradation\b", r"\bfault\b", r"\bfailure\b", r"\bcritical\b",
    r"\bsevere\b", r"\balarm\b", r"\bdrop\b", r"\bdropped\b",
    r"\bdeteriorate\b", r"\bdeterioration\b", r"\bexcessively\b",
    r"\bplummet\b", r"\bplummeted\b", r"\bsurge\b", r"\bspiked?\b",
    r"\bcollapse\b", r"\bcongestion\b", r"\bcongested\b",
    r"\bbottleneck\b", r"\boverloaded?\b", r"\bexhausted\b",
    r"\binterferen\w*\b", r"\bhigh.?error\b", r"\blow.?quality\b",
]
MASK_PATTERN = re.compile("|".join(ANOMALY_WORDS), flags=re.IGNORECASE)


def mask_text(t: str) -> str:
    return MASK_PATTERN.sub("[MASK]", str(t))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. Per-dataset configuration & data loaders
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    npz_train: str
    npz_val: str
    npz_test: str
    cache_dir: str
    npz_kpis_kind: str          # "X_array" or "kpis_dict"
    chronos_context: int
    alpha_supcon: float
    beta_kpi: float
    lr_head: float


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "TelecomTS": DatasetConfig(
        name="TelecomTS",
        npz_train="data/TelecomTS_train.npz",
        npz_val="data/TelecomTS_val.npz",
        npz_test="data/TelecomTS_test.npz",
        cache_dir="data/features_cache",
        npz_kpis_kind="X_array",
        chronos_context=20,
        alpha_supcon=1.0,
        beta_kpi=1.0,
        lr_head=5e-4,
    ),
    "SpotLight": DatasetConfig(
        name="SpotLight",
        npz_train="data/SpotLight_train.npz",
        npz_val="data/SpotLight_val.npz",
        npz_test="data/SpotLight_test.npz",
        cache_dir="data/features_cache",
        npz_kpis_kind="X_array",
        chronos_context=20,
        alpha_supcon=1.0,
        beta_kpi=1.0,
        lr_head=5e-4,
    ),
    "Production": DatasetConfig(
        name="Production",
        npz_train="data/Production_train_final_llm.npz",
        npz_val="data/Production_val_final_llm.npz",
        npz_test="data/Production_test_final_llm.npz",
        cache_dir="data/features_cache",
        npz_kpis_kind="kpis_dict",
        chronos_context=8,
        alpha_supcon=0.1,
        beta_kpi=0.3,
        lr_head=1e-3,
    ),
}


def load_dataset(cfg: DatasetConfig):
    """Load one of the three datasets exactly as the notebooks do."""
    train_npz = np.load(cfg.npz_train, allow_pickle=True)
    val_npz = np.load(cfg.npz_val, allow_pickle=True)
    test_npz = np.load(cfg.npz_test, allow_pickle=True)

    if cfg.npz_kpis_kind == "X_array":
        # TelecomTS / SpotLight: numpy 3-D arrays under key "X" + label key "y"
        X_train_raw = train_npz["X"]
        X_val_raw = val_npz["X"]
        X_test_raw = test_npz["X"]
        y_train = np.asarray(train_npz["y"], dtype=np.int64)
        y_val = np.asarray(val_npz["y"], dtype=np.int64)
        y_test = np.asarray(test_npz["y"], dtype=np.int64)
        kpi_names = (
            [str(c) for c in train_npz["feature_cols"]]
            if "feature_cols" in train_npz.files
            else [f"KPI_{i}" for i in range(X_train_raw.shape[-1])]
        )
    else:
        # Production: list of dicts under key "KPIs" + label key "label"
        sample_kpis = train_npz["KPIs"][0]
        kpi_names = sorted(sample_kpis.keys())

        def to_array(npz_kpis):
            arrs = []
            for d in npz_kpis:
                arrs.append(np.column_stack([
                    np.asarray(d[k], dtype=np.float32) for k in kpi_names
                ]))
            return np.stack(arrs, axis=0)

        X_train_raw = to_array(train_npz["KPIs"])
        X_val_raw = to_array(val_npz["KPIs"])
        X_test_raw = to_array(test_npz["KPIs"])
        y_train = np.asarray(train_npz["label"], dtype=np.int64)
        y_val = np.asarray(val_npz["label"], dtype=np.int64)
        y_test = np.asarray(test_npz["label"], dtype=np.int64)

    texts_train = [str(t) for t in train_npz["descriptions"]]
    texts_val = [str(t) for t in val_npz["descriptions"]]
    texts_test = [str(t) for t in test_npz["descriptions"]]

    cache = Path(cfg.cache_dir)
    R_train_raw = np.load(cache / "residuals_train.npy").astype(np.float32)
    R_val_raw = np.load(cache / "residuals_val.npy").astype(np.float32)
    R_test_raw = np.load(cache / "residuals_test.npy").astype(np.float32)

    n_kpis = len(kpi_names)
    feats_per_kpi = R_train_raw.shape[-1] // n_kpis
    if R_train_raw.shape[-1] != n_kpis * feats_per_kpi:
        raise ValueError(
            f"Residual dimension {R_train_raw.shape[-1]} not divisible by K={n_kpis}"
        )

    width_cols = [k * feats_per_kpi + WIDTH_IDX for k in range(n_kpis)]
    W_train_raw = np.abs(R_train_raw[:, :, width_cols]).astype(np.float32)
    W_val_raw = np.abs(R_val_raw[:, :, width_cols]).astype(np.float32)
    W_test_raw = np.abs(R_test_raw[:, :, width_cols]).astype(np.float32)

    r_mean = R_train_raw.mean(axis=(0, 1), keepdims=True)
    r_std = R_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    R_train = ((R_train_raw - r_mean) / r_std).astype(np.float32)
    R_val = ((R_val_raw - r_mean) / r_std).astype(np.float32)
    R_test = ((R_test_raw - r_mean) / r_std).astype(np.float32)

    return {
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "texts_train": [mask_text(t) for t in texts_train],
        "texts_val":   [mask_text(t) for t in texts_val],
        "texts_test":  [mask_text(t) for t in texts_test],
        "R_train": R_train, "R_val": R_val, "R_test": R_test,
        "W_train": W_train_raw, "W_val": W_val_raw, "W_test": W_test_raw,
        "kpi_names": kpi_names,
        "n_kpis": n_kpis,
        "feats_per_kpi": feats_per_kpi,
    }


# ---------------------------------------------------------------------------
# 2. Loss functions (identical to the main KAC notebooks)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        sim = features @ features.T / self.temperature
        lab = labels.view(-1, 1)
        mask_pos = (lab == lab.T).float()
        mask_pos.fill_diagonal_(0)
        n_pos = mask_pos.sum(dim=1)
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / n_pos.clamp(min=1)
        return -mean_log_prob[valid].mean()


def _kpi_uniform_weights(B: int, K: int, device) -> torch.Tensor:
    return torch.ones(B, K, device=device)


def _kpi_uncertainty_weights(width_raw, eps: float = 1e-6) -> torch.Tensor:
    if width_raw.dim() != 3:
        raise ValueError(
            f"width_raw must have shape [B, T_r, K], got {tuple(width_raw.shape)}"
        )
    mean_width = width_raw.abs().mean(dim=1)            # [B, K]
    W = 1.0 / (mean_width + eps)
    W = W / (W.mean(dim=1, keepdim=True) + eps)
    return W


def kpi_contrastive_loss(z_text_kpi, z_resid_kpi, width_raw,
                         use_uncertainty: bool, temperature: float = TEMPERATURE):
    """Per-KPI bidirectional InfoNCE.

    If ``use_uncertainty`` is True the per-KPI loss is weighted by
    inverse mean Chronos prediction-interval width; otherwise weights
    are uniform (this is variant V3 vs V4 in the ablation).
    """
    B, K, _ = z_text_kpi.shape
    device = z_text_kpi.device
    labels = torch.arange(B, device=device)
    if use_uncertainty:
        Wmat = _kpi_uncertainty_weights(width_raw.to(device))
    else:
        Wmat = _kpi_uniform_weights(B, K, device)

    w_list, l_list = [], []
    for k in range(K):
        logits = z_text_kpi[:, k] @ z_resid_kpi[:, k].T / temperature
        loss_k = (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        ) / 2
        w_list.append(Wmat[:, k].mean())
        l_list.append(loss_k)
    w = torch.stack(w_list)
    ell = torch.stack(l_list)
    return (w * ell).sum() / w.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# 3. KAC model (identical layers to main notebook; ablation flag bypasses text)
# ---------------------------------------------------------------------------

class KPIAwareContrastiveModel(nn.Module):
    def __init__(self, text_encoder, resid_dim, text_dim, d_model, proj_dim,
                 n_kpis, feats_per_kpi, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.text_encoder = text_encoder
        self.n_kpis = n_kpis
        self.feats_per_kpi = feats_per_kpi

        self.text_proj = nn.Linear(text_dim, d_model)
        self.resid_proj = nn.Linear(resid_dim, d_model)

        self.kpi_queries = nn.Parameter(torch.randn(n_kpis, d_model) * 0.02)
        self.kpi_text_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.resid_kpi_proj = nn.Linear(feats_per_kpi, d_model)

        self.kpi_text_cp = nn.Linear(d_model, proj_dim)
        self.kpi_resid_cp = nn.Linear(d_model, proj_dim)

        self.supcon_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids, attention_mask, x_resid, use_text: bool = True):
        B = input_ids.shape[0]
        if use_text:
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            t = self.text_proj(text_out.last_hidden_state)            # [B, L, d]
            mask_f = attention_mask.unsqueeze(-1).float()
            t_pool = (t * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # [B, d]
            key_pad = (attention_mask == 0)
        else:
            # Residual-only variant: replace text branch with zeros so the
            # gate saturates at 0.5 and the KPI queries see a constant.
            L = input_ids.shape[1]
            t = torch.zeros(B, L, self.text_proj.out_features, device=input_ids.device)
            t_pool = torch.zeros(B, self.text_proj.out_features, device=input_ids.device)
            key_pad = None  # all positions valid (zeros)

        r = self.resid_proj(x_resid)                                  # [B, T_r, d]

        kpi_q = self.kpi_queries.unsqueeze(0).expand(B, -1, -1)
        kpi_text, kpi_attn_w = self.kpi_text_attn(
            query=kpi_q, key=t, value=t,
            key_padding_mask=key_pad, need_weights=True,
        )

        kpi_resid_list = []
        for k in range(self.n_kpis):
            s, e = k * self.feats_per_kpi, (k + 1) * self.feats_per_kpi
            r_k = self.resid_kpi_proj(x_resid[:, :, s:e]).mean(dim=1)
            kpi_resid_list.append(r_k)
        kpi_resid = torch.stack(kpi_resid_list, dim=1)

        z_t_kpi = F.normalize(self.kpi_text_cp(kpi_text), dim=-1)
        z_r_kpi = F.normalize(self.kpi_resid_cp(kpi_resid), dim=-1)

        gate = torch.sigmoid(self.gate(t_pool)).unsqueeze(1)
        r_g = r * gate

        x = torch.cat([t, r_g], dim=1)
        x = self.norm(x + self.drop(self.conv(x.transpose(1, 2)).transpose(1, 2)))
        cls = self.cls_token.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query=cls, key=x, value=x)
        pooled = pooled.squeeze(1)

        z_sc = F.normalize(self.supcon_proj(pooled), dim=-1)
        logits = self.head(pooled).squeeze(-1)
        return logits, z_sc, z_t_kpi, z_r_kpi, kpi_attn_w


def build_model(n_kpis: int, feats_per_kpi: int) -> KPIAwareContrastiveModel:
    # ``local_files_only=True`` mirrors the headline KAC notebooks, which
    # avoid hitting Hugging Face Hub at training time. This sidesteps the
    # SSL cert errors that occur on some macOS Python installs.
    try:
        text_encoder = AutoModel.from_pretrained(
            TEXT_MODEL_NAME, local_files_only=True,
        )
    except OSError:
        text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    for p in text_encoder.parameters():
        p.requires_grad = False
    for layer in text_encoder.transformer.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True

    resid_dim = n_kpis * feats_per_kpi
    return KPIAwareContrastiveModel(
        text_encoder=text_encoder, resid_dim=resid_dim,
        text_dim=TEXT_DIM, d_model=HIDDEN, proj_dim=PROJ_DIM,
        n_kpis=n_kpis, feats_per_kpi=feats_per_kpi,
        num_heads=NUM_HEADS, dropout=DROPOUT,
    )


# ---------------------------------------------------------------------------
# 4. Dataset wrapper
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    def __init__(self, input_ids, attention_mask, residuals, labels, widths):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.residuals = torch.tensor(residuals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.widths = torch.tensor(widths, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "resid": self.residuals[i],
            "label": self.labels[i],
            "width_raw": self.widths[i],
        }


# ---------------------------------------------------------------------------
# 5. Training & evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion_bce, *, use_text: bool, threshold: float = 0.5):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        resid = batch["resid"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        logits = model(ids, mask, resid, use_text=use_text)[0]
        loss = criterion_bce(logits, y)
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    al = torch.cat(all_logits).numpy()
    yl = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-al))
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        yl, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(yl, preds)
    try:    auroc = roc_auc_score(yl, probs)
    except: auroc = float("nan")
    try:    ap = average_precision_score(yl, probs)
    except: ap = float("nan")
    return {
        "loss": total_loss / max(total_n, 1),
        "precision": p, "recall": r, "f1": f1,
        "acc": acc, "auroc": auroc, "ap": ap,
    }


@dataclass
class VariantSpec:
    code: str        # V1 / V2 / V3 / V4
    name: str        # human-readable name for the table
    use_text: bool
    use_kpi_contrastive: bool
    use_uncertainty: bool


VARIANTS: List[VariantSpec] = [
    VariantSpec("V1", "Residual-only",                                  False, False, False),
    VariantSpec("V2", "+ Text fusion (no KPI contrastive)",             True,  False, False),
    VariantSpec("V3", "+ KPI contrastive (uniform weights)",            True,  True,  False),
    VariantSpec("V4", "Full KAC (uncertainty-weighted, ours)",          True,  True,  True),
]


def train_variant(model, train_loader, val_loader, test_loader, y_train,
                  variant: VariantSpec, alpha_supcon: float, beta_kpi: float,
                  lr_head: float, *, epochs: int = EPOCHS,
                  patience: int = PATIENCE, seed: int = 42):
    set_seed(seed)
    model = model.to(DEVICE)

    enc_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("text_encoder")]
    optim = torch.optim.AdamW([
        {"params": enc_params,   "lr": 2e-5, "weight_decay": 1e-2},
        {"params": other_params, "lr": lr_head, "weight_decay": 1e-2},
    ])

    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    pw = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    crit_sc = SupConLoss(temperature=TEMPERATURE)

    best_f1, best_state, pat = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            resid = batch["resid"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            width_raw = batch["width_raw"].to(DEVICE)

            logits, z_sc, z_t_kpi, z_r_kpi, _ = model(
                ids, mask, resid, use_text=variant.use_text
            )

            l_bce = crit_bce(logits, y)
            l_sc = crit_sc(z_sc, y.long())

            if variant.use_kpi_contrastive:
                l_kpi = kpi_contrastive_loss(
                    z_t_kpi, z_r_kpi, width_raw,
                    use_uncertainty=variant.use_uncertainty,
                )
                loss = l_bce + alpha_supcon * l_sc + beta_kpi * l_kpi
            else:
                loss = l_bce + alpha_supcon * l_sc

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        val = evaluate(model, val_loader, crit_bce, use_text=variant.use_text)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, crit_bce, use_text=variant.use_text)
    test["best_val_f1"] = best_f1
    test["epochs_run"] = epoch
    return test


# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------

def make_loaders(data, tokens_train, tokens_val, tokens_test, seed: int):
    train_ds = ContrastiveDataset(
        tokens_train["input_ids"], tokens_train["attention_mask"],
        data["R_train"], data["y_train"], data["W_train"],
    )
    val_ds = ContrastiveDataset(
        tokens_val["input_ids"], tokens_val["attention_mask"],
        data["R_val"], data["y_val"], data["W_val"],
    )
    test_ds = ContrastiveDataset(
        tokens_test["input_ids"], tokens_test["attention_mask"],
        data["R_test"], data["y_test"], data["W_test"],
    )
    g = torch.Generator(); g.manual_seed(seed)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def run_ablation(
    dataset: str,
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
    out: str = "results/kac_ablation.csv",
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> Tuple[List[Dict], List[Dict]]:
    """Programmatic entry point used by the per-dataset notebooks.

    Parameters
    ----------
    dataset
        One of ``"TelecomTS"``, ``"SpotLight"``, ``"Production"``. The
        function expects to be called from inside the corresponding
        ``evaluation_ver2/<dataset>`` directory so that the relative
        paths to ``data/...`` resolve to the same files the headline
        ``KAC_Uncertainty_*_rawwidth.ipynb`` notebooks use.
    seeds
        List of integer seeds. Defaults to ``[42, 123, 456, 789, 1337]``.
    variants
        List of variant codes to run. Defaults to ``["V1", "V2", "V3", "V4"]``.
    out
        Path for the per-(variant, seed) CSV. The aggregate summary is
        always written next to it as ``kac_ablation_summary.csv``.

    Returns
    -------
    rows, summary_rows
        ``rows`` has one entry per (variant, seed); ``summary_rows`` has
        one entry per variant with mean/std across seeds.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1337]
    if variants is None:
        variants = [v.code for v in VARIANTS]

    cfg = DATASET_CONFIGS[dataset]
    print(f"=== KAC Component Ablation — {cfg.name} ===")
    print(f"Device: {DEVICE}")

    data = load_dataset(cfg)
    print(f"Train/Val/Test: {len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}")
    print(f"K (KPIs): {data['n_kpis']}, feats/KPI: {data['feats_per_kpi']}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL_NAME, local_files_only=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    tokens_train = tokenizer(data["texts_train"], padding="max_length",
                             truncation=True, max_length=MAX_LEN, return_tensors="pt")
    tokens_val = tokenizer(data["texts_val"], padding="max_length",
                           truncation=True, max_length=MAX_LEN, return_tensors="pt")
    tokens_test = tokenizer(data["texts_test"], padding="max_length",
                            truncation=True, max_length=MAX_LEN, return_tensors="pt")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for v in VARIANTS:
        if v.code not in variants:
            continue
        print(f"\n--- Variant {v.code}: {v.name} ---")
        for seed in seeds:
            t0 = time.time()
            train_loader, val_loader, test_loader = make_loaders(
                data, tokens_train, tokens_val, tokens_test, seed=seed,
            )
            model = build_model(data["n_kpis"], data["feats_per_kpi"])
            test = train_variant(
                model, train_loader, val_loader, test_loader,
                data["y_train"], variant=v,
                alpha_supcon=cfg.alpha_supcon,
                beta_kpi=cfg.beta_kpi,
                lr_head=cfg.lr_head,
                epochs=epochs, patience=patience, seed=seed,
            )
            elapsed = time.time() - t0
            row = {
                "dataset": cfg.name,
                "variant_code": v.code,
                "variant_name": v.name,
                "use_text": int(v.use_text),
                "use_kpi_contrastive": int(v.use_kpi_contrastive),
                "use_uncertainty": int(v.use_uncertainty),
                "seed": seed,
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "auroc": test["auroc"],
                "ap": test["ap"],
                "best_val_f1": test["best_val_f1"],
                "epochs_run": test["epochs_run"],
                "wall_seconds": round(elapsed, 1),
            }
            rows.append(row)
            print(f"  seed={seed:>5d} | F1={test['f1']:.4f} | AUROC={test['auroc']:.4f}"
                  f" | AP={test['ap']:.4f} | {elapsed:.0f}s")

            # Append-as-you-go so partial runs are not lost.
            file_exists = out_path.exists() and out_path.stat().st_size > 0
            with open(out_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

    # ------------------------------------------------------------------
    # Aggregate (mean +- std over seeds)
    # ------------------------------------------------------------------
    print("\n=== Per-variant mean +- std over seeds ===")
    summary_path = out_path.parent / "kac_ablation_summary.csv"
    summary_rows: List[Dict] = []
    print(f"{'Code':<4}  {'Variant':<48}  {'F1 (mean+-std)':<18}  {'AUROC':<18}  {'AP':<18}")
    for v in VARIANTS:
        if v.code not in variants:
            continue
        sel = [r for r in rows if r["variant_code"] == v.code]
        if not sel:
            continue
        f1 = np.array([r["f1"] for r in sel])
        auroc = np.array([r["auroc"] for r in sel])
        ap = np.array([r["ap"] for r in sel])
        prec = np.array([r["precision"] for r in sel])
        rec = np.array([r["recall"] for r in sel])
        srow = {
            "dataset": cfg.name,
            "variant_code": v.code,
            "variant_name": v.name,
            "n_seeds": len(sel),
            "precision_mean": prec.mean(), "precision_std": prec.std(),
            "recall_mean": rec.mean(), "recall_std": rec.std(),
            "f1_mean": f1.mean(), "f1_std": f1.std(),
            "auroc_mean": auroc.mean(), "auroc_std": auroc.std(),
            "ap_mean": ap.mean(), "ap_std": ap.std(),
        }
        summary_rows.append(srow)
        print(f"{v.code:<4}  {v.name:<48}  {f1.mean():.3f} +- {f1.std():.3f}     "
              f"{auroc.mean():.3f} +- {auroc.std():.3f}     "
              f"{ap.mean():.3f} +- {ap.std():.3f}")

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for srow in summary_rows:
                w.writerow(srow)
    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")
    return rows, summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[42, 123, 456, 789, 1337])
    ap.add_argument("--variants", type=str, nargs="*",
                    default=[v.code for v in VARIANTS],
                    help="Variant codes to run, e.g. V1 V2 V3 V4")
    ap.add_argument("--out", type=str, default="results/kac_ablation.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()
    run_ablation(
        dataset=args.dataset, seeds=args.seeds, variants=args.variants,
        out=args.out, epochs=args.epochs, patience=args.patience,
    )


if __name__ == "__main__":
    main()
