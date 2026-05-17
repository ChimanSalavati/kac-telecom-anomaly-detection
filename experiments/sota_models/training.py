"""Training/inference loops for the 5 vendored SOTA AD baselines.

Each ``train_*`` function follows the original paper protocol:

* DCdetector (KDD'23): train on normal-only windows with the symmetric-KL
  contrastive loss; anomaly score is per-step KL on the unsupervised attention
  pair.
* TimesNet (ICLR'23): MSE reconstruction on normal-only windows; per-step MSE
  is the anomaly score.
* ModernTCN (ICLR'24): same recipe (MSE reconstruction on normal-only).
* MEMTO (NeurIPS'23): two-phase training -- random-init memory + gathering
  + entropy + reconstruction; then re-init memory with k-means on phase-1
  encoder outputs and continue training; anomaly score is bi-dim deviation.
* D3R (NeurIPS'23): joint stable+trend+reconstruction loss on normal data;
  anomaly score is per-step diffusion reconstruction MSE.

All functions take ``(B, T, C)`` tensors and return per-step scores of shape
``(N_test_windows, T)``; ``common.reduce_window_score`` collapses the time
axis to a window-level score.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    best_f1_threshold,
    compute_metrics,
    reduce_window_score,
    select_normal_only,
)
from .dcdetector import (
    DCdetector,
    DCdetectorConfig,
    dcdetector_loss,
    dcdetector_score,
)
from .timesnet import TimesNet, TimesNetConfig
from .moderntcn import ModernTCN, ModernTCNConfig
from .memto import (
    MEMTO,
    MemtoConfig,
    memto_loss,
    memto_score,
)
from .d3r import D3R, D3RConfig


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_time_features(T: int, time_num: int = 4, batch: int = 1, device: str = "cpu") -> torch.Tensor:
    """Synthetic positional time features for D3R.

    The original D3R paper uses minute/hour features. For our pre-segmented
    KPI windows, we use a small bank of sin/cos positional features of the
    same shape ``(B, T, time_num)``.
    """
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T)
    feats = []
    for k in range(1, time_num // 2 + 1):
        feats.append(torch.sin(2 * math.pi * k * pos / T))
        feats.append(torch.cos(2 * math.pi * k * pos / T))
    feats = torch.stack(feats[:time_num], dim=-1)  # (1, T, time_num)
    return feats.expand(batch, T, time_num).contiguous()


# ---------------------------------------------------------------------------
# DCdetector
# ---------------------------------------------------------------------------
def train_dcdetector(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    win_size: int,
    enc_in: int,
    *,
    patch_size=(3, 5, 7),
    d_model: int = 256,
    e_layers: int = 3,
    n_heads: int = 1,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train DCdetector on normal-only windows; return per-step scores on test."""
    _seed_all(seed)
    # restrict patch_size list to those that divide win_size
    patch_size = [p for p in patch_size if win_size % p == 0]
    if not patch_size:
        patch_size = [d for d in (4, 2, 1) if win_size % d == 0][:1]
    cfg = DCdetectorConfig(
        win_size=win_size, enc_in=enc_in, c_out=enc_in,
        n_heads=n_heads, d_model=d_model, e_layers=e_layers,
        patch_size=list(patch_size),
    )
    model = DCdetector(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xn = torch.from_numpy(X_train_normal).float()
    loader = DataLoader(TensorDataset(Xn), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            series, prior = model(xb)
            loss = dcdetector_loss(series, prior)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  [DCdetector] ep {ep+1}/{epochs} loss={ep_loss/max(1,len(loader)):.4f} ({time.time()-t0:.1f}s)")

    # score on test
    model.eval()
    Xt = torch.from_numpy(X_test).float()
    test_loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    scores_per_step = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            series, prior = model(xb)
            s = dcdetector_score(series, prior, win_size).cpu().numpy()
            scores_per_step.append(s)
    return np.concatenate(scores_per_step, axis=0)


# ---------------------------------------------------------------------------
# TimesNet
# ---------------------------------------------------------------------------
def train_timesnet(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    win_size: int,
    enc_in: int,
    *,
    d_model: int = 64,
    d_ff: int = 64,
    e_layers: int = 2,
    top_k: int = 3,
    num_kernels: int = 6,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train TimesNet on normal-only data; return per-step MSE on test."""
    _seed_all(seed)
    cfg = TimesNetConfig(seq_len=win_size, enc_in=enc_in, c_out=enc_in,
                         d_model=d_model, d_ff=d_ff, e_layers=e_layers,
                         top_k=top_k, num_kernels=num_kernels)
    model = TimesNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xn = torch.from_numpy(X_train_normal).float()
    loader = DataLoader(TensorDataset(Xn), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            yhat = model(xb)
            loss = crit(yhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  [TimesNet] ep {ep+1}/{epochs} loss={ep_loss/max(1,len(loader)):.4f} ({time.time()-t0:.1f}s)")

    model.eval()
    Xt = torch.from_numpy(X_test).float()
    test_loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    out_scores = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            yhat = model(xb)
            mse = ((yhat - xb) ** 2).mean(dim=-1)  # (B, T)
            out_scores.append(mse.cpu().numpy())
    return np.concatenate(out_scores, axis=0)


# ---------------------------------------------------------------------------
# ModernTCN
# ---------------------------------------------------------------------------
def _largest_divisor_le(n: int, cap: int = 8) -> int:
    """Largest integer in [1, cap] that divides n; used to pick patch_size."""
    for k in range(min(cap, n), 0, -1):
        if n % k == 0:
            return k
    return 1


def train_moderntcn(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    win_size: int,
    enc_in: int,
    *,
    patch_size: Optional[int] = None,
    large_size: int = 51,
    small_size: int = 5,
    d_model: int = 64,
    ffn_ratio: int = 1,
    num_blocks: int = 1,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    _seed_all(seed)
    if patch_size is None:
        patch_size = _largest_divisor_le(win_size, cap=8)
    # large kernel must be odd and <= a reasonable factor of seq_len after patching
    if large_size % 2 == 0:
        large_size += 1
    if small_size % 2 == 0:
        small_size += 1
    cfg = ModernTCNConfig(
        seq_len=win_size, enc_in=enc_in,
        patch_size=patch_size, patch_stride=patch_size,
        large_size=[large_size], small_size=[small_size],
        dims=[d_model], ffn_ratio=ffn_ratio, num_blocks=[num_blocks],
    )
    model = ModernTCN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xn = torch.from_numpy(X_train_normal).float()
    loader = DataLoader(TensorDataset(Xn), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            yhat = model(xb)
            loss = crit(yhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  [ModernTCN] ep {ep+1}/{epochs} loss={ep_loss/max(1,len(loader)):.4f} ({time.time()-t0:.1f}s)")

    model.eval()
    Xt = torch.from_numpy(X_test).float()
    test_loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            yhat = model(xb)
            mse = ((yhat - xb) ** 2).mean(dim=-1)
            out.append(mse.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# MEMTO (two-phase training)
# ---------------------------------------------------------------------------
def train_memto(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    win_size: int,
    enc_in: int,
    *,
    n_memory: int = 10,
    d_model: int = 256,
    n_heads: int = 4,
    e_layers: int = 3,
    d_ff: int = 256,
    lambda_gather: float = 0.1,
    lambda_entropy: float = 0.01,
    epochs_phase1: int = 3,
    epochs_phase2: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    _seed_all(seed)
    cfg = MemtoConfig(
        win_size=win_size, enc_in=enc_in, c_out=enc_in,
        n_memory=n_memory, d_model=d_model, n_heads=n_heads,
        e_layers=e_layers, d_ff=d_ff,
    )
    model = MEMTO(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xn = torch.from_numpy(X_train_normal).float()
    loader = DataLoader(TensorDataset(Xn), batch_size=batch_size, shuffle=True, drop_last=False)

    # ---- phase 1: random-init memory ---------------------------------------
    model.train()
    for ep in range(epochs_phase1):
        t0 = time.time()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)
            loss = memto_loss(out, xb, lambda_gather=lambda_gather, lambda_entropy=lambda_entropy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  [MEMTO p1] ep {ep+1}/{epochs_phase1} loss={ep_loss/max(1,len(loader)):.4f} ({time.time()-t0:.1f}s)")

    # ---- k-means re-init of memory ----------------------------------------
    model.eval()
    feats = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            feats.append(model.encode(xb).cpu())
    feats = torch.cat(feats, dim=0)
    if verbose:
        print(f"  [MEMTO] running k-means on {feats.shape} ...")
    model.kmeans_init_memory(feats, seed=seed)

    # ---- phase 2: same loss, same model, with kmeans-initialized memory ----
    model.train()
    for ep in range(epochs_phase2):
        t0 = time.time()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)
            loss = memto_loss(out, xb, lambda_gather=lambda_gather, lambda_entropy=lambda_entropy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  [MEMTO p2] ep {ep+1}/{epochs_phase2} loss={ep_loss/max(1,len(loader)):.4f} ({time.time()-t0:.1f}s)")

    # ---- inference: freeze memory --------------------------------------------
    model.eval()
    model.mem.freeze()
    Xt = torch.from_numpy(X_test).float()
    test_loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    out_scores = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            out = model(xb)
            s = memto_score(out, xb).cpu().numpy()  # (B, T)
            out_scores.append(s)
    return np.concatenate(out_scores, axis=0)


# ---------------------------------------------------------------------------
# D3R
# ---------------------------------------------------------------------------
def train_d3r(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    win_size: int,
    enc_in: int,
    *,
    model_dim: int = 128,
    ff_dim: int = 128,
    atten_dim: int = 16,
    block_num: int = 3,
    head_num: int = 4,
    time_num: int = 4,
    t: int = 200,
    d: int = 5,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    _seed_all(seed)
    cfg = D3RConfig(window_size=win_size, feature_num=enc_in, time_num=time_num,
                    model_dim=model_dim, ff_dim=ff_dim, atten_dim=atten_dim,
                    block_num=block_num, head_num=head_num, t=t, d=d)
    model = D3R(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xn = torch.from_numpy(X_train_normal).float()
    loader = DataLoader(TensorDataset(Xn), batch_size=batch_size, shuffle=True, drop_last=False)

    # Pre-build a single time-feature tensor of size (1, T, time_num)
    time_template = _build_time_features(win_size, time_num=time_num, batch=1, device=device)

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            tb = time_template.expand(xb.shape[0], -1, -1).contiguous()
            stable, trend, recon = model(xb, tb, p=0.0)
            target = xb
            # reconstruction loss + trend regularizer + stable regularizer
            loss = F.mse_loss(recon, target) + 0.1 * F.mse_loss(stable, target - trend)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  [D3R] ep {ep+1}/{epochs} loss={ep_loss/max(1,len(loader)):.4f} ({time.time()-t0:.1f}s)")

    model.eval()
    Xt = torch.from_numpy(X_test).float()
    test_loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    out_scores = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            tb = time_template.expand(xb.shape[0], -1, -1).contiguous()
            _, _, recon = model(xb, tb, p=0.0)
            mse = ((recon - xb) ** 2).mean(dim=-1)  # (B, T)
            out_scores.append(mse.cpu().numpy())
    return np.concatenate(out_scores, axis=0)


# ---------------------------------------------------------------------------
# Run-and-evaluate convenience wrapper
# ---------------------------------------------------------------------------
def evaluate_method(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
):
    """Train ``method`` on normal-only ``X_train``, score val/test, pick the best
    F1 threshold on ``X_val`` (or the union train+val), and return metrics on
    test along with the raw test scores.

    ``X_train``, ``X_val``, ``X_test`` are ``(N, T, C)`` float arrays.
    """
    win_size = X_train.shape[1]
    enc_in = X_train.shape[2]
    X_train_normal = select_normal_only(X_train, y_train)

    fn = {
        "DCdetector": train_dcdetector,
        "TimesNet":   train_timesnet,
        "ModernTCN":  train_moderntcn,
        "MEMTO":      train_memto,
        "D3R":        train_d3r,
    }[method]

    # We score on val+test in one go (concatenated) so train and threshold
    # selection use the same model weights.
    X_score = np.concatenate([X_val, X_test], axis=0)
    point_scores = fn(X_train_normal, X_score, win_size, enc_in,
                      seed=seed, device=device, verbose=verbose, **kwargs)

    # split scores back into val/test
    val_scores_pt = point_scores[: X_val.shape[0]]
    test_scores_pt = point_scores[X_val.shape[0]:]

    val_scores = reduce_window_score(val_scores_pt, "mean")
    test_scores = reduce_window_score(test_scores_pt, "mean")

    # threshold selection: use validation labels
    threshold, _ = best_f1_threshold(val_scores, y_val)
    metrics = compute_metrics(test_scores, y_test, threshold)
    return {
        "method": method,
        "seed": seed,
        **metrics,
        "test_scores": test_scores,
    }
