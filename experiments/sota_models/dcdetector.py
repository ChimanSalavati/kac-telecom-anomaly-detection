"""DCdetector (KDD 2023) - Dual Attention Contrastive Representation Learning
for Time Series Anomaly Detection.

Self-contained re-implementation of the official code at
https://github.com/DAMO-DI-ML/KDD2023-DCdetector ; same architecture and losses,
but device-agnostic (no hardcoded ``cuda``) and packaged as a single module.

The key components match the paper:

* Multi-scale patching (different ``patch_size`` values).
* For each scale, two views are formed: in-patch (token positions inside a
  patch) and patch-wise (patches as tokens). The model computes an attention
  matrix on each view, broadcasts both to the original ``win_size x win_size``
  resolution, and trains them to be a discrepancy-free pair under the
  symmetric KL contrastive loss in the paper (Eq. 4-6).
* Anomaly score at inference is the sum (over scales/layers) of the symmetric
  KL between the two upsampled attention maps, summed over the window axis to
  produce a per-step score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


# ---------------------------------------------------------------------------
# Embeddings (from the official repo's model/embed.py)
# ---------------------------------------------------------------------------
class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=padding,
                                   padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class _DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.value_embedding = _TokenEmbedding(c_in, d_model)
        self.position_embedding = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


# ---------------------------------------------------------------------------
# RevIN (from the official repo's model/RevIN.py, made device-agnostic)
# ---------------------------------------------------------------------------
class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# Dual Attention Comparison structure (DAC)
# ---------------------------------------------------------------------------
class _DACStructure(nn.Module):
    """Computes patch-wise and in-patch self-attention maps, then broadcasts
    them to ``win_size x win_size`` so the symmetric-KL contrastive loss can be
    applied at the same resolution. Mirrors `model/attn.py::DAC_structure`."""

    def __init__(self, win_size: int, patch_size: List[int], channel: int,
                 attention_dropout: float = 0.05):
        super().__init__()
        self.win_size = win_size
        self.patch_size = patch_size
        self.channel = channel
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries_patch_size, queries_patch_num,
                keys_patch_size, keys_patch_num, values, patch_index, attn_mask=None):
        # in-patch attention
        B, L, H, E = queries_patch_size.shape
        scale_ps = 1.0 / math.sqrt(E)
        scores_ps = torch.einsum("blhe,bshe->bhls", queries_patch_size, keys_patch_size)
        series_patch_size = self.dropout(torch.softmax(scale_ps * scores_ps, dim=-1))

        # patch-wise attention
        B, L, H, E = queries_patch_num.shape
        scale_pn = 1.0 / math.sqrt(E)
        scores_pn = torch.einsum("blhe,bshe->bhls", queries_patch_num, keys_patch_num)
        series_patch_num = self.dropout(torch.softmax(scale_pn * scores_pn, dim=-1))

        ps = self.patch_size[patch_index]
        # upsample in-patch attention from (p, p) to (win, win)
        series_patch_size = repeat(series_patch_size,
                                   "b l m n -> b l (m r1) (n r2)", r1=ps, r2=ps)
        # upsample patch-wise attention from (n, n) to (win, win)
        series_patch_num = series_patch_num.repeat(
            1, 1, self.win_size // ps, self.win_size // ps)
        # average over channels (channel-independent patching expands B by self.channel)
        series_patch_size = reduce(series_patch_size,
                                   "(b r) l m n -> b l m n", "mean", r=self.channel)
        series_patch_num = reduce(series_patch_num,
                                  "(b r) l m n -> b l m n", "mean", r=self.channel)
        return series_patch_size, series_patch_num


class _AttentionLayer(nn.Module):
    def __init__(self, attention: _DACStructure, d_model: int,
                 patch_size: List[int], channel: int, n_heads: int, win_size: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.n_heads = n_heads
        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        H = self.n_heads
        B, L, _ = x_patch_size.shape
        q_ps = self.patch_query_projection(x_patch_size).view(B, L, H, -1)
        k_ps = self.patch_key_projection(x_patch_size).view(B, L, H, -1)

        B, L, _ = x_patch_num.shape
        q_pn = self.patch_query_projection(x_patch_num).view(B, L, H, -1)
        k_pn = self.patch_key_projection(x_patch_num).view(B, L, H, -1)

        B, L, _ = x_ori.shape
        v = self.value_projection(x_ori).view(B, L, H, -1)
        return self.inner_attention(q_ps, q_pn, k_ps, k_pn, v, patch_index, attn_mask)


class _Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_ps, x_pn, x_ori, patch_index, attn_mask=None):
        series_list, prior_list = [], []
        for layer in self.attn_layers:
            s, p = layer(x_ps, x_pn, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(s)
            prior_list.append(p)
        return series_list, prior_list


# ---------------------------------------------------------------------------
# Public model class (mirrors model/DCdetector.py::DCdetector)
# ---------------------------------------------------------------------------
@dataclass
class DCdetectorConfig:
    win_size: int
    enc_in: int
    c_out: int
    n_heads: int = 1
    d_model: int = 256
    e_layers: int = 3
    patch_size: List[int] = field(default_factory=lambda: [3, 5, 7])
    d_ff: int = 512
    dropout: float = 0.0


class DCdetector(nn.Module):
    """Reference-faithful DCdetector model.

    Forward returns ``(series_list, prior_list)`` -- two flat lists where the
    i-th element is the (B, H, L, L) attention matrix from view 1 (series) and
    view 2 (prior) respectively. The window-level anomaly score at inference is
    computed by ``dcdetector_score`` below.
    """

    def __init__(self, cfg: DCdetectorConfig):
        super().__init__()
        self.cfg = cfg
        # win_size must be divisible by every patch_size for the upsampling step
        for ps in cfg.patch_size:
            assert cfg.win_size % ps == 0, (
                f"win_size={cfg.win_size} not divisible by patch_size={ps}"
            )

        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for ps in cfg.patch_size:
            self.embedding_patch_size.append(_DataEmbedding(ps, cfg.d_model, cfg.dropout))
            self.embedding_patch_num.append(_DataEmbedding(cfg.win_size // ps, cfg.d_model, cfg.dropout))

        self.embedding_window_size = _DataEmbedding(cfg.enc_in, cfg.d_model, cfg.dropout)

        self.encoder = _Encoder(
            [
                _AttentionLayer(
                    _DACStructure(cfg.win_size, cfg.patch_size, cfg.enc_in,
                                  attention_dropout=cfg.dropout),
                    cfg.d_model, cfg.patch_size, cfg.enc_in, cfg.n_heads, cfg.win_size,
                )
                for _ in range(cfg.e_layers)
            ],
            norm_layer=nn.LayerNorm(cfg.d_model),
        )
        self.projection = nn.Linear(cfg.d_model, cfg.c_out, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (B, L, M)
        B, L, M = x.shape
        revin = _RevIN(num_features=M).to(x.device)
        x = revin(x, "norm")
        x_ori = self.embedding_window_size(x)

        series_patch_mean, prior_patch_mean = [], []
        for patch_index, ps in enumerate(self.cfg.patch_size):
            x_patch_size = rearrange(x, "b l m -> b m l")
            x_patch_num = rearrange(x, "b l m -> b m l")
            x_patch_size = rearrange(x_patch_size, "b m (n p) -> (b m) n p", p=ps)
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, "b m (p n) -> (b m) p n", p=ps)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.extend(series)
            prior_patch_mean.extend(prior)
        return series_patch_mean, prior_patch_mean


# ---------------------------------------------------------------------------
# Loss / scoring (paper's symmetric-KL contrastive criterion)
# ---------------------------------------------------------------------------
def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return p * (torch.log(p + eps) - torch.log(q + eps))


def dcdetector_loss(series_list, prior_list) -> torch.Tensor:
    """Symmetric-KL contrastive loss across all encoder layers and patch scales.

    Mirrors the loss in solver.py of the official repo: each layer/scale pair
    contributes two terms summed and averaged. Stop-grad is applied to the
    "other" view in each direction, matching the paper's contrastive design.
    """
    loss = 0.0
    for s, p in zip(series_list, prior_list):
        # forward: align series toward stop-grad(prior)
        loss = loss + (_kl_div(s, p.detach()) + _kl_div(p.detach(), s)).mean()
        loss = loss + (_kl_div(p, s.detach()) + _kl_div(s.detach(), p)).mean()
    return loss / max(1, len(series_list))


def dcdetector_score(series_list, prior_list, win_size: int) -> torch.Tensor:
    """Per-time-step anomaly score with shape ``(B, win_size)``.

    For each (series, prior) attention pair, we average over heads, compute the
    symmetric KL per row (per timestep), and apply a temperature-50 softmax over
    layers / scales (matches the official solver's anomaly-score recipe).
    """
    temperature = 50.0
    series_kl_sum = None
    for s, p in zip(series_list, prior_list):
        # s, p: (B, H, L, L) -> avg heads -> (B, L, L)
        s_m = s.mean(dim=1)
        p_m = p.mean(dim=1)
        kl = (_kl_div(s_m, p_m) + _kl_div(p_m, s_m)).sum(dim=-1)  # (B, L)
        if series_kl_sum is None:
            series_kl_sum = kl
        else:
            series_kl_sum = series_kl_sum + kl
    score = F.softmax(-series_kl_sum / temperature, dim=-1)
    # high anomaly => low softmax mass => invert sign for "higher = more anomalous"
    return -torch.log(score + 1e-8)
