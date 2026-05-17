"""MEMTO (NeurIPS 2023) - Memory-guided Transformer for Multivariate Time
Series Anomaly Detection.

Self-contained re-implementation of https://github.com/gunny97/MEMTO. Trained
in two phases:

1. *first_train* with random-initialized memory items (memory updated via the
   gated update rule).
2. *second_train* with memory items initialized by k-means on the encoder
   outputs of the first phase.

At inference, ``second_train`` weights are reused and memory is **frozen**. The
anomaly score is the bi-dimensional deviation: per-step squared reconstruction
error * latent gathering distance to the nearest memory item.

Differences from upstream:
* Removed all ``.cuda()`` and on-disk memory-item save/load; memory state is
  carried entirely in ``MemoryModule.mem`` (a buffer).
* Single-file packaging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class _TokenEmbedding(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        pad = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(in_dim, d_model, kernel_size=3, padding=pad,
                              padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class _InputEmbedding(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.token = _TokenEmbedding(in_dim, d_model)
        self.pos = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.token(x) + self.pos(x))


# ---------------------------------------------------------------------------
# Self-attention encoder layer
# ---------------------------------------------------------------------------
class _Attention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        N, L, H, C = q.shape
        scale = 1.0 / math.sqrt(C)
        scores = torch.einsum("nlhd,nshd->nhls", q, k)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("nhls,nshd->nlhd", attn, v).contiguous()


class _AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dk = d_model // n_heads
        self.dv = d_model // n_heads
        self.h = n_heads
        self.W_Q = nn.Linear(d_model, n_heads * self.dk)
        self.W_K = nn.Linear(d_model, n_heads * self.dk)
        self.W_V = nn.Linear(d_model, n_heads * self.dv)
        self.out = nn.Linear(n_heads * self.dv, d_model)
        self.attn = _Attention(dropout=dropout)

    def forward(self, x):
        N, L, _ = x.shape
        Q = self.W_Q(x).view(N, L, self.h, -1)
        K = self.W_K(x).view(N, L, self.h, -1)
        V = self.W_V(x).view(N, L, self.h, -1)
        out = self.attn(Q, K, V).view(N, L, -1)
        return self.out(out)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.attn = _AttentionLayer(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        out = self.attn(x)
        x = self.norm1(x + self.dropout(out))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Memory module with gated update
# ---------------------------------------------------------------------------
class _MemoryModule(nn.Module):
    def __init__(self, n_memory: int, fea_dim: int, shrink_thres: float = 0.0):
        super().__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)
        # mem is a non-parameter buffer (carries gradients off; updated by gated rule)
        init = F.normalize(torch.rand(n_memory, fea_dim), dim=1)
        self.register_buffer("mem", init)

    def init_memory(self, mem: torch.Tensor):
        """Replace memory state (used after k-means initialization)."""
        with torch.no_grad():
            self.mem = mem.detach().to(self.mem.device)

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    @staticmethod
    def hard_shrink_relu(x, lambd=0.0025, eps=1e-12):
        return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + eps)

    def get_attn_score(self, query, key):
        attn = torch.matmul(query, torch.t(key))
        attn = F.softmax(attn, dim=-1)
        if self.shrink_thres > 0:
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            attn = F.normalize(attn, p=1, dim=1)
        return attn

    def read(self, query):
        attn = self.get_attn_score(query, self.mem.detach())
        add = torch.matmul(attn, self.mem.detach())
        return torch.cat([query, add], dim=1), attn

    def update(self, query):
        attn = self.get_attn_score(self.mem, query.detach())  # M x T
        add_mem = torch.matmul(attn, query.detach())
        gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem))
        with torch.no_grad():
            self.mem = (1 - gate) * self.mem + gate * add_mem

    def forward(self, query):
        s = query.shape
        l = len(s)
        q2 = query.contiguous().view(-1, s[-1])
        if not getattr(self, "_frozen", False):
            self.update(q2)
        out, attn = self.read(q2)
        if l == 3:
            out = out.view(s[0], s[1], 2 * s[2])
            attn = attn.view(s[0], s[1], self.n_memory)
        return out, attn


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class _Decoder(nn.Module):
    def __init__(self, d_in: int, c_out: int):
        super().__init__()
        self.out_linear = nn.Linear(d_in, c_out)

    def forward(self, x):
        return self.out_linear(x)


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------
@dataclass
class MemtoConfig:
    win_size: int
    enc_in: int
    c_out: int
    n_memory: int = 10
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.0


class MEMTO(nn.Module):
    def __init__(self, cfg: MemtoConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = _InputEmbedding(cfg.enc_in, cfg.d_model, dropout=cfg.dropout)
        self.encoder = _Encoder(
            [_EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff,
                           dropout=cfg.dropout, activation="gelu")
             for _ in range(cfg.e_layers)],
            norm_layer=nn.LayerNorm(cfg.d_model),
        )
        self.mem = _MemoryModule(cfg.n_memory, cfg.d_model)
        self.decoder = _Decoder(2 * cfg.d_model, cfg.c_out)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embedding(x))

    def forward(self, x: torch.Tensor) -> dict:
        x_emb = self.embedding(x)
        queries = self.encoder(x_emb)
        out, attn = self.mem(queries)
        recon = self.decoder(out)
        return {
            "out": recon,
            "queries": queries,
            "attn": attn,
            "mem": self.mem.mem,
        }

    @torch.no_grad()
    def kmeans_init_memory(self, encoder_features: torch.Tensor, seed: int = 42):
        """Initialize memory items via k-means on the flat encoder features."""
        feats = encoder_features.detach().cpu().numpy()
        if feats.ndim == 3:
            feats = feats.reshape(-1, feats.shape[-1])
        km = KMeans(n_clusters=self.cfg.n_memory, n_init=10, random_state=seed)
        km.fit(feats)
        centers = torch.from_numpy(km.cluster_centers_).float()
        self.mem.init_memory(centers)


# ---------------------------------------------------------------------------
# Loss helpers (gathering loss + entropy regularizer + reconstruction MSE)
# ---------------------------------------------------------------------------
def _gathering_loss(queries: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    """Mean squared distance from each query to its nearest memory item."""
    B, L, C = queries.shape
    q = queries.reshape(-1, C)
    score = F.softmax(q @ items.t(), dim=1)
    _, idx = torch.topk(score, 1, dim=1)
    nearest = items[idx.squeeze(1)]  # (B*L, C)
    return F.mse_loss(q, nearest)


def _entropy_loss(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (-attn * torch.log(attn + eps)).sum(dim=-1).mean()


def memto_loss(out_dict: dict, x_target: torch.Tensor,
               lambda_gather: float = 0.1, lambda_entropy: float = 0.01) -> torch.Tensor:
    recon_loss = F.mse_loss(out_dict["out"], x_target)
    gather = _gathering_loss(out_dict["queries"], out_dict["mem"])
    ent = _entropy_loss(out_dict["attn"])
    return recon_loss + lambda_gather * gather + lambda_entropy * ent


@torch.no_grad()
def memto_score(out_dict: dict, x_target: torch.Tensor) -> torch.Tensor:
    """Bi-dimensional anomaly score: per-step recon MSE * gathering distance."""
    recon_err = F.mse_loss(out_dict["out"], x_target, reduction="none").mean(dim=-1)  # (B, T)
    queries = out_dict["queries"]  # (B, T, C)
    items = out_dict["mem"]  # (M, C)
    B, T, C = queries.shape
    q = queries.reshape(-1, C)
    score = F.softmax(q @ items.t(), dim=1)
    _, idx = torch.topk(score, 1, dim=1)
    nearest = items[idx.squeeze(1)]
    gather = ((q - nearest) ** 2).sum(dim=-1).reshape(B, T)
    return recon_err * gather
