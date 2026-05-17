"""TimesNet (ICLR 2023) - Temporal 2D-Variation Modeling for General Time
Series Analysis.

Self-contained re-implementation of the anomaly-detection branch from
``thuml/Time-Series-Library/models/TimesNet.py``. Trained as a one-step
reconstruction model on normal-only windows; the per-step squared error is
the anomaly score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embedding (TimesNet uses Conv1d-token + sinusoidal-position; no time features
# for the AD task, matching the official `enc_embedding(x, None)` call).
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
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = _TokenEmbedding(c_in, d_model)
        self.position_embedding = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


# ---------------------------------------------------------------------------
# Inception 2D block (from layers/Conv_Blocks.py::Inception_Block_V1)
# ---------------------------------------------------------------------------
class _InceptionBlockV1(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_kernels: int = 6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_c, out_c, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


# ---------------------------------------------------------------------------
# Period-aware 2D conv block (TimesBlock)
# ---------------------------------------------------------------------------
def _fft_for_period(x: torch.Tensor, k: int = 2):
    xf = torch.fft.rfft(x, dim=1)
    freq = abs(xf).mean(0).mean(-1)
    freq[0] = 0
    _, top_list = torch.topk(freq, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class _TimesBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, top_k: int,
                 d_model: int, d_ff: int, num_kernels: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            _InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            _InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        period_list, period_weight = _fft_for_period(x, self.k)
        res = []
        L = self.seq_len + self.pred_len
        for i in range(self.k):
            period = max(int(period_list[i]), 1)
            if L % period != 0:
                length = ((L // period) + 1) * period
                padding = torch.zeros(B, length - L, N, device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = L
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :L, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        return torch.sum(res * period_weight, -1) + x


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------
@dataclass
class TimesNetConfig:
    seq_len: int
    enc_in: int
    c_out: int
    d_model: int = 64
    d_ff: int = 64
    e_layers: int = 2
    top_k: int = 3
    num_kernels: int = 6
    dropout: float = 0.1


class TimesNet(nn.Module):
    """Anomaly-detection branch of TimesNet (reconstruction).

    Forward expects ``x`` of shape ``(B, T, C)`` and returns the reconstructed
    tensor of the same shape; train with MSE on normal-only data.
    """

    def __init__(self, cfg: TimesNetConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_embedding = _DataEmbedding(cfg.enc_in, cfg.d_model, cfg.dropout)
        self.model = nn.ModuleList([
            _TimesBlock(seq_len=cfg.seq_len, pred_len=0, top_k=cfg.top_k,
                        d_model=cfg.d_model, d_ff=cfg.d_ff, num_kernels=cfg.num_kernels)
            for _ in range(cfg.e_layers)
        ])
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.projection = nn.Linear(cfg.d_model, cfg.c_out, bias=True)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # Non-stationary normalization (per-window)
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        enc = self.enc_embedding(x)
        for block in self.model:
            enc = self.layer_norm(block(enc))
        dec = self.projection(enc)
        # de-normalize
        T = self.cfg.seq_len
        dec = dec * stdev[:, 0, :].unsqueeze(1).repeat(1, T, 1)
        dec = dec + means[:, 0, :].unsqueeze(1).repeat(1, T, 1)
        return dec
