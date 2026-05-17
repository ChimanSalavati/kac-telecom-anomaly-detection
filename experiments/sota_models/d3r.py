"""D3R (NeurIPS 2023) - Drift doesn't Matter: Dynamic Decomposition with
Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly
Detection.

Self-contained re-implementation of https://github.com/ForestsKing/D3R . The
model jointly learns

* a *dynamic decomposition* head that separates ``stable`` and ``trend``
  components using data-time mix-attention;
* a *diffusion reconstruction* head that reconstructs the (data + disturbance)
  signal conditioned on the trend and a small diffusion-noise step.

At training, three reconstruction signals are matched: the stable component to
the original window minus learned trend, the trend to the leftover after
``OffsetSubtraction``, and the diffusion reconstruction to the input. At test,
the per-step reconstruction error of the diffusion head is the anomaly score.

Differences from upstream:
* Single-file packaging.
* Time covariates default to a learned positional vector when none are
  provided; the original repo built sin/cos hour/minute features but we adapt
  to telecom KPI windows where time-of-day is not always available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class _PositionEmbedding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float()
                             * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        return self.norm(self.pe[:, : x.size(1), :])


class _DataEmbedding(nn.Module):
    def __init__(self, model_dim: int, feature_num: int):
        super().__init__()
        self.conv = nn.Conv1d(feature_num, model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class _TimeEmbedding(nn.Module):
    def __init__(self, model_dim: int, time_num: int):
        super().__init__()
        self.conv = nn.Conv1d(time_num, model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        return self.norm(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))


# ---------------------------------------------------------------------------
# Attention + transformer blocks
# ---------------------------------------------------------------------------
class _OrdAttention(nn.Module):
    def __init__(self, model_dim: int, atten_dim: int, head_num: int, dropout: float, residual: bool):
        super().__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual
        self.W_Q = nn.Linear(model_dim, atten_dim * head_num)
        self.W_K = nn.Linear(model_dim, atten_dim * head_num)
        self.W_V = nn.Linear(model_dim, atten_dim * head_num)
        self.fc = nn.Linear(atten_dim * head_num, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V):
        residual = Q.clone()
        B, L, _ = Q.shape
        Q = self.W_Q(Q).view(B, L, self.head_num, self.atten_dim).transpose(1, 2)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).reshape(residual.size(0), residual.size(1), -1)
        out = self.dropout(self.fc(ctx))
        return self.norm(out + residual) if self.residual else self.norm(out)


class _MixAttention(nn.Module):
    def __init__(self, model_dim: int, atten_dim: int, head_num: int, dropout: float, residual: bool):
        super().__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual
        self.W_Q_data = nn.Linear(model_dim, atten_dim * head_num)
        self.W_Q_time = nn.Linear(model_dim, atten_dim * head_num)
        self.W_K_data = nn.Linear(model_dim, atten_dim * head_num)
        self.W_K_time = nn.Linear(model_dim, atten_dim * head_num)
        self.W_V_time = nn.Linear(model_dim, atten_dim * head_num)
        self.fc = nn.Linear(atten_dim * head_num, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Qd, Qt, Kd, Kt, Vt):
        residual = Qd.clone()
        Qd = self.W_Q_data(Qd).view(Qd.size(0), Qd.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        Qt = self.W_Q_time(Qt).view(Qt.size(0), Qt.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        Kd = self.W_K_data(Kd).view(Kd.size(0), Kd.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        Kt = self.W_K_time(Kt).view(Kt.size(0), Kt.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        Vt = self.W_V_time(Vt).view(Vt.size(0), Vt.size(1), self.head_num, self.atten_dim).transpose(1, 2)
        sd = torch.matmul(Qd, Kd.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        st = torch.matmul(Qt, Kt.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = F.softmax(sd + st, dim=-1)
        ctx = torch.matmul(attn, Vt).transpose(1, 2).reshape(residual.size(0), residual.size(1), -1)
        out = self.dropout(self.fc(ctx))
        return self.norm(out + residual) if self.residual else self.norm(out)


class _TemporalTransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super().__init__()
        self.attention = _OrdAttention(model_dim, atten_dim, head_num, dropout, residual=True)
        self.conv1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.attention(x, x, x)
        residual = x.clone()
        x = F.gelu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(x + residual)


class _SpatialTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super().__init__()
        self.attention = _OrdAttention(window_size, atten_dim, head_num, dropout, residual=True)
        self.conv1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attention(x, x, x)
        x = x.permute(0, 2, 1)
        residual = x.clone()
        x = F.gelu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(x + residual)


class _SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super().__init__()
        self.time_block = _TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.feature_block = _SpatialTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)
        self.conv1 = nn.Conv1d(2 * model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        time_x = self.time_block(x)
        feature_x = self.feature_block(x)
        x = self.norm1(torch.cat([time_x, feature_x], dim=-1))
        x = F.gelu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm2(x)


class _DecompositionBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, feature_num, head_num, dropout):
        super().__init__()
        self.mixed = _MixAttention(model_dim, atten_dim, head_num, dropout, residual=False)
        self.ord = _OrdAttention(model_dim, atten_dim, head_num, dropout, residual=True)
        self.conv1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.conv2 = nn.Conv1d(ff_dim, model_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, feature_num)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, trend, time):
        stable = self.mixed(trend, time, trend, time, time)
        stable = self.ord(stable, stable, stable)
        residual = stable.clone()
        stable = F.gelu(self.conv1(stable.permute(0, 2, 1)))
        stable = self.dropout(self.conv2(stable).permute(0, 2, 1))
        stable = self.norm1(stable + residual)
        trend = self.norm2(trend - stable)
        stable = self.fc2(F.gelu(self.fc1(stable)))
        return stable, trend


class _OffsetSubtraction(nn.Module):
    def __init__(self, window_size: int, feature_num: int, d: int):
        super().__init__()
        init_index = (torch.arange(window_size) + window_size).unsqueeze(-1).unsqueeze(-1)
        init_index = init_index.repeat(1, feature_num, 2 * d + 1)
        delay = torch.tensor([0] + list(range(1, d + 1)) + [-i for i in range(1, d + 1)],
                             dtype=torch.int64)
        delay = delay.unsqueeze(0).unsqueeze(0).repeat(window_size, feature_num, 1)
        self.register_buffer("index", init_index + delay)
        self.d = d

    def forward(self, subed, sub):
        B = subed.shape[0]
        index = self.index.unsqueeze(0).repeat(B, 1, 1, 1).to(sub.device)
        front = sub[:, 0:1, :].repeat(1, sub.shape[1], 1)
        end = sub[:, -1:, :].repeat(1, sub.shape[1], 1)
        sub = torch.cat([front, sub, end], dim=1)
        sub = torch.gather(sub.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1), dim=1, index=index)
        res = subed.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1) - sub
        res = torch.gather(res, dim=-1, index=torch.argmin(torch.abs(res), dim=-1).unsqueeze(-1))
        return res.reshape(subed.shape)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class _DataEncoder(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, block_num, head_num, dropout):
        super().__init__()
        self.data_emb = _DataEmbedding(model_dim, feature_num)
        self.pos_emb = _PositionEmbedding(model_dim)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0.0 if i == block_num - 1 else dropout
            self.blocks.append(_SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp))

    def forward(self, x):
        x = self.data_emb(x) + self.pos_emb(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _TimeEncoder(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, time_num, block_num, head_num, dropout):
        super().__init__()
        self.time_emb = _TimeEmbedding(model_dim, time_num)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0.0 if i == block_num - 1 else dropout
            self.blocks.append(_TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dp))

    def forward(self, x):
        x = self.time_emb(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _DynamicDecomposition(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num,
                 block_num, head_num, dropout, d):
        super().__init__()
        self.data_encoder = _DataEncoder(window_size, model_dim, ff_dim, atten_dim, feature_num,
                                         block_num, head_num, dropout)
        self.time_encoder = _TimeEncoder(model_dim, ff_dim, atten_dim, time_num, block_num,
                                         head_num, dropout)
        self.blocks = nn.ModuleList([
            _DecompositionBlock(model_dim, ff_dim, atten_dim, feature_num, head_num,
                                dropout if i < block_num - 1 else 0.0)
            for i in range(block_num)
        ])
        self.minus = _OffsetSubtraction(window_size, feature_num, d)

    def forward(self, data, time):
        residual = data.clone()
        data_e = self.data_encoder(data)
        time_e = self.time_encoder(time)
        stable = torch.zeros_like(residual).to(data_e.device)
        for blk in self.blocks:
            tmp_stable, data_e = blk(data_e, time_e)
            stable = stable + tmp_stable
        trend = torch.mean(self.minus(residual, stable), dim=1).unsqueeze(1).repeat(1, data_e.shape[1], 1)
        return stable, trend


class _Diffusion:
    def __init__(self, time_steps: int, beta_start: float, beta_end: float, device):
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = 1.0 - torch.sqrt(self.alphas_cumprod)

    @staticmethod
    def _extract(data, t, shape):
        out = torch.gather(data, -1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, trend, t, noise):
        a = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        c = self._extract(self.one_minus_sqrt_alphas_cumprod, t, x_start.shape)
        return a * x_start + b * noise + c * trend


class _Reconstruction(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num,
                 block_num, head_num, dropout):
        super().__init__()
        self.time_emb = _TimeEmbedding(model_dim, time_num)
        self.data_emb = _DataEmbedding(model_dim, feature_num)
        self.pos_emb = _PositionEmbedding(model_dim)
        self.blocks = nn.ModuleList([
            _SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num,
                                             dropout if i < block_num - 1 else 0.0)
            for i in range(block_num)
        ])
        self.fc1 = nn.Linear(model_dim, feature_num)

    def forward(self, noise, trend, time):
        trend_e = self.data_emb(trend)
        x = self.data_emb(noise) - trend_e + self.pos_emb(noise) + self.time_emb(time)
        for blk in self.blocks:
            x = blk(x)
        return self.fc1(x + trend_e)


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------
@dataclass
class D3RConfig:
    window_size: int
    feature_num: int
    time_num: int = 4
    model_dim: int = 128
    ff_dim: int = 128
    atten_dim: int = 16
    block_num: int = 3
    head_num: int = 4
    dropout: float = 0.1
    time_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    t: int = 200
    d: int = 5


class D3R(nn.Module):
    """Drift-robust D3R for multivariate window reconstruction."""

    def __init__(self, cfg: D3RConfig):
        super().__init__()
        self.cfg = cfg
        self.window_size = cfg.window_size
        self.t = cfg.t
        self.dynamic_decomposition = _DynamicDecomposition(
            window_size=cfg.window_size, model_dim=cfg.model_dim, ff_dim=cfg.ff_dim,
            atten_dim=cfg.atten_dim, feature_num=cfg.feature_num, time_num=cfg.time_num,
            block_num=cfg.block_num, head_num=cfg.head_num, dropout=cfg.dropout, d=cfg.d,
        )
        self.reconstruction = _Reconstruction(
            window_size=cfg.window_size, model_dim=cfg.model_dim, ff_dim=cfg.ff_dim,
            atten_dim=cfg.atten_dim, feature_num=cfg.feature_num, time_num=cfg.time_num,
            block_num=cfg.block_num, head_num=cfg.head_num, dropout=cfg.dropout,
        )
        # diffusion is rebuilt on the right device on each forward
        self._diffusion = None

    def _ensure_diffusion(self, device):
        if self._diffusion is None or self._diffusion.betas.device != device:
            self._diffusion = _Diffusion(self.cfg.time_steps, self.cfg.beta_start,
                                         self.cfg.beta_end, device)
        return self._diffusion

    def forward(self, data: torch.Tensor, time: torch.Tensor, p: float = 0.0):
        device = data.device
        diff = self._ensure_diffusion(device)
        # additive disturbance per (batch, feature)
        if p > 0:
            disturb = (torch.rand(data.shape[0], data.shape[2], device=device) * p)
            disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1).float()
            data = data + disturb
        else:
            disturb = torch.zeros_like(data)
        stable, trend = self.dynamic_decomposition(data, time)
        bt = torch.full((data.shape[0],), self.t, device=device, dtype=torch.long)
        sample_noise = torch.randn_like(data).float()
        noise_data = diff.q_sample(data, trend, bt, sample_noise)
        recon = self.reconstruction(noise_data, trend, time)
        return stable, trend - disturb, recon - disturb
