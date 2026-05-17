"""ModernTCN (ICLR 2024 Spotlight) - A Modern Pure Convolution Structure for
General Time Series Analysis.

Self-contained re-implementation of the ``ModernTCN-detection`` variant from
https://github.com/luodhhh/ModernTCN. The detection task is unsupervised
reconstruction: input ``(B, T, C)`` -> output ``(B, T, C)``; train on
normal-only windows with MSE; per-step squared error is the anomaly score.

Bug fixes vs the upstream:

* The unused ``LayerNorm`` helper class in upstream contained
  ``nn.Layernorm`` (typo). We omit it -- the architecture only uses
  ``nn.BatchNorm1d`` in practice.
* Removed broken ``PaddingTwoEdge1d`` (uses non-existent ``dims=`` kwarg of
  ``torch.cat``); we don't use re-parameterized inference here, only training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RevIN (matches upstream `layers/RevIN.py`)
# ---------------------------------------------------------------------------
class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            dim2 = tuple(range(1, x.ndim - 1))
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
            else:
                self.mean = torch.mean(x, dim=dim2, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2, keepdim=True, unbiased=False) + self.eps).detach()
            if self.subtract_last:
                x = x - self.last
            else:
                x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            return x + (self.last if self.subtract_last else self.mean)
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# Re-parameterizable large-kernel conv (no merge step at inference here).
# ---------------------------------------------------------------------------
def _conv_bn(in_c: int, out_c: int, kernel_size: int, stride: int, padding: Optional[int],
             groups: int, dilation: int = 1, bias: bool = False) -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, bias=bias),
        nn.BatchNorm1d(out_c),
    )


class _ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 groups: int, small_kernel: Optional[int]):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        self.lkb_origin = _conv_bn(in_channels, out_channels, kernel_size, stride,
                                   padding, groups=groups, dilation=1, bias=False)
        if small_kernel is not None:
            assert small_kernel <= kernel_size
            self.small_conv = _conv_bn(in_channels, out_channels, small_kernel, stride,
                                       padding=small_kernel // 2, groups=groups,
                                       dilation=1, bias=False)

    def forward(self, x):
        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out = out + self.small_conv(x)
        return out


# ---------------------------------------------------------------------------
# ModernTCN block, stage, backbone
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, large_size: int, small_size: int, dmodel: int, dff: int,
                 nvars: int, drop: float = 0.1):
        super().__init__()
        self.dw = _ReparamLargeKernelConv(
            in_channels=nvars * dmodel, out_channels=nvars * dmodel,
            kernel_size=large_size, stride=1, groups=nvars * dmodel,
            small_kernel=small_size,
        )
        self.norm = nn.BatchNorm1d(dmodel)
        # convffn1
        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        # convffn2
        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, M, D, N)
        residual = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N).reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)

        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)
        return residual + x


class _Stage(nn.Module):
    def __init__(self, ffn_ratio: int, num_blocks: int, large_size: int, small_size: int,
                 dmodel: int, nvars: int, drop: float = 0.1):
        super().__init__()
        d_ffn = dmodel * ffn_ratio
        self.blocks = nn.ModuleList([
            _Block(large_size, small_size, dmodel, d_ffn, nvars, drop)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Public model class (anomaly-detection task only)
# ---------------------------------------------------------------------------
@dataclass
class ModernTCNConfig:
    seq_len: int
    enc_in: int
    patch_size: int = 8
    patch_stride: int = 8
    stem_ratio: int = 6
    downsample_ratio: int = 2
    ffn_ratio: int = 1
    num_blocks: List[int] = field(default_factory=lambda: [1])
    large_size: List[int] = field(default_factory=lambda: [51])
    small_size: List[int] = field(default_factory=lambda: [5])
    dims: List[int] = field(default_factory=lambda: [64])
    dropout: float = 0.1
    revin: bool = True
    affine: bool = True


class ModernTCN(nn.Module):
    """Anomaly-detection ModernTCN (one-step reconstruction)."""

    def __init__(self, cfg: ModernTCNConfig):
        super().__init__()
        assert len(cfg.num_blocks) == len(cfg.large_size) == len(cfg.small_size) == len(cfg.dims), (
            "num_blocks/large_size/small_size/dims must all be the same length"
        )
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.revin = cfg.revin
        if self.revin:
            self.revin_layer = _RevIN(cfg.enc_in, affine=cfg.affine)

        # patch stem
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Linear(cfg.patch_size, cfg.dims[0]))
        for i in range(len(cfg.num_blocks) - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm1d(cfg.dims[i]),
                nn.Conv1d(cfg.dims[i], cfg.dims[i + 1],
                          kernel_size=cfg.downsample_ratio, stride=cfg.downsample_ratio),
            ))
        self.patch_size = cfg.patch_size
        self.patch_stride = cfg.patch_stride
        self.downsample_ratio = cfg.downsample_ratio

        self.num_stage = len(cfg.num_blocks)
        self.stages = nn.ModuleList([
            _Stage(cfg.ffn_ratio, cfg.num_blocks[i], cfg.large_size[i], cfg.small_size[i],
                   dmodel=cfg.dims[i], nvars=cfg.enc_in, drop=cfg.dropout)
            for i in range(self.num_stage)
        ])
        self.head_detect = nn.Linear(cfg.dims[-1], cfg.patch_size)

    def _forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L)
        B, M, L = x.shape
        x = x.unsqueeze(-2)  # (B, M, 1, L)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                x = x.reshape(B, M, 1, -1).squeeze(-2)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 1, 3, 2)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                x = self.downsample_layers[i](x)
                _, D_, N_ = x.shape
                x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.revin:
            x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self._forward_feature(x)
        # x: (B, M, D, N) -> (B, M, N, D) -> head -> (B, M, N, patch) -> (B, M, N*patch)
        x = x.permute(0, 1, 3, 2)
        x = self.head_detect(x)
        B, M, N, _ = x.shape
        x = x.reshape(B, M, -1)[:, :, : self.seq_len]
        x = x.permute(0, 2, 1)  # (B, T, C)
        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x
