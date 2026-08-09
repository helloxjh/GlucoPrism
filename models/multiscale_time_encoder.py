"""TCN/Transformer hybrid multi-scale temporal encoder for GlucoPrism."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


class MultiResolutionPositionalEncoding(nn.Module):
    """Multi-resolution sinusoidal + learnable positional encoding.

    Input/Output:
        x: [batch_size, seq_len, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        max_len: int = 512,
        periods: Sequence[float] = (8.0, 24.0, 72.0),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.periods = tuple(float(p) for p in periods)
        self.learnable_pe = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.learnable_pe, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,H], got {tuple(x.shape)}")
        b, t, h = x.shape
        if h != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: expected {self.hidden_dim}, got {h}")
        if t > self.learnable_pe.shape[1]:
            raise ValueError(f"seq_len {t} exceeds max_len {self.learnable_pe.shape[1]}")

        pos = torch.arange(t, device=x.device, dtype=x.dtype).view(1, t, 1)
        freqs = torch.arange(0, h, 2, device=x.device, dtype=x.dtype)
        freqs = torch.exp(freqs * (-math.log(10000.0) / h)).view(1, 1, -1)
        pe = torch.zeros(1, t, h, device=x.device, dtype=x.dtype)
        pe[..., 0::2] = torch.sin(pos * freqs)
        pe[..., 1::2] = torch.cos(pos * freqs[..., : pe[..., 1::2].shape[-1]])

        # Extra periodic components encode short/medium/long temporal rhythms.
        for period in self.periods:
            phase = 2.0 * math.pi * pos / period
            pe[..., 0::2] = pe[..., 0::2] + torch.sin(phase)
            pe[..., 1::2] = pe[..., 1::2] + torch.cos(phase[..., : pe[..., 1::2].shape[-1]])
        pe = pe / (len(self.periods) + 1.0)

        out = x + pe + self.learnable_pe[:, :t, :].to(dtype=x.dtype, device=x.device)
        return self.dropout(out)


class TemporalConvBlock(nn.Module):
    """Residual dilated temporal convolution block.

    Input/Output:
        x: [batch_size, seq_len, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.out_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, h = x.shape
        residual = x
        y = self.norm(x).transpose(1, 2)  # [B,H,T]
        y = self.depthwise(y)
        y = self.pointwise(y)
        value, gate = y.chunk(2, dim=1)
        y = value * torch.sigmoid(gate)
        y = self.out_proj(y).transpose(1, 2)  # [B,T,H]
        y = self.dropout(y)
        assert y.shape == (b, t, h)
        return residual + y


class MultiScaleTimeEncoder(nn.Module):
    """Hybrid temporal encoder: multi-resolution PE + TCN + Transformer.

    The old implementation applied LayerNorm over feature_dim=1, which erased
    scalar CGM/physio values. This version preserves scalar amplitudes and adds
    both local TCN receptive fields and global self-attention.

    Input:
        x: [batch_size, seq_len, feature_dim]
    Output:
        out: [batch_size, seq_len, hidden_dim]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        short_kernel_size: int = 3,
        long_kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        transformer_layers: int = 2,
        transformer_ff_multiplier: int = 4,
        dropout: float = 0.1,
        temporal_mode: str = "multi_scale",
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not dilations:
            raise ValueError("dilations must not be empty.")
        if temporal_mode not in {"multi_scale", "single_scale"}:
            raise ValueError("temporal_mode must be 'multi_scale' or 'single_scale'.")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.temporal_mode = temporal_mode

        # Critical correction: LayerNorm(1) collapses every scalar time step to
        # zero. Use Identity for scalar inputs and LayerNorm otherwise.
        self.input_norm = nn.LayerNorm(feature_dim) if feature_dim > 1 else nn.Identity()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.position_encoding = MultiResolutionPositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * transformer_ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        if temporal_mode == "multi_scale":
            self.short_tcn: nn.Module | None = nn.Sequential(
                TemporalConvBlock(hidden_dim, short_kernel_size, 1, dropout),
                TemporalConvBlock(hidden_dim, short_kernel_size, 2, dropout),
            )
            self.long_tcn: nn.Module | None = nn.Sequential(
                *[
                    TemporalConvBlock(hidden_dim, long_kernel_size, int(dilation), dropout)
                    for dilation in dilations
                ]
            )
            self.branch_gate: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            refine_dilation = max(1, min(4, int(max(dilations))))
            self.temporal_refinement: nn.Module | None = TemporalConvBlock(
                hidden_dim=hidden_dim,
                kernel_size=3,
                dilation=refine_dilation,
                dropout=dropout,
            )
            self.reuse_gate: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid(),
            )
            self.output_dropout: nn.Module | None = nn.Dropout(dropout)
        else:
            self.short_tcn = None
            self.long_tcn = None
            self.branch_gate = None
            self.temporal_refinement = None
            self.reuse_gate = None
            self.output_dropout = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,F], got {tuple(x.shape)}")
        b, t, f = x.shape
        if f != self.feature_dim:
            raise ValueError(f"feature_dim mismatch: expected {self.feature_dim}, got {f}")
        if not torch.isfinite(x).all():
            raise ValueError("Temporal encoder input contains NaN/Inf.")

        x = self.input_norm(x)
        h = self.input_projection(x)  # [B,T,H]
        h = self.position_encoding(h)  # [B,T,H]

        if self.temporal_mode == "single_scale":
            out = self.output_norm(self.global_transformer(h))
            assert out.shape == (b, t, self.hidden_dim)
            return out

        assert self.short_tcn is not None
        assert self.long_tcn is not None
        assert self.branch_gate is not None
        assert self.temporal_refinement is not None
        assert self.reuse_gate is not None
        assert self.output_dropout is not None
        short = self.short_tcn(h)  # [B,T,H]
        long = self.long_tcn(h)  # [B,T,H]
        global_ctx = self.global_transformer(h)  # [B,T,H]

        stacked = torch.stack([short, long, global_ctx], dim=-2)  # [B,T,3,H]
        gate_input = torch.cat([short, long, global_ctx], dim=-1)  # [B,T,3H]
        gates = torch.softmax(self.branch_gate(gate_input), dim=-1)  # [B,T,3]
        multi_scale = torch.sum(stacked * gates.unsqueeze(-1), dim=-2)  # [B,T,H]

        # Lightweight temporal refinement keeps the dual-branch structure intact
        # while correcting long-horizon drift through a residual dilated filter.
        refined = self.temporal_refinement(multi_scale)  # [B,T,H]
        reuse = self.reuse_gate(torch.cat([h, multi_scale, refined], dim=-1))  # [B,T,H]
        out = reuse * refined + (1.0 - reuse) * multi_scale  # [B,T,H]
        out = self.output_norm(h + self.output_dropout(out))
        assert out.shape == (b, t, self.hidden_dim)
        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(4, 24, 3)
    encoder = MultiScaleTimeEncoder(feature_dim=3, hidden_dim=32, num_heads=4)
    y = encoder(x)
    assert y.shape == (4, 24, 32)
    print("MultiScaleTimeEncoder OK", tuple(y.shape))
