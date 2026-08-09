"""Bidirectional co-attention fusion with modality confidence gating."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional co-attention + anti-collapse modality gating.

    Inputs:
        f_cgm: [batch_size, seq_len_a, dim_a]
        f_physio: [batch_size, seq_len_b, dim_b]
    Output:
        fused: [batch_size, seq_len_a, embed_dim]
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        min_modality_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if not 0.0 <= min_modality_weight < 1.0 / 3.0:
            raise ValueError("min_modality_weight must be in [0, 1/3).")

        self.dim_a = dim_a
        self.dim_b = dim_b
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.min_modality_weight = min_modality_weight

        self.cgm_projection = nn.Linear(dim_a, embed_dim)
        self.physio_projection = nn.Linear(dim_b, embed_dim)
        self.cgm_norm = nn.LayerNorm(embed_dim)
        self.physio_norm = nn.LayerNorm(embed_dim)

        self.cgm_to_physio = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.physio_to_cgm = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.align_physio_to_cgm_axis = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.cgm_guided_norm = nn.LayerNorm(embed_dim)
        self.physio_guided_norm = nn.LayerNorm(embed_dim)
        self.axis_norm = nn.LayerNorm(embed_dim)

        self.confidence_network = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )
        self.dropout = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_multiplier, embed_dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.last_modality_weights: Optional[Tensor] = None

    def forward(
        self,
        f_cgm: Tensor,
        f_physio: Tensor,
        cgm_key_padding_mask: Optional[Tensor] = None,
        physio_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        self._check_inputs(f_cgm, f_physio, cgm_key_padding_mask, physio_key_padding_mask)
        b, ta, _ = f_cgm.shape
        _, tb, _ = f_physio.shape

        cgm = self.cgm_norm(self.cgm_projection(f_cgm))  # [B,Ta,E]
        physio = self.physio_norm(self.physio_projection(f_physio))  # [B,Tb,E]

        cgm_guided, attn_c2p = self.cgm_to_physio(
            query=cgm,
            key=physio,
            value=physio,
            key_padding_mask=physio_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        cgm_guided = self.cgm_guided_norm(cgm + self.dropout(cgm_guided))  # [B,Ta,E]

        physio_guided, attn_p2c = self.physio_to_cgm(
            query=physio,
            key=cgm,
            value=cgm,
            key_padding_mask=cgm_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        physio_guided = self.physio_guided_norm(physio + self.dropout(physio_guided))  # [B,Tb,E]

        physio_aligned, _ = self.align_physio_to_cgm_axis(
            query=cgm,
            key=physio_guided,
            value=physio_guided,
            key_padding_mask=physio_key_padding_mask,
            need_weights=False,
        )
        physio_aligned = self.axis_norm(cgm + self.dropout(physio_aligned))  # [B,Ta,E]

        # Confidence gating uses absolute disagreement and multiplicative agreement.
        diff_cgm_physio = torch.abs(cgm - physio_aligned)
        diff_guided = torch.abs(cgm_guided - physio_aligned)
        agreement = cgm_guided * physio_aligned
        confidence_input = torch.cat(
            [cgm, cgm_guided, physio_aligned, diff_cgm_physio, agreement + diff_guided],
            dim=-1,
        )  # [B,Ta,5E]
        logits = self.confidence_network(confidence_input)  # [B,Ta,3]
        probs = torch.softmax(logits, dim=-1)
        min_w = self.min_modality_weight
        gates = min_w + (1.0 - 3.0 * min_w) * probs  # each modality keeps non-zero mass
        self.last_modality_weights = gates.detach()

        fused = (
            gates[..., 0:1] * cgm
            + gates[..., 1:2] * cgm_guided
            + gates[..., 2:3] * physio_aligned
        )
        fused = self.fusion_norm(fused)
        out = self.out_norm(fused + self.ffn(fused))
        assert out.shape == (b, ta, self.embed_dim)
        if not torch.isfinite(out).all():
            raise ValueError("Fusion output contains NaN/Inf.")

        if need_weights:
            if attn_c2p is None:
                attn_c2p = torch.empty(b, ta, tb, device=f_cgm.device)
            if attn_p2c is None:
                attn_p2c = torch.empty(b, tb, ta, device=f_cgm.device)
            return out, attn_c2p, attn_p2c
        return out

    def modality_balance_loss(self) -> Tensor:
        """Entropy-based anti-collapse regularization for fusion gates."""
        if self.last_modality_weights is None:
            return torch.tensor(0.0)
        weights = self.last_modality_weights
        entropy = -(weights * weights.clamp_min(1e-6).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(3.0, device=weights.device, dtype=weights.dtype))
        return max_entropy - entropy

    def _check_inputs(
        self,
        f_cgm: Tensor,
        f_physio: Tensor,
        cgm_key_padding_mask: Optional[Tensor],
        physio_key_padding_mask: Optional[Tensor],
    ) -> None:
        if f_cgm.ndim != 3:
            raise ValueError(f"f_cgm must be [B,Ta,Da], got {tuple(f_cgm.shape)}")
        if f_physio.ndim != 3:
            raise ValueError(f"f_physio must be [B,Tb,Db], got {tuple(f_physio.shape)}")
        b, ta, da = f_cgm.shape
        bp, tb, db = f_physio.shape
        if b != bp or da != self.dim_a or db != self.dim_b:
            raise ValueError("fusion input shape mismatch.")
        if cgm_key_padding_mask is not None and cgm_key_padding_mask.shape != (b, ta):
            raise ValueError("invalid cgm_key_padding_mask shape.")
        if physio_key_padding_mask is not None and physio_key_padding_mask.shape != (b, tb):
            raise ValueError("invalid physio_key_padding_mask shape.")


if __name__ == "__main__":
    torch.manual_seed(42)
    module = BidirectionalCrossAttention(64, 64, 64, num_heads=4)
    a = torch.randn(4, 24, 64)
    b = torch.randn(4, 6, 64)
    y = module(a, b)
    assert y.shape == (4, 24, 64)
    print("BidirectionalCrossAttention OK", tuple(y.shape))
