"""Physiological-prior-masked GATv2 graph encoder for GlucoPrism."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class PriorMaskedGATv2Layer(nn.Module):
    """GATv2-style graph attention with a hard physiological prior mask.

    Input:
        x: [batch_size, num_nodes, input_dim]
        edge_bias: [num_nodes, num_nodes]
        edge_mask: [num_nodes, num_nodes], bool
    Output:
        out: [batch_size, num_nodes, output_dim]
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            while num_heads > 1 and output_dim % num_heads != 0:
                num_heads -= 1
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.src_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.dst_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.val_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_vector = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.edge_head_scale = nn.Parameter(torch.ones(num_heads))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.attn_vector)

    def forward(self, x: Tensor, edge_bias: Tensor, edge_mask: Tensor) -> Tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,N,F], got {tuple(x.shape)}")
        b, n, _ = x.shape
        if n != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: expected {self.num_nodes}, got {n}")

        src = self.src_proj(x).view(b, n, self.num_heads, self.head_dim)  # [B,N,H,D]
        dst = self.dst_proj(x).view(b, n, self.num_heads, self.head_dim)  # [B,N,H,D]
        val = self.val_proj(x).view(b, n, self.num_heads, self.head_dim)  # [B,N,H,D]

        pair = torch.tanh(src.unsqueeze(2) + dst.unsqueeze(1))  # [B,N,N,H,D]
        logits = (pair * self.attn_vector.view(1, 1, 1, self.num_heads, self.head_dim)).sum(-1)
        logits = logits.permute(0, 3, 1, 2)  # [B,H,N,N]

        bias = edge_bias.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        scale = self.edge_head_scale.to(dtype=x.dtype, device=x.device).view(1, self.num_heads, 1, 1)
        logits = logits + scale * bias

        mask = edge_mask.to(device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)  # [B,H,N,N]
        attn = self.dropout(attn)

        val = val.permute(0, 2, 1, 3)  # [B,H,N,D]
        out = torch.einsum("bhij,bhjd->bhid", attn, val)  # [B,H,N,D]
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.output_dim)  # [B,N,O]
        out = self.out_proj(out)
        out = self.norm(self.residual(x) + self.dropout(self.activation(out)))
        return out, attn.detach()


class PhysioGraphEncoder(nn.Module):
    """Prior-guided adaptive graph attention encoder.

    The adaptive mode combines a learnable prior weight with masked residual
    edges. The fixed-prior ablation uses A_prior directly as the edge bias.

    Input:
        x: [batch_size, num_nodes, feature_dim]
    Output:
        out: [batch_size, num_nodes, output_dim]
    """

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        node_embedding_dim: int = 16,
        adaptive_rank: int = 16,
        num_gcn_layers: int = 2,
        dropout: float = 0.1,
        A_prior: Optional[Tensor] = None,
        eps: float = 1e-6,
        gat_heads: int = 4,
        residual_edge_scale: float = 0.10,
        edge_dropout: float = 0.05,
        use_residual_graph: bool = True,
        use_edge_importance: bool = True,
        learn_prior_weight: bool = True,
    ) -> None:
        super().__init__()
        del node_embedding_dim, adaptive_rank
        output_dim = hidden_dim if output_dim is None else output_dim
        if num_nodes <= 0 or feature_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("num_nodes/feature_dim/hidden_dim/output_dim must be positive.")

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gcn_layers = num_gcn_layers
        self.eps = eps
        self.residual_edge_scale = residual_edge_scale
        self.use_residual_graph = use_residual_graph
        self.use_edge_importance = use_edge_importance
        self.learn_prior_weight = learn_prior_weight
        if not 0.0 <= edge_dropout < 1.0:
            raise ValueError("edge_dropout must be in [0, 1).")
        self.edge_dropout = edge_dropout

        prior = self._prepare_prior_adjacency(A_prior, num_nodes)
        identity = torch.eye(num_nodes, dtype=torch.float32)
        prior_mask = (prior > 0) | identity.bool()
        prior = torch.maximum(prior, identity)
        self.register_buffer("A_prior", prior)
        self.register_buffer("prior_mask", prior_mask)
        self.register_buffer("identity", identity)

        # Learnable residual edge weights are masked by physiological priors.
        self.edge_residual_logits = nn.Parameter(
            torch.full((num_nodes, num_nodes), -4.0),
            requires_grad=use_residual_graph,
        )
        self.edge_importance_logits = nn.Parameter(
            torch.zeros(num_nodes, num_nodes),
            requires_grad=use_residual_graph and use_edge_importance,
        )
        self.prior_weight_logit = nn.Parameter(
            torch.zeros(()),
            requires_grad=learn_prior_weight,
        )

        self.input_norm = nn.LayerNorm(feature_dim)
        dims = [feature_dim] + [hidden_dim] * (num_gcn_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [
                PriorMaskedGATv2Layer(
                    num_nodes=num_nodes,
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    num_heads=gat_heads,
                    dropout=dropout,
                )
                for i in range(num_gcn_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.last_attention: Optional[Tensor] = None
        self.last_node_features: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,N,F], got {tuple(x.shape)}")
        b, n, f = x.shape
        if n != self.num_nodes or f != self.feature_dim:
            raise ValueError(f"expected [B,{self.num_nodes},{self.feature_dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("Graph input contains NaN/Inf.")

        edge_bias, edge_mask = self.build_edge_bias(dtype=x.dtype, device=x.device, apply_dropout=True)
        out = self.input_norm(x)
        self.last_node_features = out
        attentions = []
        for layer in self.layers:
            out, attn = layer(out, edge_bias, edge_mask)
            attentions.append(attn)
        self.last_attention = torch.stack(attentions).mean(dim=0)  # [B,H,N,N]
        out = self.output_norm(out)
        assert out.shape == (b, self.num_nodes, self.output_dim)
        return out

    def build_edge_bias(
        self,
        dtype: torch.dtype,
        device: torch.device,
        apply_dropout: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        prior = self.A_prior.to(dtype=dtype, device=device)
        mask = self.prior_mask.to(device=device)
        if apply_dropout and self.training and self.edge_dropout > 0.0:
            identity = self.identity.to(device=device).bool()
            keep = torch.rand(self.num_nodes, self.num_nodes, device=device) >= self.edge_dropout
            mask = (mask & keep) | identity

        if self.learn_prior_weight:
            prior_weight = 0.5 + torch.sigmoid(
                self.prior_weight_logit.to(dtype=dtype, device=device)
            )
        else:
            prior_weight = prior.new_tensor(1.0)
        A_final = prior_weight * prior
        if self.use_residual_graph:
            residual = torch.nn.functional.softplus(
                self.edge_residual_logits.to(dtype=dtype, device=device)
            )
            if self.use_edge_importance:
                importance = 0.5 + torch.sigmoid(
                    self.edge_importance_logits.to(dtype=dtype, device=device)
                )
                residual = residual * importance
            residual = residual * mask.to(dtype=dtype) * self.residual_edge_scale
            A_final = A_final + residual
        edge_bias = torch.log(A_final.clamp_min(self.eps))
        return edge_bias, mask

    def build_adjacency(self, dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Tensor]:
        edge_bias, edge_mask = self.build_edge_bias(dtype=dtype, device=device, apply_dropout=False)
        A_final = torch.exp(edge_bias) * edge_mask.to(dtype=dtype, device=device)
        A_norm = A_final / A_final.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return A_final, A_norm

    def regularization_loss(self) -> Tensor:
        """L1 sparsity on learnable residual graph, masked by prior structure."""
        mask = self.prior_mask.to(
            device=self.edge_residual_logits.device,
            dtype=self.edge_residual_logits.dtype,
        )
        loss = self.edge_residual_logits.new_zeros(())
        if self.use_residual_graph:
            residual = torch.nn.functional.softplus(self.edge_residual_logits) * mask
            loss = loss + residual.mean()
            if self.use_edge_importance:
                importance = 0.5 + torch.sigmoid(self.edge_importance_logits)
                loss = loss + 0.10 * ((importance - 1.0).abs() * mask).mean()
        if self.learn_prior_weight:
            prior_weight = 0.5 + torch.sigmoid(self.prior_weight_logit)
            loss = loss + 0.10 * (prior_weight - 1.0).square()
        return loss

    def graph_smoothness_loss(self) -> Tensor:
        """Encourage prior-connected node representations to vary smoothly."""
        if self.last_node_features is None:
            return self.edge_residual_logits.new_tensor(0.0)
        features = self.last_node_features
        dtype = features.dtype
        device = features.device
        _, A_norm = self.build_adjacency(dtype=dtype, device=device)
        diff = features.unsqueeze(2) - features.unsqueeze(1)  # [B,N,N,F]
        pairwise_dist = diff.square().mean(dim=-1)  # [B,N,N]
        smoothness = (pairwise_dist * A_norm.unsqueeze(0)).sum(dim=(1, 2)).mean()
        return smoothness

    def prior_consistency_loss(self) -> Tensor:
        """Keep learned adjacency close to row-normalized physiological priors."""
        device = self.edge_residual_logits.device
        dtype = self.edge_residual_logits.dtype
        A_final, A_norm = self.build_adjacency(dtype=dtype, device=device)
        prior = self.A_prior.to(device=device, dtype=dtype)
        prior = prior * self.prior_mask.to(device=device, dtype=dtype)
        prior_norm = prior / prior.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        consistency = torch.nn.functional.smooth_l1_loss(A_norm, prior_norm, beta=0.05)
        sparsity = self.regularization_loss()
        entropy = -(A_norm * A_norm.clamp_min(self.eps).log()).sum(dim=-1).mean()
        smoothness = self.graph_smoothness_loss()
        return consistency + 0.05 * sparsity + 0.01 * entropy + 0.01 * smoothness

    @staticmethod
    def _prepare_prior_adjacency(A_prior: Optional[Tensor], num_nodes: int) -> Tensor:
        if A_prior is None:
            return torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        prior = torch.as_tensor(A_prior, dtype=torch.float32)
        if prior.shape != (num_nodes, num_nodes):
            raise ValueError(f"A_prior must be [{num_nodes},{num_nodes}], got {tuple(prior.shape)}")
        if (prior < 0).any() or not torch.isfinite(prior).all():
            raise ValueError("A_prior must be finite and non-negative.")
        prior = 0.5 * (prior + prior.t())
        prior.fill_diagonal_(1.0)
        return prior


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(4, 6, 32)
    prior = torch.ones(6, 6)
    encoder = PhysioGraphEncoder(6, 32, 64, output_dim=64, A_prior=prior)
    y = encoder(x)
    A, A_norm = encoder.build_adjacency(dtype=x.dtype, device=x.device)
    assert y.shape == (4, 6, 64)
    assert A.shape == (6, 6)
    assert A_norm.shape == (6, 6)
    print("PhysioGraphEncoder OK", tuple(y.shape))
