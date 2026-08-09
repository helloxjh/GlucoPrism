"""Multi-horizon glucose prediction head for GlucoPrism.

Architecture: horizon-aware queries + temporal cross-attention + lightweight
long-context calibration + CGM residual prediction. The encoder and fusion
backbone remain shared, while each forecast horizon receives an explicit
time-distance condition and a small horizon-specific output head.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn


class MultiHorizonPredictionHead(nn.Module):
    """Multi-horizon prediction head with per-horizon learnable queries.

    Input:
        x: Tensor [batch_size, seq_len, input_dim] – fused temporal features.
        last_cgm: Tensor [batch_size, 1] – last observed CGM value (standardized).

    Output:
        pred: Tensor [batch_size, num_horizons] – predicted glucose values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
        num_attn_heads: Optional[int] = None,
        dropout: float = 0.1,
        enable_horizon_refinement: bool = False,
    ) -> None:
        super().__init__()
        self._validate(input_dim, hidden_dim, horizon_minutes, sampling_interval_minutes, dropout)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon_minutes = tuple(int(v) for v in horizon_minutes)
        self.sampling_interval_minutes = sampling_interval_minutes
        self.horizon_steps = tuple(h // sampling_interval_minutes for h in self.horizon_minutes)
        self.num_horizons = len(self.horizon_minutes)
        self.required_horizon_steps = max(self.horizon_steps)
        self.enable_horizon_refinement = bool(enable_horizon_refinement)
        normalized_horizons = torch.tensor(self.horizon_steps, dtype=torch.float32)
        normalized_horizons = normalized_horizons / normalized_horizons.max()
        self.register_buffer("normalized_horizons", normalized_horizons)

        # Per-horizon learnable queries [K, hidden_dim].
        self.horizon_queries = nn.Parameter(torch.zeros(self.num_horizons, hidden_dim))
        nn.init.xavier_uniform_(self.horizon_queries)

        # Cross-attention: horizon queries [B, K, H] attend to temporal features [B, T, H].
        if num_attn_heads is None:
            # Pick largest divisor <= 8 that divides hidden_dim.
            num_attn_heads = min(8, hidden_dim)
            while hidden_dim % num_attn_heads != 0:
                num_attn_heads -= 1
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Individual per-horizon MLP heads.
        # Each head transforms its horizon-specific context vector into a delta
        # from the last observed CGM value.
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ) for _ in range(self.num_horizons)
        ])

        # Extra horizon-aware refinement layers are initialized after the shared
        # original head components, so baseline/refined ablations share the same
        # common initialization under a fixed seed.
        refinement_dim = max(8, min(32, hidden_dim // 4))
        self.horizon_conditioner = nn.Sequential(
            nn.Linear(1, refinement_dim),
            nn.GELU(),
            nn.Linear(refinement_dim, hidden_dim),
        )
        self.long_context_refiner = nn.Sequential(
            nn.Linear(hidden_dim * 3, refinement_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(refinement_dim, hidden_dim),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.progressive_refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, refinement_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(refinement_dim, hidden_dim),
        )
        self.progressive_norm = nn.LayerNorm(hidden_dim)
        self.progressive_gate_logits = nn.Parameter(torch.full((self.num_horizons,), -2.0))
        self.refinement_dropout = nn.Dropout(dropout)
        self.last_horizon_attention: Optional[Tensor] = None

        # Residual refinements start from zero, preserving the original head at
        # initialization and reducing the risk of degrading the 15 min branch.
        self._zero_initialize_residual_refinements()

    def forward(self, x: Tensor, last_cgm: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [B, T, input_dim] fused temporal features.
            last_cgm: [B, 1] last observed CGM value in standardized space.

        Returns:
            pred: [B, num_horizons] predicted values (standardized).
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,F], got {tuple(x.shape)}")
        if last_cgm.ndim != 2 or last_cgm.shape[1] != 1:
            raise ValueError(f"last_cgm must be [B,1], got {tuple(last_cgm.shape)}")
        if x.shape[0] != last_cgm.shape[0]:
            raise ValueError("x and last_cgm must share batch dimension.")

        B, T, F = x.shape
        if F != self.input_dim:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {F}")

        # Project to hidden_dim and normalize.  [B, T, H]
        h = self.input_proj(x)
        h = self.input_norm(h)
        assert h.shape == (B, T, self.hidden_dim)

        # Explicit time-distance conditioning prevents the four learned queries
        # from becoming interchangeable. It can be disabled for graph_loss-style
        # ablations, where the original query-only head is recovered.
        horizon_position = self.normalized_horizons.to(
            device=h.device, dtype=h.dtype
        ).unsqueeze(-1)
        if self.enable_horizon_refinement:
            horizon_condition = self.horizon_conditioner(horizon_position)
            query_seed = self.horizon_queries + horizon_condition
        else:
            horizon_condition = torch.zeros_like(self.horizon_queries)
            query_seed = self.horizon_queries
        queries = query_seed.unsqueeze(0).expand(B, -1, -1)
        assert queries.shape == (B, self.num_horizons, self.hidden_dim)

        # Cross-attention: each horizon query independently attends to the
        # full temporal sequence, extracting relevant context for its horizon.
        attended, attention_weights = self.cross_attn(
            queries,
            h,
            h,
            need_weights=True,
            average_attn_weights=True,
        )  # attended [B,K,H], weights [B,K,T]
        self.last_horizon_attention = attention_weights.detach()
        attended = self.attn_norm(attended + queries)  # residual around attention
        assert attended.shape == (B, self.num_horizons, self.hidden_dim)

        if self.enable_horizon_refinement:
            # Parameter-efficient long-context calibration. The global mean captures
            # the shared historical state; the feature-space linear trend receives
            # stronger weight as the prediction horizon increases.
            global_context = h.mean(dim=1, keepdim=True).expand(-1, self.num_horizons, -1)
            time_axis = torch.linspace(-1.0, 1.0, T, device=h.device, dtype=h.dtype)
            trend_denominator = time_axis.square().sum().clamp_min(1.0)
            trend_context = torch.sum(h * time_axis.view(1, T, 1), dim=1) / trend_denominator
            trend_context = trend_context.unsqueeze(1).expand(-1, self.num_horizons, -1)
            horizon_scale = horizon_position.view(1, self.num_horizons, 1)
            refinement_input = torch.cat(
                [attended, global_context, horizon_scale * trend_context], dim=-1
            )
            long_refinement = self.context_norm(self.long_context_refiner(refinement_input))
            calibration_gate = torch.sigmoid(horizon_condition).unsqueeze(0)
            calibrated = (
                attended
                + self.refinement_dropout(horizon_scale * calibration_gate * long_refinement)
            )
            assert calibrated.shape == (B, self.num_horizons, self.hidden_dim)

            # Reuse the preceding horizon representation without recursively using
            # its scalar prediction. This introduces ordered horizon evolution while
            # avoiding autoregressive error accumulation.
            progressive_contexts: list[Tensor] = [calibrated[:, 0, :]]
            for i in range(1, self.num_horizons):
                current = calibrated[:, i, :]
                previous = progressive_contexts[-1]
                transition = self.progressive_norm(
                    self.progressive_refiner(torch.cat([current, previous], dim=-1))
                )
                gate = torch.sigmoid(self.progressive_gate_logits[i]) * horizon_scale[:, i, :]
                progressive_contexts.append(
                    current + self.refinement_dropout(gate * transition)
                )
            horizon_features = torch.stack(progressive_contexts, dim=1)  # [B,K,H]
        else:
            horizon_features = attended
        assert horizon_features.shape == (B, self.num_horizons, self.hidden_dim)

        # Each horizon head predicts a delta from last_cgm.
        deltas: list[Tensor] = []
        for i, head in enumerate(self.horizon_heads):
            delta_i = head(horizon_features[:, i, :])  # [B, 1]
            assert delta_i.shape == (B, 1)
            deltas.append(delta_i)

        delta = torch.cat(deltas, dim=-1)  # [B, K]
        assert delta.shape == (B, self.num_horizons)

        # Residual anchor: final prediction = last observed CGM + learned delta.
        pred = last_cgm + delta  # [B, K]
        assert pred.shape == (B, self.num_horizons)
        return pred

    @torch.no_grad()
    def diagnostic_metrics(self) -> Dict[str, float]:
        """Measure horizon separation without affecting model optimization."""
        horizon_position = self.normalized_horizons.unsqueeze(-1)
        if self.enable_horizon_refinement:
            conditioned_queries = self.horizon_queries.detach() + self.horizon_conditioner(
                horizon_position
            ).detach()
        else:
            conditioned_queries = self.horizon_queries.detach()
        query_similarity = self._mean_off_diagonal_cosine(conditioned_queries)
        metrics = {
            "horizon_query_similarity": float(query_similarity.cpu()),
            "horizon_progressive_gate": float(
                torch.sigmoid(self.progressive_gate_logits[1:]).mean().cpu()
            )
            if self.num_horizons > 1
            else 0.0,
        }
        if self.last_horizon_attention is not None:
            attention = self.last_horizon_attention.float().mean(dim=0)  # [K,T]
            metrics["horizon_attention_similarity"] = float(
                self._mean_off_diagonal_cosine(attention).cpu()
            )
            entropy = -(
                attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
            ).sum(dim=-1)
            max_entropy = torch.log(
                attention.new_tensor(float(max(2, attention.shape[-1])))
            )
            metrics["horizon_attention_entropy"] = float(
                (entropy / max_entropy).mean().cpu()
            )
        return metrics

    @staticmethod
    def _mean_off_diagonal_cosine(features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be [K,D], got {tuple(features.shape)}")
        if features.shape[0] < 2:
            return features.new_tensor(0.0)
        normalized = torch.nn.functional.normalize(features, dim=-1, eps=1e-8)
        similarity = normalized @ normalized.transpose(0, 1)
        mask = ~torch.eye(
            features.shape[0], device=features.device, dtype=torch.bool
        )
        return similarity[mask].mean()

    def _zero_initialize_residual_refinements(self) -> None:
        """Make the added horizon refinements identity-like at initialization."""
        for module in (
            self.horizon_conditioner[-1],
            self.long_context_refiner[-1],
            self.progressive_refiner[-1],
        ):
            if not isinstance(module, nn.Linear):
                raise TypeError("Residual refinement output must be an nn.Linear layer.")
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def select_targets(self, future_glucose: Tensor) -> Tensor:
        """Extract supervision labels from full future glucose sequence.

        Args:
            future_glucose: [B, future_steps] – must have at least required_horizon_steps.

        Returns:
            [B, num_horizons] – target values for each prediction horizon.
        """
        if future_glucose.ndim != 2:
            raise ValueError(f"future_glucose must be [B,S], got {tuple(future_glucose.shape)}")
        B, S = future_glucose.shape
        if S < self.required_horizon_steps:
            raise ValueError(
                f"future_glucose needs {self.required_horizon_steps} steps, got {S}"
            )
        # horizon_steps are 1-based (step 3 = index 2).
        indices = torch.tensor(
            [s - 1 for s in self.horizon_steps],
            device=future_glucose.device,
            dtype=torch.long,
        )
        return future_glucose.index_select(dim=1, index=indices)

    @staticmethod
    def _validate(
        input_dim: int,
        hidden_dim: int,
        horizon_minutes: Sequence[int],
        sampling_interval_minutes: int,
        dropout: float,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not horizon_minutes:
            raise ValueError("horizon_minutes must not be empty")
        if any(h <= 0 for h in horizon_minutes):
            raise ValueError("horizon_minutes must all be positive")
        if any(h % sampling_interval_minutes != 0 for h in horizon_minutes):
            raise ValueError("horizons must be multiples of sampling_interval_minutes")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, F, H = 4, 24, 64, 64
    head = MultiHorizonPredictionHead(input_dim=F, hidden_dim=H, dropout=0.1)
    x = torch.randn(B, T, F)
    last = torch.randn(B, 1)
    out = head(x, last)
    assert out.shape == (B, 4), f"Expected (4,4), got {tuple(out.shape)}"
    print(f"OK – output shape: {tuple(out.shape)}")
    print(f"  horizon_minutes: {head.horizon_minutes}")
    print(f"  horizon_steps:   {head.horizon_steps}")
    print(f"  required_steps:  {head.required_horizon_steps}")

    # Verify select_targets
    future = torch.randn(B, 12)
    targets = head.select_targets(future)
    assert targets.shape == (B, 4)
    # Verify indices: step 3->idx 2, step 6->idx 5, step 9->idx 8, step 12->idx 11
    assert torch.allclose(targets[:, 0], future[:, 2])
    assert torch.allclose(targets[:, 3], future[:, 11])
    print("  select_targets: OK")
