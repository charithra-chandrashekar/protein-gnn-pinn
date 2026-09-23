"""Phase 5: task-specific prediction heads."""

from __future__ import annotations

import torch
from torch import nn


class StructureHead(nn.Module):
    """Predict per-atom or per-residue coordinates."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_embeddings)


class EnergyHead(nn.Module):
    """Optional scalar energy regression head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(graph_embedding).squeeze(-1)
