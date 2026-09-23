"""
encoder.py

Phase 1, step 1 (model): a deliberately simple GNN encoder for the
binding-site classification smoke test.

Uses stacked GCNConv layers with DISTANCE-WEIGHTED edges — NOT the
SE(3)-equivariant architecture planned for later phases. GCNConv
supports an optional per-edge scalar weight; we derive it from the
CA-CA distance already stored in edge_attr (inverse distance, so
closer / more physically meaningful contacts get more influence in
each residue's aggregated representation, rather than every edge
within the 8A cutoff being treated identically).

Swapping in a more expressive/equivariant architecture (EGNN, GAT,
IPA-style) is a later, isolated change once this baseline is proven
to work — keeping the two concerns (data correctness vs. architecture
quality) separate makes debugging much easier if something goes wrong.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BindingSiteGNN(nn.Module):
    """
    Per-residue binary classifier: binding-site residue or not.

    Input:  Data.x [n_nodes, 20] one-hot amino acid identity
            Data.edge_index [2, n_edges]
            Data.edge_attr [n_edges, 2] -> column 0 is CA-CA distance
             (Angstrom), column 1 is is_backbone (0/1) — see
             graph_builder.py. Only the distance column is used here;
             is_backbone is available for a future refinement but not
             consumed by this simple GCNConv-based model yet.
    Output: [n_nodes] raw logits (apply sigmoid externally, or use
             BCEWithLogitsLoss directly on these logits during training)
    """

    # Small constant added to distance before inverting, so a
    # (theoretically impossible but not worth crashing over)
    # zero-distance edge can't produce a divide-by-zero / inf weight.
    _EPS = 1e-2

    def __init__(
        self,
        in_channels: int = 20,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 2, "need at least an input and output layer"

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Final conv outputs 1 channel directly (raw logit per node) —
        # avoids a separate linear head for this simple first version.
        self.convs.append(GCNConv(hidden_channels, 1))

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_weight = None
        if edge_attr is not None:
            distances = edge_attr[:, 0]
            # Inverse distance: closer contacts get a LARGER weight
            # (more influence on the aggregated neighbor representation).
            # This is the opposite of using raw distance directly, which
            # would incorrectly give MORE influence to farther-apart,
            # weaker contacts.
            edge_weight = 1.0 / (distances + self._EPS)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            is_last = i == len(self.convs) - 1
            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.squeeze(-1)  # [n_nodes, 1] -> [n_nodes]
