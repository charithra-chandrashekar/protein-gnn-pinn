"""Phase 4: wraps energy_module as regularizer."""

from __future__ import annotations

from typing import Any

from src.physics.energy_module import EnergyModule


def physics_loss(
    energy_module: EnergyModule,
    pos: Any,
    edge_index: Any,
    node_features: Any | None = None,
    bond_types: Any | None = None,
) -> Any:
    """Physics regularizer based on predicted structure energy."""
    breakdown = energy_module(
        pos=pos,
        edge_index=edge_index,
        node_features=node_features,
        bond_types=bond_types,
    )
    return breakdown["total"]
