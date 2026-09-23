"""Inspect graphs/structures (Phase 0 sanity checks)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def plot_graph(pos: Any, edge_index: Any, title: str = "Protein Graph") -> None:
    """2D projection of graph nodes and edges."""
    raise NotImplementedError("Graph visualization not yet implemented")


def show_structure(pos: Any, title: str = "Structure") -> None:
    """Quick 3D scatter of atomic coordinates."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
    ax.set_title(title)
    plt.show()
