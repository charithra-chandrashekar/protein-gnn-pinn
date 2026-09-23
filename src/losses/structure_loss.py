"""Data-fit loss (e.g., FAPE/lDDT-style)."""

from __future__ import annotations

import torch


def structure_loss(
    pred_pos: torch.Tensor,
    target_pos: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute structure reconstruction loss."""
    diff = pred_pos - target_pos
    per_atom = diff.pow(2).sum(dim=-1)
    if mask is not None:
        per_atom = per_atom * mask
        return per_atom.sum() / mask.sum().clamp_min(1.0)
    return per_atom.mean()


def fape_loss(
    pred_pos: torch.Tensor,
    target_pos: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Frame-aligned point error (placeholder)."""
    raise NotImplementedError("FAPE loss not yet implemented")
