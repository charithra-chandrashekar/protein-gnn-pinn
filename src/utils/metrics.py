"""Evaluation metrics."""

from __future__ import annotations

import torch


def rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Root-mean-square deviation between coordinate sets."""
    diff = pred - target
    sq = diff.pow(2).sum(dim=-1)
    if mask is not None:
        sq = sq * mask
        return torch.sqrt(sq.sum() / mask.sum().clamp_min(1.0)).item()
    return torch.sqrt(sq.mean()).item()


def decoy_ranking_score(native_energy: float, decoy_energies: list[float]) -> float:
    """Fraction of decoys with higher energy than native (higher is better)."""
    if not decoy_energies:
        return 0.0
    worse = sum(e > native_energy for e in decoy_energies)
    return worse / len(decoy_energies)
