"""Metrics and validation."""

from __future__ import annotations

from typing import Any

import torch


def evaluate(model: torch.nn.Module, dataloader: Any, device: torch.device) -> dict[str, float]:
    """Run validation and return metric dictionary."""
    model.eval()
    metrics: dict[str, float] = {}
    raise NotImplementedError("Evaluation not yet implemented")
    return metrics
