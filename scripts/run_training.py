#!/usr/bin/env python3
"""CLI entry point for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protein PINN-GNN training")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--physics-config", default="configs/physics_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    args = parser.parse_args()

    train(
        model_config_path=args.model_config,
        physics_config_path=args.physics_config,
        training_config_path=args.training_config,
    )


if __name__ == "__main__":
    main()
