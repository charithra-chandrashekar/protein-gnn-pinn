"""
taxonomy_loader.py

Loads configs/bond_taxonomy.yaml into a plain dict, with a couple of
convenience accessors for the specific numeric parameters energy_terms.py
needs. Deliberately thin — this file's only job is "get taxonomy data
from disk into Python", not any physics logic itself.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = PROJECT_ROOT / "configs" / "bond_taxonomy.yaml"


def load_taxonomy(path: Path | None = None) -> dict:
    path = path or TAXONOMY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"bond_taxonomy.yaml not found at {path}. This file is required "
            f"by src/physics/energy_terms.py — see Phase 2 of the project."
        )
    with open(path) as f:
        return yaml.safe_load(f)
