"""Phase 2: bond type definitions and reference geometries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BondType(Enum):
    """Supported bond categories for physics terms."""

    PEPTIDE = "peptide"
    DISULFIDE = "disulfide"
    HYDROGEN = "hydrogen"
    OTHER = "other"


@dataclass(frozen=True)
class BondGeometry:
    """Reference geometry for a bond type."""

    ideal_length: float
    force_constant: float


@dataclass(frozen=True)
class AngleGeometry:
    """Reference geometry for a bond angle."""

    ideal_deg: float
    force_constant: float


BOND_GEOMETRIES: dict[BondType, BondGeometry] = {
    BondType.PEPTIDE: BondGeometry(ideal_length=1.33, force_constant=300.0),
    BondType.DISULFIDE: BondGeometry(ideal_length=2.05, force_constant=200.0),
}


def get_bond_geometry(bond_type: BondType) -> BondGeometry:
    """Return reference geometry for a bond type."""
    return BOND_GEOMETRIES[bond_type]
