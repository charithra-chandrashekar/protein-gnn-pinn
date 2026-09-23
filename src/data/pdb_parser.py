"""PDB → atom/residue records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class AtomRecord:
    """Single atom parsed from a PDB file."""

    atom_id: int
    name: str
    resname: str
    chain_id: str
    resseq: int
    x: float
    y: float
    z: float
    element: str


@dataclass
class ResidueRecord:
    """Residue grouping with constituent atoms."""

    chain_id: str
    resseq: int
    resname: str
    atoms: list[AtomRecord]


def parse_pdb(path: str | Path) -> list[ResidueRecord]:
    """Parse a PDB file into residue records."""
    raise NotImplementedError("PDB parsing not yet implemented")


def iter_atoms(records: list[ResidueRecord]) -> Iterator[AtomRecord]:
    """Yield all atoms from parsed residue records."""
    for residue in records:
        yield from residue.atoms
