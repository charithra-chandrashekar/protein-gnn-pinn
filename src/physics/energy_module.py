"""
energy_module.py

Phase 3, step 2: combines all 5 terms from energy_terms.py into a
single total_energy(structure) function, given a parsed protein
structure (from src/data/cif_parser.py's ParsedStructure/ResidueRecord
objects).

This module's job is purely ASSEMBLY — extracting the right atom-coordinate
subsets for each energy term (backbone donor/acceptor atoms, charged
side-chain atoms, nonpolar side-chain atoms) from a ParsedStructure, and
summing the 5 resulting scalar energies. All the actual physics lives in
energy_terms.py; this file has no energy formulas of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.physics import energy_terms as et
from src.physics.taxonomy_loader import load_taxonomy


@dataclass
class EnergyBreakdown:
    """Per-term energy values, so a caller can inspect which term(s)
    drive a structure's total energy rather than only seeing one
    opaque scalar — useful for debugging and for the decoy-ranking
    validation in this same module."""
    covalent: float
    hydrogen_bond: float
    van_der_waals: float
    electrostatic: float
    hydrophobic: float
    total: float


def _extract_backbone_bond_coords(residues: list) -> torch.Tensor:
    """[n_residues, 2, 3]: [C coord, N coord] per residue, for the
    covalent peptide-bond term. Residues missing either atom are
    skipped (returns a shorter tensor) rather than raising, consistent
    with this project's "skip and report, don't crash" pattern —
    though note the caller does not currently print a skip count; a
    silent gap here would only matter for structures with missing
    backbone atoms, which verify_and_build_index.py's checks already
    guard against upstream."""
    coords = []
    for r in residues:
        c = r.atom_coords.get("C")
        n = r.atom_coords.get("N")
        if c is not None and n is not None:
            coords.append([list(c), list(n)])
    if not coords:
        return torch.empty((0, 2, 3))
    return torch.tensor(coords, dtype=torch.float32)


def _extract_hbond_donor_acceptor_coords(residues: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Backbone N (donor proxy) and O (acceptor proxy) coordinates,
    per the v1 scoping decision in energy_terms.py's module docstring
    (side-chain donors/acceptors not modeled in this version)."""
    donors, acceptors = [], []
    for r in residues:
        n = r.atom_coords.get("N")
        o = r.atom_coords.get("O")
        if n is not None:
            donors.append(list(n))
        if o is not None:
            acceptors.append(list(o))
    donor_t = torch.tensor(donors, dtype=torch.float32) if donors else torch.empty((0, 3))
    acceptor_t = torch.tensor(acceptors, dtype=torch.float32) if acceptors else torch.empty((0, 3))
    return donor_t, acceptor_t


def _extract_all_heavy_atom_coords(residues: list) -> torch.Tensor:
    """Every atom coordinate across all residues, flattened, for the
    van der Waals term (applies between all atom pairs per the
    taxonomy)."""
    coords = []
    for r in residues:
        for coord in r.atom_coords.values():
            coords.append(list(coord))
    if not coords:
        return torch.empty((0, 3))
    return torch.tensor(coords, dtype=torch.float32)


def _extract_charged_atom_coords(residues: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Formally-charged side-chain atom coordinates + matching charge
    signs, per et.CHARGED_ATOMS."""
    coords, charges = [], []
    for r in residues:
        charged_map = et.CHARGED_ATOMS.get(r.res_name)
        if not charged_map:
            continue
        for atom_name, charge in charged_map.items():
            coord = r.atom_coords.get(atom_name)
            if coord is not None:
                coords.append(list(coord))
                charges.append(float(charge))
    coords_t = torch.tensor(coords, dtype=torch.float32) if coords else torch.empty((0, 3))
    charges_t = torch.tensor(charges, dtype=torch.float32) if charges else torch.empty((0,))
    return coords_t, charges_t


def _extract_hydrophobic_atom_coords(residues: list) -> torch.Tensor:
    """Nonpolar side-chain atom coordinates, per et.HYDROPHOBIC_PROXY_ATOMS."""
    coords = []
    for r in residues:
        atom_names = et.HYDROPHOBIC_PROXY_ATOMS.get(r.res_name)
        if not atom_names:
            continue
        for atom_name in atom_names:
            coord = r.atom_coords.get(atom_name)
            if coord is not None:
                coords.append(list(coord))
    if not coords:
        return torch.empty((0, 3))
    return torch.tensor(coords, dtype=torch.float32)


# Default per-term weights applied before summing into the total.
# All terms are weighted 1.0 (full trust) EXCEPT hydrophobic, which is
# down-weighted per the investigation in PHASE_3_ADDENDUM.md: the
# hydrophobic pairwise-contact proxy fails the near-native decoy
# ranking test (unlike the other 4 terms, which pass reliably), so its
# contribution is capped low here rather than removed entirely — it
# still carries some real signal (it passes the easier scrambled-decoy
# test), just not enough to be trusted at full weight in a coupled
# loss (Phase 4). Override via the term_weights argument if a specific
# use case needs different values.
DEFAULT_TERM_WEIGHTS = {
    "covalent": 1.0,
    "hydrogen_bond": 1.0,
    "van_der_waals": 1.0,
    "electrostatic": 1.0,
    "hydrophobic": 0.1,  # down-weighted — see PHASE_3_ADDENDUM.md
}


def total_energy(
    residues: list,
    taxonomy: dict | None = None,
    term_weights: dict | None = None,
) -> EnergyBreakdown:
    """
    Computes total energy for a list of ResidueRecord objects (e.g.
    ParsedStructure.residues from cif_parser.py), summing all 5
    WEIGHTED terms.

    Args:
        residues: list of ResidueRecord (must have .res_name and
            .atom_coords populated — .ca_coord alone is not sufficient
            since several terms need specific named atoms).
        taxonomy: pre-loaded bond_taxonomy.yaml dict; loaded fresh if
            not provided (allows callers to load once and reuse across
            many structures without re-reading the file each time).
        term_weights: dict mapping term name -> weight, applied to
            each term BEFORE summing into the total. Defaults to
            DEFAULT_TERM_WEIGHTS (all 1.0 except hydrophobic at 0.1 —
            see module-level comment and PHASE_3_ADDENDUM.md for why).
            EnergyBreakdown's per-term fields report the RAW (unweighted)
            values regardless of term_weights, so the breakdown always
            reflects the true underlying energies — only `total` is
            affected by weighting. Pass an explicit dict to override
            (e.g. {**DEFAULT_TERM_WEIGHTS, "hydrophobic": 0.0} to
            exclude it entirely, or 1.0 to trust it fully).
        taxonomy: pre-loaded bond_taxonomy.yaml dict; loaded fresh if
            not provided (allows callers to load once and reuse across
            many structures without re-reading the file each time).

    Returns:
        EnergyBreakdown with per-term values and the summed total.
        LOWER total = more physically stable, following standard
        convention — this is the property the decoy-ranking test
        below (rank_real_vs_decoy) checks.
    """
    if taxonomy is None:
        taxonomy = load_taxonomy()
    if term_weights is None:
        term_weights = DEFAULT_TERM_WEIGHTS

    backbone_bond_coords = _extract_backbone_bond_coords(residues)
    donor_coords, acceptor_coords = _extract_hbond_donor_acceptor_coords(residues)
    all_atom_coords = _extract_all_heavy_atom_coords(residues)
    charged_coords, charges = _extract_charged_atom_coords(residues)
    hydrophobic_coords = _extract_hydrophobic_atom_coords(residues)

    e_covalent = et.covalent_bond_energy(backbone_bond_coords, taxonomy)
    e_hbond = et.hydrogen_bond_energy(donor_coords, acceptor_coords, taxonomy)
    e_vdw = et.van_der_waals_energy(all_atom_coords, taxonomy)
    e_elec = et.electrostatic_energy(charged_coords, charges, taxonomy)
    e_hydrophobic = et.hydrophobic_energy(hydrophobic_coords, taxonomy)

    # total uses WEIGHTED terms; the returned per-term fields below
    # remain RAW/unweighted, so a caller inspecting the breakdown sees
    # true values, not values silently scaled by the mitigation weight.
    total = (
        term_weights.get("covalent", 1.0) * e_covalent
        + term_weights.get("hydrogen_bond", 1.0) * e_hbond
        + term_weights.get("van_der_waals", 1.0) * e_vdw
        + term_weights.get("electrostatic", 1.0) * e_elec
        + term_weights.get("hydrophobic", 1.0) * e_hydrophobic
    )

    return EnergyBreakdown(
        covalent=e_covalent.item(),
        hydrogen_bond=e_hbond.item(),
        van_der_waals=e_vdw.item(),
        electrostatic=e_elec.item(),
        hydrophobic=e_hydrophobic.item(),
        total=total.item(),
    )
