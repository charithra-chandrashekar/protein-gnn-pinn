"""
energy_terms.py

Phase 3, step 1: differentiable energy functions for each interaction
type in configs/bond_taxonomy.yaml, implemented as PyTorch operations
so gradients can flow from a scalar energy back to atomic coordinates
(required for Phase 4, where this becomes a loss term).

Each function takes coordinate tensors and returns a SCALAR energy
(lower = more stable, following standard physics convention — this
matters for the decoy-ranking test in energy_module.py: a real
structure should have LOWER total energy than a scrambled one).

Scoping decisions made explicit here (see conversation for full
rationale — these are simplifications relative to a real force field,
not hidden shortcuts):
  - hydrogen_bond: approximated via backbone N (donor) / O (acceptor)
    atoms only — catches the dominant secondary-structure H-bonds, but
    does NOT model side-chain donor/acceptor atoms (e.g. Ser/Thr -OH,
    Asn/Gln amide, His imidazole). A real per-residue donor/acceptor
    atom map is a natural v2 addition.
  - van_der_waals: uses ONE generic sigma/epsilon (carbon-like) for
    ALL atom pairs, not a real per-element AMBER-style table. This is
    a real simplification — van der Waals contact strength genuinely
    varies by atom type — acceptable for a first-pass ranking test,
    flagged for replacement with real force-field parameters later.
  - electrostatic: only applied between formally charged side-chain
    atoms (Asp/Glu carboxylate O, Lys ammonium N, Arg guanidinium N),
    not all atoms.
  - hydrophobic: uses the explicitly-flagged PAIRWISE PROXY from the
    taxonomy file (nonpolar side-chain atoms in contact), not a real
    solvent-accessible-surface-area calculation.
"""

from __future__ import annotations

import torch

from src.physics.taxonomy_loader import load_taxonomy

# --- Atom-type classification, used across several terms below ---

BACKBONE_DONOR_ATOMS = {"N"}    # backbone amide N (H-bond donor proxy)
BACKBONE_ACCEPTOR_ATOMS = {"O"}  # backbone carbonyl O (H-bond acceptor proxy)

# Formally charged side-chain atoms, by residue name -> atom name -> charge sign.
# Deliberately a small, explicit table (not "every atom in a charged
# residue") — only the atom(s) actually carrying the formal charge.
CHARGED_ATOMS = {
    "ASP": {"OD1": -1, "OD2": -1},
    "GLU": {"OE1": -1, "OE2": -1},
    "LYS": {"NZ": +1},
    "ARG": {"NH1": +1, "NH2": +1},
}

# Nonpolar side-chain atoms used as the hydrophobic pairwise-contact
# proxy (see module docstring) — one representative carbon atom per
# hydrophobic residue type, not an exhaustive list of every nonpolar
# atom, to keep the v1 proxy simple and fast.
HYDROPHOBIC_PROXY_ATOMS = {
    "ALA": ["CB"], "VAL": ["CB", "CG1", "CG2"], "LEU": ["CB", "CG", "CD1", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"], "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "PRO": ["CB", "CG", "CD"],
}


def _pairwise_distances(coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
    """coords_a: [N, 3], coords_b: [M, 3] -> [N, M] distance matrix.
    Kept as a shared helper so every energy term uses the same
    (differentiable) distance computation."""
    diff = coords_a.unsqueeze(1) - coords_b.unsqueeze(0)  # [N, M, 3]
    return torch.norm(diff, dim=-1)


def covalent_bond_energy(
    backbone_coords: torch.Tensor,
    taxonomy: dict,
    force_constant: float = 300.0,
) -> torch.Tensor:
    """
    Harmonic penalty on consecutive-residue peptide bond length
    (C_i to N_i+1 distance) deviating from equilibrium.

    Args:
        backbone_coords: [n_residues, 2, 3] — for each residue,
            [C atom coord, N atom coord] (the two atoms forming the
            peptide bond TO THE NEXT residue; last residue's C is
            unused since there's no i+1).
        taxonomy: loaded bond_taxonomy.yaml dict
        force_constant: k in E = k*(r-r0)^2. NOT specified numerically
            in bond_taxonomy.yaml (flagged there as force-field-
            dependent) — 300 kcal/mol/A^2 is a commonly-used AMBER-
            range default for a C-N bond, used here as a reasonable
            default since the taxonomy deliberately left this open.

    Returns:
        Scalar tensor (sum over all consecutive-residue bonds).
    """
    r0 = taxonomy["covalent_peptide_bond"]["energy_scale"]["parameters"]["r0_angstrom"]

    if backbone_coords.shape[0] < 2:
        return torch.tensor(0.0)

    c_coords = backbone_coords[:-1, 0, :]   # C of residue i
    n_coords = backbone_coords[1:, 1, :]    # N of residue i+1
    bond_lengths = torch.norm(c_coords - n_coords, dim=-1)

    energy = force_constant * (bond_lengths - r0) ** 2
    return energy.sum()


def hydrogen_bond_energy(
    donor_coords: torch.Tensor,
    acceptor_coords: torch.Tensor,
    taxonomy: dict,
    cutoff: float = 4.0,
) -> torch.Tensor:
    """
    Mayo-style radial hydrogen bond potential (distance term only —
    angular term explicitly deferred, see bond_taxonomy.yaml note)
    between all donor-acceptor pairs within `cutoff` Angstrom.

    Args:
        donor_coords: [n_donors, 3] (backbone N atoms)
        acceptor_coords: [n_acceptors, 3] (backbone O atoms)
        taxonomy: loaded bond_taxonomy.yaml dict
        cutoff: distance beyond which a pair is not considered
            (avoids summing a near-zero contribution over every
            possible pair in a large structure)

    Returns:
        Scalar tensor (sum over all donor-acceptor pairs within cutoff).
    """
    params = taxonomy["hydrogen_bond"]["energy_scale"]["parameters"]
    R0 = params["R0_angstrom"]
    D0 = params["D0_kcal_per_mol"]

    if donor_coords.shape[0] == 0 or acceptor_coords.shape[0] == 0:
        return torch.tensor(0.0)

    dist = _pairwise_distances(donor_coords, acceptor_coords)  # [n_donors, n_acceptors]
    mask = (dist <= cutoff) & (dist > 0.1)  # exclude self/degenerate zero-distance pairs

    # Mayo radial term: E = D0 * (5*(R0/R)^12 - 6*(R0/R)^10)
    # Distance is clamped to a safe floor BEFORE computing the ratio —
    # same rationale as van_der_waals_energy: an unclamped near-zero
    # distance can produce inf/nan in the 12th-power term, which survives
    # multiplication by a zero mask (0 * nan = nan in PyTorch), silently
    # corrupting the sum even for pairs meant to be excluded.
    safe_dist = torch.clamp(dist, min=0.5)
    ratio = R0 / safe_dist
    radial_term = D0 * (5 * ratio**12 - 6 * ratio**10)
    energy = radial_term * mask.float()
    return energy.sum()


def van_der_waals_energy(
    all_coords: torch.Tensor,
    taxonomy: dict,
    sigma: float = 3.5,
    epsilon: float = 0.1,
    cutoff: float = 6.0,
    exclude_self_pairs: int = 1,
    min_nonbonded_distance: float = 2.0,
) -> torch.Tensor:
    """
    Lennard-Jones 12-6 potential between all NON-BONDED atom pairs
    within cutoff.

    Args:
        all_coords: [n_atoms, 3] — ALL heavy atoms in the structure
            (this term applies between every non-bonded atom pair, per
            the taxonomy's description — not residue-type-specific).
        taxonomy: loaded bond_taxonomy.yaml dict (present for API
            consistency with the other terms; sigma/epsilon are NOT
            read from it since the taxonomy explicitly flags these as
            atom-type-pair-dependent / force-field-sourced, not a
            single constant — see module docstring scoping note).
        sigma: generic distance parameter (Angstrom), carbon-like default.
        epsilon: generic well-depth parameter (kcal/mol), carbon-like default.
        cutoff: distance beyond which a pair is ignored.
        exclude_self_pairs: minimum atom-index separation to include
            (default 1 excludes only literal self-pairs).
        min_nonbonded_distance: pairs closer than this (Angstrom) are
            EXCLUDED from this term entirely. Real covalent bond
            lengths (e.g. C-C ~1.5A, C=O ~1.2A) and 1-2/1-3 bonded
            neighbors fall well under 2.0A; without this exclusion, the
            generic non-bonded LJ parameters (calibrated for genuine
            non-bonded contacts) produce an enormous, unphysical
            repulsive spike at real bonded distances — found directly
            when validating this function against a real structure
            (EGFR/1M17): a single directly-bonded atom pair at 1.21A
              contributed +138,506 to the total energy on its own,
            dwarfing every other physically meaningful contribution in
            the structure by several orders of magnitude. This mirrors
            standard force-field practice of excluding 1-2/1-3 bonded
            neighbors from non-bonded interaction terms — those
            interactions are already handled by the covalent bond term
            (and, for a full force field, angle/torsion terms not yet
            in this v1 taxonomy scope).

    Returns:
        Scalar tensor.
    """
    n = all_coords.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    dist = _pairwise_distances(all_coords, all_coords)  # [n, n]

    # Mask: within cutoff, not a self-pair, NOT closer than the
    # bonded-neighbor exclusion distance, upper triangle only (avoid
    # double-counting each pair and avoid the zero-distance diagonal
    # blowing up the 1/r^12 term).
    upper_tri = torch.triu(torch.ones(n, n), diagonal=exclude_self_pairs).bool()
    mask = (dist <= cutoff) & (dist >= min_nonbonded_distance) & upper_tri

    # IMPORTANT: clamp distance to a safe floor BEFORE computing ratio
    # powers, not just mask afterward. At self-pairs (distance=0), an
    # unclamped 1/r^12 term produces `inf`, and inf**2 - inf = nan —
    # multiplying that nan by a mask of 0 afterward STILL yields nan in
    # PyTorch (0 * nan = nan, not 0), silently poisoning the sum and any
    # gradient through it. Clamping avoids ever computing inf/nan in the
    # first place, regardless of what the mask later zeroes out.
    safe_dist = torch.clamp(dist, min=0.5)
    ratio6 = (sigma / safe_dist) ** 6
    lj = 4 * epsilon * (ratio6**2 - ratio6)
    energy = lj * mask.float()
    return energy.sum()


def electrostatic_energy(
    charged_atom_coords: torch.Tensor,
    charges: torch.Tensor,
    taxonomy: dict,
    dielectric: float = 10.0,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """
    Coulomb potential with a fixed effective dielectric constant,
    between all formally-charged atom pairs within cutoff.

    Args:
        charged_atom_coords: [n_charged, 3]
        charges: [n_charged] tensor of +1/-1 values matching
            charged_atom_coords row-for-row
        taxonomy: loaded bond_taxonomy.yaml dict (present for API
            consistency; dielectric is NOT read from it since the
            taxonomy explicitly flags epsilon_r as model-dependent,
            not a fixed constant)
        dielectric: effective screened dielectric constant. 10.0 is a
            commonly-used mid-range value for a protein-interior-ish
            environment (taxonomy notes protein interior ~4-20,
            solvent-exposed approaches ~80) — a genuine simplification
            versus a real distance-dependent or Debye-Huckel model.
        cutoff: distance beyond which a pair is ignored.

    Returns:
        Scalar tensor. Coulomb's constant is folded into a single
        kcal/mol-Angstrom-appropriate prefactor (332.0), the standard
        conversion factor used in biomolecular force fields for
        charges in elementary-charge units and distances in Angstrom.
    """
    COULOMB_CONSTANT_KCAL = 332.0  # standard biomolecular-units Coulomb prefactor

    n = charged_atom_coords.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    dist = _pairwise_distances(charged_atom_coords, charged_atom_coords)
    charge_products = charges.unsqueeze(1) * charges.unsqueeze(0)  # [n, n]

    upper_tri = torch.triu(torch.ones(n, n), diagonal=1).bool()
    mask = (dist <= cutoff) & upper_tri

    # Same distance-clamping rationale as the other terms: avoid
    # dividing by a near-zero self-pair distance before masking, since
    # a resulting inf/nan survives multiplication by a zero mask.
    safe_dist = torch.clamp(dist, min=0.5)
    energy_matrix = (COULOMB_CONSTANT_KCAL * charge_products) / (dielectric * safe_dist)
    energy = energy_matrix * mask.float()
    return energy.sum()


def hydrophobic_energy(
    nonpolar_coords: torch.Tensor,
    taxonomy: dict,
    cutoff: float = 4.5,
    saturation_density: float = 8.0,
) -> torch.Tensor:
    """
    Pairwise-contact PROXY for hydrophobic burial (see module
    docstring: this is explicitly NOT a real solvent-accessible-
    surface-area calculation, per the simplification flagged in
    bond_taxonomy.yaml itself). Applies a smooth, distance-dependent
    stabilizing well to nonpolar side-chain atom pairs within contact
    range.

    NOTE ON FUNCTIONAL FORM: an earlier version of this function used
    a flat per-contact energy bonus (-E if within cutoff, 0 otherwise).
    That is intuitive but NOT differentiable with respect to atom
    coordinates — a boolean mask times a constant carries no gradient
    back to the positions that produced it, which silently breaks the
    "differentiable everywhere" requirement this whole module exists
    for (see module docstring; this matters for Phase 4, where this
    needs to backpropagate into a GNN's coordinate predictions). This
    version instead uses a smooth cosine switching function that
    ramps the stabilizing energy from full strength at close range
    down to zero at the cutoff, which is both continuous AND
    differentiable — while still being a simplified proxy, not a real
    SASA-based calculation, exactly as before.

    Args:
        nonpolar_coords: [n_atoms, 3] — nonpolar side-chain atoms
            (see HYDROPHOBIC_PROXY_ATOMS for which atoms qualify)
        taxonomy: loaded bond_taxonomy.yaml dict
        cutoff: contact distance (Angstrom) beyond which the
            stabilizing contribution smoothly reaches zero.
        saturation_density: the local-density value (see below) at
            which an atom is considered "fully buried" — density
            beyond this point contributes only marginally more
            stabilization, rather than growing without bound. Chosen
            as a reasonable middle-of-the-pack neighbor count for a
            genuinely buried atom in a folded protein core; not
            precision-tuned against a real dataset, same "starting
            point, not ground truth" caveat as bond_taxonomy.yaml's
            own numbers.

    Returns:
        Scalar tensor (negative = stabilizing, following convention).

    REFORMULATION NOTE (supersedes the pairwise-contact-count version):
    The original version summed a smooth per-PAIR bonus for every atom
    pair within cutoff — energy scaled with raw contact COUNT, with no
    upper limit. This was found to be a genuine bug, not just an
    approximation: validating against NEAR-NATIVE decoys (small
    Gaussian coordinate noise, not full scrambling) showed the term
    failing to rank the real structure as more stable in 0/5 trials,
    consistently, at every noise level tested (0.5-3.0A) — because
    independent per-residue noise has more opportunities to nudge
    near-cutoff atom pairs INTO contact range than to push already-
    close pairs apart, so random noise systematically produces MORE
    raw contacts than the real, compact fold (confirmed directly:
    real EGFR structure had 1,343 contact pairs; every tested noise
    level produced 1,349-1,465, i.e. more, in 5/5 trials each).

    This version instead computes, per atom, a smooth LOCAL DENSITY
    (how many nonpolar neighbors it has, weighted by proximity) and
    applies a SATURATING response (tanh) before summing — so an atom
    that's already deeply buried (high local density) gains very
    little additional "credit" from one more marginal neighbor,
    which is much closer to how real burial works (an atom is either
    buried or it isn't) and removes the specific unbounded-accumulation
    mechanism that made the previous version noise-sensitive.
    """
    per_contact_energy = taxonomy["hydrophobic"]["energy_scale"][
        "typical_single_interaction_anchor_kcal_per_mol"
    ]

    n = nonpolar_coords.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    dist = _pairwise_distances(nonpolar_coords, nonpolar_coords)
    self_mask = ~torch.eye(n, dtype=torch.bool)

    # Smooth cosine switching function per PAIR: 1.0 at distance=0,
    # ramps down to 0.0 at distance=cutoff. Same continuous/
    # differentiable weighting as before, but now used to build a
    # PER-ATOM local density rather than summed directly as energy.
    safe_dist = torch.clamp(dist, min=0.0, max=cutoff)
    pair_weight = 0.5 * (1.0 + torch.cos(torch.pi * safe_dist / cutoff))
    within_cutoff = dist <= cutoff
    pair_weight = pair_weight * (within_cutoff & self_mask).float()

    # Per-atom local density: sum of proximity-weighted neighbor
    # contributions (each atom's row sums its weighted contacts with
    # every other atom — self excluded via self_mask above).
    local_density = pair_weight.sum(dim=1)  # [n]

    # Saturating response: tanh grows ~linearly for small density
    # (few neighbors -> proportionate credit, same intuition as the
    # old version for a lightly-packed atom) but flattens out well
    # before density becomes large, so an already-buried atom cannot
    # keep gaining stabilization from ever more marginal contacts —
    # this is the mechanism that removes the noise-sensitivity found
    # during near-native decoy testing.
    saturated = torch.tanh(local_density / saturation_density)

    # Total energy: negative (stabilizing) contribution proportional
    # to each atom's saturated burial level, summed over atoms, scaled
    # by the taxonomy's per-contact anchor value and halved (since
    # local_density double-counts each pair from both atoms' rows).
    energy = -0.5 * per_contact_energy * saturated.sum()
    return energy
