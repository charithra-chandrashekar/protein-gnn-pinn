"""
validate_energy_ranking.py

Phase 3, step 3: the actual validation bar for this energy module,
per the original Phase 3 plan — does it correctly rank a REAL protein
structure as more stable (lower energy) than a SCRAMBLED/DECOY version
of the same structure? If not, this energy function has no business
being coupled into a GNN's loss (Phase 4) — it would inject noise, not
a useful physical prior.

Decoy generation strategy: given a real structure's residues, generate
a decoy by RANDOMLY PERMUTING each residue's CA-centered atom cluster
to a different residue's CA position, preserving each residue's
internal (local) atomic geometry but destroying the real tertiary
fold. This is a deliberately "obviously wrong" decoy — a proper decoy
benchmark (e.g. from Rosetta or CASP decoy sets) would include
near-native decoys too, which is a MUCH harder discrimination task.
This script only claims to test the easy case: can the energy function
tell a real fold from scrambled garbage. Passing this is a necessary,
not sufficient, condition for the energy function being useful.

Run from the project root:
    python src/physics/validate_energy_ranking.py
"""

from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cif_parser import parse_cif  # noqa: E402
from src.physics.energy_module import total_energy  # noqa: E402
from src.physics.taxonomy_loader import load_taxonomy  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_scrambled_decoy(residues: list, seed: int = 42) -> list:
    """
    Generates a decoy by randomly shuffling WHICH residue occupies
    each spatial position, while keeping each residue's own internal
    atom geometry (bond lengths etc. within that residue) intact.

    Concretely: takes the list of (residue identity + internal atom
    coords relative to its own CA) and the list of (CA positions in
    the real structure), then randomly re-assigns identities to
    positions. This destroys the real fold/packing/contacts while
    keeping local per-residue geometry realistic — which is important,
    since a decoy with broken bond lengths would trivially fail on the
    covalent term alone and wouldn't test whether the NON-covalent
    terms (the harder, more interesting ones) can tell real from fake.
    """
    rng = random.Random(seed)

    # Extract (residue_name, {atom_name: coord relative to own CA}) per residue
    relative_geometries = []
    ca_positions = []
    for r in residues:
        if r.ca_coord is None:
            continue
        ca = r.ca_coord
        rel_coords = {name: (c[0] - ca[0], c[1] - ca[1], c[2] - ca[2])
                      for name, c in r.atom_coords.items()}
        relative_geometries.append((r.res_name, rel_coords, r.res_num))
        ca_positions.append(ca)

    shuffled_positions = ca_positions.copy()
    rng.shuffle(shuffled_positions)

    decoy_residues = []
    for (res_name, rel_coords, orig_res_num), new_ca in zip(relative_geometries, shuffled_positions):
        new_atom_coords = {
            name: (new_ca[0] + dx, new_ca[1] + dy, new_ca[2] + dz)
            for name, (dx, dy, dz) in rel_coords.items()
        }
        decoy_residue = deepcopy(residues[0])  # get a ResidueRecord-shaped object
        decoy_residue.res_name = res_name
        decoy_residue.res_num = orig_res_num
        decoy_residue.atom_coords = new_atom_coords
        decoy_residue.ca_coord = new_ca
        decoy_residues.append(decoy_residue)

    return decoy_residues


def make_near_native_decoy(residues: list, noise_std_angstrom: float, seed: int = 42) -> list:
    """
    Generates a MILDER decoy than make_scrambled_decoy: instead of
    fully reassigning which residue occupies which spatial position
    (destroying the fold entirely), this perturbs EVERY residue's
    position by small, independent Gaussian noise, while preserving
    each residue's own internal atom geometry (same relative-coordinate
    technique as the scrambled decoy).

    Why this matters: the original scrambled decoy is "obviously
    wrong" — real vs. decoy differ so drastically that the covalent
    and hydrogen-bond terms alone can discriminate them by many orders
    of magnitude, which can mask problems in weaker terms (this is
    exactly what happened with the hydrophobic term — see
    PHASE_3_README.md section 7). A near-native decoy with SMALL,
    controlled perturbation is a harder, more informative test: it
    keeps the overall fold recognizable while still measurably
    disturbing local packing, which is a fairer test of whether each
    INDIVIDUAL energy term (not just the dominant ones) can tell real
    from almost-real.

    Args:
        residues: list of ResidueRecord (from a real parsed structure)
        noise_std_angstrom: standard deviation of the Gaussian noise
            applied to each residue's CA position (and, by extension,
            all its atoms, since the internal geometry is preserved).
            Small values (e.g. 0.5-1.0 A) produce a mild, near-native
            decoy; larger values (e.g. 3-5 A) approach the difficulty
            of the fully-scrambled decoy without fully destroying
            sequential/local structure.
        seed: RNG seed for reproducibility.

    Returns:
        List of decoy ResidueRecord objects, same length as input.
    """
    rng = random.Random(seed)

    decoy_residues = []
    for r in residues:
        if r.ca_coord is None:
            continue
        ca = r.ca_coord
        # Independent Gaussian displacement per residue (not per atom —
        # this preserves each residue's internal geometry exactly,
        # while still perturbing WHERE that residue sits overall).
        dx = rng.gauss(0, noise_std_angstrom)
        dy = rng.gauss(0, noise_std_angstrom)
        dz = rng.gauss(0, noise_std_angstrom)

        new_atom_coords = {
            name: (c[0] + dx, c[1] + dy, c[2] + dz)
            for name, c in r.atom_coords.items()
        }
        decoy_residue = deepcopy(r)
        decoy_residue.atom_coords = new_atom_coords
        decoy_residue.ca_coord = (ca[0] + dx, ca[1] + dy, ca[2] + dz)
        decoy_residues.append(decoy_residue)

    return decoy_residues


def run_validation(cif_path: Path, chain: str, n_decoys: int = 5) -> bool:
    """
    Returns True if the real structure's total energy is lower than
    EVERY generated decoy's total energy (the validation passes).
    """
    taxonomy = load_taxonomy()

    print(f"Parsing {cif_path.name}, chain {chain}...")
    parsed = parse_cif(cif_path, target_chain=chain)
    print(f"  {len(parsed.residues)} residues")

    print("\nComputing energy for REAL structure...")
    real_breakdown = total_energy(parsed.residues, taxonomy)
    print(f"  Real structure total energy: {real_breakdown.total:.2f}")
    print(f"    covalent={real_breakdown.covalent:.2f}  "
          f"hbond={real_breakdown.hydrogen_bond:.2f}  "
          f"vdw={real_breakdown.van_der_waals:.2f}  "
          f"elec={real_breakdown.electrostatic:.2f}  "
          f"hydrophobic={real_breakdown.hydrophobic:.2f}")

    print(f"\nGenerating {n_decoys} scrambled decoys and computing their energy...")
    decoy_energies = []
    for i in range(n_decoys):
        decoy_residues = make_scrambled_decoy(parsed.residues, seed=i)
        decoy_breakdown = total_energy(decoy_residues, taxonomy)
        decoy_energies.append(decoy_breakdown.total)
        print(f"  Decoy {i+1}: total energy = {decoy_breakdown.total:.2f} "
              f"(covalent={decoy_breakdown.covalent:.2f}, "
              f"vdw={decoy_breakdown.van_der_waals:.2f})")

    print(f"\n{'=' * 60}")
    print("Validation result (scrambled decoy)")
    print(f"{'=' * 60}")
    n_real_lower = sum(1 for d in decoy_energies if real_breakdown.total < d)
    print(f"Real structure energy ({real_breakdown.total:.2f}) is LOWER than "
          f"{n_real_lower}/{n_decoys} decoys")

    passed = n_real_lower == n_decoys
    if passed:
        print("PASSED: real structure ranks as more stable than every decoy tested.")
    else:
        print("FAILED: at least one decoy scored as more stable than the real "
              "structure. This energy function is NOT yet reliable enough to "
              "couple into a GNN loss (Phase 4) — investigate which term(s) "
              "are misbehaving before proceeding.")
    return passed


def run_near_native_validation(
    cif_path: Path, chain: str, noise_levels: list[float], n_decoys_per_level: int = 5
) -> dict:
    """
    Harder validation: for each noise level, generates several
    near-native decoys and checks BOTH the total energy ranking AND
    each individual term's ranking — since the original scrambled-decoy
    test showed a term can be "wrong" (hydrophobic) while still being
    masked by the aggregate total passing (see PHASE_3_README.md
    section 7). This test is specifically designed to surface that
    kind of per-term problem rather than hide it behind the total.

    Returns a dict of {noise_level: {term_name: n_correct_out_of_n_decoys}}
    so the caller can see exactly which terms hold up at which
    difficulty level, rather than a single pass/fail.
    """
    taxonomy = load_taxonomy()
    parsed = parse_cif(cif_path, target_chain=chain)
    real_breakdown = total_energy(parsed.residues, taxonomy)

    term_names = ["covalent", "hydrogen_bond", "van_der_waals", "electrostatic", "hydrophobic", "total"]
    results = {}

    for noise_std in noise_levels:
        print(f"\n{'=' * 60}")
        print(f"Near-native decoy validation: noise_std = {noise_std} Angstrom")
        print(f"{'=' * 60}")

        term_correct_counts = {name: 0 for name in term_names}
        for i in range(n_decoys_per_level):
            decoy_residues = make_near_native_decoy(parsed.residues, noise_std_angstrom=noise_std, seed=i)
            decoy_breakdown = total_energy(decoy_residues, taxonomy)

            print(f"  Decoy {i+1}: total={decoy_breakdown.total:.2f} "
                  f"(real={real_breakdown.total:.2f}) | "
                  f"hydrophobic={decoy_breakdown.hydrophobic:.2f} "
                  f"(real={real_breakdown.hydrophobic:.2f})")

            for name in term_names:
                if getattr(real_breakdown, name) < getattr(decoy_breakdown, name):
                    term_correct_counts[name] += 1

        print(f"\n  Per-term results at noise_std={noise_std}A "
              f"(real ranked more stable than decoy, out of {n_decoys_per_level}):")
        for name in term_names:
            marker = "OK" if term_correct_counts[name] == n_decoys_per_level else "WEAK"
            print(f"    {name:16s}: {term_correct_counts[name]}/{n_decoys_per_level}  [{marker}]")

        results[noise_std] = term_correct_counts

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdb_id", nargs="?", default="1M17",
        help="PDB ID of the structure to validate against, expected at "
             "data/raw/pdb/{pdb_id}.cif (default: 1M17, the EGFR structure "
             "used throughout Phase 0-3 testing)"
    )
    parser.add_argument("--chain", default="A")
    parser.add_argument("--n_decoys", type=int, default=5)
    parser.add_argument(
        "--noise_levels", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0],
        help="Gaussian noise std (Angstrom) levels to test for the "
             "near-native decoy validation. Smaller = harder/more realistic."
    )
    parser.add_argument(
        "--skip_scrambled", action="store_true",
        help="skip the original scrambled-decoy test and only run the "
             "near-native validation"
    )
    args = parser.parse_args()

    cif_path = PROJECT_ROOT / "data" / "raw" / "pdb" / f"{args.pdb_id}.cif"
    if not cif_path.exists():
        # also try lowercase, matching the convention used elsewhere in
        # this project (e.g. verify_and_build_index.py's find_cif_file)
        cif_path_lower = PROJECT_ROOT / "data" / "raw" / "pdb" / f"{args.pdb_id.lower()}.cif"
        if cif_path_lower.exists():
            cif_path = cif_path_lower
        else:
            print(f"ERROR: {cif_path} not found.")
            sys.exit(1)

    scrambled_passed = True
    if not args.skip_scrambled:
        scrambled_passed = run_validation(cif_path, chain=args.chain, n_decoys=args.n_decoys)

    near_native_results = run_near_native_validation(
        cif_path, chain=args.chain, noise_levels=args.noise_levels, n_decoys_per_level=args.n_decoys
    )

    print(f"\n{'=' * 60}")
    print("Overall summary")
    print(f"{'=' * 60}")
    print(f"Scrambled-decoy test: {'PASSED' if scrambled_passed else 'FAILED'}")
    print("Near-native decoy test, hydrophobic term specifically:")
    for noise_std, counts in near_native_results.items():
        print(f"  noise_std={noise_std}A: hydrophobic {counts['hydrophobic']}/{args.n_decoys} correct")

    sys.exit(0 if scrambled_passed else 1)


if __name__ == "__main__":
    main()
