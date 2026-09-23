"""
verify_and_build_index.py

Phase 0, step 1: verification + index building.

For each protein in configs/protein_list.yaml, this script:
  1. Confirms the expected .cif file exists in data/raw/pdb/
  2. Parses it (via src/data/cif_parser.py)
  3. Runs validation checks (resolution, chain presence, residue count,
     ligand presence, UniProt cross-check)
  4. Writes a human-readable verification report (pass/fail + reasons)
  5. Emits data/processed/index.yaml containing ONLY entries that passed
     all checks — failing entries are reported, not silently included

Run from the project root:
    python src/data/verify_and_build_index.py

Exit behavior: does not raise on a single protein's failure — collects
all results and reports at the end, since the point of this step is to
tell you which of your 15 downloads are good, not to halt on the first
problem.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

# Allow running this script directly (python src/data/verify_and_build_index.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cif_parser import parse_cif, ParsedStructure  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTEIN_LIST_PATH = PROJECT_ROOT / "configs" / "protein_list.yaml"
RAW_PDB_DIR = PROJECT_ROOT / "data" / "raw" / "pdb"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "index.yaml"
REPORT_PATH = PROCESSED_DIR / "verification_report.txt"


@dataclass
class ValidationResult:
    uniprot_id: str
    gene_name: str
    pdb_id: str
    passed: bool
    errors: list[str]
    warnings: list[str]
    parsed: ParsedStructure | None = None


def find_cif_file(pdb_id: str) -> Path | None:
    """Look for the .cif file under a few common naming conventions,
    since manual downloads vary (e.g. '1M17.cif' vs '1m17.cif')."""
    candidates = [
        RAW_PDB_DIR / f"{pdb_id.upper()}.cif",
        RAW_PDB_DIR / f"{pdb_id.lower()}.cif",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def validate_entry(entry: dict) -> ValidationResult:
    uniprot_id = entry["uniprot_id"]
    gene_name = entry["gene_name"]
    pdb_id = entry["pdb_id"]
    expected_chain = entry["chain"]

    errors: list[str] = []
    warnings: list[str] = []

    # --- Check 1: file exists ---
    filepath = find_cif_file(pdb_id)
    if filepath is None:
        errors.append(
            f"File not found for PDB ID '{pdb_id}' in {RAW_PDB_DIR} "
            f"(looked for {pdb_id.upper()}.cif / {pdb_id.lower()}.cif)"
        )
        return ValidationResult(uniprot_id, gene_name, pdb_id, False, errors, warnings)

    # --- Check 2: file parses ---
    try:
        parsed = parse_cif(filepath, target_chain=expected_chain)
    except Exception as e:
        errors.append(f"Failed to parse {filepath.name}: {e}")
        return ValidationResult(uniprot_id, gene_name, pdb_id, False, errors, warnings)

    # --- Check 3: expected chain present ---
    if expected_chain not in parsed.chains_present:
        errors.append(
            f"Expected chain '{expected_chain}' not found. "
            f"Chains present: {parsed.chains_present}"
        )
        # Can't validate further checks meaningfully without the right chain
        return ValidationResult(uniprot_id, gene_name, pdb_id, False, errors, warnings, parsed)

    # --- Check 4: resolution ---
    max_res = entry.get("max_resolution")
    if parsed.resolution is None:
        warnings.append("Resolution not found in file metadata (could not verify)")
    elif max_res is not None and parsed.resolution > max_res:
        errors.append(
            f"Resolution {parsed.resolution:.2f} \u00c5 exceeds max allowed {max_res} \u00c5"
        )

    # --- Check 5: minimum residue count (catches truncated/wrong chain) ---
    min_residues = entry.get("min_residues", 0)
    n_residues = len(parsed.residues)
    if n_residues < min_residues:
        errors.append(
            f"Chain '{expected_chain}' has {n_residues} standard residues, "
            f"expected at least {min_residues}"
        )

    # --- Check 6: ligand presence (if expected) ---
    if entry.get("expect_ligand", False):
        if len(parsed.ligands) == 0:
            errors.append(
                f"Expected at least one ligand (non-water HETATM group) in "
                f"chain '{expected_chain}', found none"
            )
    else:
        if len(parsed.ligands) == 0:
            warnings.append(
                "No ligand found (expected — entry marked expect_ligand: false)"
            )

    # --- Check 7: UniProt cross-check ---
    if parsed.uniprot_accessions:
        if uniprot_id not in parsed.uniprot_accessions:
            errors.append(
                f"UniProt mismatch: expected {uniprot_id}, "
                f"file references {parsed.uniprot_accessions}"
            )
    else:
        warnings.append("No UniProt accession found in file metadata (could not cross-check)")

    passed = len(errors) == 0
    return ValidationResult(uniprot_id, gene_name, pdb_id, passed, errors, warnings, parsed)


def build_index_entry(entry: dict, result: ValidationResult) -> dict:
    """Construct the index.yaml record for a protein that passed validation,
    following the schema designed in Phase 0 planning (UniProt-anchored,
    with an empty `variants` list reserved for Phase 6)."""
    parsed = result.parsed
    sequence = "".join(
        _three_to_one(r.res_name) for r in parsed.residues
    )
    binding_residues = sorted({
        r.res_num for lig in parsed.ligands
        for r in parsed.residues
        if _is_near(r, lig, cutoff=4.5)
    })

    return {
        "uniprot_id": entry["uniprot_id"],
        "gene_name": entry["gene_name"],
        "reference_structure": {
            "pdb_id": entry["pdb_id"],
            "chain": entry["chain"],
            "resolution": parsed.resolution,
            "source": "PDB",
            "family": entry.get("family"),
        },
        "binding_sites": {
            "source": "computed_heavy_atom_contact_4.5A",  # all-heavy-atom
            # distance method, matching BioLiP2's contact definition —
            # replaces the earlier CA-only proxy (see _is_near docstring)
            "ligands": _summarize_ligands(parsed.ligands),
            "binding_residues": binding_residues,
        },
        "variants": [],  # reserved for Phase 6 (ClinVar/gnomAD)
        "sequence": sequence,
        "notes": entry.get("notes", ""),
    }


def _summarize_ligands(ligands) -> list[str]:
    """Small-molecule HET codes are listed individually (e.g. 'AQ4').
    Peptide-like ligand chains (is_peptide_like=True) are collapsed to
    one entry per chain (e.g. 'peptide_ligand:C'), since listing every
    amino acid code in a 7-residue inhibitor separately would be
    misleading — it's one ligand, not seven."""
    small_molecule = {lig.res_name for lig in ligands if not lig.is_peptide_like}
    peptide_chains = {lig.chain_id for lig in ligands if lig.is_peptide_like}
    summary = sorted(small_molecule) + sorted(f"peptide_ligand:{c}" for c in peptide_chains)
    return summary


def _three_to_one(res_name: str) -> str:
    table = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return table.get(res_name, "X")


def _is_near(residue, ligand, cutoff: float) -> bool:
    """
    Heavy-atom contact check: True if ANY non-hydrogen atom of the
    residue is within `cutoff` Angstroms of ANY non-hydrogen atom of
    the ligand.

    This replaces the earlier CA-only proxy. CA-only distance
    systematically misses real contacts where a residue's SIDE CHAIN
    (not its backbone CA) is the part actually touching the ligand —
    e.g. a long Lysine/Arginine side chain can reach a ligand while its
    CA sits well outside the cutoff. All-heavy-atom distance at 4.5A is
    the definition BioLiP2 itself uses for ligand-binding contacts, so
    this is a correctness upgrade toward that standard, not just a
    different arbitrary threshold.

    Hydrogens are excluded deliberately: most X-ray structures at
    typical resolution (>1.2A) don't have experimentally resolved
    hydrogens anyway, but if a file DOES include them (e.g. from a
    computational/refined source), including H atoms would bias contact
    distances shorter than the heavy-atom-based cutoff was calibrated
    for, so they're filtered out either way for consistency.
    """
    import math

    residue_atoms = [
        coord for name, coord in residue.atom_coords.items()
        if not name.strip().startswith("H")
    ]
    ligand_atoms = [
        coord for name, coord in ligand.atom_coords.items()
        if not name.strip().startswith("H")
    ]

    if not residue_atoms or not ligand_atoms:
        return False

    for r_coord in residue_atoms:
        for l_coord in ligand_atoms:
            if math.dist(r_coord, l_coord) <= cutoff:
                return True
    return False


def main():
    if not PROTEIN_LIST_PATH.exists():
        print(f"ERROR: protein list not found at {PROTEIN_LIST_PATH}")
        sys.exit(1)
    else:
        print(f"Protein list found at {PROTEIN_LIST_PATH}")

    with open(PROTEIN_LIST_PATH) as f:
        protein_list = yaml.safe_load(f)["proteins"]

    RAW_PDB_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ValidationResult] = []
    for entry in protein_list:
        print(f"Validating {entry['gene_name']} ({entry['uniprot_id']}, PDB {entry['pdb_id']})")
        result = validate_entry(entry)
        results.append(result)

    # --- Write report ---
    n_passed = sum(r.passed for r in results)
    n_total = len(results)

    report_lines = [
        f"Phase 0 Verification Report",
        f"{'=' * 60}",
        f"Total proteins checked: {n_total}",
        f"Passed: {n_passed}",
        f"Failed: {n_total - n_passed}",
        "",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        report_lines.append(f"[{status}] {r.gene_name} ({r.uniprot_id}, PDB {r.pdb_id})")
        for e in r.errors:
            report_lines.append(f"    ERROR:   {e}")
        for w in r.warnings:
            report_lines.append(f"    WARNING: {w}")
    report_text = "\n".join(report_lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nFull report written to {REPORT_PATH}")

    # --- Build index.yaml from passing entries only ---
    entry_by_uniprot = {e["uniprot_id"]: e for e in protein_list}
    index_entries = [
        build_index_entry(entry_by_uniprot[r.uniprot_id], r)
        for r in results if r.passed
    ]

    with open(INDEX_PATH, "w") as f:
        yaml.dump({"proteins": index_entries}, f, sort_keys=False, default_flow_style=False)

    print(f"index.yaml written with {len(index_entries)} verified entries -> {INDEX_PATH}")

    if n_passed < n_total:
        print(
            f"\n{n_total - n_passed} protein(s) failed validation and were "
            f"EXCLUDED from index.yaml. Review the report above, fix the "
            f"underlying issue (wrong PDB ID, missing file, wrong chain, etc.), "
            f"and re-run this script."
        )


if __name__ == "__main__":
    main()
