"""
cif_parser.py

Parses mmCIF protein structure files into structured records:
polymer residues (for graph construction in Phase 1) and
ligand/HETATM groups (for binding-site labels).

Deliberately has NO knowledge of the pilot protein list or validation
rules — verify_and_build_index.py owns that. This module only answers
"what is physically in this file", so it stays reusable when the
dataset scales past the pilot list, and later for variant/isoform
structures in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

# Standard 20 amino acids (3-letter codes) — used to separate polymer
# residues from ligands/waters/ions/modified residues.
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

WATER = {"HOH", "WAT"}

# Chains at or below this residue count, sitting on a chain ID other than
# the target polymer chain, are treated as candidate peptide-like ligands
# (e.g. peptidomimetic inhibitors such as BACE1's OM99-2, deposited as a
# short standalone chain of standard amino acids rather than a HETATM
# group). This catches ligands that plain HETATM scanning on the target
# chain misses. Tune if a real target polymer subunit is ever this short.
PEPTIDE_LIGAND_MAX_RESIDUES = 15


@dataclass
class ResidueRecord:
    """A single polymer (amino acid) residue."""
    chain_id: str
    res_num: int
    res_name: str
    ca_coord: tuple[float, float, float] | None  # CA atom coordinate, if present
    atom_coords: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class LigandRecord:
    """A non-polymer HETATM group (excludes water), OR a residue that is
    part of a short peptide-like ligand chain (see PEPTIDE_LIGAND_MAX_RESIDUES).
    `res_name` is the 3-letter residue code either way; for peptide-like
    ligands this will be a standard amino acid code, not a HET code —
    use `is_peptide_like` to distinguish."""
    chain_id: str
    res_num: int
    res_name: str
    atom_coords: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    is_peptide_like: bool = False


@dataclass
class ParsedStructure:
    pdb_id: str
    resolution: float | None
    chains_present: list[str]
    residues: list[ResidueRecord]      # filtered to the requested chain
    ligands: list[LigandRecord]        # filtered to the requested chain
    uniprot_accessions: list[str]      # from _struct_ref, for cross-checking
    raw_structure: Structure           # full Biopython object, for downstream use


def _extract_resolution(mmcif_dict: dict) -> float | None:
    """mmCIF stores resolution under different keys depending on the
    experimental method (X-ray vs cryo-EM etc). Try the common ones."""
    for key in (
        "_refine.ls_d_res_high",
        "_reflns.d_resolution_high",
        "_em_3d_reconstruction.resolution",
    ):
        val = mmcif_dict.get(key)
        if val:
            try:
                # mmCIF values are often lists even for single values
                v = val[0] if isinstance(val, list) else val
                return float(v)
            except (ValueError, TypeError):
                continue
    return None


def _extract_uniprot_accessions(mmcif_dict: dict) -> list[str]:
    """Pull UniProt accessions from _struct_ref, used to cross-check
    that the downloaded structure actually corresponds to the expected
    protein (catches wrong-PDB-ID mistakes)."""
    db_names = mmcif_dict.get("_struct_ref.db_name", [])
    db_accessions = mmcif_dict.get("_struct_ref.pdbx_db_accession", [])
    if isinstance(db_names, str):
        db_names = [db_names]
    if isinstance(db_accessions, str):
        db_accessions = [db_accessions]

    accessions = []
    for name, acc in zip(db_names, db_accessions):
        if name.upper() == "UNP":  # UniProt Knowledgebase
            accessions.append(acc)
    return accessions


def parse_cif(filepath: Path, target_chain: str) -> ParsedStructure:
    """
    Parse a single mmCIF file.

    Args:
        filepath: path to the .cif file
        target_chain: chain ID to extract residues/ligands from
                       (structure may contain other chains; we keep
                       all chain IDs in `chains_present` for validation,
                       but `residues`/`ligands` are filtered to target_chain)

    Raises:
        FileNotFoundError, ValueError on unparseable files — callers
        (verify_and_build_index.py) are responsible for catching these
        and recording them as validation failures, not crashing the batch.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CIF file not found: {filepath}")

    pdb_id = filepath.stem.upper()

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(filepath))

    mmcif_dict = MMCIF2Dict(str(filepath))
    resolution = _extract_resolution(mmcif_dict)
    uniprot_accessions = _extract_uniprot_accessions(mmcif_dict)

    model = next(structure.get_models())  # first model only (crystal structures
                                           # have 1; NMR/cryo-EM ensembles may have
                                           # more — Phase 0 uses model 1 only)
    chains_present = [chain.id for chain in model]

    residues: list[ResidueRecord] = []
    ligands: list[LigandRecord] = []

    if target_chain in [c for c in chains_present]:
        chain = model[target_chain]
        for residue in chain:
            hetflag, res_num, _ = residue.id
            res_name = residue.resname

            if hetflag == " " and res_name in STANDARD_AA:
                # Standard polymer residue
                atom_coords = {atom.get_name(): tuple(atom.coord) for atom in residue}
                ca_coord = atom_coords.get("CA")
                residues.append(ResidueRecord(
                    chain_id=target_chain,
                    res_num=res_num,
                    res_name=res_name,
                    ca_coord=ca_coord,
                    atom_coords=atom_coords,
                ))
            elif hetflag != " " and res_name not in WATER:
                # Ligand / non-standard HETATM group, excluding water
                atom_coords = {atom.get_name(): tuple(atom.coord) for atom in residue}
                ligands.append(LigandRecord(
                    chain_id=target_chain,
                    res_num=res_num,
                    res_name=res_name,
                    atom_coords=atom_coords,
                    is_peptide_like=False,
                ))

    # Also scan OTHER chains for short peptide-like ligands (e.g. BACE1's
    # OM99-2 inhibitor, deposited as its own short chain rather than a
    # HETATM group on the target chain). Such inhibitors are frequently
    # peptidomimetics: a mix of standard amino acids PLUS non-standard
    # "residues" (e.g. '1OL', a statine-like transition-state-mimetic
    # unit) — so purity (100% standard AA) is NOT required, only that
    # the chain is short and not water-dominated.
    for chain in model:
        if chain.id == target_chain:
            continue
        chain_residues = [r for r in chain if r.resname not in WATER]
        if 0 < len(chain_residues) <= PEPTIDE_LIGAND_MAX_RESIDUES:
            for residue in chain_residues:
                _, res_num, _ = residue.id
                atom_coords = {atom.get_name(): tuple(atom.coord) for atom in residue}
                ligands.append(LigandRecord(
                    chain_id=chain.id,
                    res_num=res_num,
                    res_name=residue.resname,
                    atom_coords=atom_coords,
                    is_peptide_like=True,
                ))

    return ParsedStructure(
        pdb_id=pdb_id,
        resolution=resolution,
        chains_present=chains_present,
        residues=residues,
        ligands=ligands,
        uniprot_accessions=uniprot_accessions,
        raw_structure=structure,
    )
