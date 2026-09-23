"""
visualize_structures.py

Phase 0, step 3: visual sanity check.

For a small set of proteins from data/processed/index.yaml, renders the
protein backbone, ligand(s), and labeled binding-site residues as a
standalone interactive HTML file (py3Dmol), so you can visually confirm:
  - the structure looks like an intact, sensible protein fold
  - the ligand is positioned where you'd expect (in a pocket, not floating
    off in space or embedded oddly)
  - the highlighted binding-site residues actually cluster AROUND the
    ligand, rather than being scattered randomly across the structure
    (this is the main thing worth checking, since binding_residues in
    index.yaml currently comes from a crude Phase 0 placeholder method,
    not real BioLiP2 annotations — see build_index_entry in
    verify_and_build_index.py)

Output: one .html file per protein in data/processed/structure_previews/,
viewable directly in any browser (double-click to open) — no Jupyter
notebook required.

Run from the project root:
    python src/data/visualize_structures.py                  # first 3 proteins
    python src/data/visualize_structures.py EGFR DHFR TP53    # specific genes
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import py3Dmol
import yaml
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.PDBExceptions import PDBIOException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.yaml"
RAW_PDB_DIR = PROJECT_ROOT / "data" / "raw" / "pdb"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "structure_previews"

# Visual style constants — kept as named colors rather than scattered
# magic strings, so the "what does each color mean" mapping is legible
# in one place and easy to change.
COLOR_BACKBONE = "#CFCFCF"        # light grey — de-emphasized, context only
COLOR_BINDING_RESIDUES = "#2E86AB"  # blue — the residues we're checking
COLOR_LIGAND = "#E94F37"          # red/orange — the ligand, high contrast


def find_cif_file(pdb_id: str) -> Path | None:
    for candidate in (RAW_PDB_DIR / f"{pdb_id.upper()}.cif", RAW_PDB_DIR / f"{pdb_id.lower()}.cif"):
        if candidate.exists():
            return candidate
    return None


def cif_to_pdb_string(cif_path: Path) -> str:
    """
    Convert mmCIF to legacy PDB format text via Biopython, and hand THAT
    to py3Dmol instead of raw CIF text.

    Why: 3Dmol.js's own CIF parser resolves chain/residue-number selectors
    (`chain`, `resi`) from the file's auth_asym_id / auth_seq_id fields,
    falling back to label_* fields if those are absent. In practice this
    has caused selection mismatches on real deposited structures (see
    3Dmol.js issue #817), which is the likely cause of binding-site
    residues silently not rendering. Legacy PDB format has a single,
    unambiguous numbering scheme, and 3Dmol's PDB parser/selector path is
    far more mature — converting removes this whole class of bug rather
    than chasing CIF-specific selector quirks.

    Caveat: PDB format truncates for very large structures (>99,999 atoms
    or >62 chains). None of the Phase 0 pilot proteins are anywhere close
    to this, but if a future protein hits it, this function will raise
    and the caller should fall back to raw CIF (with the caveat above).
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    io_ = PDBIO()
    io_.set_structure(structure)
    buffer = io.StringIO()
    with warnings.catch_warnings():
        # Biopython warns on chain IDs >1 char, discontinuous numbering,
        # etc. when downgrading to PDB format — expected and harmless for
        # a visualization-only conversion, so suppressed here rather than
        # spamming the console per-protein.
        warnings.simplefilter("ignore")
        try:
            io_.save(buffer)
        except PDBIOException as e:
            raise RuntimeError(
                f"Could not convert {cif_path.name} to PDB format "
                f"(likely too large for legacy PDB format): {e}"
            )
    return buffer.getvalue()


def build_view(entry: dict, cif_path: Path) -> py3Dmol.view:
    pdb_data = cif_to_pdb_string(cif_path)

    view = py3Dmol.view(width=900, height=650)
    view.addModel(pdb_data, "pdb")

    chain = entry["reference_structure"]["chain"]
    binding_residues = entry["binding_sites"]["binding_residues"]

    # 1. Whole structure as a faint cartoon, for overall fold context
    view.setStyle({}, {"cartoon": {"color": COLOR_BACKBONE, "opacity": 0.6}})

    # 2. Highlight binding-site residues on the target chain
    if binding_residues:
        view.addStyle(
            {"chain": chain, "resi": binding_residues},
            {"stick": {"color": COLOR_BINDING_RESIDUES, "radius": 0.25}},
        )
    else:
        print(f"    (note: no binding_residues listed for this entry)")

    # 3. Highlight ligand(s) — anything HETATM that isn't water, on any chain
    #    (this selector is not chain-scoped, so it already covers ligands
    #    regardless of which chain they sit on).
    view.addStyle(
        {"hetflag": True, "resn": ["HOH", "WAT"], "invert": True},
        {"stick": {"color": COLOR_LIGAND, "radius": 0.35}},
    )
    # NOTE: some peptide-like ligands (e.g. BACE1's OM99-2) may be recorded
    # as ATOM rather than HETATM in the source file, matching the same
    # edge case handled in cif_parser.py's peptide-ligand detection. Such
    # cases won't be colored red by the selector above and will instead
    # render in the default cartoon/backbone style.

    view.zoomTo()
    return view


def render_protein(entry: dict) -> Path | None:
    pdb_id = entry["reference_structure"]["pdb_id"]
    gene_name = entry["gene_name"]

    cif_path = find_cif_file(pdb_id)
    if cif_path is None:
        print(f"  SKIP {gene_name}: could not find {pdb_id}.cif in {RAW_PDB_DIR}")
        return None

    view = build_view(entry, cif_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{gene_name}_{pdb_id}.html"

    html = view._make_html()
    legend = f"""
    <div style="font-family: sans-serif; padding: 10px; background: #f5f5f5;">
      <h3>{gene_name} ({pdb_id}), chain {entry['reference_structure']['chain']}</h3>
      <p>
        <span style="color:{COLOR_BACKBONE}; font-weight:bold;">&#9632;</span> backbone &nbsp;
        <span style="color:{COLOR_BINDING_RESIDUES}; font-weight:bold;">&#9632;</span> labeled binding-site residues &nbsp;
        <span style="color:{COLOR_LIGAND}; font-weight:bold;">&#9632;</span> ligand
      </p>
      <p><b>Check:</b> do the blue binding-site residues visually cluster around
      the red ligand? If they're scattered across the whole structure instead,
      the Phase 0 placeholder binding-site labels for this protein need review
      before Phase 1.</p>
    </div>
    """
    with open(out_path, "w") as f:
        f.write(legend + html)

    print(f"  OK   {gene_name}: {out_path}")
    return out_path


def main():
    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} not found. Run verify_and_build_index.py first.")
        sys.exit(1)

    with open(INDEX_PATH) as f:
        proteins = yaml.safe_load(f)["proteins"]

    requested_genes = sys.argv[1:]
    if requested_genes:
        selected = [p for p in proteins if p["gene_name"] in requested_genes]
        missing = set(requested_genes) - {p["gene_name"] for p in selected}
        if missing:
            print(f"WARNING: not found in index.yaml, skipping: {missing}")
    else:
        selected = proteins[:3]  # default: first 3 for a quick spot-check

    print(f"Rendering {len(selected)} structure(s) to {OUTPUT_DIR}/\n")
    for entry in selected:
        render_protein(entry)

    print("\nOpen the .html file(s) above in any browser to inspect.")


if __name__ == "__main__":
    main()
