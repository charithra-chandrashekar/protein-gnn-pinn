"""
build_dataset.py

Phase 1, step 2 (assembly): the "glue" step that turns
  data/processed/index.yaml + data/splits/{train,val,test}.txt + .cif files
into
  ready-to-train PyTorch Geometric graphs, split and saved to disk.

For each protein in each split:
  1. Look up its entry in index.yaml (pdb_id, chain, binding_residues)
  2. Find and parse its .cif file (cif_parser.py)
  3. Build its graph (graph_builder.py)
  4. On failure: report which protein and why, SKIP it, and keep going
     — one bad protein must not crash the whole assembly run, same
     "verify and report, don't silently break" pattern used in
     verify_and_build_index.py

Output: data/processed/graphs.pt — a dict with keys 'train'/'val'/'test',
each a list of PyG Data objects, loadable directly by the training script
via torch.load(..., weights_only=False) (PyG Data objects require this).

Run from the project root:
    python src/data/build_dataset.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cif_parser import parse_cif  # noqa: E402
from src.data.graph_builder import build_graph, DEFAULT_RADIUS_CUTOFF  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.yaml"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RAW_PDB_DIR = PROJECT_ROOT / "data" / "raw" / "pdb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "graphs.pt"


def find_cif_file(pdb_id: str) -> Path | None:
    for candidate in (RAW_PDB_DIR / f"{pdb_id.upper()}.cif", RAW_PDB_DIR / f"{pdb_id.lower()}.cif"):
        if candidate.exists():
            return candidate
    return None


def load_split(name: str) -> list[str]:
    path = SPLITS_DIR / f"{name}.txt"
    if not path.exists():
        print(f"WARNING: {path} not found (run make_splits.py first). Treating as empty.")
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_graph_for_protein(entry: dict, radius_cutoff: float) -> tuple:
    """Returns (Data, GraphBuildStats) on success, or (None, error_message) on failure."""
    gene_name = entry["gene_name"]
    pdb_id = entry["reference_structure"]["pdb_id"]
    chain = entry["reference_structure"]["chain"]
    binding_residues = entry["binding_sites"]["binding_residues"]

    cif_path = find_cif_file(pdb_id)
    if cif_path is None:
        return None, f"{gene_name}: .cif file not found for PDB ID '{pdb_id}' in {RAW_PDB_DIR}"

    try:
        parsed = parse_cif(cif_path, target_chain=chain)
    except Exception as e:
        return None, f"{gene_name}: failed to parse {cif_path.name}: {e}"

    try:
        data, stats = build_graph(parsed, binding_residues, radius_cutoff=radius_cutoff)
    except ValueError as e:
        return None, f"{gene_name}: failed to build graph: {e}"

    # Attach identifying metadata directly on the Data object — makes it
    # possible to trace a graph back to its source protein later (e.g.
    # when inspecting per-protein predictions/errors during training).
    data.gene_name = gene_name
    data.uniprot_id = entry["uniprot_id"]
    data.pdb_id = pdb_id

    if stats.n_binding_site_residues == 0:
        print(
            f"  NOTE: {gene_name} has 0 binding-site residues in this graph "
            f"(label will be all-zero — usable for training but won't "
            f"contribute positive examples)"
        )

    return data, stats


def assemble_split(split_name: str, uniprot_ids: list[str], structures_by_uniprot: dict,
                    radius_cutoff: float) -> list:
    n_structures_expected = sum(len(structures_by_uniprot.get(uid, [])) for uid in uniprot_ids)
    print(f"\n--- {split_name} ({len(uniprot_ids)} unique proteins, "
          f"{n_structures_expected} structures) ---")
    graphs = []
    for uid in uniprot_ids:
        entries = structures_by_uniprot.get(uid)
        if not entries:
            print(f"  SKIP: UniProt ID {uid} (from {split_name}.txt) not found in index.yaml")
            continue

        for entry in entries:
            result, info = build_graph_for_protein(entry, radius_cutoff)
            if result is None:
                print(f"  FAIL: {info}")
                continue

            data = result
            stats = info
            print(
                f"  OK:   {entry['gene_name']:12s} ({entry['reference_structure']['pdb_id']}) "
                f"nodes={stats.n_nodes:4d} edges={stats.n_edges:5d} "
                f"binding_site_residues={stats.n_binding_site_residues}"
            )
            graphs.append(data)

    return graphs


def main():
    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} not found. Run verify_and_build_index.py first.")
        sys.exit(1)

    with open(INDEX_PATH) as f:
        proteins = yaml.safe_load(f)["proteins"]

    # ONE-TO-MANY mapping: a single UniProt ID can have multiple structure
    # entries in index.yaml (e.g. 38 different Thrombin crystal structures).
    # A plain {uniprot_id: entry} dict would silently keep only the LAST
    # entry seen per UniProt ID and discard the rest — this was a real bug
    # found when auditing for train/test leakage; fixed here by collecting
    # a LIST of entries per UniProt ID instead.
    structures_by_uniprot: dict[str, list] = defaultdict(list)
    for p in proteins:
        structures_by_uniprot[p["uniprot_id"]].append(p)

    splits = {
        "train": load_split("train"),
        "val": load_split("val"),
        "test": load_split("test"),
    }

    assembled = {}
    for split_name, uniprot_ids in splits.items():
        assembled[split_name] = assemble_split(
            split_name, uniprot_ids, structures_by_uniprot, DEFAULT_RADIUS_CUTOFF
        )

    print("\n" + "=" * 50)
    print("Assembly summary")
    print("=" * 50)
    total_structures_built = sum(len(v) for v in assembled.values())
    total_structures_expected = sum(
        len(structures_by_uniprot.get(uid, [])) for uids in splits.values() for uid in uids
    )
    for split_name in ("train", "val", "test"):
        n_proteins = len(splits[split_name])
        n_expected = sum(len(structures_by_uniprot.get(uid, [])) for uid in splits[split_name])
        n_built = len(assembled[split_name])
        status = "OK" if n_built == n_expected else "SOME FAILED"
        print(
            f"{split_name:6s}: {n_built}/{n_expected} structures built "
            f"({n_proteins} unique proteins) [{status}]"
        )
    print(f"total : {total_structures_built}/{total_structures_expected} structures")

    if total_structures_built == 0:
        print("\nERROR: no graphs were built at all. Nothing to save. "
              "Check the FAIL messages above.")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(assembled, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(
        "Load later with: "
        "torch.load('data/processed/graphs.pt', weights_only=False)"
    )

    if total_structures_built < total_structures_expected:
        print(
            f"\n{total_structures_expected - total_structures_built} structure(s) "
            f"failed to build and were excluded. Review FAIL messages above "
            f"before training — a training run on fewer structures than "
            f"expected can silently look like a smaller dataset was intentional."
        )


if __name__ == "__main__":
    main()
