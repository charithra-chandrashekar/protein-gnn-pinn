"""
graph_builder.py

Phase 1, step 2: converts a parsed protein structure (from cif_parser.py)
into a PyTorch Geometric graph, ready for GNN training/evaluation.

Design (see conversation for full rationale):
  - Nodes: one per polymer residue, feature = one-hot amino acid identity
  - Edges: radius graph on CA-CA distance (default 8 A cutoff) PLUS
    explicit sequential backbone edges (i -> i+1), flagged separately
  - Edge features: [distance, is_backbone] — physics/chemistry-richer
    edge features (bond angles, bond types) are deliberately deferred to
    Phase 2-3, not included here
  - Target (y): per-residue binary binding-site label, kept as a SEPARATE
    tensor from node features `x` — must never be concatenated into x,
    or the model would be trained on a label leaked into its own input

This module has no knowledge of index.yaml file paths or the pilot
protein list — it only knows how to turn a ParsedStructure + a set of
binding-site residue numbers into a graph. Orchestration (looping over
index.yaml, loading each .cif, saving graphs to disk) belongs in a
separate script, not here — same separation-of-concerns principle used
for cif_parser.py vs verify_and_build_index.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.cif_parser import ParsedStructure

# Fixed vocabulary and ordering for one-hot amino acid encoding. Order
# doesn't matter functionally, but MUST stay stable across the whole
# project once training starts — changing this later would silently
# invalidate any already-trained model's input encoding.
STANDARD_AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AA_ORDER)}

DEFAULT_RADIUS_CUTOFF = 8.0  # Angstrom, CA-CA distance for spatial edges


@dataclass
class GraphBuildStats:
    """Diagnostics returned alongside the graph, so callers can sanity-check
    a build without having to re-inspect the Data object's internals."""
    n_nodes: int
    n_edges: int
    n_backbone_edges: int
    n_spatial_edges: int
    n_binding_site_residues: int
    n_residues_missing_ca: int  # residues dropped because CA coord was absent


def _one_hot_aa(res_name: str) -> np.ndarray:
    """One-hot encode a residue's amino acid identity. Unknown/non-standard
    residue names map to an all-zero vector rather than raising — this
    keeps the builder robust to occasional non-standard residues without
    silently mis-assigning them to an arbitrary standard AA."""
    vec = np.zeros(len(STANDARD_AA_ORDER), dtype=np.float32)
    idx = AA_TO_INDEX.get(res_name)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def build_graph(
    parsed: ParsedStructure,
    binding_residue_numbers: set[int] | list[int],
    radius_cutoff: float = DEFAULT_RADIUS_CUTOFF,
) -> tuple[Data, GraphBuildStats]:
    """
    Build a PyG graph from a parsed structure.

    Args:
        parsed: output of cif_parser.parse_cif(...) — uses parsed.residues
                 (already filtered to the target chain)
        binding_residue_numbers: residue numbers (matching ResidueRecord.res_num)
                 that are labeled as binding-site residues (from index.yaml)
        radius_cutoff: CA-CA distance threshold (Angstrom) for spatial edges

    Returns:
        (Data, GraphBuildStats)

    Data fields:
        x:          [n_nodes, 20] float32 — one-hot amino acid identity
        y:          [n_nodes] float32 — binary binding-site label (0/1)
        edge_index: [2, n_edges] long — COO format, UNDIRECTED (both directions included)
        edge_attr:  [n_edges, 2] float32 — [ca_distance, is_backbone (0/1)]
        pos:        [n_nodes, 3] float32 — CA coordinates (kept for physics
                     terms in later phases; NOT used as a node feature here)
        res_num:    [n_nodes] long — original residue numbers, for traceability
                     back to the source structure / index.yaml

    Raises:
        ValueError if fewer than 2 residues have valid CA coordinates
        (can't build a meaningful graph).
    """
    binding_set = set(binding_residue_numbers)

    # Filter to residues with a valid CA coordinate — a residue with
    # missing backbone atoms (common in lower-resolution or partially
    # disordered structures) can't be placed in the graph.
    usable_residues = [r for r in parsed.residues if r.ca_coord is not None]
    n_missing_ca = len(parsed.residues) - len(usable_residues)

    if len(usable_residues) < 2:
        raise ValueError(
            f"Only {len(usable_residues)} residue(s) with valid CA coordinates "
            f"found (need at least 2 to build a graph). Structure: {parsed.pdb_id}"
        )

    n_nodes = len(usable_residues)
    positions = np.array([r.ca_coord for r in usable_residues], dtype=np.float32)
    res_nums = np.array([r.res_num for r in usable_residues], dtype=np.int64)

    # --- Node features ---
    x = np.stack([_one_hot_aa(r.res_name) for r in usable_residues])
    y = np.array(
        [1.0 if r.res_num in binding_set else 0.0 for r in usable_residues],
        dtype=np.float32,
    )

    # --- Edges: spatial radius graph ---
    # Pairwise distance matrix — fine at Phase 1 scale (single-protein
    # graphs, hundreds of residues). Revisit with a proper neighbor-search
    # structure (e.g. scipy cKDTree) if graphs grow much larger later.
    diffs = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)

    spatial_edges = []
    spatial_dists = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if dist_matrix[i, j] <= radius_cutoff:
                spatial_edges.append((i, j))
                spatial_dists.append(dist_matrix[i, j])

    # --- Edges: explicit sequential backbone (i -> i+1 and i+1 -> i) ---
    # Based on ARRAY POSITION in usable_residues, not on res_num arithmetic
    # — res_num can have gaps (missing/unresolved loop residues), so
    # "res_num + 1" would incorrectly skip edges across a gap, or
    # incorrectly bridge one if two chains happened to share numbering.
    # Consecutive entries in usable_residues are consecutive in the actual
    # resolved chain, which is the correct notion of "sequential" here.
    backbone_edge_set = set()
    for i in range(n_nodes - 1):
        backbone_edge_set.add((i, i + 1))
        backbone_edge_set.add((i + 1, i))

    # --- Merge: build final edge list, tagging is_backbone, avoiding duplicates ---
    all_edges: dict[tuple[int, int], float] = {}
    for (i, j), d in zip(spatial_edges, spatial_dists):
        all_edges[(i, j)] = d
    for (i, j) in backbone_edge_set:
        if (i, j) not in all_edges:
            all_edges[(i, j)] = float(dist_matrix[i, j])

    edge_index_list = []
    edge_attr_list = []
    n_backbone = 0
    n_spatial_only = 0
    for (i, j), d in all_edges.items():
        is_backbone = 1.0 if (i, j) in backbone_edge_set else 0.0
        if is_backbone:
            n_backbone += 1
        else:
            n_spatial_only += 1
        edge_index_list.append((i, j))
        edge_attr_list.append((d, is_backbone))

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float32),
        res_num=torch.tensor(res_nums, dtype=torch.long),
    )

    stats = GraphBuildStats(
        n_nodes=n_nodes,
        n_edges=edge_index.shape[1],
        n_backbone_edges=n_backbone,
        n_spatial_edges=n_spatial_only,
        n_binding_site_residues=int(y.sum()),
        n_residues_missing_ca=n_missing_ca,
    )

    return data, stats
