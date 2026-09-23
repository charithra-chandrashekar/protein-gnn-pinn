"""
select_proteins.py

Phase 0 (scaled): programmatically populates configs/protein_list.yaml
by querying the live RCSB PDB Search + Data APIs, replacing hand-curation.

Uses the official `rcsb-api` package (pip install rcsb-api), maintained
by RCSB PDB itself — NOT hand-built HTTP requests — so query syntax is
validated against RCSB's live schema rather than guessed.

IMPORTANT — network requirement: this script fetches RCSB's schema and
queries live over the network on EVERY run (the rcsbapi package does
this on import). It cannot run in a network-restricted sandbox; it must
run on a machine with normal internet access to search.rcsb.org and
data.rcsb.org.

Pipeline:
  1. Search API: find PDB entries matching quality criteria (resolution,
     experimental method, human protein, has at least one bound ligand)
  2. Data API (entries-level): for each match, fetch resolution, chain
     info, ligand identity/details, and UniProt cross-reference (batched,
     since the Data API allows up to 1000 IDs per batch request)
  3. Filter out entries whose only "ligand" is a known crystallization
     artifact (water is already excluded by PDB conventions, but
     glycerol, sulfate, PEG fragments, etc. are real HETATM ligands
     that are NOT biologically meaningful binding partners)
  4. Data API (polymer_entity_instances-level): for each SURVIVING
     {pdb_id, chain} pair, fetch Pfam and CATH classification —
     these live as "positional features" on the instance object, not
     as scalar fields on the entry, so this is a genuinely separate
     query, deliberately run only after step 3's filtering to avoid
     fetching classification data for entries about to be discarded
  5. Write configs/protein_list.yaml with pfam_family and
     cath_classification as separate fields (both may independently
     be "unclassified" if RCSB has no such annotation for that chain)

Output is INCREMENTAL: results are written to disk as they're fetched
(not held in memory until the very end), and running with --resume
skips PDB IDs already present in the existing protein_list.yaml. This
matters at 500-1000+ scale where a network hiccup partway through
should not mean starting over.

Run:
    python src/data/select_proteins.py --target_count 500
    python src/data/select_proteins.py --target_count 1000 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "configs" / "protein_list.yaml"

# Common crystallization additives / buffer components that show up as
# HETATM ligands but are not biologically meaningful binding partners.
# This list is deliberately NOT exhaustive — it catches the most common
# offenders; verify_and_build_index.py's per-structure checks remain the
# final correctness gate, this is just a first-pass filter to avoid
# populating the list with mostly-useless entries.
ARTIFACT_LIGAND_CODES = {
    "HOH", "WAT",                      # water (usually already excluded)
    "GOL", "EDO", "PEG", "PG4", "1PE",  # glycerol, ethylene glycol, PEG fragments
    "SO4", "PO4", "NO3", "ACT",         # sulfate, phosphate, nitrate, acetate
    "NA", "CL", "K", "MG", "CA", "ZN",  # common buffer/crystallization ions
    "MPD", "TRS", "BME", "DTT",         # common crystallization/reducing agents
    "IOD", "BR", "FMT", "EPE",          # iodide, bromide, formate, HEPES
}

DEFAULT_MAX_RESOLUTION = 2.8
DEFAULT_MIN_RESIDUES = 50
DATA_API_BATCH_SIZE = 1000  # RCSB Data API hard limit per batch request


def build_search_query(max_resolution: float, organism: str | None):
    """Constructs the RCSB Search API query. Imports rcsbapi lazily
    (inside the function, not at module level) so this file can still
    be inspected/edited without a network connection — the import
    itself triggers a live schema fetch."""
    from rcsbapi.search import AttributeQuery, search_attributes as attrs

    q_method = AttributeQuery(
        attribute="exptl.method", operator="exact_match", value="X-RAY DIFFRACTION"
    )
    q_resolution = attrs.rcsb_entry_info.resolution_combined <= max_resolution
    # has at least one non-polymer (ligand) entity bound
    q_has_ligand = attrs.rcsb_entry_info.nonpolymer_entity_count >= 1
    # restrict to protein polymers only (excludes nucleic-acid-only structures)
    q_is_protein = attrs.entity_poly.rcsb_entity_polymer_type == "Protein"

    query = q_method & q_resolution & q_has_ligand & q_is_protein

    if organism:
        q_organism = attrs.rcsb_entity_source_organism.scientific_name == organism
        query = query & q_organism

    return query


def search_candidate_pdb_ids(max_resolution: float, target_count: int, organism: str | None) -> list[str]:
    """Runs the Search API query and returns up to target_count PDB IDs.
    Note: Search API returns ENTRY-level IDs (4-character PDB codes),
    not chain-specific — chain selection happens later using Data API
    results, since we need per-chain polymer/ligand info anyway."""
    query = build_search_query(max_resolution, organism)
    print(f"Querying RCSB Search API (max_resolution<={max_resolution}, organism={organism})...")

    results = []
    for pdb_id in query():
        results.append(pdb_id)
        if len(results) >= target_count:
            break

    print(f"Search API returned {len(results)} candidate PDB IDs")
    return results


def fetch_entry_details_batch(pdb_ids: list[str]) -> dict:
    """Fetches resolution, polymer entity / chain info, ligand identities,
    and UniProt cross-references for a batch of PDB IDs via the Data API.
    Returns {pdb_id: {...}} — entries that fail to fetch are simply
    absent from the returned dict (caller treats missing as unusable,
    consistent with the "skip and report" pattern used elsewhere in
    this pipeline)."""
    from rcsbapi.data import DataQuery

    query = DataQuery(
        input_type="entries",
        input_ids=pdb_ids,
        return_data_list=[
            "rcsb_id",
            "rcsb_entry_info.resolution_combined",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.asym_ids",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.auth_asym_ids",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
            "polymer_entities.rcsb_entity_source_organism.scientific_name",
            "polymer_entities.entity_poly.rcsb_sample_sequence_length",
            "polymer_entities.rcsb_polymer_entity.pdbx_description",
            "nonpolymer_entities.pdbx_entity_nonpoly.comp_id",
            "nonpolymer_entities.pdbx_entity_nonpoly.name",
        ],
    )
    result = query.exec()
    return result


def fetch_classifications_batch(instance_ids: list[str]) -> dict[str, dict]:
    """
    Fetches Pfam and CATH classification for a batch of polymer entity
    instances, keyed as "{pdb_id}.{chain}" (RCSB's own instance ID
    format, confirmed from official docs example "1NDO.A").

    Returns {instance_id: {"pfam_family": str, "cath_classification": str}}.
    Either or both fields fall back to "unclassified" if that particular
    annotation source has no entry for this chain — CATH/Pfam coverage
    is NOT universal across the PDB, so a missing annotation is expected
    and normal, not a fetch failure.

    This is a SEPARATE query from fetch_entry_details_batch because Pfam/
    CATH assignments live on polymer_entity_instances as a
    "positional feature" list (rcsb_polymer_instance_feature), not as a
    scalar field on polymer_entities — confirmed from RCSB's own Data
    API documentation and examples, not assumed.
    """
    from rcsbapi.data import DataQuery

    query = DataQuery(
        input_type="polymer_entity_instances",
        input_ids=instance_ids,
        return_data_list=[
            "rcsb_polymer_entity_instance_container_identifiers.entry_id",
            "rcsb_polymer_entity_instance_container_identifiers.auth_asym_id",
            "rcsb_polymer_instance_feature.type",
            "rcsb_polymer_instance_feature.name",
        ],
    )
    result = query.exec()

    classifications: dict[str, dict] = {}
    instances_data = result.get("data", {}).get("polymer_entity_instances", []) or []
    for inst in instances_data:
        ids = inst.get("rcsb_polymer_entity_instance_container_identifiers", {})
        entry_id = ids.get("entry_id", "")
        auth_asym_id = ids.get("auth_asym_id", "")
        instance_key = f"{entry_id}.{auth_asym_id}"

        pfam_name = "unclassified"
        cath_name = "unclassified"
        features = inst.get("rcsb_polymer_instance_feature") or []
        for feat in features:
            feat_type = (feat.get("type") or "").upper()
            feat_name = feat.get("name")
            if not feat_name:
                continue
            if feat_type == "PFAM" and pfam_name == "unclassified":
                pfam_name = feat_name
            elif feat_type == "CATH" and cath_name == "unclassified":
                cath_name = feat_name

        classifications[instance_key] = {
            "pfam_family": pfam_name,
            "cath_classification": cath_name,
        }

    return classifications


def enrich_with_classifications(entries: list[dict], sleep_between_batches: float) -> list[dict]:
    """Takes already-filtered protein entries (post parse_entry_result)
    and fills in their pfam_family / cath_classification fields via a
    second batched Data API call. Runs AFTER filtering so we never fetch
    classification data for entries that were already rejected — no
    point spending API calls on discarded candidates."""
    if not entries:
        return entries

    instance_ids = [f"{e['pdb_id']}.{e['chain']}" for e in entries]

    all_classifications: dict[str, dict] = {}
    for i in range(0, len(instance_ids), DATA_API_BATCH_SIZE):
        batch = instance_ids[i:i + DATA_API_BATCH_SIZE]
        try:
            batch_result = fetch_classifications_batch(batch)
            all_classifications.update(batch_result)
        except Exception as e:
            print(f"  WARNING: classification batch fetch failed, leaving "
                  f"'unclassified' for this batch: {e}")
        time.sleep(sleep_between_batches)

    for entry in entries:
        instance_key = f"{entry['pdb_id']}.{entry['chain']}"
        classification = all_classifications.get(instance_key)
        if classification:
            entry["pfam_family"] = classification["pfam_family"]
            entry["cath_classification"] = classification["cath_classification"]
        # else: leave the "unclassified" defaults already set in
        # parse_entry_result — a missing lookup result (e.g. this
        # instance simply wasn't in the batch response) is treated the
        # same as "RCSB has no annotation for this chain", not an error

    return entries


def parse_entry_result(pdb_id: str, entry_data: dict, max_resolution: float,
                        min_residues: int) -> dict | None:
    """Extracts a single protein_list.yaml entry from raw Data API JSON
    for one PDB entry. Returns None if the entry doesn't have everything
    needed (no UniProt accession, no non-artifact ligand, etc.) — such
    entries are skipped, not force-included with missing data."""
    try:
        resolution = entry_data.get("rcsb_entry_info", {}).get("resolution_combined")
        if isinstance(resolution, list):
            resolution = resolution[0] if resolution else None
        if resolution is None or resolution > max_resolution:
            return None

        polymer_entities = entry_data.get("polymer_entities") or []
        nonpolymer_entities = entry_data.get("nonpolymer_entities") or []

        # Identify a real (non-artifact) ligand
        real_ligands = []
        for np_entity in nonpolymer_entities:
            comp_id = np_entity.get("pdbx_entity_nonpoly", {}).get("comp_id")
            if comp_id and comp_id.upper() not in ARTIFACT_LIGAND_CODES:
                real_ligands.append(comp_id)
        if not real_ligands:
            return None

        # Find the first polymer entity with a UniProt cross-reference
        # and use it as the target chain/protein for this entry.
        for pe in polymer_entities:
            ids = pe.get("rcsb_polymer_entity_container_identifiers", {})
            ref_ids = ids.get("reference_sequence_identifiers") or []
            uniprot_id = None
            for ref in ref_ids:
                if ref.get("database_name") == "UniProt":
                    uniprot_id = ref.get("database_accession")
                    break
            if uniprot_id is None:
                continue

            auth_chains = ids.get("auth_asym_ids") or []
            if not auth_chains:
                continue
            chain = auth_chains[0]  # first author chain for this entity

            seq_length = pe.get("entity_poly", {}).get("rcsb_sample_sequence_length")
            if seq_length is None or seq_length < min_residues:
                continue

            description = pe.get("rcsb_polymer_entity", {}).get("pdbx_description", "")
            organism = ""
            src = pe.get("rcsb_entity_source_organism")
            if src:
                organism = src[0].get("scientific_name", "") if isinstance(src, list) else src.get("scientific_name", "")

            gene_name = (description or pdb_id).split(",")[0].strip()[:40]

            return {
                "uniprot_id": uniprot_id,
                "gene_name": gene_name,
                "pdb_id": pdb_id,
                "chain": chain,
                "pfam_family": "unclassified",       # filled in by
                "cath_classification": "unclassified",  # fetch_classifications_batch()
                "max_resolution": round(float(resolution) + 0.1, 2),  # small buffer
                                            # above the OBSERVED resolution, same
                                            # pattern used to fix BRAF/ALDH2/etc.
                                            # earlier, but computed from real data
                                            # this time instead of guessed
                "expect_ligand": True,
                "min_residues": min_residues,
                "notes": f"auto-selected via RCSB API; organism={organism or 'unknown'}; "
                         f"ligand(s)={','.join(real_ligands)}",
            }

        return None
    except (KeyError, TypeError, IndexError) as e:
        print(f"  WARNING: could not parse entry {pdb_id}: {e}")
        return None


def load_existing_pdb_ids() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    with open(OUTPUT_PATH) as f:
        data = yaml.safe_load(f) or {}
    return {p["pdb_id"] for p in data.get("proteins", [])}


def append_entries(new_entries: list[dict]):
    """Writes entries to protein_list.yaml, merging with existing content
    (if any) rather than overwriting — supports incremental/resumable runs."""
    existing = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            data = yaml.safe_load(f) or {}
        existing = data.get("proteins", [])

    combined = existing + new_entries
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump({"proteins": combined}, f, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_count", type=int, default=500,
                         help="number of proteins to select")
    parser.add_argument("--max_resolution", type=float, default=DEFAULT_MAX_RESOLUTION)
    parser.add_argument("--min_residues", type=int, default=DEFAULT_MIN_RESIDUES)
    parser.add_argument("--organism", type=str, default="Homo sapiens",
                         help="restrict to this organism; pass '' to disable")
    parser.add_argument("--resume", action="store_true",
                         help="skip PDB IDs already present in protein_list.yaml")
    parser.add_argument("--sleep_between_batches", type=float, default=0.5,
                         help="seconds to wait between Data API batch calls, "
                              "as a courtesy to RCSB's servers even though no "
                              "hard rate limit is documented")
    args = parser.parse_args()

    organism = args.organism if args.organism else None

    try:
        candidate_ids = search_candidate_pdb_ids(
            args.max_resolution, args.target_count * 2, organism  # over-fetch;
            # many candidates will be filtered out in parsing (no UniProt xref,
            # only artifact ligands, etc.), so request more than target_count
        )
    except Exception as e:
        print(f"ERROR: Search API query failed: {e}")
        print("This script requires a live network connection to search.rcsb.org.")
        sys.exit(1)

    if args.resume:
        already_have = load_existing_pdb_ids()
        before = len(candidate_ids)
        candidate_ids = [pid for pid in candidate_ids if pid not in already_have]
        print(f"--resume: skipping {before - len(candidate_ids)} already-present PDB IDs")

    n_selected = 0
    n_batches_processed = 0

    for i in range(0, len(candidate_ids), DATA_API_BATCH_SIZE):
        if n_selected >= args.target_count:
            break

        batch_ids = candidate_ids[i:i + DATA_API_BATCH_SIZE]
        print(f"\nFetching Data API batch {n_batches_processed + 1} ({len(batch_ids)} IDs)...")

        try:
            batch_result = fetch_entry_details_batch(batch_ids)
        except Exception as e:
            print(f"  WARNING: batch fetch failed, skipping this batch: {e}")
            continue

        entries_data = batch_result.get("data", {}).get("entries", [])
        batch_entries = []
        for entry_data in entries_data:
            pdb_id = entry_data.get("rcsb_id", "UNKNOWN")
            parsed = parse_entry_result(pdb_id, entry_data, args.max_resolution, args.min_residues)
            if parsed is not None:
                batch_entries.append(parsed)
                n_selected += 1
                if n_selected >= args.target_count:
                    break

        if batch_entries:
            print(f"  Fetching Pfam/CATH classification for {len(batch_entries)} entries...")
            batch_entries = enrich_with_classifications(batch_entries, args.sleep_between_batches)
            append_entries(batch_entries)
            print(f"  Added {len(batch_entries)} proteins (total so far: {n_selected})")

        n_batches_processed += 1
        time.sleep(args.sleep_between_batches)

    print(f"\n{'=' * 50}")
    print(f"Selection complete: {n_selected} proteins written to {OUTPUT_PATH}")
    print(f"{'=' * 50}")
    print(
        "\nNOTE: pfam_family / cath_classification are set to 'unclassified' "
        "for entries where RCSB has no such annotation for that chain — this "
        "is expected (CATH/Pfam coverage is not universal across the PDB), "
        "not a fetch error. Spot-check a few entries to confirm the fields "
        "look sane before trusting the full batch."
    )
    print(
        "Next steps: run verify_and_build_index.py to validate structures "
        "against real downloaded .cif files (still required — this script "
        "only confirms proteins MATCH your query criteria in RCSB's metadata, "
        "not that the actual downloaded file parses cleanly)."
    )


if __name__ == "__main__":
    main()
