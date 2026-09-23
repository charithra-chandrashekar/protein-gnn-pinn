"""
make_splits.py

Generates data/splits/{train,val,test}.txt from data/processed/index.yaml.

CRITICAL: splits by UNIQUE UniProt ID, not by structure entry. index.yaml
can contain multiple structures of the same protein (e.g. 38 different
Thrombin crystal structures) — if those were split naively at the
structure level, the same protein could end up in both train and test,
so the model would be "tested" on a protein it already learned from
via a different crystal form. This was found to be a real, serious
issue in this project: with structure-level splitting, ~68% of the
test set shared a UniProt ID with something in train. Splitting by
UNIQUE PROTEIN first, then assigning ALL of that protein's structures
to the same split, eliminates this class of leakage entirely.

Each output file lists UniProt IDs (one per line) — matching the
anchor-identity design from Phase 0 and what build_dataset.py already
expects (it looks up ALL index.yaml entries for each UniProt ID in a
split, so multiple structures per protein are still all included).

Re-run this script (it's deterministic, same seed) whenever index.yaml
grows.

Run from the project root:
    python src/data/make_splits.py
    python src/data/make_splits.py --seed 123 --train_frac 0.75 --val_frac 0.13 --test_frac 0.12
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.yaml"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_frac", type=float, default=0.75,
                         help="fraction of UNIQUE proteins for train split")
    parser.add_argument("--val_frac", type=float, default=0.13,
                         help="fraction of UNIQUE proteins for val split")
    parser.add_argument("--test_frac", type=float, default=0.12,
                         help="fraction of UNIQUE proteins for test split "
                              "(remainder after train/val)")
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} not found. Run verify_and_build_index.py first.")
        sys.exit(1)

    with open(INDEX_PATH) as f:
        proteins = yaml.safe_load(f)["proteins"]

    # Group structure entries by unique UniProt ID FIRST — this is the
    # fix. Splitting happens on the GROUPS (unique proteins), never on
    # individual structure entries.
    structures_by_uniprot: dict[str, list] = defaultdict(list)
    for p in proteins:
        structures_by_uniprot[p["uniprot_id"]].append(p)

    unique_uniprot_ids = list(structures_by_uniprot.keys())
    n_unique = len(unique_uniprot_ids)
    n_structures = len(proteins)

    if n_structures > n_unique:
        print(
            f"NOTE: {n_structures} total structures map to only {n_unique} "
            f"unique proteins ({n_structures - n_unique} are additional "
            f"structures of an already-represented protein). Splitting by "
            f"unique protein, not by structure, to avoid leakage."
        )

    rng = random.Random(args.seed)
    shuffled = unique_uniprot_ids.copy()
    rng.shuffle(shuffled)

    n_train = round(n_unique * args.train_frac)
    n_val = round(n_unique * args.val_frac)
    n_test = n_unique - n_train - n_val  # remainder, guarantees no protein dropped

    if n_test < 0:
        print(
            f"ERROR: train_frac + val_frac ({args.train_frac + args.val_frac}) "
            f"exceeds 1.0 — nothing left for test. Adjust the fractions."
        )
        sys.exit(1)

    train_uids = shuffled[:n_train]
    val_uids = shuffled[n_train:n_train + n_val]
    test_uids = shuffled[n_train + n_val:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, uids in [("train", train_uids), ("val", val_uids), ("test", test_uids)]:
        out_path = SPLITS_DIR / f"{name}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(uids) + "\n")
        n_structures_in_split = sum(len(structures_by_uniprot[u]) for u in uids)
        print(
            f"{name}: {len(uids)} unique proteins "
            f"({n_structures_in_split} total structures) -> {out_path}"
        )

    # Explicit leakage self-check before finishing — this is the exact
    # bug this script was rewritten to prevent, so verify it directly
    # rather than just trusting the grouping logic.
    train_set, val_set, test_set = set(train_uids), set(val_uids), set(test_uids)
    assert not (train_set & val_set), "BUG: train/val overlap after fix"
    assert not (train_set & test_set), "BUG: train/test overlap after fix"
    assert not (val_set & test_set), "BUG: val/test overlap after fix"
    print("\nLeakage self-check passed: no UniProt ID appears in more than one split.")

    if min(n_train, n_val, n_test) == 0:
        print(
            "\nNOTE: at least one split is empty. That's fine for a smoke test "
            "but any metric computed on an empty split is meaningless — check "
            "before drawing conclusions from val/test results."
        )


if __name__ == "__main__":
    main()
