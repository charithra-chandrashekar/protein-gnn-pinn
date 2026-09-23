#!/usr/bin/env python3
"""Download a subset of PDB structures."""

from __future__ import annotations

import argparse
from pathlib import Path


def download_pdb_subset(output_dir: Path, pdb_ids: list[str]) -> None:
    """Download PDB files for the given IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("PDB download not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a PDB subset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded PDB files",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=["1CRN", "1UBQ"],
        help="PDB IDs to download",
    )
    args = parser.parse_args()
    download_pdb_subset(args.output_dir, args.ids)


if __name__ == "__main__":
    main()
