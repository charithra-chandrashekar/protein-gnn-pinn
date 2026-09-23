"""
download_structures.py

Phase 0 (scaled), step 2: downloads .cif files for every PDB ID in
configs/protein_list.yaml, into data/raw/pdb/ — this is the missing
step between select_proteins.py (which only fetches METADATA) and
verify_and_build_index.py (which needs the actual structure files on
disk).

Uses RCSB's direct file-download service (files.rcsb.org), confirmed
from RCSB's own documentation to be intended for exactly this kind of
programmatic, batch use — NOT the Search/Data APIs used elsewhere in
this pipeline, which only return metadata, not structure files.

URL pattern (confirmed from RCSB docs): 
    https://files.rcsb.org/download/{PDB_ID}.cif

Resumable: any PDB ID whose .cif already exists in data/raw/pdb/ is
skipped, so an interrupted run (network drop, closed terminal, etc.)
can just be re-run rather than starting over. This matters at
hundreds-of-files scale.

Run from the project root:
    python src/data/download_structures.py
    python src/data/download_structures.py --sleep 0.2   # faster, still polite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTEIN_LIST_PATH = PROJECT_ROOT / "configs" / "protein_list.yaml"
RAW_PDB_DIR = PROJECT_ROOT / "data" / "raw" / "pdb"
FAILED_LOG_PATH = PROJECT_ROOT / "data" / "raw" / "download_failures.txt"

RCSB_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.cif"

DEFAULT_SLEEP_SECONDS = 0.3  # light courtesy delay between requests;
                              # RCSB's file service is documented as not
                              # rate-limited for programmatic use, but a
                              # small delay is still good practice and
                              # costs little time at this batch size


def load_pdb_ids() -> list[str]:
    if not PROTEIN_LIST_PATH.exists():
        print(f"ERROR: {PROTEIN_LIST_PATH} not found. Run select_proteins.py first.")
        sys.exit(1)
    with open(PROTEIN_LIST_PATH) as f:
        proteins = yaml.safe_load(f)["proteins"]
    # de-duplicate while preserving order — multiple proteins in the list
    # could in principle reference the same PDB ID via different chains
    seen = set()
    pdb_ids = []
    for p in proteins:
        pdb_id = p["pdb_id"].upper()
        if pdb_id not in seen:
            seen.add(pdb_id)
            pdb_ids.append(pdb_id)
    return pdb_ids


def already_downloaded(pdb_id: str) -> bool:
    return (RAW_PDB_DIR / f"{pdb_id}.cif").exists() or (RAW_PDB_DIR / f"{pdb_id.lower()}.cif").exists()


def download_one(pdb_id: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    url = RCSB_DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id)
    dest = RAW_PDB_DIR / f"{pdb_id}.cif"
    tmp_dest = RAW_PDB_DIR / f"{pdb_id}.cif.partial"

    try:
        urlretrieve(url, tmp_dest)
    except HTTPError as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False, f"HTTP {e.code} for {url}"
    except URLError as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False, f"network error for {url}: {e.reason}"
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False, f"unexpected error for {url}: {e}"

    # basic sanity check: a real .cif should be more than a trivial
    # size and should start with 'data_' — catches the case where RCSB
    # returns a small HTML error page with a 200 status instead of a
    # real structure file (rare, but worth guarding against silently
    # writing garbage into data/raw/pdb/)
    try:
        content = tmp_dest.read_text(errors="ignore")
    except Exception as e:
        tmp_dest.unlink(missing_ok=True)
        return False, f"could not read downloaded file for {pdb_id}: {e}"

    if len(content) < 100 or not content.lstrip().startswith("data_"):
        tmp_dest.unlink(missing_ok=True)
        return False, f"downloaded content for {pdb_id} does not look like a valid .cif file"

    tmp_dest.rename(dest)
    return True, f"downloaded {dest.name} ({len(content)} bytes)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS,
                         help="seconds to wait between download requests")
    args = parser.parse_args()

    RAW_PDB_DIR.mkdir(parents=True, exist_ok=True)

    pdb_ids = load_pdb_ids()
    print(f"Found {len(pdb_ids)} unique PDB ID(s) in {PROTEIN_LIST_PATH}")

    to_download = [pid for pid in pdb_ids if not already_downloaded(pid)]
    n_already_present = len(pdb_ids) - len(to_download)
    if n_already_present:
        print(f"{n_already_present} file(s) already present in {RAW_PDB_DIR}, skipping those")
    print(f"Downloading {len(to_download)} file(s)...\n")

    n_success = 0
    n_failed = 0
    failures = []

    for i, pdb_id in enumerate(to_download, 1):
        success, message = download_one(pdb_id)
        status = "OK" if success else "FAIL"
        print(f"  [{i}/{len(to_download)}] {status}: {pdb_id} - {message}")

        if success:
            n_success += 1
        else:
            n_failed += 1
            failures.append(f"{pdb_id}: {message}")

        time.sleep(args.sleep)

    print(f"\n{'=' * 50}")
    print(f"Download summary")
    print(f"{'=' * 50}")
    print(f"Already present: {n_already_present}")
    print(f"Newly downloaded: {n_success}")
    print(f"Failed: {n_failed}")
    print(f"Total available in {RAW_PDB_DIR}: {n_already_present + n_success}/{len(pdb_ids)}")

    if failures:
        with open(FAILED_LOG_PATH, "w") as f:
            f.write("\n".join(failures) + "\n")
        print(f"\n{n_failed} failure(s) logged to {FAILED_LOG_PATH}")
        print(
            "Common causes: obsolete/superseded PDB ID, temporary network "
            "issue, or a typo carried over from select_proteins.py's output. "
            "Re-running this script will retry only the missing ones "
            "(already-downloaded files are skipped)."
        )

    print(
        "\nNext step: run verify_and_build_index.py to validate these "
        "downloaded files against protein_list.yaml's expectations."
    )


if __name__ == "__main__":
    main()
