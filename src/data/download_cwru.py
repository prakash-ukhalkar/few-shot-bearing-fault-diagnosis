"""Download the CWRU 12kHz Drive-End bearing dataset subset.

No credentials/API key required — files are served publicly by the Case
Western Reserve University Bearing Data Center. Downloads are deterministic:
re-running this script re-fetches the exact same fixed set of file numbers
defined in `src/data/cwru_manifest.py`.

Usage:
    python -m src.data.download_cwru --out data/raw/cwru
"""
import argparse
import time
from pathlib import Path

import requests

from src.data.cwru_manifest import build_index, file_number_to_url


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as e:
            print(f"  attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/cwru", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_index()
    print(f"Downloading {len(index)} CWRU .mat files to {out_dir} ...")
    failures = []
    for filename, meta in index.items():
        dest = out_dir / filename
        url = file_number_to_url(meta["file_number"])
        print(f"[{meta['label']} @ {meta['load_hp']}HP] {url} -> {dest.name}")
        ok = download_file(url, dest)
        if not ok:
            failures.append(filename)

    if failures:
        print(f"\nWARNING: {len(failures)} files failed to download: {failures}")
        print("Retry the script, or check https://engineering.case.edu/bearingdatacenter for mirrors.")
    else:
        print("\nAll files downloaded successfully.")


if __name__ == "__main__":
    main()
