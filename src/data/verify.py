"""Check a local dataset copy against the committed manifests BEFORE training.

    python -m src.data.verify              # existence + per-folder file counts
    python -m src.data.verify --hashes     # also compare SHA-256 of every file (slower)

Reports missing files, unexpected extra files, file-count differences per folder and
(optionally) content mismatches. Exit code 0 = consistent, 1 = mismatch found.
Training runs a quick existence check for its task automatically (see train.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..paths import METADATA_DIR, SOURCES, SPLITS_DIR, resolve_dataset_root
from .scan import list_files, sha256_file


def missing_files(relpaths, local: set[str]) -> list[str]:
    return [rp for rp in relpaths if rp not in local]


def check_task_files(root: Path, task: str, local: set[str] | None = None) -> list[str]:
    """Quick check used before training: files referenced by one task manifest exist.

    Lists the dataset folders once (fast) instead of stat-ing every file.
    """
    df = pd.read_csv(SPLITS_DIR / "tasks" / f"{task}.csv", usecols=["relpath"])
    return missing_files(df["relpath"], local if local is not None else set(list_files(root)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--hashes", action="store_true", help="Compare SHA-256 against the file index")
    args = ap.parse_args(argv)

    root = resolve_dataset_root(args.dataset_root)
    print(f"Dataset root: {root}")
    problems = 0

    index_path = METADATA_DIR / "file_index.csv.gz"
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    expected = set(index["relpath"])
    local = set(list_files(root))

    # 1) per-folder counts
    print("\nFile counts (local vs. committed index):")
    for s in SOURCES:
        n_exp = sum(1 for p in expected if p.startswith(s + "/"))
        n_loc = sum(1 for p in local if p.startswith(s + "/"))
        flag = "OK" if n_exp == n_loc else "MISMATCH"
        problems += flag != "OK"
        print(f"  {s:11s} local={n_loc:6d} expected={n_exp:6d}  {flag}")

    # 2) missing / extra files
    missing, extra = sorted(expected - local), sorted(local - expected)
    print(f"\nMissing files: {len(missing)}   Extra files: {len(extra)}")
    for p in missing[:10]:
        print(f"  missing: {p}")
    for p in extra[:10]:
        print(f"  extra:   {p}")
    problems += bool(missing) + bool(extra)

    # 3) every manifest entry exists
    for path in sorted((SPLITS_DIR / "tasks").glob("*.csv")):
        n_missing = len(check_task_files(root, path.stem, local))
        print(f"Task manifest {path.stem:10s}: {'OK' if n_missing == 0 else f'{n_missing} files missing'}")
        problems += n_missing > 0

    # 4) optional content hashes
    if args.hashes:
        print("\nHashing files (this takes a few minutes)...")
        sha = dict(zip(index["relpath"], index["sha256"]))
        bad = [rp for rp in sorted(expected & local) if sha.get(rp) and sha256_file(root / rp) != sha[rp]]
        print(f"Content mismatches: {len(bad)}")
        for p in bad[:10]:
            print(f"  differs: {p}")
        problems += bool(bad)

    print("\nRESULT:", "consistent with committed manifests" if problems == 0 else "MISMATCH — fix before training")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
