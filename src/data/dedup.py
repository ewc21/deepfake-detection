"""Exact and near-duplicate detection.

Exact duplicates: identical SHA-256.
Near duplicates: 64-bit pHash with Hamming distance <= max_distance.

Near-duplicate search avoids the O(n^2) all-pairs comparison with the pigeonhole trick:
the 64-bit hash is cut into 8 bytes; two hashes that differ in <= 7 bits must agree
exactly on at least one byte. We bucket by (byte position, byte value) and only compare
inside buckets. Exact for max_distance <= 7.

CLI (run after src.data.split):
    python -m src.data.dedup
writes data/reports/duplicates.json and data/reports/duplicates_cross_split.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from ..paths import METADATA_DIR, REPORTS_DIR, SPLITS_DIR
from .match import split_source, stem_key

N_BANDS = 8


def phash_to_uint64(hex_hashes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    # fallback: byte-wise lookup table
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def near_duplicate_pairs(hashes: np.ndarray, max_distance: int) -> set[tuple[int, int]]:
    """All index pairs (i < j) whose pHash Hamming distance is <= max_distance."""
    if max_distance > N_BANDS - 1:
        raise ValueError(f"max_distance must be <= {N_BANDS - 1} for exact band search")
    pairs: set[tuple[int, int]] = set()
    idx_all = np.arange(len(hashes))
    for band in range(N_BANDS):
        keys = (hashes >> np.uint64(8 * band)) & np.uint64(0xFF)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        bounds = np.flatnonzero(np.diff(sorted_keys)) + 1
        for bucket in np.split(idx_all[order], bounds):
            if len(bucket) < 2:
                continue
            h = hashes[bucket]
            dist = _popcount(h[:, None] ^ h[None, :])
            ii, jj = np.nonzero(np.triu(dist <= max_distance, k=1))
            for a, b in zip(bucket[ii], bucket[jj]):
                pairs.add((int(min(a, b)), int(max(a, b))))
    return pairs


def exact_duplicate_groups(sha: pd.Series) -> list[list[int]]:
    groups = defaultdict(list)
    for i, h in enumerate(sha):
        if h:
            groups[h].append(i)
    return [g for g in groups.values() if len(g) > 1]


def file_splits(index: pd.DataFrame, master: pd.DataFrame) -> pd.Series:
    """Split of every file = split of its source wiki image (fakes inherit)."""
    src_split = dict(zip(master["source_key"], master["split"]))
    keys = index["relpath"].map(lambda rp: stem_key(split_source(rp)[1]))
    return keys.map(src_split)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-distance", type=int, default=None, help="pHash distance (default from config)")
    args = ap.parse_args(argv)

    from ..config import load_config

    cfg = load_config()
    max_d = args.max_distance if args.max_distance is not None else cfg["split"]["phash_max_distance"]
    index = pd.read_csv(METADATA_DIR / "file_index.csv.gz", dtype=str, keep_default_na=False)
    master = pd.read_csv(SPLITS_DIR / "master.csv", dtype=str, keep_default_na=False)
    index = index[index["ok"] == "1"].reset_index(drop=True)
    index["source"] = index["relpath"].str.split("/").str[0]
    index["split"] = file_splits(index, master)

    # ---- exact duplicates
    ex_groups = exact_duplicate_groups(index["sha256"])
    ex_cross = [g for g in ex_groups if index.loc[g, "split"].nunique() > 1]

    # ---- near duplicates (all images, all fake types)
    pairs = near_duplicate_pairs(phash_to_uint64(index["phash"]), max_d)
    rows = []
    for i, j in pairs:
        si, sj = index.at[i, "split"], index.at[j, "split"]
        rows.append({
            "a": index.at[i, "relpath"], "b": index.at[j, "relpath"],
            "split_a": si, "split_b": sj,
            "type_pair": "+".join(sorted([index.at[i, "source"], index.at[j, "source"]])),
            "cross_split": si != sj,
        })
    pairs_df = pd.DataFrame(rows, columns=["a", "b", "split_a", "split_b", "type_pair", "cross_split"])
    cross = pairs_df[pairs_df["cross_split"]] if len(pairs_df) else pairs_df

    # Per task: only pairs where both images are in that task's manifest matter.
    per_task = {}
    for path in sorted((SPLITS_DIR / "tasks").glob("*.csv")):
        members = set(pd.read_csv(path, usecols=["relpath"])["relpath"])
        sub = cross[cross["a"].isin(members) & cross["b"].isin(members)] if len(cross) else cross
        ex_sub = [g for g in ex_cross if len({index.at[k, "relpath"] for k in g} & members) > 1]
        per_task[path.stem] = {"near_duplicate_pairs_across_splits": int(len(sub)),
                               "exact_duplicate_groups_across_splits": len(ex_sub)}

    report = {
        "files_checked": int(len(index)),
        "phash_max_distance": max_d,
        "exact_duplicate_groups": len(ex_groups),
        "exact_duplicate_files": int(sum(len(g) for g in ex_groups)),
        "exact_duplicate_groups_across_splits": len(ex_cross),
        "near_duplicate_pairs": int(len(pairs_df)),
        "near_duplicate_pairs_by_type": pairs_df["type_pair"].value_counts().to_dict() if len(pairs_df) else {},
        "near_duplicate_pairs_across_splits": int(len(cross)),
        "near_duplicate_pairs_across_splits_by_type": cross["type_pair"].value_counts().to_dict() if len(cross) else {},
        "per_task": per_task,
        "note": "Pairs between a wiki image and fakes made from it are expected and always share a split.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "duplicates.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cross.to_csv(REPORTS_DIR / "duplicates_cross_split.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2))
    print("per task:", json.dumps(per_task, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
