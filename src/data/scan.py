"""Inspect the extracted dataset and build the file index (Phase 1).

    python -m src.data.scan                    # uses configs/local.yaml / DFF_ROOT
    python -m src.data.scan --workers 8

For every image in wiki/, insight/, text2img/, inpainting/ it records:
  size in bytes, SHA-256 (exact duplicates + teammate verification), image format, mode,
  width/height, whether it decodes fully (corruption check) and a 64-bit perceptual hash
  (pHash; near-duplicate detection).

Outputs (committed, small):
  data/metadata/file_index.csv.gz   one row per file, paths relative to the dataset root
  data/reports/scan_summary.json    counts, file types, dimensions, corrupted files

The scan is resumable: finished rows are cached in outputs/cache/scan_cache.csv, so an
interrupted scan (or --max-seconds) continues where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ..paths import (
    EXPECTED_FILES_PER_SOURCE,
    METADATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SOURCES,
    resolve_dataset_root,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLUMNS = ["relpath", "bytes", "sha256", "format", "mode", "width", "height", "ok", "error", "phash"]
CACHE = PROJECT_ROOT / "outputs" / "cache" / "scan_cache.csv"
INDEX_PATH = METADATA_DIR / "file_index.csv.gz"


def list_files(root: Path) -> list[str]:
    """All files under the four source folders, as sorted posix relpaths."""
    # os.walk uses directory-entry types (no per-file stat), which is much faster than
    # Path.rglob + is_file() on network drives, synced folders and Windows.
    out = []
    for source in SOURCES:
        folder = root / source
        if not folder.is_dir():
            print(f"[warn] missing folder: {folder}")
            continue
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            out.extend(f"{rel_dir}/{name}" for name in filenames)
    return sorted(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_file(args: tuple[str, str]) -> dict:
    """Worker: inspect one file. Must be module-level (picklable on Windows)."""
    from PIL import Image
    import imagehash

    root, rel = args
    path = Path(root) / rel
    row = {c: "" for c in COLUMNS}
    row["relpath"] = rel
    try:
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
        with Image.open(path) as im:
            row["format"], row["mode"] = im.format, im.mode
            row["width"], row["height"] = im.size
            im.load()  # full decode: catches truncated/corrupted files
            row["phash"] = str(imagehash.phash(im.convert("RGB")))
        row["ok"] = 1
    except Exception as exc:  # corrupted / unreadable file
        row["ok"] = 0
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def _valid_row(r: dict) -> bool:
    if r.get("ok") == "0":
        return bool(r.get("relpath"))
    return r.get("ok") == "1" and len(r.get("sha256", "")) == 64 and len(r.get("phash", "")) == 16


def load_cache() -> dict[str, dict]:
    """Load finished rows. Rows cut off by an interruption are dropped (and re-scanned),
    and the cache is rewritten cleanly so new rows are never appended to a partial line."""
    if not CACHE.exists():
        return {}
    df = pd.read_csv(CACHE, dtype=str, keep_default_na=False, on_bad_lines="skip")
    rows = [r for r in df.to_dict("records") if _valid_row(r)]
    with open(CACHE, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        ends_clean = True
        if size > 0:
            f.seek(-1, os.SEEK_END)
            ends_clean = f.read(1) == b"\n"
    if len(rows) != len(df) or not ends_clean:
        print(f"[cache] dropping {len(df) - len(rows)} incomplete row(s); rewriting cache")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(CACHE, index=False)
    return {r["relpath"]: r for r in rows}


def summarise(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["source"] = df["relpath"].str.split("/").str[0]
    df["ext"] = df["relpath"].str.extract(r"(\.[^./]+)$")[0].str.lower()
    summary = {"total_files": int(len(df)), "expected_per_source": EXPECTED_FILES_PER_SOURCE, "sources": {}}
    for source, g in df.groupby("source"):
        ok = g[g["ok"].astype(int) == 1]
        sizes = Counter(zip(ok["width"].astype(int), ok["height"].astype(int)))
        summary["sources"][source] = {
            "files": int(len(g)),
            "extensions": dict(Counter(g["ext"])),
            "formats": dict(Counter(ok["format"])),
            "modes": dict(Counter(ok["mode"])),
            "corrupted": int((g["ok"].astype(int) == 0).sum()),
            "corrupted_files": g.loc[g["ok"].astype(int) == 0, ["relpath", "error"]].to_dict("records")[:50],
            "width_min_max": [int(ok["width"].astype(int).min()), int(ok["width"].astype(int).max())] if len(ok) else None,
            "height_min_max": [int(ok["height"].astype(int).min()), int(ok["height"].astype(int).max())] if len(ok) else None,
            "most_common_sizes": [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(5)],
            "distinct_sizes": len(sizes),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=None, help="Stop after this long (resume later)")
    args = ap.parse_args(argv)

    root = resolve_dataset_root(args.dataset_root)
    files = list_files(root)
    print(f"Found {len(files)} files under {root}")
    cache = load_cache()
    todo = [f for f in files if f not in cache]
    print(f"{len(files) - len(todo)} already scanned (cache), {len(todo)} to go")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CACHE.exists()
    start, done = time.time(), 0
    with open(CACHE, "a", newline="", encoding="utf-8") as fh, Pool(max(1, args.workers)) as pool:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row in pool.imap_unordered(inspect_file, [(str(root), f) for f in todo], chunksize=64):
            writer.writerow(row)
            cache[row["relpath"]] = row
            done += 1
            if done % 2000 == 0:
                fh.flush()
                rate = done / (time.time() - start)
                print(f"  {done}/{len(todo)}  ({rate:.0f} files/s)")
            if args.max_seconds and time.time() - start > args.max_seconds:
                pool.terminate()
                break

    if len(cache.keys() & set(files)) < len(files):
        print(f"Scan incomplete ({len(cache)}/{len(files)}). Re-run the same command to resume.")
        return 2

    df = pd.DataFrame([cache[f] for f in files], columns=COLUMNS)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_PATH, index=False, compression={"method": "gzip", "mtime": 0})
    summary = summarise(df)
    with open(REPORTS_DIR / "scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {INDEX_PATH.relative_to(PROJECT_ROOT)} and data/reports/scan_summary.json")
    for s, v in summary["sources"].items():
        print(f"  {s:11s} files={v['files']:6d} corrupted={v['corrupted']} sizes={v['most_common_sizes'][:2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
