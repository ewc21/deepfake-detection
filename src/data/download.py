"""OPT-IN download of DeepFakeFace from the official Hugging Face dataset (OpenRL/DeepFakeFace).

    python -m src.data.download --dest dataset
    python -m src.data.download --dest D:/data/dff --archives wiki insight

Nothing in the pipeline calls this automatically (no downloads on import/training/eval).
Archives whose folder already exists and is non-empty are skipped. Each zip contains a
top-level folder (wiki/, insight/, ...), so extraction into --dest yields <dest>/<name>/...
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from ..paths import EXPECTED_FILES_PER_SOURCE, SOURCES

REPO_ID = "OpenRL/DeepFakeFace"
# Pinned dataset commit for reproducibility (latest commit as of 2026-11; README update only).
DEFAULT_REVISION = "eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c"


def count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(len(files) for _, _, files in os.walk(folder))


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing entries that would escape `dest` (zip-slip protection)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")]
        for m in members:
            target = (dest / m.filename).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {m.filename}")
        for i, m in enumerate(members, 1):
            zf.extract(m, dest)
            if i % 5000 == 0:
                print(f"    extracted {i}/{len(members)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="Folder to place wiki/, insight/, text2img/, inpainting/ in")
    ap.add_argument("--archives", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face commit/branch to download")
    ap.add_argument("--keep-zips", action="store_true", help="Keep the downloaded .zip files")
    args = ap.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed: pip install -r requirements.txt", file=sys.stderr)
        return 1

    dest = Path(args.dest).expanduser().resolve()
    zip_dir = dest / "_zips"
    dest.mkdir(parents=True, exist_ok=True)

    for name in args.archives:
        folder = dest / name
        n = count_files(folder)
        if n > 0:
            note = "" if n == EXPECTED_FILES_PER_SOURCE else f" (expected {EXPECTED_FILES_PER_SOURCE}!)"
            print(f"[skip] {name}: already extracted, {n} files{note}")
            continue
        print(f"[download] {name}.zip (~1 GB) from {REPO_ID}@{args.revision[:8]} ...")
        zip_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{name}.zip",
                repo_type="dataset",
                revision=args.revision,
                local_dir=zip_dir,
            )
        )
        print(f"[extract] {zip_path.name} -> {dest}")
        safe_extract(zip_path, dest)
        n = count_files(folder)
        print(f"[done] {name}: {n} files")
        if not args.keep_zips:
            zip_path.unlink(missing_ok=True)

    if zip_dir.exists() and not args.keep_zips:
        shutil.rmtree(zip_dir, ignore_errors=True)
    print(
        "\nNext: set dataset_root in configs/local.yaml (or DFF_ROOT) to\n"
        f"  {dest}\nthen run: python -m src.data.verify"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
