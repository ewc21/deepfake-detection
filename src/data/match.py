"""Fake -> source-wiki mapping and identity extraction.

DeepFakeFace keeps the IMDB-WIKI directory layout in every archive, so a fake image is
matched to its source by having the same relative path inside its folder:

    wiki/00/10049200_1891-09-16_1958.jpg      (real source)
    insight/00/10049200_1891-09-16_1958.jpg   (face swap made from it)

Matching uses the path WITHOUT extension, to tolerate .jpg/.png differences.

IMDB-WIKI wiki filenames are <person_id>_<date_of_birth>_<year_photo_taken>.jpg and the
sub-folder is the last two digits of person_id. person_id is used as the identity key.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath

WIKI_NAME = re.compile(r"^(?P<pid>\d+)_(?P<dob>[\d-]+)_(?P<taken>\d+)$")


def stem_key(relpath_in_source: str) -> str:
    """'00/123_1990-01-01_2010.jpg' -> '00/123_1990-01-01_2010' (posix, no extension)."""
    p = PurePosixPath(relpath_in_source)
    return str(p.with_suffix(""))


def split_source(relpath: str) -> tuple[str, str]:
    """'insight/00/x.jpg' -> ('insight', '00/x.jpg')."""
    source, _, rest = relpath.partition("/")
    return source, rest


def parse_wiki_name(relpath_in_source: str) -> dict | None:
    """Parse identity fields from a wiki filename; None if it does not follow the pattern."""
    m = WIKI_NAME.match(PurePosixPath(relpath_in_source).stem)
    if not m:
        return None
    return {"person_id": m["pid"], "dob": m["dob"], "year_taken": m["taken"]}


def identity_key(relpath_in_source: str) -> str:
    """Identity used for grouping: person_id if parseable, else the file itself."""
    parsed = parse_wiki_name(relpath_in_source)
    return f"pid:{parsed['person_id']}" if parsed else f"file:{stem_key(relpath_in_source)}"


def match_sources(relpaths: list[str]) -> tuple[dict, dict]:
    """Match every fake to its source wiki image.

    Args:
        relpaths: dataset-root-relative posix paths of ALL files (e.g. from the file index).
    Returns:
        mapping: {fake_source: {wiki_stem_key: fake_relpath}}
        report:  counts of matched / unmatched fakes and wiki images without a fake.
    """
    by_source: dict[str, dict[str, str]] = {}
    for rp in relpaths:
        source, rest = split_source(rp)
        by_source.setdefault(source, {})[stem_key(rest)] = rp

    wiki = by_source.get("wiki", {})
    mapping, report = {}, {"wiki_images": len(wiki), "fake_types": {}}
    for source, files in sorted(by_source.items()):
        if source == "wiki":
            continue
        matched = {k: v for k, v in files.items() if k in wiki}
        unmatched = sorted(v for k, v in files.items() if k not in wiki)
        mapping[source] = matched
        report["fake_types"][source] = {
            "total": len(files),
            "matched": len(matched),
            "unmatched": len(unmatched),
            "unmatched_examples": unmatched[:20],
            "wiki_without_this_fake": len(set(wiki) - set(files)),
            "handling": "unmatched fakes are excluded from all manifests",
        }
    return mapping, report


def identity_report(wiki_relpaths: list[str]) -> dict:
    """How well identities can be recovered from wiki filenames."""
    rests = [split_source(rp)[1] for rp in wiki_relpaths]
    parsed = [parse_wiki_name(r) for r in rests]
    pids = Counter(p["person_id"] for p in parsed if p)
    dobs_per_pid: dict[str, set] = {}
    for p in parsed:
        if p:
            dobs_per_pid.setdefault(p["person_id"], set()).add(p["dob"])
    folder_ok = sum(
        1 for r, p in zip(rests, parsed) if p and r.split("/")[0] == p["person_id"][-2:].zfill(2)
    )
    return {
        "wiki_images": len(rests),
        "filenames_matching_pattern": sum(p is not None for p in parsed),
        "unique_person_ids": len(pids),
        "person_ids_with_multiple_images": sum(1 for c in pids.values() if c > 1),
        "max_images_per_person_id": max(pids.values()) if pids else 0,
        "person_ids_with_conflicting_dob": sum(1 for s in dobs_per_pid.values() if len(s) > 1),
        "subfolder_equals_last_two_digits_of_id": folder_ok,
    }
