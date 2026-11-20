"""Matching, splitting, balancing and duplicate detection on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_tasks
from src.data.dedup import near_duplicate_pairs
from src.data.match import identity_key, match_sources, parse_wiki_name
from src.data.split import build_master, build_task_manifest

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_parse_wiki_name():
    p = parse_wiki_name("00/10049200_1891-09-16_1958.jpg")
    assert p == {"person_id": "10049200", "dob": "1891-09-16", "year_taken": "1958"}
    assert identity_key("00/10049200_1891-09-16_1958.jpg") == "pid:10049200"
    assert identity_key("00/weird.jpg").startswith("file:")


def test_matching_reports_unmatched(synthetic_index):
    _, report = match_sources(synthetic_index["relpath"].tolist())
    assert report["fake_types"]["insight"]["matched"] == 40
    assert report["fake_types"]["insight"]["unmatched"] == 1
    assert report["fake_types"]["text2img"]["unmatched"] == 0


def test_master_split_inheritance_and_ratios(synthetic_index):
    master, _ = build_master(synthetic_index, seed=1, ratios=RATIOS, phash_max_distance=4)
    assert len(master) == 40
    assert set(master["split"]) == {"train", "val", "test"}
    assert master["split"].value_counts()["train"] >= 26
    # deterministic
    master2, _ = build_master(synthetic_index, seed=1, ratios=RATIOS, phash_max_distance=4)
    pd.testing.assert_frame_equal(master, master2)
    # groups never straddle splits
    assert (master.groupby("group")["split"].nunique() == 1).all()


def test_task_manifests_balanced_and_leak_free(synthetic_index):
    master, _ = build_master(synthetic_index, seed=1, ratios=RATIOS, phash_max_distance=4)
    src_split = dict(zip(master["source_key"], master["split"]))
    for name, task in load_tasks().items():
        df = build_task_manifest(master, task, seed=1)
        for split, g in df.groupby("split"):
            assert (g["label"] == 0).sum() == (g["label"] == 1).sum(), (name, split)
        # every image (real or fake) is in the split of its source wiki image
        assert (df["source"].map(src_split) == df["split"]).all()
        # each source contributes at most one fake
        assert df[df["label"] == 1]["source"].is_unique
    comb = build_task_manifest(master, load_tasks()["combined"], seed=1)
    counts = comb[comb["label"] == 1]["fake_type"].value_counts()
    assert abs(counts["insight"] - counts["text2img"]) <= 3  # 50/50 up to per-split rounding


def test_near_duplicate_pairs_bruteforce_agreement():
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**63, size=300, dtype=np.int64).astype(np.uint64)
    hashes[10] = hashes[5] ^ np.uint64(0b1011)  # distance 3
    hashes[20] = hashes[5]                      # distance 0
    found = near_duplicate_pairs(hashes, 4)
    brute = {
        (i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
        if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= 4
    }
    assert found == brute
    assert (5, 10) in found and (5, 20) in found


def test_near_duplicates_are_grouped(synthetic_index):
    idx = synthetic_index.copy()
    wiki = idx[idx["relpath"].str.startswith("wiki/")].index
    idx.loc[wiki[1], "phash"] = idx.loc[wiki[0], "phash"]  # make two wiki images near-identical
    master, info = build_master(idx, seed=3, ratios=RATIOS, phash_max_distance=4)
    a, b = idx.loc[wiki[0], "relpath"], idx.loc[wiki[1], "relpath"]
    ga = master.loc[master["wiki"] == a, "group"].item()
    gb = master.loc[master["wiki"] == b, "group"].item()
    assert ga == gb and info["near_duplicate_links_between_different_sources"].get("wiki+wiki", 0) >= 1


def test_faceswap_target_photo_is_grouped_with_its_donor(synthetic_index):
    """An insight fake named after wiki A whose pixels copy wiki B must tie A and B together."""
    idx = synthetic_index.copy()
    wiki = idx[idx["relpath"].str.startswith("wiki/")]
    a, b = wiki["relpath"].iloc[2], wiki["relpath"].iloc[7]
    fake_a = a.replace("wiki/", "insight/", 1)
    idx.loc[idx["relpath"] == fake_a, "phash"] = idx.loc[idx["relpath"] == b, "phash"].item()
    master, info = build_master(idx, seed=5, ratios=RATIOS, phash_max_distance=6)
    group = dict(zip(master["wiki"], master["group"]))
    assert group[a] == group[b]
    assert info["near_duplicate_links_between_different_sources"].get("insight+wiki", 0) >= 1
