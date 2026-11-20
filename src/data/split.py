"""Master split + per-task manifests (Phase 1).

    python -m src.data.split

Works from the committed file index (data/metadata/file_index.csv.gz), so it does not
need the images themselves. Steps:

1. Match every fake to its source wiki image (same relative path; see match.py).
2. Build leakage groups over wiki images with union-find:
     - same identity (person_id parsed from the IMDB-WIKI filename), and
     - perceptual near-duplicates (pHash distance <= split.phash_max_distance) between ANY
       two images of any type; each image stands for its source wiki image. This catches
       InsightFace swaps, whose pixels come from a different (target) wiki photo than the
       one they are named after.
3. Shuffle groups with split.seed and assign them to train/val/test (70/15/15 of images).
4. Every fake inherits the split of its source wiki image -> one MASTER split shared by
   all tasks and fake types (needed for leakage-free cross-task tests in Phase 5).
5. For each task in configs/tasks.yaml write a class-balanced manifest. With several fake
   types, the sources of each split are partitioned between fake types (e.g. 50/50), so
   each source contributes exactly one real and one fake image.

Outputs (committed):
  data/splits/master.csv           one row per wiki source image
  data/splits/tasks/<task>.csv     relpath,label,fake_type,source,split
  data/reports/matching.json, split_summary.json, identity_overlap.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from ..config import load_config, load_tasks
from ..paths import FAKE_SOURCES, METADATA_DIR, REPORTS_DIR, SPLITS_DIR
from .dedup import near_duplicate_pairs, phash_to_uint64
from .match import identity_key, identity_report, match_sources, split_source, stem_key

SPLIT_NAMES = ("train", "val", "test")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_master(index: pd.DataFrame, seed: int, ratios: dict, phash_max_distance: int) -> tuple[pd.DataFrame, dict]:
    """One row per usable wiki source image with its group and split."""
    ok = index[index["ok"].astype(str) == "1"]
    mapping, matching = match_sources(index["relpath"].tolist())
    matching["corrupted_files_excluded"] = int((index["ok"].astype(str) != "1").sum())

    wiki = ok[ok["relpath"].str.startswith("wiki/")].sort_values("relpath").reset_index(drop=True)
    wiki["source_key"] = wiki["relpath"].map(lambda rp: stem_key(split_source(rp)[1]))
    wiki["identity"] = wiki["relpath"].map(lambda rp: identity_key(split_source(rp)[1]))

    # ---- leakage groups: identity + near-duplicate wiki images
    uf = UnionFind(len(wiki))
    first_of_identity: dict[str, int] = {}
    for i, ident in enumerate(wiki["identity"]):
        if ident in first_of_identity:
            uf.union(i, first_of_identity[ident])
        else:
            first_of_identity[ident] = i
    # Near-duplicates across ALL images (any type) tie their source wiki images together.
    # Needed because an InsightFace swap is named after the identity donor (wiki A) but its
    # pixels — pose, background, body — come from a different target photo (wiki B).
    # Without this link, B could be a real test image while a near-copy of it is a
    # training fake. Fakes map to their source by name; wiki images map to themselves.
    ok_all = ok.reset_index(drop=True)
    key_to_wiki = {k: i for i, k in enumerate(wiki["source_key"])}
    wiki_idx = [key_to_wiki.get(stem_key(split_source(rp)[1])) for rp in ok_all["relpath"]]
    kinds = ok_all["relpath"].str.split("/").str[0].tolist()
    nd_pairs = near_duplicate_pairs(phash_to_uint64(ok_all["phash"]), phash_max_distance)
    links_by_type: dict[str, int] = {}
    for a, b in nd_pairs:
        ia, ib = wiki_idx[a], wiki_idx[b]
        if ia is None or ib is None or ia == ib:
            continue  # unmatched file, or a fake vs. its own source (same group anyway)
        t = "+".join(sorted((kinds[a], kinds[b])))
        links_by_type[t] = links_by_type.get(t, 0) + 1
        uf.union(ia, ib)
    roots = [uf.find(i) for i in range(len(wiki))]
    wiki["group"] = [wiki.at[r, "source_key"] for r in roots]  # group named after its first member

    # ---- assign whole groups to splits (cumulative image counts hit the target ratios)
    groups = sorted(wiki["group"].unique())
    rng = np.random.default_rng(seed)
    order = [groups[i] for i in rng.permutation(len(groups))]
    sizes = wiki["group"].value_counts()
    n = len(wiki)
    cut_train, cut_val = ratios["train"] * n, (ratios["train"] + ratios["val"]) * n
    assignment, running = {}, 0
    for g in order:
        assignment[g] = "train" if running < cut_train else ("val" if running < cut_val else "test")
        running += int(sizes[g])
    wiki["split"] = wiki["group"].map(assignment)

    # ---- attach each fake type (only fakes that decode fine)
    ok_set = set(ok["relpath"])
    for ft in FAKE_SOURCES:
        m = mapping.get(ft, {})
        wiki[ft] = wiki["source_key"].map(lambda k, m=m: m.get(k, ""))
        wiki.loc[~wiki[ft].isin(ok_set), ft] = ""

    master = wiki[["source_key", "relpath", "identity", "group", "split", *FAKE_SOURCES]].rename(
        columns={"relpath": "wiki"}
    )
    matching["phash_max_distance"] = phash_max_distance
    matching["near_duplicate_links_between_different_sources"] = links_by_type
    matching["leakage_groups"] = len(groups)
    matching["groups_with_more_than_one_source"] = int((sizes > 1).sum())
    matching["largest_group_size"] = int(sizes.max())
    matching["group_size_counts"] = {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
    return master, matching


def build_task_manifest(master: pd.DataFrame, task: dict, seed: int) -> pd.DataFrame:
    """Balanced real/fake manifest for one task, derived from the master split."""
    fake_types = list(task["fakes"].keys())
    fracs = np.array([task["fakes"][t] for t in fake_types], dtype=float)
    # Sources must have every fake type the task uses, so that the partition is unbiased.
    usable = master[(master[fake_types] != "").all(axis=1)]
    rows = []
    for split in SPLIT_NAMES:
        part = usable[usable["split"] == split].sort_values("source_key")
        n = len(part)
        rng = np.random.default_rng([seed, SPLIT_NAMES.index(split)])
        perm = part.iloc[rng.permutation(n)]
        # Partition sources between fake types (largest-remainder rounding, sums to n).
        counts = np.floor(fracs * n).astype(int)
        counts[np.argsort(-(fracs * n - counts))[: n - counts.sum()]] += 1
        start = 0
        for ft, c in zip(fake_types, counts):
            for r in perm.iloc[start:start + c].itertuples():
                rows.append((r.wiki, 0, "real", r.source_key, split))
                rows.append((getattr(r, ft), 1, ft, r.source_key, split))
            start += c
    df = pd.DataFrame(rows, columns=["relpath", "label", "fake_type", "source", "split"])
    return df.sort_values(["split", "source", "label"]).reset_index(drop=True)


def summarise(master: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> dict:
    out = {"master": master["split"].value_counts().reindex(SPLIT_NAMES).to_dict(), "tasks": {}}
    for name, df in manifests.items():
        t = {}
        for split in SPLIT_NAMES:
            s = df[df["split"] == split]
            t[split] = {
                "real": int((s["label"] == 0).sum()),
                "fake": int((s["label"] == 1).sum()),
                "fake_by_type": s.loc[s["label"] == 1, "fake_type"].value_counts().to_dict(),
            }
        out["tasks"][name] = t
    return out


def identity_overlap(master: pd.DataFrame, ident_report: dict) -> dict:
    per_identity = master.groupby("identity")["split"].nunique()
    per_group = master.groupby("group")["split"].nunique()
    return {
        **ident_report,
        "identities_in_more_than_one_split": int((per_identity > 1).sum()),
        "leakage_groups_in_more_than_one_split": int((per_group > 1).sum()),
        "limitation": (
            "Identity = IMDB-WIKI person_id from the filename. The same celebrity under a "
            "different person_id cannot be detected from filenames; visually near-identical "
            "photos are still caught by the pHash grouping."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = load_config()
    sp = cfg["split"]
    index = pd.read_csv(METADATA_DIR / "file_index.csv.gz", dtype=str, keep_default_na=False)

    master, matching = build_master(index, sp["seed"], sp["ratios"], sp["phash_max_distance"])
    manifests = {
        name: build_task_manifest(master, task, sp["seed"]) for name, task in load_tasks().items()
    }

    (SPLITS_DIR / "tasks").mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(SPLITS_DIR / "master.csv", index=False)
    for name, df in manifests.items():
        df.to_csv(SPLITS_DIR / "tasks" / f"{name}.csv", index=False)

    wiki_paths = index.loc[index["relpath"].str.startswith("wiki/"), "relpath"].tolist()
    reports = {
        "matching.json": matching,
        "split_summary.json": {"seed": sp["seed"], "ratios": sp["ratios"], **summarise(master, manifests)},
        "identity_overlap.json": identity_overlap(master, identity_report(wiki_paths)),
    }
    for fname, rep in reports.items():
        with open(REPORTS_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    print(json.dumps(reports["split_summary.json"], indent=2))
    print("identity overlap:", json.dumps(reports["identity_overlap.json"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
