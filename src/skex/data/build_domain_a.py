"""Write Domain A JSONL from SciRIFF and SciER, then copy the files.

One document id is kept in the most held-out split only. test_ood outranks
test, then dev, then train.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from skex.data.convert_scier import build_rows as build_scier
from skex.data.convert_scirex import build_rows as build_scirex
from skex.data.convert_sciriff import build_rows as build_sciriff
from skex.data.convert_sciriff import load_instances
from skex.data.io import write_jsonl
from skex.data.splits import assert_no_doc_leak, assert_no_id_leak

RANK = {"test_ood": 3, "test": 2, "dev": 1, "train": 0}
SPLITS = ("train", "dev", "test", "test_ood")


def merge_groups(parts: list[dict[str, list[dict[str, Any]]]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    best: dict[str, tuple[int, str, dict[str, Any]]] = {}
    seen = 0
    for grouped in parts:
        for split, rows in grouped.items():
            if split not in RANK:
                raise ValueError(f"unknown split {split}")
            for row in rows:
                seen += 1
                doc = row["doc_id"]
                rank = RANK[split]
                current = best.get(doc)
                if current is None or rank > current[0]:
                    best[doc] = (rank, split, row)
    grouped = {split: [] for split in SPLITS}
    for _doc, (_rank, split, row) in best.items():
        grouped[split].append(row)
    _reject_leaks(grouped)
    return grouped, seen - sum(len(rows) for rows in grouped.values())


def _reject_leaks(grouped: dict[str, list[dict[str, Any]]]) -> None:
    names = [split for split in SPLITS if grouped.get(split)]
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert_no_id_leak(grouped[left], grouped[right])
            assert_no_doc_leak(grouped[left], grouped[right])


def _copy(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = src / f"{split}.jsonl"
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def build(
    sciriff_src: Path,
    scier_src: Path,
    dest: Path,
    copy_to: Path | None = None,
    scirex_src: Path | None = None,
) -> dict[str, Any]:
    sciriff, sciriff_stats = build_sciriff(load_instances(sciriff_src))
    scier, scier_stats = build_scier(scier_src)
    parts = [sciriff, scier]
    scirex_stats: dict[str, Any] = {"status": "not on disk"}
    if scirex_src is not None and Path(scirex_src).is_dir() and any(Path(scirex_src).glob("*.jsonl")):
        scirex, scirex_stats = build_scirex(scirex_src)
        scirex_stats = {"status": "converted", **scirex_stats}
        parts.append(scirex)
    grouped, dropped = merge_groups(parts)
    dest = Path(dest)
    for split in SPLITS:
        write_jsonl(dest / f"{split}.jsonl", grouped[split])
    if copy_to is not None:
        _copy(dest, copy_to)
    counts = {split: len(grouped[split]) for split in SPLITS}
    return {
        "counts": counts,
        "dropped_cross_split": dropped,
        "sciriff": {key: sciriff_stats[key] for key in sciriff_stats if str(key).startswith("kept_") or str(key).startswith("docs_")},
        "scier": scier_stats,
        "scirex": scirex_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build decontaminated Domain A JSONL")
    parser.add_argument("--sciriff", type=Path, default=Path("Dataset/sciriff"))
    parser.add_argument("--scier", type=Path, default=Path("Dataset/scier/LLM"))
    parser.add_argument("--dest", type=Path, default=Path("data/processed/domain_a"))
    parser.add_argument("--copy-to", type=Path, default=Path("Dataset/processed/domain_a"))
    parser.add_argument("--scirex", type=Path, default=Path("Dataset/scirex/release_data"))
    args = parser.parse_args(argv)
    report = build(args.sciriff, args.scier, args.dest, args.copy_to, args.scirex)
    print(json.dumps(report, indent=2))
    counts = report["counts"]
    if counts["train"] == 0 or counts["test"] == 0:
        print("FAIL: train or test is empty", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
