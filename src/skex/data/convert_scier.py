"""Map SciER sentence rows onto one research card per document.

Keeps Task, Method, and Dataset only. A value is kept only when it is a
contiguous substring of the joined sentences. test_ood stays its own split.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skex.data.convert_sciriff import INSTRUCTION, _LIST_FIELDS, _NER_MAP, _finish_card, _norm_key, document_key
from skex.data.io import load_jsonl
from skex.data.pack import pack_document

SPLITS = ("train", "dev", "test", "test_ood")


def _mentions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {field: [] for field in _LIST_FIELDS}
    for row in rows:
        for ent in row.get("ner") or []:
            if not isinstance(ent, list) or len(ent) < 2:
                continue
            text, label = ent[0], ent[1]
            field = _NER_MAP.get(_norm_key(label))
            if field and isinstance(text, str) and text.strip():
                buckets[field].append(text.strip())
    return buckets


def build_rows(src: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    src = Path(src)
    grouped: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    stats = {"read_docs": 0, "kept": 0, "skipped_empty": 0}
    for split in SPLITS:
        path = src / f"{split}.jsonl"
        if not path.is_file():
            continue
        order: list[str] = []
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for row in load_jsonl(path):
            doc = str(row.get("doc_id") or "").strip()
            if not doc:
                continue
            if doc not in by_doc:
                by_doc[doc] = []
                order.append(doc)
            by_doc[doc].append(row)
        for doc in order:
            stats["read_docs"] += 1
            sentences = [str(row.get("sentence") or "").strip() for row in by_doc[doc]]
            body = "\n".join(sentence for sentence in sentences if sentence)
            if not body:
                stats["skipped_empty"] += 1
                continue
            packed = pack_document({"abstract": body})
            card = _finish_card(_mentions(by_doc[doc]), packed)
            if card is None:
                stats["skipped_empty"] += 1
                continue
            grouped[split].append(
                {
                    "id": f"scier:{split}:{doc}",
                    "domain": "A",
                    "instruction": INSTRUCTION,
                    "input": packed,
                    "output": card,
                    "doc_id": document_key(body),
                    "source_task": "scier",
                }
            )
            stats["kept"] += 1
    return grouped, stats


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert SciER JSONL into document research cards")
    parser.add_argument("--src", type=Path, default=Path("Dataset/scier/LLM"))
    args = parser.parse_args()
    grouped, stats = build_rows(args.src)
    print(json.dumps({split: len(rows) for split, rows in grouped.items()} | stats, indent=2))
    return 0 if stats["kept"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
