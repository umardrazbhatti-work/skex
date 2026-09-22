"""Integrity checks for the local Dataset/ drop. Does not rewrite source files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from skex.data.convert_sciriff import load_instances, parse_output
from skex.data.io import load_jsonl

SCIRIFF_EXPECTED = {"train": 70521, "dev": 30736, "test": 35875}


def _sciriff(root: Path) -> dict[str, Any]:
    parquet = list((root / "sciriff").rglob("*.parquet"))
    report: dict[str, Any] = {"parquet_files": [str(path.relative_to(root)) for path in parquet]}
    if not parquet:
        report["status"] = "missing"
        return report
    counts: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    families: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    seen: set[str] = set()
    nulls = 0
    dups = 0
    json_ok = 0
    json_bad = 0
    clinical = 0
    sample: dict[str, Any] | None = None
    for row in load_instances(root / "sciriff"):
        split = row.get("_split") or "?"
        counts[split] += 1
        instance_id = str(row.get("_instance_id") or "")
        if not instance_id or instance_id in seen:
            dups += 1
        seen.add(instance_id)
        if not row.get("input") or row.get("output") in (None, ""):
            nulls += 1
        meta = row.get("metadata") or {}
        if isinstance(meta, dict):
            contexts[str(meta.get("output_context"))] += 1
            families[str(meta.get("task_family"))] += 1
            domains = meta.get("domains") or []
            if "clinical_medicine" in domains:
                clinical += 1
        task = instance_id.split(":", 1)[0]
        tasks[task] += 1
        if str(meta.get("output_context") if isinstance(meta, dict) else "") in {"json", "jsonlines"}:
            parsed = parse_output(row.get("output"))
            if parsed is None:
                json_bad += 1
            else:
                json_ok += 1
        if sample is None and task == "scierc_ner":
            sample = {
                "id": instance_id,
                "output_head": str(row.get("output"))[:500],
                "input_tail": str(row.get("input"))[-400:],
            }
    report.update(
        {
            "status": "ok" if dict(counts) == SCIRIFF_EXPECTED and nulls == 0 and dups == 0 and json_bad == 0 else "check",
            "counts": dict(counts),
            "expected": SCIRIFF_EXPECTED,
            "null_input_or_output": nulls,
            "duplicate_or_blank_ids": dups,
            "json_outputs_ok": json_ok,
            "json_outputs_bad": json_bad,
            "clinical_medicine_rows": clinical,
            "output_context": dict(contexts),
            "task_family": dict(families),
            "tasks": dict(tasks.most_common()),
            "scierc_ner_sample": sample,
        }
    )
    return report


def _jsonl_docs(path: Path, id_key: str) -> dict[str, Any]:
    rows = load_jsonl(path)
    empty = 0
    bad = 0
    ids: list[str] = []
    keys: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            bad += 1
            continue
        keys.update(row.keys())
        text = row.get("sentence") or row.get("text") or row.get("tokens") or row.get("sentences")
        if text in (None, "", [], {}):
            empty += 1
        if row.get(id_key) is not None:
            ids.append(str(row.get(id_key)))
    return {
        "rows": len(rows),
        "empty_text": empty,
        "non_objects": bad,
        "keys": dict(keys),
        "unique_ids": len(set(ids)),
        "ids": ids,
    }


def _scier(root: Path) -> dict[str, Any]:
    base = root / "scier" / "LLM"
    report: dict[str, Any] = {"dir": str(base)}
    if not base.exists():
        report["status"] = "missing"
        return report
    per_split = {}
    id_sets = {}
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "test_ood.jsonl"):
        path = base / name
        if not path.exists():
            per_split[name] = {"status": "missing"}
            continue
        info = _jsonl_docs(path, "doc_id")
        id_sets[name] = set(info.pop("ids"))
        per_split[name] = info
    leaks = {}
    for left, right in (("train.jsonl", "dev.jsonl"), ("train.jsonl", "test.jsonl"), ("dev.jsonl", "test.jsonl"), ("train.jsonl", "test_ood.jsonl")):
        if left in id_sets and right in id_sets:
            leaks[f"{left}|{right}"] = len(id_sets[left] & id_sets[right])
    report["splits"] = per_split
    report["shared_doc_ids"] = leaks
    report["status"] = "ok" if all(item.get("non_objects", 1) == 0 and item.get("empty_text", 1) == 0 for item in per_split.values() if "rows" in item) else "check"
    return report


def _scierc(root: Path) -> dict[str, Any]:
    base = root / "scierc"
    note = base / "DOWNLOAD_FAILED.txt"
    if note.exists() and not any(base.rglob("*.json")):
        return {"status": "download_failed", "note": note.read_text(encoding="utf-8")}
    files = [path for path in base.rglob("*") if path.suffix.lower() in {".json", ".jsonl"}]
    parsed = 0
    failed = 0
    for path in files:
        try:
            if path.suffix.lower() == ".jsonl":
                load_jsonl(path)
            else:
                json.loads(path.read_text(encoding="utf-8"))
            parsed += 1
        except Exception:
            failed += 1
    return {
        "status": "missing" if not files else ("ok" if failed == 0 else "check"),
        "files": len(files),
        "parsed": parsed,
        "failed": failed,
    }


def _cord(root: Path) -> dict[str, Any]:
    text = root / "cord" / "text"
    report: dict[str, Any] = {"dir": str(text)}
    if not text.exists():
        report["status"] = "missing"
        return report
    per_split = {}
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        path = text / name
        if not path.exists():
            per_split[name] = {"status": "missing"}
            continue
        rows = load_jsonl(path)
        parsed = 0
        bad = 0
        missing_parse = 0
        empty_name = 0
        image_keys = 0
        for row in rows:
            raw = row.get("ground_truth")
            if "image" in row:
                image_keys += 1
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(payload, dict):
                bad += 1
                continue
            parsed += 1
            gt = payload.get("gt_parse")
            if not isinstance(gt, dict):
                missing_parse += 1
                continue
            menu = gt.get("menu")
            items = menu if isinstance(menu, list) else [menu] if isinstance(menu, dict) else []
            for item in items:
                if isinstance(item, dict) and not str(item.get("nm") or "").strip():
                    empty_name += 1
        per_split[name] = {
            "rows": len(rows),
            "parsed": parsed,
            "bad_json": bad,
            "missing_gt_parse": missing_parse,
            "empty_menu_names": empty_name,
            "image_fields": image_keys,
        }
    report["splits"] = per_split
    report["images_kept"] = False
    report["status"] = "ok"
    for item in per_split.values():
        if item.get("bad_json", 1) or item.get("missing_gt_parse", 1) or item.get("status") == "missing":
            report["status"] = "check"
    return report


def inspect_all(dest: Path) -> dict[str, Any]:
    dest = Path(dest)
    return {
        "sciriff_4096": _sciriff(dest),
        "scier_llm": _scier(dest),
        "scierc": _scierc(dest),
        "cord_text": _cord(dest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Dataset/ without modifying it")
    parser.add_argument("--dest", type=Path, default=Path("Dataset"))
    args = parser.parse_args(argv)
    report = inspect_all(args.dest)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
