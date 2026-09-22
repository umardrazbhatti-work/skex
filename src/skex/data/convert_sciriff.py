"""Map public allenai/SciRIFF rows onto the research-card JSONL splits.

Reads a local SciRIFF drop (parquet or JSONL). Does not invent fields.
A gold value is kept only when it is a contiguous substring of the packed
document. Clinical-medicine rows and multi-document inputs are dropped.
Split assignment is by document text id: if one document appears in a
more held-out SciRIFF split, copies in the other splits are removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from skex.data.io import load_jsonl, write_jsonl
from skex.data.pack import pack_document
from skex.data.splits import assert_no_doc_leak, assert_no_id_leak

INSTRUCTION = (
    "Extract a research card from the document. "
    "Return JSON using only these keys when a value is a contiguous substring of the document: "
    "task, method, datasets, metrics, scores, claims, limitations, evidence_spans. "
    "Omit keys you cannot support. Do not add values that are not written in the document. "
    'evidence_spans is a list of objects {"field", "quote"} where quote is an exact substring of the document.'
)

_JSON_CONTEXTS = {"json", "jsonlines"}
_SPLIT_ALIAS = {"train": "train", "dev": "dev", "validation": "dev", "valid": "dev", "test": "test"}
_RANK = {"test": 2, "dev": 1, "train": 0}
_NER_MAP = {"task": "task", "method": "method", "metric": "metrics", "material": "datasets"}
_NER_EXTRA = {"generic", "otherscientificterm"}
_CARD_KEYS = ("task", "method", "datasets", "metrics", "scores", "claims", "limitations")
_LIST_FIELDS = ("task", "method", "datasets", "metrics", "claims", "limitations")
_BOUNDARY = re.compile(r"(?im)^[ \t]*(abstract|title|paper|full text|fulltext|document)\s*:\s*$")
_PACK_CHARS = 8000


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z]", "", str(key).lower())


def attested_quote(value: str, document: str) -> str | None:
    """Return a document slice for value.

    Exact match first. If SciRIFF spaced the punctuation ("word , next"),
    recover the contiguous document span. Do not invent a span that is not
    in the document.
    """
    value = " ".join(str(value).split())
    if len(value) < 2 or not document:
        return None
    if value in document:
        return value
    if len(document.lower()) == len(document):
        idx = document.lower().find(value.lower())
        if idx >= 0:
            return document[idx : idx + len(value)]
    tightened = re.sub(r"\s+([,.;:)\]])", r"\1", value)
    tightened = re.sub(r"([(\[])\s+", r"\1", tightened)
    tokens = re.findall(r"\w+|[^\w\s]", tightened)
    if len(tokens) < 1:
        return None
    pattern = r"\s*".join(re.escape(token) for token in tokens)
    match = re.search(pattern, document, flags=re.IGNORECASE)
    if not match:
        return None
    return document[match.start() : match.end()]


def extract_document(prompt: str) -> str | None:
    """Keep the source passage after the last template boundary. Do not keep the instructions."""
    if not prompt or not str(prompt).strip():
        return None
    text = str(prompt).replace("\r\n", "\n")
    matches = list(_BOUNDARY.finditer(text))
    if matches:
        body = text[matches[-1].end() :].strip()
        return body or None
    inline = re.search(r"(?is)\babstract\s*:\s*(.+)$", text)
    if not inline:
        return None
    body = inline.group(1).strip()
    return body or None


def document_key(body: str) -> str:
    """Content id of the packed source. SciRIFF does not ship a paper id."""
    norm = " ".join(body.split()).lower()
    return "sha256:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:20]


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("text") or item.get("mention") or item.get("span") or item.get("name")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


def _is_ner(obj: dict[str, Any]) -> bool:
    norms = {_norm_key(k) for k in obj}
    if not norms or not norms <= (set(_NER_MAP) | _NER_EXTRA):
        return False
    return bool(norms & set(_NER_MAP))


def _finish_card(buckets: dict[str, list[str]], document: str) -> dict[str, Any] | None:
    card: dict[str, Any] = {}
    spans: list[dict[str, str]] = []
    for field in _LIST_FIELDS:
        kept: list[str] = []
        for raw in buckets.get(field) or []:
            quote = attested_quote(raw, document)
            if not quote:
                continue
            if quote not in kept:
                kept.append(quote)
                spans.append({"field": field, "quote": quote})
        if kept:
            card[field] = kept
    if not card:
        return None
    card["evidence_spans"] = spans
    return card


def _map_ner(obj: dict[str, Any], document: str) -> dict[str, Any] | None:
    buckets: dict[str, list[str]] = {field: [] for field in _LIST_FIELDS}
    for key, value in obj.items():
        field = _NER_MAP.get(_norm_key(key))
        if not field:
            continue
        buckets[field].extend(_string_items(value))
    return _finish_card(buckets, document)


def _map_scores(value: Any, document: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(value, list):
        return [], []
    kept: list[dict[str, Any]] = []
    spans: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or "metric" not in item or "value" not in item:
            continue
        metric = item.get("metric")
        raw_value = item.get("value")
        if not isinstance(metric, str) or raw_value in (None, ""):
            continue
        metric_quote = attested_quote(metric, document)
        value_quote = attested_quote(str(raw_value), document)
        if not metric_quote or not value_quote:
            continue
        score: dict[str, Any] = {"metric": metric_quote, "value": raw_value}
        condition = item.get("condition")
        if isinstance(condition, str) and condition.strip():
            condition_quote = attested_quote(condition, document)
            if condition_quote:
                score["condition"] = condition_quote
        kept.append(score)
        spans.append({"field": "metrics", "quote": metric_quote})
        spans.append({"field": "scores", "quote": value_quote})
    return kept, spans


def _map_card(obj: dict[str, Any], document: str) -> dict[str, Any] | None:
    buckets: dict[str, list[str]] = {field: [] for field in _LIST_FIELDS}
    for key, value in obj.items():
        field = _norm_key(key)
        if field in _LIST_FIELDS:
            buckets[field].extend(_string_items(value))
    card = _finish_card(buckets, document) or {}
    scores, score_spans = _map_scores(obj.get("scores"), document)
    if scores:
        card["scores"] = scores
        metrics = list(card.get("metrics") or [])
        for span in score_spans:
            if span["field"] == "metrics" and span["quote"] not in metrics:
                metrics.append(span["quote"])
        if metrics:
            card["metrics"] = metrics
        card.setdefault("evidence_spans", [])
        card["evidence_spans"].extend(score_spans)
    if not card:
        return None
    card.setdefault("evidence_spans", [])
    return card


def _map_relations(items: list[Any], document: str) -> dict[str, Any] | None:
    buckets: dict[str, list[str]] = {field: [] for field in _LIST_FIELDS}
    typed = False
    for item in items:
        if not isinstance(item, dict):
            return None
        for side in ("head", "tail", "subject", "object"):
            node = item.get(side)
            if not isinstance(node, dict):
                continue
            text = node.get("text") or node.get("mention") or node.get("name")
            label = node.get("type") or node.get("label")
            if not isinstance(text, str) or not isinstance(label, str):
                continue
            field = _NER_MAP.get(_norm_key(label))
            if not field:
                continue
            typed = True
            buckets[field].append(text)
    if not typed:
        return None
    return _finish_card(buckets, document)


def map_output(parsed: Any, document: str) -> dict[str, Any] | None:
    """Map one parsed SciRIFF output. Return None when no attested card field exists."""
    if isinstance(parsed, dict):
        if _is_ner(parsed):
            return _map_ner(parsed, document)
        norms = {_norm_key(k) for k in parsed}
        if norms & set(_CARD_KEYS):
            return _map_card(parsed, document)
        return None
    if isinstance(parsed, list) and parsed:
        if all(isinstance(item, dict) and _is_ner(item) for item in parsed):
            merged: dict[str, list[Any]] = {}
            for item in parsed:
                for key, value in item.items():
                    merged.setdefault(key, [])
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)
            return _map_ner(merged, document)
        return _map_relations(parsed, document)
    return None


def parse_output(raw: Any) -> Any | None:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    rows: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return rows or None


def split_from_filename(name: str) -> str | None:
    stem = name.lower()
    if "validation" in stem or re.search(r"(^|[_\-.])dev([_\-.]|$)", stem):
        return "dev"
    if "test" in stem:
        return "test"
    if "train" in stem:
        return "train"
    return None


def _task_name(instance_id: str) -> str:
    return instance_id.split(":", 1)[0] if instance_id else ""


def _domains(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    domains = metadata.get("domains") or []
    if isinstance(domains, str):
        return [domains]
    return [str(item) for item in domains]


def _output_context(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("output_context") or "")


def _source_type(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("source_type") or "")


def build_rows(instances: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    stats: dict[str, Any] = Counter()
    pending: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for instance in instances:
        stats["read"] += 1
        instance_id = str(instance.get("_instance_id") or "").strip()
        split = _SPLIT_ALIAS.get(str(instance.get("_split") or "").lower(), "")
        if not instance_id:
            stats["skipped_bad_id"] += 1
            continue
        if instance_id in seen_ids:
            stats["skipped_duplicate_id"] += 1
            continue
        if split not in _RANK:
            stats["skipped_bad_split"] += 1
            continue
        seen_ids.add(instance_id)
        if "clinical_medicine" in _domains(instance.get("metadata")):
            stats["skipped_clinical"] += 1
            continue
        if _source_type(instance.get("metadata")) == "multiple_source":
            stats["skipped_multi_source"] += 1
            continue
        if _output_context(instance.get("metadata")) not in _JSON_CONTEXTS:
            stats["skipped_not_json"] += 1
            continue
        body = extract_document(str(instance.get("input") or ""))
        if not body:
            stats["skipped_no_document"] += 1
            continue
        packed = pack_document({"abstract": body}, max_chars=_PACK_CHARS)
        parsed = parse_output(instance.get("output"))
        if parsed is None:
            stats["skipped_bad_json"] += 1
            continue
        card = map_output(parsed, packed)
        if card is None:
            stats["skipped_unmapped"] += 1
            continue
        pending.append(
            {
                "id": instance_id,
                "domain": "A",
                "instruction": INSTRUCTION,
                "input": packed,
                "output": card,
                "doc_id": document_key(body),
                "source_task": _task_name(instance_id),
                "_split": split,
            }
        )

    present: dict[str, set[str]] = defaultdict(set)
    for row in pending:
        present[row["doc_id"]].add(row["_split"])
    keep = {doc: max(splits, key=lambda item: _RANK[item]) for doc, splits in present.items()}
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    kept_tasks: Counter[str] = Counter()
    for row in pending:
        if row["_split"] != keep[row["doc_id"]]:
            stats["dropped_cross_split"] += 1
            continue
        split = row.pop("_split")
        grouped[split].append(row)
        kept_tasks[row["source_task"]] += 1
    for split, rows in grouped.items():
        stats[f"kept_{split}"] = len(rows)
        stats[f"docs_{split}"] = len({row["doc_id"] for row in rows})
    _reject_leaks(grouped)
    stats["kept_by_task"] = dict(kept_tasks.most_common())
    return grouped, dict(stats)


def _reject_leaks(grouped: dict[str, list[dict[str, Any]]]) -> None:
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for left, right in pairs:
        assert_no_id_leak(grouped[left], grouped[right])
        assert_no_doc_leak(grouped[left], grouped[right])


def load_instances(src: Path) -> Iterator[dict[str, Any]]:
    src = Path(src)
    if src.is_file():
        yield from _load_file(src, split_from_filename(src.name))
        return
    files = sorted(path for path in src.rglob("*") if path.suffix.lower() in {".jsonl", ".parquet"})
    if not files:
        raise FileNotFoundError(f"No JSONL or parquet files under {src}")
    for path in files:
        yield from _load_file(path, split_from_filename(path.name))


def _load_file(path: Path, file_split: str | None) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        yield from _load_parquet(path, file_split)
        return
    for row in load_jsonl(path):
        split = file_split or _SPLIT_ALIAS.get(str(row.get("_split") or "").lower())
        yield {**row, "_split": split or ""}


def _load_parquet(path: Path, file_split: str | None) -> Iterator[dict[str, Any]]:
    if file_split is None:
        raise ValueError(f"Cannot tell train/dev/test from parquet name: {path.name}")
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
    except Exception:
        yield from _load_parquet_duckdb(path, file_split)
        return
    for batch in parquet.iter_batches(batch_size=256):
        columns = batch.to_pydict()
        count = len(columns["_instance_id"])
        for index in range(count):
            yield {
                "input": columns["input"][index],
                "output": columns["output"][index],
                "metadata": columns["metadata"][index],
                "_instance_id": columns["_instance_id"][index],
                "_split": file_split,
            }


def _load_parquet_duckdb(path: Path, file_split: str) -> Iterator[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "Reading SciRIFF parquet needs pyarrow or duckdb. pip install duckdb"
        ) from exc
    cursor = duckdb.connect().cursor()
    cursor.execute(
        "SELECT input, output, metadata, _instance_id FROM read_parquet(?)",
        [str(path)],
    )
    while True:
        batch = cursor.fetchmany(256)
        if not batch:
            break
        for input_text, output, metadata, instance_id in batch:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            yield {
                "input": input_text,
                "output": output,
                "metadata": metadata,
                "_instance_id": instance_id,
                "_split": file_split,
            }


def convert(src: Path, dest: Path) -> dict[str, Any]:
    grouped, stats = build_rows(load_instances(Path(src)))
    dest = Path(dest)
    for split, rows in grouped.items():
        write_jsonl(dest / f"{split}.jsonl", rows)
    return stats


def _print_stats(stats: dict[str, Any]) -> None:
    print("split counts:")
    for split in ("train", "dev", "test"):
        print(f"  {split}: rows={stats.get(f'kept_{split}', 0)} docs={stats.get(f'docs_{split}', 0)}")
    print(json.dumps(stats, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a local allenai/SciRIFF drop into Domain-A JSONL")
    parser.add_argument("--src", type=Path, required=True, help="Directory or file of SciRIFF parquet/JSONL")
    parser.add_argument("--dest", type=Path, default=Path("data/processed/domain_a"))
    args = parser.parse_args(argv)
    stats = convert(args.src, args.dest)
    _print_stats(stats)
    if stats.get("kept_train", 0) == 0 or stats.get("kept_test", 0) == 0:
        print("FAIL: train or test is empty after decontamination", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
