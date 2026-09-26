"""Map SciREX full papers onto research cards.

A gold string is kept only when it is a contiguous substring of the packed
input. Relation names use underscores as word joins; the kept value is the
document's own slice, not the underscore form. Material is stored under datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from skex.data.convert_sciriff import INSTRUCTION, _NER_MAP, _blank_card, _norm_key, attested_quote, document_key
from skex.data.io import load_jsonl
from skex.data.pack import pack_document

_PACK_CHARS = 8000


def _bucket(text: str) -> str:
    head = " ".join(text.lower().split()[:8])
    if ":" in head[:24]:
        title = head.split(":", 1)[1]
    else:
        title = head
    if "abstract" in title:
        return "abstract"
    if any(word in title for word in ("method", "approach", "model", "architecture")):
        return "method"
    if any(word in title for word in ("experiment", "result")):
        return "experiments"
    if "limitation" in title:
        return "limitations"
    return "body"


def _sections(words: list[str], spans: list[Any]) -> dict[str, str]:
    sections = {"title": "", "abstract": "", "method": "", "experiments": "", "limitations": "", "body": ""}
    if spans and isinstance(spans[0], list) and len(spans[0]) >= 2 and spans[0][0] > 0:
        sections["title"] = " ".join(words[: spans[0][0]]).strip()
    for span in spans:
        if not isinstance(span, list) or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        chunk = " ".join(words[start:end]).strip()
        if not chunk:
            continue
        key = _bucket(chunk)
        sections[key] = (sections[key] + "\n\n" + chunk).strip()
    return sections


def _span_text(words: list[str], span: list[Any]) -> tuple[str, str] | None:
    if len(span) < 3:
        return None
    start, end, label = int(span[0]), int(span[1]), span[2]
    field = _NER_MAP.get(_norm_key(label))
    if not field or start < 0 or end <= start or end > len(words):
        return None
    return field, " ".join(words[start:end]).strip()


def _add(card: dict[str, Any], field: str, raw: str, document: str) -> None:
    quote = attested_quote(raw, document)
    if not quote:
        return
    if field == "scores":
        return
    values = list(card.get(field) or [])
    if quote not in values:
        values.append(quote)
        card[field] = values
        card["evidence_spans"].append({"field": field, "quote": quote})


def card_from_document(row: dict[str, Any]) -> dict[str, Any] | None:
    words = [str(word) for word in row.get("words") or []]
    if not words:
        return None
    packed = pack_document(_sections(words, row.get("sections") or []), max_chars=_PACK_CHARS)
    if not packed:
        return None
    card = _blank_card()
    for span in row.get("ner") or []:
        if not isinstance(span, list):
            continue
        found = _span_text(words, span)
        if found:
            _add(card, found[0], found[1], packed)
    for rel in row.get("n_ary_relations") or []:
        if not isinstance(rel, dict):
            continue
        for key, value in rel.items():
            field = _NER_MAP.get(_norm_key(key))
            if not field or not isinstance(value, str):
                continue
            _add(card, field, " ".join(value.replace("_", " ").split()), packed)
        metric = rel.get("Metric")
        score = rel.get("score")
        if isinstance(metric, str) and score not in (None, ""):
            metric_quote = attested_quote(" ".join(metric.replace("_", " ").split()), packed)
            value_quote = attested_quote(str(score), packed)
            if metric_quote and value_quote:
                scores = list(card.get("scores") or [])
                item = {"metric": metric_quote, "value": score if not isinstance(score, str) else value_quote}
                if item not in scores:
                    scores.append(item)
                    card["scores"] = scores
                    metrics = list(card.get("metrics") or [])
                    if metric_quote not in metrics:
                        metrics.append(metric_quote)
                        card["metrics"] = metrics
                    card["evidence_spans"].append({"field": "metrics", "quote": metric_quote})
                    card["evidence_spans"].append({"field": "scores", "quote": value_quote})
    if all(card.get(field) is None for field in ("task", "method", "datasets", "metrics", "scores", "claims", "limitations")):
        return None
    body = " ".join(words)
    return {
        "id": f"scirex:{row.get('_split')}:{row.get('doc_id')}",
        "domain": "A",
        "instruction": INSTRUCTION,
        "input": packed,
        "output": card,
        "doc_id": document_key(body),
        "source_task": "scirex",
    }


def build_rows(src: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    src = Path(src)
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    stats = {"read": 0, "kept": 0, "skipped": 0}
    files = {
        "train": src / "train.jsonl",
        "dev": src / "dev.jsonl",
        "test": src / "test.jsonl",
    }
    for split, path in files.items():
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            stats["read"] += 1
            built = card_from_document({**row, "_split": split})
            if built is None:
                stats["skipped"] += 1
                continue
            grouped[split].append(built)
            stats["kept"] += 1
    return grouped, stats
