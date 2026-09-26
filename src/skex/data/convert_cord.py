"""Map CORD receipt text onto the frozen receipt schema.

Images are not used. A gold string is kept only when attested_quote finds it
in the word lines of that receipt. Missing totals stay null.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skex.data.convert_sciriff import attested_quote
from skex.data.io import load_jsonl, write_jsonl

INSTRUCTION = (
    "Extract the receipt. Return JSON with every key: menu, subtotal_price, tax_price, "
    "service_price, discount_price, total_price, cashprice, changeprice, creditcardprice, "
    "emoneyprice, evidence_spans. Use null when the receipt text does not contain that value. "
    "Each menu item has nm, cnt, price, and unitprice. "
    'evidence_spans is a list of objects {"field", "quote"} and quote is an exact substring of the receipt.'
)

MENU_KEYS = ("nm", "cnt", "price", "unitprice")
SUB_KEYS = ("subtotal_price", "tax_price", "service_price", "discount_price")
TOTAL_KEYS = ("total_price", "cashprice", "changeprice", "creditcardprice", "emoneyprice")
SCALAR_KEYS = SUB_KEYS + TOTAL_KEYS


def receipt_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for row in payload.get("valid_line") or []:
        words: list[str] = []
        for word in row.get("words") or []:
            if not isinstance(word, dict):
                continue
            text = str(word.get("text") or "").strip()
            if text:
                words.append(text)
        if words:
            lines.append(" ".join(words))
    return "\n".join(lines)


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return None


def _quote(value: Any, document: str) -> str | None:
    if value is None:
        return None
    return attested_quote(str(value).strip(), document)


def _menu(gt: dict[str, Any], document: str) -> tuple[list[dict[str, Any]] | None, list[dict[str, str]]]:
    raw = gt.get("menu")
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    kept: list[dict[str, Any]] = []
    spans: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        line = {key: _quote(item.get(key), document) for key in MENU_KEYS}
        if all(value is None for value in line.values()):
            continue
        kept.append(line)
        for key, quote in line.items():
            if quote:
                spans.append({"field": key, "quote": quote})
    return (kept or None), spans


def _scalars(gt: dict[str, Any], document: str) -> tuple[dict[str, str | None], list[dict[str, str]]]:
    sub = _as_dict(gt.get("sub_total"))
    if sub is None and isinstance(gt.get("sub_total"), str):
        sub = {"subtotal_price": gt.get("sub_total")}
    total = _as_dict(gt.get("total"))
    if total is None and isinstance(gt.get("total"), str):
        total = {"total_price": gt.get("total")}
    source = {}
    source.update(sub or {})
    source.update(total or {})
    values: dict[str, str | None] = {}
    spans: list[dict[str, str]] = []
    for key in SCALAR_KEYS:
        quote = _quote(source.get(key), document)
        values[key] = quote
        if quote:
            spans.append({"field": key, "quote": quote})
    return values, spans


def card_from_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    document = receipt_text(payload)
    if not document:
        return None
    gt = payload.get("gt_parse")
    if not isinstance(gt, dict):
        return None
    menu, menu_spans = _menu(gt, document)
    scalars, scalar_spans = _scalars(gt, document)
    card: dict[str, Any] = {"menu": menu, **scalars, "evidence_spans": menu_spans + scalar_spans}
    if menu is None and all(scalars[key] is None for key in SCALAR_KEYS):
        return None
    return document, card


def build_rows(src: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    src = Path(src)
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    stats = {"read": 0, "kept": 0, "skipped": 0}
    for split in grouped:
        path = src / f"{split}.jsonl"
        if not path.is_file():
            continue
        for index, row in enumerate(load_jsonl(path)):
            stats["read"] += 1
            raw = row.get("ground_truth")
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                stats["skipped"] += 1
                continue
            if not isinstance(payload, dict):
                stats["skipped"] += 1
                continue
            built = card_from_payload(payload)
            if built is None:
                stats["skipped"] += 1
                continue
            document, card = built
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            image_id = meta.get("image_id", index)
            grouped[split].append(
                {
                    "id": f"cord:{split}:{image_id}",
                    "domain": "B",
                    "instruction": INSTRUCTION,
                    "input": document,
                    "output": card,
                    "doc_id": f"cord:{split}:{image_id}",
                    "source_task": "cord",
                }
            )
            stats["kept"] += 1
    return grouped, stats


def convert(src: Path, dest: Path) -> dict[str, int]:
    grouped, stats = build_rows(src)
    dest = Path(dest)
    for split, rows in grouped.items():
        write_jsonl(dest / f"{split}.jsonl", rows)
        stats[split] = len(rows)
    return stats


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert CORD text JSONL into Domain B receipts")
    parser.add_argument("--src", type=Path, default=Path("Dataset/cord/text"))
    parser.add_argument("--dest", type=Path, default=Path("data/processed/domain_b"))
    args = parser.parse_args()
    stats = convert(args.src, args.dest)
    print(json.dumps(stats, indent=2))
    return 0 if stats.get("train") and stats.get("test") else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
