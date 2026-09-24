from __future__ import annotations
from collections import defaultdict
from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def quote_in_doc(quote: str, document: str) -> bool:
    if not quote or not document:
        return False
    if quote in document:
        return True
    return _norm(quote) in _norm(document)


def _quotes_by_field(pred: dict[str, Any]) -> dict[str, list[str]]:
    """Keep every quote. A later quote for the same field must not erase an earlier one."""
    quotes: dict[str, list[str]] = defaultdict(list)
    for span in pred.get("evidence_spans") or []:
        if not isinstance(span, dict):
            continue
        field = span.get("field")
        if not field:
            continue
        quotes[str(field)].append(str(span.get("quote") or ""))
    return quotes


def span_support(pred: dict[str, Any] | None, document: str) -> dict[str, Any]:
    """Every non-empty field is supported when any of its quotes is in the document."""
    if not isinstance(pred, dict) or not pred:
        return {"n_non_null": 0, "n_supported": 0, "span_support": 0.0, "unsupported_fields": []}
    quotes = _quotes_by_field(pred)
    fields = []
    if "fields" in pred and isinstance(pred["fields"], dict):
        for k, v in pred["fields"].items():
            if v not in (None, "", []):
                fields.append(k)
    else:
        for k, v in pred.items():
            if k == "evidence_spans":
                continue
            if v in (None, "", []):
                continue
            fields.append(k)
    supported = []
    unsupported = []
    for f in fields:
        if any(quote_in_doc(q, document) for q in quotes.get(f, [])):
            supported.append(f)
        else:
            unsupported.append(f)
    n = len(fields)
    return {
        "n_non_null": n,
        "n_supported": len(supported),
        "span_support": (len(supported) / n) if n else 1.0,
        "unsupported_fields": unsupported,
    }
