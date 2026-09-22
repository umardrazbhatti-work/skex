from __future__ import annotations
from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def quote_in_doc(quote: str, document: str) -> bool:
    if not quote or not document:
        return False
    if quote in document:
        return True
    return _norm(quote) in _norm(document)


def span_support(pred: dict[str, Any] | None, document: str) -> dict[str, Any]:
    """Every non-empty atomic field should have a quote ⊆ document."""
    if not pred:
        return {"n_non_null": 0, "n_supported": 0, "span_support": 0.0, "unsupported_fields": []}
    spans = {s.get("field"): s.get("quote", "") for s in pred.get("evidence_spans") or [] if s.get("field")}
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
        q = spans.get(f, "")
        if quote_in_doc(str(q), document):
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
