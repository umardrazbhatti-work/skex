from __future__ import annotations
from typing import Any
from skex.eval.schema_check import parse_and_validate
from skex.eval.span import span_support
from skex.eval.field_match import field_scores


def score_example(
    raw_pred: str | dict,
    gold: dict,
    document: str,
    schema_path: str,
    score_unattested: bool = False,
) -> dict[str, Any]:
    chk = parse_and_validate(raw_pred, schema_path)
    pred = chk["parsed"] if chk["parse_ok"] else None
    fields = field_scores(gold, pred if isinstance(pred, dict) else None, document, score_unattested)
    spans = span_support(pred if isinstance(pred, dict) else None, document)
    correct = fields["fp"] == 0 and fields["fn"] == 0 and chk["valid"]
    wrong_valid = bool(chk["valid"] and not correct)
    gated_tp = 0
    gated_n = 0
    if isinstance(pred, dict):
        unsupported = set(spans["unsupported_fields"])
        fmap = pred.get("fields") if "fields" in pred else {k: v for k, v in pred.items() if k != "evidence_spans"}
        gmap = gold.get("fields") if "fields" in gold else {k: v for k, v in gold.items() if k != "evidence_spans"}
        for k, v in (fmap or {}).items():
            if v in (None, "", []) or k in unsupported:
                continue
            gated_n += 1
            # crude exact membership
            gv = gmap.get(k) if gmap else None
            if gv == v or (isinstance(gv, list) and v in gv) or (isinstance(v, list) and gv in v):
                gated_tp += 1
    return {
        "parse_ok": chk["parse_ok"],
        "schema_valid": chk["valid"],
        "schema_errors": chk["errors"],
        "field_f1": fields["field_f1"],
        "precision": fields["precision"],
        "recall": fields["recall"],
        "wrong_valid": wrong_valid,
        "span_support": spans["span_support"],
        "unsupported_fields": spans["unsupported_fields"],
        "unscorable": fields["unscorable"],
        "gated_precision": (gated_tp / gated_n) if gated_n else 0.0,
        "gated_n": gated_n,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    def mean(k):
        return sum(float(r.get(k) or 0.0) for r in rows) / n
    return {
        "n": len(rows),
        "parse_rate": sum(1 for r in rows if r.get("parse_ok")) / n,
        "schema_valid": sum(1 for r in rows if r.get("schema_valid")) / n,
        "field_f1": mean("field_f1"),
        "wrong_valid": sum(1 for r in rows if r.get("wrong_valid")) / n,
        "span_support": mean("span_support"),
        "gated_precision": mean("gated_precision"),
    }
