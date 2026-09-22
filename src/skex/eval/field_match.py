from __future__ import annotations
from typing import Any


ALIASES = {
    "f-1": "f1",
    "f-measure": "f1",
    "f measure": "f1",
    "accuracy": "acc",
}


def _norm_atom(x: Any) -> str:
    s = " ".join(str(x).strip().lower().split())
    return ALIASES.get(s, s)


def _as_set(value: Any) -> set[str]:
    if value in (None, "", []):
        return set()
    if isinstance(value, list):
        out = set()
        for v in value:
            if isinstance(v, dict):
                out.add(_norm_atom(v.get("metric") or v.get("value") or v))
            else:
                out.add(_norm_atom(v))
        return {x for x in out if x not in ("none", "null")}
    return {_norm_atom(value)}


def _field_map(card: dict[str, Any] | None) -> dict[str, Any]:
    if not card:
        return {}
    if "fields" in card and isinstance(card["fields"], dict):
        return card["fields"]
    return {k: v for k, v in card.items() if k != "evidence_spans"}


def field_scores(gold: dict[str, Any], pred: dict[str, Any] | None, document: str, score_unattested: bool = False) -> dict[str, Any]:
    gmap = _field_map(gold)
    pmap = _field_map(pred)
    keys = sorted(set(gmap) | set(pmap))
    per = {}
    tp = fp = fn = 0
    unscorable = []
    for k in keys:
        g = gmap.get(k)
        p = pmap.get(k)
        gset = _as_set(g)
        pset = _as_set(p)
        attested = True
        if not score_unattested and gset:
            joined = " ".join(gset)
            if joined and joined not in document.lower() and not any(tok in document.lower() for tok in gset if len(tok) > 3):
                # only mark unscorable when *no* gold atom appears in the doc
                if not any(atom in document.lower() for atom in gset):
                    attested = False
        if not attested:
            unscorable.append(k)
            per[k] = {"status": "unscorable", "gold": list(gset), "pred": list(pset)}
            continue
        inter = gset & pset
        only_g = gset - pset
        only_p = pset - gset
        tp += len(inter)
        fn += len(only_g)
        fp += len(only_p)
        per[k] = {"status": "scored", "tp": len(inter), "fp": len(only_p), "fn": len(only_g)}
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "field_f1": f1, "unscorable": unscorable, "per_field": per}
