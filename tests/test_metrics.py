from skex.eval.metrics import score_example
from skex.paths import ROOT

SCHEMA = str(ROOT / "schemas" / "research_card.schema.json")
DOC = "We propose a BiLSTM-CRF for named entity recognition. We evaluate on CoNLL-2003 and report 91.2 F1 on the test set. English only."
GOLD = {
    "task": ["named entity recognition"],
    "method": ["BiLSTM-CRF"],
    "datasets": ["CoNLL-2003"],
    "metrics": ["F1"],
    "scores": [{"metric": "F1", "value": 91.2, "condition": "test"}],
    "claims": [],
    "limitations": ["English only"],
    "evidence_spans": [{"field": "method", "quote": "We propose a BiLSTM-CRF"}],
}


def test_perfect_match():
    s = score_example(GOLD, GOLD, DOC, SCHEMA)
    assert s["parse_ok"] and s["schema_valid"]
    assert s["field_f1"] == 1.0
    assert s["span_support"] > 0


def test_valid_but_wrong():
    pred = dict(GOLD)
    pred = {**GOLD, "method": ["Transformer-XL"], "evidence_spans": [{"field": "method", "quote": "We propose a BiLSTM-CRF"}]}
    s = score_example(pred, GOLD, DOC, SCHEMA)
    assert s["schema_valid"]
    assert s["wrong_valid"] is True
