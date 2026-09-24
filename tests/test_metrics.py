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


def test_an_earlier_quote_still_supports_the_field():
    pred = {
        **GOLD,
        "evidence_spans": [
            {"field": "method", "quote": "We propose a BiLSTM-CRF"},
            {"field": "method", "quote": "not in the document at all"},
        ],
    }
    s = score_example(pred, GOLD, DOC, SCHEMA)
    assert "method" not in s["unsupported_fields"]


def test_string_evidence_span_does_not_crash():
    pred = {
        **GOLD,
        "evidence_spans": ["We propose a BiLSTM-CRF", {"field": "method", "quote": "BiLSTM-CRF"}],
    }
    s = score_example(pred, GOLD, DOC, SCHEMA)
    assert s["parse_ok"]
    assert s["schema_valid"] is False
    assert "method" not in s["unsupported_fields"]


def test_gated_precision_matches_case_and_order():
    pred = {
        **GOLD,
        "method": ["bilstm-crf"],
        "datasets": ["CoNLL-2003"],
        "evidence_spans": [
            {"field": "method", "quote": "BiLSTM-CRF"},
            {"field": "task", "quote": "named entity recognition"},
            {"field": "datasets", "quote": "CoNLL-2003"},
            {"field": "metrics", "quote": "F1"},
            {"field": "scores", "quote": "91.2 F1"},
            {"field": "limitations", "quote": "English only"},
        ],
    }
    s = score_example(pred, GOLD, DOC, SCHEMA)
    assert s["gated_precision"] == 1.0
    assert s["gated_n"] == 6
