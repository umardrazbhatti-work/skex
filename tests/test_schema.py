from pathlib import Path
import json
from skex.eval.schema_check import parse_and_validate
from skex.paths import ROOT

SCHEMA = ROOT / "schemas" / "research_card.schema.json"
EXAMPLE = json.loads((ROOT / "schemas" / "example_record.json").read_text())


def test_example_card_validates():
    r = parse_and_validate(EXAMPLE["output"], SCHEMA)
    assert r["parse_ok"] and r["valid"], r["errors"]


def test_extra_key_fails():
    bad = dict(EXAMPLE["output"])
    bad["extra"] = "nope"
    r = parse_and_validate(bad, SCHEMA)
    assert r["parse_ok"] and not r["valid"]


def test_omitted_keys_are_valid():
    card = {"task": ["named entity recognition"], "evidence_spans": [{"field": "task", "quote": "named entity recognition"}]}
    r = parse_and_validate(card, SCHEMA)
    assert r["parse_ok"] and r["valid"], r["errors"]


def test_score_without_value_is_invalid():
    card = dict(EXAMPLE["output"])
    card["scores"] = [{"metric": "F1"}]
    r = parse_and_validate(card, SCHEMA)
    assert r["parse_ok"] and not r["valid"]


def test_fence_and_lead_in_parse_as_the_card():
    card = json.dumps(EXAMPLE["output"])
    fenced = "```json\n" + card + "\n```"
    lead = "Here is the extracted JSON:\n" + fenced
    for raw in (fenced, lead):
        r = parse_and_validate(raw, SCHEMA)
        assert r["parse_ok"] and r["valid"], r["errors"]
        assert r["parsed"]["task"] == EXAMPLE["output"]["task"]


def test_schema_echo_parses_and_is_not_a_card():
    raw = 'Here is the research card:\n```\n{"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "SKEX"}\n```'
    r = parse_and_validate(raw, SCHEMA)
    assert r["parse_ok"] and not r["valid"]


def test_truncated_fence_does_not_parse():
    r = parse_and_validate('```json\n{"task": ["named', SCHEMA)
    assert r["parse_ok"] is False


def test_prose_does_not_parse():
    r = parse_and_validate("I cannot extract a card.", SCHEMA)
    assert r["parse_ok"] is False
