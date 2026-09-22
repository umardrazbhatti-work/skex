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
