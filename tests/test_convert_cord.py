import json

from skex.data.convert_cord import card_from_payload
from skex.eval.schema_check import parse_and_validate
from skex.paths import ROOT

SCHEMA = ROOT / "schemas" / "cord_receipt.schema.json"


def _payload():
    return {
        "gt_parse": {
            "menu": [
                {"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"},
                {"nm": "NOT ON RECEIPT", "cnt": "1", "price": "9,999"},
            ],
            "sub_total": {"subtotal_price": "16,500", "tax_price": "1,500"},
            "total": {"total_price": "18,000", "cashprice": "20,000", "changeprice": "2,000"},
        },
        "valid_line": [
            {"words": [{"text": "1"}, {"text": "REAL"}, {"text": "GANACHE"}, {"text": "16,500"}]},
            {"words": [{"text": "18,000"}, {"text": "20,000"}, {"text": "2,000"}]},
            {"words": [{"text": "1,500"}]},
        ],
        "meta": {"image_id": 0},
    }


def test_unattested_menu_line_is_dropped_and_missing_totals_are_null():
    document, card = card_from_payload(_payload())
    assert "REAL GANACHE" in document
    assert card["menu"] == [
        {"nm": "REAL GANACHE", "cnt": None, "price": "16,500", "unitprice": None}
    ]
    assert card["total_price"] == "18,000"
    assert card["cashprice"] == "20,000"
    assert card["changeprice"] == "2,000"
    assert card["subtotal_price"] == "16,500"
    assert card["tax_price"] == "1,500"
    assert card["service_price"] is None
    assert card["discount_price"] is None
    assert card["creditcardprice"] is None
    assert card["emoneyprice"] is None
    assert "NOT ON RECEIPT" not in json.dumps(card)
    for span in card["evidence_spans"]:
        assert span["quote"] in document
    checked = parse_and_validate(card, SCHEMA)
    assert checked["valid"], checked["errors"]
