from skex.data.convert_scirex import card_from_document


def test_scirex_keeps_only_text_that_was_packed():
    row = {
        "doc_id": "paper-1",
        "words": [
            "BERT", "for", "parsing", "section", ":", "Abstract",
            "We", "use", "BERT", "on", "PTB", ".",
            "section", ":", "Appendix", "The", "hidden", "metric", "is", "LAS", ".",
        ],
        "sections": [[0, 6], [6, 12], [12, 20]],
        "ner": [
            [8, 9, "Method"],
            [10, 11, "Material"],
        ],
        "n_ary_relations": [
            {"Task": "parsing", "Method": "BERT", "Metric": "not_in_paper", "Material": "PTB", "score": "91.2"},
        ],
    }
    built = card_from_document({**row, "_split": "train"})
    assert built is not None
    card = built["output"]
    assert "BERT" in card["method"]
    assert card["datasets"] == ["PTB"]
    assert card["task"] == ["parsing"]
    assert card["metrics"] is None
    assert card["scores"] is None
    assert "91.2" not in built["input"]
    for span in card["evidence_spans"]:
        assert span["quote"] in built["input"]
