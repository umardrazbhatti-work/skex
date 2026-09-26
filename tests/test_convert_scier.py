from skex.data.convert_scier import build_rows


def test_scier_document_keeps_task_method_and_dataset(tmp_path):
    src = tmp_path / "LLM"
    src.mkdir()
    rows = [
        {"doc_id": "1", "sentence": "We propose CornerNet for object detection.", "ner": [["CornerNet", "Method"], ["object detection", "Task"]], "rel": [], "rel_plus": []},
        {"doc_id": "1", "sentence": "Results are reported on COCO.", "ner": [["COCO", "Dataset"]], "rel": [], "rel_plus": []},
        {"doc_id": "2", "sentence": "Nothing annotated here.", "ner": [], "rel": [], "rel_plus": []},
    ]
    import json
    (src / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    for name in ("dev.jsonl", "test.jsonl", "test_ood.jsonl"):
        (src / name).write_text("", encoding="utf-8")
    grouped, stats = build_rows(src)
    assert stats["kept"] == 1
    card = grouped["train"][0]["output"]
    assert card["method"] == ["CornerNet"]
    assert card["task"] == ["object detection"]
    assert card["datasets"] == ["COCO"]
    assert card["metrics"] is None
    assert card["scores"] is None
    for span in card["evidence_spans"]:
        assert span["quote"] in grouped["train"][0]["input"]
