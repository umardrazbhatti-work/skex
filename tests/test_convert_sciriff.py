import json
from pathlib import Path

from skex.data.convert_sciriff import build_rows, convert
from skex.data.io import load_jsonl

DOC_A = (
    "We present a syntax-based constraint for word alignment, known as the cohesion constraint."
)
DOC_B = "We use a BiLSTM-CRF for named entity recognition on the CoNLL-2003 benchmark."
DOC_C = "The proposed graph parser improves dependency parsing on the Penn Treebank."
DOC_CLINICAL = "This trial measured fasting glucose after metformin treatment in adults with diabetes."


def _row(instance_id, split, document, output, **metadata):
    meta = {
        "domains": ["artificial_intelligence"],
        "output_context": "json",
        "source_type": "single_source",
        "task_family": "ie.named_entity_recognition",
        "input_context": "paragraph",
    }
    meta.update(metadata)
    return {
        "_instance_id": instance_id,
        "_split": split,
        "input": "Extract entities from the abstract.\n\nAbstract:\n" + document,
        "output": json.dumps(output),
        "metadata": meta,
    }


def _fixture_rows():
    ner_a = {
        "Task": ["word alignment"],
        "Method": ["not-in-document-method"],
        "Material": ["English phrases"],
        "Metric": ["alignment quality"],
        "Generic": ["algorithms"],
        "OtherScientificTerm": ["cohesion constraint"],
    }
    return [
        _row("scierc_ner:train:0", "train", DOC_A, ner_a),
        _row("scierc_ner:test:0", "test", DOC_A, ner_a),
        _row(
            "scierc_ner:train:1",
            "train",
            DOC_B,
            {"Task": ["named entity recognition"], "Method": ["bilstm-crf"], "Material": ["CoNLL-2003"], "Metric": [], "Generic": [], "OtherScientificTerm": []},
        ),
        _row(
            "scierc_ner:validation:0",
            "dev",
            DOC_C,
            {"Task": ["dependency parsing"], "Method": ["graph parser"], "Material": [], "Metric": [], "Generic": [], "OtherScientificTerm": []},
        ),
        _row(
            "scifact_entailment:train:0",
            "train",
            DOC_CLINICAL,
            {"Task": ["diabetes"]},
            domains=["biomedicine", "clinical_medicine"],
        ),
        _row(
            "ncbi_ner:train:0",
            "train",
            "The patient has influenza according to the abstract note.",
            {"Disease": ["influenza"]},
        ),
    ]


def test_spaced_punctuation_uses_the_document_slice():
    from skex.data.convert_sciriff import attested_quote

    document = "estimating affine camera parameters, illumination, shape, and albedo"
    quote = attested_quote(
        "estimating affine camera parameters , illumination , shape , and albedo",
        document,
    )
    assert quote == document
    assert attested_quote("not in the abstract at all", document) is None


def test_omits_unattested_and_unmapped_types():
    grouped, stats = build_rows(_fixture_rows())
    test_rows = grouped["test"]
    assert len(test_rows) == 1
    card = test_rows[0]["output"]
    assert card["task"] == ["word alignment"]
    assert card["method"] is None
    assert card["datasets"] is None
    assert card["metrics"] is None
    assert card["claims"] is None
    assert card["scores"] is None
    blob = json.dumps(card)
    assert "cohesion constraint" not in blob
    assert "English phrases" not in blob
    assert "not-in-document-method" not in blob
    assert card["evidence_spans"] == [{"field": "task", "quote": "word alignment"}]
    assert stats["skipped_clinical"] == 1
    assert stats["skipped_unmapped"] == 1
    assert stats["dropped_cross_split"] == 1


def test_case_recovers_document_substring_and_blocks_doc_leak(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    src.write_text("\n".join(json.dumps(row) for row in _fixture_rows()) + "\n", encoding="utf-8")
    dest = tmp_path / "domain_a"
    stats = convert(src, dest)
    train = load_jsonl(dest / "train.jsonl")
    dev = load_jsonl(dest / "dev.jsonl")
    test = load_jsonl(dest / "test.jsonl")
    assert stats["kept_train"] == 1
    assert stats["kept_dev"] == 1
    assert stats["kept_test"] == 1
    assert train[0]["output"]["method"] == ["BiLSTM-CRF"]
    assert train[0]["output"]["datasets"] == ["CoNLL-2003"]
    assert "BiLSTM-CRF" in train[0]["input"]
    train_docs = {row["doc_id"] for row in train}
    test_docs = {row["doc_id"] for row in test}
    dev_docs = {row["doc_id"] for row in dev}
    assert train_docs.isdisjoint(test_docs)
    assert train_docs.isdisjoint(dev_docs)
    assert dev_docs.isdisjoint(test_docs)
    for split in (train, dev, test):
        for row in split:
            assert set(row) >= {"id", "domain", "instruction", "input", "output"}
            assert row["domain"] == "A"
            for span in row["output"]["evidence_spans"]:
                assert span["quote"] in row["input"]
