from skex.experiments.registry import Registry
from skex.experiments.tax_table import build_table, expected_cells, main


def _metrics(spec: dict, n_override: int | None = None) -> dict:
    arm = spec["decode_arm"]
    return {
        "n": n_override if n_override is not None else (126 if spec["split"] == "dev" else 175),
        "field_f1": 0.200 if arm == "prompt_json" else 0.150,
        "wrong_valid": 0.100 if arm == "prompt_json" else 0.900,
        "parse_rate": 1.0,
        "schema_valid": 1.0,
        "span_support": 0.010,
        "gated_precision": 0.0,
        "latency_ms": 1000.0,
        "tokens_in": 800.0,
        "tokens_out": 100.0,
    }


def _fill(tmp_path, n_override: int | None = None):
    registry = Registry(tmp_path / "registry.jsonl", tmp_path / "failures.jsonl")
    for index, cell in enumerate(expected_cells()):
        registry.finish(
            f"run{index}",
            cell["fingerprint"],
            cell["spec"],
            _metrics(cell["spec"], n_override if index == 0 else None),
        )
    return registry


def test_tax_table_writes_the_sixteen_cell_comparison(tmp_path):
    registry = _fill(tmp_path)
    text, problems = build_table(registry.latest_by_fp())
    assert problems == []
    assert text.count("A prompt-JSON") == 8
    assert text.count("B Outlines") == 8
    assert "Llama 3.2 1B" in text
    assert "Llama 3.2 3B" in text
    assert "Qwen2.5 3B" in text
    assert "Qwen2.5 1.5B" in text
    assert "field F1 -0.050" in text
    assert "wrong-valid +0.800" in text
    out = tmp_path / "phase2_tax.md"
    assert main(["--registry", str(tmp_path / "registry.jsonl"), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == text


def test_tax_table_refuses_a_short_split(tmp_path):
    registry = _fill(tmp_path, n_override=50)
    text, problems = build_table(registry.latest_by_fp())
    assert text is None
    assert any("needs 126" in item for item in problems)
    assert main(["--registry", str(tmp_path / "registry.jsonl"), "--out", str(tmp_path / "nope.md")]) == 1
    assert not (tmp_path / "nope.md").exists()
