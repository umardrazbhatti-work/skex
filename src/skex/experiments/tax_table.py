"""Before-training constraint-tax table for the sixteen Phase 2 cells."""

from __future__ import annotations

import argparse
import sys

from skex.config import load_config
from skex.experiments.fingerprint import fingerprint
from skex.experiments.plan import load_plan
from skex.experiments.registry import Registry
from skex.experiments.runner import _spec_from_job
from skex.paths import resolve

PHASE2_PLANS = [
    "experiments/plans/01f_llama1b_dev.yaml",
    "experiments/plans/01g_llama1b_test.yaml",
    "experiments/plans/01h_llama3b_dev.yaml",
    "experiments/plans/01i_llama3b_test.yaml",
    "experiments/plans/01l_qwen3b_dev.yaml",
    "experiments/plans/01e_qwen3b_test.yaml",
    "experiments/plans/01j_qwen15b_dev.yaml",
    "experiments/plans/01k_qwen15b_test.yaml",
]

EXPECTED_N = {"dev": 126, "test": 175}

MODEL_LABEL = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5 3B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5 1.5B",
}

ARM_LABEL = {
    "prompt_json": "A prompt-JSON",
    "constrained": "B Outlines",
}


def expected_cells(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    cells = []
    for relative in PHASE2_PLANS:
        plan = load_plan(relative)
        for job in plan["jobs"]:
            spec = _spec_from_job(plan, job, cfg)
            cells.append({"fingerprint": fingerprint(spec), "spec": spec})
    fingerprints = [cell["fingerprint"] for cell in cells]
    if len(fingerprints) != 16 or len(set(fingerprints)) != 16:
        raise RuntimeError("Phase 2 plans must yield 16 distinct cells")
    return cells


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _signed(value: float) -> str:
    return f"{float(value):+.3f}"


def build_table(latest: dict, cfg: dict | None = None) -> tuple[str | None, list[str]]:
    problems: list[str] = []
    rows = []
    paired: dict[tuple, dict] = {}
    for cell in expected_cells(cfg):
        spec = cell["spec"]
        label = (
            f"{MODEL_LABEL.get(spec['model_id'], spec['model_id'])} "
            f"{spec['split']} {ARM_LABEL.get(spec['decode_arm'], spec['decode_arm'])}"
        )
        row = latest.get(cell["fingerprint"])
        if row is None or row.get("status") != "succeeded":
            problems.append(f"missing {label} ({cell['fingerprint']})")
            continue
        stored = row.get("spec") or {}
        if stored.get("schema_id") != "research_card.v1" or stored.get("data_rev") != "v1":
            problems.append(f"{label} is not research_card.v1 data_rev v1")
            continue
        metrics = row.get("metrics") or {}
        want = EXPECTED_N[spec["split"]]
        if metrics.get("n") != want:
            problems.append(f"{label} scored {metrics.get('n')} papers; Phase 2 needs {want}")
            continue
        rows.append({"spec": spec, "metrics": metrics})
        paired[(spec["model_id"], spec["split"], spec["decode_arm"])] = metrics
    if problems:
        return None, problems

    lines = [
        "# Phase 2 before-training tax",
        "",
        "Schema research_card.v1. Domain A dev has 126 papers and test has 175. No training and no adapter.",
        "",
        "| Model | Split | Cell | n | Field F1 | Wrong-valid | Parse | Schema valid | Span support | Gated precision | Latency ms | Tokens in | Tokens out | Tax, B against A |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in rows:
        spec = entry["spec"]
        metrics = entry["metrics"]
        tax = ""
        if spec["decode_arm"] == "constrained":
            base = paired[(spec["model_id"], spec["split"], "prompt_json")]
            tax = (
                f"field F1 {_signed(metrics['field_f1'] - base['field_f1'])}, "
                f"wrong-valid {_signed(metrics['wrong_valid'] - base['wrong_valid'])}"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABEL.get(spec["model_id"], spec["model_id"]),
                    spec["split"],
                    ARM_LABEL.get(spec["decode_arm"], spec["decode_arm"]),
                    str(metrics["n"]),
                    _fmt(metrics.get("field_f1")),
                    _fmt(metrics.get("wrong_valid")),
                    _fmt(metrics.get("parse_rate")),
                    _fmt(metrics.get("schema_valid")),
                    _fmt(metrics.get("span_support")),
                    _fmt(metrics.get("gated_precision")),
                    _fmt(metrics.get("latency_ms"), 0),
                    _fmt(metrics.get("tokens_in"), 0),
                    _fmt(metrics.get("tokens_out"), 0),
                    tax,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines), []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Phase 2 before-training tax table")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out", default="experiments/phase2_tax.md")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    registry = Registry(args.registry or cfg["registry"]["path"])
    text, problems = build_table(registry.latest_by_fp(), cfg)
    if problems:
        print("Phase 2 tax table is incomplete:")
        for item in problems:
            print("-", item)
        return 1
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
