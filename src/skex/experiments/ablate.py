from __future__ import annotations
import argparse
import csv
import sys
from skex.config import load_config
from skex.experiments.registry import Registry
from skex.paths import resolve

COLS = [
    "fingerprint", "status", "plan_id", "cell", "model_id", "adapter_id",
    "decode_arm", "decode_engine", "split", "n", "parse_rate", "schema_valid",
    "field_f1", "wrong_valid", "span_support", "gated_precision", "failure_class",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the ablation table from the registry")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="experiments/ablations/latest.csv")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    reg = Registry(cfg["registry"]["path"], cfg["registry"]["failures_path"])
    latest = reg.latest_by_fp()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for fp, r in latest.items():
            spec = r.get("spec") or {}
            m = r.get("metrics") or {}
            w.writerow({
                "fingerprint": fp,
                "status": r.get("status"),
                "plan_id": spec.get("plan_id"),
                "cell": spec.get("cell"),
                "model_id": spec.get("model_id"),
                "adapter_id": spec.get("adapter_id"),
                "decode_arm": spec.get("decode_arm"),
                "decode_engine": spec.get("decode_engine"),
                "split": spec.get("split"),
                "n": m.get("n"),
                "parse_rate": m.get("parse_rate"),
                "schema_valid": m.get("schema_valid"),
                "field_f1": m.get("field_f1"),
                "wrong_valid": m.get("wrong_valid"),
                "span_support": m.get("span_support"),
                "gated_precision": m.get("gated_precision"),
                "failure_class": r.get("failure_class"),
            })
    print(f"wrote {out} rows={len(latest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
