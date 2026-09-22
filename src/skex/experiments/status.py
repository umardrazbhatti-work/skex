from __future__ import annotations
import argparse
import sys
from collections import Counter
from skex.config import load_config
from skex.experiments.registry import Registry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    reg = Registry(cfg["registry"]["path"], cfg["registry"]["failures_path"])
    latest = reg.latest_by_fp()
    if not latest:
        print("registry empty")
        return 0
    counts = Counter(r.get("status") for r in latest.values())
    print("fingerprints:", len(latest), dict(counts))
    print(f"{'status':10} {'fp':16} {'cell':8} {'arm':14} {'model':40} {'note'}")
    for fp, r in latest.items():
        spec = r.get("spec") or {}
        note = r.get("failure_class") or ""
        if r.get("metrics"):
            m = r["metrics"]
            note = f"f1={m.get('field_f1', 0):.2f} wv={m.get('wrong_valid', 0):.2f}"
        print(f"{r.get('status','?'):10} {fp:16} {str(spec.get('cell')):8} {str(spec.get('decode_arm')):14} {str(spec.get('model_id'))[:40]:40} {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
