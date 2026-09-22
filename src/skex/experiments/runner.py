from __future__ import annotations
import argparse
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from skex.config import load_config
from skex.data.io import load_jsonl
from skex.eval.classify import classify
from skex.eval.metrics import aggregate, score_example
from skex.experiments.fingerprint import fingerprint
from skex.experiments.plan import load_plan
from skex.experiments.registry import Registry
from skex.logging.events import EventLogger
from skex.models.load import describe_backend
from skex.paths import ROOT, resolve


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _spec_from_job(plan: dict, job: dict, cfg: dict) -> dict:
    return {
        "plan_id": plan["plan_id"],
        "cell": job.get("cell"),
        "model_id": job.get("model_id") or cfg["models"]["primary"],
        "adapter_id": job.get("adapter_id"),
        "decode_arm": job.get("decode_arm", "prompt_json"),
        "decode_engine": cfg["decode"]["engine"] if job.get("decode_arm") == "constrained" else "none",
        "schema_id": cfg["schema_id"],
        "split": job.get("split", "dev"),
        "data_rev": job.get("data_rev", "v0"),
        "seq_len": cfg["train"]["seq_len"],
        "lora_r": cfg["train"]["lora_r"] if job.get("adapter_id") else None,
        "train_examples": job.get("train_examples"),
        "epochs": job.get("epochs"),
    }


def _load_split(job: dict, cfg: dict) -> list[dict]:
    split = job.get("split", "dev")
    domain = job.get("domain", "A")
    key = "domain_a_dir" if domain == "A" else "domain_b_dir"
    path = resolve(cfg["data"][key]) / f"{split}.jsonl"
    if path.exists():
        return load_jsonl(path)
    # smoke fallback
    demo = resolve("schemas/example_record.json")
    return [json.loads(demo.read_text())]


def _predict(job: dict, record: dict, cfg: dict) -> str:
    arm = job.get("decode_arm", "smoke")
    if arm == "smoke" or job.get("backend") == "smoke":
        from skex.decode.interface import generate
        return generate(record["input"], arm="smoke", engine="none")
    from skex.decode.interface import generate
    return generate(
        record["input"],
        arm=job.get("decode_arm", "prompt_json"),
        engine=cfg["decode"]["engine"],
    )


def run_job(plan: dict, job: dict, cfg: dict, registry: Registry, *, retry_failed: bool, force: bool) -> dict:
    spec = _spec_from_job(plan, job, cfg)
    fp = fingerprint(spec)
    ok, reason = registry.should_run(fp, retry_failed=retry_failed)
    if force:
        ok, reason = True, "forced"
    run_id = _run_id()
    run_dir = resolve("outputs/runs") / run_id
    log = EventLogger(run_dir)
    log.write_json("spec.json", spec)
    log.write_json("PLAN.md", {"plan": plan["plan_id"], "job": job, "reason": reason})
    (run_dir / "PLAN.md").write_text(
        f"# Plan\n\nplan_id: {plan['plan_id']}\njob: {job.get('cell')}\n"
        f"fingerprint: {fp}\nshould_run: {ok}\nreason: {reason}\n"
    )
    log.emit("gate", fingerprint=fp, ok=ok, reason=reason)
    result = {"run_id": run_id, "fingerprint": fp, "reason": reason, "status": None}
    if not ok:
        result["status"] = "skipped"
        log.emit("skipped", reason=reason)
        print(f"SKIP {fp} {reason}")
        return result
    registry.start(spec, run_id, fp)
    log.emit("start", backend=describe_backend())
    try:
        rows = _load_split(job, cfg)
        limit = job.get("max_docs")
        if limit:
            rows = rows[: int(limit)]
        scored = []
        gens = []
        schema_path = str(resolve(cfg["schema_path"]))
        for rec in rows:
            raw = _predict(job, rec, cfg)
            s = score_example(raw, rec["output"], rec["input"], schema_path, cfg["eval"]["score_unattested"])
            s["id"] = rec["id"]
            scored.append(s)
            gens.append({"id": rec["id"], "raw": raw, "scores": s})
        metrics = aggregate(scored)
        log.write_json("metrics.json", metrics)
        log.write_json("generations.json", gens)
        registry.finish(run_id, fp, spec, metrics)
        log.emit("finish", metrics=metrics)
        result["status"] = "succeeded"
        result["metrics"] = metrics
        print(f"OK   {fp} run={run_id} n={metrics['n']} f1={metrics['field_f1']:.3f} wrong_valid={metrics['wrong_valid']:.3f}")
    except NotImplementedError as e:
        registry.fail(run_id, fp, spec, failure_class="unknown", error=str(e))
        log.emit("fail", failure_class="unknown", error=str(e))
        result["status"] = "failed"
        print(f"FAIL {fp} not-implemented: {e}")
    except Exception as e:
        klass = classify(e)
        registry.fail(run_id, fp, spec, failure_class=klass, error=traceback.format_exc())
        log.emit("fail", failure_class=klass, error=str(e))
        result["status"] = "failed"
        print(f"FAIL {fp} class={klass} {e}")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SKEX plan runner — checks registry before every job")
    p.add_argument("--plan", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore registry (debug only)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    plan = load_plan(args.plan)
    registry = Registry(cfg["registry"]["path"], cfg["registry"]["failures_path"])
    print(f"plan={plan['plan_id']} jobs={len(plan['jobs'])}")
    rc = 0
    for job in plan["jobs"]:
        res = run_job(plan, job, cfg, registry, retry_failed=args.retry_failed, force=args.force)
        if res["status"] == "failed":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
