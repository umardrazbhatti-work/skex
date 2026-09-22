from skex.experiments.fingerprint import fingerprint
from skex.experiments.registry import Registry


def _spec(**kw):
    base = dict(
        plan_id="t", cell="A", model_id="m", adapter_id=None,
        decode_arm="prompt_json", decode_engine="none", schema_id="s",
        split="dev", data_rev="v0", seq_len=2048, lora_r=None,
        train_examples=None, epochs=None,
    )
    base.update(kw)
    return base


def test_fingerprint_stable():
    a = fingerprint(_spec())
    b = fingerprint(_spec())
    assert a == b
    assert fingerprint(_spec(seq_len=1024)) != a


def test_block_failed(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl", tmp_path / "fail.jsonl")
    spec = _spec()
    fp = fingerprint(spec)
    assert reg.should_run(fp)[0] is True
    reg.start(spec, "r1", fp)
    assert reg.should_run(fp)[0] is False
    reg.fail("r1", fp, spec, failure_class="oom", error="CUDA OOM")
    ok, reason = reg.should_run(fp)
    assert ok is False and "oom" in reason
    ok, reason = reg.should_run(fp, retry_failed=True)
    assert ok is False and "non-retryable" in reason
    spec2 = _spec(seq_len=1024)
    assert reg.should_run(fingerprint(spec2))[0] is True


def test_block_success(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl")
    spec = _spec(cell="C")
    fp = fingerprint(spec)
    reg.start(spec, "r2", fp)
    reg.finish("r2", fp, spec, {"field_f1": 0.5})
    ok, reason = reg.should_run(fp)
    assert ok is False and "succeeded" in reason
