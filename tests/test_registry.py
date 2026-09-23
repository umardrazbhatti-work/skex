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


def test_slot_reuses_the_same_model_and_releases_on_change():
    from skex.models.load import ModelSlot

    calls = []
    released = []

    def loader(model_id):
        calls.append(model_id)
        return ("weights", model_id)

    slot = ModelSlot(releaser=released.append)
    first, how = slot.get("Qwen/Qwen2.5-1.5B-Instruct", loader)
    again, how2 = slot.get("Qwen/Qwen2.5-1.5B-Instruct", loader)
    other, how3 = slot.get("Qwen/Qwen2.5-3B-Instruct", loader)
    assert first is again
    assert how == "load" and how2 == "reuse" and how3 == "load"
    assert calls == ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    assert released == [("weights", "Qwen/Qwen2.5-1.5B-Instruct")]
    slot.release()
    assert released[-1] == ("weights", "Qwen/Qwen2.5-3B-Instruct")
    slot.release()
    assert len(released) == 2


def test_pack_kaggle_skips_when_working_dir_is_missing(tmp_path):
    from skex.experiments.pack import pack_kaggle
    assert pack_kaggle("note", working=tmp_path / "absent") is None


def test_pack_kaggle_writes_the_zip(tmp_path):
    from skex.experiments.pack import pack_kaggle
    import zipfile

    working = tmp_path / "working"
    repo = tmp_path / "repo"
    run = repo / "outputs" / "runs" / "r1"
    run.mkdir(parents=True)
    working.mkdir()
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    dest = pack_kaggle("plan 01 finished", repo=repo, working=working)
    assert dest == working / "skex-output.zip"
    with zipfile.ZipFile(dest) as archive:
        assert archive.read("RUN.txt") == b"plan 01 finished\n"
        assert "outputs/runs/r1/metrics.json" in archive.namelist()


def test_sealed_outlines_cells_import_as_succeeded(tmp_path):
    """Outlines dev cells from 22 Sep travel in the repo. Prompt-JSON does not.

    registry.jsonl is gitignored, so a Kaggle clone only learns about
    finished cells through experiments/sealed.jsonl. The 23 Sep prompt-JSON
    1.5B cell finished in the log and then the kernel was killed before its
    generations were zipped, so those two fingerprints must still run.
    """
    from skex.paths import ROOT

    reg = Registry(tmp_path / "reg.jsonl", tmp_path / "fail.jsonl")
    sealed = ROOT / "experiments" / "sealed.jsonl"
    assert reg.import_sealed(sealed) == 2
    assert reg.import_sealed(sealed) == 0
    for fp in ("5ea1cba6acc2e8a2", "85edb1fb3d92e4c8"):
        ok, reason = reg.should_run(fp, retry_failed=True)
        assert ok is False and "succeeded" in reason
    for fp in ("998f8fafc5edf958", "cb0d9b71c2836671"):
        ok, _reason = reg.should_run(fp, retry_failed=True)
        assert ok is True


def test_block_success(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl")
    spec = _spec(cell="C")
    fp = fingerprint(spec)
    reg.start(spec, "r2", fp)
    reg.finish("r2", fp, spec, {"field_f1": 0.5})
    ok, reason = reg.should_run(fp)
    assert ok is False and "succeeded" in reason
