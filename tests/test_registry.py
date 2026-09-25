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


def test_v02_cells_run_and_sealed_v01_rows_stay_blocked(tmp_path):
    """Schema v0.2 mints new fingerprints. The eight sealed v0.1 Qwen rows stay blocked.

    registry.jsonl is gitignored, so a Kaggle clone only skips cells that
    are listed in experiments/sealed.jsonl. Those rows use research_card.v0.1.
    """
    from skex.config import load_config
    from skex.experiments.fingerprint import fingerprint
    from skex.experiments.plan import load_plan
    from skex.experiments.runner import _spec_from_job
    from skex.paths import ROOT

    sealed_dev = [
        "998f8fafc5edf958",
        "5ea1cba6acc2e8a2",
        "cb0d9b71c2836671",
        "85edb1fb3d92e4c8",
    ]
    sealed_test = [
        "5a187165dcc1169b",
        "d2e84f99a8669900",
        "0250975feee544d1",
        "8d9cfcb6c44bdcc0",
    ]
    cfg = load_config()
    assert cfg["schema_id"] == "research_card.v0.2"
    dev = load_plan("experiments/plans/01_tax_zeroshot.yaml")
    test = load_plan("experiments/plans/01b_tax_zeroshot_test.yaml")
    dev_fps = [fingerprint(_spec_from_job(dev, job, cfg)) for job in dev["jobs"]]
    test_fps = [fingerprint(_spec_from_job(test, job, cfg)) for job in test["jobs"]]
    assert dev_fps == [
        "ccc915c32da36d81",
        "da1d605e7c7fe7d5",
        "c78b3bc828f5789f",
        "b3239df5871e2cfa",
    ]
    assert [job["split"] for job in test["jobs"]] == ["test", "test", "test", "test"]
    assert [job["max_docs"] for job in test["jobs"]] == [99, 99, 99, 99]
    assert test_fps == [
        "783b1f72c0e3595f",
        "3f848b71557eab7e",
        "c0d773de9383c17a",
        "2d5294ea57d5f13f",
    ]
    assert set(dev_fps).isdisjoint(sealed_dev)
    assert set(test_fps).isdisjoint(sealed_test)
    assert set(dev_fps).isdisjoint(test_fps)
    reg = Registry(tmp_path / "reg.jsonl", tmp_path / "fail.jsonl")
    sealed = ROOT / "experiments" / "sealed.jsonl"
    assert reg.import_sealed(sealed) == 8
    assert reg.import_sealed(sealed) == 0
    for fp in sealed_dev + sealed_test:
        ok, reason = reg.should_run(fp, retry_failed=True)
        assert ok is False and "succeeded" in reason
    for fp in dev_fps + test_fps:
        ok, reason = reg.should_run(fp, retry_failed=True)
        assert ok is True and reason == "new"
    llama_dev = load_plan("experiments/plans/01c_tax_zeroshot_llama_dev.yaml")
    llama_test = load_plan("experiments/plans/01d_tax_zeroshot_llama_test.yaml")
    llama_dev_fps = [fingerprint(_spec_from_job(llama_dev, job, cfg)) for job in llama_dev["jobs"]]
    llama_test_fps = [fingerprint(_spec_from_job(llama_test, job, cfg)) for job in llama_test["jobs"]]
    assert [job["model_id"] for job in llama_dev["jobs"]] == [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    assert [job["split"] for job in llama_dev["jobs"]] == ["dev", "dev", "dev", "dev"]
    assert [job["split"] for job in llama_test["jobs"]] == ["test", "test", "test", "test"]
    assert llama_dev_fps == [
        "00662886ed0e4b07",
        "fc167651daa9443b",
        "6a22d0beb474a64a",
        "d721454cb0a44913",
    ]
    assert llama_test_fps == [
        "8a161e35e9122b8e",
        "877bf9f2a5277765",
        "19eb8aac99ba8a53",
        "f4579c32f1f5375a",
    ]
    known = set(dev_fps) | set(test_fps) | set(sealed_dev) | set(sealed_test)
    assert known.isdisjoint(llama_dev_fps)
    assert known.isdisjoint(llama_test_fps)
    assert set(llama_dev_fps).isdisjoint(llama_test_fps)
    for fp in llama_dev_fps + llama_test_fps:
        ok, reason = reg.should_run(fp, retry_failed=True)
        assert ok is True and reason == "new"


def test_block_success(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl")
    spec = _spec(cell="C")
    fp = fingerprint(spec)
    reg.start(spec, "r2", fp)
    reg.finish("r2", fp, spec, {"field_f1": 0.5})
    ok, reason = reg.should_run(fp)
    assert ok is False and "succeeded" in reason


def test_notebook_runs_the_ten_unfinished_cells_one_model_at_a_time():
    """The 25 Sep run finished six Qwen cells. The notebook runs the other ten.

    Each phase file keeps the original plan_id, so the fingerprint matches the
    full plan, and each process loads one model.
    """
    import ast
    import json

    from skex.config import load_config
    from skex.experiments.fingerprint import fingerprint
    from skex.experiments.plan import load_plan
    from skex.experiments.runner import _spec_from_job
    from skex.paths import ROOT

    finished = {
        "ccc915c32da36d81",
        "da1d605e7c7fe7d5",
        "c78b3bc828f5789f",
        "b3239df5871e2cfa",
        "783b1f72c0e3595f",
        "3f848b71557eab7e",
    }
    nb = json.loads((ROOT / "notebooks" / "skex_kaggle.ipynb").read_text(encoding="utf-8"))
    source = ""
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if "PHASE_FILES" in text:
            source = text
            break
    assert source
    for old in (
        "01_tax_zeroshot.yaml",
        "01b_tax_zeroshot_test.yaml",
        "01c_tax_zeroshot_llama_dev.yaml",
        "01d_tax_zeroshot_llama_test.yaml",
    ):
        assert old not in source
    phase_files = None
    phase_order = None
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "PHASE_FILES":
                phase_files = ast.literal_eval(node.value)
            if isinstance(target, ast.Name) and target.id == "PHASES":
                phase_order = ast.literal_eval(node.value)
    assert [item[1] for item in phase_order] == [
        "experiments/plans/01e_qwen3b_test.yaml",
        "experiments/plans/01f_llama1b_dev.yaml",
        "experiments/plans/01g_llama1b_test.yaml",
        "experiments/plans/01h_llama3b_dev.yaml",
        "experiments/plans/01i_llama3b_test.yaml",
    ]
    assert list(phase_files) == [item[1] for item in phase_order]
    cfg = load_config()
    seen = []
    for relative, text in phase_files.items():
        assert (ROOT / relative).read_text(encoding="utf-8") == text
        plan = load_plan(relative)
        models = {job["model_id"] for job in plan["jobs"]}
        assert len(models) == 1
        assert [job["decode_arm"] for job in plan["jobs"]] == ["prompt_json", "constrained"]
        seen.extend(fingerprint(_spec_from_job(plan, job, cfg)) for job in plan["jobs"])
    full = []
    for relative in (
        "experiments/plans/01_tax_zeroshot.yaml",
        "experiments/plans/01b_tax_zeroshot_test.yaml",
        "experiments/plans/01c_tax_zeroshot_llama_dev.yaml",
        "experiments/plans/01d_tax_zeroshot_llama_test.yaml",
    ):
        plan = load_plan(relative)
        full.extend(fingerprint(_spec_from_job(plan, job, cfg)) for job in plan["jobs"])
    assert len(seen) == 10
    assert len(set(seen)) == 10
    assert set(seen).isdisjoint(finished)
    assert set(seen) | finished == set(full)
    assert set(full) - finished == set(seen)


def test_notebook_tries_both_kaggle_dataset_mounts_before_failing():
    import json

    from skex.paths import ROOT

    nb = json.loads((ROOT / "notebooks" / "skex_kaggle.ipynb").read_text(encoding="utf-8"))
    source = ""
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if "DATA_CANDIDATES" in text:
            source = text
            break
    assert source
    first = "/kaggle/input/datasets/umardrazbhatti/skex-datasets"
    second = "/kaggle/input/datasets/umardrazbhatti999/skex-datasets"
    assert source.index(first) < source.index(second)
    assert source.index(second) < source.index("Dataset not found")
    assert "if DATA is None" in source
