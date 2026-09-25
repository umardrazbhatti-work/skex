# Session status

Read this file first at the next session, then `GROK.md`, `docs/LOCKED_DECISIONS.md`, `docs/EXECUTION_PLAN.md`, and `docs/DATASET_INSPECTION.md`.

## Resume here — ten cells left, one model per step

`Results/25-9-26 1400Hrs` is Kaggle version 13. The v0.2 checker worked. Six Qwen cells finished in about 59 minutes. The next load, Qwen2.5-3B on the test split, stuck at 49% of the weights until the 12-hour cap (exit 137). Llama did not start.

Keep those six scores. Re-upload `notebooks/skex_kaggle.ipynb` before Save & Run. The last cell writes the five phase plans into the clone and runs them in order, each in its own process: Qwen2.5-3B test, Llama 3.2 1B dev, Llama 3.2 1B test, Llama 3.2 3B dev, Llama 3.2 3B test. That is the other ten cells. `HF_TOKEN` stays checked. Accelerator GPU T4 x2, Internet on, dataset `skex-datasets`. Budget about 4 hours. Download `skex-output.zip` into a new Results folder. Plan 02 stays off.

The version 13 zip still has the six finished cells. Download it into `Results/25-9-26 1400Hrs` if that file is not already there.

SciERC `Material` stays under `datasets`. That mapping is the recorded decision in `docs/DATASET_INSPECTION.md`.

## Previous resume — 2026-09-24 after `Results/24-9-26 1400Hrs`

Qwen dev and Qwen test are finished and sealed. Eight fingerprints. Do not re-run them on schema v0.1. Do not start plan 02. Do not install Unsloth. Do not download a 7B.

`24-9-26 1400Hrs` is the Qwen-only test notebook. `RUN.txt` says `test plan finished`. The log ends at about 64 minutes. No crash. No Llama. That notebook did not need `HF_TOKEN`. Smoke passed. The four Qwen dev cells skipped. The four Qwen test cells finished, and the generations are in the zip.

Saved test numbers, 99 papers each.

| Fingerprint | Arm | Model | Parse | Schema valid | Field F1 | Wrong-valid | Span support |
|---|---|---|---:|---:|---:|---:|---:|
| `5a187165dcc1169b` | prompt-JSON | Qwen2.5 1.5B | 0.00 | 0.00 | 0.000 | 0.00 | 0.000 |
| `d2e84f99a8669900` | Outlines | Qwen2.5 1.5B | 1.00 | 0.990 | 0.118 | 0.990 | 0.011 |
| `0250975feee544d1` | prompt-JSON | Qwen2.5 3B | 1.00 | 0.990 | 0.151 | 0.990 | 0.025 |
| `8d9cfcb6c44bdcc0` | Outlines | Qwen2.5 3B | 1.00 | 1.00 | 0.148 | 1.00 | 0.004 |

Run ids: `20260923T111720Z-4ceeb1` (1.5B prompt-JSON), `20260923T113203Z-c38d53` (1.5B Outlines), `20260923T114833Z-f37e2a` (3B prompt-JSON), `20260923T120610Z-4b7df8` (3B Outlines). All four are in `experiments/sealed.jsonl`.

Fence, kept separate from the table. All 99 of the 1.5B prompt completions on this test split start with a markdown ` ```json ` fence, so the saved parse rate is 0.00. The other three test cells are raw JSON.

That next notebook became `Results/24-9-26 2030Hrs`. Llama finished on the v0.1 checker. The checker fix at the top of this file is the follow-up. Low field F1 on an untouched model stays the measurement.

## Previous resume — 2026-09-23 after `Results/23-9-26 1515Hrs`

Qwen dev is finished and sealed. The Qwen-only test notebook that was running then became `Results/24-9-26 1400Hrs`, read above. Do not re-run the four Qwen dev fingerprints.

Saved dev numbers, 50 papers each. Outlines rows are the 22 Sep runs. Prompt-JSON rows are this folder.

| Fingerprint | Arm | Model | Parse | Schema valid | Field F1 | Wrong-valid | Span support |
|---|---|---|---:|---:|---:|---:|---:|
| `998f8fafc5edf958` | prompt-JSON | Qwen2.5 1.5B | 0.00 | 0.00 | 0.000 | 0.00 | 0.000 |
| `5ea1cba6acc2e8a2` | Outlines | Qwen2.5 1.5B | 1.00 | 1.00 | 0.118 | 1.00 | 0.004 |
| `cb0d9b71c2836671` | prompt-JSON | Qwen2.5 3B | 1.00 | 1.00 | 0.154 | 1.00 | 0.035 |
| `85edb1fb3d92e4c8` | Outlines | Qwen2.5 3B | 1.00 | 1.00 | 0.211 | 1.00 | 0.000 |

Run ids for the new cells: `20260923T081503Z-80feff` (1.5B) and `20260923T082300Z-4d7647` (3B). Generations are in the zip. All four dev fingerprints are in `experiments/sealed.jsonl`.

Fence, kept separate from the table above. All 50 of the 1.5B prompt completions start with a markdown ` ```json ` fence, so the saved parse rate is 0.00 and field F1 was not scored. That 0.00 stays the saved metric. The 3B prompt completions are raw JSON. A one-off count of the card inside the 1.5B fence, which does not replace the table, was parse 0.94, schema valid 0.72, field F1 0.140, wrong-valid 0.72, span support 0. Three texts still failed to parse after the fence was removed. Eight parsed cards had `scores` as an object, and three had a list where the schema wants a number or string.

The notebook to re-upload after that test zip was `notebooks/skex_kaggle.ipynb`. That upload is the Llama run described at the top of this file.

## Previous resume — 2026-09-23 after `Results/23-9-26 1030Hrs`

Both prompt-JSON cells were still open when that note was written. The 1515 run above closed them.

`23-9-26 1030Hrs` is Kaggle version 8, GPU T4 x2, killed after 43213s (12h limit, exit 137). `RUN.txt` says `smoke finished` because the zip was written before plan 01. Pytest 16 passed. Smoke skipped on the second call.

| Fingerprint | Cell | Model | This run | Note |
|---|---|---|---|---|
| `998f8fafc5edf958` | A prompt-JSON | Qwen2.5 1.5B | process finished | `run=20260922T102328Z-d766b0`, n=50, field F1 0.000, wrong-valid 0.000. Logged docs 1/10/20/30/40/50 were all `valid=False`. Generations were not in the zip. Not sealed in `experiments/registry.jsonl`, so the next run repeats it and keeps the raw text. |
| `5ea1cba6acc2e8a2` | B constrained | Qwen2.5 1.5B | did not finish | Started a second 4-bit load of the same 1.5B and sat at `Loading weights: 49%` until the kill. Already succeeded on 22 Sep. Now sealed in the registry. Must stay skipped. |
| `cb0d9b71c2836671` | A prompt-JSON | Qwen2.5 3B | did not start | Still the open cell. |
| `85edb1fb3d92e4c8` | B constrained | Qwen2.5 3B | did not start | Already succeeded on 22 Sep. Now sealed. Must stay skipped. |

Outlines numbers from `Results/22-9-26 1515Hrs` still stand. They are not the prompt-JSON arm. The 1.5B prompt-JSON log line is not a saved paper result: there is no `generations.json` and no `metrics.json` for `20260922T102328Z-d766b0`. Do not fill the missing parse rate, span-support, or token counts.

What changed after reading that folder:

- The runner keeps one loaded model. A second cell with the same `model_id` reuses it instead of calling `from_pretrained` again.
- Releasing a model drops the reference before `gc` and `empty_cache`. That is the 49% hang.
- On Kaggle the runner rewrites `/kaggle/working/skex-output.zip` every 10 documents and after every job.
- Doc 1 prints a 200-character preview plus `parse=` and `valid=`.
- `experiments/sealed.jsonl` is tracked and holds the two Outlines successes above. `experiments/registry.jsonl` is gitignored, so the runner copies those rows in at the start of every plan. The two prompt-JSON fingerprints are absent, so `--retry-failed` runs them.

Kaggle clones GitHub. These edits do nothing on the next Save & Run until they are pushed and the notebook is re-uploaded. Re-upload `notebooks/skex_kaggle.ipynb` as well. The notebook text no longer says the run will download Llama.

Accelerator stays GPU T4 x2. Use GPU 0 only. Models stay public Qwen. No `HF_TOKEN`.

## Previous resume — 2026-09-22 15:36 local

Git `main` was `42fd4fc`. The run then in progress became `Results/23-9-26 1030Hrs`, read above.

Already finished, from `Results/22-9-26 1515Hrs`. Do not quote these as the prompt-JSON arm:

| Fingerprint | Cell | Model | Status | Note |
|---|---|---|---|---|
| `5ea1cba6acc2e8a2` | B constrained | Qwen2.5 1.5B | succeeded | 50 docs, field F1 0.118, wrong-valid 1.0, span-support 0.004 |
| `85edb1fb3d92e4c8` | B constrained | Qwen2.5 3B | succeeded | 50 docs, field F1 0.211, wrong-valid 1.0, span-support 0.0 |
| `998f8fafc5edf958` | A prompt-JSON | Qwen2.5 1.5B | failed on 22 Sep, ran on 23 Sep | 22 Sep was the token-dict crash. 23 Sep finished with field F1 0.000 and the artifacts were lost. |
| `cb0d9b71c2836671` | A prompt-JSON | Qwen2.5 3B | failed on 22 Sep, not rerun | same 22 Sep crash. Still open. |

The 15:15 crash was `AttributeError: shape` in `src/skex/decode/interface.py`. `apply_chat_template` returned a token dict and `model.generate` treated it as a tensor. `chat_tensors()` unpacks `input_ids`. That crash did not recur on 23 Sep.

## Results folder (fixed)

`C:\Users\PC1\Desktop\Specialization Beats Scale G\Results`

This path stays the same. Each Kaggle run is a new subfolder with its own name. The first run is `22-9-26 1400Hrs` (dual T4 abort, before the dataset cell). When a new run is added, read that subfolder before changing the notebook or the loader. Do not move this directory.

Date: 2026-09-22. Topic is closed. Do not write a new research proposal.

`AGENTS.md`, `docs/GITHUB.md`, and `docs/KAGGLE.md` were not in the tree. Operating rules are in `GROK.md` and `docs/LOCKED_DECISIONS.md`.

## Folder

The workspace is now the repo root. The nested `skex\skex` directory was moved up and removed.

`C:\Users\PC1\Desktop\Specialization Beats Scale G\skex`

`src\skex` and `GROK.md` are in this folder. There is one `skex` directory, not two.

## Done this session

- SciRIFF Domain-A converter: `src/skex/data/convert_sciriff.py`
- No-GPU test: `tests/test_convert_sciriff.py`
- Download/inspect helpers: `src/skex/data/fetch_public.py`, `src/skex/data/inspect_public.py`
- Kaggle notebook: `notebooks/skex_kaggle.ipynb` (clone `https://github.com/umardrazbhatti-work/skex`, single T4, no plan 01, no Unsloth, no 7B)
- `pytest -q`: 11 passed
- Smoke plan run twice. Both printed `SKIP 5ccd9202035a454f blocked: succeeded run_id=20260922T060737Z-ce62a0`
- `python -m skex.experiments.status`: 1 fingerprint, succeeded

Domain-A JSONL (gitignored; also copied to `Dataset/processed/domain_a/`):

| Split | Rows | Docs |
|---|---:|---:|
| train | 344 | 344 |
| dev | 50 | 50 |
| test | 99 | 99 |

Source task kept: `scierc_ner` only. Evidence quotes checked: 0 missing from the packed input. Train/dev/test document ids do not overlap.

This Windows install blocks PyArrow's parquet DLL (Application Control). The converter falls back to DuckDB. Kaggle's Linux image can use PyArrow. DuckDB is installed locally; it is not a required package dependency.

## Datasets on disk

| Source | Path | State |
|---|---|---|
| SciRIFF 4096 | `Dataset/sciriff/4096/` | Complete. 70521 / 30736 / 35875 |
| SciER | `Dataset/scier/` | Complete. No document overlap. Not a research card |
| SciERC JSONL | `Dataset/scierc/extracted/processed_data/json/` | Complete 350/50/100. ELMo tar was truncated and removed |
| CORD text | `Dataset/cord/text/` | Complete. 800 / 100 / 100. Images deleted. 0 bad JSON |
| Domain-A JSONL | `Dataset/processed/domain_a/` | Written |

Details and the "do not relabel the raw files" decision: `docs/DATASET_INSPECTION.md`.

## Git

Pushed to https://github.com/umardrazbhatti-work/skex on `main`.

Commit: `13ddd7da323ced622db2f3307d8a669e2130d1ab`

Message: `Initial SKEX workspace: plans, registry, Kaggle notebook.`

That commit also contains the SciRIFF converter, the Kaggle notebook, and these notes. There is no second commit. Unstaging the converter before the first commit failed because the repository had no HEAD yet, and the push had already happened before a split was possible. Do not force-push to rewrite it.

## Not done — next session

- Do not force-push. Remote is `https://github.com/umardrazbhatti-work/skex.git`, branch `main`.
- CORD text is downloaded. Do not re-download it. The Domain-B converter is still unwritten.
- Do not install Unsloth. Do not download a 7B. Do not run `experiments/plans/01_tax_zeroshot.yaml` until you mean to spend T4 quota. Zero-shot cells are defined in that plan. Domain-A JSONL now exists, so the data gate for that plan is met, but it was not run.
- QLoRA plan `02` stays blocked until those zero-shot cells have been run.
- Domain B converter is not written. Freeze stays `cord` in `configs/default.yaml`.
- SciER and raw SciERC are downloaded only. They are not converted.

## Kaggle run 22-9-26 15:15

Log: `Results/22-9-26 1515Hrs/`. Two T4s worked. Qwen downloaded. The prompt-JSON cells crashed: `apply_chat_template` returned a token dict, and `model.generate` treated that dict as a tensor (`AttributeError: shape`). The Outlines cells finished all 50 dev rows: 1.5B field F1 0.118, 3B field F1 0.211. Both have wrong-valid 1.0. Those two succeeded fingerprints must not be repeated. The next run passes `--retry-failed`, so only the two prompt-JSON cells run again.

## Kaggle run 22-9-26 14:45

Log: `Results/22-9-26 1445Hrs/`. Two T4s were visible and accepted. Smoke passed. Plan 01 stopped because the notebook has no secret named `HF_TOKEN` (`No user secrets exist ... label HF_TOKEN`). Llama 3.2 cannot be downloaded without that secret. Plan 01 now uses public `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`. No token required.

## Kaggle run 22-9-26 14:30

Log: `Results/22-9-26 1430Hrs/`. Smoke passed again. Plan 01 did not start. The session had no GPU (`nvidia-smi` missing). The cell stopped before Llama. This account cannot switch the accelerator off T4 x2. That is accepted. The notebook now pins `CUDA_VISIBLE_DEVICES=0` and does not refuse two T4s. A CPU session still cannot run plan 01.

## Next: plan 01, untouched models

Wired in code, not run yet. `experiments/plans/01_tax_zeroshot.yaml` is four dev cells: Llama-3.2-1B and 3B, each with prompt-JSON and Outlines. Domain A dev has 50 rows, so each cell scores 50. No QLoRA. No 7B.

The Kaggle notebook's last cell runs that plan. Turn the Accelerator on. T4 x2 is accepted and only GPU 0 is used. A secret named `HF_TOKEN` is required. The Llama 3.2 license must be accepted on Hugging Face. The cell rewrites `/kaggle/working/skex-output.zip`.

## Kaggle run 22-9-26 14:25

Log and zip: `Results/22-9-26 1425Hrs/`. Verdict: plumbing pass. Not a paper result.

Dataset path worked. Domain A 344 / 50 / 99. Pytest 11 passed. Smoke `20260922T090235Z-fcb52a` succeeded on one real dev row (`scierc_ner:validation:0`) with the dummy empty card: schema valid, field F1 0, wrong-valid 1. Second call skipped the same fingerprint. `skex-output.zip` was written (11 files) and is in the Results folder. No GPU was attached. Do not cite these metrics as model performance.

## Kaggle run 22-9-26 14:15

Log: `Results/22-9-26 1415Hrs/specialization-beats-scale.log`.

The run succeeded. Dataset paths resolved. Domain A copied: 344 / 50 / 99. Pytest: 11 passed. First smoke wrote `20260922T085500Z-7f6443` (`OK`, f1=0.000 on the dummy smoke cell). Second smoke printed `SKIP`. No GPU was visible. There is no zip in that Results folder because the notebook never built one. Kaggle's log download does not include `outputs/runs/`. The notebook now writes `/kaggle/working/skex-output.zip` after the smoke cell. Re-upload `notebooks/skex_kaggle.ipynb`, then download that zip from the Output tab into the Results subfolder next to the log.

## Kaggle run 22-9-26 14:00

Log: `Results/22-9-26 1400Hrs/specialization-beats-scale.log`.

Failed in cell 1 before any dataset load. Kaggle showed two T4s (`GPU 0` and `GPU 1`). The notebook raised `SystemExit: Stop. Expected one T4, found 2.` The dataset path was never reached.

The notebook no longer aborts a non-training run for that. It warns and sets `CUDA_VISIBLE_DEVICES=0`. A 2x T4 session still costs double quota until the Accelerator is set to GPU T4 x1. Re-upload `notebooks/skex_kaggle.ipynb` before the next Save & Run.

## Kaggle, when you are ready

Dataset name / slug to type on Kaggle: `skex-datasets`

Zip to upload: `C:\Users\PC1\Desktop\Specialization Beats Scale G\skex-datasets.zip`

Kaggle mounts this account's copy at `/kaggle/input/datasets/umardrazbhatti/skex-datasets`. The notebook uses that path, not `/kaggle/input/skex-datasets`. One zip holds every source (SciRIFF, SciER, SciERC, CORD text, and the processed Domain-A JSONL). The runner only copies `processed/domain_a`.

1. Create a Kaggle dataset named `skex-datasets` and upload that zip.
2. Upload `notebooks/skex_kaggle.ipynb`. Add Data -> `skex-datasets`.
3. Settings: Internet on, one T4. Do not enable a second GPU.
4. Run all cells. The data cell must print the five paths and the Domain-A row counts before smoke runs.
