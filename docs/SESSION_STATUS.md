# Session status

Read this file first at the next session, then `GROK.md`, `docs/LOCKED_DECISIONS.md`, `docs/EXECUTION_PLAN.md`, and `docs/DATASET_INSPECTION.md`.

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

## Kaggle run 22-9-26 14:30

Log: `Results/22-9-26 1430Hrs/`. Smoke passed again. Plan 01 did not start. The session had no GPU (`nvidia-smi` missing). The cell stopped with `Plan 01 needs one T4`. That is a session setting, not a model bug. Next run must use Accelerator GPU T4 x1. Notebook metadata now requests `nvidiaTeslaT4`.

## Next: plan 01, untouched models

Wired in code, not run yet. `experiments/plans/01_tax_zeroshot.yaml` is four dev cells: Llama-3.2-1B and 3B, each with prompt-JSON and Outlines. Domain A dev has 50 rows, so each cell scores 50. No QLoRA. No 7B.

The Kaggle notebook's last cell runs that plan. It needs Accelerator **GPU T4 x1** and a secret named `HF_TOKEN`. The Llama 3.2 license must be accepted on Hugging Face. The cell rewrites `/kaggle/working/skex-output.zip`.

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
