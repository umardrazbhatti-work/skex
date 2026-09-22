# Session status

Read this file first at the next session, then `GROK.md`, `docs/LOCKED_DECISIONS.md`, `docs/EXECUTION_PLAN.md`, and `docs/DATASET_INSPECTION.md`.

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

## Not done — next session

- Push if the GitHub credentials were not available in this session. Remote: `https://github.com/umardrazbhatti-work/skex.git`. Branch `main` only. No force-push.
- CORD text is downloaded. Do not re-download it. The Domain-B converter is still unwritten.
- Do not install Unsloth. Do not download a 7B. Do not run `experiments/plans/01_tax_zeroshot.yaml` until you mean to spend T4 quota. Zero-shot cells are defined in that plan. Domain-A JSONL now exists, so the data gate for that plan is met, but it was not run.
- QLoRA plan `02` stays blocked until those zero-shot cells have been run.
- Domain B converter is not written. Freeze stays `cord` in `configs/default.yaml`.
- SciER and raw SciERC are downloaded only. They are not converted.

## Kaggle, when you are ready

1. Upload `notebooks/skex_kaggle.ipynb`.
2. Settings: Internet on, one T4. Do not enable a second GPU.
3. Upload the `Dataset/` folder as a Kaggle dataset and attach it.
4. Run the notebook. It clones the GitHub repo. The second smoke cell must print SKIP after the first smoke succeeds on that machine.
