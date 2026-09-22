# Dataset drop for Kaggle

This folder is not committed. `Dataset/**` is gitignored except this file.

Download the public sources from the skex repo root:

```bash
pip install -e ".[dev]" pyarrow huggingface_hub
python -m skex.data.fetch_public --dest Dataset
```

What lands here:

- `sciriff/` — `allenai/SciRIFF` config `4096` (the configuration used in the SciRIFF paper)
- `scier/` — SciER LLM and PLM JSONL from `github.com/edzq/SciER`
- `scierc/` — official SciERC tarball when the University of Washington URL answers
- `cord/text/` — CORD-v2 `ground_truth` text only (800/100/100). Receipt images are not kept.
- `processed/domain_a/` — SciRIFF mapped to the research card (344/50/99).

Upload one Kaggle dataset named `skex-datasets` (that exact slug) using `skex-datasets.zip`. Do not put model weights in it.

The inspection write-up is `docs/DATASET_INSPECTION.md`.
