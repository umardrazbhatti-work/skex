# Dataset inspection — 2026-09-22

The public files that downloaded are valid releases. They are not corrupt, and the raw files were not edited. They are **not** already SKEX training JSONL. Domain A still needs the converter in `src/skex/data/convert_sciriff.py`. Do not relabel the source annotations.

Local drop: `Dataset/` (gitignored except this note's sibling `Dataset/README.md`). Upload that folder to Kaggle. Processed Domain-A JSONL is also copied to `Dataset/processed/domain_a/`.

## SciRIFF `allenai/SciRIFF` config 4096

Paper configuration. Three parquet files, 294.5 MB.

| Split | Rows in the file | Rows on the dataset card |
|---|---:|---:|
| train | 70521 | 70521 |
| validation | 30736 | 30736 |
| test | 35875 | 35875 |

Total 137,132. Counts match the Hugging Face card. Columns are `input`, `output`, `metadata`, `_instance_id`. This is instruction text, not the research-card schema.

What the converter did, without changing the parquet:

- Dropped 28,186 rows whose metadata includes `clinical_medicine`. The raw file still has them. Clinical is out of v1.
- Dropped 25,014 `multiple_source` rows. Those inputs are not one paper, so they have no single document id.
- Dropped 39,181 rows whose output is not JSON.
- Dropped 18,030 JSON rows with no `Abstract:` / `Title:` / `Paper:` boundary. The instructions would otherwise be treated as the document.
- Dropped 26,228 JSON rows whose labels are not research-card fields (biomedical NER types, untyped relation triples, and similar). `scierc_re` is a list of `[head, tail, RELATION]` with no entity type, so those rows are not mapped.
- SciRIFF often writes a space before commas (`parameters , illumination`). The converter keeps the document's own slice (`parameters, illumination`) only when that span is contiguous. It does not invent a span.

Only `scierc_ner` produced a card. SciRIFF contains 495 of those rows, not the full 500 SciERC abstracts. Two of the 495 have no Task, Method, Material, or Metric spans (only Generic / OtherScientificTerm, or empty lists). Those two are omitted. 493 rows remain.

| SKEX split | Rows | Document ids |
|---|---:|---:|
| train | 344 | 344 |
| dev | 50 | 50 |
| test | 99 | 99 |

No document id is in more than one split. Every stored evidence quote is a substring of that row's packed input (0 failures on the written JSONL).

Known schema mismatch, not a file defect: SciERC's `Material` type is broader than "dataset". Example from `scierc_ner:train:0`: the mention `French` is stored under `datasets` because that is the SciERC data/resource type. The string is in the abstract. It was not relabeled.

## SciER

From `github.com/edzq/SciER`, LLM and PLM JSONL. LLM split:

| File | Rows | Documents | Empty sentences |
|---|---:|---:|---:|
| train.jsonl | 5575 | 80 | 0 |
| dev.jsonl | 713 | 10 | 0 |
| test.jsonl | 854 | 10 | 0 |
| test_ood.jsonl | 580 | 6 | 0 |

Keys: `doc_id`, `sentence`, `ner`, `rel`, `rel_plus`. Shared `doc_id`s across train/dev/test/test_ood: 0. The file is sentence-level NER/RE. It does not need repair. It does need its own converter before it can be a research card. That converter is not in this step.

## SciERC

Annotation JSONL from the University of Washington `sciERC_processed` bundle:

| Split | Documents | Overlap with the other splits |
|---|---:|---:|
| train | 350 | 0 |
| dev | 50 | 0 |
| test | 100 | 0 |

Every line parses. Keys: `doc_key`, `sentences`, `ner`, `relations`, `clusters`. This is the standard 500-abstract release. Token-level JSON, not a research card. No label edits.

The official tarball is 695,340,151 bytes and also contains ELMo `.hdf5` files. The download stopped inside those embedding files. The partial tar and the partial hdf5 files were deleted so a corrupt binary is not uploaded. The annotation JSONL had already been extracted whole. This project does not use ELMo.

## CORD (frozen Domain B)

`configs/default.yaml` already sets `data.domain_b: cord`. The Hugging Face release `naver-clova-ix/cord-v2` is about 2.3 GB because each parquet row mixes a receipt image with a `ground_truth` JSON string. Vision is out of v1. `Dataset/cord/text/` keeps `ground_truth` only. The parquet files were deleted after extraction. No `.parquet` file remains under `Dataset/cord/`.

| Split | Receipts | Bad JSON | Missing `gt_parse` | Menu lines | Empty `nm` |
|---|---:|---:|---:|---:|---:|
| train | 800 | 0 | 0 | 2105 | 8 |
| dev | 100 | 0 | 0 | 221 | 0 |
| test | 100 | 0 | 0 | 251 | 0 |

Every `ground_truth` string parses. Every receipt has `gt_parse.menu`. `sub_total` is absent on some receipts (548/800 train, 67/100 dev, 66/100 test). `total` is absent on 2 train receipts. That is how those receipts were annotated, not a broken download. The 8 empty `nm` values are empty in the source. Do not invent item names or totals.

This text is not yet SKEX Domain-B JSONL. A converter still has to map `gt_parse` onto `schemas/document_kie.schema.json` and keep only quotes that occur in the receipt text. That converter was not written in this session.

## What not to "fix"

- Do not delete clinical rows inside the SciRIFF parquet. Filter them in the converter.
- Do not rewrite SciERC or SciER labels.
- Do not treat SciRIFF `output` as a research card. Most tasks are a different JSON shape.
- Do not upload the CORD images.
