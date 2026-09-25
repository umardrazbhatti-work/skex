# Execution plan

Agent: finish the current phase before starting the next. Update `experiments/registry.jsonl` after every run.

## Phase 0 — repo smoke (no GPU)
- [ ] `pytest -q` green
- [ ] smoke plan writes a run folder and a registry row
- [ ] `python -m skex.experiments.status` shows the row

## Phase 1 — data
- [ ] SciRIFF structured/JSON tasks → `data/processed/domain_a/{train,dev,test}.jsonl`
- [ ] Document-level decontamination by paper id
- [ ] Packer with section priority: title, abstract, method, experiments, limitations
- [ ] Domain B converter behind `data.domain_b` (CORD or SROIE)
- [ ] Print split counts; stop if test ids leak into train

## Phase 2 — zero-shot tax (GPU)
Qwen dev: `experiments/plans/01_tax_zeroshot.yaml` (v0.1 sealed; v0.2 finished in `25-9-26 1400Hrs`).
Qwen test: `experiments/plans/01b_tax_zeroshot_test.yaml`. The 1.5B half finished on 25 Sep. The 3B half is `experiments/plans/01e_qwen3b_test.yaml`.
Llama 1B: `experiments/plans/01f_llama1b_dev.yaml`, then `experiments/plans/01g_llama1b_test.yaml`.
Llama 3B: `experiments/plans/01h_llama3b_dev.yaml`, then `experiments/plans/01i_llama3b_test.yaml`.
The Kaggle notebook runs those five plans in that order. Each plan is one process and one model.
- [x] Cells A/B on Qwen2.5 1.5B and 3B, dev first (50 docs). Results: `23-9-26 1515Hrs`. v0.2 rerun finished in `25-9-26 1400Hrs`.
- [x] Same four Qwen cells on the 99 test papers. Results: `24-9-26 1400Hrs`. v0.2 rerun finished the 1.5B half only.
- [ ] Qwen2.5-3B test, then Llama 3.2 1B dev and test, then Llama 3.2 3B dev and test. Needs Kaggle secret `HF_TOKEN`.
- [x] Dev `metrics.json` has validity, field F1, wrong-valid, span-support

## Phase 3 — QLoRA 3B Domain A
Plan: `experiments/plans/02_sft_3b_domain_a.yaml`
- [ ] Checkpoint adapter every 200 steps to `outputs/adapters/` **and** Hugging Face if token present
- [ ] Cells C/D on the same test ids as Phase 2

## Phase 4 — generalist + transfer
- [ ] Frozen API on the same items (prompt + JSON-mode if available)
- [ ] Domain B zero-shot with the Domain-A adapter
- [ ] Optional A+B SFT only after A numbers exist

## Phase 5 — encoder + human
- [ ] SciBERT span baseline
- [ ] 100-item human file exported from `skex.eval.export_human`

## Phase 6 — ablation table
```bash
python -m skex.experiments.ablate
```
Writes `experiments/ablations/latest.csv` from the registry. That CSV is the paper table draft.

If a phase fails, log the failure class and stop that fingerprint. Propose a *new* fingerprint (shorter seq, smaller batch) rather than rerunning the dead one.
