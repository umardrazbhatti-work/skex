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
Qwen dev: `experiments/plans/01_tax_zeroshot.yaml` (finished, sealed).
Qwen test: `experiments/plans/01b_tax_zeroshot_test.yaml`.
Llama dev: `experiments/plans/01c_tax_zeroshot_llama_dev.yaml`.
Llama test: `experiments/plans/01d_tax_zeroshot_llama_test.yaml`.
- [x] Cells A/B on Qwen2.5 1.5B and 3B, dev first (50 docs). Results: `23-9-26 1515Hrs`.
- [x] Same four Qwen cells on the 99 test papers. Results: `24-9-26 1400Hrs`.
- [ ] Llama 3.2 1B and 3B, prompt-JSON and Outlines, dev then test. License accepted 22 Sep 2026. Needs Kaggle secret `HF_TOKEN`.
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
