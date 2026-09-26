# Execution plan

The authority is the supervisor proposal of 21 September 2026: `Proposal V1 - Specialization Beats Scale 21 - 9 -26.docx`. These phases are that work plan. Finish the phase that is open before starting the next one. Do not drop a proposal step because an earlier, narrower run already exists.

The v0.2 Kaggle scores in `Results/25-9-26 1400Hrs` and `Results/25-9-26 2000Hrs` are a pilot. The schema then allowed a missing key. The proposal says a missing value is null, and every non-null value needs a quote from the document. Those pilot numbers are not the paper table. Do not start the next Kaggle run until Phase 1 has been copied into the Kaggle dataset.

## Phase 1 — Freeze the schema and both datasets

Proposal weeks 1–2. Done on disk. The Kaggle upload must be replaced before Phase 2.

- [x] Domain A schema `research_card.v1`: every card key is present. An unsupported value is null, not a missing key. `evidence_spans` quotes every non-null value.
- [x] Domain B frozen as CORD. Schema `cord_receipt.v1`. Receipt images stay out. Gold values are kept only when they are substrings of the receipt text.
- [x] SciRIFF `scierc_ner` cards, with clinical rows and multi-document inputs removed. 344 train, 50 dev, 99 test.
- [x] SciER document cards. Labels stay Task, Method, and Dataset. Dataset is stored under `datasets`. 80 train, 10 dev, 10 test, 6 test_ood.
- [x] SciREX, only for values that are substrings of the packed input. Material is stored under `datasets`. 306 train, 66 dev, 66 test. None were dropped.
- [x] No shared document id across train, dev, test, or SciER test_ood.
- [x] Counts printed. Domain A together: 730 train, 126 dev, 175 test, 6 test_ood. Domain B: 800 train, 100 dev, 100 test. Every card validated. Every kept quote is in its document.

The proposal's target is about 1,000 scientific cards if the filtered split allows, plus 200–400 receipt cards. The scientific total is 1,037. Biomedical NER and clinical tasks were not relabeled.

Exit met: both domains have train, dev, and test JSONL on disk.

## Phase 2 — Zero-shot tax, before any training

Proposal week 3. Starts only after Phase 1 is closed and the Kaggle dataset matches those files.

Cells A and B on the same items. A is prompt-JSON. B is Outlines. One engine only.

- [ ] Llama 3.2 1B, dev and test
- [ ] Llama 3.2 3B, dev and test
- [ ] Qwen2.5 3B, dev and test
- [ ] Qwen2.5 1.5B, dev and test (already the small Qwen probe; it stays so the pair is complete)
- [ ] Write the before-training tax table: field F1 and wrong-valid, A against B, plus parse rate, schema validity, span support, gated precision, latency, and tokens

Exit: that table exists, and no training has been run.

## Phase 3 — QLoRA on Domain A

Proposal weeks 4–6, training half.

- [ ] QLoRA 4-bit, Qwen2.5 3B or Llama 3.2 3B, on the Domain A training cards. The primary 3B is the one that leads the zero-shot table. Checkpoint every 200 steps.
- [ ] Cells C and D on the same test items as Phase 2. C is prompt-JSON with the adapter. D is Outlines with the adapter.
- [ ] Report the tax again: A→B before training, C→D after training.
- [ ] Consistency: three instruction paraphrases on the base model and on the tuned model.

Exit: cells C and D have metrics, and the before/after tax is written down.

## Phase 4 — Generalist and the 7B ceiling

Proposal weeks 4–6, comparison half. Research question 3.

- [ ] Freeze one API model id and the date in `configs/default.yaml` before calling it.
- [ ] Score that API on the same test items, prompt-JSON and JSON-mode if the API has it.
- [ ] Zero-shot cells A and B for the ceiling model: `Qwen/Qwen2.5-7B-Instruct`, or `microsoft/Phi-3.5-mini-instruct` if the 7B does not fit the GPU.
- [ ] Small-versus-large table: field F1, span support, consistency, latency, tokens, and cost.

Exit: that table exists.

## Phase 5 — Transfer and the encoder

Proposal weeks 7–8.

- [ ] Score the Domain A adapter on Domain B with no further training.
- [ ] Optional: train on Domain A and Domain B together, only after the Domain A numbers exist.
- [ ] SciBERT span baseline on SciERC.

Exit: a transfer table and an encoder table.

## Phase 6 — Human review

Proposal week 9.

- [ ] 100 items, 50 from each domain, reviewed by a person.
- [ ] Agreement numbers written down.
- [ ] API snapshot id and date already frozen in Phase 4.

Exit: the agreement numbers exist.

## Phase 7 — Paper

Proposal weeks 10–12.

- [ ] Write the paper from the tables. A negative "small beats large" result stays in scope if the tax table is complete.
- [ ] Package the code and the frozen schemas.

Exit: a draft for the supervisor.

## Out of this plan

Clinical data. A contract-versus-GPT title. A SciRIFF-only title. Vision-first receipts. A second paper. DPO or GRPO before the supervised numbers exist.
