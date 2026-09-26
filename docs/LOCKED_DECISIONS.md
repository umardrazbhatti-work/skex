# Locked decisions

The supervisor proposal of 21 September 2026 is the authority. The phase list is `docs/EXECUTION_PLAN.md`. Where this file is narrower than that proposal, follow the proposal.

These choices stay closed. Do not re-litigate them in code comments or new markdown files.

## Paper
- Title: When Does Specialization Beat Scale for Structured Knowledge Extraction?
- Topic 1 is the paper. Topic 2 (constraint tax × SFT) is the central experiment. Topic 3 (span-support) is a required metric.
- Do not publish Topic 1 and Topic 2 as two papers.

## Domains
- A: scientific papers. Public sources: SciRIFF (`allenai/SciRIFF`), SciERC, SciER. SciREX only for fields attested in the packed input.
- B: pick one before training and freeze it in `configs/default.yaml` → `data.domain_b`. Preferred: CORD or SROIE text fields. RealKIE is allowed. Do not add a third domain in v1.
- Out: clinical / EHR, vision-first Donut as the main system, contract-titled replication of Olava Extract.

## Models
- Probe: `meta-llama/Llama-3.2-1B-Instruct`
- Primary: `meta-llama/Llama-3.2-3B-Instruct` or `Qwen/Qwen2.5-3B-Instruct`
- Ceiling: `Qwen/Qwen2.5-7B-Instruct` or `microsoft/Phi-3.5-mini-instruct`
- Encoder baseline: SciBERT on SciERC spans
- Generalist: one API snapshot, id + date frozen in config. Default planning rate card used GPT-4.1 list prices; the actual id is whatever the human sets.

## Training
- QLoRA 4-bit only for v1. No DPO/GRPO until SFT numbers exist.
- One GPU. Unsloth-class stack when available, PEFT otherwise.

## Decoding
- Arm P: prompt-JSON, schema in the instruction, no token mask.
- Arm C: same prompt, grammar/schema enforced by **one** engine (`decode.engine` in config).
- Never mix engines inside one table.

## Metrics (all required)
parse_rate, schema_valid, field_f1, wrong_valid, span_support, gated_precision, consistency (3 paraphrases on Base), latency_ms, tokens_in, tokens_out, cost_usd.

## Scoring rule
A non-null field without a quote that is a contiguous substring of the packed document is `unsupported`, even if the value matches gold.
Gold fields not present in the packed input are `unscorable` and excluded from F1.

## Compute
- Lean can finish on Kaggle (~30 h/week T4).
- Base eval should be sliced across weeks or moved to a rented 24GB weekend box.
- Dual T4 on Kaggle costs 2× quota. Default is single T4.
