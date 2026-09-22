# Short context for Grok Bot (non-code chat)

This project already chose a topic. Do not run another literature-review loop.

Question we will measure:
Given a document and a frozen JSON schema, can a QLoRA model ≤7B produce schema-valid, source-faithful, field-correct objects more accurately and cheaply than a larger general model that is only prompted or constrained — and does domain SFT remove the constraint tax?

Constraint tax = constrained decoding raises schema validity while raising the rate of valid-but-wrong outputs, or lowering field F1. Constraint Tax (arXiv:2605.26128) measured this on untuned sub-3B models. We measure it on documents, before and after SFT.

Public data only. Kaggle T4 is the default GPU. Log every run. Do not repeat failures.
