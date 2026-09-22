# Paste this as the first Grok Code message

You are implementing SKEX in this workspace. Read AGENTS.md, GROK.md, docs/LOCKED_DECISIONS.md, and docs/EXECUTION_PLAN.md before writing code.

Locked title: When Does Specialization Beat Scale for Structured Knowledge Extraction?
Central experiment: four cells — base vs QLoRA SFT, each with prompt-JSON vs one frozen constrained decoder.
Required extra metric: evidence span must be a substring of the packed document.
Hardware assumption: Kaggle single T4 16GB, 30 GPU hours/week, or one local 16–24GB GPU.
Do not change the topic. Do not invent datasets.

Current job: make `pytest -q` and `python -m skex.experiments.runner --plan experiments/plans/00_smoke.yaml` pass, then implement Domain-A converters against the frozen schema. Plan first. Log every run through the registry. Never repeat a failed fingerprint.
