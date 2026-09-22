# SKEX — Structured Knowledge EXtraction

Research code for the locked paper:

**When Does Specialization Beat Scale for Structured Knowledge Extraction?**

Small QLoRA models (≤7B) vs a frozen generalist API, on a frozen JSON schema, with paired prompt-JSON vs constrained decoding. Headline metrics are field F1, wrong-valid rate, and span-support — not “JSON parsed.”

## Open this folder in VS Code
This directory is the Grok Code workspace root.

```
skex/
  AGENTS.md          standing rules for the agent
  GROK.md            Grok Code entry
  docs/              locked decisions + week plan
  schemas/           frozen JSON Schemas
  configs/           default experiment config
  src/skex/          modular library
  experiments/plans  YAML plans the runner executes
  scripts/           CLI
  tests/             no-GPU tests
```

## Install (CPU / tests)

```bash
pip install -e ".[dev]"
pytest -q
python -m skex.experiments.runner --plan experiments/plans/00_smoke.yaml
python -m skex.experiments.status
```

GPU extra (`[train]`) is Unsloth / PEFT and is optional until you are on Kaggle T4 or a 16–24GB card.

## Do not
- Fine-tune first. Zero-shot tax cells come first.
- Title the work as SciRIFF-only or contracts-vs-GPT.
- Re-run a fingerprint marked `failed` without `--retry-failed`.
