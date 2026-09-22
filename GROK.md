# Grok Code / Grok Bot instructions

Read `AGENTS.md` first, then `docs/LOCKED_DECISIONS.md`, then `docs/EXECUTION_PLAN.md`.

You are Grok Code working in this VS Code workspace (`skex/`).

Run logs from Kaggle live outside this repo. The parent path does not change:

`C:\Users\PC1\Desktop\Specialization Beats Scale G\Results`

Each run is a new subfolder there (the first one is `22-9-26 1400Hrs`). When the user says a run failed or names a new results folder, read that subfolder before editing code. Do not move or rename the Results directory.

On every user request:
1. State the smallest plan (files + commands).
2. Check `experiments/registry.jsonl` so you do not repeat a failed or finished fingerprint.
3. Implement inside `src/skex/` only.
4. Run the smallest test that can fail (`pytest -q` or `python -m skex.experiments.runner --plan experiments/plans/00_smoke.yaml`).
5. Write run artifacts under `outputs/runs/<run_id>/`.
6. Update the registry. Do not leave a running status hanging.

If the user pastes the supervisor proposal, treat it as context, not as a request to change the topic.
