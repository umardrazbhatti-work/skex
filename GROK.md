# Grok Code / Grok Bot instructions

Read `AGENTS.md` first, then `docs/LOCKED_DECISIONS.md`, then `docs/EXECUTION_PLAN.md`.

You are Grok Code working in this VS Code workspace (`skex/`).

On every user request:
1. State the smallest plan (files + commands).
2. Check `experiments/registry.jsonl` so you do not repeat a failed or finished fingerprint.
3. Implement inside `src/skex/` only.
4. Run the smallest test that can fail (`pytest -q` or `python -m skex.experiments.runner --plan experiments/plans/00_smoke.yaml`).
5. Write run artifacts under `outputs/runs/<run_id>/`.
6. Update the registry. Do not leave a running status hanging.

If the user pastes the supervisor proposal, treat it as context, not as a request to change the topic.
