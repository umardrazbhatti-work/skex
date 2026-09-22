# Open in VS Code + Grok Code

1. File → Open Folder → `skex/` (this directory, not the parent `artifacts/` folder).
2. Enable Grok Code / Grok Bot on the workspace.
3. Paste `docs/HANDOFF.md` as the first chat message.
4. Optional context to attach:
   - `AGENTS.md`
   - `docs/LOCKED_DECISIONS.md`
   - `../Research Paper/Supervisor_Topic_Approval_Proposal.docx` (human proposal; do not re-derive the topic)
5. First command the agent should run:

```bash
pip install -e ".[dev]"
pytest -q
python -m skex.experiments.runner --plan experiments/plans/00_smoke.yaml
python -m skex.experiments.runner --plan experiments/plans/00_smoke.yaml
python -m skex.experiments.status
```

The second smoke run must print SKIP. That is the anti-repeat gate.
