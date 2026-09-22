from __future__ import annotations
from typing import Any
import json


def generate(prompt: str, *, arm: str, engine: str, schema: dict | None = None, **kwargs: Any) -> str:
    """Decode one prompt. Smoke path returns a valid empty card so the runner can be tested without a GPU."""
    if arm not in {"prompt_json", "constrained", "smoke"}:
        raise ValueError(f"unknown decode arm: {arm}")
    if arm == "smoke":
        return json.dumps({
            "task": [], "method": [], "datasets": [], "metrics": [],
            "scores": [], "claims": [], "limitations": [], "evidence_spans": [],
        })
    raise NotImplementedError(
        f"decode arm={arm} engine={engine} is not wired yet. Implement Outlines or XGrammar here. Do not add a second engine to the comparison."
    )
