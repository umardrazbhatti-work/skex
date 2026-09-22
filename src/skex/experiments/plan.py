from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from skex.paths import resolve


def load_plan(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    with path.open() as f:
        plan = yaml.safe_load(f)
    if "plan_id" not in plan or "jobs" not in plan:
        raise ValueError(f"plan {path} needs plan_id and jobs")
    return plan
