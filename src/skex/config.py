from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from skex.paths import ROOT, resolve


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = resolve(path or ROOT / "configs" / "default.yaml")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg
