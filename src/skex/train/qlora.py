from __future__ import annotations
from pathlib import Path
from typing import Any


def train_qlora(cfg: dict[str, Any], train_path: Path, out_dir: Path) -> Path:
    raise NotImplementedError(
        "Implement QLoRA in skex.train.qlora.train_qlora. "
        "Save adapter under outputs/adapters/<run_id> and checkpoint every cfg['train']['save_every_steps'] steps."
    )
