from __future__ import annotations
from typing import Any


def describe_backend() -> dict[str, Any]:
    info: dict[str, Any] = {"unsloth": False, "transformers": False, "cuda": False}
    try:
        import unsloth  # noqa: F401
        info["unsloth"] = True
    except Exception:
        pass
    try:
        import transformers  # noqa: F401
        info["transformers"] = True
    except Exception:
        pass
    try:
        import torch
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def load_causal(model_id: str, *, fourbit: bool = True):
    """GPU entry point. Implemented when [train] extras are present."""
    raise NotImplementedError(
        f"load_causal({model_id!r}) needs the train extra and a GPU. "
        "Implement in skex.models.load using Unsloth FastLanguageModel when available."
    )
