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
    """Load one causal LM onto cuda:0. 4-bit so a 3B model fits a 16GB T4.

    Returns (model, tokenizer). Does not download a 7B and does not start QLoRA.
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(f"load_causal({model_id!r}) needs a CUDA GPU. Turn the Kaggle accelerator on. T4 x2 is accepted.")
    if torch.cuda.device_count() > 1:
        print("More than one GPU is visible. Weights stay on cuda:0.")
    quant = None
    if fourbit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    if model_id.startswith("meta-llama/") and not token:
        raise RuntimeError(
            "Llama 3.2 is gated. Add a read token from the Hugging Face account "
            "that accepted the license as a Kaggle secret named HF_TOKEN."
        )
    auth = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **auth)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.float16,
        **auth,
    )
    model.eval()
    return model, tokenizer


def release_causal(loaded) -> None:
    """Free a 4-bit model before another from_pretrained.

    On 23 Sep 2026 the second load of Qwen2.5-1.5B hung at 49% of weight
    materialization for the rest of the 12-hour Kaggle session. The first
    model was still referenced, so the CUDA allocator waited forever.
    The caller has to drop its own reference before calling this.
    """
    from skex.decode.interface import clear_decoder_cache
    clear_decoder_cache()
    if loaded is None:
        return
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    if hasattr(model, "past_key_values"):
        try:
            model.past_key_values = None
        except Exception:
            pass
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass
    del model
    del loaded
    import gc
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


class ModelSlot:
    """One causal LM at a time. The same model_id is not loaded twice."""

    def __init__(self, releaser=release_causal):
        self.model_id: str | None = None
        self.loaded = None
        self._releaser = releaser

    def get(self, model_id: str, loader):
        if self.loaded is not None and self.model_id == model_id:
            return self.loaded, "reuse"
        self.release()
        loaded = loader(model_id)
        self.loaded = loaded
        self.model_id = model_id
        return loaded, "load"

    def release(self) -> None:
        loaded = self.loaded
        self.loaded = None
        self.model_id = None
        if loaded is None:
            return
        self._releaser(loaded)
