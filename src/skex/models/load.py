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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def release_causal(loaded) -> None:
    """Drop decoder state. The caller must also drop its own reference to `loaded`."""
    from skex.decode.interface import clear_decoder_cache
    clear_decoder_cache()
    del loaded
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
