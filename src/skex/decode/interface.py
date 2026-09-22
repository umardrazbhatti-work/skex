"""One decoder interface. Constrained arm uses Outlines only."""
from __future__ import annotations

import json
from typing import Any

FROZEN_ENGINE = "outlines"
_OUTLINES: dict[int, Any] = {}
_CARD = None


def clear_decoder_cache() -> None:
    _OUTLINES.clear()


def card_model():
    """Pydantic copy of the research card. Imported only for the constrained arm."""
    global _CARD
    if _CARD is not None:
        return _CARD
    from pydantic import BaseModel, ConfigDict

    class Score(BaseModel):
        model_config = ConfigDict(extra="forbid")
        metric: str
        value: float | str | None = None
        condition: str | None = None

    class Span(BaseModel):
        model_config = ConfigDict(extra="forbid")
        field: str
        quote: str

    class ResearchCard(BaseModel):
        model_config = ConfigDict(extra="forbid")
        task: list[str]
        method: list[str]
        datasets: list[str]
        metrics: list[str]
        scores: list[Score]
        claims: list[str]
        limitations: list[str]
        evidence_spans: list[Span]

    _CARD = ResearchCard
    return _CARD


def empty_card() -> str:
    return json.dumps({
        "task": [], "method": [], "datasets": [], "metrics": [],
        "scores": [], "claims": [], "limitations": [], "evidence_spans": [],
    })


def build_messages(instruction: str, document: str, schema_text: str) -> list[dict[str, str]]:
    """Same prompt for prompt-JSON and constrained. The document stays in the user turn."""
    system = (
        instruction.strip()
        + "\n\nFollow this JSON schema. Do not add keys.\n"
        + schema_text.strip()
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": document},
    ]


def _usage(tokens_in: int, tokens_out: int) -> None:
    generate.last_usage = {"tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": 0.0}


def _as_json_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump())
    return json.dumps(result)


def _render(tokenizer, messages: list[dict[str, str]], max_input_tokens: int):
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )


def _prompt_json(messages, model, tokenizer, *, max_new_tokens: int, max_input_tokens: int) -> str:
    import torch

    inputs = _render(tokenizer, messages, max_input_tokens).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, inputs.shape[-1]:]
    _usage(int(inputs.shape[-1]), int(new_tokens.shape[-1]))
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _constrained(messages, model, tokenizer, *, max_new_tokens: int, max_input_tokens: int) -> str:
    import outlines

    key = id(model)
    if key not in _OUTLINES:
        _OUTLINES[key] = outlines.from_transformers(model, tokenizer)
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    # The rendered string is not truncated here; the caller already keeps documents short.
    result = _OUTLINES[key](prompt, output_type=card_model(), max_new_tokens=max_new_tokens)
    text = _as_json_text(result)
    encoded_in = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_input_tokens)
    _usage(len(encoded_in["input_ids"]), len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    return text


def generate(
    prompt: str,
    *,
    arm: str,
    engine: str,
    schema: dict | None = None,
    instruction: str | None = None,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    max_input_tokens: int = 1536,
    **kwargs: Any,
) -> str:
    """Decode one document. Smoke returns an empty valid card and does not touch a GPU."""
    del kwargs
    if arm not in {"prompt_json", "constrained", "smoke"}:
        raise ValueError(f"unknown decode arm: {arm}")
    if arm == "smoke":
        _usage(0, 0)
        return empty_card()
    if engine != FROZEN_ENGINE:
        raise ValueError(f"frozen decode engine is {FROZEN_ENGINE}, not {engine}")
    if model is None or tokenizer is None:
        raise RuntimeError(f"decode arm={arm} needs a loaded model. Refusing to guess one.")
    schema_text = json.dumps(schema or {}, indent=2)
    messages = build_messages(instruction or "Extract a research card.", prompt, schema_text)
    if arm == "prompt_json":
        return _prompt_json(messages, model, tokenizer, max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens)
    return _constrained(messages, model, tokenizer, max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens)


generate.last_usage = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
