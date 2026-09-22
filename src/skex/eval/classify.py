"""Map exceptions to failure_class so the registry can block repeats."""
from __future__ import annotations


def classify(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in msg or "cuda oom" in msg or "cuda out of memory" in msg:
        return "oom"
    if "sm_60" in msg or "pascal" in msg or "cuda capability" in msg:
        return "cuda_arch"
    if "jsonschema" in msg or "schema" in msg and "valid" in msg:
        return "schema_invalid"
    if "file not found" in msg or "dataset" in msg or "no such file" in msg:
        return "data"
    if "timeout" in msg or "deadline" in msg:
        return "timeout"
    if "api" in msg or "rate limit" in msg or "openai" in msg or "anthropic" in msg:
        return "api"
    return "unknown"
