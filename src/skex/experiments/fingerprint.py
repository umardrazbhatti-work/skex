from __future__ import annotations
from typing import Any
import hashlib
import json


FINGERPRINT_KEYS = (
    "plan_id",
    "cell",
    "model_id",
    "adapter_id",
    "decode_arm",
    "decode_engine",
    "schema_id",
    "split",
    "data_rev",
    "seq_len",
    "lora_r",
    "train_examples",
    "epochs",
)


def canonical(spec: dict[str, Any]) -> dict[str, Any]:
    out = {k: spec.get(k) for k in FINGERPRINT_KEYS}
    return out


def fingerprint(spec: dict[str, Any]) -> str:
    blob = json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
