from __future__ import annotations
from typing import Any


DEFAULT_PRIORITY = ("title", "abstract", "method", "experiments", "limitations")


def pack_document(sections: dict[str, str], *, max_chars: int = 8000, priority: list[str] | None = None) -> str:
    """Deterministic packer. Keep this frozen so fingerprints stay stable."""
    order = list(priority or DEFAULT_PRIORITY)
    parts = []
    for key in order:
        text = (sections.get(key) or "").strip()
        if text:
            parts.append(f"## {key}\n{text}")
    extra = [k for k in sections if k not in order and (sections.get(k) or "").strip()]
    for key in extra:
        parts.append(f"## {key}\n{sections[key].strip()}")
    packed = "\n\n".join(parts)
    if len(packed) > max_chars:
        packed = packed[:max_chars]
    return packed
