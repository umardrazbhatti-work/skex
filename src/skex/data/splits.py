from __future__ import annotations
from collections import defaultdict
from typing import Any


def assert_no_id_leak(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    tr = {r["id"] for r in train}
    te = {r["id"] for r in test}
    leak = tr & te
    if leak:
        raise ValueError(f"train/test id leak: {sorted(leak)[:10]}")


def assert_no_doc_leak(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    key: str = "doc_id",
) -> None:
    """Fail when the same paper/document id is in both splits."""
    tr = {r[key] for r in train if r.get(key)}
    te = {r[key] for r in test if r.get(key)}
    leak = tr & te
    if leak:
        raise ValueError(f"train/test {key} leak: {sorted(leak)[:10]}")


def counts_by_domain(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[str(r.get("domain"))] += 1
    return dict(c)
