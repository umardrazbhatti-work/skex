from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator
try:
    import orjson
except ImportError:
    import json as _json
    class orjson:
        @staticmethod
        def dumps(obj, option=None):
            return (_json.dumps(obj) + '\n').encode() if False else _json.dumps(obj).encode()
        @staticmethod
        def loads(b):
            return _json.loads(b)



REQUIRED = ("id", "domain", "instruction", "input", "output")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_bytes().splitlines():
        line = line.strip()
        if line:
            rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in Path(path).read_bytes().splitlines():
        line = line.strip()
        if line:
            yield orjson.loads(line)


def record_is_valid(row: dict[str, Any]) -> bool:
    return all(k in row for k in REQUIRED)
