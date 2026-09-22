from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
try:
    import orjson
except ImportError:
    import json as _json
    class orjson:
        OPT_INDENT_2 = 1

        @staticmethod
        def dumps(obj, option=None):
            indent = 2 if option else None
            return _json.dumps(obj, indent=indent).encode()

        @staticmethod
        def loads(b):
            return _json.loads(b)



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLogger:
    """Append-only JSONL event log inside a run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"

    def emit(self, kind: str, **payload: Any) -> None:
        rec = {"ts": _now(), "kind": kind, **payload}
        with self.path.open("ab") as f:
            f.write(orjson.dumps(rec) + b"\n")

    def write_json(self, name: str, obj: Any) -> Path:
        dest = self.run_dir / name
        dest.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
        return dest
