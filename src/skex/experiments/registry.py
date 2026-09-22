from __future__ import annotations
from datetime import datetime, timezone
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

from skex.paths import resolve


TERMINAL = {"succeeded", "failed", "aborted"}
RETRYABLE_DEFAULT = {"timeout", "api", "unknown"}
NON_RETRYABLE = {"oom", "cuda_arch", "schema_invalid", "data"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Registry:
    """Append-only experiment log. One JSON object per line.

    should_run() is the gate that stops the agent repeating work.
    """

    def __init__(self, path: str | Path, failures_path: str | Path | None = None):
        self.path = resolve(path)
        self.failures_path = resolve(failures_path) if failures_path else self.path.parent / "failures.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_bytes().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(orjson.loads(line))
        return rows

    def latest_by_fp(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._read():
            fp = row.get("fingerprint")
            if fp:
                latest[fp] = row
        return latest

    def find(self, fingerprint: str) -> dict[str, Any] | None:
        return self.latest_by_fp().get(fingerprint)

    def should_run(
        self,
        fingerprint: str,
        *,
        retry_failed: bool = False,
        retryable: set[str] | None = None,
    ) -> tuple[bool, str]:
        row = self.find(fingerprint)
        if row is None:
            return True, "new"
        status = row.get("status")
        if status == "running":
            return False, f"blocked: already running run_id={row.get('run_id')}"
        if status == "succeeded":
            return False, f"blocked: succeeded run_id={row.get('run_id')}"
        if status == "aborted":
            return False, f"blocked: aborted run_id={row.get('run_id')} (pass --retry-failed to retry)"
        if status == "failed":
            klass = row.get("failure_class") or "unknown"
            if not retry_failed:
                return False, f"blocked: failed ({klass}) run_id={row.get('run_id')} — change a fingerprint field or pass --retry-failed"
            allowed = retryable if retryable is not None else set(RETRYABLE_DEFAULT)
            if klass in NON_RETRYABLE and klass not in allowed:
                return False, f"blocked: non-retryable failure_class={klass}. Change seq_len/batch/model to mint a new fingerprint."
            if klass not in allowed and klass not in RETRYABLE_DEFAULT:
                return False, f"blocked: failure_class={klass} not in retryable set {sorted(allowed)}"
            return True, f"retry_failed:{klass}"
        return True, f"unrecognized_status:{status}"

    def append(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("logged_at", _now())
        with self.path.open("ab") as f:
            f.write(orjson.dumps(record) + b"\n")
        if record.get("status") == "failed":
            self.failures_path.parent.mkdir(parents=True, exist_ok=True)
            with self.failures_path.open("ab") as f:
                f.write(orjson.dumps(record) + b"\n")

    def start(self, spec: dict[str, Any], run_id: str, fingerprint: str) -> dict[str, Any]:
        rec = {
            "run_id": run_id,
            "fingerprint": fingerprint,
            "status": "running",
            "started_at": _now(),
            "spec": spec,
        }
        self.append(rec)
        return rec

    def finish(self, run_id: str, fingerprint: str, spec: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
        self.append({
            "run_id": run_id,
            "fingerprint": fingerprint,
            "status": "succeeded",
            "ended_at": _now(),
            "spec": spec,
            "metrics": metrics or {},
        })

    def fail(
        self,
        run_id: str,
        fingerprint: str,
        spec: dict[str, Any],
        *,
        failure_class: str,
        error: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        rec = {
            "run_id": run_id,
            "fingerprint": fingerprint,
            "status": "failed",
            "ended_at": _now(),
            "spec": spec,
            "failure_class": failure_class,
            "error": error[:2000],
        }
        if extra:
            rec["extra"] = extra
        self.append(rec)

    def abort(self, run_id: str, fingerprint: str, spec: dict[str, Any], reason: str) -> None:
        self.append({
            "run_id": run_id,
            "fingerprint": fingerprint,
            "status": "aborted",
            "ended_at": _now(),
            "spec": spec,
            "error": reason,
        })

    def rows(self) -> Iterator[dict[str, Any]]:
        yield from self._read()
