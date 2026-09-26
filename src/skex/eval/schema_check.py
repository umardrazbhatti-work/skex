from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import re
from skex.paths import resolve

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?(.*)$", re.DOTALL)

_CACHE: dict[str, dict] = {}


def _load_schema(schema_path: str | Path) -> dict:
    path = str(resolve(schema_path))
    if path not in _CACHE:
        _CACHE[path] = json.loads(Path(path).read_text())
    return _CACHE[path]


def _validate(obj: Any, schema: dict) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
        return [e.message for e in Draft202012Validator(schema).iter_errors(obj)]
    except ImportError:
        return _lite_validate(obj, schema)


def _lite_validate(obj: Any, schema: dict) -> list[str]:
    errs: list[str] = []
    if schema.get("type") == "object":
        if not isinstance(obj, dict):
            return ["root is not an object"]
        for k in schema.get("required") or []:
            if k not in obj:
                errs.append(f"missing required property {k}")
        if schema.get("additionalProperties") is False:
            allowed = set((schema.get("properties") or {}).keys())
            extra = set(obj) - allowed
            if extra:
                errs.append(f"additional properties {sorted(extra)}")
        props = schema.get("properties") or {}
        for k, v in obj.items():
            sub = props.get(k)
            if not sub:
                continue
            types = sub.get("type")
            type_list = types if isinstance(types, list) else [types]
            if v is None and "null" in type_list:
                continue
            if "array" in type_list and not isinstance(v, list):
                errs.append(f"{k} is not an array")
            if "object" in type_list and not isinstance(v, dict):
                errs.append(f"{k} is not an object")
            if isinstance(v, list) and isinstance(sub.get("items"), dict):
                item = sub["items"]
                if item.get("type") == "object":
                    for i, el in enumerate(v):
                        if not isinstance(el, dict):
                            errs.append(f"{k}[{i}] is not an object")
                            continue
                        for rk in item.get("required") or []:
                            if rk not in el:
                                errs.append(f"{k}[{i}] missing {rk}")
                        if item.get("additionalProperties") is False:
                            allowed = set((item.get("properties") or {}).keys())
                            extra = set(el) - allowed
                            if extra:
                                errs.append(f"{k}[{i}] extra {sorted(extra)}")
    return errs


def _json_object(text: str) -> Any:
    """Parse one JSON object. A fence, a lead-in sentence, or trailing text may surround it."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    last = None
    for match in re.finditer(r"\{", stripped):
        try:
            obj, _end = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError as exc:
            last = exc
            continue
        if isinstance(obj, dict):
            return obj
    if last is not None:
        raise last
    raise json.JSONDecodeError("no JSON object", stripped, 0)


def _candidates(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("{") or text.startswith("["):
        return [text]
    found: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        found.append(fenced.group(1))
    else:
        opened = _OPEN_FENCE.search(text)
        if opened:
            found.append(opened.group(1))
    found.append(text)
    return [item.strip() for item in found if item and item.strip()]


def parse_and_validate(raw: str | dict, schema_path: str | Path) -> dict[str, Any]:
    if isinstance(raw, dict):
        obj: Any = raw
        parse_ok = True
    else:
        obj = None
        parse_ok = False
        error = "not JSON"
        for text in _candidates(str(raw)):
            try:
                obj = _json_object(text)
                parse_ok = True
                break
            except json.JSONDecodeError as exc:
                error = str(exc)
        if not parse_ok:
            return {"parsed": None, "parse_ok": False, "valid": False, "errors": [error]}
    if not isinstance(obj, dict):
        return {"parsed": obj, "parse_ok": True, "valid": False, "errors": ["root is not an object"]}
    errs = _validate(obj, _load_schema(schema_path))
    return {"parsed": obj, "parse_ok": parse_ok, "valid": not errs, "errors": errs}
