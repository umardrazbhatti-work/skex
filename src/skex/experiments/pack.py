"""Rewrite the Kaggle output zip while a plan is still running.

The 23 Sep 2026 notebook wrote that zip only after the plan process
returned. The kernel was killed during the next model load, so the
finished prompt-JSON cell never landed in the zip.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from skex.paths import ROOT


def pack_kaggle(note: str, repo: Path | None = None) -> Path | None:
    working = Path("/kaggle/working")
    if not working.is_dir():
        return None
    repo = Path(repo) if repo is not None else ROOT
    zip_path = working / "skex-output.zip"
    tmp = working / "skex-output.zip.tmp"
    include: list[Path] = []
    runs = repo / "outputs" / "runs"
    if runs.is_dir():
        include.extend(path for path in runs.rglob("*") if path.is_file())
    for relative in ("experiments/registry.jsonl", "experiments/failures.jsonl", "data_paths.json"):
        path = repo / relative
        if path.is_file():
            include.append(path)
    text = note if note.endswith("\n") else note + "\n"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("RUN.txt", text)
        for path in include:
            archive.write(path, path.relative_to(repo).as_posix())
    tmp.replace(zip_path)
    print(f"Wrote {zip_path} with {len(include)} files.", flush=True)
    return zip_path
