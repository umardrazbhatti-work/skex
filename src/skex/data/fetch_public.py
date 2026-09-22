"""Download the public datasets SKEX is allowed to use.

Writes files under Dataset/. Does not download model weights.
CORD images are not kept: the locked protocol uses receipt text only.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

SCIRIFF_REPO = "allenai/SciRIFF"
SCIRIFF_ALLOW = ["4096/*", "README.md", "card.md"]
SCIER_BASE = "https://raw.githubusercontent.com/edzq/SciER/main/SciER"
SCIER_FILES = (
    "LLM/train.jsonl",
    "LLM/dev.jsonl",
    "LLM/test.jsonl",
    "LLM/test_ood.jsonl",
    "PLM/train.jsonl",
    "PLM/dev.jsonl",
    "PLM/test.jsonl",
    "PLM/test_ood.jsonl",
)
SCIERC_URLS = (
    "https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz",
    "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz",
)
CORD_REPO = "naver-clova-ix/cord-v2"
CORD_FILES = (
    "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    "data/train-00001-of-00004-7dbbe248962764c5.parquet",
    "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
    "data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
    "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    print(f"GET {url}")
    urllib.request.urlretrieve(url, dest)


def fetch_sciriff(dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = dest / "sciriff"
    snapshot_download(
        SCIRIFF_REPO,
        repo_type="dataset",
        allow_patterns=SCIRIFF_ALLOW,
        local_dir=str(target),
    )
    return target


def fetch_scier(dest: Path) -> Path:
    target = dest / "scier"
    for name in SCIER_FILES:
        _download(f"{SCIER_BASE}/{name}", target / name)
    return target


def fetch_scierc(dest: Path) -> Path:
    target = dest / "scierc"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "sciERC_processed.tar.gz"
    last_error = "no url tried"
    for url in SCIERC_URLS:
        try:
            _download(url, archive)
            break
        except Exception as exc:
            last_error = f"{url} -> {exc}"
            print(f"SciERC download failed: {last_error}")
            if archive.exists() and archive.stat().st_size == 0:
                archive.unlink()
    else:
        note = target / "DOWNLOAD_FAILED.txt"
        note.write_text(
            "The official SciERC tarball could not be downloaded.\n"
            f"Last error: {last_error}\n"
            "SciERC annotations are still present inside allenai/SciRIFF as scierc_ner and scierc_re.\n",
            encoding="utf-8",
        )
        return target
    extracted = target / "extracted"
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(extracted, filter="data")
    return target


def _cord_split(name: str) -> str:
    if "validation" in name:
        return "dev"
    if "test" in name:
        return "test"
    return "train"


def fetch_cord_text(dest: Path) -> Path:
    """Download CORD parquet, keep ground_truth text, delete the image parquet."""
    import duckdb
    from huggingface_hub import hf_hub_download

    raw = dest / "cord" / "_parquet"
    text_dir = dest / "cord" / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    handles = {split: (text_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "dev", "test")}
    counts = {"train": 0, "dev": 0, "test": 0}
    try:
        for name in CORD_FILES:
            path = Path(hf_hub_download(CORD_REPO, name, repo_type="dataset", local_dir=str(raw)))
            split = _cord_split(path.name)
            rows = duckdb.sql(
                f"SELECT ground_truth FROM read_parquet('{path.as_posix()}')"
            ).fetchall()
            for (value,) in rows:
                record = {"ground_truth": value, "split": split}
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
            path.unlink(missing_ok=True)
    finally:
        for handle in handles.values():
            handle.close()
    for leftover in raw.rglob("*"):
        if leftover.is_file() and leftover.suffix == ".parquet":
            leftover.unlink()
    (text_dir / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"CORD text rows: {counts}")
    return text_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download public SKEX datasets into Dataset/")
    parser.add_argument("--dest", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--only",
        default="sciriff,scier,scierc,cord",
        help="Comma-separated subset of sciriff,scier,scierc,cord",
    )
    args = parser.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    if "scier" in wanted:
        fetch_scier(dest)
    if "scierc" in wanted:
        fetch_scierc(dest)
    if "sciriff" in wanted:
        fetch_sciriff(dest)
    if "cord" in wanted:
        fetch_cord_text(dest)
    print(f"dataset dir: {dest.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
