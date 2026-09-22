from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
