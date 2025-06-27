"""Common file system helper functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` as a directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str | Path, exts: Iterable[str] | None = None) -> List[str]:
    """Return a list of image file paths under ``directory``."""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
    else:
        exts = {e.lower() for e in exts}

    results: List[str] = []
    for entry in Path(directory).rglob("*"):
        if entry.suffix.lower() in exts:
            results.append(str(entry))
    return results