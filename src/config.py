"""Configuration loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

# Attempt to import PyYAML, but fall back to a simple parser if unavailable.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependency might not be installed
    yaml = None  # type: ignore


def _parse_simple_yaml(path: str) -> Dict[str, Any]:
    """Parse a very small subset of YAML consisting of simple key-value pairs."""
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            data[key] = parsed
    return data


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return _parse_simple_yaml(path)


@dataclass
class Config:
    model_path: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    data_dir: str = "test_data"
    save_predictions: bool = True
    output_dir: str = "output"

    @classmethod
    def from_file(cls, path: str | None = None) -> "Config":
        """Load configuration and resolve relative paths."""
        if path is None:
            root = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(root, "configs", "default.yaml")
        data = load_config(path)
        cfg = cls(**data)

        repo_root = os.path.dirname(os.path.dirname(__file__))
        for key in ("model_path", "data_dir", "output_dir"):
            value = getattr(cfg, key)
            if not os.path.isabs(value):
                setattr(cfg, key, os.path.join(repo_root, value))
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()