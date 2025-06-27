"""Utility for loading YOLO models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - ultralytics optional
    YOLO = None  # type: ignore


@dataclass
class ModelManager:
    """Simple manager that loads a YOLO model if possible."""

    model_path: str
    model: Optional[object] = None

    def load(self) -> None:
        if YOLO is not None:
            self.model = YOLO(self.model_path)
        else:  # pragma: no cover - fallback
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_path: str):
        if self.model is None:
            raise RuntimeError("Model not loaded or ultralytics unavailable")
        return self.model(image_path)