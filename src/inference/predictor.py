from __future__ import annotations
import os


from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging
from src.datasets.xml_loader import Box

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - ultralytics may not be installed
    logging.exception("Failed to import ultralytics YOLO: %s", exc)
    YOLO = None  # type: ignore


@dataclass
class Predictor:
    """Simple wrapper around YOLOv8 inference."""

    model_path: str
    confidence: float = 0.25
    image_size: tuple[int, int] | list[int] = (192, 320)
    batch_size: int = 1

    def __post_init__(self) -> None:
        """Load the YOLO model or abort if unavailable."""
        if YOLO is None:
            # Log the missing dependency and stop execution
            logging.exception("ultralytics YOLO is not installed")
            raise RuntimeError("YOLO library not available")

        try:
            self.model = YOLO(self.model_path)
            self.model.conf = self.confidence
            self.model.overrides["imgsz"] = list(self.image_size)
        except Exception as exc:  # pragma: no cover - model loading may fail
            logging.exception("Failed to load YOLO model: %s", exc)
            raise

    def predict(self, image_path: str) -> List[Box]:
        """Run inference on a single image."""
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized")

        results = self.model(image_path)
        boxes: List[Box] = []
        for r in results:
            for b in r.boxes:
                label = self.model.names[int(b.cls)]
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                boxes.append(
                    Box(
                        label,
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                        conf,
                    )
                )
        return boxes

    def batch_predict(self, image_paths: List[str]) -> List[List[Box]]:
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized")

        results = self.model.predict(
            image_paths, imgsz=list(self.image_size), batch=self.batch_size
        )
        all_boxes: List[List[Box]] = []
        for r in results:
            boxes: List[Box] = []
            for b in r.boxes:
                label = self.model.names[int(b.cls)]
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                boxes.append(
                    Box(
                        label,
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                        conf,
                    )
                )
            all_boxes.append(boxes)
        return all_boxes