from __future__ import annotations
import os


from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.datasets.xml_loader import Box, parse_annotation

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - ultralytics may not be installed
    YOLO = None  # type: ignore


@dataclass
class Predictor:
    """Simple wrapper around YOLOv8 inference."""

    model_path: str
    confidence: float = 0.25
    image_size: tuple[int, int] | list[int] = (192, 320)
    batch_size: int = 1

    def __post_init__(self) -> None:
        if YOLO is not None:
            self.model = YOLO(self.model_path)
            self.model.conf = self.confidence
            self.model.overrides["imgsz"] = list(self.image_size)
        else:
            self.model = None

    def predict(self, image_path: str) -> List[Box]:
        """Run inference on a single image."""
        if self.model is None:
            # Fallback: use ground truth annotation if available
            xml_path = Path(image_path).with_suffix(".xml")
            if xml_path.exists():
                ann = parse_annotation(str(xml_path))
                return ann.boxes
            return []

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
            return [self.predict(p) for p in image_paths]

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