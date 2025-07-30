from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


@dataclass
class GroundingDinoPredictor:
    """Wrapper around Grounding DINO for batch inference."""

    model_id: str = "IDEA-Research/grounding-dino-base"
    cache_dir: Path = Path(__file__).resolve().parents[1] / "models" / "grounding-dino-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=str(self.cache_dir))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id, cache_dir=str(self.cache_dir)
        ).to(self.device)

    def predict(
        self,
        image_paths: Sequence[str],
        text_labels: Sequence[str],
        box_threshold: float,
        text_threshold: float,
    ) -> List[Tuple[List[List[float]], List[str], List[float], Tuple[int, int]]]:
        """Run inference on ``image_paths`` and return detection results."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, text=[text_labels] * len(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img.size[::-1] for img in images],
        )

        detections: List[Tuple[List[List[float]], List[str], List[float], Tuple[int, int]]] = []
        for res, img in zip(results, images):
            boxes = res["boxes"].cpu().tolist()
            labels = res["labels"]
            scores = res["scores"].cpu().tolist()
            detections.append((boxes, labels, scores, img.size))
            # print(detections)
        return detections