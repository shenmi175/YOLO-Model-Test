from __future__ import annotations
"""Utilities for zero-shot detection using Grounding DINO.

The implementation follows the example usage from the official
documentation: https://huggingface.co/docs/transformers/model_doc/grounding-dino

@misc{liu2023grounding,
      title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection}, 
      author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
      year={2023},
      eprint={2303.05499},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

"""
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
            self.model_id,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device)
        self.model.eval()

    def predict(
        self,
        image_paths: Sequence[str],
        text_labels: Sequence[str],
        box_threshold: float,
        text_threshold: float,
    ) -> List[Tuple[List[List[float]], List[str], List[float], Tuple[int, int]]]:
        """Run inference on ``image_paths`` and return detection results."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        # According to the official documentation, multiple queries should be
        # concatenated with a trailing period so the model can parse them.
        text_prompt = ". ".join(text_labels) + "."
        inputs = self.processor(images=images, text=[text_prompt] * len(images), return_tensors="pt").to(self.device)
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
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return detections