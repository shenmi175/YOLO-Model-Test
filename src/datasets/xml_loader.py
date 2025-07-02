"""XML dataset loader for simple object detection annotations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET


@dataclass
class Box:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float | None = None


@dataclass
class Annotation:
    image_path: str
    boxes: List[Box]

class DatasetConsistencyError(Exception):
    """Raised when images and annotations do not match."""

    def __init__(self, errors: List[str], annotations: List[Annotation]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.annotations = annotations

def parse_annotation(xml_file: str) -> Annotation:
    """Parse a single Pascal VOC style XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.findtext("filename")
    folder = Path(xml_file).parent
    image_path = folder / filename if filename else Path()

    boxes: List[Box] = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or "unknown"
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
        except ValueError:
            continue
        boxes.append(Box(name, xmin, ymin, xmax, ymax, 1.0))

    return Annotation(str(image_path), boxes)


def load_dataset(root_dir: str) -> List[Annotation]:
    """Load all XML annotations under ``root_dir``.

    Any missing image/annotation pairs are collected and reported via
    :class:`DatasetConsistencyError` but do not halt execution."""

    annotations: List[Annotation] = []
    errors: List[str] = []
    xml_files: set[str] = set()
    seen_images: set[str] = set()

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(".xml"):
                continue
            xml_path = os.path.join(dirpath, name)
            xml_files.add(os.path.join(dirpath, name))
            try:
                ann = parse_annotation(xml_path)
            except Exception as exc:  # pragma: no cover - malformed XML
                errors.append(f"Failed to parse {xml_path}: {exc}")
                continue

            img_path = ann.image_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(dirpath, os.path.basename(img_path))
                ann.image_path = img_path
            if not os.path.exists(img_path):
                errors.append(f"Missing image for {xml_path}")
                continue
            annotations.append(ann)
            seen_images.add(os.path.abspath(img_path))

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(dirpath, name)
                    if os.path.abspath(img_path) not in seen_images:
                        xml_path = os.path.splitext(img_path)[0] + ".xml"
                        if xml_path not in xml_files and not os.path.exists(xml_path):
                            errors.append(f"Missing annotation for {img_path}")

        if errors:
            raise DatasetConsistencyError(errors, annotations)

        return annotations