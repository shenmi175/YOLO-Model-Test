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


@dataclass
class Annotation:
    image_path: str
    boxes: List[Box]


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
        boxes.append(Box(name, xmin, ymin, xmax, ymax))

    return Annotation(str(image_path), boxes)


def load_dataset(root_dir: str) -> List[Annotation]:
    """Load all XML annotations under ``root_dir``."""
    annotations: List[Annotation] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(".xml"):
                continue
            xml_path = os.path.join(dirpath, name)
            ann = parse_annotation(xml_path)
            annotations.append(ann)
    return annotations