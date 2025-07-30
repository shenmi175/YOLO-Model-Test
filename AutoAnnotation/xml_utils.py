from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence, Tuple


def save_voc_xml(
    xml_path: Path,
    image_name: str,
    img_size: Tuple[int, int],
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float],
) -> None:
    """Write detection results to Pascal VOC style XML."""
    width, height = img_size
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name

    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = "3"

    for box, label, score in zip(boxes, labels, scores):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "confidence").text = f"{score:.4f}"
        bnd = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = map(int, box)
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(xml_path, encoding="utf-8")