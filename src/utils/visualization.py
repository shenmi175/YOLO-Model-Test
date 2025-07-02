"""Utilities for drawing detection results on images."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.datasets.xml_loader import Box

# simple palette of distinct colors
_PALETTE: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
]

_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {}

try:
    _FONT = ImageFont.truetype("G:\\A_Share\\YOLO-Model-Test\\DejaVuSans.ttf", 20)
    print("truetype 字体加载成功！")
except Exception as e:
    print("truetype 字体加载失败，进入 except。错误：", e)
    _FONT = ImageFont.load_default()


def get_color(label: str) -> Tuple[int, int, int]:
    """Return a color tuple for ``label``."""
    if label not in _LABEL_COLORS:
        _LABEL_COLORS[label] = _PALETTE[len(_LABEL_COLORS) % len(_PALETTE)]
    return _LABEL_COLORS[label]


def draw_boxes(image: Image.Image, boxes: Iterable[Box]) -> None:
    """Draw ``boxes`` on ``image`` in-place."""
    draw = ImageDraw.Draw(image)
    for b in boxes:
        color = get_color(b.label)
        draw.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=color, width=2)
        label = b.label
        if b.confidence is not None:
            label += f" {b.confidence:.2f}"
        draw.text((b.xmin, b.ymin), label, fill=color, font=_FONT)
