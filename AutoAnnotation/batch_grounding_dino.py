#!/usr/bin/env python3
"""Batch zero-shot detection with optional GUI."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

from dino_predictor import GroundingDinoPredictor
from xml_utils import save_voc_xml


def iter_images(root_dir: Path, exts: set[str]):
    """Yield image paths under ``root_dir`` matching ``exts``."""
    for p in root_dir.rglob("*"):
        if p.suffix.lower() in exts:
            yield p

_DEF_BATCH = 4


def run_batch(
    text_labels: Sequence[str],
    img_root: str | Path,
    xml_root: str | Path,
    box_threshold: float,
    text_threshold: float,
    batch_size: int = _DEF_BATCH,
) -> None:
    """Process all images under ``img_root`` and save XML files."""
    predictor = GroundingDinoPredictor()
    img_root = Path(img_root)
    xml_root = Path(xml_root)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total = sum(1 for _ in iter_images(img_root, exts))
    pbar = tqdm(total=total, desc="Processing", unit="img")

    image_iter = iter_images(img_root, exts)
    batch_paths: list[Path] = []
    for img_path in image_iter:
        batch_paths.append(img_path)
        if len(batch_paths) < batch_size:
            continue
        results = predictor.predict(
            [str(p) for p in batch_paths], text_labels, box_threshold, text_threshold
        )
        for path, (boxes, labels, scores, size) in zip(batch_paths, results):
            rel = path.relative_to(img_root)
            xml_path = xml_root / rel.with_suffix(".xml")
            save_voc_xml(xml_path, path.name, size, boxes, labels, scores)
            pbar.update(1)
        batch_paths = []

    if batch_paths:
        results = predictor.predict(
            [str(p) for p in batch_paths], text_labels, box_threshold, text_threshold
        )
        for path, (boxes, labels, scores, size) in zip(batch_paths, results):
            rel = path.relative_to(img_root)
            xml_path = xml_root / rel.with_suffix(".xml")
            save_voc_xml(xml_path, path.name, size, boxes, labels, scores)
            pbar.update(1)

    pbar.close()


def launch_gui() -> None:
    """Launch a simple Tkinter GUI for batch processing."""
    root = tk.Tk()
    root.title("Grounding DINO Batch")

    tk.Label(root, text="Text labels (comma separated):").grid(row=0, column=0, sticky="e")
    labels_var = tk.StringVar()
    tk.Entry(root, textvariable=labels_var, width=60).grid(row=0, column=1, columnspan=2)

    tk.Label(root, text="Images root:").grid(row=1, column=0, sticky="e")
    img_var = tk.StringVar()
    tk.Entry(root, textvariable=img_var, width=50).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=lambda: img_var.set(filedialog.askdirectory())).grid(row=1, column=2)

    tk.Label(root, text="XML output:").grid(row=2, column=0, sticky="e")
    out_var = tk.StringVar()
    tk.Entry(root, textvariable=out_var, width=50).grid(row=2, column=1)
    tk.Button(root, text="Browse", command=lambda: out_var.set(filedialog.askdirectory())).grid(row=2, column=2)

    tk.Label(root, text="Box threshold:").grid(row=3, column=0, sticky="e")
    box_var = tk.StringVar(value="0.3")
    tk.Entry(root, textvariable=box_var, width=10).grid(row=3, column=1, sticky="w")

    tk.Label(root, text="Text threshold:").grid(row=4, column=0, sticky="e")
    text_var = tk.StringVar(value="0.25")
    tk.Entry(root, textvariable=text_var, width=10).grid(row=4, column=1, sticky="w")

    tk.Label(root, text="Batch size:").grid(row=5, column=0, sticky="e")
    batch_var = tk.StringVar(value=str(_DEF_BATCH))
    tk.Entry(root, textvariable=batch_var, width=10).grid(row=5, column=1, sticky="w")

    def start() -> None:
        labels = [l.strip() for l in labels_var.get().split(',') if l.strip()]
        if not labels or not img_var.get() or not out_var.get():
            messagebox.showerror("Error", "Please provide all parameters")
            return
        threading.Thread(
            target=run_batch,
            args=(
                labels,
                img_var.get(),
                out_var.get(),
                float(box_var.get()),
                float(text_var.get()),
                int(batch_var.get()),
            ),
            daemon=True,
        ).start()

    tk.Button(root, text="Start", command=start).grid(row=6, columnspan=3, pady=5)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()