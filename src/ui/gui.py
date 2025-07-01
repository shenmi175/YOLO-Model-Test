"""Simple Tkinter GUI for launching model evaluation with progress."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import Config
from datasets.xml_loader import load_dataset, Annotation
from inference.predictor import Predictor
from metrics.evaluator import Evaluator
from metrics.confusion import plot_confusion_matrix
from log_setup import setup_logging


def run_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str,
    save_predictions: bool,
    progress_cb: callable | None = None,
) -> None:
    """Run evaluation, reporting progress via ``progress_cb``."""
    cfg = Config.from_file(None)
    cfg.model_path = model_path
    cfg.data_dir = data_dir
    cfg.output_dir = output_dir
    cfg.save_predictions = save_predictions

    out_root = Path(cfg.output_dir)
    data_name = Path(cfg.data_dir).name
    idx = 1
    while (out_root / f"{data_name}{idx}").exists():
        idx += 1
    run_dir = out_root / f"{data_name}{idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"
    setup_logging(str(log_file))

    logging.info("Loading dataset from %s", cfg.data_dir)
    annotations = load_dataset(cfg.data_dir)
    predictor = Predictor(cfg.model_path, cfg.confidence_threshold)

    predictions: dict[str, list] = {}
    total = len(annotations)
    for i, ann in enumerate(annotations, 1):
        boxes = predictor.predict(ann.image_path)
        predictions[ann.image_path] = boxes
        if progress_cb:
            progress_cb(i, total)

    evaluator = Evaluator(cfg.iou_threshold)

    def save_result(name: str, anns: list[Annotation]) -> None:
        preds = {a.image_path: predictions[a.image_path] for a in anns}
        res = evaluator.evaluate(anns, preds)
        logging.info(
            "%s - Precision: %.3f Recall: %.3f F1: %.3f mAP50: %.3f",
            name,
            res.precision,
            res.recall,
            res.f1,
            res.map50,
        )
        labels = ["positive", "negative"]
        sub_dir = run_dir / name if name != "overall" else run_dir
        sub_dir.mkdir(parents=True, exist_ok=True)
        cm_path = sub_dir / "confusion_matrix.png"
        cmp_path = sub_dir / "confusion_probability.png"
        try:
            plot_confusion_matrix(res.confusion_matrix, labels, False, str(cm_path))
            plot_confusion_matrix(res.confusion_prob, labels, True, str(cmp_path))
        except Exception as exc:  # pragma: no cover - matplotlib optional
            logging.error("Failed to plot confusion matrix: %s", exc)

    save_result("overall", annotations)

    groups: dict[str, list[Annotation]] = {}
    root_dir = Path(cfg.data_dir)
    for ann in annotations:
        rel = Path(ann.image_path).relative_to(root_dir)
        grp = rel.parts[0] if len(rel.parts) > 1 else root_dir.name
        groups.setdefault(grp, []).append(ann)

    for name, anns in groups.items():
        save_result(name, anns)

    if cfg.save_predictions:
        pred_file = run_dir / "predictions.txt"
        with pred_file.open("w", encoding="utf-8") as fh:
            for img, boxes in predictions.items():
                b_str = " ".join(
                    f"{b.label},{b.xmin},{b.ymin},{b.xmax},{b.ymax}" for b in boxes
                )
                fh.write(f"{img} {b_str}\n")
        logging.info("Predictions saved to %s", pred_file)


def launch() -> None:
    """Launch the parameter selection window."""
    root = tk.Tk()
    root.title("YOLO Model Test")

    tk.Label(root, text="Model path:").grid(row=0, column=0, sticky="e")
    model_var = tk.StringVar(value="models/best.pt")
    tk.Entry(root, textvariable=model_var, width=40).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: model_var.set(
        filedialog.askopenfilename(initialdir="models", filetypes=[("Model", "*.pt *.onnx"), ("All", "*.*")])
    )).grid(row=0, column=2)

    tk.Label(root, text="Data directory:").grid(row=1, column=0, sticky="e")
    data_var = tk.StringVar(value="test_data")
    tk.Entry(root, textvariable=data_var, width=40).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=lambda: data_var.set(
        filedialog.askdirectory(initialdir="test_data")
    )).grid(row=1, column=2)

    tk.Label(root, text="Output directory:").grid(row=2, column=0, sticky="e")
    out_var = tk.StringVar(value="output")
    tk.Entry(root, textvariable=out_var, width=40).grid(row=2, column=1)
    tk.Button(root, text="Browse", command=lambda: out_var.set(
        filedialog.askdirectory(initialdir="output")
    )).grid(row=2, column=2)

    save_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Save predictions", variable=save_var).grid(row=3, columnspan=3)

    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.grid(row=4, columnspan=3, pady=5)

    def update_progress(current: int, total: int) -> None:
        progress["maximum"] = total
        progress["value"] = current
        root.update_idletasks()

    def run() -> None:
        def _worker() -> None:
            try:
                run_evaluation(
                    model_var.get(),
                    data_var.get(),
                    out_var.get(),
                    save_var.get(),
                    update_progress,
                )
                messagebox.showinfo("YOLO Model Test", "Evaluation complete")
            except Exception as exc:  # pragma: no cover - UI errors
                messagebox.showerror("YOLO Model Test", str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    tk.Button(root, text="Run", command=run).grid(row=5, columnspan=3, pady=5)
    root.mainloop()


if __name__ == "__main__":
    launch()