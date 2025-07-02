"""Simple Tkinter GUI for launching model evaluation with progress."""

from __future__ import annotations


import logging
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import Config
from src.datasets.xml_loader import load_dataset, Annotation, DatasetConsistencyError
from src.inference.predictor import Predictor
from src.metrics.evaluator import Evaluator
from src.metrics.confusion import plot_confusion_matrix
from src.log_setup import setup_logging


def run_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str,
    save_predictions: bool,
    save_images: bool = False,
    progress_cb: callable | None = None,
    image_output_dir: str | None = None,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> tuple[Path, Path | None]:
    """Run evaluation, reporting progress via ``progress_cb``.

    Returns the run directory and the directory where annotated images were
    stored (``None`` if images were not saved)."""
    """Run evaluation, reporting progress via ``progress_cb``."""
    cfg = Config.from_file(None)
    # ``gui.py`` lives under ``src/ui`` so we need to go two levels up to reach
    # the repository root where the ``models`` and ``test_data`` directories
    # reside. Previously this used ``parents[1]`` which resolved to the ``src``
    # directory, causing all relative paths to incorrectly point inside ``src``.
    # Using ``parents[2]`` ensures we always resolve paths relative to the
    # project root.
    repo_root = Path(__file__).resolve().parents[2]
    # repo_root = Path(__file__).resolve().parents[1]

    cfg.model_path = str((repo_root / model_path).resolve()) if not Path(model_path).is_absolute() else model_path
    cfg.data_dir = str((repo_root / data_dir).resolve()) if not Path(data_dir).is_absolute() else data_dir
    cfg.output_dir = str((repo_root / output_dir).resolve()) if not Path(output_dir).is_absolute() else output_dir

    cfg.save_predictions = save_predictions
    cfg.save_images = save_images

    if conf_threshold is not None:
        cfg.confidence_threshold = conf_threshold
    if iou_threshold is not None:
        cfg.iou_threshold = iou_threshold


    out_root = Path(cfg.output_dir)
    data_name = Path(cfg.data_dir).name
    idx = 1
    while (out_root / f"{data_name}{idx}").exists():
        idx += 1
    run_dir = out_root / f"{data_name}{idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    img_dir_path: Path | None = None

    log_file = run_dir / "run.log"
    setup_logging(str(log_file))

    logging.info("Loading dataset from %s", cfg.data_dir)
    try:
        annotations = load_dataset(cfg.data_dir)
    except DatasetConsistencyError as exc:
        logging.debug("Dataset issue: %s", exc)
        annotations = exc.annotations
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
        labels = res.labels
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

    if cfg.save_images:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:  # pragma: no cover - PIL optional
            logging.error("Saving images failed: %s", exc)
        else:
            from src.utils.visualization import draw_boxes
            img_dir = Path(image_output_dir) if image_output_dir else run_dir / "images"
            if not img_dir.is_absolute():
                img_dir = repo_root / img_dir
            img_dir.mkdir(parents=True, exist_ok=True)
            for img_path, boxes in predictions.items():
                img = Image.open(img_path).convert("RGB")
                draw_boxes(img, boxes)

                out_path = img_dir / Path(img_path).name
                img.save(out_path)
            logging.info("Images saved to %s", img_dir)
            img_dir_path = img_dir

    return run_dir, img_dir_path

def launch() -> None:
    """Launch the parameter selection window."""
    cfg = Config.from_file(None)
    root = tk.Tk()
    root.title("YOLO Model Test")

    tk.Label(root, text="Model path:").grid(row=0, column=0, sticky="e")
    model_var = tk.StringVar(value=cfg.model_path)
    tk.Entry(root, textvariable=model_var, width=60).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: model_var.set(
        filedialog.askopenfilename(initialdir="models", filetypes=[("Model", "*.pt *.onnx"), ("All", "*.*")])
    )).grid(row=0, column=2)

    tk.Label(root, text="Data directory:").grid(row=1, column=0, sticky="e")
    data_var = tk.StringVar(value=cfg.data_dir)
    tk.Entry(root, textvariable=data_var, width=60).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=lambda: data_var.set(
        filedialog.askdirectory(initialdir="test_data")
    )).grid(row=1, column=2)

    tk.Label(root, text="Output directory:").grid(row=2, column=0, sticky="e")
    out_var = tk.StringVar(value=cfg.output_dir)
    tk.Entry(root, textvariable=out_var, width=60).grid(row=2, column=1)
    tk.Button(root, text="Browse", command=lambda: out_var.set(
        filedialog.askdirectory(initialdir="output")
    )).grid(row=2, column=2)

    save_var = tk.BooleanVar(value=cfg.save_predictions)
    tk.Checkbutton(root, text="Save predictions", variable=save_var).grid(row=3, columnspan=3)

    image_var = tk.BooleanVar(value=cfg.save_images)
    tk.Checkbutton(root, text="Save images", variable=image_var).grid(row=4, columnspan=3)

    tk.Label(root, text="Image output dir:").grid(row=5, column=0, sticky="e")
    img_out_var = tk.StringVar()
    tk.Entry(root, textvariable=img_out_var, width=60).grid(row=5, column=1)
    tk.Button(root, text="Browse", command=lambda: img_out_var.set(
        filedialog.askdirectory(initialdir="output")
    )).grid(row=5, column=2)

    tk.Label(root, text="Confidence:").grid(row=6, column=0, sticky="e")
    conf_var = tk.StringVar(value=str(cfg.confidence_threshold))
    tk.Entry(root, textvariable=conf_var, width=10).grid(row=6, column=1, sticky="w")

    tk.Label(root, text="IoU threshold:").grid(row=7, column=0, sticky="e")
    iou_var = tk.StringVar(value=str(cfg.iou_threshold))
    tk.Entry(root, textvariable=iou_var, width=10).grid(row=7, column=1, sticky="w")

    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.grid(row=8, columnspan=3, pady=5)

    def update_progress(current: int, total: int) -> None:
        progress["maximum"] = total
        progress["value"] = current
        root.update_idletasks()

    def run() -> None:
        def _worker() -> None:
            try:
                run_dir, img_dir = run_evaluation(
                    model_var.get(),
                    data_var.get(),
                    out_var.get(),
                    save_var.get(),
                    image_var.get(),
                    update_progress,
                    img_out_var.get() or None,
                    float(conf_var.get()),
                    float(iou_var.get()),
                )
                msg = "Evaluation complete"
                if img_dir:
                    msg += f"\nImages saved to: {img_dir}"
                messagebox.showinfo("YOLO Model Test", msg)
            except Exception as exc:  # pragma: no cover - UI errors
                messagebox.showerror("YOLO Model Test", str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    tk.Button(root, text="Run", command=run).grid(row=9, columnspan=3, pady=5)
    root.mainloop()


if __name__ == "__main__":
    launch()