import logging
from pathlib import Path
import sys

from config import Config
from datasets.xml_loader import load_dataset, DatasetConsistencyError
from inference.predictor import Predictor
from metrics.evaluator import Evaluator
from log_setup import setup_logging
import argparse

try:  # optional dependency
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm may not be installed
    tqdm = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 evaluation CLI")
    parser.add_argument("--config", help="Path to config YAML", default=None)
    parser.add_argument("--model", help="Model weights", default=None)
    parser.add_argument("--data", help="Dataset directory", default=None)
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), help="Inference image size", default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not save predictions")
    parser.add_argument("--log-dir", help="Directory for logs", default="logs")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar while running predictions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)
    cfg = Config.from_file(args.config)
    if args.model:
        cfg.model_path = args.model
    if args.data:
        cfg.data_dir = args.data
    if args.output:
        cfg.output_dir = args.output
    if args.no_save:
        cfg.save_predictions = False
    if args.img_size:
        cfg.img_size = tuple(args.img_size)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    # Log run parameters
    logging.info("Model path: %s", cfg.model_path)
    logging.info("Dataset dir: %s", cfg.data_dir)
    logging.info("Output dir: %s", cfg.output_dir)
    logging.info("Confidence threshold: %.3f", cfg.confidence_threshold)
    logging.info("IoU threshold: %.3f", cfg.iou_threshold)
    logging.info("Image size: %s", cfg.img_size)
    logging.info("Batch size: %d", cfg.batch_size)
    logging.info("Save predictions: %s", cfg.save_predictions)
    logging.info("Loading dataset from %s", cfg.data_dir)
    try:
        annotations = load_dataset(cfg.data_dir)
    except DatasetConsistencyError as exc:
        logging.debug("Dataset issue: %s", exc)
        annotations = exc.annotations
    predictor = Predictor(
        cfg.model_path,
        cfg.confidence_threshold,
        cfg.img_size,
        cfg.batch_size,
    )

    predictions = {}

    iterable = annotations
    if args.progress:
        if tqdm is not None:
            iterable = tqdm(annotations, desc="Predicting", unit="img")
        else:
            total = len(annotations)

            def _simple_progress(items: list) -> "list":
                for idx, item in enumerate(items, 1):
                    print(f"{idx}/{total}", end="\r", file=sys.stderr)
                    yield item
                print(file=sys.stderr)

            iterable = _simple_progress(annotations)

    for ann in iterable:
        boxes = predictor.predict(ann.image_path)
        predictions[ann.image_path] = boxes

    names_attr = getattr(predictor.model, "names", None) if predictor.model is not None else None
    if isinstance(names_attr, dict):
        class_names = list(names_attr.values())
    elif names_attr is not None:
        class_names = list(names_attr)
    else:
        class_names = None
    evaluator = Evaluator(cfg.iou_threshold, class_names)

    result = evaluator.evaluate(annotations, predictions)

    logging.info("Precision: %.3f", result.precision)
    logging.info("Recall: %.3f", result.recall)
    logging.info("F1: %.3f", result.f1)
    logging.info("mAP50: %.3f", result.map50)

    if cfg.save_predictions:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_file = out_dir / "predictions.txt"
        with pred_file.open("w", encoding="utf-8") as fh:
            for img, boxes in predictions.items():
                b_str = " ".join(
                    f"{b.label},{b.xmin},{b.ymin},{b.xmax},{b.ymax}" for b in boxes
                )
                fh.write(f"{img} {b_str}\n")
        logging.info("Predictions saved to %s", pred_file)


if __name__ == "__main__":
    main()
