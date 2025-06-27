import argparse
import logging
from pathlib import Path

from config import Config
from datasets.xml_loader import load_dataset
from inference.predictor import Predictor
from metrics.evaluator import Evaluator
from log_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 evaluation CLI")
    parser.add_argument("--config", help="Path to config YAML", default=None)
    parser.add_argument("--model", help="Model weights", default=None)
    parser.add_argument("--data", help="Dataset directory", default=None)
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not save predictions")
    parser.add_argument("--log-dir", help="Directory for logs", default="logs")
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

    logging.info("Loading dataset from %s", cfg.data_dir)
    annotations = load_dataset(cfg.data_dir)
    predictor = Predictor(cfg.model_path, cfg.confidence_threshold)

    predictions = {}
    for ann in annotations:
        boxes = predictor.predict(ann.image_path)
        predictions[ann.image_path] = boxes

    evaluator = Evaluator(cfg.iou_threshold)
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