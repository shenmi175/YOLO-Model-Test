import argparse
import logging
from pathlib import Path


from config import Config
from datasets.xml_loader import load_dataset, DatasetConsistencyError
from inference.predictor import Predictor
from metrics.evaluator import Evaluator
from metrics.confusion import plot_confusion_matrix
from log_setup import setup_logging

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be available
    tqdm = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation and save confusion matrix")
    parser.add_argument("--config", help="Path to config file", default=None)
    parser.add_argument("--model", help="Model weights", default=None)
    parser.add_argument("--data", help="Dataset directory", default=None)
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), help="Inference image size", default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not save prediction txt")
    parser.add_argument("--log-dir", help="Directory for logs", default="logs")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
                    print(f"{idx}/{total}", end="\r")
                    yield item
                print()
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

    def save_result(name: str, anns: list) -> None:
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
        except Exception as exc:
            logging.error("Failed to plot confusion matrix: %s", exc)

    # overall
    save_result("overall", annotations)

    # per folder (all levels)
    groups: dict[str, list] = {}
    root = Path(cfg.data_dir)
    for ann in annotations:
        rel = Path(ann.image_path).relative_to(root)
        parts = rel.parts
        for i in range(1, len(parts)):
            key = Path(*parts[:i]).as_posix()
            groups.setdefault(key, []).append(ann)

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


if __name__ == "__main__":
    main()