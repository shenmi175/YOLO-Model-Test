import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from datasets.xml_loader import load_dataset
from datasets.dataset_stats import compute_stats
from inference.predictor import Predictor
from metrics.evaluator import Evaluator


def test_full_pipeline():
    # Load annotations
    dataset = load_dataset('test_data')
    stats = compute_stats(dataset)

    assert stats['num_images'] == 2
    assert stats['num_boxes'] == 4
    assert stats['class_counts']['person'] == 3
    assert stats['class_counts']['face'] == 1

    predictor = Predictor('models/best.pt')
    predictions = {ann.image_path: predictor.predict(ann.image_path) for ann in dataset}

    evaluator = Evaluator()
    result = evaluator.evaluate(dataset, predictions)

    assert result.tp == 4
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0