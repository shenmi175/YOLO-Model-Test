import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from datasets.xml_loader import load_dataset
from inference.predictor import Predictor
from metrics.evaluator import Evaluator


def test_evaluator_metrics():
    dataset = load_dataset('test_data')
    predictor = Predictor('models/best.pt')
    preds = {ann.image_path: predictor.predict(ann.image_path) for ann in dataset}
    evaluator = Evaluator()
    result = evaluator.evaluate(dataset, preds)
    assert result.tp == 4
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0