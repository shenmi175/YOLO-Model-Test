import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from inference.predictor import Predictor
from datasets.xml_loader import Box


def test_predictor_fallback():
    predictor = Predictor('models/best.pt')
    img = 'test_data/test1/20221126152950_002730.jpg'
    boxes = predictor.predict(img)
    assert boxes
    box = boxes[0]
    assert isinstance(box, Box)
    assert box.xmin == 741
    assert box.ymax == 684