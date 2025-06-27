import sys
from pathlib import Path

# ensure src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from datasets.xml_loader import load_dataset, Annotation, Box


def test_load_dataset():
    dataset = load_dataset('test_data')
    assert len(dataset) == 2
    ann_paths = [a.image_path for a in dataset]
    assert any(p.endswith('20221126152950_002730.jpg') for p in ann_paths)
    assert any(p.endswith('22112604_002940.jpg') for p in ann_paths)

    # check first annotation's boxes
    first = next(a for a in dataset if '20221126152950_002730.jpg' in a.image_path)
    assert len(first.boxes) == 1
    box = first.boxes[0]
    assert isinstance(box, Box)
    assert box.xmin == 741
    assert box.ymax == 684