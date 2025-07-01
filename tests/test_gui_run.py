import importlib
from pathlib import Path

import pytest

from src.ui.gui import run_evaluation


@pytest.mark.skipif(importlib.util.find_spec('matplotlib') is None, reason='requires matplotlib')
def test_run_evaluation(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    run_evaluation('models/best.pt', 'test_data', str(out_root), False)
    run_dir = out_root / 'test_data1'
    assert (run_dir / 'confusion_matrix.png').exists()
    assert (run_dir / 'test1' / 'confusion_matrix.png').exists()
    assert (run_dir / 'run.log').exists()
