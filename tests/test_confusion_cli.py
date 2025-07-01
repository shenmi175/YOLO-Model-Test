import subprocess
import sys
from pathlib import Path
import importlib
import pytest


@pytest.mark.skipif(importlib.util.find_spec('matplotlib') is None, reason='requires matplotlib')
def test_confusion_cli(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    cmd = [
        sys.executable,
        str(Path('src') / 'confusion_cli.py'),
        '--model', 'models/best.pt',
        '--data', 'test_data',
        '--output', str(out_root),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0
    run_dir = out_root / 'test_data1'
    assert (run_dir / 'confusion_matrix.png').exists()
    assert (run_dir / 'test1' / 'confusion_matrix.png').exists()
    assert (run_dir / 'test2' / 'confusion_matrix.png').exists()
    assert (run_dir / 'run.log').exists()