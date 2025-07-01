import subprocess
import sys
from pathlib import Path
import importlib
import pytest


@pytest.mark.skipif(importlib.util.find_spec('matplotlib') is None, reason='requires matplotlib')
def test_confusion_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "out"
    cmd = [
        sys.executable,
        str(root / "src" / "confusion_cli.py"),
        "--model",
        str(root / "models" / "best.pt"),
        "--data",
        str(root / "test_data"),
        "--output",
        str(out_root),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0
    run_dir = out_root / 'test_data1'
    assert (run_dir / 'confusion_matrix.png').exists()
    assert (run_dir / 'test1' / 'confusion_matrix.png').exists()
    assert (run_dir / 'test2' / 'confusion_matrix.png').exists()
    assert (run_dir / 'run.log').exists()
