import os
import subprocess
import sys
from pathlib import Path


def test_cli_runs(tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(Path('src') / 'cli.py'),
        '--model', 'models/best.pt',
        '--data', 'test_data',
        '--output', str(out_dir),
        '--no-save',
        '--log-dir', str(log_dir)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0
    assert "Precision:" in result.stdout