import subprocess
import sys
from pathlib import Path


def test_cli_end_to_end(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / 'src' / 'cli.py'),
        '--model', str(root / 'models' / 'best.pt'),
        '--data', str(root / 'test_data'),
        '--output', str(out_dir),
        '--log-dir', str(log_dir),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0
    assert "Precision:" in result.stdout
    assert "mAP50:" in result.stdout
    pred_file = out_dir / 'predictions.txt'
    assert pred_file.exists()
    lines = pred_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert log_dir.is_dir() and any(log_dir.iterdir())
