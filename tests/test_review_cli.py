import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_review_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "pdiseg.cli.review", "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Web UI" in result.stdout
    assert "--result" in result.stdout
    assert "--calibration" in result.stdout
