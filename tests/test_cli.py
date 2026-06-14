"""CLI entry points: help and smoke checks."""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pdiseg_module_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "pdiseg", "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Segment poultry-packaging" in result.stdout


def test_calibrate_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "pdiseg.calibrate_cli", "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "calibration harness" in result.stdout


def test_calibrate_console_entry_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "pdiseg.calibrate_cli", "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "--per-class-limit" in result.stdout
