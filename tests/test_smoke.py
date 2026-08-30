"""The smoke experiment must import and run without touching the GPU heavily."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_experiment_runs():
    r = subprocess.run(
        [sys.executable, "experiments/exp001_smoke/run.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    runs = sorted((ROOT / "results").glob("*__exp001_smoke__*"))
    assert runs, "no results directory written"
    assert (runs[-1] / "summary.json").exists()
