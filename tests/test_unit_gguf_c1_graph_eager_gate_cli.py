"""C1 qualification CLI validates its shape before model or HIP access."""
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gguf_c1_graph_eager_gate.py"


def test_help_requires_no_model_or_gpu():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--decode-steps" in result.stdout


@pytest.mark.parametrize("steps,capacity", [(0, 4096), (32, 32)])
def test_invalid_shape_fails_before_loading(tmp_path, steps, capacity):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--model", str(tmp_path / "absent.gguf"),
            "--output", str(tmp_path / "gate.json"),
            "--decode-steps", str(steps), "--max-sequence-length", str(capacity),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires at least 32 decode steps" in result.stderr
    assert not (tmp_path / "gate.json").exists()
