from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.gguf_linear_boundary_capture import _array_summary, _parse_tokens, _resolve_position


def test_linear_boundary_capture_dry_run_writes_plan_artifact(tmp_path: Path) -> None:
    out = tmp_path / "capture-plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_linear_boundary_capture.py",
            "--dry-run",
            "--tokens",
            "10,11,12",
            "--position",
            "1",
            "--layer",
            "0",
            "--output",
            str(out),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    stdout = json.loads(result.stdout)
    artifact = json.loads(out.read_text())
    assert stdout == {
        "output": str(out),
        "status": "dry_run",
        "position": 1,
        "token_id": 11,
    }
    assert artifact["kind"] == "mtp_gguf_linear_attention_boundary_capture"
    assert artifact["status"] == "dry_run"
    assert artifact["prompt_tokens"] == [10, 11, 12]
    assert artifact["warmup_tokens"] == [10]
    assert artifact["api"] == "Qwen35GGUFResidentSession.capture_linear_attention_boundary"


def test_linear_boundary_capture_helpers_validate_inputs() -> None:
    assert _parse_tokens("1, 2,3") == (1, 2, 3)
    assert _resolve_position(-1, 3) == 2
    with pytest.raises(ValueError, match="at least one token"):
        _parse_tokens(",")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_tokens("1,-2")
    with pytest.raises(ValueError, match="outside prompt token range"):
        _resolve_position(3, 3)


def test_array_summary_is_json_friendly() -> None:
    summary = _array_summary(np.asarray([1.0, -2.0, 3.0], dtype=np.float32))

    assert summary == {
        "shape": [3],
        "finite": True,
        "min": -2.0,
        "max": 3.0,
        "mean": pytest.approx(2.0 / 3.0),
        "rms": pytest.approx(np.sqrt(14.0 / 3.0)),
        "sample": [1.0, -2.0, 3.0],
    }
