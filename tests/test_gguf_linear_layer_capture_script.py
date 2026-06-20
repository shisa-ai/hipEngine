from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.gguf_linear_layer_capture import _array_payload_for_json, _plan_artifact


def test_linear_layer_capture_dry_run_writes_plan_artifact(tmp_path: Path) -> None:
    out = tmp_path / "layer-capture-plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_linear_layer_capture.py",
            "--dry-run",
            "--tokens",
            "10,11,12",
            "--position",
            "1",
            "--layer",
            "0",
            "--iteration",
            "1001",
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
        "layer_id": 0,
    }
    assert artifact["kind"] == "mtp_gguf_linear_attention_layer_capture"
    assert artifact["iteration"] == 1001
    assert artifact["prompt_tokens"] == [10, 11, 12]
    assert artifact["warmup_tokens"] == [10]
    assert artifact["api"] == "Qwen35GGUFResidentSession.capture_linear_attention_layer"


def test_linear_layer_capture_requires_include_arrays_for_array_keys(tmp_path: Path) -> None:
    out = tmp_path / "layer-capture-plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_linear_layer_capture.py",
            "--dry-run",
            "--array-keys",
            "hidden_in_f32",
            "--output",
            str(out),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "--array-keys requires --include-arrays" in result.stderr


def test_linear_layer_capture_serializes_integer_arrays_as_ints() -> None:
    payload = _array_payload_for_json(np.asarray([200, 140, 67], dtype=np.int64))

    assert payload == [200, 140, 67]
    assert all(isinstance(item, int) for item in payload)


def test_linear_layer_capture_plan_records_selected_position() -> None:
    artifact = _plan_artifact(
        model=Path("model.gguf"),
        tokens=(7, 8, 9),
        position=2,
        layer=1,
        status="dry_run",
        iteration=42,
    )

    assert artifact["position"] == 2
    assert artifact["token_id"] == 9
    assert artifact["layer_id"] == 1
    assert artifact["warmup_tokens"] == [7, 8]
    assert np.isfinite(float(artifact["schema"]))
