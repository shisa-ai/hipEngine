from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gguf_boundary_cast_error import build_cast_error_artifact, compare_recurrent_bf16_cast


def test_compare_recurrent_bf16_cast_matches_host_rounding() -> None:
    recurrent = np.asarray([0.0, 1.2345, -2.75, 128.125], dtype=np.float32)
    rounded = bf16_to_float32(float_array_to_bf16_bits(recurrent)).astype(np.float32)

    comparison = compare_recurrent_bf16_cast(recurrent, rounded)

    assert comparison["device_matches_expected_bf16"] is True
    assert comparison["device_expected_bf16_diff"]["max_abs_diff"] == 0.0
    assert comparison["cast_error"]["count"] == 4
    assert comparison["cast_error"]["max_abs_diff"] > 0.0


def test_compare_recurrent_bf16_cast_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compare_recurrent_bf16_cast(
            np.ones((4,), dtype=np.float32),
            np.ones((3,), dtype=np.float32),
        )


def test_build_cast_error_artifact_requires_full_arrays(tmp_path: Path) -> None:
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"arrays": {"recurrent_out_f32": [1.0]}}) + "\n")

    with pytest.raises(ValueError, match="recurrent_bf16_f32"):
        build_cast_error_artifact(capture)

    capture.write_text(json.dumps({"buffers": {}}) + "\n")
    with pytest.raises(ValueError, match="include full arrays"):
        build_cast_error_artifact(capture)
