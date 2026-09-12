from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_layer_capture_summary import summarize_array


def test_summarize_array_records_hash_samples_and_top_abs() -> None:
    values = np.asarray([0.25, -2.0, 1.5, 0.5], dtype=np.float32)

    summary = summarize_array(values, top_n=2)

    assert summary["shape"] == [4]
    assert summary["dtype"] == "float32"
    assert summary["count"] == 4
    assert len(summary["sha256"]) == 64
    assert summary["sample_first"] == pytest.approx([0.25, -2.0])
    assert summary["sample_last"] == pytest.approx([1.5, 0.5])
    assert summary["top_abs"] == [
        {"index": 1, "value": -2.0, "abs": 2.0},
        {"index": 2, "value": 1.5, "abs": 1.5},
    ]
    assert summary["rms"] == pytest.approx(float(np.sqrt(np.mean(values * values))))


def test_summarize_array_preserves_integer_samples() -> None:
    values = np.asarray([200, 140, 67], dtype=np.int64)

    summary = summarize_array(values, top_n=2)

    assert summary["dtype"] == "int64"
    assert summary["sample_first"] == [200, 140]
    assert all(isinstance(item, int) for item in summary["sample_first"])
    assert summary["top_abs"][0] == {
        "index": 0,
        "value": 200.0,
        "abs": 200.0,
    }


def test_summarize_array_requires_positive_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        summarize_array(np.asarray([1.0], dtype=np.float32), top_n=0)
