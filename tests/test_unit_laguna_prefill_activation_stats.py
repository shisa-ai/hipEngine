from __future__ import annotations

import numpy as np
import pytest

from scripts.laguna_prefill_activation_stats import (
    ACTIVATION_ROWS,
    _bf16_bits,
    _summarize_bf16_activation,
)


def test_activation_rows_cover_lap0_and_natural_expert_shapes() -> None:
    assert ACTIVATION_ROWS == (32, 55, 64, 122, 128, 256, 512)


def test_bf16_summary_preserves_bits_and_reports_compact_statistics() -> None:
    source = np.array(
        [
            [0.0, -0.0, 1.0, -2.0],
            [0.5, -0.5, 4.0, -8.0],
        ],
        dtype=np.float32,
    )
    values = _bf16_bits(source)

    summary = _summarize_bf16_activation(values)

    assert summary["shape"] == [2, 4]
    assert summary["count"] == 8
    assert summary["finite_count"] == 8
    assert summary["nonfinite_count"] == 0
    assert summary["zero_count"] == 2
    assert summary["zero_fraction"] == 0.25
    assert summary["minimum"] == -8.0
    assert summary["maximum"] == 4.0
    assert summary["mean"] == pytest.approx(-0.625)
    assert summary["rms"] == pytest.approx(np.sqrt(85.5 / 8.0))
    assert summary["absolute"]["p50"] == pytest.approx(0.75)
    assert summary["absolute"]["maximum"] == 8.0
    assert summary["row_rms"]["minimum"] == pytest.approx(np.sqrt(5.0 / 4.0))
    assert summary["row_rms"]["maximum"] == pytest.approx(np.sqrt(80.5 / 4.0))
    assert len(summary["bf16_sha256"]) == 64


def test_bf16_summary_rejects_wrong_or_empty_inputs() -> None:
    with pytest.raises(TypeError, match="uint16"):
        _summarize_bf16_activation(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="two-dimensional"):
        _summarize_bf16_activation(np.ones(4, dtype=np.uint16))
    with pytest.raises(ValueError, match="non-empty"):
        _summarize_bf16_activation(np.empty((0, 2), dtype=np.uint16))
