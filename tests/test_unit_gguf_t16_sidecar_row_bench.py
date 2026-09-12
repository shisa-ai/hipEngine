from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_t16_sidecar_row_bench import (
    ARMS,
    DEFAULT_ROWS,
    _bf16_comparison,
    _counterbalanced_order,
    _sample_summary,
    build_parser,
)


def test_sidecar_row_bench_counterbalances_every_arm_position() -> None:
    orders = [
        _counterbalanced_order(ARMS, cell_index=2, sample_index=index)
        for index in range(3)
    ]

    assert orders == [
        ("strict_shared_b", "incumbent", "smallm"),
        ("incumbent", "smallm", "strict_shared_b"),
        ("smallm", "strict_shared_b", "incumbent"),
    ]
    assert all({order[position] for order in orders} == set(ARMS) for position in range(3))


def test_sidecar_row_bench_parser_requires_rows2_through8_and_output() -> None:
    parser = build_parser()
    args = parser.parse_args(["--rows", "2,4,8", "--output", "/tmp/result.json"])

    assert args.rows == (2, 4, 8)
    assert args.output.as_posix() == "/tmp/result.json"
    assert build_parser().parse_args(["--output", "/tmp/result.json"]).rows == DEFAULT_ROWS
    with pytest.raises(SystemExit):
        parser.parse_args(["--rows", "1,8", "--output", "/tmp/result.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--rows", "2,2", "--output", "/tmp/result.json"])


def test_sidecar_row_bench_reports_bf16_exactness_and_ulp_distance() -> None:
    expected = np.asarray([0x0000, 0x3F80, 0xBF80], dtype=np.uint16)
    exact = _bf16_comparison(expected.copy(), expected)
    one_ulp = _bf16_comparison(
        np.asarray([0x0000, 0x3F81, 0xBF80], dtype=np.uint16),
        expected,
    )

    assert exact == {
        "exact": True,
        "mismatched_values": 0,
        "total_values": 3,
        "max_bf16_ulp": 0,
        "max_abs_error": 0.0,
    }
    assert one_ulp["exact"] is False
    assert one_ulp["mismatched_values"] == 1
    assert one_ulp["max_bf16_ulp"] == 1
    assert one_ulp["max_abs_error"] == pytest.approx(1.0 / 128.0)


def test_sidecar_row_bench_sample_summary_is_sorted_and_uses_median() -> None:
    summary = _sample_summary((0.4, 0.2, 0.3))

    assert summary == {
        "samples": 3,
        "median_ms": 0.3,
        "min_ms": 0.2,
        "max_ms": 0.4,
        "values_ms": [0.2, 0.3, 0.4],
    }
    with pytest.raises(ValueError, match="zero samples"):
        _sample_summary(())
