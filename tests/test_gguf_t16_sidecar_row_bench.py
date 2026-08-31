from __future__ import annotations

import pytest

from scripts.gguf_t16_sidecar_row_bench import (
    ARMS,
    DEFAULT_ROWS,
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
