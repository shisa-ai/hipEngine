from __future__ import annotations

import pytest

from scripts.laguna_grouped_down_prefill_bench import (
    MODES,
    _comparison_summary,
    _mode_order,
    _set_mode,
)


def _samples(
    direct: dict[int, list[float]],
    candidate: dict[int, list[float]],
) -> dict[str, dict[int, list[float]]]:
    return {
        "direct": direct,
        "adaptive_grouped_smallm": candidate,
    }


def _tokens(
    rows: tuple[int, ...], candidate_offset: int = 0
) -> dict[str, dict[int, list[int]]]:
    return {
        "direct": {row: [row, row] for row in rows},
        "adaptive_grouped_smallm": {
            row: [row + candidate_offset, row + candidate_offset] for row in rows
        },
    }


def test_grouped_down_mode_order_is_counterbalanced() -> None:
    assert MODES == ("direct", "adaptive_grouped_smallm")
    assert _mode_order(0, 0) == MODES
    assert _mode_order(1, 0) == tuple(reversed(MODES))
    assert _mode_order(0, 1) == tuple(reversed(MODES))


def test_grouped_down_summary_allows_exact_direct_fallback_and_gates_grouped_gain() -> None:
    rows = (16, 32)
    result = _comparison_summary(
        _samples(
            {16: [1.0, 1.0], 32: [1.0, 1.0]},
            {16: [1.001, 1.001], 32: [0.9, 0.9]},
        ),
        _tokens(rows),
        rows=rows,
    )

    assert result["correctness"]["pass"] is True
    assert result["promotion"]["pass"] is True
    assert result["shapes"]["16"]["route"] == "direct_fallback"
    assert result["shapes"]["32"]["route"] == "grouped_smallm"


def test_grouped_down_summary_fails_mismatch_or_slow_grouped_shape() -> None:
    rows = (16, 32)
    result = _comparison_summary(
        _samples(
            {16: [1.0, 1.0], 32: [1.0, 1.0]},
            {16: [1.0, 1.0], 32: [1.1, 1.1]},
        ),
        _tokens(rows, candidate_offset=1),
        rows=rows,
    )

    assert result["correctness"]["pass"] is False
    assert result["promotion"]["pass"] is False
    assert "output_ids_not_exact" in result["promotion"]["failed_checks"]
    assert "grouped_route_not_faster" in result["promotion"]["failed_checks"]


def test_grouped_down_set_mode_uses_session_local_selector() -> None:
    class Session:
        selected: str | None = None

        def set_selected_down_mode(self, mode: str) -> None:
            self.selected = mode

    session = Session()
    _set_mode(session, "adaptive_grouped_smallm")  # type: ignore[arg-type]
    assert session.selected == "adaptive_grouped_smallm"
    with pytest.raises(ValueError, match="unknown Laguna grouped-down mode"):
        _set_mode(session, "bogus")  # type: ignore[arg-type]
