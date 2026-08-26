from __future__ import annotations

import pytest

from scripts.gguf_mtp_c1c8_server_bench import (
    _diagnostic_plan,
    _generated_ids,
    _mtp_engaged,
    _parse_widths,
    _render_messages,
    summarize,
)


def test_mtp_c1c8_parses_complete_widths() -> None:
    assert _parse_widths("1,2,3,4,5,6,7,8") == tuple(range(1, 9))
    with pytest.raises(Exception, match="unique subset"):
        _parse_widths("1,1")
    with pytest.raises(Exception, match="unique subset"):
        _parse_widths("9")


def test_mtp_c1c8_renders_canonical_messages() -> None:
    assert _render_messages([{"role": "user", "content": "hello"}]) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_mtp_c1c8_extracts_authoritative_ids_and_engagement() -> None:
    payload = {
        "choices": [{"hipengine": {"generated_token_ids": [9]}}],
        "hipengine": {
            "token_accounting": {"choice_generated_token_ids": [[1, 2, 3]]}
        },
    }
    assert _generated_ids(payload) == [1, 2, 3]
    assert _mtp_engaged(
        "speculative_mtp",
        {"used": True, "draft_tokens": 4, "draft_cycles": 2},
    )
    assert not _mtp_engaged(
        "speculative_mtp",
        {"used": False, "draft_tokens": 0, "draft_cycles": 0},
    )


def test_mtp_c1c8_diagnostic_plan_is_content_agnostic_and_bounded() -> None:
    base = {
        "realized_group_rows": 4,
        "candidate_budget": 2,
        "sampling_mode": "greedy_fast",
        "context_tokens": 64,
        "output_horizon_tokens": 24,
        "memory_fit": True,
    }
    first = _diagnostic_plan(**base)
    second = _diagnostic_plan(**base)

    assert first == second
    assert first["admitted"] is True
    assert first["selected_candidate_count"] == 2
    assert _diagnostic_plan(**{**base, "realized_group_rows": 5})["admitted"] is False
    assert _diagnostic_plan(**{**base, "context_tokens": 96})["admitted"] is False


def test_mtp_c1c8_summary_uses_complete_wall() -> None:
    cells = [
        {
            "width": 2,
            "exact": True,
            "mtp_engaged": True,
            "ar": {"wall_seconds": 2.0, "generated_tokens": 48},
            "mtp": {"wall_seconds": 1.5, "generated_tokens": 48},
        },
        {
            "width": 2,
            "exact": True,
            "mtp_engaged": True,
            "ar": {"wall_seconds": 2.0, "generated_tokens": 48},
            "mtp": {"wall_seconds": 1.5, "generated_tokens": 48},
        },
    ]

    row = summarize(cells, widths=(2,))["2"]
    assert row["ar"]["tok_s"] == pytest.approx(24.0)
    assert row["mtp"]["tok_s"] == pytest.approx(32.0)
    assert row["mtp_vs_ar_percent"] == pytest.approx(100.0 / 3.0)
    assert row["exact_cells"] == row["engaged_cells"] == row["cells"] == 2
