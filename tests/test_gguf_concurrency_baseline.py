from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import gguf_concurrency_baseline as baseline


def _output(tokens: tuple[int, ...], timing: dict[str, float], *, owner: bool = True):
    return SimpleNamespace(
        generated_token_ids=tokens,
        telemetry=SimpleNamespace(
            timing=timing,
            timing_scope="choice",
            timing_owner=owner,
            group_rows=1,
            batch_id=None,
        ),
    )


def test_parse_concurrencies_requires_unique_positive_widths() -> None:
    assert baseline._parse_concurrencies("1,2,4") == (1, 2, 4)
    with pytest.raises(ValueError, match="positive"):
        baseline._parse_concurrencies("1,0")
    with pytest.raises(ValueError, match="unique"):
        baseline._parse_concurrencies("1,2,2")


def test_serial_execution_environment_disables_every_multirow_route() -> None:
    assert baseline._execution_environment("serial") == {
        "HIPENGINE_GGUF_AR_PACKED_PREFILL": "0",
        "HIPENGINE_GGUF_AR_PACKED_DECODE": "0",
        "HIPENGINE_GGUF_AR_STREAM_DECODE": "0",
    }
    assert baseline._execution_environment("package") == {
        "HIPENGINE_GGUF_AR_PACKED_PREFILL": "1",
        "HIPENGINE_GGUF_AR_PACKED_DECODE": "1",
        "HIPENGINE_GGUF_AR_STREAM_DECODE": "1",
    }


def test_measurement_sample_uses_only_timing_owners_and_exact_denominators() -> None:
    outputs = [
        _output((10, 11, 12), {"prefill_ms": 20.0, "decode_ms": 100.0}),
        _output((10, 11, 12), {"prefill_ms": 21.0, "decode_ms": 110.0}),
        _output(
            (99, 98, 97),
            {"prefill_ms": 999.0, "decode_ms": 999.0},
            owner=False,
        ),
    ]
    sample = baseline._measurement_sample(
        outputs,
        wall_seconds=0.5,
        prompt_length=512,
        last_batch_generation={
            "path": "gguf_serial_greedy_decode",
            "batch_size": 3,
            "native_caware_decode": False,
            "serial_decode_fallback": True,
        },
    )

    assert sample["accounting"] == {
        "prompt_tokens": 1536,
        "generated_tokens": 9,
        "continuation_decode_tokens": 6,
        "rows": 3,
    }
    assert sample["timing_ownership"] == {
        "timed_rows": 3,
        "owned_timing_rows": 2,
        "timing_scopes": ["choice"],
        "batch_ids": [],
    }
    assert sample["owned_timing_ms"] == {"decode_ms": 210.0, "prefill_ms": 41.0}
    assert sample["rates"]["prefill_tok_s"] == pytest.approx(1536 / 0.041)
    assert sample["rates"]["decode_tok_s"] == pytest.approx(6 / 0.210)
    assert sample["rates"]["generated_wall_tok_s"] == pytest.approx(18.0)
    assert sample["all_rows_same_trajectory"] is False
    assert sample["route"]["path"] == "gguf_serial_greedy_decode"


def test_summary_requires_repeatable_trajectories_and_reports_medians() -> None:
    sample_a = {
        "rates": {"prefill_tok_s": 100.0, "decode_tok_s": 20.0, "generated_wall_tok_s": 15.0},
        "wall_seconds": 2.0,
        "row_trajectories": [{"token_ids_sha256": "same"}],
        "route": {"path": "serial"},
    }
    sample_b = {
        "rates": {"prefill_tok_s": 110.0, "decode_tok_s": 22.0, "generated_wall_tok_s": 14.0},
        "wall_seconds": 2.1,
        "row_trajectories": [{"token_ids_sha256": "same"}],
        "route": {"path": "serial"},
    }
    summary = baseline._summarize_samples([sample_a, sample_b])
    assert summary["repeatable_trajectories"] is True
    assert summary["rates"]["prefill_tok_s"]["median"] == pytest.approx(105.0)
    assert summary["rates"]["decode_tok_s"]["samples"] == [20.0, 22.0]

    sample_b["row_trajectories"] = [{"token_ids_sha256": "different"}]
    assert baseline._summarize_samples([sample_a, sample_b])["repeatable_trajectories"] is False
