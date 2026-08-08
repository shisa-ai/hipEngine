from __future__ import annotations

import pytest

from scripts.pm4_packed_graph_bench import (
    _context_teardown_ok,
    _paired_transport_summary,
    _validate_sample_transport,
)


def _sample(
    transport: str,
    *,
    decode_seconds: float,
    capture_seconds: float,
    prompt_seconds: float = 0.5,
    token_hash: str = "exact",
) -> dict[str, object]:
    native = transport == "pm4"
    proof = {
        "transport": transport,
        "launches": 128,
        "native_fallbacks": 0,
    }
    if native:
        proof.update(
            {
                "stateful_registers": True,
                "local_cache_dependencies": True,
                "context": {"unretired_submissions": 0, "callback_status": 0},
                "executable": {"retired": True},
            }
        )
    return {
        "passed": True,
        "timings": {
            "decode_seconds": decode_seconds,
            "graph_capture_seconds": capture_seconds,
            "prefill_seconds": prompt_seconds,
        },
        "trajectory_fingerprints": [{"sha256": token_hash}],
        "graph_manifests": [
            {
                "graph": {
                    "transport": proof,
                }
            }
        ],
        "flush_results": [True],
    }


def test_paired_transport_summary_reports_replay_and_request_tradeoffs() -> None:
    pairs = [
        {
            "round": 0,
            "hipgraph": _sample("hipgraph", decode_seconds=2.0, capture_seconds=0.02),
            "pm4": _sample("pm4", decode_seconds=1.8, capture_seconds=0.08),
        },
        {
            "round": 1,
            "hipgraph": _sample("hipgraph", decode_seconds=2.2, capture_seconds=0.02),
            "pm4": _sample("pm4", decode_seconds=1.9, capture_seconds=0.08),
        },
    ]

    summary = _paired_transport_summary(pairs, logical_rows=4, decode_steps=128)

    assert summary["paired_replay_wins"] == 2
    assert summary["paired_rounds"] == 2
    assert summary["hipgraph"]["replay_ms_per_step"] == pytest.approx(2100.0 / 128)
    assert summary["pm4"]["replay_ms_per_step"] == pytest.approx(1850.0 / 128)
    assert summary["replay_wall_delta_percent"] == pytest.approx((1.85 / 2.1 - 1.0) * 100.0)
    assert summary["pm4"]["capture_ms"] == pytest.approx(80.0)
    assert summary["trajectories_exact"] is True


def test_validate_sample_transport_requires_canonical_retired_pm4() -> None:
    pm4 = _sample("pm4", decode_seconds=1.0, capture_seconds=0.1)
    hip = _sample("hipgraph", decode_seconds=1.0, capture_seconds=0.1)

    assert _validate_sample_transport(pm4, expected="pm4", steps=128) == []
    assert _validate_sample_transport(hip, expected="hipgraph", steps=128) == []

    proof = pm4["graph_manifests"][0]["graph"]["transport"]
    proof["local_cache_dependencies"] = False
    blockers = _validate_sample_transport(pm4, expected="pm4", steps=128)
    assert blockers == ["graph 0 did not use the canonical local-cache dependency encoder"]


def test_context_teardown_accepts_native_counters_nested_in_owner_proof() -> None:
    proofs = {
        "pm4": {
            "before": {
                "children": 0,
                "native": {"unretired_submissions": 0, "callback_status": 0},
            },
            "after": {"closed": True},
        }
    }

    assert _context_teardown_ok(proofs) is True
    proofs["pm4"]["before"]["native"]["unretired_submissions"] = 1
    assert _context_teardown_ok(proofs) is False
