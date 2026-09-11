"""P6e resumable-prefill GPU proof: the gate logic and the production defect.

Two things are checked here, both of which the earlier P6 CPU tests missed:

1. The four P6e gates must be *able to fail*. A gate that only ever passes is not
   evidence, and the failure paths never occur on a healthy GPU run, so they are
   driven here with synthetic measurements.
2. The resumable prefill must seed the plan counters its sampling tail increments.
   The P6c fixture pre-populated exactly those counters, which hid a real
   ``KeyError`` that the first GPU proof run hit on a fresh session; the fixture
   now uses the dataclass default and these tests pin the production behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.gguf_resumable_prefill_gpu_proof import (
    DECLARED_DECODE_GAP_MAX_FACTOR,
    DECLARED_DECODE_GAP_P95_FACTOR,
    DECLARED_SEGMENT_WORK_FLOOR_MS,
    _percentiles,
    evaluate_gates,
)

# A healthy set of measurements: 16 real segments, a decode gap at parity with
# the standalone step, exact continuation, and the scratch released.
_HEALTHY = {
    "arm_a_token": 265,
    "arm_b_token": 265,
    "segments": {"count": 16, "median_ms": 273.478, "p95_ms": 297.094, "min_ms": 270.152, "max_ms": 300.136},
    "baseline": {"count": 24, "median_ms": 34.203, "p95_ms": 34.473, "min_ms": 33.063, "max_ms": 39.795},
    "gap": {"count": 15, "median_ms": 34.011, "p95_ms": 34.132, "min_ms": 33.796, "max_ms": 34.25},
    "scratch_released": True,
    "peak_suspended_bytes": 221_773_824,
}


def _gates(**overrides: object) -> dict:
    payload = dict(_HEALTHY)
    payload.update(overrides)
    return evaluate_gates(**payload)  # type: ignore[arg-type]


def test_all_four_gates_pass_on_a_healthy_run() -> None:
    gates = _gates()

    assert set(gates) == {
        "exact_continuation",
        "real_yield",
        "bounded_decode_gap",
        "cleanup",
    }
    assert all(gate["passed"] for gate in gates.values())


def test_exact_continuation_fails_when_the_arms_disagree() -> None:
    gates = _gates(arm_b_token=266)

    assert gates["exact_continuation"]["passed"] is False
    assert gates["exact_continuation"]["arm_a_token_id"] == 265
    assert gates["exact_continuation"]["arm_b_token_id"] == 266


def test_exact_continuation_fails_when_the_resumable_arm_produced_nothing() -> None:
    gates = _gates(arm_b_token=None)

    assert gates["exact_continuation"]["passed"] is False


def test_real_yield_fails_when_the_prefill_never_yielded() -> None:
    gates = _gates(
        segments={"count": 1, "median_ms": 4400.0, "p95_ms": 4400.0, "min_ms": 4400.0, "max_ms": 4400.0}
    )

    assert gates["real_yield"]["passed"] is False


def test_real_yield_fails_when_a_segment_only_advanced_bookkeeping() -> None:
    """A sub-millisecond segment means the yield did no GPU work."""

    gates = _gates(
        segments={"count": 16, "median_ms": 273.0, "p95_ms": 297.0, "min_ms": 0.004, "max_ms": 300.0}
    )

    assert gates["real_yield"]["passed"] is False
    assert gates["real_yield"]["segment_min_ms"] < DECLARED_SEGMENT_WORK_FLOOR_MS


def test_bounded_decode_gap_fails_when_p95_exceeds_the_declared_factor() -> None:
    baseline_p95 = _HEALTHY["baseline"]["p95_ms"]  # type: ignore[index]
    gates = _gates(
        gap={
            "count": 15,
            "median_ms": 34.0,
            "p95_ms": baseline_p95 * (DECLARED_DECODE_GAP_P95_FACTOR + 0.5),
            "min_ms": 33.8,
            "max_ms": 34.3,
        }
    )

    assert gates["bounded_decode_gap"]["passed"] is False


def test_bounded_decode_gap_fails_when_max_exceeds_the_declared_factor() -> None:
    baseline_max = _HEALTHY["baseline"]["max_ms"]  # type: ignore[index]
    gates = _gates(
        gap={
            "count": 15,
            "median_ms": 34.0,
            "p95_ms": 34.1,
            "min_ms": 33.8,
            "max_ms": baseline_max * (DECLARED_DECODE_GAP_MAX_FACTOR + 0.5),
        }
    )

    assert gates["bounded_decode_gap"]["passed"] is False


def test_bounded_decode_gap_fails_when_no_decode_ran() -> None:
    gates = _gates(gap={})

    assert gates["bounded_decode_gap"]["passed"] is False


def test_cleanup_fails_when_the_scratch_outlives_the_checkpoint() -> None:
    gates = _gates(scratch_released=False)

    assert gates["cleanup"]["passed"] is False


def test_cleanup_records_the_peak_suspension_as_evidence() -> None:
    gates = _gates()

    assert gates["cleanup"]["peak_suspended_bytes"] > 0


def test_percentiles_are_empty_for_no_samples() -> None:
    assert _percentiles([]) == {}


def test_percentiles_report_the_observed_range() -> None:
    stats = _percentiles([3.0, 1.0, 2.0, 4.0])

    assert stats["count"] == 4
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 4.0
    assert stats["median_ms"] == 2.5


def test_resumable_plan_seeds_every_counter_the_sampling_tail_increments() -> None:
    """The production defect the first GPU proof run hit.

    A fresh session's ``last_packed_prefill_plan`` is the dataclass default, and
    the resumable sampling tail increments four counters with ``+=``. The
    resumable setup must therefore seed all four.
    """

    plan_field = Qwen35GGUFResidentSession.__dataclass_fields__["last_packed_prefill_plan"]

    assert plan_field.default_factory() == {}  # type: ignore[misc]

    source = (
        Path(__file__).resolve().parents[1]
        / "hipengine"
        / "runtime"
        / "qwen35_gguf_runner.py"
    ).read_text(encoding="utf-8")
    # Every counter the tail mutates must be seeded by a setdefault in the same
    # module, otherwise a fresh-session resumable prefill raises KeyError.
    for counter in (
        "output_norm_rows",
        "lm_head_sample_rows",
        "target_hidden_norm_rows",
        "hidden_seed_norm_rows",
    ):
        assert 'setdefault(_plan_counter, 0)' in source
        assert f'last_packed_prefill_plan["{counter}"] +=' in source
        assert f'"{counter}",' in source


def test_the_p6c_fixture_does_not_pre_populate_the_plan() -> None:
    """The fixture must not supply state production is responsible for."""

    fixture = (
        Path(__file__).resolve().parent / "test_gguf_resumable_layer_outer_prefill.py"
    ).read_text(encoding="utf-8")

    assert 'last_packed_prefill_plan={"output_norm_rows": 0' not in fixture
    assert "last_packed_prefill_plan={}," in fixture


@pytest.mark.parametrize("counter", ("output_norm_rows", "lm_head_sample_rows"))
def test_the_tail_counters_are_the_ones_that_crashed(counter: str) -> None:
    """Pin the exact counters from the observed traceback line."""

    source = (
        Path(__file__).resolve().parents[1]
        / "hipengine"
        / "runtime"
        / "qwen35_gguf_runner.py"
    ).read_text(encoding="utf-8")

    assert f'self.last_packed_prefill_plan["{counter}"] += sample_output_rows' in source
