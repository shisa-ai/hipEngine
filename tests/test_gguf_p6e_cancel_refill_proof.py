"""P6e cancel/refill service proof: the gate logic.

The gates only pass on a healthy service run, so their failure paths are driven
here with synthetic measurements. A gate that cannot fail is not evidence, and
the failure paths are exactly what a broken cancel or a stalled yield would
produce.
"""

from __future__ import annotations

from scripts.gguf_p6e_cancel_refill_proof import (
    CANCELLATION_COUNTER,
    DECLARED_ACK_P95_LIMIT_MS,
    DECLARED_GAP_MAX_FACTOR,
    evaluate_gates,
)

_REFERENCE_TEXT = "The quick brown fox jumps over the lazy dog, and then it rests."
_REFERENCE_GAPS = [33.0, 34.0, 35.0, 36.0]

# A healthy run: the cancel reached the backend, the refill reproduced the
# reference exactly, and both latency gates are inside their declared limits.
_HEALTHY = {
    "reference_text": _REFERENCE_TEXT,
    "refill_text": _REFERENCE_TEXT,
    "cancellation_delta": 1.0,
    "acknowledgement_ms": 120.0,
    "refill_gaps_ms": [33.5, 34.2, 34.9],
    "reference_gaps_ms": _REFERENCE_GAPS,
    "drained": True,
    "prefill_owner_released": True,
    # The blocking and slow-consumer arms submit the same short prompt as the
    # reference arm, so the healthy expectation is the same text.
    "blocking_survivor_text": _REFERENCE_TEXT,
    "slow_consumer_survivor_text": _REFERENCE_TEXT,
    "slow_consumer_completed": True,
    # The layer-outer route must be seen to engage, or the other gates are
    # evidence about the default chunk-outer path instead of the deliverable.
    "oracle_peak_owners": 1.0,
    "oracle_peak_bytes": 912907308.0,
}


def _gates(**overrides: object) -> dict:
    payload = dict(_HEALTHY)
    payload.update(overrides)
    return evaluate_gates(**payload)  # type: ignore[arg-type]


def test_all_gates_pass_on_a_healthy_run() -> None:
    gates = _gates()

    assert set(gates) == {
        "cancellation_reached_backend",
        "resumable_path_engaged",
        "survivors_exact",
        "blocking_survivor_exact",
        "slow_consumer_survivor_exact",
        "bounded_acknowledgement",
        "bounded_decode_gap",
        "cleanup",
    }
    assert all(gate["passed"] for gate in gates.values())


def test_resumable_gate_fails_when_the_layer_outer_route_never_engaged() -> None:
    """The gate that stops this proof from silently testing the default route."""

    gates = _gates(oracle_peak_owners=0.0, oracle_peak_bytes=0.0)

    assert gates["resumable_path_engaged"]["passed"] is False
    # Every other gate can still pass, which is exactly the hazard: they would be
    # evidence about a route that does not contain the deliverable.
    assert gates["survivors_exact"]["passed"] is True
    assert gates["cancellation_reached_backend"]["passed"] is True


def test_resumable_gate_fails_when_the_oracle_metrics_are_absent() -> None:
    gates = _gates(oracle_peak_owners=None, oracle_peak_bytes=None)

    assert gates["resumable_path_engaged"]["passed"] is False


def test_resumable_gate_fails_when_owners_rose_but_bytes_did_not() -> None:
    # A partial reading must not be accepted as engagement.
    gates = _gates(oracle_peak_owners=1.0, oracle_peak_bytes=0.0)

    assert gates["resumable_path_engaged"]["passed"] is False


def test_blocking_survivor_gate_fails_when_the_blocking_arm_perturbs_it() -> None:
    """A blocking long request must not perturb the short survivor either."""

    gates = _gates(blocking_survivor_text=_REFERENCE_TEXT.replace("rests", "sleeps"))

    assert gates["blocking_survivor_exact"]["passed"] is False
    assert gates["blocking_survivor_exact"]["sha256"] != gates["survivors_exact"][
        "reference_sha256"
    ]


def test_blocking_survivor_gate_fails_when_the_arm_never_ran() -> None:
    # An empty survivor means the blocking arm did not produce a completion, which
    # must fail rather than pass vacuously.
    gates = _gates(blocking_survivor_text="")

    assert gates["blocking_survivor_exact"]["passed"] is False


def test_slow_consumer_gate_fails_when_the_slow_reader_stalls_the_engine() -> None:
    gates = _gates(slow_consumer_survivor_text=_REFERENCE_TEXT[:12])

    assert gates["slow_consumer_survivor_exact"]["passed"] is False


def test_slow_consumer_gate_reports_whether_the_reader_finished() -> None:
    # The survivor can be exact while the slow reader itself never completed; the
    # gate must surface that separately rather than hiding it behind a pass.
    gates = _gates(slow_consumer_completed=False)

    assert gates["slow_consumer_survivor_exact"]["passed"] is True
    assert gates["slow_consumer_survivor_exact"]["slow_consumer_completed"] is False


def test_slow_consumer_gate_fails_when_the_slow_reader_stalls_and_truncates() -> None:
    gates = _gates(slow_consumer_survivor_text="", slow_consumer_completed=False)

    assert gates["slow_consumer_survivor_exact"]["passed"] is False


def test_cancellation_gate_fails_when_the_counter_did_not_move() -> None:
    """The whole point: a client abort that never reached the backend."""

    gates = _gates(cancellation_delta=0.0)

    assert gates["cancellation_reached_backend"]["passed"] is False
    assert gates["cancellation_reached_backend"]["counter"] == CANCELLATION_COUNTER


def test_cancellation_gate_fails_when_the_counter_is_unavailable() -> None:
    gates = _gates(cancellation_delta=None)

    assert gates["cancellation_reached_backend"]["passed"] is False


def test_survivors_gate_fails_when_the_refill_diverges() -> None:
    gates = _gates(refill_text=_REFERENCE_TEXT.replace("rests", "sleeps"))

    assert gates["survivors_exact"]["passed"] is False
    assert gates["survivors_exact"]["refill_sha256"] != gates["survivors_exact"][
        "reference_sha256"
    ]


def test_survivors_gate_fails_when_the_refill_is_truncated() -> None:
    gates = _gates(refill_text=_REFERENCE_TEXT[:20])

    assert gates["survivors_exact"]["passed"] is False


def test_survivors_gate_fails_when_the_refill_produced_nothing() -> None:
    gates = _gates(refill_text="")

    assert gates["survivors_exact"]["passed"] is False


def test_acknowledgement_gate_fails_past_the_declared_limit() -> None:
    gates = _gates(acknowledgement_ms=DECLARED_ACK_P95_LIMIT_MS + 1.0)

    assert gates["bounded_acknowledgement"]["passed"] is False


def test_acknowledgement_gate_fails_when_the_refill_never_emitted() -> None:
    gates = _gates(acknowledgement_ms=None)

    assert gates["bounded_acknowledgement"]["passed"] is False


def test_decode_gap_gate_fails_past_the_declared_factor() -> None:
    worst = max(_REFERENCE_GAPS) * (DECLARED_GAP_MAX_FACTOR + 0.5)
    gates = _gates(refill_gaps_ms=[33.5, worst])

    assert gates["bounded_decode_gap"]["passed"] is False
    assert gates["bounded_decode_gap"]["declared_factor"] == DECLARED_GAP_MAX_FACTOR


def test_decode_gap_gate_fails_when_no_gaps_were_observed() -> None:
    gates = _gates(refill_gaps_ms=[])

    assert gates["bounded_decode_gap"]["passed"] is False


def test_decode_gap_gate_fails_without_a_reference_to_bound_against() -> None:
    gates = _gates(reference_gaps_ms=[])

    assert gates["bounded_decode_gap"]["passed"] is False


def test_cleanup_gate_fails_when_the_scheduler_never_drained() -> None:
    """A cancelled request that never releases its slot is the leak this catches."""

    gates = _gates(drained=False)

    assert gates["cleanup"]["passed"] is False


def test_cleanup_gate_fails_when_suspended_prefill_state_leaked() -> None:
    """The cancelled prefill's hidden/oracle owner bytes must return to baseline."""

    gates = _gates(prefill_owner_released=False)

    assert gates["cleanup"]["passed"] is False
    assert gates["cleanup"]["prefill_owner_released"] is False


def test_a_single_divergent_character_fails_the_survivors_gate() -> None:
    """Exactness matters: one wrong character is not a survivor."""

    gates = _gates(refill_text=_REFERENCE_TEXT.replace("lazy", "la2y"))

    assert gates["survivors_exact"]["passed"] is False
