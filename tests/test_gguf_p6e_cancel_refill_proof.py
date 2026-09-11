"""P6e cancel/refill service proof: the gate logic.

The five gates only pass on a healthy service run, so their failure paths are
driven here with synthetic measurements. A gate that cannot fail is not evidence,
and the failure paths are exactly what a broken cancel would produce.
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
}


def _gates(**overrides: object) -> dict:
    payload = dict(_HEALTHY)
    payload.update(overrides)
    return evaluate_gates(**payload)  # type: ignore[arg-type]


def test_all_five_gates_pass_on_a_healthy_run() -> None:
    gates = _gates()

    assert set(gates) == {
        "cancellation_reached_backend",
        "survivors_exact",
        "bounded_acknowledgement",
        "bounded_decode_gap",
        "cleanup",
    }
    assert all(gate["passed"] for gate in gates.values())


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
