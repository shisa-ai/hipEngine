"""P6e cancel/refill service proof: the gate logic.

The gates only pass on a healthy service run, so their failure paths are driven
here with synthetic measurements. A gate that cannot fail is not evidence, and
the failure paths are exactly what a broken cancel or a stalled yield would
produce.
"""

from __future__ import annotations

import pytest

from scripts.gguf_p6e_cancel_refill_proof import (
    CANCELLATION_COUNTER,
    DECLARED_ACK_P95_LIMIT_MS,
    DECLARED_GAP_MAX_FACTOR,
    _labeled_metric_values,
    _metric_sample_labels,
    _split_prometheus_labels,
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
    # The indicator is the WHILE-LIVE observed peak: the live owner gauge reads
    # zero even when the route ran, because /metrics scrapes block on the loop
    # lock during a synchronous prefill.
    "oracle_observed_peak_owners": 1.0,
    "oracle_observed_peak_bytes": 912907308.0,
    "live_oracle_owners": 0.0,
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

    gates = _gates(oracle_observed_peak_owners=0.0, oracle_observed_peak_bytes=0.0)

    assert gates["resumable_path_engaged"]["passed"] is False
    # Every other gate can still pass, which is exactly the hazard: they would be
    # evidence about a route that does not contain the deliverable.
    assert gates["survivors_exact"]["passed"] is True
    assert gates["cancellation_reached_backend"]["passed"] is True


def test_resumable_gate_ignores_the_live_gauge_which_always_reads_zero() -> None:
    """The false negative this gate was rebuilt to avoid.

    The live owner gauge is unobservable from out of process while a synchronous
    prefill holds the loop lock, so a gate keyed on it fails no matter what the
    engine did. The while-live observed peak is the indicator that can pass.
    """

    gates = _gates(
        oracle_observed_peak_owners=1.0,
        oracle_observed_peak_bytes=912907308.0,
        live_oracle_owners=0.0,
    )

    assert gates["resumable_path_engaged"]["passed"] is True
    assert gates["resumable_path_engaged"]["live_gauge_owners"] == 0.0


def test_resumable_gate_fails_when_the_oracle_metrics_are_absent() -> None:
    gates = _gates(oracle_observed_peak_owners=None, oracle_observed_peak_bytes=None)

    assert gates["resumable_path_engaged"]["passed"] is False


def test_resumable_gate_fails_when_owners_rose_but_bytes_did_not() -> None:
    # A partial reading must not be accepted as engagement.
    gates = _gates(oracle_observed_peak_owners=1.0, oracle_observed_peak_bytes=0.0)

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


def test_layer_outer_flag_helper_reads_the_env_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engagement gate is only meaningful if the engine can read the flag.

    Without this, a run that passes --packed-layer-outer could still leave the
    route disabled by plumbing, and the gate would be blamed on the engine.
    """

    from hipengine.runtime import qwen35_gguf_runner as runner

    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "1")
    monkeypatch.setattr(runner, "_gguf_packed_layer_outer_enabled_cache", None)
    assert runner._gguf_packed_layer_outer_enabled() is True

    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "0")
    monkeypatch.setattr(runner, "_gguf_packed_layer_outer_enabled_cache", None)
    assert runner._gguf_packed_layer_outer_enabled() is False

    monkeypatch.delenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", raising=False)
    monkeypatch.setattr(runner, "_gguf_packed_layer_outer_enabled_cache", None)
    # Default OFF: the corrected chunk-outer executor remains the fallback.
    assert runner._gguf_packed_layer_outer_enabled() is False


def test_engagement_gate_reports_what_actually_ran_not_just_that_it_failed() -> None:
    """A red gate must name the reason, not leave it to be inferred.

    Before the executor-mode export existed, a decline could only be inferred
    from an absent fallback counter. The gate now carries the observed modes so
    the artifact records which executor ran.
    """

    gates = _gates(
        oracle_observed_peak_owners=0.0,
        oracle_observed_peak_bytes=0.0,
        executor_modes={"none": 2.0},
    )

    gate = gates["resumable_path_engaged"]
    assert gate["passed"] is False
    assert gate["executor_modes"] == {"none": 2.0}
    assert gate["detail"].endswith(
        "decline reason instead of leaving it to be inferred"
    )


def test_engagement_gate_can_pass_when_the_layer_outer_mode_is_observed() -> None:
    gates = _gates(
        oracle_observed_peak_owners=1.0,
        oracle_observed_peak_bytes=912907308.0,
        executor_modes={"layer_outer_packed": 1.0},
    )

    assert gates["resumable_path_engaged"]["passed"] is True
    assert gates["resumable_path_engaged"]["executor_modes"] == {
        "layer_outer_packed": 1.0
    }


def test_labeled_metric_values_keeps_the_label_that_identifies_the_path() -> None:
    """_metrics_values sums labels; this one must not, or the mode is lost."""

    text = (
        "# HELP hipengine_resident_prefill_executor_mode_total executor\n"
        "# TYPE hipengine_resident_prefill_executor_mode_total counter\n"
        'hipengine_resident_prefill_executor_mode_total{mode="layer_outer_packed"} 1\n'
        'hipengine_resident_prefill_executor_mode_total{mode="none"} 2\n'
        "hipengine_resident_other_total 5\n"
    )

    class _Response:
        status_code = 200

        def __init__(self, body: str) -> None:
            self.text = body

    class _Client:
        def get(self, url: str) -> _Response:
            return _Response(text)

    values = _labeled_metric_values(
        _Client(), "http://x", "hipengine_resident_prefill_executor_mode_total"
    )  # type: ignore[arg-type]

    assert values == {"layer_outer_packed": 1.0, "none": 2.0}
    # A metric family that is absent yields nothing rather than a fabricated zero.
    assert (
        _labeled_metric_values(
            _Client(), "http://x", "hipengine_resident_absent_total"
        )  # type: ignore[arg-type]
        == {}
    )


def test_metric_sample_labels_reads_a_multi_label_family() -> None:
    """The route manifest carries its meaning across several labels."""

    text = (
        "# HELP hipengine_resident_route_manifest_info identity\n"
        "# TYPE hipengine_resident_route_manifest_info gauge\n"
        "hipengine_resident_route_manifest_info{"
        'claim_level="none",kind="gguf_ar_serial_fallback_execution_manifest",'
        'kv_attention_source="bf16_mirror",mode="serial_c1_per_row",rows="2"} 1\n'
    )

    class _Response:
        status_code = 200

        def __init__(self, body: str) -> None:
            self.text = body

    class _Client:
        def get(self, url: str) -> _Response:
            return _Response(text)

    labels = _metric_sample_labels(
        _Client(), "http://x", "hipengine_resident_route_manifest_info"
    )  # type: ignore[arg-type]

    assert labels["kv_attention_source"] == "bf16_mirror"
    assert labels["kind"] == "gguf_ar_serial_fallback_execution_manifest"
    assert len(labels) == 5
    # An absent family yields nothing rather than a fabricated source.
    assert (
        _metric_sample_labels(_Client(), "http://x", "hipengine_absent_info")
        == {}  # type: ignore[arg-type]
    )


def test_split_prometheus_labels_ignores_commas_inside_quoted_values() -> None:
    """A label value containing a comma must not split into two labels."""

    assert _split_prometheus_labels('a="1",b="2"') == ['a="1"', 'b="2"']
    assert _split_prometheus_labels('mode="a,b",c="d"') == ['mode="a,b"', 'c="d"']
    # An escaped quote does not close the value.
    assert _split_prometheus_labels('mode="a\\",b",c="d"') == [
        'mode="a\\",b"',
        'c="d"',
    ]
    assert _split_prometheus_labels("") == []


def test_engagement_gate_reports_whether_the_resumable_path_was_even_attempted() -> None:
    """The outer int8_direct gate must be visible, not inferred.

    An empty fallback counter only implies the int8_direct branch was not
    entered. Recording kv_attention_source states it.
    """

    gates = _gates(
        oracle_observed_peak_owners=0.0,
        oracle_observed_peak_bytes=0.0,
        executor_modes={"None": 2.0},
        kv_attention_source="bf16_mirror",
    )

    gate = gates["resumable_path_engaged"]
    assert gate["passed"] is False
    assert gate["kv_attention_source"] == "bf16_mirror"
    assert gate["resumable_kv_source"] == "int8_direct"
    assert "whether the resumable path was even attempted" in gate["detail"]


def test_engagement_gate_still_passes_when_the_source_is_int8_direct() -> None:
    gates = _gates(
        oracle_observed_peak_owners=1.0,
        oracle_observed_peak_bytes=912907308.0,
        executor_modes={"layer_outer_packed": 1.0},
        kv_attention_source="int8_direct",
    )

    assert gates["resumable_path_engaged"]["passed"] is True
