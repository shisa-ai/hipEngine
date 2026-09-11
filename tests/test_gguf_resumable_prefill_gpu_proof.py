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
    _capture_prefill_state,
    _committed_prefix_nbytes,
    _percentiles,
    _state_mismatches,
    evaluate_gates,
)

# A healthy set of measurements: 16 real segments, a decode gap at parity with
# the standalone step, exact continuation, the scratch released, and the
# segmented arm's committed state fingerprinting identically to the reference.
_HEALTHY = {
    "arm_a_token": 265,
    "arm_b_token": 265,
    "segments": {"count": 16, "median_ms": 273.478, "p95_ms": 297.094, "min_ms": 270.152, "max_ms": 300.136},
    "baseline": {"count": 24, "median_ms": 34.203, "p95_ms": 34.473, "min_ms": 33.063, "max_ms": 39.795},
    "gap": {"count": 15, "median_ms": 34.011, "p95_ms": 34.132, "min_ms": 33.796, "max_ms": 34.25},
    "scratch_released": True,
    "peak_suspended_bytes": 221_773_824,
    "state_layers_compared": 14,
    "state_mismatches": [],
    "final_round_rows": 1024,
    "row_capacity": 1024,
    "ragged_declared": False,
    "hidden_planes_aliased": True,
    "hidden_planes_aliased_reference": True,
    "alias_declared": True,
}


def _gates(**overrides: object) -> dict:
    payload = dict(_HEALTHY)
    payload.update(overrides)
    return evaluate_gates(**payload)  # type: ignore[arg-type]


def test_all_seven_gates_pass_on_a_healthy_run() -> None:
    gates = _gates()

    assert set(gates) == {
        "exact_continuation",
        "real_yield",
        "bounded_decode_gap",
        "cleanup",
        "layer_boundary_state",
        "ragged_final_round",
        "hidden_plane_alias",
    }
    assert all(gate["passed"] for gate in gates.values())


def test_hidden_plane_alias_passes_for_the_declared_two_plane_control() -> None:
    gates = _gates(
        hidden_planes_aliased=False,
        hidden_planes_aliased_reference=False,
        alias_declared=False,
    )

    assert gates["hidden_plane_alias"]["passed"] is True


def test_hidden_plane_alias_fails_when_the_alias_did_not_apply() -> None:
    """Declaring the alias while the run measured two planes is not evidence."""

    gates = _gates(hidden_planes_aliased=False, hidden_planes_aliased_reference=False)

    assert gates["hidden_plane_alias"]["passed"] is False


def test_hidden_plane_alias_fails_when_the_arms_disagree() -> None:
    """One arm aliased and the other not is a configuration defect."""

    gates = _gates(hidden_planes_aliased_reference=False)

    assert gates["hidden_plane_alias"]["passed"] is False


def test_ragged_final_round_passes_when_a_short_round_was_declared() -> None:
    gates = _gates(final_round_rows=100, ragged_declared=True)

    assert gates["ragged_final_round"]["passed"] is True
    assert gates["ragged_final_round"]["final_round_rows"] == 100


def test_ragged_final_round_fails_when_a_ragged_prompt_was_padded() -> None:
    """A declared ragged shape that ran full rounds is not ragged coverage."""

    gates = _gates(final_round_rows=1024, ragged_declared=True)

    assert gates["ragged_final_round"]["passed"] is False


def test_ragged_final_round_fails_when_a_full_round_was_declared_but_short() -> None:
    """A shape that silently became ragged did not exercise full rounds."""

    gates = _gates(final_round_rows=100, ragged_declared=False)

    assert gates["ragged_final_round"]["passed"] is False


def test_ragged_final_round_fails_on_a_degenerate_shape() -> None:
    assert _gates(final_round_rows=0) ["ragged_final_round"]["passed"] is False
    assert _gates(row_capacity=0)["ragged_final_round"]["passed"] is False
    assert (
        _gates(final_round_rows=2048, row_capacity=1024)["ragged_final_round"][
            "passed"
        ]
        is False
    )


def test_layer_boundary_state_fails_when_a_layer_mismatches() -> None:
    """A single differing layer's K/V or linear state must fail the gate."""

    gates = _gates(
        state_mismatches=[{"section": "kv", "row": 3, "layer": 3}],
    )

    assert gates["layer_boundary_state"]["passed"] is False
    assert gates["layer_boundary_state"]["mismatch_count"] == 1
    assert gates["layer_boundary_state"]["mismatches"][0]["layer"] == 3


def test_layer_boundary_state_fails_when_nothing_was_compared() -> None:
    """A gate with no captured layers is not evidence and must not pass."""

    gates = _gates(state_layers_compared=0)

    assert gates["layer_boundary_state"]["passed"] is False


def test_state_mismatches_reports_section_row_and_count() -> None:
    expected = {
        "position": 32,
        "linear": [{"layer": 1, "conv": "a", "recurrent": "b"}],
        "kv": [{"layer": 0, "key_payload": "c", "value_payload": "d"}],
    }
    actual = {
        "position": 32,
        "linear": [{"layer": 1, "conv": "a", "recurrent": "CHANGED"}],
        "kv": [{"layer": 0, "key_payload": "c", "value_payload": "d"}],
    }

    mismatches = _state_mismatches(actual, expected)

    assert len(mismatches) == 1
    assert mismatches[0]["section"] == "linear"
    assert mismatches[0]["row"] == 0
    assert mismatches[0]["layer"] == 1
    assert mismatches[0]["actual_sha256"] != mismatches[0]["expected_sha256"]


def test_state_mismatches_reports_a_row_count_difference() -> None:
    expected = {"position": 32, "linear": [], "kv": [{"layer": 0}]}
    actual = {"position": 32, "linear": [], "kv": [{"layer": 0}, {"layer": 1}]}

    mismatches = _state_mismatches(actual, expected)

    assert len(mismatches) == 1
    assert mismatches[0]["section"] == "kv"
    assert mismatches[0]["detail"] == "row count differs"
    assert mismatches[0]["actual_rows"] == 2
    assert mismatches[0]["expected_rows"] == 1


def test_state_mismatches_is_empty_for_identical_states() -> None:
    state = {
        "position": 32,
        "linear": [{"layer": 1, "conv": "a", "recurrent": "b"}],
        "kv": [{"layer": 0, "key_payload": "c", "value_payload": "d"}],
    }

    assert _state_mismatches(state, dict(state)) == []


def test_state_mismatches_names_the_differing_fields() -> None:
    expected = {"position": 32, "linear": [], "kv": [{"layer": 0, "key_scale": "a"}]}
    actual = {"position": 32, "linear": [], "kv": [{"layer": 0, "key_scale": "b"}]}

    mismatches = _state_mismatches(actual, expected)

    assert mismatches[0]["fields"] == ["key_scale"]


# ---------------------------------------------------------------------------
# P6f: the layer-boundary state comparison must not read a plane's dead tail
# ---------------------------------------------------------------------------


def test_committed_prefix_nbytes_uses_the_planes_own_row_width() -> None:
    """Scale planes are narrower per row than payload planes; derive, not guess.

    A full-context INT8 payload plane holds 16,384 positions at 512 bytes each;
    the matching per-token-head fp32 scale plane holds the same 16,384 rows at
    16 bytes each. Committing 3,072 rows must select 3,072 rows of *each* plane,
    not the payload row width applied to the scale plane.
    """

    assert (
        _committed_prefix_nbytes(16_384 * 512, total_rows=16_384, committed_rows=3_072)
        == 3_072 * 512
    )
    assert (
        _committed_prefix_nbytes(16_384 * 16, total_rows=16_384, committed_rows=3_072)
        == 3_072 * 16
    )


def test_committed_prefix_nbytes_clamps_and_handles_degenerate_inputs() -> None:
    assert _committed_prefix_nbytes(0, total_rows=10, committed_rows=5) == 0
    assert _committed_prefix_nbytes(100, total_rows=0, committed_rows=5) == 0
    # More committed rows than the plane holds selects the whole plane.
    assert _committed_prefix_nbytes(100, total_rows=10, committed_rows=99) == 100
    assert _committed_prefix_nbytes(100, total_rows=10, committed_rows=0) == 0


def _capture_session(
    *,
    position: int,
    head_count_kv: int = 4,
    key_length: int = 128,
    total_positions: int = 16_384,
    scale_dtype_itemsize: int = 4,
):
    """Minimal session surface for ``_capture_prefill_state``."""

    from types import SimpleNamespace

    payload_row_nbytes = head_count_kv * key_length
    return SimpleNamespace(
        position=int(position),
        runtime=SimpleNamespace(device_synchronize=lambda: None),
        runner=SimpleNamespace(
            weights=SimpleNamespace(
                config=SimpleNamespace(
                    head_count_kv=head_count_kv,
                    key_length=key_length,
                )
            )
        ),
        scratch=SimpleNamespace(
            layer_conv_states=(None, None),
            layer_recurrent_states=(None, None),
            full_key_caches=(
                SimpleNamespace(ptr=0x1000, nbytes=total_positions * payload_row_nbytes),
            ),
            full_value_caches=(
                SimpleNamespace(ptr=0x2000, nbytes=total_positions * payload_row_nbytes),
            ),
            full_scale_metadata=lambda layer_id: SimpleNamespace(
                k_scale=SimpleNamespace(
                    ptr=0x3000,
                    numel=total_positions * head_count_kv,
                    dtype=SimpleNamespace(itemsize=scale_dtype_itemsize),
                ),
                v_scale=SimpleNamespace(
                    ptr=0x4000,
                    numel=total_positions * head_count_kv,
                    dtype=SimpleNamespace(itemsize=scale_dtype_itemsize),
                ),
            ),
        ),
    )


def test_capture_hashes_only_the_committed_prefix_of_every_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dead tail of a full-context plane must never enter the comparison.

    This is the defect the first GPU run of the gate hit: the scale planes were
    hashed in full, so their uninitialized tail differed between two
    independently allocated sessions and eight full-attention layers were
    reported as mismatches while their committed K/V was in fact identical.
    """

    import scripts.gguf_resumable_prefill_gpu_proof as proof

    recorded: list[int] = []

    def fake_hash(session: object, ptr: int, nbytes: int) -> str:
        recorded.append(int(nbytes))
        return "hash"

    monkeypatch.setattr(proof, "_device_hash", fake_hash)
    _capture_prefill_state(_capture_session(position=3_072))

    payload_row_nbytes = 4 * 128
    committed_payload = 3_072 * payload_row_nbytes
    # One key + one value payload, then one key + one value scale, all clamped
    # to the committed rows rather than the full 16,384-position context.
    assert recorded == [committed_payload, committed_payload, 3_072 * 16, 3_072 * 16]
    assert committed_payload < 16_384 * payload_row_nbytes
    assert 3_072 * 16 < 16_384 * 16


def test_capture_reports_full_plane_sizes_for_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.gguf_resumable_prefill_gpu_proof as proof

    monkeypatch.setattr(proof, "_device_hash", lambda *a, **k: "hash")
    state = _capture_prefill_state(_capture_session(position=3_072))

    row = state["kv"][0]
    assert row["payload_nbytes"] == 3_072 * 4 * 128
    assert row["key_scale_nbytes"] == 16_384 * 4 * 4
    assert row["value_scale_nbytes"] == 16_384 * 4 * 4
    assert state["position"] == 3_072


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
