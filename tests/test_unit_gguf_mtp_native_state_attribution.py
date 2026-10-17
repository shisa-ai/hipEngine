from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_mtp_forced_target_probe import ProbeError
from scripts.gguf_mtp_native_state_attribution import (
    LANE_PACKED,
    LANE_SINGLETON,
    VERDICT_CLEAN,
    VERDICT_NUMERICAL_DRIFT,
    VERDICT_OWNERSHIP_DEFECT,
    VERDICT_OWNERSHIP_UNVERIFIED,
    VERDICT_REPEATABILITY,
    VERDICT_SERIAL_CONTROL,
    _record_reference_row,
    _top1_only_agreement,
    attribute_divergence,
    build_arg_parser,
    commit_ownership_check,
    logits_agreement,
    repeatability_check,
    require_reproducible_lane,
    resolve_lane,
    state_numeric_delta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fingerprint(
    *,
    label: str,
    position: int,
    current_prev: int,
    digest: str,
    layer_digest: str = "layer",
) -> dict[str, object]:
    return {
        "label": label,
        "position": int(position),
        "current_prev": int(current_prev),
        "hidden_seed": {"blake2b_128": digest},
        "linear_state_layers": [
            {
                "layer": 0,
                "conv": {"blake2b_128": layer_digest, "nbytes": 64},
                "recurrent": {"blake2b_128": layer_digest, "nbytes": 64},
            }
        ],
    }


def _logits(rows: int, vocab: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(rows, vocab)), dtype=np.float32)


def _flip_row(logits: np.ndarray, row: int) -> np.ndarray:
    """Return a copy whose ``row`` argmax moved to a different vocabulary id."""

    flipped = logits.copy()
    vocab = int(flipped.shape[1])
    winner = int(np.argmax(flipped[row]))
    other = (winner + 1) % vocab
    flipped[row] = np.float32(0.0)
    flipped[row, other] = np.float32(50.0)
    return flipped


def _exact_ownership() -> dict[str, object]:
    return {
        "status": "exact",
        "layers_checked": 1,
        "contracts": {
            "strict_state_copy": {"status": "exact", "mismatches": []},
            "ownership": {"status": "exact", "mismatches": []},
        },
        "mismatches": [],
    }


def _unverified_ownership() -> dict[str, object]:
    return {
        "status": "not_checked",
        "layers_checked": 0,
        "contracts": {
            "strict_state_copy": {"status": "not_checked", "mismatches": []},
            "ownership": {"status": "exact", "mismatches": []},
        },
        "mismatches": [],
    }


def _commit_record(
    *,
    accepted: int = 1,
    rows: int = 3,
    commit_row: int | None = None,
    commit_token: int | None = None,
    commit_position: int | None = None,
    start_position: int = 100,
    end_position: int | None = None,
    cursor_after: int | None = None,
    input_token_ids: list[int] | None = None,
    visible_tokens: list[int] | None = None,
    draft_tokens: list[int] | None = None,
    target_top1: list[int] | None = None,
    source: bytes = b"conv-row",
    destination: bytes | None = None,
    other_row: bytes | None = None,
    hidden_source: bytes | None = None,
    hidden_destination: bytes | None = None,
) -> dict[str, object]:
    tokens = input_token_ids if input_token_ids is not None else [11, 22, 33]
    top1 = target_top1 if target_top1 is not None else [22, 33, 44]
    drafts = draft_tokens if draft_tokens is not None else [22, 33]
    row = accepted if commit_row is None else commit_row
    visible = (
        visible_tokens
        if visible_tokens is not None
        else [
            *drafts[:accepted],
            top1[accepted] if accepted < len(top1) else 0,
        ]
    )
    layer: dict[str, object] = {
        "layer": 0,
        "conv": {
            "source_row": source,
            "destination": source if destination is None else destination,
        },
        "recurrent": {
            "source_row": source,
            "destination": source if destination is None else destination,
        },
    }
    if other_row is not None:
        layer["conv"]["other_row"] = {"index": 0, "bytes": other_row}
        layer["recurrent"]["other_row"] = {"index": 0, "bytes": other_row}
    record: dict[str, object] = {
        "accepted": int(accepted),
        "rows": int(rows),
        "commit_row": int(row),
        "commit_token": (
            int(tokens[row])
            if commit_token is None and row < len(tokens)
            else (0 if commit_token is None else int(commit_token))
        ),
        "commit_position": (
            int(start_position + row)
            if commit_position is None
            else int(commit_position)
        ),
        "start_position": int(start_position),
        "end_position": (
            int(start_position + row + 1)
            if end_position is None
            else int(end_position)
        ),
        "cursor_after": (
            int(start_position + row + 1)
            if cursor_after is None
            else int(cursor_after)
        ),
        "input_token_ids": [int(token) for token in tokens],
        "visible_tokens": [int(token) for token in visible],
        "draft_tokens": [int(token) for token in drafts],
        "target_top1": [int(token) for token in top1],
        "layers": [layer],
        "hidden_row": (
            None
            if hidden_source is None
            else {
                "source_row": hidden_source,
                "destination": (
                    hidden_source
                    if hidden_destination is None
                    else hidden_destination
                ),
            }
        ),
    }
    return record


def _arms(
    *,
    ar_logits: np.ndarray,
    native_logits: np.ndarray,
    accepted: int,
    ownership: dict[str, object] | None = None,
    native_state_digest: str = "ar_state",
    native_layer_digest: str = "layer",
    serial_state_digest: str = "ar_state",
    ar_state_digest: str = "ar_state",
) -> dict[str, dict[str, object]]:
    rows = int(ar_logits.shape[0])
    ar_top1 = [int(value) for value in np.argmax(ar_logits, axis=1)]
    native_top1 = [int(value) for value in np.argmax(native_logits, axis=1)]
    visible = [*native_top1[:accepted], native_top1[accepted]]
    position = 100 + accepted + 1
    return {
        "ar": {
            "row_top1": ar_top1,
            "row_logits": ar_logits,
            "state": _fingerprint(
                label="ar",
                position=100 + rows,
                current_prev=ar_top1[-1],
                digest=ar_state_digest,
            ),
        },
        "native": {
            "row_top1": native_top1,
            "row_logits": native_logits,
            "accepted": accepted,
            "visible": visible,
            "verifier_drafts": native_top1[:accepted],
            "commit": {
                "accepted": accepted,
                "commit_row": accepted,
                "commit_token": int(native_top1[0]),
                "commit_position": 100 + accepted,
                "end_position": position,
                "cursor_after": position,
            },
            "logits_digest": "logits",
            "ownership": (
                _exact_ownership() if ownership is None else ownership
            ),
            "state": _fingerprint(
                label="native",
                position=position,
                current_prev=visible[-1],
                digest=native_state_digest,
                layer_digest=native_layer_digest,
            ),
            "state_digest": "state",
        },
        "ar_control": {
            "state": _fingerprint(
                label="ar_control",
                position=position,
                current_prev=visible[-1],
                digest=ar_state_digest,
            )
        },
        "serial": {
            "row_top1": ar_top1,
            "row_logits": None,
            "state": _fingerprint(
                label="serial",
                position=100 + rows,
                current_prev=ar_top1[-1],
                digest=serial_state_digest,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Lane selection
# ---------------------------------------------------------------------------


def test_parser_takes_a_record_and_output_index_not_a_cycle_trace() -> None:
    args = build_arg_parser().parse_args(
        [
            "--record",
            "/tmp/mtp-repair/moe-hostproposal-native.json",
            "--output-index",
            "35",
        ]
    )

    assert args.output_index == 35
    assert args.draft_budget == 2
    assert args.lane == "auto"
    assert args.repeat == 1
    assert args.packed_c1_target is False
    assert args.native_bulk_attention_mode == "native"
    assert not hasattr(args, "moe_c1")


def test_lane_auto_keeps_a_lone_request_on_the_singleton_native_cycle() -> None:
    """Serving capacity 4 does not move one speculative request to packed."""

    lane = resolve_lane(
        requested="auto",
        request_count=1,
        packed_c1_target=False,
        max_active_requests=4,
    )

    assert lane["lane"] == LANE_SINGLETON
    assert lane["reproduced_by_this_diagnostic"] is True
    assert lane["max_active_requests"] == 4
    assert lane["launch"].endswith("verify_target_block_native_cycle")
    assert "max-active-requests 4" in lane["reason"]


def test_lane_packed_needs_a_packed_group_or_the_explicit_evidence_row() -> None:
    with pytest.raises(ProbeError):
        resolve_lane(
            requested="packed",
            request_count=1,
            packed_c1_target=False,
            max_active_requests=4,
        )

    lane = resolve_lane(
        requested="packed",
        request_count=1,
        packed_c1_target=True,
        max_active_requests=4,
    )
    assert lane["lane"] == LANE_PACKED
    assert lane["reproduced_by_this_diagnostic"] is False


def test_lane_singleton_rejects_the_packed_c1_evidence_row() -> None:
    with pytest.raises(ProbeError):
        resolve_lane(
            requested="singleton",
            request_count=1,
            packed_c1_target=True,
            max_active_requests=1,
        )


def test_require_reproducible_lane_refuses_to_relabel_a_packed_run() -> None:
    lane = resolve_lane(
        requested="packed",
        request_count=1,
        packed_c1_target=True,
        max_active_requests=1,
    )

    with pytest.raises(ProbeError) as error:
        require_reproducible_lane(lane)
    assert "verify_target_blocks_batch" in str(error.value)

    require_reproducible_lane(
        resolve_lane(requested="auto", request_count=1, max_active_requests=4)
    )


# ---------------------------------------------------------------------------
# Logit comparison
# ---------------------------------------------------------------------------


def test_logits_agreement_reports_identical_matrices_as_exact() -> None:
    logits = _logits(rows=3, vocab=11, seed=1)

    agreement = logits_agreement(logits, logits)

    assert agreement["compared_rows"] == 3
    assert agreement["top1_mismatch_rows"] == []
    assert agreement["top1_agreement"] == 1.0
    assert agreement["max_abs_logit_delta"] == 0.0
    assert agreement["mean_kl_nats"] == 0.0
    assert agreement["gating"] is False
    assert agreement["metric_source"].endswith("compare_logits")


def test_logits_agreement_finds_the_flipped_row_and_its_delta() -> None:
    reference = _logits(rows=2, vocab=9, seed=2)
    candidate = _flip_row(reference, 1)

    agreement = logits_agreement(reference, candidate)

    assert agreement["top1_mismatch_rows"] == [1]
    assert agreement["top1_agreement"] == 0.5
    assert agreement["max_abs_logit_delta"] > 0.0
    assert agreement["mean_kl_nats"] > 0.0
    diagnostics = agreement["shared_metrics"]["diagnostics"]
    assert diagnostics["gating"] is False
    assert diagnostics["top1_flips"]["flips"] == 1


def test_logits_agreement_restricts_to_the_accepted_prefix_rows() -> None:
    reference = _logits(rows=3, vocab=7, seed=3)
    candidate = _flip_row(reference, 2)

    agreement = logits_agreement(reference, candidate, rows=2)

    assert agreement["compared_rows"] == 2
    assert agreement["top1_mismatch_rows"] == []
    assert logits_agreement(reference, candidate)["top1_mismatch_rows"] == [2]


def test_top1_only_agreement_handles_the_serial_control() -> None:
    agreement = _top1_only_agreement([4, 5, 6], [4, 9, 6], rows=None)

    assert agreement["compared_rows"] == 3
    assert agreement["top1_mismatch_rows"] == [1]
    assert agreement["logits_available"] is False
    assert agreement["gating"] is False


# ---------------------------------------------------------------------------
# Commit provenance
# ---------------------------------------------------------------------------


def test_commit_ownership_is_exact_for_the_selected_row_and_cursor() -> None:
    check = commit_ownership_check(_commit_record(accepted=1))

    assert check["status"] == "exact"
    assert check["selected_row"] == 1
    assert check["expected_end_position"] == 102
    assert check["mismatches"] == []
    assert check["contracts"]["strict_state_copy"]["status"] == "exact"
    assert check["contracts"]["ownership"]["status"] == "exact"


def test_commit_ownership_proves_a_wrong_source_row_copy() -> None:
    check = commit_ownership_check(
        _commit_record(
            accepted=1,
            source=b"selected-row",
            destination=b"row-zero----",
            other_row=b"row-zero----",
        )
    )

    assert check["status"] == "broken"
    assert check["contracts"]["ownership"]["status"] == "exact"
    copy = check["contracts"]["strict_state_copy"]
    assert copy["status"] == "broken"
    components = {row["component"] for row in copy["mismatches"]}
    assert components == {"linear_state_conv", "linear_state_recurrent"}
    assert all(row["reason"] == "bytes" for row in copy["mismatches"])
    assert all(
        row["destination_matches_other_row"] == 0 for row in copy["mismatches"]
    )


def test_commit_ownership_proves_a_commit_row_that_is_not_the_accepted_row() -> None:
    check = commit_ownership_check(
        _commit_record(
            accepted=1,
            commit_row=0,
            commit_position=101,
            end_position=101,
        )
    )

    assert check["status"] == "broken"
    ownership = check["contracts"]["ownership"]
    assert ownership["status"] == "broken"
    components = {row["component"] for row in ownership["mismatches"]}
    assert {"accept.commit_row", "cursor.commit_position"} <= components


def test_commit_ownership_proves_a_commit_token_that_is_not_the_selected_row() -> None:
    check = commit_ownership_check(
        _commit_record(accepted=1, commit_token=33)
    )

    assert check["status"] == "broken"
    components = {
        row["component"]
        for row in check["contracts"]["ownership"]["mismatches"]
    }
    assert components == {"accept.commit_token"}


def test_commit_ownership_proves_a_cursor_that_did_not_advance_by_the_accept() -> None:
    check = commit_ownership_check(
        _commit_record(accepted=1, end_position=101, cursor_after=101)
    )

    assert check["status"] == "broken"
    components = {
        row["component"]
        for row in check["contracts"]["ownership"]["mismatches"]
    }
    assert {"cursor.end_position", "cursor.session_position"} <= components


def test_commit_ownership_proves_committed_tokens_that_are_not_the_accept() -> None:
    check = commit_ownership_check(
        _commit_record(accepted=1, visible_tokens=[22, 99])
    )

    assert check["status"] == "broken"
    assert [
        row["component"]
        for row in check["contracts"]["ownership"]["mismatches"]
    ] == ["accept.committed_tokens"]


def test_commit_ownership_checks_the_hidden_tap_copy() -> None:
    exact = commit_ownership_check(
        _commit_record(accepted=1, hidden_source=b"hidden-row")
    )
    assert exact["status"] == "exact"
    assert exact["hidden_row_checked"] is True

    broken = commit_ownership_check(
        _commit_record(
            accepted=1,
            hidden_source=b"hidden-row",
            hidden_destination=b"hidden-ROW",
        )
    )
    assert broken["status"] == "broken"
    components = {
        row["component"]
        for row in broken["contracts"]["strict_state_copy"]["mismatches"]
    }
    assert components == {"hidden_seed"}


def test_commit_ownership_checks_the_hidden_tap_without_linear_layers() -> None:
    """A dense artifact has no Conv/GDN rows; the hidden tap still proves a copy."""

    record = _commit_record(accepted=0, hidden_source=b"hidden-row")
    record["layers"] = []

    exact = commit_ownership_check(record)
    assert exact["status"] == "exact"
    assert exact["layers_checked"] == 0
    assert exact["hidden_row_checked"] is True

    record["hidden_row"]["destination"] = b"hidden-ROW"
    broken = commit_ownership_check(record)
    assert broken["status"] == "broken"
    assert broken["contracts"]["strict_state_copy"]["status"] == "broken"


def test_commit_ownership_is_not_checked_without_captured_rows() -> None:
    record = _commit_record(accepted=0)
    record["layers"] = []
    record["hidden_row"] = None

    check = commit_ownership_check(record)

    assert check["status"] == "not_checked"
    assert check["layers_checked"] == 0
    assert check["hidden_row_checked"] is False
    assert check["contracts"]["strict_state_copy"]["status"] == "not_checked"
    assert check["contracts"]["ownership"]["status"] == "exact"


def test_commit_ownership_proves_a_size_mismatch() -> None:
    check = commit_ownership_check(
        _commit_record(accepted=0, source=b"abcd", destination=b"abc")
    )

    assert check["status"] == "broken"
    assert check["mismatches"][0]["reason"] == "size"


def test_commit_ownership_rejects_an_accept_outside_the_verifier_rows() -> None:
    with pytest.raises(ProbeError):
        commit_ownership_check(_commit_record(accepted=3, rows=3))


# ---------------------------------------------------------------------------
# Measured state drift
# ---------------------------------------------------------------------------


def _fp32_bytes(values: list[float]) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(values, dtype=np.float32).view(np.uint8)
    ).reshape(-1)


def test_state_numeric_delta_reports_identical_states() -> None:
    native = {0: {"conv": _fp32_bytes([1.0, 2.0]), "recurrent": _fp32_bytes([3.0])}}
    ar = {0: {"conv": _fp32_bytes([1.0, 2.0]), "recurrent": _fp32_bytes([3.0])}}

    delta = state_numeric_delta(native, ar)

    assert delta["checked"] is True
    assert delta["all_identical"] is True
    assert delta["layers"][0]["conv"]["max_abs_delta"] == 0.0


def test_state_numeric_delta_measures_the_changed_elements() -> None:
    native = {0: {"conv": _fp32_bytes([1.0, 2.5]), "recurrent": _fp32_bytes([3.0])}}
    ar = {0: {"conv": _fp32_bytes([1.0, 2.0]), "recurrent": _fp32_bytes([3.0])}}

    delta = state_numeric_delta(native, ar)

    assert delta["all_identical"] is False
    conv = delta["layers"][0]["conv"]
    assert conv["bitwise_equal"] is False
    assert conv["changed_elements"] == 1
    assert conv["max_abs_delta"] == pytest.approx(0.5)
    assert conv["native_summary"]["max"] == pytest.approx(2.5)
    assert delta["layers"][0]["recurrent"]["bitwise_equal"] is True


def test_state_numeric_delta_reports_missing_capture_instead_of_guessing() -> None:
    delta = state_numeric_delta({}, {})

    assert delta["checked"] is False


# ---------------------------------------------------------------------------
# Repeatability
# ---------------------------------------------------------------------------


def _repeat_run(*, top1: int = 7, logits: str = "logits", state: str = "state") -> dict[str, object]:
    return {
        "row_top1": [top1, 8],
        "logits_digest": logits,
        "state_digest": state,
        "visible": [top1, 8],
        "commit": {
            "accepted": 1,
            "commit_row": 1,
            "commit_token": 8,
            "commit_position": 100,
            "end_position": 101,
            "cursor_after": 101,
        },
    }


def test_repeatability_is_not_checked_for_a_single_run() -> None:
    check = repeatability_check([_repeat_run()])

    assert check["checked"] is False
    assert check["identical"] is None


def test_repeatability_compares_row_top1_logits_state_and_commit() -> None:
    assert repeatability_check([_repeat_run(), _repeat_run()])["identical"] is True

    check = repeatability_check(
        [_repeat_run(), _repeat_run(logits="other-logits")]
    )
    assert check["checked"] is True
    assert check["identical"] is False
    assert [row["field"] for row in check["differences"]] == ["logits_digest"]


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_top1_difference_alone_is_measured_drift_not_a_commit_defect() -> None:
    """The corrected semantics: equal/unequal top-1 is diagnostic, not decisive."""

    reference = _logits(rows=3, vocab=13, seed=5)
    candidate = _flip_row(reference, 1)

    attribution = attribute_divergence(
        _arms(ar_logits=reference, native_logits=candidate, accepted=1)
    )

    assert attribution["verdict"] == VERDICT_NUMERICAL_DRIFT
    assert attribution["verdict_class"] == "numerical_diagnostic"
    assert attribution["verdict_is_decisive"] is False
    assert attribution["verdict"] != VERDICT_OWNERSHIP_DEFECT
    assert attribution["numerical"]["gating"] is False
    assert attribution["numerical"]["top1_divergence_at_consumed_rows"] == [1]
    assert attribution["accepted_decision"]["correction_token_matches_ar"] is False
    assert (
        attribution["numerical"]["row_arithmetic_accepted_prefix"][
            "max_abs_logit_delta"
        ]
        > 0.0
    )


def test_clean_cycle_needs_identical_logits_and_state() -> None:
    logits = _logits(rows=3, vocab=13, seed=4)

    attribution = attribute_divergence(
        _arms(ar_logits=logits, native_logits=logits, accepted=1)
    )

    assert attribution["verdict"] == VERDICT_CLEAN
    assert attribution["verdict_is_decisive"] is False
    assert attribution["consumed_rows"] == 2
    assert attribution["accepted_state"]["ar_control_vs_native"] == []


def test_unequal_state_with_exact_ownership_is_measured_drift() -> None:
    logits = _logits(rows=3, vocab=13, seed=7)

    attribution = attribute_divergence(
        _arms(
            ar_logits=logits,
            native_logits=logits,
            accepted=1,
            native_layer_digest="drifted_conv_row",
        )
    )

    assert attribution["verdict"] == VERDICT_NUMERICAL_DRIFT
    assert attribution["verdict_class"] == "numerical_diagnostic"
    assert attribution["verdict_is_decisive"] is False
    mismatches = attribution["accepted_state"]["ar_control_vs_native"]
    assert any(row["component"] == "linear_state_conv" for row in mismatches)


def test_a_proven_wrong_copy_is_a_decisive_commit_defect() -> None:
    logits = _logits(rows=3, vocab=13, seed=8)
    broken = commit_ownership_check(
        _commit_record(
            accepted=1,
            source=b"selected-row",
            destination=b"row-zero----",
            other_row=b"row-zero----",
        )
    )

    attribution = attribute_divergence(
        _arms(
            ar_logits=logits,
            native_logits=logits,
            accepted=1,
            ownership=broken,
        )
    )

    assert attribution["verdict"] == VERDICT_OWNERSHIP_DEFECT
    assert attribution["verdict_class"] == "ownership_defect"
    assert attribution["verdict_is_decisive"] is True
    assert (
        attribution["contracts"]["strict_state_copy"]["status"] == "broken"
    )


def test_ownership_defect_outranks_numerical_drift() -> None:
    reference = _logits(rows=3, vocab=13, seed=9)
    candidate = _flip_row(reference, 1)
    broken = commit_ownership_check(
        _commit_record(accepted=1, end_position=101, cursor_after=101)
    )

    attribution = attribute_divergence(
        _arms(
            ar_logits=reference,
            native_logits=candidate,
            accepted=1,
            ownership=broken,
        )
    )

    assert attribution["verdict"] == VERDICT_OWNERSHIP_DEFECT
    assert attribution["numerical"]["gating"] is False


def test_repeatability_mismatch_outranks_ownership_and_drift() -> None:
    logits = _logits(rows=3, vocab=13, seed=10)

    attribution = attribute_divergence(
        _arms(ar_logits=logits, native_logits=logits, accepted=1),
        repeatability=repeatability_check(
            [_repeat_run(), _repeat_run(state="other-state")]
        ),
    )

    assert attribution["verdict"] == VERDICT_REPEATABILITY
    assert attribution["verdict_class"] == "repeatability_defect"
    assert attribution["verdict_is_decisive"] is True
    assert attribution["contracts"]["repeatability"]["identical"] is False


def test_state_difference_without_a_captured_commit_stays_inconclusive() -> None:
    logits = _logits(rows=3, vocab=13, seed=11)

    attribution = attribute_divergence(
        _arms(
            ar_logits=logits,
            native_logits=logits,
            accepted=1,
            ownership=_unverified_ownership(),
            native_layer_digest="drifted_conv_row",
        )
    )

    assert attribution["verdict"] == VERDICT_OWNERSHIP_UNVERIFIED
    assert attribution["verdict_class"] == "inconclusive"
    assert attribution["verdict_is_decisive"] is False


def test_serial_control_mismatch_is_inconclusive() -> None:
    logits = _logits(rows=3, vocab=13, seed=12)

    attribution = attribute_divergence(
        _arms(
            ar_logits=logits,
            native_logits=logits,
            accepted=1,
            serial_state_digest="serial_drifted",
        )
    )

    assert attribution["verdict"] == VERDICT_SERIAL_CONTROL
    assert attribution["accepted_state"]["ar_vs_serial"]


def test_attribution_records_the_lane_it_resolved() -> None:
    logits = _logits(rows=3, vocab=13, seed=13)
    lane = resolve_lane(
        requested="auto",
        request_count=1,
        packed_c1_target=False,
        max_active_requests=4,
    )

    attribution = attribute_divergence(
        _arms(ar_logits=logits, native_logits=logits, accepted=1),
        lane=lane,
    )

    assert attribution["lane"]["lane"] == LANE_SINGLETON
    assert attribution["lane"]["max_active_requests"] == 4


def test_attribute_divergence_requires_the_three_scoring_arms() -> None:
    with pytest.raises(ProbeError):
        attribute_divergence({"native": {}})


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _moe_record() -> dict[str, object]:
    return {
        "rows": [
            {
                "label": "pair0_False",
                "status": 200,
                "request": {
                    "messages": [{"role": "user", "content": "word word"}],
                    "speculative_mtp": False,
                },
                "ids": [5, 6, 7],
                "mtp": {"used": False},
                "response": {"usage": {"prompt_tokens": 512}},
            },
            {
                "label": "pair0_True",
                "status": 200,
                "request": {
                    "messages": [{"role": "user", "content": "word word"}],
                    "speculative_mtp": True,
                },
                "ids": [5, 6, 9],
                "mtp": {"used": True},
                "response": {"usage": {"prompt_tokens": 512}},
            },
        ]
    }


def _openai_record() -> dict[str, object]:
    return {
        "rows": [
            {
                "label": "ar_128",
                "status": 200,
                "request": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "speculative_mtp": False,
                },
                "response": {
                    "usage": {"prompt_tokens": 22},
                    "hipengine": {
                        "token_accounting": {
                            "choice_generated_token_ids": [[1, 2, 3, 4]]
                        }
                    },
                },
            }
        ]
    }


def test_record_reference_row_reads_row_level_ids() -> None:
    parsed = _record_reference_row(_moe_record(), row_label="pair0_False")

    assert parsed["reference_ids"] == [5, 6, 7]
    assert parsed["prompt"] == "word word"
    assert parsed["expected_prompt_tokens"] == 512


def test_record_reference_row_prefers_the_row_that_did_not_use_mtp() -> None:
    parsed = _record_reference_row(_moe_record(), row_label=None)

    assert parsed["label"] == "pair0_False"
    assert parsed["reference_ids"] == [5, 6, 7]


def test_record_reference_row_reads_the_openai_response_shape() -> None:
    parsed = _record_reference_row(_openai_record(), row_label=None)

    assert parsed["label"] == "ar_128"
    assert parsed["reference_ids"] == [1, 2, 3, 4]
    assert parsed["expected_prompt_tokens"] == 22


def test_record_reference_row_rejects_a_multi_message_prompt() -> None:
    record = _moe_record()
    record["rows"][0]["request"]["messages"] = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "word word"},
    ]

    with pytest.raises(SystemExit):
        _record_reference_row(record, row_label="pair0_False")
