"""Unit coverage for per-request MTP-versus-AR output attribution.

The response layer derives autoregressive output as
``completion_tokens - mtp2_mtp_output_tokens``, so these counters are the only
thing standing between "the route said MTP" and "these tokens came from MTP".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.speculative.accounting import (
    PROVIDER_ABSENT,
    PROVIDER_DECLINED,
    PROVIDER_PROMPT_BUFFERED,
    PROVIDER_READY,
    accounting_timing_fields,
    record_autoregressive_step,
    record_provider_readiness,
    record_speculative_outputs,
    record_speculative_plan,
    span_accounting,
    speculative_output_accounting,
)


def _intent_row(budget: int = 3) -> SimpleNamespace:
    return SimpleNamespace(mtp2_requested_budget=budget, mtp2_candidate_budget=budget)


def test_accounting_is_absent_for_a_request_without_speculative_intent() -> None:
    row = SimpleNamespace(mtp2_cycles=0, mtp2_candidate_budget=0)

    assert speculative_output_accounting(row) is None
    assert accounting_timing_fields(None) == {}
    assert accounting_timing_fields(speculative_output_accounting(row)) == {}


def test_accounting_is_absent_for_plain_autoregressive_decoding() -> None:
    """AR-only activity on a non-speculative row must not invent a split."""

    row = SimpleNamespace(mtp2_cycles=0, mtp2_candidate_budget=0)
    record_autoregressive_step(row, plan_reason="policy_selected_ar")

    assert speculative_output_accounting(row) is None


def test_accounting_splits_speculative_and_autoregressive_output() -> None:
    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=2,
        visible_count=4,
        output_position=1,
        plan_reason="speculative_qualified",
    )
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=1,
        visible_count=3,
        output_position=5,
        plan_reason="speculative_qualified",
    )
    record_speculative_outputs(
        row,
        candidate_count=0,
        accepted_count=0,
        visible_count=1,
        output_position=8,
        plan_reason="target_graph_context_bucket_miss",
    )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["requested_budget"] == 3
    assert accounting["candidate_budget"] == 3
    assert accounting["cycles"] == 3
    assert accounting["generated_draft_tokens"] == 6
    assert accounting["accepted_draft_tokens"] == 3
    assert accounting["selected_depth_histogram"] == {"0": 1, "3": 2}
    assert accounting["mtp_output_tokens"] == 7
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] == 8
    assert accounting["ar_step_reason_counts"] == {
        "target_graph_context_bucket_miss": 1
    }
    assert accounting["prompt_fallback_reason"] is None
    assert accounting_timing_fields(accounting) == {
        "mtp_cycles_count": 3.0,
        "mtp_generated_draft_tokens": 6.0,
        "mtp_accepted_draft_tokens": 3.0,
        "mtp_visible_output_tokens": 7.0,
        "mtp_ar_output_tokens": 1.0,
        "mtp_accept_per_draft": 0.5,
    }


def test_accounting_publishes_the_four_refusal_facts_separately() -> None:
    """One row can be route-qualified, plan-refused, and provider-absent at once.

    Every fact here is deliberately a *different* value from the others, so a
    single published field cannot satisfy two assertions: the row's own
    activation refusal, its provider readiness, the width of the group it was
    planned in, and that group's autoregressive decision must each come from
    their own attribute.
    """

    row = _intent_row()
    row.mtp2_prompt_fallback_reason = "prefix_reuse_k0"
    row.mtp2_provider_readiness = PROVIDER_ABSENT
    row.mtp2_provider_decline_reason = "provider_state_absent"
    row.mtp2_plan_group_rows = 4
    row.mtp2_plan_ar_only = True
    row.mtp2_plan_reason = "no_provider"
    record_autoregressive_step(row, plan_reason="no_provider")

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["activation_reason"] == "prefix_reuse_k0"
    assert accounting["provider_readiness"] == PROVIDER_ABSENT
    assert accounting["provider_decline_reason"] == "provider_state_absent"
    assert accounting["plan_group_rows"] == 4
    assert accounting["plan_ar_only"] is True
    assert accounting["plan_reason"] == "no_provider"
    # The event fold is a different quantity from the activation reason: this row
    # emitted one autoregressive step for the same planner reason, and a reader
    # must be able to see both without one standing in for the other.
    assert accounting["ar_step_reason_counts"] == {"no_provider": 1}
    assert accounting["prompt_fallback_reason"] == "prefix_reuse_k0"


def test_accounting_reports_an_engaged_row_without_a_refusal_reason() -> None:
    """An engaged row keeps ``no_provider`` events and a ready provider at once."""

    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=2,
        visible_count=3,
        output_position=0,
        plan_reason="speculative_qualified",
    )
    record_autoregressive_step(row, plan_reason="no_provider")
    record_provider_readiness(row, readiness=PROVIDER_READY)
    record_speculative_plan(
        row, group_rows=1, ar_only=False, plan_reason="speculative_qualified"
    )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["cycles"] == 1
    assert accounting["activation_reason"] is None
    assert accounting["provider_readiness"] == PROVIDER_READY
    assert accounting["provider_decline_reason"] is None
    assert accounting["plan_group_rows"] == 1
    assert accounting["plan_ar_only"] is False
    assert accounting["plan_reason"] == "speculative_qualified"
    assert accounting["ar_step_reason_counts"] == {"no_provider": 1}


def test_record_provider_readiness_clears_a_stale_decline_reason() -> None:
    """A row that acquired a provider must not keep reporting why it had not."""

    row = _intent_row()
    record_provider_readiness(
        row, readiness=PROVIDER_DECLINED, decline_reason="first_token_pending"
    )
    assert row.mtp2_provider_readiness == PROVIDER_DECLINED
    assert row.mtp2_provider_decline_reason == "first_token_pending"

    record_provider_readiness(row, readiness=PROVIDER_PROMPT_BUFFERED)
    assert row.mtp2_provider_readiness == PROVIDER_PROMPT_BUFFERED
    assert row.mtp2_provider_decline_reason is None

    with pytest.raises(ValueError):
        record_provider_readiness(row, readiness="probably")


def test_record_speculative_plan_keeps_group_rows_and_decision_apart() -> None:
    """The group's width and its AR decision are independent published facts."""

    wide = _intent_row()
    record_speculative_plan(
        wide, group_rows=4, ar_only=False, plan_reason="speculative_qualified"
    )
    assert wide.mtp2_plan_group_rows == 4
    assert wide.mtp2_plan_ar_only is False
    assert wide.mtp2_plan_reason == "speculative_qualified"

    singleton = _intent_row()
    record_speculative_plan(
        singleton, group_rows=1, ar_only=True, plan_reason="no_provider"
    )
    assert singleton.mtp2_plan_group_rows == 1
    assert singleton.mtp2_plan_ar_only is True
    assert singleton.mtp2_plan_reason == "no_provider"


def test_accounting_publishes_the_cycle_phase_split_with_its_divisor() -> None:
    """The measured phase windows are published, and are not a per-cycle cost.

    These are host-wall windows: a window that ends by reading a result
    includes the wait for it. Publishing the sums beside ``cycles`` is what
    makes a per-cycle cost computable without another profiler run.
    """

    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=2,
        visible_count=3,
        output_position=0,
        plan_reason="speculative_qualified",
    )
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=1,
        visible_count=2,
        output_position=3,
        plan_reason="speculative_qualified",
    )
    row.mtp2_proposal_ms = 1.5
    row.mtp2_target_ms = 30.0
    row.mtp2_provider_update_ms = 2.25
    row.mtp2_accept_ms = 4.0
    row.mtp2_target_readback_ms = 0.5
    row.mtp2_accept_upload_ms = 0.25
    row.mtp2_accept_tail_ms = 0.0
    row.mtp2_accept_enqueue_ms = 0.75
    row.mtp2_selected_commit_ms = 0.5
    row.mtp2_candidate_readback_ms = 0.25

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["cycles"] == 2
    assert accounting["cycle_timing_ms"] == {
        "proposal": 1.5,
        "target": 30.0,
        "provider_update": 2.25,
        "accept": 4.0,
        "candidate_readback": 0.25,
        "target_readback": 0.5,
        "accept_upload": 0.25,
        "accept_tail": 0.0,
        "accept_enqueue": 0.75,
        "selected_commit": 0.5,
    }
    # A row that never ran a cycle still reports zeroed windows rather than
    # omitting the block, so a reader never has to guess whether the phase was
    # unmeasured or merely absent from the response.
    idle = _intent_row()
    idle_accounting = speculative_output_accounting(idle)
    assert idle_accounting is not None
    assert idle_accounting["cycles"] == 0
    assert set(idle_accounting["cycle_timing_ms"].values()) == {0.0}


def test_accounting_keeps_the_first_fallback_position_only() -> None:
    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=2,
        accepted_count=1,
        visible_count=2,
        output_position=1,
        plan_reason="speculative_qualified",
    )
    record_speculative_outputs(
        row,
        candidate_count=0,
        accepted_count=0,
        visible_count=1,
        output_position=3,
        plan_reason="target_graph_output_room_miss",
    )
    record_speculative_outputs(
        row,
        candidate_count=0,
        accepted_count=0,
        visible_count=1,
        output_position=4,
        plan_reason="target_graph_output_room_miss",
    )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["first_fallback_position"] == 3
    assert accounting["ar_output_tokens_in_cycles"] == 2
    assert accounting["ar_step_reason_counts"] == {"target_graph_output_room_miss": 2}


def test_accounting_locates_the_fallback_of_an_autoregressive_only_cycle() -> None:
    """After the plan turns AR-only, each step reports its position and token."""

    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=2,
        visible_count=3,
        output_position=1,
        plan_reason="speculative_qualified",
    )
    # The plan turns AR-only above the context window: the resident owner runs
    # one plain decode step per cycle and prepare_k0 reports each of them.
    for position in (4, 5, 6):
        record_autoregressive_step(
            row,
            plan_reason="target_graph_context_bucket_miss",
            output_position=position,
        )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["mtp_output_tokens"] == 3
    assert accounting["ar_output_tokens_in_cycles"] == 3
    assert accounting["first_fallback_position"] == 4
    assert accounting["ar_step_reason_counts"] == {
        "target_graph_context_bucket_miss": 3
    }


def test_accounting_ignores_steps_that_precede_speculative_output() -> None:
    """An AR step before the first speculative cycle is not a fallback."""

    row = _intent_row()
    record_autoregressive_step(
        row,
        plan_reason="no_provider",
        output_position=1,
    )
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=3,
        visible_count=4,
        output_position=2,
        plan_reason="speculative_qualified",
    )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["mtp_output_tokens"] == 4
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] is None


def test_accounting_does_not_attribute_cycle_tokens_twice() -> None:
    """A depth-zero cycle commit owns its token once, not once per recorder."""

    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=2,
        accepted_count=2,
        visible_count=3,
        output_position=1,
        plan_reason="speculative_qualified",
    )
    record_speculative_outputs(
        row,
        candidate_count=0,
        accepted_count=0,
        visible_count=1,
        output_position=4,
        plan_reason="target_graph_output_room_miss",
    )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert sum(accounting["ar_step_reason_counts"].values()) == 1


def test_accounting_reports_a_refusal_without_publishing_timing_mirrors() -> None:
    """A refused activation stays AR in ``timing`` and reports why in its block."""

    row = _intent_row()
    row.mtp2_candidate_budget = 0
    row.mtp2_prompt_fallback_reason = "target_context_k0"
    record_autoregressive_step(row, plan_reason="no_provider")

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["cycles"] == 0
    assert accounting["mtp_output_tokens"] == 0
    assert accounting["prompt_fallback_reason"] == "target_context_k0"
    assert accounting["ar_step_reason_counts"] == {"no_provider": 1}
    # Zero committed cycles must not look like realized MTP to the response.
    assert accounting_timing_fields(accounting) == {}


def test_accounting_counts_failure_reason_pairs_by_category() -> None:
    row = _intent_row()
    row.mtp2_recoverable_failures = 2
    row.mtp2_failure_reasons = [
        "precommit_failure_ar_fallback",
        "RuntimeError:target cursor moved",
        "precommit_failure_ar_fallback",
        "RuntimeError:target cursor moved again",
    ]

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["recoverable_failures"] == 2
    assert accounting["failure_reason_counts"] == {
        "precommit_failure_ar_fallback": 2
    }


def test_accounting_records_a_speculative_cycle_without_visible_tokens() -> None:
    """A zero-visible record must not create a fallback position or a count."""

    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=0,
        accepted_count=0,
        visible_count=0,
        output_position=0,
        plan_reason="target_physical_bucket_miss",
    )

    accounting = speculative_output_accounting(row)

    assert accounting is not None
    assert accounting["mtp_output_tokens"] == 0
    assert accounting["ar_output_tokens_in_cycles"] == 0
    assert accounting["first_fallback_position"] is None
    assert accounting["ar_step_reason_counts"] == {}


def test_accounting_records_committed_output_spans_only_when_enabled(
    monkeypatch,
) -> None:
    """Spans are diagnostic evidence, and their absence must be distinguishable.

    A response read without diagnostic recording must be unchanged, so the span
    block is added only when the switch is on; an enabled-but-empty span list
    still reports its (failed) proof rather than looking like production.
    """

    monkeypatch.delenv("HIPENGINE_MTP2_OUTPUT_SPANS", raising=False)
    row = _intent_row()
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=2,
        visible_count=3,
        output_position=1,
        plan_reason="speculative_qualified",
    )
    record_autoregressive_step(
        row,
        plan_reason="target_graph_context_bucket_miss",
        output_position=4,
    )

    assert "output_spans" not in speculative_output_accounting(row)

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    enabled = _intent_row()
    record_speculative_outputs(
        enabled,
        candidate_count=3,
        accepted_count=2,
        visible_count=3,
        output_position=1,
        plan_reason="speculative_qualified",
    )
    record_speculative_outputs(
        enabled,
        candidate_count=0,
        accepted_count=0,
        visible_count=1,
        output_position=4,
        plan_reason="target_graph_context_bucket_miss",
    )
    record_autoregressive_step(
        enabled,
        plan_reason="no_provider",
        output_position=5,
    )
    # A mixed-plan K0 row attributes its tokens through the cycle commit, so the
    # step record must not add a second span for them.
    record_autoregressive_step(
        enabled,
        plan_reason="target_graph_context_bucket_miss",
        emitted_tokens=0,
    )

    accounting = speculative_output_accounting(enabled)

    assert accounting is not None
    assert accounting["output_spans"] == [
        {
            "mode": "mtp",
            "reason": "speculative_qualified",
            "position": 1,
            "tokens": 3,
        },
        {
            "mode": "ar",
            "reason": "target_graph_context_bucket_miss",
            "position": 4,
            "tokens": 1,
        },
        {"mode": "ar", "reason": "no_provider", "position": 5, "tokens": 1},
    ]
    assert accounting["mtp_output_tokens"] == 3
    assert accounting["ar_output_tokens_in_cycles"] == 2


def test_span_accounting_tiles_the_output_and_matches_both_counters() -> None:
    spans = [
        {"mode": "mtp", "reason": "speculative_qualified", "position": 1, "tokens": 3},
        {"mode": "mtp", "reason": "speculative_qualified", "position": 4, "tokens": 2},
        {
            "mode": "ar",
            "reason": "target_graph_context_bucket_miss",
            "position": 6,
            "tokens": 1,
        },
    ]

    proof = span_accounting(
        spans,
        completion_tokens=7,
        mtp_output_tokens=5,
        ar_output_tokens=2,
        ar_output_tokens_in_cycles=1,
    )

    assert proof["spans"] == 3
    assert proof["tokens"] == 6
    assert proof["mtp_tokens"] == 5
    assert proof["ar_tokens"] == 1
    assert proof["ar_tokens_by_reason"] == {"target_graph_context_bucket_miss": 1}
    assert proof["first_position"] == 1
    assert proof["last_end"] == 7
    assert proof["contiguous"] is True
    # One token is unspanned and it is exactly the autoregressive token emitted
    # outside the plan, which is what makes the split attributable.
    assert proof["unspanned_tokens"] == 1
    assert proof["expected_unspanned_ar_tokens"] == 1
    assert proof["unspanned_tokens_match"] is True
    assert proof["reconciled"] is True
    assert proof["reconciled_reasons"] == []

    # A span starting at 0 is allowed: the prefill root token is counted as
    # autoregressive output but is not itself a committed span.
    rooted = span_accounting(
        [
            {"mode": "ar", "reason": "root", "position": 0, "tokens": 1},
            {"mode": "mtp", "reason": "speculative_qualified", "position": 1, "tokens": 2},
        ],
        completion_tokens=3,
        mtp_output_tokens=2,
        ar_output_tokens=1,
        ar_output_tokens_in_cycles=1,
    )
    assert rooted["unspanned_tokens"] == 0
    assert rooted["reconciled"] is True

    # A gap fails, and so does an overlap.
    gapped = span_accounting(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 2},
            {"mode": "mtp", "reason": "r", "position": 4, "tokens": 2},
        ],
        completion_tokens=6,
        mtp_output_tokens=4,
        ar_output_tokens=2,
        ar_output_tokens_in_cycles=0,
    )
    assert gapped["contiguous"] is False
    assert "spans_not_contiguous" in gapped["reconciled_reasons"]

    overlapped = span_accounting(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 3},
            {"mode": "mtp", "reason": "r", "position": 3, "tokens": 2},
        ],
        completion_tokens=5,
        mtp_output_tokens=5,
        ar_output_tokens=0,
        ar_output_tokens_in_cycles=0,
    )
    assert overlapped["contiguous"] is False
    assert "spans_not_contiguous" in overlapped["reconciled_reasons"]

    # Spans that tile the output but disagree with the reported split.
    mislabelled = span_accounting(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 1},
            {"mode": "ar", "reason": "r", "position": 2, "tokens": 2},
        ],
        completion_tokens=4,
        mtp_output_tokens=2,
        ar_output_tokens=2,
        ar_output_tokens_in_cycles=1,
    )
    assert mislabelled["unspanned_tokens"] == 1
    assert mislabelled["reconciled"] is False
    assert mislabelled["reconciled_reasons"] == [
        "span_mtp_tokens_do_not_match_mtp_output",
        "span_ar_tokens_do_not_match_in_cycle_ar_output",
    ]


def test_span_recording_never_breaks_a_row_that_cannot_hold_spans(
    monkeypatch,
) -> None:
    """A slots row without the span field must drop the span, not raise.

    The diagnostic switch is operator-set, so a row type that does not declare
    ``mtp2_output_spans`` must degrade to "no proof recorded" instead of failing
    the generation that enabled the flag.
    """

    class SlotsRow:
        __slots__ = (
            "mtp2_requested_budget",
            "mtp2_candidate_budget",
            "mtp2_cycles",
            "mtp2_candidate_counts",
            "mtp2_accepted_counts",
            "mtp2_mtp_output_tokens",
            "mtp2_ar_output_tokens",
            "mtp2_ar_step_output_tokens",
            "mtp2_first_fallback_position",
            "mtp2_ar_step_reasons",
        )

        def __init__(self) -> None:
            self.mtp2_requested_budget = 3
            self.mtp2_candidate_budget = 3
            self.mtp2_cycles = 0
            self.mtp2_candidate_counts = []
            self.mtp2_accepted_counts = []
            self.mtp2_mtp_output_tokens = 0
            self.mtp2_ar_output_tokens = 0
            self.mtp2_ar_step_output_tokens = 0
            self.mtp2_first_fallback_position = None
            self.mtp2_ar_step_reasons = {}

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    row = SlotsRow()
    record_speculative_outputs(
        row,
        candidate_count=3,
        accepted_count=2,
        visible_count=3,
        output_position=0,
        plan_reason="speculative_qualified",
    )
    record_autoregressive_step(row, plan_reason="no_provider", output_position=3)

    # The counters are unaffected; only the diagnostic proof is missing.
    accounting = speculative_output_accounting(row)
    assert accounting is not None
    assert accounting["mtp_output_tokens"] == 3
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["output_spans"] == []


def test_resident_loop_row_declares_the_diagnostic_span_field() -> None:
    """The shared accounting helper writes the span list onto the live row.

    ``_GGUFResidentLoopRow`` is a slots dataclass, so the field has to be
    declared there or the diagnostic switch raises mid-generation (which is
    exactly how this was found).
    """

    from dataclasses import fields

    from hipengine.generation.qwen35_gguf import _GGUFResidentLoopRow

    names = {entry.name for entry in fields(_GGUFResidentLoopRow)}
    assert "mtp2_output_spans" in names
    # Every attribute the accounting layer writes belongs in this list. A
    # missing declaration is not a missing fact: setattr on a slots dataclass
    # raises inside the engine loop, so the request fails with a 500 instead of
    # reporting the fact. That is how mtp2_provider_state_present was found.
    for name in (
        "mtp2_mtp_output_tokens",
        "mtp2_ar_output_tokens",
        "mtp2_ar_step_output_tokens",
        "mtp2_first_fallback_position",
        "mtp2_ar_step_reasons",
        "mtp2_cycles",
        "mtp2_candidate_counts",
        "mtp2_accepted_counts",
        "mtp2_provider_readiness",
        "mtp2_provider_decline_reason",
        "mtp2_provider_state_present",
        "mtp2_plan_group_rows",
        "mtp2_plan_ar_only",
        "mtp2_plan_reason",
        "mtp2_requested_budget",
        "mtp2_candidate_budget",
        "mtp2_prompt_fallback_reason",
    ):
        assert name in names


def test_provider_state_presence_is_reported_beside_a_declined_readiness() -> None:
    """A refused group says nothing about whether the row still holds state.

    Readiness folds both into ``declined``, and the lifecycle answers are
    opposite: a row refused by its realized width keeps the provider it may use
    again when its group narrows, while a row whose state is gone cannot
    speculate until something primes it again.
    """

    held = _intent_row()
    record_provider_readiness(
        held,
        readiness=PROVIDER_DECLINED,
        decline_reason="width 4 exceeds static group bound 1",
        state_present=True,
    )
    held_accounting = speculative_output_accounting(held)
    assert held_accounting is not None
    assert held_accounting["provider_readiness"] == "declined"
    assert held_accounting["provider_decline_reason"] == (
        "width 4 exceeds static group bound 1"
    )
    assert held_accounting["provider_state_present"] is True

    gone = _intent_row()
    record_provider_readiness(
        gone,
        readiness=PROVIDER_ABSENT,
        decline_reason="provider_state_absent",
        state_present=False,
    )
    gone_accounting = speculative_output_accounting(gone)
    assert gone_accounting is not None
    assert gone_accounting["provider_readiness"] == "absent"
    assert gone_accounting["provider_state_present"] is False

    # A ready row reports presence too, and an older caller that passes no
    # presence keeps the field absent rather than inventing a value for it.
    ready = _intent_row()
    record_provider_readiness(ready, readiness=PROVIDER_READY)
    ready_accounting = speculative_output_accounting(ready)
    assert ready_accounting is not None
    assert ready_accounting["provider_state_present"] is None
    assert ready_accounting["provider_decline_reason"] is None
