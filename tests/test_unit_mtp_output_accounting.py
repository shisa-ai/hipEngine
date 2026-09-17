"""Unit coverage for per-request MTP-versus-AR output attribution.

The response layer derives autoregressive output as
``completion_tokens - mtp2_mtp_output_tokens``, so these counters are the only
thing standing between "the route said MTP" and "these tokens came from MTP".
"""

from __future__ import annotations

from types import SimpleNamespace

from hipengine.speculative.accounting import (
    accounting_timing_fields,
    record_autoregressive_step,
    record_speculative_outputs,
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
