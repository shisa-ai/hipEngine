"""Per-request committed-output accounting for the Qwen4Exp MTP provider.

The provider publishes ``diagnostics.specdec2_mtp2`` through the shared
speculative accounting helper, so a Qwen4Exp response reports the same
MTP-versus-AR split as the resident routes: committed speculative output, the
autoregressive tokens a K0 step emitted inside the plan, and nothing else.

Speculative tokens are the tokens a cycle actually committed — the accepted
drafts plus the target's own correction or bonus token — never the proposals
that were rejected and never the prompt-prefill root token.
"""

from __future__ import annotations

import pytest

from hipengine.generation.deadline import (
    GenerationCancelled,
    GenerationCancellationToken,
)
from hipengine.server.api import _mtp_response_summary
from tests.test_unit_qwen4exp_mtp_steps import (
    _BatchedTargetRunner,
    _DraftRunner,
    _drain,
    _provider,
    _request,
)


def _block(output):
    diagnostics = output.telemetry.diagnostics
    assert "specdec2_mtp2" in diagnostics
    return diagnostics["specdec2_mtp2"]


def test_serial_cycles_attribute_committed_tokens_not_proposals():
    provider, _target, _draft = _provider()

    steps, output = _drain(provider.iter_generate((1, 2), _request(max_tokens=5)))

    assert [step.token_ids for step in steps] == [(10,), (11, 12), (13, 14)]
    block = _block(output)
    assert block["requested_budget"] == block["candidate_budget"] == 2
    assert block["cycles"] == 2
    # Two depth-2 cycles proposed four drafts; three matched.
    assert block["generated_draft_tokens"] == 4
    assert block["accepted_draft_tokens"] == 3
    assert block["selected_depth_histogram"] == {"2": 2}
    # Cycle one committed its accepted draft plus the target correction, cycle
    # two committed two accepted drafts at the length horizon: four speculative
    # tokens, never the four proposals and never the rejected 98.
    assert block["mtp_output_tokens"] == 4
    assert block["ar_output_tokens_in_cycles"] == 0
    assert block["first_fallback_position"] is None
    assert block["ar_step_reason_counts"] == {}
    # The prompt-prefill root token is the only autoregressive token and it is
    # not attributed to the plan.
    assert output.generated_token_ids == (10, 11, 12, 13, 14)
    assert len(output.generated_token_ids) == block["mtp_output_tokens"] + 1


def test_batched_bonus_row_is_attributed_to_the_cycle_that_committed_it():
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (14, 15)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    steps, output = _drain(provider.iter_generate((1, 2), _request(max_tokens=5)))

    assert [step.token_ids for step in steps] == [(10,), (11, 12, 13), (14,)]
    block = _block(output)
    # The first cycle verified three rows: two accepted drafts plus the bonus
    # row, which is speculative output. The one-token tail cycle was a plain
    # serial target step, so the histogram records the depth actually selected.
    assert block["selected_depth_histogram"] == {"1": 1, "2": 1}
    assert block["generated_draft_tokens"] == 3
    assert block["accepted_draft_tokens"] == 3
    assert block["mtp_output_tokens"] == 4
    assert block["ar_output_tokens_in_cycles"] == 0
    assert output.generated_token_ids == (10, 11, 12, 13, 14)


def test_cycle_truncated_by_eos_counts_only_committed_tokens():
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (14, 15)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    steps, output = _drain(
        provider.iter_generate(
            (1, 2), _request(max_tokens=8, ignore_eos=False, eos_token_id=12)
        )
    )

    assert [step.token_ids for step in steps] == [(10,), (11, 12)]
    assert output.finish_details.reason == "eos"
    assert output.generated_token_ids == (10, 11, 12)
    block = _block(output)
    # The target's own EOS ended the cycle inside the verified rows, so the
    # bonus row was never committed and must not be counted even though it was
    # scored.
    assert block["cycles"] == 1
    assert block["accepted_draft_tokens"] == 2
    assert block["selected_depth_histogram"] == {"2": 1}
    assert block["mtp_output_tokens"] == 2
    assert block["ar_output_tokens_in_cycles"] == 0


def test_ar_only_scheduler_steps_report_no_speculative_output():
    provider, _target, _draft = _provider()

    steps = provider.iter_generate((1, 2), _request(max_tokens=3))
    assert next(steps).token_ids == (10,)
    assert steps.send(0).token_ids == (11,)
    assert steps.send(0).token_ids == (12,)
    with pytest.raises(StopIteration) as terminal:
        next(steps)
    output = terminal.value.value

    block = _block(output)
    assert output.generated_token_ids == (10, 11, 12)
    assert block["requested_budget"] == 2
    assert block["cycles"] == 0
    assert block["generated_draft_tokens"] == 0
    assert block["accepted_draft_tokens"] == 0
    assert block["selected_depth_histogram"] == {}
    assert block["mtp_output_tokens"] == 0
    # Both K0 tokens were emitted while the request was still owned by the plan.
    assert block["ar_output_tokens_in_cycles"] == 2
    assert block["ar_step_reason_counts"] == {"scheduler_k0": 2}
    assert block["first_fallback_position"] is None

    summary = _mtp_response_summary("speculative", [output])
    # ``used`` is route-level for this provider: its execution path names the
    # MTP verifier even when every step ran as K0.
    assert summary["used"] is True
    assert summary["draft_cycles"] == 0
    assert summary["mtp_output_tokens"] == 0
    assert summary["ar_output_tokens"] == 3
    assert summary["first_fallback_position"] == 0
    assert summary["fallback_reason"] == "scheduler_k0"
    assert summary["output_accounting"]["reconciled"] is True


def test_k0_step_after_a_cycle_reports_where_speculation_stopped():
    provider, _target, _draft = _provider()

    steps = provider.iter_generate((1, 2), _request(max_tokens=4))
    assert next(steps).token_ids == (10,)
    assert steps.send(2).token_ids == (11, 12)
    assert steps.send(0).token_ids == (13,)
    with pytest.raises(StopIteration) as terminal:
        next(steps)
    output = terminal.value.value

    block = _block(output)
    assert output.generated_token_ids == (10, 11, 12, 13)
    assert block["cycles"] == 1
    assert block["mtp_output_tokens"] == 2
    assert block["ar_output_tokens_in_cycles"] == 1
    # The K0 token occupies generated index 3, so that is where the request
    # left speculative execution.
    assert block["first_fallback_position"] == 3
    assert block["ar_step_reason_counts"] == {"scheduler_k0": 1}


def test_cancelled_step_publishes_no_partial_accounting():
    cancellation = GenerationCancellationToken()
    provider, target, draft = _provider()

    steps = provider.iter_generate(
        (1, 2), _request(max_tokens=5, cancellation_token=cancellation)
    )
    assert next(steps).token_ids == (10,)
    cancellation.cancel()
    with pytest.raises(GenerationCancelled):
        steps.send(2)
    # Nothing was committed by the cancelled step, so nothing can be attributed
    # and no terminal output (the only publication point) exists.
    assert target.runner.position == draft.position == 2
    steps.close()


def test_output_spans_tile_the_committed_output(monkeypatch):
    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (14, 15)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    _steps, output = _drain(provider.iter_generate((1, 2), _request(max_tokens=5)))

    block = _block(output)
    # Diagnostic mode records where each committed span starts, so the counters
    # above are checkable against the emitted tokens instead of a subtraction.
    # The prompt root at index 0 is deliberately unspanned.
    assert block["output_spans"] == [
        {"mode": "mtp", "reason": None, "position": 1, "tokens": 3},
        {"mode": "mtp", "reason": None, "position": 4, "tokens": 1},
    ]
    spans = _mtp_response_summary("speculative", [output])["output_accounting"][
        "span_accounting"
    ]
    assert spans["contiguous"] is True
    assert spans["mtp_tokens"] == 4
    assert spans["unspanned_tokens"] == 1
    assert spans["unspanned_tokens_match"] is True
    assert spans["reconciled"] is True


def test_server_accounting_reconciles_the_published_block():
    runner = _BatchedTargetRunner(truth=(10, 11, 12, 13, 14))
    draft = _DraftRunner(proposals=((11, 12), (14, 15)))
    provider, _target, _draft = _provider(runner=runner, draft=draft)

    _steps, output = _drain(provider.iter_generate((1, 2), _request(max_tokens=5)))

    summary = _mtp_response_summary("speculative", [output])
    assert summary["used"] is True
    assert summary["draft_cycles"] == 2
    assert summary["draft_tokens"] == 3
    assert summary["accepted_draft_tokens"] == 3
    assert summary["mtp_output_tokens"] == 4
    assert summary["ar_output_tokens"] == 1
    assert summary["first_fallback_position"] is None
    accounting = summary["output_accounting"]
    assert accounting["completion_tokens"] == 5
    assert accounting["mtp_coverage"] == pytest.approx(0.8)
    assert accounting["mtp_output_tokens_explained_by_cycles"] == 5
    # The explained count is an upper bound (accepted drafts plus one verified
    # token per speculative cycle), so a cycle that committed no bonus or
    # correction token leaves it above the actual speculative output.
    assert accounting["unexplained_mtp_output_tokens"] <= 0
    assert accounting["reconciled"] is True
    assert accounting["reconciled_reasons"] == []
