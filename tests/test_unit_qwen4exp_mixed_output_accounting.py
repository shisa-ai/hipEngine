"""Per-row MTP-versus-AR accounting for the shared Flash-Next mixed pass.

``Qwen4ExpServingMixed`` commits a speculative cycle for an MTP row and one
autoregressive token for each ordinary batch peer in the same target pass. The
MTP row's committed output is published through the shared accounting helper as
``diagnostics.specdec2_mtp2``; a peer carries no speculative intent and must
publish no split at all.

Speculative tokens are the tokens a cycle committed: the accepted drafts plus
the target's own correction or bonus token. Proposals that were rejected and
the prompt-prefill root token are never counted.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.qwen4_exp_mtp_stages import Qwen4ExpMTPProposal
from hipengine.generation.qwen4_exp_serving_mixed import Qwen4ExpServingMixed
from hipengine.generation.qwen4_exp_serving_mtp import _RequestRetired
from hipengine.generation.registry import GenerationOutput
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetResult
from hipengine.server.api import _mtp_response_summary
from hipengine.speculative import SpecCycleStage
from tests.test_unit_qwen4exp_serving_mixed import _cycle_fixture


def _accounting(native, row):
    """The block ``reclaim`` publishes for this row, read the way it does."""
    telemetry = native._mixed_telemetry(row)
    return None if telemetry is None else telemetry.diagnostics.get("specdec2_mtp2")


def _summary(native, row):
    """The public summary for the output ``reclaim`` would publish."""

    return _mtp_response_summary(
        "speculative_mtp",
        [
            GenerationOutput(
                text="",
                generated_token_ids=tuple(row.generated_ids),
                telemetry=native._mixed_telemetry(row),
            )
        ],
    )


def _draft(*, position):
    """The draft-owner stub ``_cycle_fixture`` builds, for a second MTP row."""

    draft = SimpleNamespace(position=position)
    draft.snapshot = lambda: SimpleNamespace(position=draft.position)
    draft.restore = lambda snap: setattr(draft, "position", snap.position)
    draft.trim = lambda pos: setattr(draft, "position", pos)
    draft.forward = lambda *args, **kwargs: setattr(
        draft, "position", draft.position + 1
    )
    return draft


def _plan(plan, *, candidate_counts, cycle_id=2):
    """A plan shaped like ``SpecRequestPlan``, without the reasons field."""

    return SimpleNamespace(
        request_ids=plan.request_ids,
        candidate_counts=tuple(candidate_counts),
        operation_id=plan.operation_id,
        cycle_id=cycle_id,
        target_transaction_mode=plan.target_transaction_mode,
        provider_transaction_mode=plan.provider_transaction_mode,
    )


def _k0_plan(plan, *, cycle_id=2):
    """The plan ``_mixed_k0_decode`` builds: every row's candidate count is 0."""

    return _plan(
        plan, candidate_counts=(0,) * len(plan.request_ids), cycle_id=cycle_id
    )


def test_mixed_cycle_publishes_the_split_for_the_mtp_row_only(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)

    mtp_row, peer = f.rows
    assert mtp_row.generated_ids == [11, 12, 29]
    block = _accounting(f.native, mtp_row)
    # Two committed tokens from one depth-2 cycle: the accepted draft 12 and the
    # target's own correction 29. The rejected candidate 13 is not output, and
    # the prefill root 11 at index 0 is not speculative output.
    assert block["requested_budget"] == block["candidate_budget"] == 2
    assert block["cycles"] == 1
    assert block["selected_depth_histogram"] == {"2": 1}
    assert block["generated_draft_tokens"] == 2
    assert block["accepted_draft_tokens"] == 1
    assert block["mtp_output_tokens"] == 2
    assert block["ar_output_tokens_in_cycles"] == 0
    assert block["first_fallback_position"] is None
    assert len(mtp_row.generated_ids) == block["mtp_output_tokens"] + 1

    # The peer shared the pass but never carried speculative intent, so it
    # publishes no split: its whole output is ordinary autoregressive decode.
    assert peer.generated_ids == [11, 17]
    assert _accounting(f.native, peer) is None

    # Legacy provider counters are untouched beside the new block.
    legacy = f.native._mixed_telemetry(mtp_row).diagnostics
    assert legacy["cycle_count"] == legacy["cycles_count"] == 1
    assert legacy["candidate_budget"] == 2
    assert legacy["proposed_draft_tokens"] == 2
    assert legacy["accepted_draft_tokens"] == 1
    assert "cycle_count" not in f.native._mixed_telemetry(peer).diagnostics


def test_mixed_bonus_row_is_attributed_to_the_cycle_that_committed_it(monkeypatch):
    f = _cycle_fixture(monkeypatch, truth=(12, 13, 31, 18))

    f.native._execute_mixed_cycle(f.plan)

    mtp_row = f.rows[0]
    assert mtp_row.generated_ids == [11, 12, 13, 31]
    block = _accounting(f.native, mtp_row)
    # Both drafts matched, so the cycle committed the bonus row as well: three
    # speculative tokens from one depth-2 cycle.
    assert block["accepted_draft_tokens"] == 2
    assert block["mtp_output_tokens"] == 3
    assert block["ar_output_tokens_in_cycles"] == 0
    assert len(mtp_row.generated_ids) == block["mtp_output_tokens"] + 1


def test_mixed_cycle_truncated_at_the_output_horizon_counts_committed_tokens(monkeypatch):
    f = _cycle_fixture(monkeypatch, truth=(12, 13, 31, 18))
    f.rows[0].request = replace(f.rows[0].request, max_tokens=3)

    f.native._execute_mixed_cycle(f.plan)

    mtp_row = f.rows[0]
    # The horizon stopped the cycle after the second accepted draft, so the
    # scored bonus row 31 was never committed.
    assert mtp_row.generated_ids == [11, 12, 13]
    block = _accounting(f.native, mtp_row)
    assert block["accepted_draft_tokens"] == 2
    assert block["selected_depth_histogram"] == {"2": 1}
    assert block["mtp_output_tokens"] == 2


def test_mixed_k0_pass_attributes_in_plan_autoregressive_output(monkeypatch):
    f = _cycle_fixture(monkeypatch)
    plan = _k0_plan(f.plan)

    f.native._execute_mixed_cycle(plan)

    mtp_row = f.rows[0]
    assert mtp_row.generated_ids == [11, 12]
    block = _accounting(f.native, mtp_row)
    # A K0 pass commits one target token per row and no draft: a depth-zero
    # cycle whose token is autoregressive output emitted inside the plan.
    assert block["cycles"] == 1
    assert block["selected_depth_histogram"] == {"0": 1}
    assert block["generated_draft_tokens"] == 0
    assert block["accepted_draft_tokens"] == 0
    assert block["mtp_output_tokens"] == 0
    assert block["ar_output_tokens_in_cycles"] == 1
    assert block["ar_step_reason_counts"] == {"policy_selected_ar": 1}
    # No speculative output preceded it, so this is not a fallback position.
    assert block["first_fallback_position"] is None
    assert len(mtp_row.generated_ids) == 1 + block["ar_output_tokens_in_cycles"]


def test_k0_pass_after_a_cycle_reports_where_speculation_stopped(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    f.native._execute_mixed_cycle(_k0_plan(f.plan))

    mtp_row = f.rows[0]
    block = _accounting(f.native, mtp_row)
    assert mtp_row.generated_ids == [11, 12, 29, 12]
    assert block["cycles"] == 2
    assert block["selected_depth_histogram"] == {"0": 1, "2": 1}
    assert block["mtp_output_tokens"] == 2
    assert block["ar_output_tokens_in_cycles"] == 1
    # The K0 token occupies generated index 3, so that is where the request
    # left speculative execution.
    assert block["first_fallback_position"] == 3


def test_cancelled_pass_before_commit_attributes_nothing(monkeypatch):
    f = _cycle_fixture(monkeypatch)
    visits = []

    def check(row):
        visits.append(row.request_id)
        if len(visits) == 3:
            raise _RequestRetired()

    f.native._check_request_live = check
    f.native.request_cancel_reason = lambda rid: "disconnect" if rid == 1 else None

    result = f.native._execute_mixed_cycle(f.plan)

    assert result.stage is SpecCycleStage.CANCELLED
    assert f.rows[0].generated_ids == [11]
    block = _accounting(f.native, f.rows[0])
    # Nothing committed, so nothing is attributed: intent with zero cycles.
    assert block["cycles"] == 0
    assert block["mtp_output_tokens"] == 0
    assert block["ar_output_tokens_in_cycles"] == 0
    # The block is still authoritative: the response reports the delivered
    # token as autoregressive instead of falling back to cycle-derived counts.
    summary = _summary(f.native, f.rows[0])
    assert summary["draft_cycles"] == 0
    assert summary["draft_tokens"] == 0
    assert summary["mtp_output_tokens"] == 0
    assert summary["ar_output_tokens"] == 1
    assert summary["output_accounting"]["reconciled"] is True


def test_reclaim_trim_inside_a_cycle_reports_only_the_delivered_prefix(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    row = f.rows[0]
    assert row.generated_ids == [11, 12, 29]
    assert _accounting(f.native, row)["mtp_output_tokens"] == 2

    # ``reclaim`` keeps only the ids the scheduler committed for a cancelled
    # request. The cycle committed 12 and 29; only 12 was delivered.
    row.generated_ids[:] = [11, 12]

    block = _accounting(f.native, row)
    assert block["cycles"] == 1
    assert block["selected_depth_histogram"] == {"2": 1}
    assert block["accepted_draft_tokens"] == 1
    assert block["mtp_output_tokens"] == 1
    assert block["ar_output_tokens_in_cycles"] == 0

    summary = _summary(f.native, row)
    assert summary["output_accounting"]["completion_tokens"] == 2
    assert summary["mtp_output_tokens"] == 1
    assert summary["ar_output_tokens"] == 1
    # The cycle did propose two drafts; only the split is cut to what was
    # delivered, so the proposal count is not rewritten after the fact.
    assert summary["draft_tokens"] == 2
    assert summary["accepted_draft_tokens"] == 1
    assert summary["output_accounting"]["reconciled"] is True
    assert summary["output_accounting"]["reconciled_reasons"] == []


def test_reclaim_trim_to_the_root_reports_intent_with_no_split(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    row = f.rows[0]
    row.generated_ids[:] = [11]

    block = _accounting(f.native, row)
    assert block["requested_budget"] == 2
    assert block["cycles"] == 0
    assert block["mtp_output_tokens"] == 0
    assert block["ar_output_tokens_in_cycles"] == 0
    # The prompt root is the only delivered token, so the request is fully
    # autoregressive and the cycle it committed is not published.
    summary = _summary(f.native, row)
    assert summary["output_accounting"]["completion_tokens"] == 1
    assert summary["mtp_output_tokens"] == 0
    assert summary["ar_output_tokens"] == 1
    assert summary["first_fallback_position"] == 0
    assert summary["output_accounting"]["reconciled"] is True
    # The legacy counters keep describing the pass that ran.
    assert f.native._mixed_telemetry(row).diagnostics["cycle_count"] == 1


@pytest.mark.parametrize("delivered,mtp", [([11, 12], 1), ([11, 12, 13], 2)])
def test_partial_bonus_cycle_keeps_only_delivered_accepted_drafts(
    monkeypatch, delivered, mtp
):
    f = _cycle_fixture(monkeypatch, truth=(12, 13, 31, 18))

    f.native._execute_mixed_cycle(f.plan)
    row = f.rows[0]
    assert row.generated_ids == [11, 12, 13, 31]
    row.generated_ids[:] = delivered

    block = _accounting(f.native, row)
    assert block["accepted_draft_tokens"] == mtp
    assert block["mtp_output_tokens"] == mtp
    summary = _summary(f.native, row)
    assert summary["output_accounting"]["mtp_output_tokens"] == mtp
    assert summary["output_accounting"]["reconciled"] is True


def test_trim_between_cycles_drops_only_the_undelivered_cycle(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    f.native._execute_mixed_cycle(_k0_plan(f.plan))
    row = f.rows[0]
    assert row.generated_ids == [11, 12, 29, 12]
    assert _accounting(f.native, row)["first_fallback_position"] == 3

    row.generated_ids[:] = [11, 12, 29]

    block = _accounting(f.native, row)
    assert block["cycles"] == 1
    assert block["selected_depth_histogram"] == {"2": 1}
    assert block["mtp_output_tokens"] == 2
    assert block["ar_output_tokens_in_cycles"] == 0
    assert block["ar_step_reason_counts"] == {}
    assert block["first_fallback_position"] is None


def test_output_spans_straddling_a_trim_are_cut(monkeypatch):
    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    f.native._execute_mixed_cycle(_k0_plan(f.plan))
    row = f.rows[0]
    assert [
        span["mode"] for span in _accounting(f.native, row)["output_spans"]
    ] == ["mtp", "ar"]

    # Delivered: the root and the first token of the speculative cycle. The MTP
    # span is cut to one token and the AR span after it is dropped entirely.
    row.generated_ids[:] = [11, 12]

    block = _accounting(f.native, row)
    assert [
        (span["mode"], span["position"], span["tokens"])
        for span in block["output_spans"]
    ] == [("mtp", 1, 1)]
    spans = _summary(f.native, row)["output_accounting"]["span_accounting"]
    assert spans["contiguous"] is True
    assert spans["mtp_tokens"] == 1
    assert spans["ar_tokens"] == 0
    assert spans["unspanned_tokens"] == 1
    assert spans["reconciled"] is True


def test_counted_row_without_a_commit_ledger_publishes_no_split():
    """A row counted by another writer keeps its legacy counts, not zeros.

    No production route records cycles on a serving row without this module's
    ledger, so the shape is unreachable; it is kept honest anyway because an
    aggregate cannot be cut to a delivered prefix after the fact, and a
    zero-cycle block would claim the delivered output was fully autoregressive.
    """

    row = SimpleNamespace(
        prompt_ids=(1, 2), generated_ids=[3] * 8, mtp_budget=4,
        mixed_proposed=12, mixed_accepted=6, mixed_cycles=3,
        mixed_widths={2: 3}, mixed_target_rows=18,
    )

    assert _accounting(Qwen4ExpServingMixed(), row) is None


def test_row_that_finished_on_its_root_publishes_intent_with_no_cycles():
    # The prompt root is emitted by prefill, outside any cycle: a request that
    # stops there still carried speculative intent, so it reports zero cycles
    # rather than pretending it never asked to speculate.
    row = SimpleNamespace(prompt_ids=(1, 2), generated_ids=[3], mtp_budget=4,
                          mixed_proposed=0, mixed_accepted=0, mixed_cycles=0,
                          mixed_widths={}, mixed_target_rows=0)

    telemetry = Qwen4ExpServingMixed()._mixed_telemetry(row)

    block = telemetry.diagnostics["specdec2_mtp2"]
    assert block["requested_budget"] == block["candidate_budget"] == 4
    assert block["cycles"] == 0
    assert block["mtp_output_tokens"] == 0
    assert block["ar_output_tokens_in_cycles"] == 0
    summary = _mtp_response_summary(
        "speculative_mtp",
        [GenerationOutput(text="", generated_token_ids=(3,), telemetry=telemetry)],
    )
    assert summary["mtp_output_tokens"] == 0
    assert summary["ar_output_tokens"] == 1
    assert summary["first_fallback_position"] == 0
    assert summary["output_accounting"]["reconciled"] is True


def test_untrimmed_output_reproduces_the_committed_block(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    f.native._execute_mixed_cycle(_k0_plan(f.plan))

    block = _accounting(f.native, f.rows[0])
    # Nothing was trimmed, so the published split equals what the two passes
    # committed: one depth-2 cycle and one depth-zero cycle.
    assert block["cycles"] == 2
    assert block["selected_depth_histogram"] == {"0": 1, "2": 1}
    assert block["generated_draft_tokens"] == 2
    assert block["accepted_draft_tokens"] == 1
    assert block["mtp_output_tokens"] == 2
    assert block["ar_output_tokens_in_cycles"] == 1
    assert block["first_fallback_position"] == 3


def test_two_mtp_rows_in_one_pass_keep_independent_counters(monkeypatch):
    f = _cycle_fixture(monkeypatch)
    second = f.rows[1]
    second.mixed_mtp = True
    second.mtp_budget = 2
    second.mixed_root_hidden = f.rows[0].mixed_root_hidden
    second.draft = _draft(position=2)

    def proposal(draft, **kwargs):
        draft.position += 2
        return Qwen4ExpMTPProposal(
            kwargs["start_position"], kwargs["root_token"], (12, 13)
        )

    # Request 1 accepts its first draft; request 2's first draft is rejected, so
    # the two rows commit different numbers of tokens in the same pass.
    truth = {1: (12, 29, 31), 2: (28, 29, 31)}

    class Target:
        def __init__(self, batch, layout):
            self.batch, self.layout = batch, layout

        def verify(self):
            positions = [
                truth[row.request_id][row.depth] for row in self.layout.rows
            ]
            return Qwen4ExpMixedTargetResult(
                self.layout,
                np.array(positions),
                np.zeros((len(positions), 64)),
                np.zeros((len(positions), 4)),
            )

        def commit(self, depths):
            for request, depth in zip(self.layout.requests, depths, strict=True):
                self.batch._positions[request.slot] += depth + 1

        def abort(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_serving_mixed.prepare_proposal", proposal
    )
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_serving_mixed.Qwen4ExpMixedTargetPass", Target
    )

    f.native._execute_mixed_cycle(_plan(f.plan, candidate_counts=(2, 2)))

    assert f.rows[0].generated_ids == [11, 12, 29]
    assert second.generated_ids == [11, 28]
    first_block = _accounting(f.native, f.rows[0])
    second_block = _accounting(f.native, second)
    assert first_block["accepted_draft_tokens"] == 1
    assert first_block["mtp_output_tokens"] == 2
    assert second_block["accepted_draft_tokens"] == 0
    # A depth-2 cycle whose first candidate was rejected still commits the
    # target's own correction token as speculative output, and it is still a
    # depth-2 cycle rather than a depth-zero one.
    assert second_block["selected_depth_histogram"] == {"2": 1}
    assert second_block["mtp_output_tokens"] == 1
    assert second_block["ar_output_tokens_in_cycles"] == 0
    assert first_block is not second_block


def test_server_accounting_reconciles_the_published_mixed_block(monkeypatch):
    f = _cycle_fixture(monkeypatch)

    f.native._execute_mixed_cycle(f.plan)
    row = f.rows[0]
    output = GenerationOutput(
        text="",
        generated_token_ids=tuple(row.generated_ids),
        telemetry=f.native._mixed_telemetry(row),
    )

    summary = _mtp_response_summary("speculative_mtp", [output])

    assert summary["used"] is True
    assert summary["draft_cycles"] == 1
    assert summary["draft_tokens"] == 2
    assert summary["accepted_draft_tokens"] == 1
    assert summary["mtp_output_tokens"] == 2
    assert summary["ar_output_tokens"] == 1
    accounting = summary["output_accounting"]
    assert accounting["completion_tokens"] == 3
    assert accounting["mtp_output_tokens_explained_by_cycles"] == 2
    assert accounting["unexplained_mtp_output_tokens"] == 0
    assert accounting["reconciled"] is True
    assert accounting["reconciled_reasons"] == []
