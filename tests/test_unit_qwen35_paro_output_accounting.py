"""PARO publishes the shared MTP-versus-AR output accounting block.

The PARO route used to increment its cycle counters by hand and publish only
route-snapshot counters, so a served response could not answer which tokens
came from committed speculative cycles and which came from autoregressive
decoding. These tests drive the real slotted resident row through the real
adapter commit path and assert the same ``diagnostics.specdec2_mtp2`` schema the
GGUF routes publish, without changing generated tokens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
from hipengine.generation.batch_scheduler import (
    GeneratedToken,
    GeneratedTokenEvent,
)
from hipengine.generation.qwen35_paro import (
    Qwen35ParoResidentModelRunner,
    _ParoResidentLoopRow,
)
from hipengine.generation.registry import (
    FinishDetails,
    GenerationStreamChunk,
    GenerationTelemetry,
)
from hipengine.generation.qwen35_paro_mtp2 import (
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
)
from hipengine.generation.sampling import SamplingMode
from hipengine.kvcache import ResourceClaimSet
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
)
from hipengine.speculative.accounting import (
    record_speculative_outputs,
    span_accounting,
    speculative_output_accounting,
)
from hipengine.speculative.frontier import (
    SpecK0Class,
    SpecPlanReason,
    SpecTransactionMode,
)
from hipengine.speculative.transaction import SpecCycleStage

REQUEST_ID = 7
OPERATION_ID = "op-paro-output-accounting"
# Exact binary fractions, so the committed phase sums are exact after the
# helper's seconds-to-milliseconds conversion: proposal 12.5 ms, target 62.5 ms,
# provider update 31.25 ms.
_CLOCK_TICKS = (0.0, 0.0125, 0.0625, 0.125, 0.25, 0.28125)


class _SessionDouble:
    """Only the session surface one committed C1 cycle touches."""

    def __init__(self) -> None:
        self.position_arr = [0, 0]
        self.context_arr = [0, 0]
        self.tokenizer = SimpleNamespace(decode=lambda ids: "")
        self.runtime = SimpleNamespace(device_synchronize=lambda: None)
        self.slot_copies = []

    def copy_slot_state(self, source, destination, *, kv_rows) -> None:
        self.slot_copies.append((source, destination, int(kv_rows)))

    def _record_slot_position_host(self, position, *, slot) -> None:
        self.position_arr[slot] = int(position)


class _ProposerDouble:
    def __init__(self) -> None:
        self.position = 2
        self.cache_len = 2
        self.closed = False
        self.checkpoints = []
        self.advances = []

    def save_state(self, slot):
        self.checkpoints.append(int(slot))
        return ("snapshot", int(slot))

    def restore_state(self, snapshot) -> None:
        self.checkpoints.append(snapshot)

    def device_candidate_token_ids(self, rows=1):
        return None

    def advance_with_previous_hidden(self, **kwargs):
        self.advances.append(kwargs)
        return None

    def advance_with_target_hidden(self, **kwargs):
        self.advances.append(kwargs)
        return None

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _sampler_plan() -> SimpleNamespace:
    return SimpleNamespace(
        mode=SamplingMode.GREEDY_FAST,
        active_processors=(),
        fast_path_blockers=(),
        fallback_reason=None,
    )


def _row(*, generated=1) -> _ParoResidentLoopRow:
    """The real slots row, seeded with the prefill root token."""

    row = _ParoResidentLoopRow(
        request_id=REQUEST_ID,
        batch_id=0,
        row_index=0,
        request=SimpleNamespace(max_tokens=16, eos_token_id=None),
        sampling_request=SimpleNamespace(
            stop_token_sequences=(),
            stop_token_ids=(),
            ignore_eos=True,
            logprobs=False,
            top_logprobs=0,
        ),
        prompt_ids=(10, 11),
        sampler_plan=_sampler_plan(),
        native_greedy=True,
        submitted_at=0.0,
        model_slot=0,
        first_token_emitted=True,
    )
    for index in range(generated):
        row.generated_steps.append(
            Qwen35ParoAutoregressiveStepResult(
                token_id=100 + index,
                token_text="",
                logit=float("nan"),
            )
        )
    return row


def _runner(row, session, adapter) -> Qwen35ParoResidentModelRunner:
    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner.generator = SimpleNamespace(
        backend="hip_gfx1100",
        execution_profile="production",
    )
    runner.capacity = 1
    runner._session = session
    runner._rows = {REQUEST_ID: row}
    runner._route_counts = Counter()
    runner._fallback_reasons = Counter()
    runner._last_width_plan = {"groups": 1}
    runner._mtp2_adapter = adapter if adapter is not None else Qwen35ParoMTP2Adapter(runner)
    runner._mtp2_adapter_resolved = True
    runner._closed = False
    return runner


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        request_ids=(REQUEST_ID,),
        operation_id=OPERATION_ID,
        cycle_id=1,
        candidate_counts=(1,),
        reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
        k0_classes=(SpecK0Class.NOT_K0,),
        proposal_widths=(1,),
        target_row_decomposition=(2,),
        target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        provider_key="qwen_paro_mtp_bf16",
        resident_slots=(0,),
        execution_route="eager",
    )


def _frontier() -> SimpleNamespace:
    return SimpleNamespace(
        target_batch="batch",
        provider_transaction_id=5,
        root_positions=(2,),
        candidate_graph=SimpleNamespace(candidate_rows=1, candidate_counts=(1,), token_ids=None),
    )


class _CycleHarness:
    def __init__(self, *, row, session, proposer, adapter, runner) -> None:
        self.row = row
        self.session = session
        self.proposer = proposer
        self.adapter = adapter
        self.runner = runner

    def diagnostics(self) -> dict:
        telemetry = self.runner._row_telemetry(self.row)
        assert telemetry.diagnostics is not None
        return dict(telemetry.diagnostics)

    def timing(self) -> dict:
        telemetry = self.runner._row_telemetry(self.row)
        assert telemetry.timing is not None
        return dict(telemetry.timing)

    def accounting(self) -> dict | None:
        return self.diagnostics().get("specdec2_mtp2")


def _harness(*, generated=1, accepted=1, draft_token=200, bonus_token=201):
    row = _row(generated=generated)
    session = _SessionDouble()
    proposer = _ProposerDouble()
    runner = _runner(row, session, None)
    adapter = Qwen35ParoMTP2Adapter(runner)
    runner._mtp2_adapter = adapter
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(REQUEST_ID, proposer)
    adapter._verify_target = lambda session, batch, **kwargs: SimpleNamespace(
        accepted_count=int(accepted),
        next_token=int(bonus_token),
        target_top1=(int(bonus_token),),
        accepted_tokens=(int(draft_token),) if accepted else (),
        selected_target_hidden_ptr=0x9000,
        commit_row=int(accepted),
    )
    return _CycleHarness(
        row=row,
        session=session,
        proposer=proposer,
        adapter=adapter,
        runner=runner,
    )


def _ar_step(token_id: int) -> Qwen35ParoAutoregressiveStepResult:
    return Qwen35ParoAutoregressiveStepResult(
        token_id=int(token_id),
        token_text="",
        logit=float("nan"),
    )


def _scheduler_event(
    request_id: int,
    token_id: int,
    *,
    finished: bool,
    generated: int,
) -> GeneratedTokenEvent:
    """One event shaped exactly like ``record_speculative_cycle_result``.

    That method records a committed cycle through
    ``_append_generated_token_with_stream_chunk`` with no runner chunk, so the
    scheduler builds its own telemetry from the per-token decode counts and
    carries no accounting diagnostics at all. The finished event carries the
    scheduler's own ``FinishDetails`` and cumulative generated ids.
    """

    return GeneratedTokenEvent(
        request_id=int(request_id),
        token_id=int(token_id),
        finished=bool(finished),
        stream_chunk=GenerationStreamChunk(
            text="",
            finish_details=FinishDetails(reason="stop") if finished else None,
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=2,
                generated_tokens=int(generated),
                phase="answer",
                request_id=str(request_id),
                execution_path="resident_scheduler",
            ),
            generated_token_ids=(100, 200) if finished else None,
        ),
    )


def _install_clock(monkeypatch) -> None:
    state = {"index": 0}

    def _tick() -> float:
        index = state["index"]
        state["index"] = index + 1
        return _CLOCK_TICKS[index] if index < len(_CLOCK_TICKS) else _CLOCK_TICKS[-1]

    monkeypatch.setattr(paro_mtp2_module.time, "perf_counter", _tick)


def _commit_cycle(harness, *, cancelled=()):
    plan = _plan()
    adapter = harness.adapter
    harness.runner.register_speculative_request(REQUEST_ID, 1)
    adapter.propose_batch(
        plan,
        (
            SimpleNamespace(
                request_id=REQUEST_ID,
                sampling_mode="greedy",
                mode="verify_chain",
                context_tokens=3,
                remaining_decode=8,
            ),
        ),
    )
    claims = ResourceClaimSet.from_mapping(
        OPERATION_ID, {"paro_mtp2.target_rows": 2}
    )
    return adapter.execute_target_frontier(
        plan,
        _frontier(),
        claims,
        commit=True,
        cancelled_request_ids=lambda: tuple(cancelled),
    )


# --------------------------------------------------------------- the schema


def test_paro_row_declares_every_field_the_accounting_helpers_write() -> None:
    """The row is a slots dataclass, so an undeclared write raises mid-decode.

    ``record_speculative_outputs`` cannot attach a new attribute at runtime, and
    the failure surfaces inside the engine loop rather than as a missing fact.
    """

    names = {entry.name for entry in fields(_ParoResidentLoopRow)}
    for name in (
        "mtp2_requested_budget",
        "mtp2_mtp_output_tokens",
        "mtp2_ar_output_tokens",
        "mtp2_ar_step_output_tokens",
        "mtp2_first_fallback_position",
        "mtp2_ar_step_reasons",
        "mtp2_cycles",
        "mtp2_candidate_counts",
        "mtp2_accepted_counts",
        "mtp2_proposal_ms",
        "mtp2_target_ms",
        "mtp2_provider_update_ms",
    ):
        assert name in names

    row = _row()
    record_speculative_outputs(
        row,
        candidate_count=1,
        accepted_count=1,
        visible_count=2,
        output_position=1,
        plan_reason=SpecPlanReason.SPECULATIVE_QUALIFIED,
    )
    accounting = speculative_output_accounting(row)
    assert accounting is not None
    assert accounting["cycles"] == 1
    assert accounting["mtp_output_tokens"] == 2


def test_plain_autoregressive_paro_response_gains_no_speculative_block() -> None:
    """A row without speculative intent must publish exactly what it did before."""

    row = _row()
    runner = _runner(row, _SessionDouble(), None)

    diagnostics = runner._row_telemetry(row).diagnostics
    assert diagnostics is not None
    assert "specdec2_mtp2" not in diagnostics
    assert diagnostics["stable_model_slot"] == 0
    assert diagnostics["last_width_plan"] == {"groups": 1}
    timing = runner._row_telemetry(row).timing
    assert timing is not None
    assert "mtp_cycles_count" not in timing
    assert "mtp_visible_output_tokens" not in timing


def test_accepted_b1_cycle_publishes_the_shared_accounting_block(monkeypatch) -> None:
    _install_clock(monkeypatch)
    harness = _harness(accepted=1)

    result = _commit_cycle(harness)

    assert result.stage is SpecCycleStage.COMMITTED
    # Generated tokens are unchanged: accepted draft then the target bonus.
    assert [step.token_id for step in harness.row.generated_steps] == [100, 200, 201]
    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["requested_budget"] == 1
    assert accounting["cycles"] == 1
    assert accounting["generated_draft_tokens"] == 1
    assert accounting["accepted_draft_tokens"] == 1
    assert accounting["selected_depth_histogram"] == {"1": 1}
    # One accepted draft plus the target-verified bonus token.
    assert accounting["mtp_output_tokens"] == 2
    assert accounting["ar_output_tokens_in_cycles"] == 0
    assert accounting["first_fallback_position"] is None
    # Committed phase sums, in the same ``cycle_timing_ms`` block GGUF publishes.
    assert accounting["cycle_timing_ms"]["proposal"] == 12.5
    assert accounting["cycle_timing_ms"]["target"] == 62.5
    assert accounting["cycle_timing_ms"]["provider_update"] == 31.25
    # Phases this route does not measure report zero rather than a guess.
    assert accounting["cycle_timing_ms"]["accept"] == 0.0
    # Diagnostics are authoritative; upstream removed legacy timing mirrors.
    assert not any(name.startswith("mtp_") for name in harness.timing())
    # The route fields this change must not disturb.
    assert harness.row.mtp2_execution_routes == ["eager"]
    assert harness.runner._execution_metadata(harness.row)["specdec2_mtp2_cycles"] == 1


def test_committed_cycle_span_uses_the_real_output_positions(monkeypatch) -> None:
    """The diagnostic span starts where the cycle's tokens really land.

    The row already carries the prefill root at index 0, so the committed
    cycle's span starts at index 1 with two tokens. A span derived from the
    acceptance count instead of the emitted list would land at 0 or claim one
    token.
    """

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    _install_clock(monkeypatch)
    harness = _harness(accepted=1)

    _commit_cycle(harness)

    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["output_spans"] == [
        {
            "mode": "mtp",
            "reason": SpecPlanReason.SPECULATIVE_QUALIFIED.value,
            "position": 1,
            "tokens": 2,
        }
    ]


def test_committed_spans_tile_the_real_paro_output(monkeypatch) -> None:
    """The recorded spans reconcile against the emitted token list.

    The response layer derives autoregressive output as ``completion_tokens -
    mtp_output_tokens``, which holds by construction. Spans are the independent
    check: they must tile the emitted output with only out-of-plan AR tokens
    left unspanned (here the prefill root).
    """

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    _install_clock(monkeypatch)
    harness = _harness(accepted=1)
    _commit_cycle(harness)
    harness.row.generated_steps.append(
        Qwen35ParoAutoregressiveStepResult(
            token_id=400,
            token_text="",
            logit=float("nan"),
        )
    )
    harness.adapter.note_speculative_ar_commit(REQUEST_ID, "no_provider")
    accounting = harness.accounting()
    assert accounting is not None

    completion_tokens = len(harness.row.generated_steps)
    mtp_output_tokens = accounting["mtp_output_tokens"]
    reconciled = span_accounting(
        accounting["output_spans"],
        completion_tokens=completion_tokens,
        mtp_output_tokens=mtp_output_tokens,
        ar_output_tokens=completion_tokens - mtp_output_tokens,
        ar_output_tokens_in_cycles=accounting["ar_output_tokens_in_cycles"],
    )

    assert completion_tokens == 4
    assert mtp_output_tokens == 2
    assert reconciled["spans"] == 2
    assert reconciled["unspanned_tokens"] == 1
    assert reconciled["reconciled"] is True
    assert reconciled["reconciled_reasons"] == []


def test_rejected_b1_cycle_counts_the_bonus_without_an_accepted_draft(
    monkeypatch,
) -> None:
    """A rejected draft still commits one visible target-verified token."""

    _install_clock(monkeypatch)
    harness = _harness(accepted=0, bonus_token=301)

    _commit_cycle(harness)

    assert [step.token_id for step in harness.row.generated_steps] == [100, 301]
    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["cycles"] == 1
    assert accounting["generated_draft_tokens"] == 1
    assert accounting["accepted_draft_tokens"] == 0
    assert accounting["mtp_output_tokens"] == 1
    assert accounting["ar_output_tokens_in_cycles"] == 0
    assert accounting["accepted_draft_tokens"] == 0
    assert "mtp_visible_output_tokens" not in harness.timing()


def test_rolled_back_cycle_records_no_committed_output(monkeypatch) -> None:
    """A cycle that fails after verify restores both owners and commits nothing."""

    _install_clock(monkeypatch)
    harness = _harness(accepted=1)

    def _fail(**kwargs):
        raise RuntimeError("provider update failed")

    harness.proposer.advance_with_target_hidden = _fail

    with pytest.raises(RuntimeError, match="provider update failed"):
        _commit_cycle(harness)
    harness.adapter.rollback_cycle(
        _plan(), None, RuntimeError("provider update failed")
    )

    assert [step.token_id for step in harness.row.generated_steps] == [100]
    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["cycles"] == 0
    assert accounting["mtp_output_tokens"] == 0
    # The proposal ran, but no cycle committed, so no phase sum is published.
    assert harness.row.mtp2_proposal_ms == 0.0
    assert accounting["cycle_timing_ms"]["proposal"] == 0.0


def test_cancelled_cycle_publishes_no_committed_output(monkeypatch) -> None:
    """An aborted cycle rolled back its target and emitted nothing.

    The request still carries speculative intent, so the block exists; it must
    report zero cycles and no committed output rather than the work the cycle
    did before it was cancelled.
    """

    _install_clock(monkeypatch)
    harness = _harness(accepted=1)

    result = _commit_cycle(harness, cancelled=(REQUEST_ID,))

    assert result.stage is SpecCycleStage.CANCELLED
    assert [step.token_id for step in harness.row.generated_steps] == [100]
    assert harness.row.mtp2_cycles == 0
    assert harness.row.mtp2_candidate_counts == []
    assert harness.row.mtp2_accepted_counts == []
    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["cycles"] == 0
    assert accounting["mtp_output_tokens"] == 0
    assert accounting["ar_output_tokens_in_cycles"] == 0
    # A zero-cycle refusal must stay indistinguishable from plain AR in timing.
    assert "mtp_cycles_count" not in harness.timing()


def test_autoregressive_step_reports_the_real_fallback_position(monkeypatch) -> None:
    """AR output after a committed cycle is attributed where it was emitted."""

    _install_clock(monkeypatch)
    harness = _harness(accepted=1)
    _commit_cycle(harness)
    # The engine loop calls this after the AR decode emitted the token, so the
    # token is already the row's last generated step.
    harness.row.generated_steps.append(
        Qwen35ParoAutoregressiveStepResult(
            token_id=400,
            token_text="",
            logit=float("nan"),
        )
    )

    harness.adapter.note_speculative_ar_commit(REQUEST_ID, "no_provider")

    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["mtp_output_tokens"] == 2
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] == 3
    assert accounting["ar_step_reason_counts"] == {"no_provider": 1}
    assert accounting["failure_reason_counts"] == {}
    assert "mtp_ar_output_tokens" not in harness.timing()


def test_autoregressive_step_before_any_speculative_output_is_not_a_fallback() -> None:
    harness = _harness(accepted=1)
    harness.runner.register_speculative_request(REQUEST_ID, 1)

    harness.adapter.note_speculative_ar_commit(REQUEST_ID, "no_provider")

    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] is None
    assert accounting["mtp_output_tokens"] == 0
    # Intent without a committed cycle publishes no timing mirrors.
    assert "mtp_cycles_count" not in harness.timing()


def test_autoregressive_commit_for_an_unknown_request_records_nothing() -> None:
    harness = _harness(accepted=1)
    harness.runner.register_speculative_request(REQUEST_ID, 1)

    harness.adapter.note_speculative_ar_commit(REQUEST_ID + 1, "no_provider")

    accounting = harness.accounting()
    assert accounting is not None
    assert accounting["ar_output_tokens_in_cycles"] == 0
    assert accounting["ar_step_reason_counts"] == {}


def test_execution_metadata_keeps_the_route_counters_alongside_the_block(
    monkeypatch,
) -> None:
    _install_clock(monkeypatch)
    harness = _harness(accepted=1)
    _commit_cycle(harness)

    metadata = harness.runner._execution_metadata(harness.row)

    assert metadata["specdec2_mtp2_used"] is True
    assert metadata["specdec2_mtp2_cycles"] == 1
    assert metadata["specdec2_mtp2_candidate_counts"] == [1]
    assert metadata["specdec2_mtp2_accepted_counts"] == [1]
    assert metadata["specdec2_mtp2_execution_routes"] == ["eager"]
    assert harness.accounting()["cycles"] == 1


def test_registered_intent_without_a_cycle_publishes_a_zero_cycle_block() -> None:
    """Admission without realized speculation is reported, not hidden."""

    row = _row()
    runner = _runner(row, _SessionDouble(), None)

    runner.register_speculative_request(REQUEST_ID, 1)

    diagnostics = runner._row_telemetry(row).diagnostics
    assert diagnostics is not None
    accounting = diagnostics["specdec2_mtp2"]
    assert accounting["requested_budget"] == 1
    assert accounting["cycles"] == 0
    assert accounting["mtp_output_tokens"] == 0
    assert "mtp_cycles_count" not in runner._row_telemetry(row).timing


# ------------------------------------------------- publication-path refresh
#
# The counters above are only useful if they reach the response. Two paths used
# to lose them: an autoregressive token is attributed to the decode that emitted
# it *after* that decode built its chunk, and a committed speculative token is
# published by the scheduler, which never asks this route for a chunk at all.
# Both hooks are driven here at the level the engine loop drives them, so the
# parent-owned loop is not required to prove the runner side.


def test_speculative_stream_renders_tokens_after_terminal_reclaim() -> None:
    harness = _harness(accepted=1)
    harness.runner.register_speculative_request(REQUEST_ID, 1)
    harness.session.tokenizer = SimpleNamespace(
        decode=lambda ids: {200: "draft", 201: "bonus"}[ids[0]]
    )
    decorate = harness.runner.prepare_speculative_stream_events((REQUEST_ID,))
    harness.runner._rows.pop(REQUEST_ID)
    harness.runner._session = None
    events = (
        _scheduler_event(REQUEST_ID, 200, finished=False, generated=2),
        _scheduler_event(REQUEST_ID, 201, finished=True, generated=3),
    )

    result = decorate(events)

    assert "".join(event.stream_chunk.text for event in result) == "draftbonus"
    for original, rendered in zip(events, result, strict=True):
        assert original.token_id == rendered.token_id
        assert original.finished == rendered.finished
        assert original.stream_chunk.finish_details == rendered.stream_chunk.finish_details
        assert original.stream_chunk.generated_token_ids == rendered.stream_chunk.generated_token_ids
        assert original.stream_chunk.telemetry.decode_state == rendered.stream_chunk.telemetry.decode_state
    pre_rendered = replace(events[0], stream_chunk=replace(events[0].stream_chunk, text="owned text"))
    assert decorate((pre_rendered,))[0].stream_chunk.text == "owned text"


def test_autoregressive_chunk_metadata_is_stale_until_the_hook_refreshes_it(
    monkeypatch,
) -> None:
    """The chunk PARO builds precedes the attribution the loop performs later.

    ``decode_batch`` snapshots the row's accounting while it builds the chunk,
    and the engine loop attributes the token to the decode that emitted it only
    once that decode returned. A chunk published without the refresh therefore
    reports the request's earlier state: no fallback position, no AR step
    reason, and a stale AR output count.
    """

    _install_clock(monkeypatch)
    harness = _harness(accepted=1)
    _commit_cycle(harness)
    step = _ar_step(400)
    harness.runner._record_step(harness.row, step)

    stale = harness.runner._stream_chunk(harness.row, step, finished=False)
    assert stale.telemetry is not None
    assert stale.telemetry.diagnostics is not None
    stale_accounting = dict(stale.telemetry.diagnostics)["specdec2_mtp2"]
    assert stale_accounting["ar_output_tokens_in_cycles"] == 0
    assert stale_accounting["first_fallback_position"] is None
    assert stale_accounting["ar_step_reason_counts"] == {}
    assert "mtp_ar_output_tokens" not in (stale.telemetry.timing or {})

    harness.adapter.note_speculative_ar_commit(REQUEST_ID, "no_provider")
    token = GeneratedToken(
        REQUEST_ID,
        step.token_id,
        finished=False,
        stream_chunk=stale,
    )
    (decorated,) = harness.runner.decorate_autoregressive_tokens((token,))

    # Everything the decode produced is the decode's record and is untouched.
    assert decorated.request_id == token.request_id
    assert decorated.token_id == token.token_id
    assert decorated.finished is token.finished
    chunk = decorated.stream_chunk
    assert chunk is not None
    assert chunk.text == stale.text
    assert chunk.token_logprobs == stale.token_logprobs
    assert chunk.finish_details is stale.finish_details
    assert chunk.generated_token_ids is stale.generated_token_ids
    assert chunk.telemetry is not None
    # Only the accounting facts the row owns are refreshed.
    accounting = dict(chunk.telemetry.diagnostics or {})["specdec2_mtp2"]
    assert accounting["mtp_output_tokens"] == 2
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] == 3
    assert accounting["ar_step_reason_counts"] == {"no_provider": 1}
    timing = dict(chunk.telemetry.timing or {})
    assert "mtp_ar_output_tokens" not in timing
    assert "mtp_visible_output_tokens" not in timing
    # The refresh is a merge, not a rebuild: the route facts survive beside it.
    assert chunk.telemetry.diagnostics["stable_model_slot"] == 0
    assert chunk.telemetry.diagnostics["last_width_plan"] == {"groups": 1}


def test_terminal_autoregressive_chunk_is_refreshed_before_reclaim(monkeypatch) -> None:
    """The terminal chunk is the last chance to publish the attribution.

    Reclaim is what removes the row, so the loop must attribute and decorate
    while the row still exists. This asserts both halves: the terminal chunk
    keeps the scheduler-visible finish metadata and gains the refreshed
    accounting, and the same hook after a reclaim has nothing left to refresh.
    """

    _install_clock(monkeypatch)
    harness = _harness(accepted=1)
    _commit_cycle(harness)
    step = _ar_step(400)
    harness.runner._record_step(harness.row, step)

    terminal = harness.runner._stream_chunk(harness.row, step, finished=True)
    assert terminal.finish_details is not None
    harness.adapter.note_speculative_ar_commit(REQUEST_ID, "no_provider")
    token = GeneratedToken(
        REQUEST_ID,
        step.token_id,
        finished=True,
        stream_chunk=terminal,
    )

    (decorated,) = harness.runner.decorate_autoregressive_tokens((token,))

    chunk = decorated.stream_chunk
    assert chunk is not None
    assert decorated.finished is True
    assert chunk.finish_details is terminal.finish_details
    assert chunk.telemetry is not None
    accounting = dict(chunk.telemetry.diagnostics or {})["specdec2_mtp2"]
    assert accounting["ar_output_tokens_in_cycles"] == 1
    assert accounting["first_fallback_position"] == 3
    assert "mtp_ar_output_tokens" not in (chunk.telemetry.timing or {})

    harness.runner._rows.pop(REQUEST_ID)
    (after_reclaim,) = harness.runner.decorate_autoregressive_tokens((token,))
    assert after_reclaim is token


def test_refreshed_autoregressive_chunk_is_idempotent(monkeypatch) -> None:
    """A second refresh with nothing new to say republishes the same token."""

    _install_clock(monkeypatch)
    harness = _harness(accepted=1)
    _commit_cycle(harness)
    step = _ar_step(400)
    harness.runner._record_step(harness.row, step)
    harness.adapter.note_speculative_ar_commit(REQUEST_ID, "no_provider")
    token = GeneratedToken(
        REQUEST_ID,
        step.token_id,
        finished=False,
        stream_chunk=harness.runner._stream_chunk(harness.row, step, finished=False),
    )

    (once,) = harness.runner.decorate_autoregressive_tokens((token,))
    (twice,) = harness.runner.decorate_autoregressive_tokens((once,))

    assert twice is once


def test_plain_autoregressive_chunk_gains_no_speculative_block() -> None:
    """A row without speculative intent publishes exactly what it did before."""

    row = _row()
    runner = _runner(row, _SessionDouble(), None)
    step = _ar_step(400)
    runner._record_step(row, step)
    token = GeneratedToken(
        REQUEST_ID,
        step.token_id,
        finished=False,
        stream_chunk=runner._stream_chunk(row, step, finished=False),
    )

    (decorated,) = runner.decorate_autoregressive_tokens((token,))

    assert decorated is token


@pytest.mark.parametrize(("accepted", "mtp_tokens"), ((1, 2), (0, 1)))
def test_speculative_stream_events_keep_the_accounting_after_reclaim(
    monkeypatch,
    accepted,
    mtp_tokens,
) -> None:
    """A committed cycle's stream chunks carry the block the scheduler omits.

    ``record_speculative_cycle_result`` reclaims a finished request while it
    records that request's final token, so the decoration must work from what
    the hook captured beforehand and never read the row afterwards.
    """

    _install_clock(monkeypatch)
    harness = _harness(accepted=accepted, bonus_token=201)
    _commit_cycle(harness)

    decorate = harness.runner.prepare_speculative_stream_events((REQUEST_ID,))
    assert callable(decorate)
    harness.runner._rows.pop(REQUEST_ID)

    events = (
        _scheduler_event(REQUEST_ID, 200, finished=False, generated=1),
        _scheduler_event(REQUEST_ID, 201, finished=True, generated=2),
    )
    decorated = tuple(decorate(events))

    assert [event.token_id for event in decorated] == [200, 201]
    for event, source in zip(decorated, events, strict=True):
        assert event.request_id == source.request_id
        assert event.finished is source.finished
        assert event.completed is source.completed
        chunk = event.stream_chunk
        assert chunk.text == source.stream_chunk.text
        assert chunk.generated_token_ids == source.stream_chunk.generated_token_ids
        assert chunk.finish_details is source.stream_chunk.finish_details
        assert chunk.telemetry is not None
        # The scheduler counted this token; decoration must not recount it.
        assert (
            chunk.telemetry.decode_state.generated_tokens
            == source.stream_chunk.telemetry.decode_state.generated_tokens
        )
        accounting = dict(chunk.telemetry.diagnostics or {})["specdec2_mtp2"]
        assert accounting["cycles"] == 1
        assert accounting["requested_budget"] == 1
        assert accounting["accepted_draft_tokens"] == accepted
        assert accounting["mtp_output_tokens"] == mtp_tokens
        assert "mtp_visible_output_tokens" not in (chunk.telemetry.timing or {})


def test_plain_autoregressive_stream_events_gain_no_speculative_block() -> None:
    """The stream hook leaves a non-speculative route's chunks alone."""

    row = _row()
    runner = _runner(row, _SessionDouble(), None)
    decorate = runner.prepare_speculative_stream_events((REQUEST_ID,))

    event = _scheduler_event(REQUEST_ID, 400, finished=True, generated=2)
    (decorated,) = decorate((event,))

    assert decorated is event
    assert decorated.stream_chunk.telemetry is not None
    assert decorated.stream_chunk.telemetry.diagnostics is None
    assert decorated.stream_chunk.telemetry.timing is None


def test_stream_hook_for_an_unknown_request_leaves_events_alone() -> None:
    """A request the runner no longer owns has no accounting to add."""

    row = _row()
    runner = _runner(row, _SessionDouble(), None)
    runner.register_speculative_request(REQUEST_ID, 1)
    decorate = runner.prepare_speculative_stream_events((REQUEST_ID + 1,))

    event = _scheduler_event(REQUEST_ID, 400, finished=True, generated=2)
    (decorated,) = decorate((event,))

    assert decorated is event
