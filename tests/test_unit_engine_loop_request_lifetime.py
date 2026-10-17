"""Request-local lifetime hooks on the resident submit/poll engine loop.

Bounded native work needs to cancel exactly one row without raising through a
shared ``poll`` and without publishing a token that row produced after its
cancellation.  The agreed contract has two optional runner hooks:

* ``bind_request_lifetime(request_ids, request)`` runs right after
  ``register_batch`` and receives the *original* request, including the
  deadline and cancellation token that the scheduler-facing request strips.
* ``request_cancel_reason(request_id)`` returns ``None`` or one of ``cancel``,
  ``disconnect``, ``timeout`` for a row the scheduler still owns.

The loop probes that reason before admission and after prefill, AR decode, and
opaque speculative cycles, then reclaims only that row through the ordinary
scheduler cancel path.  Runners without the hooks keep the previous behaviour.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from hipengine.dispatch import WorkItem
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken
from hipengine.generation.engine_loop import (
    ResidentEngineLoop,
    SubmitPollTextGenerator,
)
from hipengine.generation.registry import GenerationOutput, GenerationRequest
from hipengine.speculative import SpecCycleResult, SpecCycleStage
from tests.test_unit_specdec2_engine_loop import _CycleRunner

CANCEL_REASONS = ("timeout", "disconnect", "cancel")


class _LifetimeRunner:
    """Loop-level native runner that owns request-local lifetime state."""

    def __init__(self) -> None:
        self.prefills: list[WorkItem] = []
        self.decodes: list[WorkItem] = []
        self.reclaimed: list[CompletedRequest] = []
        self.bound: dict[int, GenerationRequest] = {}
        self.reasons: dict[int, str] = {}
        self.probes: list[int] = []
        self.final_step_request_ids: set[int] = set()
        self.on_prefill = None
        self.on_decode = None
        self._generated: dict[int, int] = {}

    def bind_request_lifetime(self, request_ids, request):
        for request_id in request_ids:
            self.bound[int(request_id)] = request

    def request_cancel_reason(self, request_id):
        self.probes.append(int(request_id))
        return self.reasons.get(int(request_id))

    def prefill_batch(self, work, *, commit):
        assert commit
        self.prefills.append(work)
        if self.on_prefill is not None:
            self.on_prefill(work)

    def decode_batch(self, work, *, commit):
        assert commit
        self.decodes.append(work)
        if self.on_decode is not None:
            self.on_decode(work)
        tokens = []
        for request_id in work.request_ids:
            rid = int(request_id)
            count = self._generated.get(rid, 0)
            self._generated[rid] = count + 1
            tokens.append(
                GeneratedToken(
                    rid,
                    9000 + rid * 10 + count,
                    finished=rid in self.final_step_request_ids,
                )
            )
        return tuple(tokens)

    def compact_batch(self, moves):
        return None

    def reclaim(self, completed):
        self.reclaimed.append(completed)


class _CancellingCycleRunner(_CycleRunner):
    """Opaque cycle runner that cancels rows from its own lifetime state."""

    def __init__(self, *, reason: str | None, cancelled_rows: int = 1) -> None:
        super().__init__()
        self.cancel_reason = reason
        self.cancelled_rows = int(cancelled_rows)
        self.lifetime: str | None = None
        self.probes: list[int] = []

    def request_cancel_reason(self, request_id):
        self.probes.append(int(request_id))
        return self.lifetime

    def reclaim(self, completed):
        # A native runner drops its lifetime state when it releases the row.
        if completed.finish_reason in CANCEL_REASONS:
            self.lifetime = None
        return None

    def execute_speculative_cycle(self, plan, *, commit):
        result = super().execute_speculative_cycle(plan, commit=commit)
        # The native path detects the cancellation and rolls back before it
        # reports CANCELLED; the loop must not publish any of these tokens.
        self.lifetime = self.cancel_reason
        rolled_back = replace(
            result.transaction,
            target_open=False,
            provider_open=False,
            target_committed=False,
            provider_committed=False,
            rolled_back=True,
        )
        return SpecCycleResult(
            stage=SpecCycleStage.CANCELLED,
            transaction=rolled_back,
            cancelled_request_ids=plan.request_ids[: self.cancelled_rows],
        )


class _ResidentLifetimeRunner(_CycleRunner):
    """Resident runner covering both the AR and staged speculative submits."""

    def __init__(self) -> None:
        super().__init__()
        self.capacity = 2
        self.registered: list[
            tuple[tuple[int, ...], GenerationRequest, tuple[tuple[int, ...], ...]]
        ] = []
        self.bound: dict[int, GenerationRequest] = {}
        self.reasons: dict[int, str] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._rows: set[int] = set()
        self.closed = False

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._rows))

    def prompt_tokens(self, prompt):
        del prompt
        return (10,)

    def scheduler_max_new_tokens(self, request):
        return int(request.max_tokens)

    def register_batch(self, request_ids, request, *, prompt_rows):
        ids = tuple(int(request_id) for request_id in request_ids)
        self.registered.append(
            (ids, request, tuple(tuple(row) for row in prompt_rows))
        )
        self._rows.update(ids)

    def bind_request_lifetime(self, request_ids, request):
        for request_id in request_ids:
            self.bound[int(request_id)] = request

    def request_cancel_reason(self, request_id):
        return self.reasons.get(int(request_id))

    def reclaim(self, completed):
        self._outputs[int(completed.request_id)] = GenerationOutput(
            text="",
            generated_token_ids=completed.generated_tokens,
            finish_details=completed.finish_details,
        )

    def has_outputs(self, request_ids):
        return all(int(request_id) in self._outputs for request_id in request_ids)

    def missing_outputs(self, request_ids):
        return [
            int(request_id)
            for request_id in request_ids
            if int(request_id) not in self._outputs
        ]

    def take_outputs(self, request_ids):
        outputs = [
            self._outputs.pop(int(request_id)) for request_id in request_ids
        ]
        for request_id in request_ids:
            self._rows.discard(int(request_id))
        return outputs

    def discard(self, request_ids):
        for request_id in request_ids:
            self._rows.discard(int(request_id))
            self._outputs.pop(int(request_id), None)

    def close(self):
        self.closed = True


class _HooklessResidentRunner(_ResidentLifetimeRunner):
    """Runner that predates the optional lifetime hooks."""

    bind_request_lifetime = None
    request_cancel_reason = None


class _ResidentInner:
    supports_speculative_mtp = True

    def __init__(self, runner: _ResidentLifetimeRunner) -> None:
        self.runner = runner

    def create_resident_model_runner(self, *, capacity=None):
        del capacity
        return self.runner


def _request(*, prompts=("hello",), max_tokens=2, deadline_at=None):
    return GenerationRequest(
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        deadline_at=deadline_at,
    )


def test_pending_row_cancelled_before_admission_never_prefills() -> None:
    runner = _LifetimeRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit([10], max_new_tokens=2)
    runner.reasons[request_id] = "timeout"

    events = loop.poll(max_ticks=1)

    assert runner.prefills == []
    assert runner.decodes == []
    assert [event.kind for event in events] == ["completed"]
    completed = events[0].completed
    assert completed is not None
    assert completed.request_id == request_id
    assert completed.finish_reason == "timeout"
    assert completed.generated_tokens == ()
    assert [item.request_id for item in runner.reclaimed] == [request_id]
    assert loop.pending_count == 0
    assert loop.active_count == 0


@pytest.mark.parametrize("reason", ("timeout", "cancel"))
def test_cancellation_during_prompt_work_reclaims_only_that_row(reason: str) -> None:
    runner = _LifetimeRunner()
    loop = ResidentEngineLoop(runner, capacity=2, prefill_chunk_size=8)
    expired = loop.submit([10], max_new_tokens=4)
    neighbor = loop.submit([20], max_new_tokens=1)

    def expire_during_prefill(work):
        if expired in work.request_ids:
            runner.reasons[expired] = reason

    runner.on_prefill = expire_during_prefill

    events = loop.poll(max_ticks=1)

    assert [work.request_ids for work in runner.prefills] == [(expired,)]
    assert [event.request_id for event in events if event.kind == "completed"] == [
        expired
    ]
    assert loop.completed[expired].finish_reason == reason
    assert loop.completed[expired].generated_tokens == ()
    assert loop.active_count == 1

    probes_before = len(runner.probes)
    events = loop.poll(max_ticks=2)

    # The reclaimed row is never probed again; the neighbour prefills, decodes,
    # and publishes its own token without the shared poll raising.
    assert expired not in runner.probes[probes_before:]
    assert [work.request_ids for work in runner.prefills][-1] == (neighbor,)
    assert [event.token_id for event in events if event.kind == "token"] == [
        9000 + neighbor * 10
    ]
    assert loop.completed[neighbor].finish_reason == "length"
    assert loop.active_count == 0


@pytest.mark.parametrize("reason", CANCEL_REASONS)
def test_final_ar_step_cancellation_drops_only_its_token(reason: str) -> None:
    runner = _LifetimeRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    victim = loop.submit([10], max_new_tokens=1)
    neighbor = loop.submit([20], max_new_tokens=1)
    runner.final_step_request_ids = {victim}

    def cancel_during_decode(work):
        if victim in work.request_ids:
            runner.reasons[victim] = reason

    runner.on_decode = cancel_during_decode

    loop.poll(max_ticks=2)  # admit and prefill both prompts
    assert runner.decodes == []

    events = loop.poll(max_ticks=1)  # one AR decode covering both rows

    assert [work.request_ids for work in runner.decodes] == [(victim, neighbor)]
    assert [event.token_id for event in events if event.kind == "token"] == [
        9000 + neighbor * 10
    ]
    assert loop.completed[victim].finish_reason == reason
    assert loop.completed[victim].generated_tokens == ()
    assert loop.completed[neighbor].generated_tokens == (9000 + neighbor * 10,)
    assert [event.request_id for event in events if event.kind == "completed"] == [
        neighbor,
        victim,
    ]
    assert loop.active_count == 0


@pytest.mark.parametrize("reason", CANCEL_REASONS)
def test_opaque_cancelled_cycle_reclaims_with_the_native_reason(reason: str) -> None:
    runner = _CancellingCycleRunner(reason=reason)
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=4,
        desired_candidate_count=2,
    )

    events = loop.poll(max_ticks=2)

    assert len(runner.cycle_plans) == 1
    assert [event.token_id for event in events if event.kind == "token"] == []
    assert loop.completed[request_id].finish_reason == reason
    assert loop.completed[request_id].generated_tokens == ()
    assert loop.active_count == 0


def test_opaque_cancelled_cycle_without_reason_falls_back_to_cancel() -> None:
    runner = _CancellingCycleRunner(reason=None)
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=4,
        desired_candidate_count=2,
    )

    events = loop.poll(max_ticks=2)

    assert [event.token_id for event in events if event.kind == "token"] == []
    assert loop.completed[request_id].finish_reason == "cancel"


def test_opaque_cancelled_cycle_reclaims_only_the_rows_the_result_owns() -> None:
    runner = _CancellingCycleRunner(reason="timeout", cancelled_rows=1)
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    cancelled = loop.submit_speculative(
        [10],
        max_new_tokens=4,
        desired_candidate_count=1,
    )
    survivor = loop.submit_speculative(
        [20],
        max_new_tokens=4,
        desired_candidate_count=1,
    )

    events = loop.poll(max_ticks=3)

    assert runner.cycle_plans[-1].request_ids == (cancelled, survivor)
    assert [event.request_id for event in events if event.kind == "completed"] == [
        cancelled
    ]
    assert loop.completed[cancelled].finish_reason == "timeout"
    assert loop.completed.get(survivor) is None
    assert loop.active_count == 1


def test_committed_cycle_with_pending_cancellation_still_raises() -> None:
    # The contract guard is unchanged: a runner that commits despite the
    # loop's own pending-cancellation probe is a contract violation, not a row
    # the loop may silently publish.
    runner = _CycleRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    loop.submit_speculative(
        [10],
        max_new_tokens=4,
        desired_candidate_count=2,
        cancel_requested=lambda: True,
    )

    with pytest.raises(
        RuntimeError,
        match="committed speculative output despite pending cancellation",
    ):
        loop.poll(max_ticks=2)


def test_ar_submission_binds_the_original_request_after_register_batch() -> None:
    runner = _ResidentLifetimeRunner()
    driver = SubmitPollTextGenerator(_ResidentInner(runner), capacity=2)
    deadline = time.monotonic() + 60.0
    request = _request(deadline_at=deadline)
    try:
        submission = driver.submit_detailed(request)
        (request_id,) = submission.request_ids
        (registered_ids, registered_request, _rows) = runner.registered[-1]
        assert registered_ids == (request_id,)
        assert registered_request.deadline_at is None
        bound = runner.bound[request_id]
        assert bound.deadline_at == deadline
        assert bound.prompts == ("hello",)
    finally:
        driver.close()


def test_staged_speculative_submission_binds_the_original_request() -> None:
    runner = _ResidentLifetimeRunner()
    driver = SubmitPollTextGenerator(_ResidentInner(runner), capacity=2)
    deadline = time.monotonic() + 60.0
    request = _request(deadline_at=deadline)
    try:
        (submission,) = driver.submit_speculative_many_detailed((request,))
        (request_id,) = submission.request_ids
        assert submission.execution_route == "engine_service_specdec2"
        assert runner.registered[-1][1].deadline_at is None
        assert runner.bound[request_id] is request
        assert runner.bound[request_id].deadline_at == deadline
    finally:
        driver.close()


def test_runner_without_lifetime_hooks_keeps_scheduler_only_contract() -> None:
    runner = _HooklessResidentRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit([10], max_new_tokens=2)

    events = loop.poll(max_ticks=3)

    assert [event.token_id for event in events if event.kind == "token"] == [
        9000,
        9001,
    ]
    assert loop.completed[request_id].generated_tokens == (9000, 9001)


def test_cancelled_row_does_not_poison_the_next_request() -> None:
    runner = _LifetimeRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    victim = loop.submit([10], max_new_tokens=2)
    runner.reasons[victim] = "timeout"

    assert [event.kind for event in loop.poll(max_ticks=1)] == ["completed"]
    assert loop.completed[victim].finish_reason == "timeout"
    assert loop.release_completed(victim) is not None

    replacement = loop.submit([30], max_new_tokens=1)
    events = loop.poll(max_ticks=3)

    assert replacement != victim
    assert [event.token_id for event in events if event.kind == "token"] == [
        9000 + replacement * 10
    ]
    assert loop.completed[replacement].finish_reason == "length"
    assert loop.active_count == 0
