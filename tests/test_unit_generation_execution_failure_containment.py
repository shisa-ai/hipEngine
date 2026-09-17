"""Execution failure containment: one request's failure must not close the service.

The contract has three parts.  A runner may classify one failed execution step
as containable by returning ``ExecutionFailure`` after it has quiesced the
device and proven the named rows reclaimable; the loop then fails exactly those
rows and keeps scheduling everything else.  Anything a runner cannot prove stays
fatal, and a fatal failure marks the service unhealthy with a legible reason
instead of blaming the client's memory budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipError
from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation import (
    EngineService,
    ExecutionFailure,
    GenerationCancelled,
    GenerationExecutionFailed,
    GenerationOutput,
    GenerationRequest,
    GenerationSubmission,
    GeneratedToken,
    GenerationStreamChunk,
    SubmitPollTextGenerator,
)
from hipengine.generation.engine_loop import EngineLoopEvent, ResidentEngineLoop


class _ContainingLoopRunner:
    """Minimal resident runner whose prefill/decode failures can be contained."""

    packed_prefill_max_rows = 3

    def __init__(self) -> None:
        self.decodes: list[tuple[int, ...]] = []
        self.prefills: list[tuple[int, ...]] = []
        self.counts: dict[int, int] = {}
        self.targets: dict[int, int] = {}
        self.contain = True
        self.fail_decode: set[int] = set()
        self.fail_prefill: set[int] = set()
        self.scope: tuple[int, ...] | None = None
        self.mutation = "none"
        self.recover_raises = False

    def register(self, request_id: int, *, target: int) -> None:
        self.targets[int(request_id)] = int(target)

    def prefill_batch(self, work, *, commit):
        assert commit is True
        self.prefills.append(tuple(int(value) for value in work.request_ids))
        for request_id in work.request_ids:
            if int(request_id) in self.fail_prefill:
                raise RuntimeError(f"injected prefill failure for row {request_id}")

    def decode_batch(self, work, *, commit):
        assert commit is True
        self.decodes.append(tuple(int(value) for value in work.request_ids))
        for request_id in work.request_ids:
            if int(request_id) in self.fail_decode:
                raise RuntimeError(f"injected decode failure for row {request_id}")
        output = []
        for request_id in work.request_ids:
            request_id = int(request_id)
            count = self.counts.get(request_id, 0) + 1
            self.counts[request_id] = count
            output.append(
                GeneratedToken(
                    request_id,
                    5000 + request_id * 100 + count,
                    finished=count >= self.targets.get(request_id, 1),
                    stream_chunk=GenerationStreamChunk(text=f"row{request_id}:{count}"),
                )
            )
        return tuple(output)

    def compact_batch(self, moves):
        return None

    def reclaim(self, completed):
        return None

    def contain_execution_failure(self, error, *, phase, request_ids, work_kind):
        if not self.contain:
            return None
        if self.recover_raises:
            raise RuntimeError("containment recovery failed")
        return ExecutionFailure(
            request_ids=self.scope if self.scope is not None else tuple(request_ids),
            phase=phase,
            work_kind=work_kind,
            mutation=self.mutation,
            reason=f"{type(error).__name__}: {error}",
            error=error,
        )


def _drive_until_idle(loop: ResidentEngineLoop, *, ticks: int = 8) -> list[EngineLoopEvent]:
    events: list[EngineLoopEvent] = []
    for _ in range(ticks):
        events.extend(loop.poll(max_ticks=1))
    return events


def _failed_events(events: list[EngineLoopEvent]) -> list[EngineLoopEvent]:
    return [event for event in events if event.kind == "failed"]


def test_decode_failure_is_contained_to_the_named_row_and_peers_finish() -> None:
    runner = _ContainingLoopRunner()
    loop = ResidentEngineLoop(runner, capacity=3, prefill_chunk_size=8)
    request_ids = [loop.submit([10 + index], max_new_tokens=2) for index in range(3)]
    for request_id in request_ids:
        runner.register(request_id, target=2)
    runner.scope = (request_ids[1],)
    runner.fail_decode = {request_ids[1]}

    events = _drive_until_idle(loop)

    failed = _failed_events(events)
    assert [event.request_id for event in failed] == [request_ids[1]]
    error = failed[0].error
    assert isinstance(error, GenerationExecutionFailed)
    assert error.phase == "decode"
    assert error.request_ids == (request_ids[1],)
    assert error.work_kind == WorkKind.DECODE.value
    assert error.mutation == "none"
    assert error.cause_type == "RuntimeError"
    assert "injected decode failure" in str(error)
    # The packed group ran once with all three rows; only the named row died,
    # and no later decode ever includes it.
    assert runner.decodes[0] == tuple(request_ids)
    assert all(request_ids[1] not in group for group in runner.decodes[1:])
    survivors = [event for event in events if event.kind == "completed"]
    assert sorted(event.request_id for event in survivors) == [
        request_ids[0],
        request_ids[2],
    ]
    assert loop.scheduler.active_count == 0
    assert loop.scheduler.pending_count == 0

    # The same loop keeps serving: a later request completes normally.
    runner.fail_decode = set()
    later = loop.submit([77], max_new_tokens=2)
    runner.register(later, target=2)
    later_events = _drive_until_idle(loop)
    assert [event.request_id for event in later_events if event.kind == "completed"] == [
        later
    ]
    assert _failed_events(later_events) == []


def test_prefill_failure_contained_for_a_packed_group_leaves_other_work_serving() -> None:
    runner = _ContainingLoopRunner()
    loop = ResidentEngineLoop(runner, capacity=4, prefill_chunk_size=8)
    first = loop.submit([1], max_new_tokens=2)
    second = loop.submit([2], max_new_tokens=2)
    for request_id in (first, second):
        runner.register(request_id, target=2)
    runner.fail_prefill = {first, second}

    events = _drive_until_idle(loop)

    failed = _failed_events(events)
    assert sorted(event.request_id for event in failed) == [first, second]
    assert {event.error.phase for event in failed} == {"prefill"}
    assert {event.error.mutation for event in failed} == {"none"}
    # Both rows of the packed prefill group are gone, and the loop is empty.
    assert runner.prefills == [(first, second)]
    assert loop.scheduler.active_count == 0

    # An unrelated request admitted afterwards is unaffected.
    runner.fail_prefill = set()
    third = loop.submit([3], max_new_tokens=2)
    runner.register(third, target=2)
    later_events = _drive_until_idle(loop)
    assert [event.request_id for event in later_events if event.kind == "completed"] == [
        third
    ]


def test_unclassified_execution_failure_stays_fatal() -> None:
    runner = _ContainingLoopRunner()
    runner.contain = False
    runner.fail_decode = {0}
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    loop.submit([10], max_new_tokens=2)

    with pytest.raises(RuntimeError, match="injected decode failure"):
        _drive_until_idle(loop)


def test_containment_scope_outside_the_failed_work_is_refused() -> None:
    runner = _ContainingLoopRunner()
    runner.scope = (99,)
    runner.fail_decode = {0}
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    loop.submit([10], max_new_tokens=2)

    with pytest.raises(ValueError, match="outside the failed work"):
        _drive_until_idle(loop)


def test_a_recovery_that_cannot_report_its_scope_is_fatal() -> None:
    runner = _ContainingLoopRunner()
    runner.recover_raises = True
    runner.fail_decode = {0}
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    loop.submit([10], max_new_tokens=2)

    with pytest.raises(RuntimeError, match="injected decode failure"):
        _drive_until_idle(loop)


def test_containment_must_return_the_declared_failure_object() -> None:
    runner = _ContainingLoopRunner()
    runner.contain_execution_failure = lambda *args, **kwargs: "contained"
    runner.fail_decode = {0}
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    loop.submit([10], max_new_tokens=2)

    with pytest.raises(TypeError, match="must return ExecutionFailure"):
        _drive_until_idle(loop)


class _ContainingResidentRunner:
    """Service-facing runner: one injected row failure, everything else normal."""

    capacity = 3

    def __init__(self) -> None:
        self.targets: dict[int, int] = {}
        self.counts: dict[int, int] = {}
        self.outputs: dict[int, GenerationOutput] = {}
        self.fail_decode: set[int] = set()
        self.cancel_decode: set[int] = set()
        self.failed_row: int | None = None
        self.contain = True

    def prompt_tokens(self, prompt):
        return tuple(int(token) for token in prompt)

    def scheduler_max_new_tokens(self, request):
        return int(request.max_tokens)

    def register_batch(self, request_ids, request, *, prompt_rows):
        del request
        for request_id, prompt_row in zip(request_ids, prompt_rows, strict=True):
            self.targets[int(request_id)] = int(prompt_row[0])

    def prefill_batch(self, work, *, commit):
        assert commit is True

    def decode_batch(self, work, *, commit):
        assert commit is True
        generated = []
        for request_id in work.request_ids:
            rid = int(request_id)
            if rid in self.fail_decode:
                self.failed_row = rid
                raise RuntimeError(f"packed sampler row {rid} exceeds capacity 0")
            if rid in self.cancel_decode:
                self.failed_row = rid
                raise GenerationCancelled()
            count = self.counts.get(rid, 0) + 1
            self.counts[rid] = count
            generated.append(
                GeneratedToken(
                    rid,
                    3000 + rid * 100 + count,
                    finished=count >= self.targets[rid],
                    stream_chunk=GenerationStreamChunk(text=f"row{rid}:{count}"),
                )
            )
        return tuple(generated)

    def contain_execution_failure(self, error, *, phase, request_ids, work_kind):
        if not self.contain:
            return None
        rows = tuple(int(request_id) for request_id in request_ids)
        affected = (
            (int(self.failed_row),)
            if self.failed_row is not None and int(self.failed_row) in rows
            else rows
        )
        return ExecutionFailure(
            request_ids=affected,
            phase=phase,
            work_kind=work_kind,
            mutation="none",
            reason=f"{type(error).__name__}: {error}",
            error=error,
        )

    def compact_batch(self, moves):
        del moves

    def reclaim(self, completed):
        rid = int(completed.request_id)
        self.outputs[rid] = GenerationOutput(
            text=f"done:{rid}",
            generated_token_ids=completed.generated_tokens,
            finish_details=completed.finish_details,
        )

    def has_outputs(self, request_ids):
        return all(int(request_id) in self.outputs for request_id in request_ids)

    def missing_outputs(self, request_ids):
        return [
            int(request_id)
            for request_id in request_ids
            if int(request_id) not in self.outputs
        ]

    def take_outputs(self, request_ids):
        return [self.outputs.pop(int(request_id)) for request_id in request_ids]

    def discard(self, request_ids):
        for request_id in request_ids:
            self.outputs.pop(int(request_id), None)

    def close(self):
        pass


class _ContainingInner:
    def __init__(self) -> None:
        self.runner = _ContainingResidentRunner()

    def create_resident_model_runner(self, *, capacity):
        assert capacity in {None, 3}
        return self.runner


def _service_request(prompt_token: int, *, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        prompts=((prompt_token,),),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def test_a_contained_cancellation_is_reported_as_a_cancellation() -> None:
    """A step that observed a cancel is a request outcome, not an engine fault."""

    from hipengine.generation import GenerationCancelled

    class _CancellingRunner(_ContainingLoopRunner):
        def decode_batch(self, work, *, commit):
            self.decodes.append(tuple(int(value) for value in work.request_ids))
            raise GenerationCancelled()

    runner = _CancellingRunner()
    loop = ResidentEngineLoop(runner, capacity=3, prefill_chunk_size=8)
    request_ids = [loop.submit([10 + index], max_new_tokens=2) for index in range(3)]
    for request_id in request_ids:
        runner.register(request_id, target=2)
    runner.scope = (request_ids[0],)

    events = _drive_until_idle(loop, ticks=2)

    failed = _failed_events(events)
    assert [event.request_id for event in failed] == [request_ids[0]]
    assert isinstance(failed[0].error, GenerationCancelled)
    assert not isinstance(failed[0].error, GenerationExecutionFailed)


def test_service_contains_one_child_failure_and_keeps_serving() -> None:
    inner = _ContainingInner()
    service = EngineService(
        SubmitPollTextGenerator(inner, capacity=3, prefill_chunk_size=4)
    )
    try:
        handles = service.submit_children(
            [_service_request(2), _service_request(3)]
        )
        inner.runner.fail_decode = {handles[1].request_id}

        healthy = handles[0].result()
        with pytest.raises(GenerationExecutionFailed) as failure:
            handles[1].result()

        assert healthy.generated_token_ids
        error = failure.value
        assert error.phase == "decode"
        assert error.request_ids == (handles[1].request_id,)
        assert "packed sampler row" in str(error)
        assert service.health()["status"] == "ok"

        # The service is still usable: a new request completes on the same driver.
        inner.runner.fail_decode = set()
        follow_up = service.submit_child(_service_request(4))
        assert follow_up.result().generated_token_ids
        assert service.health()["status"] == "ok"
    finally:
        service.close()
    assert service.health()["status"] == "closed"


def test_service_reports_a_contained_cancellation_as_a_cancellation() -> None:
    """A client cancel observed in a step must not read as an engine failure."""

    from hipengine.generation import GenerationCancelled

    inner = _ContainingInner()
    service = EngineService(
        SubmitPollTextGenerator(inner, capacity=3, prefill_chunk_size=4)
    )
    try:
        handles = service.submit_children(
            [_service_request(2), _service_request(3)]
        )
        inner.runner.cancel_decode = {handles[1].request_id}

        assert handles[0].result().generated_token_ids
        with pytest.raises(GenerationCancelled):
            handles[1].result()
        assert service.health()["status"] == "ok"

        inner.runner.cancel_decode = set()
        assert service.submit_child(_service_request(4)).result().generated_token_ids
    finally:
        service.close()


def test_service_fatal_execution_failure_reports_unhealthy_reason() -> None:
    inner = _ContainingInner()
    inner.runner.contain = False
    service = EngineService(
        SubmitPollTextGenerator(inner, capacity=3, prefill_chunk_size=4)
    )
    try:
        handle = service.submit_child(_service_request(2))
        inner.runner.fail_decode = {handle.request_id}
        with pytest.raises(RuntimeError, match="packed sampler row"):
            handle.result()

        health = service.health()
        assert health["status"] == "unhealthy"
        assert health["serving"] is False
        unhealthy = health["unhealthy"]
        assert unhealthy["exception_type"] == "RuntimeError"
        assert "packed sampler row" in unhealthy["message"]
        assert unhealthy["location"] is not None
        assert "test_unit_generation_execution_failure_containment.py" in (
            unhealthy["location"]
        )

        with pytest.raises(RuntimeError) as closed:
            service.submit_child(_service_request(4))
        message = str(closed.value)
        assert message.startswith("engine service is closed after a fatal execution failure")
        assert "packed sampler row" in message
        assert "--max-context-tokens" not in message
    finally:
        service.close()
    # A recorded fatal reason outlives the shutdown that follows it.
    assert service.health()["status"] == "unhealthy"


def test_containment_does_not_hide_a_rejected_admission_error() -> None:
    """A refusal before execution keeps its admission-specific diagnostic."""

    inner = _ContainingInner()
    service = EngineService(
        SubmitPollTextGenerator(inner, capacity=3, prefill_chunk_size=4)
    )
    try:
        handle = service.submit_child(_service_request(2))
        assert handle.result().generated_token_ids
    finally:
        service.close()
    assert service.health()["status"] == "closed"
    assert service.health()["unhealthy"] is None


class _ContainmentSession:
    """Device session stub: no kernels, one recorded synchronization."""

    def __init__(self, slot_id: int) -> None:
        self.slot_id = int(slot_id)
        self.scratch = SimpleNamespace(max_positions=1024)
        self.position = 0
        self.allocation = None
        self.pool = None
        self.prefill_calls: list[tuple[tuple[int, ...], int, int]] = []
        self.step_calls: list[tuple[int, int, int]] = []
        self.runtime = _ContainmentRuntime()

    def prefill(self, token_ids, *, return_logits: bool):
        prompt = tuple(int(token) for token in token_ids)
        start = int(self.position)
        self.position += len(prompt)
        self.prefill_calls.append((prompt, start, int(self.position)))
        return self._result(return_logits=return_logits)

    def prefill_batch_native(self, prompt_token_ids, *, sessions, **kwargs):
        assert sessions == [self]
        return [self.prefill(prompt_token_ids[0], return_logits=bool(kwargs.get("return_logits", False)))]

    def step(self, token_id: int, *, return_logits: bool):
        start = int(self.position)
        self.position += 1
        self.step_calls.append((int(token_id), start, int(self.position)))
        return self._result(return_logits=return_logits)

    def invalidate_device_kv_graphs(self) -> int:
        return 0

    def unbind_device_kv_allocation(self):
        allocation = self.allocation
        self.allocation = None
        self.pool = None
        return allocation

    def reset(self) -> None:
        self.position = 0

    def close(self) -> None:
        pass

    @staticmethod
    def _result(*, return_logits: bool):
        return SimpleNamespace(token_id=777, logits=None)


class _ContainmentRuntime:
    def __init__(self) -> None:
        self.synchronize_calls = 0
        self.fail_synchronize = False

    def device_synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.fail_synchronize:
            raise HipError(719, "unspecified launch failure")

    def mem_get_info(self) -> tuple[int, int]:
        return (100, 200)


class _ContainmentOwner:
    """Minimal host owner for the real GGUF resident runner."""

    backend = "hip_gfx1151"
    target_arch = "gfx1151"
    _prepared_max_sequence_length = 1024
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        decode=lambda tokens: "".join(str(token) for token in tokens),
    )

    def __init__(self) -> None:
        self.sessions = [_ContainmentSession(index) for index in range(3)]
        self.runtime = _ContainmentRuntime()
        # The real owner exposes the shared full-stack runner, whose runtime is
        # the one a fallback step uses; mirror that private name here.
        self.shared = SimpleNamespace(_runtime=self.runtime)

    def _get_shared_runner(self):
        return self.shared

    def _acquire_shared_session(self, shared_runner, **kwargs):
        del shared_runner, kwargs
        return self.sessions.pop(0), ("continuous_ar_dynamic_kv", True, True, 1024), False

    def _release_shared_session(self, key, session) -> None:
        del key
        self.sessions.append(session)

    def _flush_ar_packed_decode_owners(self, slots) -> None:
        del slots


def _resident_runner_with_row(
    *,
    cancelled: bool = False,
) -> tuple[object, object, int]:
    from hipengine.generation.deadline import GenerationCancellationToken
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    owner = _ContainmentOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    token = GenerationCancellationToken()
    if cancelled:
        token.cancel()
    request = GenerationRequest(
        prompts=((11, 12),),
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        cancellation_token=token,
    )
    runner.register_batch((5,), request, prompt_rows=((11, 12),))
    runner.reserve_admission(SimpleNamespace(request_id=5))
    return runner, owner, 5


def test_real_runner_contains_only_a_pre_device_validation_failure() -> None:
    """The production classifier contains the prologue, never a device step."""

    runner, owner, request_id = _resident_runner_with_row(cancelled=True)
    decode = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(request_id,),
        row_to_request=(request_id,),
    )

    with pytest.raises(GenerationCancelled):
        runner.decode_batch(decode, commit=True)

    # The prologue failed before any device call, so containment is provable and
    # narrowed to the row whose cancellation was observed.
    assert runner._execution_mutation_window == "none"
    assert runner._execution_step_request_id == request_id
    failure = runner.contain_execution_failure(
        GenerationCancelled(),
        phase="decode",
        request_ids=(request_id,),
        work_kind="decode",
    )
    assert isinstance(failure, ExecutionFailure)
    assert failure.request_ids == (request_id,)
    assert failure.mutation == "none"
    assert failure.phase == "decode"
    # Quiescing ran on the owner's device runtime before the scope was returned.
    assert owner.runtime.synchronize_calls == 1

    # A failure at or after the first device call is never contained: the step
    # may have advanced rows the scheduler has not recorded.
    runner._execution_mutation_window = "unknown"
    assert (
        runner.contain_execution_failure(
            RuntimeError("packed sampler row 0 exceeds capacity 0"),
            phase="decode",
            request_ids=(request_id,),
            work_kind="decode",
        )
        is None
    )


def test_real_runner_refuses_containment_it_cannot_prove() -> None:
    runner, owner, request_id = _resident_runner_with_row()
    runner._execution_mutation_window = "none"

    # A device-side error means the context is in an unknown state.
    assert (
        runner.contain_execution_failure(
            HipError(719, "unspecified launch failure"),
            phase="decode",
            request_ids=(request_id,),
            work_kind="decode",
        )
        is None
    )
    # So does a quiesce that itself reports a device error.
    owner.runtime.fail_synchronize = True
    assert (
        runner.contain_execution_failure(
            RuntimeError("host-side validation"),
            phase="decode",
            request_ids=(request_id,),
            work_kind="decode",
        )
        is None
    )
    owner.runtime.fail_synchronize = False
    # The speculative phases keep their own state-aware recovery verdict.
    assert (
        runner.contain_execution_failure(
            RuntimeError("host-side validation"),
            phase="speculative_cycle",
            request_ids=(request_id,),
            work_kind="verify_chain",
        )
        is None
    )
    # A phase that never marks a pre-device window is never contained, even if
    # the last prefill/decode step left the window at "none".
    runner._execution_mutation_window = "none"
    for phase in ("speculative_prepare", "speculative_cycle", "compact"):
        assert (
            runner.contain_execution_failure(
                RuntimeError("host-side validation"),
                phase=phase,
                request_ids=(request_id,),
                work_kind="verify_chain",
            )
            is None
        )
    # An empty scope is not a containment.
    assert (
        runner.contain_execution_failure(
            RuntimeError("host-side validation"),
            phase="decode",
            request_ids=(),
            work_kind="decode",
        )
        is None
    )
