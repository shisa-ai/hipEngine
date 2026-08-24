from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from hipengine.generation.deadline import (
    GenerationCancellationToken,
    GenerationCancelled,
)
from hipengine.generation.engine_loop import SubmitPollTextGenerator
from hipengine.generation.engine_service import EngineService
from hipengine.generation.registry import FinishDetails, GenerationOutput, GenerationRequest
from tests.test_specdec2_engine_frontier import _StagedCycleRunner


class _ServiceRunner(_StagedCycleRunner):
    def __init__(self) -> None:
        super().__init__()
        self.capacity = 4
        self._requests = {}
        self._outputs = {}
        self.target_entered = threading.Event()
        self.target_release = threading.Event()

    @property
    def active_request_ids(self):
        return tuple(self._requests)

    def prompt_tokens(self, prompt):
        return (10,)

    def scheduler_max_new_tokens(self, request):
        return int(request.max_tokens)

    def speculative_desired_candidate_count(self, request):
        return min(2, int(request.max_tokens))

    def register_batch(self, request_ids, request, *, prompt_rows):
        for request_id in request_ids:
            self._requests[int(request_id)] = request

    def reclaim(self, completed):
        if completed.finish_reason in {"cancel", "disconnect", "timeout"}:
            return
        self._outputs[int(completed.request_id)] = GenerationOutput(
            text=",".join(str(token) for token in completed.generated_tokens),
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
        output = []
        for request_id in request_ids:
            rid = int(request_id)
            output.append(self._outputs.pop(rid))
            self._requests.pop(rid, None)
        return output

    def discard(self, request_ids):
        for request_id in request_ids:
            self._requests.pop(int(request_id), None)
            self._outputs.pop(int(request_id), None)

    def finalize_batch(self, request, request_ids, outputs):
        return None

    def execute_target_frontier(
        self,
        plan,
        frontier,
        complete_claims,
        *,
        commit,
        cancelled_request_ids=lambda: (),
    ):
        self.target_entered.set()
        assert self.target_release.wait(timeout=2.0)
        return super().execute_target_frontier(
            plan,
            frontier,
            complete_claims,
            commit=commit,
            cancelled_request_ids=cancelled_request_ids,
        )

    def close(self):
        self._requests.clear()
        self._outputs.clear()


class _Inner:
    supports_speculative_mtp = True

    def __init__(self, runner: _ServiceRunner) -> None:
        self.runner = runner
        self.legacy_calls = 0

    def create_resident_model_runner(self, *, capacity):
        self.runner.capacity = int(capacity or self.runner.capacity)
        return self.runner

    def generate_speculative_mtp_detailed(self, request):
        self.legacy_calls += 1
        raise AssertionError("staged EngineService route must not run legacy generation")

    def close(self):
        return None


def _request() -> GenerationRequest:
    return GenerationRequest(
        prompts=("staged",),
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def test_engine_service_staged_submission_returns_before_generation_finishes() -> None:
    runner = _ServiceRunner()
    inner = _Inner(runner)
    adapter = SubmitPollTextGenerator(inner, capacity=1, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    try:
        handle = service.submit_speculative_child(_request())
        assert runner.target_entered.wait(timeout=2.0)

        submission = adapter.last_speculative_submission
        assert submission is not None
        assert submission.execution_route == "engine_service_specdec2"
        assert submission.work_kind == "verify_chain"
        assert submission.work_item is None
        assert not handle.done
        assert inner.legacy_calls == 0
        assert adapter._speculative_outputs_by_request == {}

        runner.target_release.set()
        output = handle.result(timeout=2.0)

        assert output.generated_token_ids == (1, 2, 8000)
        assert output.text == "1,2,8000"
        assert runner.stage_order == [
            "claims",
            "reserve",
            "prepare",
            "propose",
            "target",
            "release",
        ]
        snapshot = service.live_loop_snapshot()["engine_service"]
        assert snapshot["active_children"] == 0
        assert snapshot["last_speculative_route"] == "engine_service_verify_chain"
        assert snapshot["speculative_routes"]["legacy_prelaunch_fallback"] == 0
    finally:
        runner.target_release.set()
        service.close()


def test_staged_cancellation_waits_for_safe_boundary_and_publishes_no_tokens() -> None:
    runner = _ServiceRunner()
    inner = _Inner(runner)
    adapter = SubmitPollTextGenerator(inner, capacity=1, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    token = GenerationCancellationToken()
    request = replace(_request(), max_tokens=2, cancellation_token=token)
    try:
        handle = service.submit_speculative_child(request)
        assert runner.target_entered.wait(timeout=2.0)

        token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        assert token.cancel_requested
        assert not token.cancelled
        runner.target_release.set()

        with pytest.raises(GenerationCancelled):
            handle.result(timeout=2.0)

        assert runner.active_claims is None
        assert runner._outputs == {}
        assert adapter._speculative_outputs_by_request == {}
        assert "target" in runner.stage_order
        assert runner.stage_order[-1] == "release"
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        runner.target_release.set()
        service.close()


def test_staged_blocking_and_streaming_share_committed_ids_and_finish_details() -> None:
    blocking_runner = _ServiceRunner()
    blocking_runner.target_release.set()
    blocking_adapter = SubmitPollTextGenerator(
        _Inner(blocking_runner), capacity=1, prefill_chunk_size=8
    )
    blocking_service = EngineService(blocking_adapter, idle_wait_seconds=0.001)
    streaming_runner = _ServiceRunner()
    streaming_runner.target_release.set()
    streaming_adapter = SubmitPollTextGenerator(
        _Inner(streaming_runner), capacity=1, prefill_chunk_size=8
    )
    streaming_service = EngineService(streaming_adapter, idle_wait_seconds=0.001)
    try:
        blocking = blocking_service.generate_speculative_mtp_detailed(_request())[0]
        chunks = tuple(streaming_service.stream_speculative_mtp_detailed(_request()))

        streamed_ids = tuple(
            token
            for chunk in chunks
            for token in (chunk.generated_token_ids or ())
        )
        terminal = next(
            chunk for chunk in reversed(chunks) if chunk.finish_details is not None
        )
        assert blocking.generated_token_ids == (1, 2, 8000)
        assert streamed_ids == blocking.generated_token_ids
        assert terminal.finish_details == blocking.finish_details
        assert terminal.finish_details is not None
        assert terminal.finish_details.reason == "length"
    finally:
        blocking_service.close()
        streaming_service.close()


def test_staged_shutdown_waits_for_cycle_boundary_and_drains_all_owners() -> None:
    runner = _ServiceRunner()
    adapter = SubmitPollTextGenerator(_Inner(runner), capacity=1, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    errors: list[BaseException] = []
    service.submit_speculative_child(_request())
    assert runner.target_entered.wait(timeout=2.0)

    def close_service() -> None:
        try:
            service.close()
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    closer = threading.Thread(target=close_service)
    closer.start()
    assert closer.is_alive()
    runner.target_release.set()
    closer.join(timeout=2.0)

    assert not closer.is_alive()
    assert errors == []
    assert runner.active_claims is None
    assert runner._requests == {}
    assert runner._outputs == {}
    assert adapter._loop.active_count == 0
    assert adapter._loop.pending_count == 0


def test_staged_multi_prompt_children_keep_independent_ids_and_accounting() -> None:
    runner = _ServiceRunner()
    runner.target_release.set()
    adapter = SubmitPollTextGenerator(_Inner(runner), capacity=2, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    request = replace(_request(), prompts=("first", "second"))
    try:
        outputs = service.generate_speculative_mtp_detailed(request)

        assert tuple(output.generated_token_ids for output in outputs) == (
            (1, 2, 8000),
            (101, 102, 8001),
        )
        assert len({plan.request_ids for plan in runner.cycle_plans}) == 1
        assert runner.cycle_plans[0].request_ids == (0, 1)
        assert runner._requests == {}
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_expired_staged_deadline_fails_before_provider_or_target_open() -> None:
    runner = _ServiceRunner()
    runner.target_release.set()
    adapter = SubmitPollTextGenerator(_Inner(runner), capacity=1, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    request = replace(_request(), deadline_at=time.perf_counter() - 1.0)
    try:
        handle = service.submit_speculative_child(request)
        with pytest.raises(GenerationCancelled):
            handle.result(timeout=2.0)

        assert not runner.target_entered.is_set()
        assert runner.stage_order == []
        assert runner.active_claims is None
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_unsupported_staged_sampling_selects_k0_before_provider_mutation() -> None:
    runner = _ServiceRunner()
    runner.target_release.set()
    adapter = SubmitPollTextGenerator(_Inner(runner), capacity=1, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    request = replace(_request(), max_tokens=2, temperature=0.7, top_p=0.9)
    try:
        output = service.generate_speculative_mtp_detailed(request)[0]

        assert output.generated_token_ids == (9000, 9001)
        assert runner.stage_order == []
        assert len(runner.decodes) == 2
        assert runner.active_claims is None
    finally:
        service.close()
