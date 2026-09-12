from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from hipengine import LLM
from hipengine.generation import (
    EngineLoopEvent,
    EngineService,
    FinishDetails,
    GeneratedToken,
    GenerationAdmissionRejected,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationSubmission,
    SubmitPollTextGenerator,
    register_text_generator,
)
from hipengine.server.api import SamplingParams, _GenerationBatcher, _QueuedBatchResult


def _request(
    prompt: str,
    *,
    max_tokens: int = 8,
    cancellation_token=None,
) -> GenerationRequest:
    target = int(prompt.rsplit(":", 1)[1]) if ":" in prompt else 0
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=max(int(max_tokens), target),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        cancellation_token=cancellation_token,
    )


@dataclass
class _DriverState:
    submission: GenerationSubmission
    prompt: str
    target_tokens: int
    tokens: list[int]


class _FakeSoleDriver:
    supports_controlled_streaming = True
    supports_stream_many = True
    supports_speculative_mtp = False

    def __init__(self) -> None:
        self._next_request_id = 1
        self._active: dict[int, _DriverState] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self.submitted_prompt_groups: list[tuple[str, ...]] = []
        self.poll_thread_ids: set[int] = set()
        self.release_order: list[int] = []
        self.abort_reasons: dict[int, str] = {}
        self.reconfigurations: list[tuple[int, object]] = []
        self.closed = False

    def submit_detailed(self, request: GenerationRequest) -> GenerationSubmission:
        assert len(request.prompts) == 1
        prompt = str(request.prompts[0])
        self.submitted_prompt_groups.append(tuple(str(item) for item in request.prompts))
        request_id = self._next_request_id
        self._next_request_id += 1
        target = int(prompt.rsplit(":", 1)[1]) if ":" in prompt else int(request.max_tokens)
        submission = GenerationSubmission(
            request_ids=(request_id,),
            request=request,
            max_ticks=max(4, target + 4),
        )
        self._active[request_id] = _DriverState(submission, prompt, target, [])
        return submission

    def poll(self, *, max_ticks: int = 1):
        assert max_ticks == 1
        self.poll_thread_ids.add(threading.get_ident())
        events: list[EngineLoopEvent] = []
        for request_id in tuple(sorted(self._active)):
            state = self._active[request_id]
            token_index = len(state.tokens)
            token_id = 1000 + request_id * 100 + token_index
            state.tokens.append(token_id)
            finish = len(state.tokens) >= state.target_tokens
            finish_details = None
            if finish:
                finish_details = (
                    FinishDetails(reason="stop", stop_sequence=tuple(state.tokens))
                    if state.prompt.startswith("stop:")
                    else FinishDetails(reason="length")
                )
            chunk = GenerationStreamChunk(
                text=f"{state.prompt}:{token_index}",
                finish_details=finish_details,
                generated_token_ids=tuple(state.tokens),
                telemetry={"decode_state": {"row_index": 0}},
            )
            events.append(
                EngineLoopEvent(
                    kind="token",
                    request_id=request_id,
                    request_ids=(request_id,),
                    token_id=token_id,
                    stream_chunk=chunk,
                )
            )
            if finish:
                self._outputs[request_id] = GenerationOutput(
                    text="".join(f"{state.prompt}:{index}" for index in range(state.target_tokens)),
                    finish_details=finish_details,
                    generated_token_ids=tuple(state.tokens),
                )
                events.append(
                    EngineLoopEvent(
                        kind="completed",
                        request_id=request_id,
                        request_ids=(request_id,),
                    )
                )
        return tuple(events)

    def generation_complete(self, submission: GenerationSubmission) -> bool:
        return all(request_id in self._outputs for request_id in submission.request_ids)

    def take_result(self, submission: GenerationSubmission) -> list[GenerationOutput]:
        outputs = [self._outputs.pop(request_id) for request_id in submission.request_ids]
        for request_id in submission.request_ids:
            self._active.pop(request_id, None)
            self.release_order.append(request_id)
        return outputs

    def abort_submission(self, submission: GenerationSubmission, *, reason: str = "cancel") -> None:
        for request_id in submission.request_ids:
            self._active.pop(request_id, None)
            self._outputs.pop(request_id, None)
            self.abort_reasons[request_id] = str(reason)
            self.release_order.append(request_id)

    def live_loop_snapshot(self):
        return {"active_request_ids": sorted(self._active)}

    def reconfigure_engine_loop(self, config: object) -> None:
        if self._active:
            raise RuntimeError("driver must be idle")
        self.reconfigurations.append((threading.get_ident(), config))

    def compact(self, order=None):
        return ((threading.get_ident(), None if order is None else tuple(order)),)

    def close(self) -> None:
        self.closed = True
        self._active.clear()
        self._outputs.clear()


def test_engine_service_serializes_idle_reconfiguration_on_driver_thread() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=8, idle_wait_seconds=0.001)
    config = object()
    try:
        service.reconfigure_engine_loop(config)
    finally:
        service.close()

    assert driver.reconfigurations == [(service.driver_thread_id, config)]


def test_engine_service_serializes_compaction_on_driver_thread() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=8, idle_wait_seconds=0.001)
    try:
        moves = service.compact((3, 1))
    finally:
        service.close()

    assert moves == ((service.driver_thread_id, (3, 1)),)


def test_engine_service_speculative_submission_uses_shared_child_lifecycle() -> None:
    class SpeculativeDriver(_FakeSoleDriver):
        supports_speculative_mtp = True

        def __init__(self) -> None:
            super().__init__()
            self.speculative_submit_thread_ids: list[int] = []
            self.legacy_calls = 0
            self.speculative_submissions: list[GenerationSubmission] = []

        def submit_speculative_detailed(self, request: GenerationRequest) -> GenerationSubmission:
            self.speculative_submit_thread_ids.append(threading.get_ident())
            request_id = self._next_request_id
            self._next_request_id += 1
            submission = GenerationSubmission(
                request_ids=(request_id,),
                request=request,
                max_ticks=1,
                work_kind="verify_chain",
                execution_route="engine_service_speculative",
            )
            self.speculative_submissions.append(submission)
            self._outputs[request_id] = GenerationOutput(
                text="spec",
                generated_token_ids=(700, 701),
            )
            return submission

        def generate_speculative_mtp_detailed(self, request: GenerationRequest):
            self.legacy_calls += 1
            raise AssertionError("legacy fallback must not run")

    driver = SpeculativeDriver()
    service = EngineService(driver)
    try:
        outputs = service.generate_speculative_mtp_detailed(_request("spec:2"))
        snapshot = service.live_loop_snapshot()["engine_service"]

        assert outputs[0].generated_token_ids == (700, 701)
        assert driver.legacy_calls == 0
        assert driver.speculative_submit_thread_ids == [service.driver_thread_id]
        assert driver.speculative_submissions[0].work_kind == "verify_chain"
        assert driver.release_order == [1]
        assert snapshot["active_children"] == 0
        assert snapshot["speculative_routes"] == {
            "engine_service_verify_chain": 1,
            "legacy_prelaunch_fallback": 0,
        }
    finally:
        service.close()


def test_engine_service_batches_compatible_speculative_children_once() -> None:
    class PackedSpecDriver(_FakeSoleDriver):
        supports_speculative_mtp = True

        def __init__(self) -> None:
            super().__init__()
            self.batch_calls: list[tuple[str, ...]] = []

        def submit_speculative_many_detailed(self, requests):
            self.batch_calls.append(tuple(str(request.prompts[0]) for request in requests))
            request_ids = tuple(range(self._next_request_id, self._next_request_id + len(requests)))
            self._next_request_id += len(requests)
            submissions = []
            for request_id, request in zip(request_ids, requests, strict=True):
                submission = GenerationSubmission(
                    request_ids=(request_id,), request=request, max_ticks=1,
                    work_kind="verify_chain", execution_route="packed_speculative",
                )
                self._outputs[request_id] = GenerationOutput(
                    text=f"packed:{request_id}", generated_token_ids=(900 + request_id,)
                )
                submissions.append(submission)
            return tuple(submissions)

        def submit_speculative_detailed(self, request):
            raise AssertionError("compatible children must use batch submission")

    driver = PackedSpecDriver()
    service = EngineService(driver)
    try:
        handles = service.submit_speculative_children(
            (_request("packed-a:1"), _request("packed-b:1"))
        )
        outputs = tuple(handle.result(timeout=2.0) for handle in handles)

        assert driver.batch_calls == [("packed-a:1", "packed-b:1")]
        assert tuple(output.generated_token_ids for output in outputs) == ((901,), (902,))
        assert driver.release_order == [1, 2]
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_engine_service_mixed_ar_and_speculative_children_complete_independently() -> None:
    class MixedDriver(_FakeSoleDriver):
        supports_speculative_mtp = True

        def submit_speculative_many_detailed(self, requests):
            submissions = []
            for request in requests:
                request_id = self._next_request_id
                self._next_request_id += 1
                submission = GenerationSubmission(
                    request_ids=(request_id,), request=request, max_ticks=1,
                    work_kind="verify_chain", execution_route="mixed_spec",
                )
                self._outputs[request_id] = GenerationOutput(
                    text="spec", generated_token_ids=(990 + request_id,)
                )
                submissions.append(submission)
            return tuple(submissions)

        def submit_speculative_detailed(self, request):
            return self.submit_speculative_many_detailed((request,))[0]

    driver = MixedDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        ar = service.submit_child(_request("ar-long:20"))
        speculative = service.submit_speculative_children(
            (_request("spec-a:1"), _request("spec-b:1"))
        )
        spec_outputs = tuple(handle.result(timeout=2.0) for handle in speculative)

        assert tuple(output.generated_token_ids for output in spec_outputs) == ((992,), (993,))
        assert ar.done is False
        assert ar.cancel(reason="cancel") is True
        with pytest.raises(GenerationCancelled):
            ar.result(timeout=2.0)
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_engine_service_speculative_blocking_and_streaming_trim_same_committed_tail() -> None:
    class TailDriver(_FakeSoleDriver):
        supports_speculative_mtp = True

        def submit_speculative_detailed(self, request: GenerationRequest) -> GenerationSubmission:
            request_id = self._next_request_id
            self._next_request_id += 1
            submission = GenerationSubmission(
                request_ids=(request_id,), request=request, max_ticks=1,
                work_kind="verify_chain", execution_route="spec_tail",
            )
            self._outputs[request_id] = GenerationOutput(
                text="untrimmed", generated_token_ids=(10, 2, 99)
            )
            return submission

        def detokenize(self, token_ids) -> str:
            return ",".join(str(token) for token in token_ids)

    request = replace(
        _request("tail:3", max_tokens=3),
        eos_token_id=2,
        ignore_eos=False,
    )
    driver = TailDriver()
    service = EngineService(driver)
    try:
        blocking = service.generate_speculative_mtp_detailed(request)[0]
        chunks = tuple(service.stream_speculative_mtp_detailed(request))

        assert blocking.generated_token_ids == (10, 2)
        assert blocking.text == "10,2"
        assert blocking.finish_details is not None
        assert blocking.finish_details.reason == "eos"
        assert len(chunks) == 1
        assert chunks[0].generated_token_ids == blocking.generated_token_ids
        assert chunks[0].finish_details == blocking.finish_details
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_engine_service_speculative_handle_cancels_through_shared_reclaim() -> None:
    class CancellableSpecDriver(_FakeSoleDriver):
        supports_speculative_mtp = True

        def submit_speculative_detailed(self, request: GenerationRequest) -> GenerationSubmission:
            submission = super().submit_detailed(request)
            return replace(
                submission,
                work_kind="verify_chain",
                execution_route="engine_service_speculative",
            )

    driver = CancellableSpecDriver()
    service = EngineService(driver, idle_wait_seconds=0.05)
    try:
        handle = service.submit_speculative_child(_request("spec-cancel:100"))
        backend_request_id = handle.backend_request_id

        assert handle.cancel(reason="disconnect") is True
        with pytest.raises(GenerationCancelled):
            handle.result(timeout=2.0)
        assert driver.abort_reasons == {backend_request_id: "disconnect"}
        assert driver.release_order == [backend_request_id]
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_submit_poll_adapter_admits_model_owned_mtp_as_verify_chain_submission() -> None:
    class Inner:
        supports_speculative_mtp = True

        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def generate_speculative_mtp_detailed(self, request: GenerationRequest):
            self.calls += 1
            return [GenerationOutput(text="adapter-spec", generated_token_ids=(900, 901))]

        def close(self) -> None:
            self.closed = True

    inner = Inner()
    adapter = SubmitPollTextGenerator(inner, capacity=1)
    service = EngineService(adapter)
    try:
        output = service.generate_speculative_mtp_detailed(_request("adapter:2", max_tokens=2))[0]
        snapshot = service.live_loop_snapshot()["engine_service"]

        assert output.generated_token_ids == (900, 901)
        assert inner.calls == 1
        assert adapter._speculative_outputs_by_request == {}
        assert adapter._submissions_by_request == {}
        assert snapshot["last_speculative_route"] == "engine_service_verify_chain"
        assert snapshot["last_speculative_work_kind"] == "verify_chain"
        assert snapshot["last_speculative_draft_depth"] == 2
        submission = adapter.last_speculative_submission
        assert submission is not None and submission.work_item is not None
        assert submission.work_item.row_to_request == (submission.request_ids[0],) * 2
    finally:
        service.close()
    assert inner.closed is True


def test_submit_poll_adapter_packs_compatible_speculative_children() -> None:
    class Inner:
        supports_speculative_mtp = True

        def __init__(self) -> None:
            self.prompt_batches: list[tuple[str, ...]] = []

        def generate_speculative_mtp_detailed(self, request: GenerationRequest):
            prompts = tuple(str(prompt) for prompt in request.prompts)
            self.prompt_batches.append(prompts)
            return [
                GenerationOutput(text=prompt, generated_token_ids=(950 + index,))
                for index, prompt in enumerate(prompts)
            ]

        def close(self) -> None:
            return None

    inner = Inner()
    adapter = SubmitPollTextGenerator(inner, capacity=2)
    service = EngineService(adapter)
    try:
        handles = service.submit_speculative_children(
            (_request("adapter-a:1"), _request("adapter-b:1"))
        )
        outputs = tuple(handle.result(timeout=2.0) for handle in handles)
        submission = adapter.last_speculative_submission

        assert inner.prompt_batches == [("adapter-a:1", "adapter-b:1")]
        assert tuple(output.generated_token_ids for output in outputs) == ((950,), (951,))
        assert submission is not None and submission.work_item is not None
        assert len(submission.work_item.request_ids) == 2
        assert submission.work_item.row_to_request == tuple(
            request_id
            for request_id in submission.work_item.request_ids
            for _ in range(submission.work_item.draft_depth)
        )
        assert adapter._speculative_outputs_by_request == {}
        assert adapter._submissions_by_request == {}
    finally:
        service.close()


def test_engine_service_speculative_legacy_route_is_declared_prelaunch_fallback() -> None:
    class LegacyDriver(_FakeSoleDriver):
        supports_speculative_mtp = True

        def __init__(self) -> None:
            super().__init__()
            self.legacy_thread_ids: list[int] = []

        def generate_speculative_mtp_detailed(self, request: GenerationRequest):
            self.legacy_thread_ids.append(threading.get_ident())
            return [GenerationOutput(text="legacy", generated_token_ids=(800,))]

    driver = LegacyDriver()
    service = EngineService(driver)
    try:
        outputs = service.generate_speculative_mtp_detailed(_request("legacy:1"))
        snapshot = service.live_loop_snapshot()["engine_service"]

        assert outputs[0].generated_token_ids == (800,)
        assert driver.legacy_thread_ids == [service.driver_thread_id]
        assert snapshot["speculative_routes"] == {
            "engine_service_verify_chain": 0,
            "legacy_prelaunch_fallback": 1,
        }
        assert snapshot["last_speculative_route"] == "legacy_prelaunch_fallback"
    finally:
        service.close()


def test_engine_service_rejects_one_pending_child_without_closing() -> None:
    class RejectingDriver(_FakeSoleDriver):
        def poll(self, *, max_ticks: int = 1):
            for request_id, state in tuple(self._active.items()):
                if state.prompt.startswith("reject:"):
                    self._active.pop(request_id)
                    return (
                        EngineLoopEvent(
                            kind="rejected",
                            request_id=request_id,
                            request_ids=(request_id,),
                            error=GenerationAdmissionRejected(
                                "global page pressure",
                                resource="device_kv_pool",
                                request_id=request_id,
                                requested_units=4,
                                current_units=8,
                                capacity_units=8,
                            ),
                        ),
                    )
            return super().poll(max_ticks=max_ticks)

    driver = RejectingDriver()
    service = EngineService(driver, command_queue_size=8, idle_wait_seconds=0.001)
    try:
        rejected = service.submit_child(_request("reject:1"))
        with pytest.raises(GenerationAdmissionRejected) as error:
            rejected.result(timeout=2.0)
        assert error.value.resource == "device_kv_pool"
        assert error.value.request_id == rejected.backend_request_id
        assert service.closed is False

        survivor = service.submit_child(_request("survivor:2"))
        assert len(survivor.result(timeout=2.0).generated_token_ids or ()) == 2
        assert service.closed is False
    finally:
        service.close()


def test_engine_service_is_sole_driver_and_refills_before_long_neighbor_finishes() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=32, idle_wait_seconds=0.001)
    caller_thread_id = threading.get_ident()
    try:
        long_handle = service.submit_child(_request("long:20"))
        short_handle = service.submit_child(_request("short:1"))

        short = short_handle.result(timeout=2.0)
        assert short.finish_details is not None
        assert short.finish_details.reason == "length"
        assert long_handle.done is False
        assert driver.release_order == [short_handle.backend_request_id]

        refill_handle = service.submit_child(_request("refill:1"))
        refill_handle.result(timeout=2.0)
        assert long_handle.done is False
        assert driver.release_order[:2] == [
            short_handle.backend_request_id,
            refill_handle.backend_request_id,
        ]

        long_handle.result(timeout=2.0)
        assert driver.poll_thread_ids == {service.driver_thread_id}
        assert caller_thread_id not in driver.poll_thread_ids
    finally:
        service.close()
    assert driver.closed is True


def test_engine_service_splits_parent_rows_and_preserves_public_order() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=16)
    request = GenerationRequest(
        prompts=("first:3", "second:1", "third:2"),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        row_seeds=(11, 22, 33),
    )
    try:
        outputs = service.generate_detailed(request)
    finally:
        service.close()

    assert len(outputs) == 3
    assert [output.generated_tokens for output in outputs] == [3, 1, 2]
    assert driver.submitted_prompt_groups == [("first:3",), ("second:1",), ("third:2",)]
    assert driver.release_order[0] == 2


def test_engine_service_drives_real_submit_poll_adapter_with_independent_children() -> None:
    class ResidentRunner:
        capacity = 3

        def __init__(self) -> None:
            self.targets: dict[int, int] = {}
            self.counts: dict[int, int] = {}
            self.outputs: dict[int, GenerationOutput] = {}
            self.reclaims: list[int] = []

        def prompt_tokens(self, prompt):
            return tuple(int(token) for token in prompt)

        def scheduler_max_new_tokens(self, request):
            return int(request.max_tokens)

        def register_batch(self, request_ids, request, *, prompt_rows):
            del request
            for request_id, prompt_row in zip(request_ids, prompt_rows, strict=True):
                self.targets[int(request_id)] = int(prompt_row[0])

        def prefill_batch(self, work, *, commit: bool):
            assert commit is True

        def decode_batch(self, work, *, commit: bool):
            assert commit is True
            generated = []
            for request_id in work.request_ids:
                rid = int(request_id)
                count = self.counts.get(rid, 0) + 1
                self.counts[rid] = count
                generated.append(
                    GeneratedToken(
                        rid,
                        2000 + rid * 100 + count,
                        finished=count >= self.targets[rid],
                        stream_chunk=GenerationStreamChunk(text=f"row{rid}:{count}"),
                    )
                )
            return tuple(generated)

        def compact_batch(self, moves):
            del moves

        def reclaim(self, completed):
            rid = int(completed.request_id)
            self.reclaims.append(rid)
            self.outputs[rid] = GenerationOutput(
                text=f"done:{rid}",
                generated_token_ids=completed.generated_tokens,
                finish_details=completed.finish_details,
            )

        def has_outputs(self, request_ids):
            return all(int(request_id) in self.outputs for request_id in request_ids)

        def missing_outputs(self, request_ids):
            return [int(request_id) for request_id in request_ids if int(request_id) not in self.outputs]

        def take_outputs(self, request_ids):
            return [self.outputs.pop(int(request_id)) for request_id in request_ids]

        def discard(self, request_ids):
            for request_id in request_ids:
                self.outputs.pop(int(request_id), None)
                self.targets.pop(int(request_id), None)

        def close(self):
            pass

    class Inner:
        def __init__(self) -> None:
            self.runner = ResidentRunner()

        def create_resident_model_runner(self, *, capacity):
            assert capacity in {None, 3}
            return self.runner

    inner = Inner()
    adapter = SubmitPollTextGenerator(inner, capacity=3, prefill_chunk_size=4)
    service = EngineService(adapter)
    request = GenerationRequest(
        prompts=((3,), (1,), (2,)),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    try:
        outputs = service.generate_detailed(request)
    finally:
        service.close()

    assert [output.generated_tokens for output in outputs] == [3, 1, 2]
    assert inner.runner.reclaims[0] == 1
    assert inner.runner.reclaims.index(1) < inner.runner.reclaims.index(0)


def test_engine_service_blocking_and_streaming_share_one_driver() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=16)
    blocking_output: list[GenerationOutput] = []

    def run_blocking() -> None:
        blocking_output.extend(service.generate_detailed(_request("blocking:12")))

    thread = threading.Thread(target=run_blocking)
    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while not driver._active and time.monotonic() < deadline:
            time.sleep(0.001)
        chunks = list(service.stream_detailed(_request("stream:2")))
        assert [chunk.text for chunk in chunks] == ["stream:2:0", "stream:2:1"]
        assert thread.is_alive()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert blocking_output[0].generated_tokens == 12
        assert len(driver.poll_thread_ids) == 1
    finally:
        service.close()
        thread.join(timeout=2.0)


def test_engine_service_stop_holdback_is_owned_per_child_collector() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=8)
    request = _request("stop:2")
    request = replace(
        request,
        stop_token_sequences=((1100, 1101),),
    )
    try:
        chunks = list(service.stream_detailed(request))
    finally:
        service.close()

    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].finish_details is not None
    assert chunks[0].finish_details.reason == "stop"


def test_engine_service_backpressure_and_cancel_are_child_scoped() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=16, stream_queue_max_chunks=1)
    try:
        slow = service.submit_child(_request("slow:20"), streaming=True)
        neighbor = service.submit_child(_request("neighbor:4"))

        with pytest.raises(GenerationCancelled) as overflow:
            slow.result(timeout=2.0)
        assert overflow.value.finish_details.budget_pressure == "client_backpressure"
        assert neighbor.result(timeout=2.0).generated_tokens == 4
        assert slow.backend_request_id in driver.abort_reasons

        cancelled = service.submit_child(_request("cancelled:20"))
        survivor = service.submit_child(_request("survivor:3"))
        assert cancelled.cancel(reason="disconnect") is True
        with pytest.raises(GenerationCancelled):
            cancelled.result(timeout=2.0)
        assert survivor.result(timeout=2.0).generated_tokens == 3
    finally:
        service.close()


def test_engine_service_timeout_is_child_scoped_and_reclaimed() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=8, idle_wait_seconds=0.001)
    expiring_request = replace(
        _request("expiring:100"),
        deadline_at=time.perf_counter() + 0.01,
    )
    try:
        expiring = service.submit_child(expiring_request)
        survivor = service.submit_child(_request("survivor:4"))
        with pytest.raises(GenerationCancelled) as timed_out:
            expiring.result(timeout=2.0)
        assert timed_out.value.finish_details.deadline_exceeded is True
        assert survivor.result(timeout=2.0).generated_tokens == 4
        assert driver.abort_reasons[expiring.backend_request_id] == "timeout"
    finally:
        service.close()


def test_engine_service_shutdown_reclaims_active_children_and_rejects_new_work() -> None:
    driver = _FakeSoleDriver()
    service = EngineService(driver, command_queue_size=8, idle_wait_seconds=0.001)
    active = service.submit_child(_request("active:100"))

    service.close()

    with pytest.raises(GenerationCancelled):
        active.result(timeout=1.0)
    with pytest.raises(RuntimeError, match="closed"):
        service.submit_child(_request("late:1"))
    assert driver.closed is True
    assert active.backend_request_id in driver.release_order


def test_llm_keeps_native_resident_capacity_separate_from_physical_route_cap(
    monkeypatch,
) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    observed: dict[str, int | None] = {}

    class NativeRunner:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity

        def prepare(self) -> None:
            observed["prepared_capacity"] = self.capacity

    class NativeGenerator:
        server_plain_ar_max_active_requests = 4

        def create_resident_model_runner(self, *, capacity):
            observed["capacity"] = capacity
            return NativeRunner(int(capacity))

    fake_index = SimpleNamespace(
        config={"architectures": ["FakeNativeForCausalLM"]},
        model_path="/tmp/fake-model",
    )
    fake_plugin = SimpleNamespace(name="fake_native_service")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_native_service",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: NativeGenerator(),
        replace=True,
    )
    llm = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        max_active_requests=13,
    )
    try:
        generator = llm._get_text_generator()
        assert isinstance(generator, EngineService)
        assert observed["capacity"] == 13
        assert observed["prepared_capacity"] == 13
        assert generator.inner._runner.capacity == 13
    finally:
        llm.close()


def test_independent_service_residency_is_not_clamped_by_physical_route_width() -> None:
    fake = SimpleNamespace(
        supports_independent_generation=True,
        server_plain_ar_max_active_requests=4,
    )
    batcher = _GenerationBatcher(
        engine_factory=lambda: fake,
        batch_window_seconds=0.0,
        max_active_requests=13,
        route_max_active_requests={"default": 4},
    )

    assert batcher._route_request_cap("default") == 13


def test_engine_service_submit_request_batches_admits_every_ready_child_before_poll() -> None:
    class CohortDriver(_FakeSoleDriver):
        def __init__(self) -> None:
            super().__init__()
            self.active_rows_at_poll: list[int] = []

        def poll(self, *, max_ticks: int = 1):
            self.active_rows_at_poll.append(len(self._active))
            return super().poll(max_ticks=max_ticks)

    driver = CohortDriver()
    service = EngineService(driver, command_queue_size=8, idle_wait_seconds=0.001)
    try:
        batches = service.submit_request_batches(
            (_request("slow:4"), _request("fast:1"))
        )
        assert [len(handles) for handles in batches] == [1, 1]
        assert batches[1][0].result(timeout=2.0).generated_tokens == 1
        assert batches[0][0].result(timeout=2.0).generated_tokens == 4
    finally:
        service.close()

    assert driver.active_rows_at_poll[0] == 2
    assert driver.submitted_prompt_groups == [("slow:4",), ("fast:1",)]


def test_llm_independent_batch_seam_preserves_per_item_sampling() -> None:
    class Submitter:
        def __init__(self) -> None:
            self.requests: tuple[GenerationRequest, ...] = ()

        def submit_request_batches(self, requests):
            self.requests = tuple(requests)
            return tuple((f"handle-{index}",) for index, _request in enumerate(requests))

    submitter = Submitter()
    llm = LLM("/tmp/not-loaded")
    llm._text_generator = submitter
    first_token = GenerationCancellationToken()
    second_token = GenerationCancellationToken()
    first = SamplingParams(max_tokens=2, cancellation_token=first_token)
    second = SamplingParams(max_tokens=3, cancellation_token=second_token)

    handles = llm.submit_independent_batches_detailed(
        (([11, 12], first), ([21, 22], second))
    )

    assert handles == (("handle-0",), ("handle-1",))
    assert [request.prompts for request in submitter.requests] == [
        ((11, 12),),
        ((21, 22),),
    ]
    assert [request.max_tokens for request in submitter.requests] == [2, 3]
    assert [request.cancellation_token for request in submitter.requests] == [
        first_token,
        second_token,
    ]


def test_generation_batcher_coalesces_ready_independent_admission_but_completes_items_independently() -> None:
    class Handle:
        def __init__(self, prompt: str, ready: threading.Event) -> None:
            self.prompt = prompt
            self.ready = ready

        @property
        def done(self) -> bool:
            return self.ready.is_set()

        def result(self, timeout: float | None = None) -> GenerationOutput:
            if not self.ready.wait(timeout=timeout):
                raise TimeoutError(self.prompt)
            return GenerationOutput(text=f"generated:{self.prompt}")

    class IndependentGroupedLLM:
        supports_independent_generation = True

        def __init__(self) -> None:
            self.submitted = threading.Event()
            self.release_slow = threading.Event()
            self.fast_ready = threading.Event()
            self.fast_ready.set()
            self.batch_calls: list[tuple[tuple[str, ...], ...]] = []
            self.last_batch_generation = {
                "batch_id": "ready-default-ar",
                "group_widths": [2],
            }

        def submit_independent_batches_detailed(self, batches):
            prompt_groups = tuple(
                tuple(str(prompt) for prompt in prompts)
                for prompts, _sampling in batches
            )
            self.batch_calls.append(prompt_groups)
            self.submitted.set()
            return tuple(
                (
                    Handle(
                        prompts[0],
                        self.release_slow if prompts == ("slow",) else self.fast_ready,
                    ),
                )
                for prompts in prompt_groups
            )

        def generate_detailed(self, prompts, sampling_params):
            del prompts, sampling_params
            raise AssertionError("ready compatible items must use one child cohort")

    async def run() -> None:
        fake = IndependentGroupedLLM()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=3,
        )
        sampling = SamplingParams(max_tokens=4)
        slow = asyncio.create_task(
            batcher.submit(
                ("slow",),
                sampling,
                detailed=True,
                include_batch_metadata=True,
            )
        )
        fast = asyncio.create_task(
            batcher.submit(
                ("fast",),
                sampling,
                detailed=True,
                include_batch_metadata=True,
            )
        )

        assert await asyncio.to_thread(fake.submitted.wait, 2.0)
        fast_result = await asyncio.wait_for(fast, timeout=1.0)
        assert isinstance(fast_result, _QueuedBatchResult)
        assert [output.text for output in fast_result.outputs] == ["generated:fast"]
        assert fast_result.generation_shape["queue_group"]["request_count"] == 2
        assert fast_result.generation_shape["backend_groups"][0]["actual_group_rows"] == [2]
        assert not slow.done()
        assert batcher.active_requests() == 1

        # A dynamically arriving request uses the free resident capacity; it is
        # neither held behind nor merged into the already-admitted slow child.
        late_result = await asyncio.wait_for(
            batcher.submit(
                ("late",),
                sampling,
                detailed=True,
                include_batch_metadata=True,
            ),
            timeout=1.0,
        )
        assert isinstance(late_result, _QueuedBatchResult)
        assert [output.text for output in late_result.outputs] == ["generated:late"]
        assert late_result.generation_shape["queue_group"]["request_count"] == 1
        assert not slow.done()
        assert batcher.active_requests() == 1

        fake.release_slow.set()
        slow_result = await asyncio.wait_for(slow, timeout=2.0)
        assert isinstance(slow_result, _QueuedBatchResult)
        assert [output.text for output in slow_result.outputs] == ["generated:slow"]
        assert slow_result.generation_shape["queue_group"]["id"] == fast_result.generation_shape["queue_group"]["id"]
        assert fake.batch_calls == [(('slow',), ('fast',)), (('late',),)]
        await batcher.shutdown(grace_seconds=0.1)

    asyncio.run(run())


def test_generation_batcher_ready_cohort_isolates_child_failure() -> None:
    class Handle:
        def __init__(self, prompt: str) -> None:
            self.prompt = prompt

        def result(self) -> GenerationOutput:
            if self.prompt == "bad":
                raise RuntimeError("bad child")
            return GenerationOutput(text=f"generated:{self.prompt}")

    class IndependentLLM:
        supports_independent_generation = True
        last_batch_generation = {
            "batch_id": "isolated-failure",
            "group_widths": [2],
        }

        def submit_independent_batches_detailed(self, batches):
            return tuple(
                tuple(Handle(str(prompt)) for prompt in prompts)
                for prompts, _sampling in batches
            )

    async def run() -> None:
        fake = IndependentLLM()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=2,
        )
        sampling = SamplingParams(max_tokens=1)
        bad, good = await asyncio.gather(
            batcher.submit(("bad",), sampling, detailed=True),
            batcher.submit(("good",), sampling, detailed=True),
            return_exceptions=True,
        )
        assert isinstance(bad, RuntimeError)
        assert str(bad) == "bad child"
        assert [output.text for output in good] == ["generated:good"]
        assert batcher.active_requests() == 0
        await batcher.shutdown(grace_seconds=0.1)

    asyncio.run(run())


def test_generation_batcher_ready_cohort_rollback_keeps_separate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IndependentLLM:
        supports_independent_generation = True

        def __init__(self) -> None:
            self.batch_calls = 0
            self.calls: list[tuple[str, ...]] = []

        def submit_independent_batches_detailed(self, batches):
            del batches
            self.batch_calls += 1
            raise AssertionError("rollback must bypass ready cohorts")

        def generate_detailed(self, prompts, sampling_params):
            del sampling_params
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append(prompt_tuple)
            return [GenerationOutput(text=f"generated:{prompt_tuple[0]}")]

    async def run() -> None:
        monkeypatch.setenv("HIPENGINE_SERVER_DEFAULT_AR_READY_COHORT", "0")
        fake = IndependentLLM()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=2,
        )
        sampling = SamplingParams(max_tokens=1)
        outputs = await asyncio.gather(
            batcher.submit(("one",), sampling, detailed=True),
            batcher.submit(("two",), sampling, detailed=True),
        )
        assert [[output.text for output in rows] for rows in outputs] == [
            ["generated:one"],
            ["generated:two"],
        ]
        assert fake.batch_calls == 0
        assert set(fake.calls) == {("one",), ("two",)}
        await batcher.shutdown(grace_seconds=0.1)

    asyncio.run(run())


def test_generation_batcher_publishes_fast_independent_result_before_slow_neighbor() -> None:
    class IndependentLLM:
        supports_independent_generation = True

        def __init__(self) -> None:
            self.slow_started = threading.Event()
            self.release_slow = threading.Event()
            self.calls: list[tuple[str, ...]] = []

        def generate_detailed(self, prompts, sampling_params):
            del sampling_params
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            assert len(prompt_tuple) == 1
            self.calls.append(prompt_tuple)
            if prompt_tuple == ("slow",):
                self.slow_started.set()
                assert self.release_slow.wait(timeout=5.0)
            return [GenerationOutput(text=f"generated:{prompt_tuple[0]}")]

    async def run() -> None:
        fake = IndependentLLM()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=2,
        )
        sampling = SamplingParams(max_tokens=4)
        slow = asyncio.create_task(batcher.submit(("slow",), sampling, detailed=True))
        fast = asyncio.create_task(batcher.submit(("fast",), sampling, detailed=True))

        assert await asyncio.to_thread(fake.slow_started.wait, 2.0)
        fast_result = await asyncio.wait_for(fast, timeout=1.0)
        assert [output.text for output in fast_result] == ["generated:fast"]
        assert not slow.done()

        fake.release_slow.set()
        slow_result = await asyncio.wait_for(slow, timeout=2.0)
        assert [output.text for output in slow_result] == ["generated:slow"]
        assert set(fake.calls) == {("slow",), ("fast",)}
        await batcher.shutdown(grace_seconds=0.1)

    asyncio.run(run())


def test_generation_batcher_shutdown_waits_for_independent_model_thread() -> None:
    """Forced shutdown must not orphan C2 work running in a Python thread."""

    class IndependentBlockingLLM:
        supports_independent_generation = True

        def __init__(self) -> None:
            self.started = threading.Event()
            self.cancel_seen = threading.Event()
            self.release = threading.Event()

        def generate_detailed(self, prompts, sampling_params):
            assert tuple(prompts) == ("blocked",)
            token = sampling_params.cancellation_token
            assert token is not None
            self.started.set()
            while not token.cancelled:
                time.sleep(0.001)
            self.cancel_seen.set()
            assert self.release.wait(timeout=5.0)
            token.raise_if_cancelled()
            return [GenerationOutput(text="unreachable")]

    async def run() -> None:
        fake = IndependentBlockingLLM()
        token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            max_active_requests=1,
        )
        pending = asyncio.create_task(
            batcher.submit(
                ("blocked",),
                SamplingParams(max_tokens=4, cancellation_token=token),
                detailed=True,
            )
        )
        assert await asyncio.to_thread(fake.started.wait, 5.0)

        shutdown = asyncio.create_task(batcher.shutdown(grace_seconds=0.01))
        assert await asyncio.to_thread(fake.cancel_seen.wait, 5.0)
        await asyncio.sleep(0.02)
        assert shutdown.done() is False
        assert batcher.active_requests() == 1

        fake.release.set()
        result = await asyncio.wait_for(shutdown, timeout=5.0)
        assert result["forced"] is True
        assert result["active_requests"] == 0
        with pytest.raises(GenerationCancelled):
            await pending
        assert batcher.active() is False

    asyncio.run(run())
