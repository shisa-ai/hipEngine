"""Sole-driver Generation-2 service around the resident submit/poll runtime."""

from __future__ import annotations

import queue
import threading
from collections import Counter
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Sequence

from hipengine.generation.concurrency2 import (
    BlockingOutputCollector,
    EngineOutput,
    OutputCollector,
    OutputKind,
    StreamingOutputCollector,
)
from hipengine.generation.deadline import GenerationCancelled, generation_deadline_expired
from hipengine.generation.engine_loop import (
    EngineLoopEvent,
    GenerationAdmissionRejected,
    GenerationSubmission,
)
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
)
from hipengine.speculative.streaming import trim_speculative_output


@dataclass(slots=True)
class _CommandResponse:
    ready: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: BaseException | None = None

    def finish(self, *, value: Any = None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error
        self.ready.set()

    def result(self, timeout: float | None = None) -> Any:
        if not self.ready.wait(timeout=timeout):
            raise TimeoutError("engine service command timed out")
        if self.error is not None:
            raise self.error
        return self.value


@dataclass(slots=True)
class _ChildState:
    service_request_id: int
    request: GenerationRequest
    collector: OutputCollector
    submission: GenerationSubmission | None = None
    backend_request_id: int | None = None
    token_ids: list[int] = field(default_factory=list)
    pending_stop_chunks: list[tuple[int, GenerationStreamChunk]] = field(default_factory=list)
    terminal: bool = False
    execution_mode: str = "ar"


@dataclass(frozen=True, slots=True)
class _ServiceCommand:
    kind: str
    response: _CommandResponse
    state: _ChildState | None = None
    states: tuple[_ChildState, ...] = ()
    service_request_id: int | None = None
    reason: str | None = None
    method_name: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()


class EngineServiceHandle:
    """One independently completing child submitted to an ``EngineService``."""

    def __init__(self, service: "EngineService", state: _ChildState) -> None:
        self._service = service
        self._state = state

    @property
    def request_id(self) -> int:
        return self._state.service_request_id

    @property
    def backend_request_id(self) -> int:
        request_id = self._state.backend_request_id
        if request_id is None:
            raise RuntimeError("child has not been admitted to the resident driver")
        return request_id

    @property
    def done(self) -> bool:
        return self._state.collector.result is not None

    def result(self, timeout: float | None = None) -> GenerationOutput:
        result = self._state.collector.wait(timeout=timeout)
        if result is None:
            raise TimeoutError(f"engine child {self.request_id} did not complete")
        if result.error is not None:
            raise result.error
        if result.generation_output is None:
            raise RuntimeError("terminal engine child has no GenerationOutput")
        return result.generation_output

    def cancel(self, *, reason: str = "cancel") -> bool:
        return self._service.cancel(self.request_id, reason=reason)

    def iter_chunks(self) -> Iterator[GenerationStreamChunk]:
        collector = self._state.collector
        if not isinstance(collector, StreamingOutputCollector):
            raise TypeError("iter_chunks requires a streaming child handle")
        emitted = False
        while True:
            if not collector.wait_for_event(timeout=0.1):
                if collector.result is not None:
                    break
                if self._service.closed:
                    raise RuntimeError("engine service closed before child completion")
                continue
            for event in collector.drain(max_chunks=1):
                if event.kind in {OutputKind.TOKEN, OutputKind.CHUNK}:
                    if event.stream_chunk is not None:
                        emitted = True
                        yield event.stream_chunk
                    continue
                result = collector.result
                if result is None:
                    raise RuntimeError("terminal stream event has no collector result")
                if result.error is not None:
                    raise result.error
                if result.generation_output is not None and (
                    not emitted
                    or self._state.execution_mode in {"verify_chain", "verify_tree"}
                ):
                    output = result.generation_output
                    yield GenerationStreamChunk(
                        text="" if emitted else output.text,
                        token_logprobs=() if emitted else output.token_logprobs,
                        finish_details=output.finish_details,
                        telemetry=output.telemetry,
                        generated_token_ids=output.generated_token_ids,
                    )
                return
        result = collector.result
        if result is None:
            raise RuntimeError("stream ended without terminal output")
        if result.error is not None:
            raise result.error


class EngineService:
    """One command/output driver for every child using a loaded model replica.

    Frontend threads enqueue child commands and wait on isolated collectors. Only
    this service's driver thread calls ``submit_detailed``, ``poll``,
    ``take_result``, cancellation/reclaim, or shutdown on the resident adapter.
    """

    supports_independent_generation = True

    def __init__(
        self,
        driver: Any,
        *,
        command_queue_size: int = 1024,
        stream_queue_max_chunks: int | None = None,
        idle_wait_seconds: float = 0.001,
        command_timeout_seconds: float = 30.0,
    ) -> None:
        queue_size = int(command_queue_size)
        if queue_size <= 0:
            raise ValueError("command_queue_size must be positive")
        configured_stream_bound = (
            int(getattr(driver, "stream_queue_max_chunks", 64))
            if stream_queue_max_chunks is None
            else int(stream_queue_max_chunks)
        )
        if configured_stream_bound <= 0:
            raise ValueError("stream_queue_max_chunks must be positive")
        idle_wait = float(idle_wait_seconds)
        if idle_wait < 0.0:
            raise ValueError("idle_wait_seconds must be non-negative")
        command_timeout = float(command_timeout_seconds)
        if command_timeout <= 0.0:
            raise ValueError("command_timeout_seconds must be positive")
        self._driver = driver
        self._canonical_token_events = bool(
            getattr(driver, "canonical_token_events", True)
        )
        self._commands: queue.Queue[_ServiceCommand] = queue.Queue(maxsize=queue_size)
        self._stream_queue_max_chunks = configured_stream_bound
        self._idle_wait_seconds = idle_wait
        self._command_timeout_seconds = command_timeout
        self._states_by_service_id: dict[int, _ChildState] = {}
        self._states_by_backend_id: dict[int, _ChildState] = {}
        self._next_service_request_id = 0
        self._request_id_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._output_condition = threading.Condition()
        self._speculative_route_counts: Counter[str] = Counter()
        self._last_speculative_route: str | None = None
        self._closing = False
        self._closed = False
        self._driver_thread_id: int | None = None
        self._driver_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._drive,
            name="hipengine-engine-service",
            daemon=True,
        )
        self._thread.start()
        if not self._driver_ready.wait(timeout=self._command_timeout_seconds):
            raise RuntimeError("engine service driver thread did not start")

    @property
    def inner(self) -> Any:
        return self._driver

    @property
    def driver_thread_id(self) -> int:
        if self._driver_thread_id is None:
            raise RuntimeError("engine service driver thread is not ready")
        return self._driver_thread_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def supports_controlled_streaming(self) -> bool:
        return bool(getattr(self._driver, "supports_controlled_streaming", False))

    @property
    def supports_stream_many(self) -> bool:
        return bool(
            self.supports_controlled_streaming
            or getattr(self._driver, "supports_stream_many", False)
        )

    @property
    def supports_speculative_mtp(self) -> bool:
        return bool(getattr(self._driver, "supports_speculative_mtp", False))

    @property
    def stream_queue_max_chunks(self) -> int:
        return self._stream_queue_max_chunks

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)

    def submit_child(
        self,
        request: GenerationRequest,
        *,
        streaming: bool = False,
        stream_queue_max_chunks: int | None = None,
    ) -> EngineServiceHandle:
        state = self._new_child_state(
            request,
            streaming=streaming,
            stream_queue_max_chunks=stream_queue_max_chunks,
        )
        response = _CommandResponse()
        self._enqueue(_ServiceCommand("submit", response, state=state))
        response.result(timeout=self._command_timeout_seconds)
        return EngineServiceHandle(self, state)

    def submit_children(
        self,
        requests: Sequence[GenerationRequest],
        *,
        streaming: bool = False,
    ) -> tuple[EngineServiceHandle, ...]:
        """Publish several independent children before the next model tick."""

        states = tuple(
            self._new_child_state(request, streaming=streaming)
            for request in requests
        )
        if not states:
            return ()
        response = _CommandResponse()
        self._enqueue(_ServiceCommand("submit_many", response, states=states))
        response.result(timeout=self._command_timeout_seconds)
        return tuple(EngineServiceHandle(self, state) for state in states)

    def submit_request_batches(
        self,
        requests: Sequence[GenerationRequest],
    ) -> tuple[tuple[EngineServiceHandle, ...], ...]:
        """Admit ready request batches in one command while retaining child handles."""

        child_groups = tuple(_split_generation_request(request) for request in requests)
        flat_children = tuple(child for group in child_groups for child in group)
        flat_handles = self.submit_children(flat_children)
        grouped: list[tuple[EngineServiceHandle, ...]] = []
        offset = 0
        for children in child_groups:
            end = offset + len(children)
            grouped.append(tuple(flat_handles[offset:end]))
            offset = end
        return tuple(grouped)

    def submit_speculative_child(
        self,
        request: GenerationRequest,
        *,
        streaming: bool = False,
        stream_queue_max_chunks: int | None = None,
    ) -> EngineServiceHandle:
        """Submit one guarded VERIFY_CHAIN child through the shared lifecycle."""

        state = self._new_child_state(
            request,
            streaming=streaming,
            stream_queue_max_chunks=stream_queue_max_chunks,
            execution_mode="verify_chain",
        )
        response = _CommandResponse()
        self._enqueue(
            _ServiceCommand(
                "submit_speculative_many",
                response,
                states=(state,),
            )
        )
        response.result(timeout=self._command_timeout_seconds)
        self._record_speculative_route("engine_service_verify_chain")
        return EngineServiceHandle(self, state)

    def submit_speculative_children(
        self,
        requests: Sequence[GenerationRequest],
        *,
        streaming: bool = False,
    ) -> tuple[EngineServiceHandle, ...]:
        states = tuple(
            self._new_child_state(
                request,
                streaming=streaming,
                execution_mode="verify_chain",
            )
            for request in requests
        )
        if not states:
            return ()
        response = _CommandResponse()
        self._enqueue(
            _ServiceCommand("submit_speculative_many", response, states=states)
        )
        response.result(timeout=self._command_timeout_seconds)
        self._record_speculative_route("engine_service_verify_chain")
        return tuple(EngineServiceHandle(self, state) for state in states)

    def cancel(self, service_request_id: int, *, reason: str = "cancel") -> bool:
        response = _CommandResponse()
        self._enqueue(
            _ServiceCommand(
                "cancel",
                response,
                service_request_id=int(service_request_id),
                reason=str(reason),
            )
        )
        return bool(response.result(timeout=self._command_timeout_seconds))

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        children = _split_generation_request(request)
        handles = list(self.submit_children(children))
        outputs: list[GenerationOutput] = []
        first_error: BaseException | None = None
        for handle in handles:
            try:
                outputs.append(handle.result())
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                outputs.append(GenerationOutput(text=""))
        if first_error is not None:
            raise first_error
        return outputs

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield str(chunk)

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        children = _split_generation_request(request)
        if len(children) != 1:
            raise ValueError("stream_detailed requires exactly one prompt")
        handle = self.submit_child(children[0], streaming=True)
        consumed = False
        try:
            yield from handle.iter_chunks()
            consumed = True
        finally:
            if not consumed and not handle.done:
                handle.cancel(reason="disconnect")

    def stream_many_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        children = _split_generation_request(request)
        handles = list(self.submit_children(children, streaming=True))
        active = set(range(len(handles)))
        consumed = False
        try:
            while active:
                progressed = False
                for choice_index in tuple(sorted(active)):
                    handle = handles[choice_index]
                    collector = handle._state.collector
                    assert isinstance(collector, StreamingOutputCollector)
                    for event in collector.drain(max_chunks=1):
                        progressed = True
                        if event.kind in {OutputKind.TOKEN, OutputKind.CHUNK}:
                            if event.stream_chunk is not None:
                                yield _stream_chunk_for_choice(event.stream_chunk, choice_index)
                            continue
                        result = collector.result
                        if result is None:
                            raise RuntimeError("terminal stream event has no result")
                        active.remove(choice_index)
                        if result.error is not None:
                            raise result.error
                        if result.generation_output is not None and not event.generated_token_ids:
                            output = result.generation_output
                            yield _stream_chunk_for_choice(
                                GenerationStreamChunk(
                                    text=output.text,
                                    token_logprobs=output.token_logprobs,
                                    finish_details=output.finish_details,
                                    telemetry=output.telemetry,
                                    generated_token_ids=output.generated_token_ids,
                                ),
                                choice_index,
                            )
                if not progressed and active:
                    with self._output_condition:
                        self._output_condition.wait(timeout=0.05)
            consumed = True
        finally:
            if not consumed:
                for choice_index in active:
                    if not handles[choice_index].done:
                        handles[choice_index].cancel(reason="disconnect")

    def prepare(self, **kwargs: Any) -> Any:
        return self._control("prepare", **kwargs)

    def prepare_request_scratch(self, **kwargs: Any) -> Any:
        return self._control("prepare_request_scratch", **kwargs)

    def reconfigure_engine_loop(self, config: Any) -> None:
        """Serialize an idle pool/policy generation on the sole driver."""

        self._control("reconfigure_engine_loop", config)

    def count_tokens(self, text: str) -> int:
        return int(self._control("count_tokens", str(text)))

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self._control("tokenize", str(text)))

    def detokenize(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        return str(self._control("detokenize", tuple(int(token) for token in token_ids), **kwargs))

    def generate_speculative_mtp_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        """Run guarded MTP through the shared child lifecycle when supported.

        Drivers implementing ``submit_speculative_detailed`` enter the same
        service request table, collectors, completion, cancellation, and output
        path as AR children. Older exact model-owned routes remain a declared
        pre-launch fallback and never mix with an admitted speculative child.
        """

        children = _split_generation_request(request)
        submit = getattr(self._driver, "submit_speculative_detailed", None)
        if not callable(submit):
            outputs = list(self._control("generate_speculative_mtp_detailed", request))
            self._record_speculative_route("legacy_prelaunch_fallback")
            return outputs
        handles = self.submit_speculative_children(children)
        outputs: list[GenerationOutput] = []
        first_error: BaseException | None = None
        for handle in handles:
            try:
                outputs.append(handle.result())
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                outputs.append(GenerationOutput(text=""))
        if first_error is not None:
            raise first_error
        return outputs

    def stream_speculative_mtp_detailed(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        children = _split_generation_request(request)
        if len(children) != 1:
            raise ValueError("speculative streaming requires exactly one prompt")
        submit = getattr(self._driver, "submit_speculative_detailed", None)
        if not callable(submit):
            raise NotImplementedError(
                "speculative streaming has no legacy out-of-band fallback"
            )
        handle = self.submit_speculative_child(children[0], streaming=True)
        consumed = False
        try:
            yield from handle.iter_chunks()
            consumed = True
        finally:
            if not consumed and not handle.done:
                handle.cancel(reason="disconnect")

    def compact(self, order: Sequence[int] | None = None) -> tuple[Any, ...]:
        """Serialize scheduler/model compaction on the sole driver thread."""

        requested = None if order is None else tuple(int(value) for value in order)
        return tuple(self._control("compact", requested))

    def live_loop_snapshot(self) -> dict[str, object]:
        payload = self._control("live_loop_snapshot")
        snapshot = dict(payload) if isinstance(payload, dict) else {}
        last_submission = getattr(self._driver, "last_speculative_submission", None)
        snapshot["engine_service"] = {
            "sole_driver": True,
            "driver_thread_id": self.driver_thread_id,
            "active_children": len(self._states_by_service_id),
            "command_queue_depth": self._commands.qsize(),
            "speculative_routes": {
                "engine_service_verify_chain": int(
                    self._speculative_route_counts["engine_service_verify_chain"]
                ),
                "legacy_prelaunch_fallback": int(
                    self._speculative_route_counts["legacy_prelaunch_fallback"]
                ),
            },
            "last_speculative_route": self._last_speculative_route,
            "last_speculative_work_kind": (
                None if last_submission is None else last_submission.work_kind
            ),
            "last_speculative_draft_depth": (
                None
                if last_submission is None or last_submission.work_item is None
                else int(last_submission.work_item.draft_depth)
            ),
        }
        return snapshot

    def drain_cancellations(self) -> int:
        method = getattr(self._driver, "drain_cancellations", None)
        if not callable(method):
            return 0
        return int(self._control("drain_cancellations"))

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._closing:
                thread = self._thread
                response = None
            else:
                self._closing = True
                response = _CommandResponse()
                self._put_command(_ServiceCommand("shutdown", response))
                thread = self._thread
        if response is not None:
            response.result(timeout=self._command_timeout_seconds)
        if thread is not threading.current_thread():
            thread.join(timeout=self._command_timeout_seconds)
            if thread.is_alive():
                raise RuntimeError("engine service driver did not stop")

    def _new_child_state(
        self,
        request: GenerationRequest,
        *,
        streaming: bool,
        stream_queue_max_chunks: int | None = None,
        execution_mode: str = "ar",
    ) -> _ChildState:
        if len(request.prompts) != 1:
            raise ValueError("one engine child requires exactly one prompt")
        service_request_id = self._allocate_service_request_id()
        collector: OutputCollector
        if streaming:
            collector = StreamingOutputCollector(
                max_output_tokens=request.max_tokens,
                max_chunks=(
                    self._stream_queue_max_chunks
                    if stream_queue_max_chunks is None
                    else int(stream_queue_max_chunks)
                ),
                enqueue_token_events=False,
            )
        else:
            collector = BlockingOutputCollector(max_output_tokens=request.max_tokens)
        collector.bind(service_request_id)
        mode = str(execution_mode)
        if mode not in {"ar", "verify_chain", "verify_tree"}:
            raise ValueError("execution_mode must be ar, verify_chain, or verify_tree")
        return _ChildState(
            service_request_id,
            request,
            collector,
            execution_mode=mode,
        )

    def _record_speculative_route(self, route: str) -> None:
        with self._lifecycle_lock:
            self._speculative_route_counts[str(route)] += 1
            self._last_speculative_route = str(route)

    def _allocate_service_request_id(self) -> int:
        with self._request_id_lock:
            request_id = self._next_service_request_id
            self._next_service_request_id += 1
            return request_id

    def _enqueue(self, command: _ServiceCommand) -> None:
        with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("engine service is closed")
            self._put_command(command)

    def _put_command(self, command: _ServiceCommand) -> None:
        try:
            self._commands.put_nowait(command)
        except queue.Full as exc:
            raise GenerationAdmissionRejected(
                "engine service command queue is full",
                resource="engine_command_queue",
                requested_units=1,
                current_units=self._commands.qsize(),
                capacity_units=self._commands.maxsize,
            ) from exc

    def _control(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if threading.get_ident() == self._driver_thread_id:
            return getattr(self._driver, method_name)(*args, **kwargs)
        response = _CommandResponse()
        self._enqueue(
            _ServiceCommand(
                "control",
                response,
                method_name=str(method_name),
                args=tuple(args),
                kwargs=tuple(kwargs.items()),
            )
        )
        return response.result(timeout=self._command_timeout_seconds)

    def _drive(self) -> None:
        self._driver_thread_id = threading.get_ident()
        self._driver_ready.set()
        shutdown_response: _CommandResponse | None = None
        try:
            while shutdown_response is None:
                if not self._states_by_service_id:
                    try:
                        command = self._commands.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    shutdown_response = self._process_command(command)
                    continue
                shutdown_response = self._drain_commands()
                if shutdown_response is not None:
                    break
                if self._idle_wait_seconds:
                    try:
                        command = self._commands.get(timeout=self._idle_wait_seconds)
                    except queue.Empty:
                        pass
                    else:
                        shutdown_response = self._process_command(command)
                        if shutdown_response is not None:
                            break
                        shutdown_response = self._drain_commands()
                        if shutdown_response is not None:
                            break
                events = tuple(self._driver.poll(max_ticks=1))
                self._route_events(events)
                self._finish_ready_children()
                if not events and self._states_by_service_id and self._idle_wait_seconds:
                    time.sleep(self._idle_wait_seconds)
        except BaseException as exc:
            import os
            import sys
            import traceback
            if os.environ.get("HIPENGINE_ENGINE_SERVICE_TRACEBACK"):
                print("=== engine service driver exception ===", file=sys.stderr, flush=True)
                traceback.print_exc()
            self._fail_all(exc)
            if shutdown_response is not None:
                shutdown_response.finish(error=exc)
                shutdown_response = None
        finally:
            for state in tuple(self._states_by_service_id.values()):
                self._cancel_state(state, reason="shutdown")
            closer = getattr(self._driver, "close", None)
            close_error: BaseException | None = None
            if callable(closer):
                try:
                    closer()
                except BaseException as exc:  # pragma: no cover - defensive shutdown
                    close_error = exc
            with self._lifecycle_lock:
                self._closed = True
                self._closing = True
            if shutdown_response is not None:
                shutdown_response.finish(error=close_error)

    def _drain_commands(self) -> _CommandResponse | None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return None
            shutdown = self._process_command(command)
            if shutdown is not None:
                return shutdown

    def _bind_child_submission(
        self,
        state: _ChildState,
        submission: GenerationSubmission,
        *,
        expected_mode: str | None = None,
        backend_ids: set[int] | None = None,
    ) -> None:
        if len(submission.request_ids) != 1:
            raise RuntimeError("one child submission must map to one backend request")
        if expected_mode is not None and str(submission.work_kind) != expected_mode:
            raise ValueError(
                "speculative submission work_kind must match child execution mode"
            )
        backend_request_id = int(submission.request_ids[0])
        if backend_request_id in self._states_by_backend_id or (
            backend_ids is not None and backend_request_id in backend_ids
        ):
            raise RuntimeError("backend request_id is already owned by another child")
        if backend_ids is not None:
            backend_ids.add(backend_request_id)
        state.submission = submission
        state.backend_request_id = backend_request_id
        self._states_by_service_id[state.service_request_id] = state
        self._states_by_backend_id[backend_request_id] = state

    def _admit_child_state(self, state: _ChildState) -> None:
        if state.execution_mode == "ar":
            submission = self._driver.submit_detailed(state.request)
        else:
            submit = getattr(self._driver, "submit_speculative_detailed", None)
            if not callable(submit):
                raise NotImplementedError(
                    "speculative submission is unavailable before child admission"
                )
            submission = submit(state.request)
        self._bind_child_submission(
            state,
            submission,
            expected_mode=(None if state.execution_mode == "ar" else state.execution_mode),
        )

    def _process_command(self, command: _ServiceCommand) -> _CommandResponse | None:
        if command.kind == "shutdown":
            return command.response
        if command.kind in {"submit", "submit_many", "submit_speculative_many"}:
            states = (
                (command.state,)
                if command.kind == "submit" and command.state is not None
                else command.states
            )
            admitted: list[_ChildState] = []
            batch_submissions: tuple[GenerationSubmission, ...] = ()
            try:
                batch_submit = (
                    getattr(self._driver, "submit_speculative_many_detailed", None)
                    if command.kind == "submit_speculative_many" and len(states) > 1
                    else None
                )
                if callable(batch_submit):
                    submissions = tuple(
                        batch_submit(tuple(state.request for state in states))
                    )
                    batch_submissions = submissions
                    if len(submissions) != len(states):
                        raise RuntimeError(
                            "speculative batch submission must return one submission per child"
                        )
                    backend_ids: set[int] = set()
                    for submission in submissions:
                        if len(submission.request_ids) != 1:
                            raise RuntimeError(
                                "one child submission must map to one backend request"
                            )
                        backend_request_id = int(submission.request_ids[0])
                        if backend_request_id in backend_ids or backend_request_id in self._states_by_backend_id:
                            raise RuntimeError(
                                "backend request_id is already owned by another child"
                            )
                        if str(submission.work_kind) != "verify_chain":
                            raise ValueError(
                                "speculative submission work_kind must match child execution mode"
                            )
                        backend_ids.add(backend_request_id)
                    backend_ids.clear()
                    for state, submission in zip(states, submissions, strict=True):
                        self._bind_child_submission(
                            state,
                            submission,
                            expected_mode="verify_chain",
                            backend_ids=backend_ids,
                        )
                        admitted.append(state)
                else:
                    for state in states:
                        self._admit_child_state(state)
                        admitted.append(state)
            except BaseException as exc:
                for state in admitted:
                    self._cancel_state(state, reason="parent_submit_error")
                if batch_submissions and not admitted:
                    abort = getattr(self._driver, "abort_submission", None)
                    if callable(abort):
                        for submission in batch_submissions:
                            try:
                                abort(submission, reason="parent_submit_error")
                            except BaseException:
                                pass
                command.response.finish(error=exc)
            else:
                command.response.finish(
                    value=tuple(state.service_request_id for state in states)
                )
            return None
        if command.kind == "cancel":
            state = self._states_by_service_id.get(int(command.service_request_id))
            if state is None or state.terminal:
                command.response.finish(value=False)
            else:
                self._cancel_state(state, reason=str(command.reason or "cancel"))
                command.response.finish(value=True)
            return None
        if command.kind == "control":
            try:
                method = getattr(self._driver, str(command.method_name))
                value = method(*command.args, **dict(command.kwargs))
            except BaseException as exc:
                command.response.finish(error=exc)
            else:
                command.response.finish(value=value)
            return None
        command.response.finish(error=ValueError(f"unknown engine service command {command.kind!r}"))
        return None

    def _route_events(self, events: Sequence[EngineLoopEvent]) -> None:
        for event in events:
            if event.kind == "rejected" and event.request_id is not None:
                state = self._states_by_backend_id.get(int(event.request_id))
                if state is not None and not state.terminal:
                    error = event.error
                    if error is None:
                        error = GenerationAdmissionRejected(
                            "resident admission was rejected",
                            resource="resident_admission",
                            request_id=int(event.request_id),
                        )
                    self._publish_terminal(
                        state,
                        generation_output=None,
                        finish_details=FinishDetails(reason="error"),
                        error=error,
                    )
                continue
            if (
                not self._canonical_token_events
                or event.kind != "token"
                or event.request_id is None
                or event.token_id is None
                or event.stream_chunk is None
            ):
                # Compatibility runners emit a synthetic scheduler token after
                # producing a whole GenerationOutput. Publish its real IDs at
                # terminal instead of exposing the surrogate row index.
                continue
            state = self._states_by_backend_id.get(int(event.request_id))
            if state is None or state.terminal:
                continue
            token_id = int(event.token_id)
            accepted = state.collector.publish(
                EngineOutput(
                    kind=OutputKind.TOKEN,
                    request_id=state.service_request_id,
                    token_id=token_id,
                    token_index=len(state.token_ids),
                )
            )
            if not accepted:
                self._cancel_state(state, reason="client_backpressure")
                continue
            state.token_ids.append(token_id)
            stream_chunk = event.stream_chunk
            if state.execution_mode in {"verify_chain", "verify_tree"} and (
                stream_chunk.finish_details is not None
                or stream_chunk.generated_token_ids is not None
            ):
                # One terminal service chunk owns final speculative IDs, finish,
                # and authoritative backend telemetry. Live chunks own text only.
                stream_chunk = replace(
                    stream_chunk,
                    finish_details=None,
                    generated_token_ids=None,
                )
            chunks = _stop_safe_chunks(state, token_id=token_id, chunk=stream_chunk)
            for chunk in chunks:
                accepted = state.collector.publish(
                    EngineOutput(
                        kind=OutputKind.CHUNK,
                        request_id=state.service_request_id,
                        stream_chunk=chunk,
                    )
                )
                if not accepted:
                    self._cancel_state(state, reason="client_backpressure")
                    break
            self._notify_output()

    def _finish_ready_children(self) -> None:
        for state in tuple(self._states_by_service_id.values()):
            if state.terminal or state.submission is None:
                continue
            if self._driver.generation_complete(state.submission):
                try:
                    outputs = list(self._driver.take_result(state.submission))
                    if len(outputs) != 1:
                        raise RuntimeError("one child completion must return one output")
                    output = outputs[0]
                    generation_output = (
                        output
                        if isinstance(output, GenerationOutput)
                        else GenerationOutput(text=str(output))
                    )
                    if state.execution_mode in {"verify_chain", "verify_tree"}:
                        generation_output = self._normalize_speculative_output(
                            state,
                            generation_output,
                        )
                    output_ids = generation_output.generated_token_ids
                    if output_ids is not None and not state.token_ids:
                        for token_id in output_ids:
                            accepted = state.collector.publish(
                                EngineOutput(
                                    kind=OutputKind.TOKEN,
                                    request_id=state.service_request_id,
                                    token_id=int(token_id),
                                    token_index=len(state.token_ids),
                                )
                            )
                            if not accepted:
                                raise RuntimeError(
                                    "terminal token history exceeds the child output bound"
                                )
                            state.token_ids.append(int(token_id))
                    if output_ids is not None and tuple(output_ids) != tuple(state.token_ids):
                        raise RuntimeError("driver terminal token history differs from routed token events")
                    finish_details = generation_output.finish_details or FinishDetails(reason="length")
                    self._publish_terminal(
                        state,
                        generation_output=generation_output,
                        finish_details=finish_details,
                    )
                except BaseException as exc:
                    self._cancel_state(state, reason="completion_error", error=exc)
                continue
            request = state.request
            token = request.cancellation_token
            if generation_deadline_expired(request.deadline_at):
                self._cancel_state(state, reason="timeout")
            elif token is not None and bool(getattr(token, "cancelled", False)):
                details = FinishDetails.from_value(getattr(token, "finish_details", None))
                self._cancel_state(state, reason="timeout" if details.deadline_exceeded else "disconnect")

    def _normalize_speculative_output(
        self,
        state: _ChildState,
        output: GenerationOutput,
    ) -> GenerationOutput:
        token_ids = output.generated_token_ids
        if token_ids is None:
            raise RuntimeError("speculative output must publish committed token ids")
        tail = trim_speculative_output(
            token_ids,
            max_tokens=state.request.max_tokens,
            min_tokens=state.request.min_tokens,
            eos_token_id=state.request.eos_token_id,
            stop_token_ids=state.request.stop_token_ids,
            stop_token_sequences=state.request.stop_token_sequences,
            ignore_eos=state.request.ignore_eos,
        )
        if tail.token_ids == tuple(token_ids) and tail.finish_reason is None:
            return output
        text = output.text
        if tail.token_ids != tuple(token_ids):
            detokenize = getattr(self._driver, "detokenize", None)
            if callable(detokenize):
                text = str(detokenize(tail.token_ids))
        details = output.finish_details
        if tail.finish_reason is not None and (
            details is None or details.reason != tail.finish_reason
        ):
            details = FinishDetails(
                reason=tail.finish_reason,
                stop_sequence=tail.matched_stop_sequence,
            )
        return GenerationOutput(
            text=text,
            token_logprobs=output.token_logprobs[: len(tail.token_ids)],
            finish_details=details,
            telemetry=output.telemetry,
            generated_token_ids=tail.token_ids,
        )

    def _cancel_state(
        self,
        state: _ChildState,
        *,
        reason: str,
        error: BaseException | None = None,
    ) -> None:
        if state.terminal:
            return
        submission = state.submission
        reclaim_error: BaseException | None = None
        if submission is not None:
            driver_reason = reason if reason in {"cancel", "disconnect", "timeout"} else "cancel"
            try:
                abort = getattr(self._driver, "abort_submission", None)
                if callable(abort):
                    abort(submission, reason=driver_reason)
                else:
                    cancel = getattr(self._driver, "cancel_submission", None)
                    if callable(cancel):
                        cancel(submission, reason=driver_reason)
            except BaseException as exc:  # publish terminal even if reclaim reports a bug
                reclaim_error = exc
        if error is None and reclaim_error is not None:
            error = reclaim_error
        if error is None:
            if reason == "timeout":
                details = FinishDetails(reason="deadline_exceeded", deadline_exceeded=True)
            elif reason == "client_backpressure":
                details = FinishDetails(
                    reason="cancelled",
                    cancelled=True,
                    budget_pressure="client_backpressure",
                )
            else:
                details = FinishDetails(reason="cancelled", cancelled=True)
            error = GenerationCancelled(details)
        else:
            details = FinishDetails(reason="error")
        self._publish_terminal(state, generation_output=None, finish_details=details, error=error)

    def _publish_terminal(
        self,
        state: _ChildState,
        *,
        generation_output: GenerationOutput | None,
        finish_details: FinishDetails,
        error: BaseException | None = None,
    ) -> None:
        state.terminal = True
        self._states_by_service_id.pop(state.service_request_id, None)
        if state.backend_request_id is not None:
            self._states_by_backend_id.pop(state.backend_request_id, None)
        published = state.collector.publish(
            EngineOutput(
                kind=OutputKind.TERMINAL,
                request_id=state.service_request_id,
                generated_token_ids=tuple(state.token_ids),
                finish_reason=finish_details.reason,
                generation_output=generation_output,
                error=error,
            )
        )
        if not published:
            raise AssertionError("terminal publication must not block the engine driver")
        self._notify_output()

    def _notify_output(self) -> None:
        with self._output_condition:
            self._output_condition.notify_all()

    def _fail_all(self, error: BaseException) -> None:
        for state in tuple(self._states_by_service_id.values()):
            self._cancel_state(state, reason="driver_error", error=error)


def _stop_safe_chunks(
    state: _ChildState,
    *,
    token_id: int,
    chunk: GenerationStreamChunk,
) -> tuple[GenerationStreamChunk, ...]:
    """Hold only child chunks whose token can still complete a configured stop."""

    pending = state.pending_stop_chunks
    pending.append((int(token_id), chunk))
    finish = chunk.finish_details
    if finish is not None:
        suppressed = _suppressed_stop_suffix(pending, finish, state.request)
        if suppressed <= 0:
            output = tuple(item[1] for item in pending)
        else:
            output = (
                *(item[1] for item in pending[:-suppressed]),
                replace(chunk, text="", token_logprobs=()),
            )
        pending.clear()
        return output

    prefixes = frozenset(
        tuple(int(token) for token in sequence[:width])
        for sequence in state.request.stop_token_sequences
        for width in range(1, len(sequence))
    )
    token_ids = tuple(item[0] for item in pending)
    retained = max(
        (
            len(prefix)
            for prefix in prefixes
            if len(prefix) <= len(token_ids)
            and token_ids[-len(prefix) :] == prefix
        ),
        default=0,
    )
    emit_count = len(pending) - retained
    output = tuple(item[1] for item in pending[:emit_count])
    del pending[:emit_count]
    return output


def _suppressed_stop_suffix(
    pending: Sequence[tuple[int, GenerationStreamChunk]],
    finish: FinishDetails,
    request: GenerationRequest,
) -> int:
    if finish.stop_sequence:
        return min(len(pending), len(finish.stop_sequence))
    token_ids = tuple(item[0] for item in pending)
    if finish.reason == "stop":
        for sequence in request.stop_token_sequences:
            normalized = tuple(int(token) for token in sequence)
            if (
                normalized
                and len(normalized) <= len(token_ids)
                and token_ids[-len(normalized) :] == normalized
            ):
                return len(normalized)
        if token_ids and token_ids[-1] in set(request.stop_token_ids):
            return 1
    if finish.reason == "eos" and finish.eos_token_id is not None:
        return 1
    return 0


def _split_generation_request(request: GenerationRequest) -> tuple[GenerationRequest, ...]:
    children: list[GenerationRequest] = []
    for row_index, prompt in enumerate(request.prompts):
        row_seeds = (request.row_seeds[row_index],) if request.row_seeds else ()
        children.append(replace(request, prompts=(prompt,), row_seeds=row_seeds))
    return tuple(children)


def _stream_chunk_for_choice(
    chunk: GenerationStreamChunk,
    choice_index: int,
) -> GenerationStreamChunk:
    telemetry = chunk.telemetry
    if telemetry is not None:
        decode_state = replace(telemetry.decode_state, row_index=int(choice_index))
        telemetry = replace(telemetry, decode_state=decode_state)
    else:
        telemetry = None
    return replace(chunk, telemetry=telemetry)


__all__ = ["EngineService", "EngineServiceHandle"]
