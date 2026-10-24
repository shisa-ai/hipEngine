"""CPU coverage for the GGUF NextN proposal-graph cache lifecycle.

The executor is constructed with ``__new__`` so these tests exercise the cache,
binding and cleanup paths without a HIP device. Every graph handle is a plain
object and every runtime call is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_nextn as nextn
from hipengine.speculative.graph_stage import (
    GRAPH_STAGE_BINDING_MISS,
    GRAPH_STAGE_RELEASE_BACKLOG,
    SpecGraphStageCache,
    describe_failure,
)


def _graph(slot: int, budget: int) -> nextn._Qwen35GGUFNextNProposalGraph:
    """Build the real provider record so handle-zeroing is under test."""

    return nextn._Qwen35GGUFNextNProposalGraph(
        slot=int(slot),
        budget=int(budget),
        completion_event=0xE000 + int(slot) * 0x100 + int(budget),
        graph_exec=0xF000 + int(slot) * 0x100 + int(budget),
        graph=0x10000 + int(slot) * 0x100 + int(budget),
        stream=0x11000 + int(slot) * 0x100 + int(budget),
    )


class _Runtime:
    """Record every destroy, optionally failing one and rejecting duplicates."""

    def __init__(
        self,
        *,
        fail: str | None = None,
        fail_handle: int | None = None,
        fail_times: int | None = None,
        strict: bool = False,
    ) -> None:
        self.calls: list[tuple[str, int]] = []
        self.destroyed: set[tuple[str, int]] = set()
        self._fail = fail
        self._fail_handle = fail_handle
        self._fail_times = fail_times
        self._strict = bool(strict)
        self._failures_used = 0

    def _record(self, name: str, handle: int) -> None:
        handle = int(handle)
        if self._strict and (name, handle) in self.destroyed:
            raise RuntimeError(f"{name} refused duplicate destroy for {handle:#x}")
        self.calls.append((name, handle))
        matched = name == self._fail and (
            self._fail_handle is None or handle == int(self._fail_handle)
        )
        if matched and (self._fail_times is None or self._failures_used < self._fail_times):
            self._failures_used += 1
            raise RuntimeError(f"{name} failed for handle {handle:#x}")
        self.destroyed.add((name, handle))

    def attempts(self, name: str) -> list[int]:
        return [handle for call, handle in self.calls if call == name]

    def event_destroy(self, handle: int) -> None:
        self._record("event_destroy", handle)

    def graph_exec_destroy(self, handle: int) -> None:
        self._record("graph_exec_destroy", handle)

    def graph_destroy(self, handle: int) -> None:
        self._record("graph_destroy", handle)

    def stream_destroy(self, handle: int) -> None:
        self._record("stream_destroy", handle)


def _weight(pointer: int) -> object:
    return SimpleNamespace(
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=int(pointer)))
    )


class _Weights:
    """Minimal stand-in for the resident NextN weights the capture reads."""

    def __init__(self) -> None:
        self._fallback = {
            "token_embedding": _weight(0x20000),
            "output_norm": _weight(0x20100),
        }
        self._nextn = {
            "enorm": _weight(0x20200),
            "hnorm": _weight(0x20300),
            "eh_proj": _weight(0x20400),
        }

    def fallback(self, name: str) -> object:
        return self._fallback[name]

    def nextn(self, name: str) -> object:
        return self._nextn[name]


@dataclass
class _Scratch:
    """Dataclass scratch stand-in so pointer collection runs for real."""

    position_buf: DeviceBuffer
    context_buf: DeviceBuffer
    block_table: DeviceBuffer
    norm: DeviceBuffer
    attn_out: DeviceBuffer
    full_q: DeviceBuffer
    full_attn_split_partial: DeviceBuffer
    ffn_intermediate: DeviceBuffer
    layer_conv_states: tuple
    layer_recurrent_states: tuple
    full_key_caches: tuple
    full_value_caches: tuple
    max_positions: int
    slot_count: int
    blocks_per_slot: int
    kv_storage_dtype: str
    full_attn_split_count: int


def _executor(*, max_requests: int = 2, runtime: _Runtime | None = None) -> object:
    executor = nextn.Qwen35GGUFNextNExecutor.__new__(nextn.Qwen35GGUFNextNExecutor)
    executor.runtime = runtime or _Runtime()
    executor.max_requests = int(max_requests)
    executor.hidden_size = 5_120
    executor.vocab_size = 248_320
    executor.last_lm_head_path = "unobserved"
    executor._lm_head_top1_mapped = True
    executor._lm_head_top1_vocab_size = 32_768
    executor._lm_head_top1_token_map_ptr = 0x5000
    executor._token_buf = DeviceBuffer(0x1000, 8 * max_requests)
    executor._proposal_target_hidden = DeviceBuffer(0x2000, 5120 * 2 * max_requests)
    executor._proposal_results = DeviceBuffer(0x3000, 8 * 8 * max_requests)
    executor._proposal_history_hidden = DeviceBuffer(0x4000, 8 * 5120 * 2 * max_requests)
    executor._embedding_buf = DeviceBuffer(0x7000, 5120 * 2 * max_requests)
    executor._fusion_buf = DeviceBuffer(0x7100, 2 * 5120 * 2 * max_requests)
    executor._fused_buf = DeviceBuffer(0x7200, 5120 * 2 * max_requests)
    executor._layer_out_buf = DeviceBuffer(0x7300, 5120 * 2 * max_requests)
    executor._final_hidden_buf = DeviceBuffer(0x7400, 5120 * 2 * max_requests)
    executor._logits_buf = DeviceBuffer(0x7500, 5120 * 4)
    executor._lm_head_top1_block_values = DeviceBuffer(0x7600, 128)
    executor._lm_head_top1_block_indices = DeviceBuffer(0x7700, 128)
    executor._lm_head_top1_result = DeviceBuffer(0x7800, 8)
    executor._lm_head_top1_weight = _weight(0x20500)
    executor._proposal_graph_runtime_library = object()
    executor.weights = _Weights()
    executor.runner = SimpleNamespace(weights=object())
    executor.scratch = _Scratch(
        position_buf=DeviceBuffer(0x6000, 8 * max_requests),
        context_buf=DeviceBuffer(0x6100, 8 * max_requests),
        block_table=DeviceBuffer(0x6200, 64 * max_requests),
        norm=DeviceBuffer(0x6100, 5120 * 2),
        attn_out=DeviceBuffer(0x6200, 5120 * 2),
        full_q=DeviceBuffer(0x6300, 5120 * 2),
        full_attn_split_partial=DeviceBuffer(0x6400, 4096),
        ffn_intermediate=DeviceBuffer(0x6500, 5120 * 4),
        layer_conv_states=(DeviceBuffer(0x6300, 128), None),
        layer_recurrent_states=(DeviceBuffer(0x6400, 128),),
        full_key_caches=(DeviceBuffer(0x6500, 4096),),
        full_value_caches=(DeviceBuffer(0x6600, 4096),),
        max_positions=32_768,
        slot_count=int(max_requests),
        blocks_per_slot=64,
        kv_storage_dtype="bf16",
        full_attn_split_count=1,
    )
    executor._proposal_graphs = SpecGraphStageCache(
        max_entries=nextn._PROPOSAL_GRAPH_CACHE_MAX_ENTRIES * int(max_requests)
    )
    executor._proposal_graph_partials = []
    executor._proposal_graph_captures = 0
    executor._proposal_graph_replays = 0
    executor._proposal_graph_last_status = "ready"
    executor._proposal_graph_last_error = None
    return executor


def _capture_stub(executor: object, *, fail_first: bool = False) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []
    remaining = {"failures": 1 if fail_first else 0}

    def capture(request_id: int, slot: int, budget: int) -> _graph:
        calls.append((int(slot), int(budget)))
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise RuntimeError("HIP returned a null GGUF NextN proposal graph")
        return _graph(slot, budget)

    executor._capture_exact_chain_graph = capture
    return calls


def test_nextn_proposal_graph_reuses_a_cached_capture() -> None:
    executor = _executor()
    calls = _capture_stub(executor)

    first = executor._proposal_graph(7, 0, 3)
    second = executor._proposal_graph(7, 0, 3)

    assert first is second
    assert calls == [(0, 3)]
    assert executor._proposal_graph_captures == 1
    assert executor._proposal_graph_last_status == "captured"
    assert executor._proposal_graph_last_error is None
    assert len(executor._proposal_graphs) == 1


def test_nextn_proposal_graph_recaptures_after_address_drift() -> None:
    runtime = _Runtime()
    executor = _executor(runtime=runtime)
    calls = _capture_stub(executor)

    first = executor._proposal_graph(7, 0, 3)
    assert first is not None
    stale = (first.stream, first.graph, first.graph_exec, first.completion_event)

    # A replacement scratch allocation changes the captured pointer set.
    executor._token_buf = DeviceBuffer(0x9000, 8 * executor.max_requests)

    second = executor._proposal_graph(7, 0, 3)

    assert second is not None
    assert second is not first
    assert calls == [(0, 3), (0, 3)]
    assert executor._proposal_graph_captures == 2
    assert ("stream_destroy", stale[0]) in runtime.calls
    assert ("graph_destroy", stale[1]) in runtime.calls
    # The released record zeroes its handles so a retry cannot re-destroy them.
    assert (first.stream, first.graph, first.graph_exec, first.completion_event) == (0, 0, 0, 0)
    assert executor._proposal_graph_last_status == "captured"


def test_nextn_proposal_graph_capture_failure_is_retryable() -> None:
    executor = _executor()
    calls = _capture_stub(executor, fail_first=True)

    assert executor._proposal_graph(7, 0, 3) is None
    assert executor._proposal_graph_last_status == "capture_fallback"
    assert "HIP returned a null GGUF NextN proposal graph" in executor._proposal_graph_last_error
    assert executor.proposal_graph_contract()["unavailable"] == [[0, 3]]

    graph = executor._proposal_graph(7, 0, 3)

    assert graph is not None
    assert calls == [(0, 3), (0, 3)]
    assert executor._proposal_graph_last_status == "captured"
    assert executor._proposal_graph_last_error is None
    assert executor.proposal_graph_contract()["unavailable"] == []


def test_nextn_proposal_graph_allocation_failure_keeps_route_reachable() -> None:
    executor = _executor()

    def exhausted(request_id: int, slot: int, budget: int) -> _graph:
        raise MemoryError("hipMalloc failed for the proposal stream")

    executor._capture_exact_chain_graph = exhausted

    assert executor._proposal_graph(7, 0, 2) is None
    assert executor._proposal_graph_last_status == "capture_fallback"
    assert "MemoryError" in executor._proposal_graph_last_error
    assert executor.proposal_graph_contract()["unavailable"] == [[0, 2]]

    calls = _capture_stub(executor)
    graph = executor._proposal_graph(7, 0, 2)

    assert graph is not None
    assert calls == [(0, 2)]
    assert executor.proposal_graph_contract()["unavailable"] == []


def test_nextn_cached_only_lookup_rejects_a_drifted_binding() -> None:
    executor = _executor()
    _capture_stub(executor)

    graph = executor._proposal_graph(7, 0, 3)
    key = (0, 3)
    assert executor._proposal_graphs.acquire(key, executor._proposal_graph_binding(slot=0, budget=3))[0] is graph

    executor._proposal_results = DeviceBuffer(0x9100, 8 * 8 * executor.max_requests)

    reused, reason = executor._proposal_graphs.acquire(
        key,
        executor._proposal_graph_binding(slot=0, budget=3),
    )

    assert reused is None
    assert reason == GRAPH_STAGE_BINDING_MISS
    assert len(executor._proposal_graphs) == 0


def test_nextn_proposal_graph_cleanup_releases_every_handle_after_one_failure() -> None:
    graph = _graph(0, 3)
    runtime = _Runtime(fail="graph_exec_destroy", fail_handle=graph.graph_exec)
    executor = _executor(runtime=runtime)

    with pytest.raises(BaseExceptionGroup) as excinfo:
        executor._close_proposal_graph(graph)

    attempted = {name for name, _handle in runtime.calls}
    assert attempted == {"event_destroy", "graph_exec_destroy", "graph_destroy", "stream_destroy"}
    assert len(excinfo.value.exceptions) == 1
    assert "graph_exec_destroy failed" in str(excinfo.value.exceptions[0])


def test_nextn_retire_closes_every_graph_and_reports_the_failure() -> None:
    executor = _executor()
    _capture_stub(executor)
    first = executor._proposal_graph(7, 0, 1)
    second = executor._proposal_graph(7, 1, 1)
    assert first is not None and second is not None

    runtime = _Runtime(fail="event_destroy", fail_handle=first.completion_event)
    executor.runtime = runtime

    retired = executor._retire_proposal_graphs(reason="executor_closed")

    assert sorted(retired) == [(0, 1), (1, 1)]
    assert len(executor._proposal_graphs) == 0
    # A permanent refusal is retried once, then retained for the owner.
    assert executor._proposal_graphs.close_failure_count == 2
    assert executor._proposal_graphs.pending_close_count == 1
    # The surviving graph's resources were still released.
    released = {name for name, _handle in runtime.calls}
    assert {"graph_destroy", "stream_destroy"} <= released
    cleanup = executor.proposal_graph_cleanup()
    assert cleanup["pending_close"] == [[0, 1]]
    assert cleanup["pending_close_count"] == 1
    assert cleanup["close_failure_count"] == 2
    assert "event_destroy failed" in cleanup["close_failures"][0][1]


def test_nextn_retire_recovers_a_transient_release_failure() -> None:
    executor = _executor()
    _capture_stub(executor)
    graph = executor._proposal_graph(7, 0, 2)
    assert graph is not None

    runtime = _Runtime(fail="graph_exec_destroy", fail_handle=graph.graph_exec)
    original = runtime.graph_exec_destroy
    remaining = {"failures": 1}

    def flaky(handle: int) -> None:
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            original(handle)
            return
        runtime.calls.append(("graph_exec_destroy", int(handle)))

    runtime.graph_exec_destroy = flaky
    executor.runtime = runtime

    executor._retire_proposal_graphs(reason="executor_closed")

    assert executor._proposal_graphs.close_failure_count == 1
    assert executor._proposal_graphs.pending_close_count == 0
    cleanup = executor.proposal_graph_cleanup()
    assert cleanup["pending_close"] == []
    assert cleanup["close_failure_count"] == 1


def test_nextn_proposal_graph_cleanup_reports_a_clean_teardown() -> None:
    executor = _executor()
    _capture_stub(executor)
    assert executor._proposal_graph(7, 0, 1) is not None

    executor._retire_proposal_graphs(reason="executor_closed")

    assert executor.proposal_graph_cleanup() == {
        "reusable": 0,
        "pending_partial_releases": 0,
        "pending_partial_handles": [],
        "pending_close": [],
        "pending_close_count": 0,
        "close_failure_count": 0,
        "close_failures": [],
    }


def test_nextn_proposal_graph_drift_then_capture_failure_keeps_both_causes() -> None:
    executor = _executor()
    _capture_stub(executor)
    assert executor._proposal_graph(7, 0, 3) is not None

    executor._token_buf = DeviceBuffer(0x9000, 8 * executor.max_requests)

    def exhausted(request_id: int, slot: int, budget: int) -> _graph:
        raise MemoryError("hipMalloc failed for the proposal stream")

    executor._capture_exact_chain_graph = exhausted

    assert executor._proposal_graph(7, 0, 3) is None
    assert executor._proposal_graph_last_status == "capture_fallback"
    assert GRAPH_STAGE_BINDING_MISS in executor._proposal_graph_last_error
    assert "MemoryError" in executor._proposal_graph_last_error


def test_nextn_proposal_graph_contract_keeps_its_existing_keys() -> None:
    executor = _executor()

    contract = executor.proposal_graph_contract()

    assert set(contract) == {
        "eligible",
        "budgets",
        "max_context",
        "graphs",
        "captures",
        "replays",
        "unavailable",
        "last_status",
        "last_error",
        "lm_head_path",
        "lm_head_mapped",
        "lm_head_score_vocab",
        "lm_head_full_vocab",
    }
    assert contract["eligible"] is True
    assert contract["graphs"] == 0
    assert contract["unavailable"] == []
    assert contract["last_status"] == "ready"


def test_nextn_proposal_graph_release_skips_destroyed_handles() -> None:
    graph = _graph(0, 3)
    event, graph_exec, graph_handle, stream = (
        graph.completion_event,
        graph.graph_exec,
        graph.graph,
        graph.stream,
    )
    runtime = _Runtime(fail="graph_destroy", fail_times=1, strict=True)
    executor = _executor(runtime=runtime)

    with pytest.raises(BaseExceptionGroup) as first:
        executor._close_proposal_graph(graph)

    assert len(first.value.exceptions) == 1
    assert runtime.attempts("event_destroy") == [event]
    assert runtime.attempts("graph_exec_destroy") == [graph_exec]
    assert runtime.attempts("stream_destroy") == [stream]
    # Only the refused handle survives.
    assert graph.completion_event == 0
    assert graph.graph_exec == 0
    assert graph.stream == 0
    assert graph.graph != 0

    # The retry touches the refused handle only; the strict runtime would raise
    # on any duplicate destroy.
    executor._close_proposal_graph(graph)

    assert graph.graph == 0
    assert len(runtime.attempts("graph_destroy")) == 2
    assert len(runtime.attempts("event_destroy")) == 1
    assert len(runtime.attempts("graph_exec_destroy")) == 1
    assert len(runtime.attempts("stream_destroy")) == 1


def test_nextn_proposal_graph_release_recovers_a_transient_failure() -> None:
    graph = _graph(1, 2)
    runtime = _Runtime(fail="event_destroy", fail_times=1, strict=True)
    executor = _executor(runtime=runtime)

    with pytest.raises(BaseExceptionGroup):
        executor._close_proposal_graph(graph)

    assert graph.completion_event != 0
    assert graph.graph_exec == 0 and graph.graph == 0 and graph.stream == 0

    executor._close_proposal_graph(graph)

    assert graph.completion_event == 0
    assert len(runtime.attempts("event_destroy")) == 2
    assert len(runtime.attempts("graph_exec_destroy")) == 1
    assert len(runtime.attempts("graph_destroy")) == 1
    assert len(runtime.attempts("stream_destroy")) == 1


def _closable_executor(monkeypatch, runtime: _Runtime):
    executor = _executor(runtime=runtime)
    _capture_stub(executor)
    graph = executor._proposal_graph(7, 0, 1)
    assert graph is not None

    events: list[str] = []
    buffers = (object(), object())

    def fake_free(buffer: object, *, runtime: _Runtime) -> None:
        events.append("buffer")

    monkeypatch.setattr(nextn, "free", fake_free)
    executor.closed = False
    executor._buffers = buffers
    executor._prompt_priming_staging = {}
    executor._prompt_prime_rows = {}
    executor._batch_session = SimpleNamespace(close=lambda: events.append("batch_session"))
    executor.runner = SimpleNamespace(close=lambda: events.append("runner"))
    executor._owns_weights = True
    executor.weights = SimpleNamespace(free=lambda **kwargs: events.append("weights"))
    return executor, graph, events


def test_nextn_executor_close_reports_unresolved_graph_failure(monkeypatch) -> None:
    runtime = _Runtime(fail="event_destroy", fail_handle=None)
    executor, graph, events = _closable_executor(monkeypatch, runtime)
    handle = graph.completion_event

    with pytest.raises(BaseExceptionGroup) as first:
        executor.close()

    assert events == ["buffer", "buffer", "batch_session", "runner", "weights"]
    assert len(first.value.exceptions) == 1
    assert "event_destroy failed" in describe_failure(first.value.exceptions[0])
    assert executor.closed is True
    # The first close retires every graph and then retries the refused one.
    assert runtime.attempts("event_destroy") == [handle, handle]

    # A later close retries the retained graph handle only: buffers, session,
    # runner and weights were already released.
    with pytest.raises(BaseExceptionGroup):
        executor.close()

    assert events == ["buffer", "buffer", "batch_session", "runner", "weights"]
    assert runtime.attempts("event_destroy") == [handle, handle, handle]
    assert len(runtime.attempts("graph_exec_destroy")) == 1
    assert len(runtime.attempts("graph_destroy")) == 1
    assert len(runtime.attempts("stream_destroy")) == 1

    # Once the runtime accepts the release, close stops raising.
    runtime._fail = None
    executor.close()

    assert runtime.attempts("event_destroy") == [handle, handle, handle, handle]
    assert executor.proposal_graph_cleanup()["pending_close"] == []


def test_nextn_executor_close_propagates_a_base_exception_after_cleanup(monkeypatch) -> None:
    executor = _executor()
    _capture_stub(executor)
    assert executor._proposal_graph(7, 0, 1) is not None

    events: list[str] = []
    buffers = (object(), object())

    def fake_free(buffer: object, *, runtime: _Runtime) -> None:
        if buffer is buffers[0]:
            raise KeyboardInterrupt("interrupted during teardown")
        events.append("buffer")

    monkeypatch.setattr(nextn, "free", fake_free)
    executor.closed = False
    executor._buffers = buffers
    executor._prompt_priming_staging = {}
    executor._prompt_prime_rows = {}
    executor._batch_session = SimpleNamespace(close=lambda: events.append("batch_session"))
    executor.runner = SimpleNamespace(close=lambda: events.append("runner"))
    executor._owns_weights = False
    executor.weights = None

    with pytest.raises(BaseExceptionGroup) as excinfo:
        executor.close()

    # The interrupt is carried out to the owner, not swallowed, and every other
    # resource was still released first.
    assert not isinstance(excinfo.value, ExceptionGroup)
    assert any(isinstance(item, KeyboardInterrupt) for item in excinfo.value.exceptions)
    assert events == ["buffer", "batch_session", "runner"]


def test_nextn_close_does_not_retry_away_a_graph_interrupt(monkeypatch) -> None:
    runtime = _Runtime(strict=True)
    executor, _record, events = _closable_executor(monkeypatch, runtime)
    original = runtime.graph_destroy
    interrupted = False

    def interrupt_once(handle):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("graph release interrupted")
        original(handle)

    runtime.graph_destroy = interrupt_once
    with pytest.raises(BaseExceptionGroup) as error:
        executor.close()
    assert not isinstance(error.value, Exception)
    assert "graph release interrupted" in describe_failure(error.value)
    assert events
    executor.close()
    assert executor._proposal_graphs.pending_close_count == 0


def test_nextn_proposal_graph_reuses_after_the_capture_binds_its_library() -> None:
    executor = _executor()
    executor._proposal_graph_runtime_library = None
    calls: list[tuple[int, int]] = []

    def capture(request_id: int, slot: int, budget: int) -> object:
        calls.append((int(slot), int(budget)))
        # The real capture builds the runtime state on its first call.
        executor._proposal_graph_runtime_library = object()
        return _graph(slot, budget)

    executor._capture_exact_chain_graph = capture

    first = executor._proposal_graph(7, 0, 3)
    second = executor._proposal_graph(7, 0, 3)

    assert first is second
    assert calls == [(0, 3)]
    assert executor._proposal_graph_captures == 1


def test_nextn_proposal_graph_binding_covers_intermediate_scratch() -> None:
    executor = _executor()

    addresses = dict(executor._proposal_graph_binding(slot=0, budget=3).addresses)

    # Every intermediate row the captured chain writes must be fingerprinted, not
    # just the position/context/KV set.
    for name in (
        "scratch.norm",
        "scratch.attn_out",
        "scratch.full_q",
        "scratch.full_attn_split_partial",
        "scratch.ffn_intermediate",
        "scratch.position_buf",
        "embedding",
        "fusion",
        "fused",
        "layer_out",
        "final_hidden",
        "logits",
        "weights.token_embedding",
        "weights.output_norm",
        "weights.enorm",
        "weights.hnorm",
        "weights.eh_proj",
        "weights.lm_head_top1",
    ):
        assert name in addresses, name


def test_nextn_proposal_graph_recaptures_after_intermediate_scratch_drift() -> None:
    executor = _executor()
    calls = _capture_stub(executor)

    first = executor._proposal_graph(7, 0, 3)
    assert first is not None
    stale_stream = first.stream

    # Replacing one intermediate scratch row changes a pointer the captured
    # kernels dereference. The position buffer is untouched.
    executor.scratch.attn_out = DeviceBuffer(0x9999, 5120 * 2)

    second = executor._proposal_graph(7, 0, 3)

    assert second is not None
    assert second is not first
    assert calls == [(0, 3), (0, 3)]
    assert executor._proposal_graph_captures == 2
    assert ("stream_destroy", stale_stream) in executor.runtime.calls
    assert executor._proposal_graph_last_status == "captured"


def test_nextn_binding_tracks_repacked_weights_without_raw_allocation() -> None:
    executor = _executor()
    buffers = {"quants": SimpleNamespace(tensor=SimpleNamespace(ptr=0xA001)),
               "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=0xA002))}
    executor.weights._nextn["eh_proj"] = nextn.Qwen35GGUFDeviceWeight(
        spec=None, allocations=buffers, backend="hip_gfx1151",
    )
    first = executor._proposal_graph_binding(slot=0, budget=1)
    assert 0xA001 in dict(first.addresses).values()
    assert 0xA002 in dict(first.addresses).values()
    buffers["scales"] = SimpleNamespace(tensor=SimpleNamespace(ptr=0xA003))
    second = executor._proposal_graph_binding(slot=0, budget=1)
    assert first != second


def test_nextn_partial_release_prevents_more_capture_allocations() -> None:
    runtime = _Runtime(fail="graph_destroy", strict=True)
    executor = _executor(runtime=runtime)
    executor._release_capture_partial(
        stream=0x11000, graph=0x10000, graph_exec=0xF000,
        completion_event=0xE000, capturing=False, slot=0, budget=1,
    )
    captures = []
    executor._capture_exact_chain_graph = lambda *args: captures.append(args) or _graph(0, 1)
    for _ in range(10):
        assert executor._proposal_graph(0, 0, 1) is None
    assert captures == []
    assert len(executor._proposal_graph_partials) == 1
    assert "graph_destroy failed" in executor._proposal_graph_last_error
    runtime._fail = None
    assert executor._proposal_graph(0, 0, 1) is not None
    assert len(captures) == 1
    assert executor._proposal_graph_partials == []


def test_nextn_capture_partial_release_is_retained_and_retried() -> None:
    runtime = _Runtime(fail="graph_destroy", fail_times=1, strict=True)
    executor = _executor(runtime=runtime)

    unresolved = executor._release_capture_partial(
        slot=0,
        budget=2,
        stream=0x11000,
        graph=0x10000,
        graph_exec=0xF000,
        completion_event=0xE000,
        capturing=False,
    )

    assert len(unresolved) == 1
    assert "graph_destroy failed" in describe_failure(unresolved[0])
    # The refused handle is owned, not dropped.
    assert len(executor._proposal_graph_partials) == 1
    cleanup = executor.proposal_graph_cleanup()
    assert cleanup["pending_partial_releases"] == 1
    assert cleanup["pending_partial_handles"] == [[0x10000]]
    assert runtime.attempts("event_destroy") == [0xE000]
    assert runtime.attempts("graph_exec_destroy") == [0xF000]
    assert runtime.attempts("stream_destroy") == [0x11000]

    # The retry releases the refused handle only; the strict runtime rejects any
    # duplicate destroy.
    assert executor._retry_proposal_graph_partials() == []

    assert executor._proposal_graph_partials == []
    assert runtime.attempts("graph_destroy") == [0x10000, 0x10000]
    assert runtime.attempts("event_destroy") == [0xE000]
    assert executor.proposal_graph_cleanup()["pending_partial_handles"] == []


def test_nextn_close_reports_a_retained_capture_partial(monkeypatch) -> None:
    runtime = _Runtime(fail="stream_destroy", strict=True)
    executor, _graph_record, _events = _closable_executor(monkeypatch, runtime)

    executor._release_capture_partial(
        slot=0,
        budget=2,
        stream=0x11000,
        graph=0x10000,
        graph_exec=0xF000,
        completion_event=0xE000,
        capturing=False,
    )
    assert executor.proposal_graph_cleanup()["pending_partial_handles"] == [[0x11000]]

    with pytest.raises(BaseExceptionGroup) as first:
        executor.close()

    assert any("stream_destroy failed" in describe_failure(item) for item in first.value.exceptions)
    assert executor.proposal_graph_cleanup()["pending_partial_releases"] == 1

    # A later close retries the retained partial and nothing else.
    runtime._fail = None
    executor.close()

    assert runtime.attempts("stream_destroy").count(0x11000) == 3
    assert executor.proposal_graph_cleanup()["pending_partial_releases"] == 0
