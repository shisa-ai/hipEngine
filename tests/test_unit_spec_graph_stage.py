"""CPU contract for the shared speculative stage graph cache.

The cache owns bounded lookup, capture-identity comparison, eviction,
invalidation and close-all. It owns no HIP resource: a provider graph is stored
as an opaque value with the provider's own close callable, so these tests use
plain objects and recorders instead of a device runtime.
"""

from __future__ import annotations

import pytest

from hipengine.speculative.graph_stage import (
    GRAPH_STAGE_BINDING_MISS,
    GRAPH_STAGE_CAPTURE_FAILED,
    GRAPH_STAGE_CLOSED,
    GRAPH_STAGE_RELEASE_BACKLOG,
    SpecGraphBinding,
    SpecGraphStageCache,
    describe_failure,
)


class _Recorder:
    """Close callable that records every value it is asked to release."""

    def __init__(self) -> None:
        self.closed: list[object] = []

    def __call__(self, value: object) -> None:
        self.closed.append(value)


class _Graph:
    """Stand-in for a provider-owned graph handle."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Graph({self.name})"


def _binding(*, slot: int = 0, budget: int = 1, token_ptr: int = 0x1000) -> SpecGraphBinding:
    return SpecGraphBinding.build(
        bucket=(slot, budget),
        addresses={"token": token_ptr, "position": 0x2000},
        configuration=("bf16", 5120),
    )


def test_spec_graph_binding_reports_a_mismatched_axis() -> None:
    base = _binding()

    assert base.ineligibility_reason(_binding()) is None
    assert base.ineligibility_reason(_binding(token_ptr=0x9000)) == GRAPH_STAGE_BINDING_MISS
    assert base.ineligibility_reason(_binding(slot=1)) == GRAPH_STAGE_BINDING_MISS
    assert base.ineligibility_reason(_binding(budget=3)) == GRAPH_STAGE_BINDING_MISS
    other_config = SpecGraphBinding.build(
        bucket=(0, 1),
        addresses={"token": 0x1000, "position": 0x2000},
        configuration=("bf16", 4096),
    )
    assert base.ineligibility_reason(other_config) == GRAPH_STAGE_BINDING_MISS


def test_spec_graph_binding_normalizes_address_order() -> None:
    forward = SpecGraphBinding.build(addresses=(("a", 1), ("b", 2)))
    reverse = SpecGraphBinding.build(addresses=(("b", 2), ("a", 1)))
    mapping = SpecGraphBinding.build(addresses={"b": 2, "a": 1})

    assert forward == reverse == mapping
    assert forward.addresses == (("a", 1), ("b", 2))


def test_spec_graph_stage_cache_reuses_a_matching_binding() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()
    graph = _Graph("first")
    captures: list[str] = []

    def capture() -> _Graph:
        captures.append("capture")
        return graph

    stored, reason = cache.capture((0, 1), _binding(), capture, recorder)

    assert (stored, reason) == (graph, None)
    assert captures == ["capture"]
    assert cache.acquire((0, 1), _binding()) == (graph, None)
    assert cache.acquire((0, 1), _binding()) == (graph, None)
    assert captures == ["capture"]
    assert recorder.closed == []


def test_spec_graph_stage_cache_evicts_the_least_recently_used_entry() -> None:
    cache = SpecGraphStageCache(max_entries=2)
    recorder = _Recorder()

    for slot in range(2):
        cache.capture((slot, 1), _binding(slot=slot), lambda slot=slot: _Graph(f"g{slot}"), recorder)

    # Touch slot 0 so slot 1 becomes the least recently used entry.
    assert cache.acquire((0, 1), _binding(slot=0))[0] is not None
    cache.capture((2, 1), _binding(slot=2), lambda: _Graph("g2"), recorder)

    assert len(cache) == 2
    assert cache.acquire((0, 1), _binding(slot=0))[0] is not None
    assert cache.acquire((2, 1), _binding(slot=2))[0] is not None
    assert [graph.name for graph in recorder.closed] == ["g1"]


def test_spec_graph_stage_cache_replaces_on_address_drift() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()
    first = _Graph("first")
    second = _Graph("second")
    captures: list[_Graph] = [first, second]

    cache.capture((0, 1), _binding(token_ptr=0x1000), lambda: captures.pop(0), recorder)

    # A drifted binding must not be replayed, and must close the stale entry.
    reused, reason = cache.acquire((0, 1), _binding(token_ptr=0x9000))

    assert reused is None
    assert reason == GRAPH_STAGE_BINDING_MISS
    assert recorder.closed == [first]
    assert len(cache) == 0
    # The drift leaves the key capturable again.
    assert cache.acquire((0, 1), _binding(token_ptr=0x9000)) == (None, None)
    stored, capture_reason = cache.capture((0, 1), _binding(token_ptr=0x9000), lambda: captures.pop(0), recorder)
    assert (stored, capture_reason) == (second, None)


def test_spec_graph_stage_cache_closes_a_replaced_value_once() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()
    first = _Graph("first")
    second = _Graph("second")

    cache.capture((0, 1), _binding(), lambda: first, recorder)
    stored, reason = cache.capture((0, 1), _binding(), lambda: second, recorder)

    assert (stored, reason) == (second, None)
    assert recorder.closed == [first]
    assert cache.acquire((0, 1), _binding()) == (second, None)


def test_spec_graph_stage_cache_capture_failure_stays_reachable() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()
    attempts: list[int] = []

    def failing() -> _Graph:
        attempts.append(1)
        raise RuntimeError("HIP returned a null proposal graph")

    failed, reason = cache.capture((0, 1), _binding(), failing, recorder)

    assert failed is None
    assert reason == GRAPH_STAGE_CAPTURE_FAILED
    assert len(cache) == 0
    assert cache.failed_keys() == ((0, 1),)
    assert "HIP returned a null proposal graph" in (cache.last_failure((0, 1)) or "")
    assert cache.last_reason((0, 1)) == GRAPH_STAGE_CAPTURE_FAILED

    # No latch: the same key is capturable on the next attempt.
    graph = _Graph("recovered")
    stored, capture_reason = cache.capture((0, 1), _binding(), lambda: graph, recorder)

    assert (stored, capture_reason) == (graph, None)
    assert cache.failed_keys() == ()
    assert cache.last_failure((0, 1)) is None
    assert recorder.closed == []


def test_spec_graph_stage_cache_allocation_failure_is_a_capture_failure() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()

    def exhausted() -> _Graph:
        raise MemoryError("hipMalloc failed for 2147483648 bytes")

    failed, reason = cache.capture((3, 2), _binding(slot=3, budget=2), exhausted, recorder)

    assert failed is None
    assert reason == GRAPH_STAGE_CAPTURE_FAILED
    assert len(cache) == 0
    assert "MemoryError" in (cache.last_failure((3, 2)) or "")
    assert cache.acquire((3, 2), _binding(slot=3, budget=2)) == (None, None)


def test_spec_graph_stage_cache_invalidate_clears_a_recorded_capture_failure() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()

    def failing() -> _Graph:
        raise RuntimeError("capture failed")

    cache.capture((0, 1), _binding(), failing, recorder)
    assert cache.failed_keys() == ((0, 1),)

    cache.invalidate((0, 1), reason="scratch_reserved")

    assert cache.failed_keys() == ()
    assert cache.last_failure((0, 1)) is None
    assert cache.last_reason((0, 1)) == "scratch_reserved"


def test_spec_graph_stage_cache_bounds_diagnostic_history() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()

    def failing() -> _Graph:
        raise RuntimeError("capture failed")

    for slot in range(200):
        cache.capture((slot, 1), _binding(slot=slot), failing, recorder)

    assert len(cache) == 0
    assert len(cache.failed_keys()) <= 64
    assert cache.last_failure((199, 1)) is not None
    assert cache.last_failure((0, 1)) is None

    for slot in range(200):
        cache.invalidate((slot, 1), reason="scratch_reserved")

    assert cache.last_reason((199, 1)) == "scratch_reserved"
    assert cache.last_reason((0, 1)) is None


def test_spec_graph_stage_cache_retires_every_entry_after_one_cleanup_failure() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    released: list[str] = []

    def close(value: object) -> None:
        name = getattr(value, "name", "?")
        if name == "g1":
            raise RuntimeError("hipGraphExecDestroy failed")
        released.append(str(name))

    for slot in range(3):
        cache.capture((slot, 1), _binding(slot=slot), lambda slot=slot: _Graph(f"g{slot}"), close)

    retired = cache.close()

    assert len(retired) == 3
    assert sorted(released) == ["g0", "g2"]
    assert len(cache) == 0
    assert cache.close_failure_count == 1
    assert [key for key, _detail in cache.close_failures] == [(1, 1)]
    assert "hipGraphExecDestroy failed" in cache.close_failures[0][1]
    # The refused release is retained, not dropped: its handles stay reachable.
    assert cache.pending_close_count == 1
    assert [key for key, _value in cache.pending_closes()] == [(1, 1)]
    assert cache.acquire((0, 1), _binding()) == (None, GRAPH_STAGE_CLOSED)


def test_spec_graph_stage_cache_retains_a_failed_release_for_retry() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    graph = _Graph("g0")
    attempts: list[int] = []

    def close(value: object) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("hipStreamDestroy failed")

    cache.capture((0, 1), _binding(), lambda: graph, close)

    assert cache.close() == ((0, 1),)
    assert cache.pending_close_count == 1
    assert cache.pending_closes() == (((0, 1), graph),)
    assert cache.close_failure_count == 1

    # The owner retries and the retained handle is released exactly once.
    assert cache.retry_closes() == ()
    assert cache.pending_close_count == 0
    assert cache.close_failure_count == 1
    assert attempts == [1, 1]


def test_spec_graph_stage_cache_retained_release_survives_a_base_exception() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    released: list[str] = []

    def close(value: object) -> None:
        name = getattr(value, "name", "?")
        if name == "g1":
            raise KeyboardInterrupt("interrupted during teardown")
        released.append(str(name))

    for slot in range(3):
        cache.capture((slot, 1), _binding(slot=slot), lambda slot=slot: _Graph(f"g{slot}"), close)

    # A non-Exception release failure must not abort the remaining removals.
    assert len(cache.close()) == 3

    assert sorted(released) == ["g0", "g2"]
    assert cache.pending_close_count == 1
    assert cache.pending_close_keys() == ((1, 1),)
    assert "KeyboardInterrupt" in cache.close_failures[0][1]


def test_spec_graph_stage_cache_close_failure_total_outlives_retained_details() -> None:
    cache = SpecGraphStageCache(max_entries=64)

    def close(value: object) -> None:
        raise RuntimeError("hipGraphDestroy failed")

    for slot in range(40):
        cache.capture((slot, 1), _binding(slot=slot), lambda slot=slot: _Graph(f"g{slot}"), close)

    cache.close()

    assert cache.close_failure_count == 40
    assert len(cache.close_failures) == 32
    assert cache.pending_close_count == 40


def test_spec_graph_stage_cache_invalidate_closes_one_entry_and_is_idempotent() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    recorder = _Recorder()
    graph = _Graph("g0")

    cache.capture((0, 1), _binding(), lambda: graph, recorder)

    assert cache.invalidate((0, 1), reason="scratch_reserved") is True
    assert cache.invalidate((0, 1), reason="scratch_reserved") is False
    assert recorder.closed == [graph]
    assert cache.last_reason((0, 1)) == "scratch_reserved"
    assert len(cache) == 0


def test_spec_graph_stage_cache_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        SpecGraphStageCache(max_entries=0)

    cache = SpecGraphStageCache(max_entries=1)
    with pytest.raises(TypeError):
        cache.capture((0, 1), "not-a-binding", lambda: _Graph("g"), _Recorder())
    with pytest.raises(TypeError):
        cache.capture((0, 1), _binding(), None, _Recorder())


def test_spec_graph_stage_cache_refuses_capture_while_release_backlog_is_full() -> None:
    cache = SpecGraphStageCache(max_entries=4, max_pending=2)
    graphs = [_Graph(f"g{index}") for index in range(2)]
    failing = {"on": True}

    def close(value: object) -> None:
        if failing["on"]:
            raise RuntimeError("hipGraphDestroy failed")

    for slot, graph in enumerate(graphs):
        cache.capture((slot, 1), _binding(slot=slot), lambda graph=graph: graph, close)

    cache.retire(reason="reset")
    assert cache.pending_close_count == 2

    # A named resource refusal, not an evidence gate: no new graph is allocated
    # while two provider releases are outstanding, and no handle is dropped.
    stored, reason = cache.capture((9, 1), _binding(slot=9), lambda: _Graph("g9"), close)

    assert stored is None
    assert reason == GRAPH_STAGE_RELEASE_BACKLOG
    assert len(cache) == 0
    assert cache.pending_close_count == 2
    assert [value for _key, value in cache.pending_closes()] == graphs
    assert GRAPH_STAGE_RELEASE_BACKLOG in (cache.last_failure((9, 1)) or "")
    assert (9, 1) in cache.failed_keys()

    # Draining the backlog is the command that clears the refusal.
    failing["on"] = False
    assert cache.retry_closes() == ()
    assert cache.pending_close_count == 0

    stored, reason = cache.capture((9, 1), _binding(slot=9), lambda: _Graph("g9"), close)

    assert reason is None
    assert stored is not None
    assert cache.failed_keys() == ()


def test_spec_graph_stage_cache_pending_errors_expose_the_failure() -> None:
    cache = SpecGraphStageCache(max_entries=4)
    graph = _Graph("g0")

    def close(value: object) -> None:
        raise RuntimeError("hipStreamDestroy failed")

    cache.capture((0, 1), _binding(), lambda: graph, close)
    cache.close()

    errors = cache.pending_errors()

    assert len(errors) == 1
    key, error = errors[0]
    assert key == (0, 1)
    assert isinstance(error, RuntimeError)
    assert "hipStreamDestroy failed" in str(error)
    # The handle is still reachable, not dropped.
    assert cache.pending_closes() == (((0, 1), graph),)


def test_describe_failure_includes_attached_notes() -> None:
    error = RuntimeError("hipGraphExecDestroy failed")
    error.add_note("GGUF NextN proposal graph release: graph_exec_destroy(0xf000)")

    text = describe_failure(error)

    assert "hipGraphExecDestroy failed" in text
    assert "graph_exec_destroy(0xf000)" in text


def test_describe_failure_includes_group_member_notes() -> None:
    first = RuntimeError("hipGraphDestroy failed")
    first.add_note("release: graph_destroy")
    second = MemoryError("hipFree failed")
    group = BaseExceptionGroup("cleanup failed", [first, second])

    text = describe_failure(group)

    assert "release: graph_destroy" in text
    assert "hipFree failed" in text
