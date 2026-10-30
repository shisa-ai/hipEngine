"""Focused ownership tests for the Qwen GGUF graph/KV-pin lifetime boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as runner
from hipengine.runtime.qwen35_gguf_graph_lifetime import (
    Qwen35GGUFGraphLifetime,
    close_graph_handles,
)


class _Pool:
    def __init__(self) -> None:
        self.pins: list[tuple[int, ...]] = []
        self.unpins: list[tuple[int, ...]] = []

    def pin(self, block_ids) -> None:
        self.pins.append(tuple(block_ids))

    def unpin(self, block_ids) -> None:
        self.unpins.append(tuple(block_ids))


@dataclass(frozen=True)
class _Scratch:
    block_table_tensor: object
    block_table: object
    full_key_caches: tuple
    full_value_caches: tuple
    full_bf16_mirror_key_caches: tuple
    full_bf16_mirror_value_caches: tuple
    full_k_scale_caches: tuple
    full_v_scale_caches: tuple
    full_kv_scale_metadata: tuple


class _SessionGraph:
    """A decode-graph stand-in with the real close -> unpin ordering."""

    def __init__(self, name: str, session, *, trace: list[str] | None = None) -> None:
        self.name = name
        self.session = session
        self.trace = trace
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.trace is not None:
            self.trace.append(self.name)
        self.session._unpin_device_kv_graph(self)


class _Context:
    def __init__(self) -> None:
        self.closed = False

    def provenance(self) -> dict[str, object]:
        return {"closed": self.closed}

    def close(self) -> None:
        self.closed = True


def _make_session(pool: _Pool, *, block_ids=(1, 2)):
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.runtime = object()
    session._device_kv_allocation = SimpleNamespace(block_ids=tuple(block_ids))
    session._device_kv_pool = pool
    session.scratch = _Scratch(
        block_table_tensor=SimpleNamespace(shape=(4,)),
        block_table=SimpleNamespace(nbytes=16),
        full_key_caches=(None,),
        full_value_caches=(None,),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
        full_kv_scale_metadata=(None,),
    )
    return session


def test_graph_lifetime_pins_each_graph_identity_once() -> None:
    owner = Qwen35GGUFGraphLifetime()
    pool = _Pool()
    allocation = SimpleNamespace(block_ids=(4, 5))
    graph = SimpleNamespace(closed=False)

    assert owner.pin(graph, allocation=allocation, pool=pool) is True
    assert owner.pin(graph, allocation=allocation, pool=pool) is False
    assert pool.pins == [(4, 5)]
    assert owner.pinned_count == 1

    assert owner.unpin(graph, allocation=allocation, pool=pool) is True
    assert owner.unpin(graph, allocation=allocation, pool=pool) is False
    assert pool.unpins == [(4, 5)]
    assert owner.pinned_count == 0


def test_graph_lifetime_pin_without_a_binding_is_a_noop() -> None:
    owner = Qwen35GGUFGraphLifetime()
    pool = _Pool()
    allocation = SimpleNamespace(block_ids=(4, 5))
    graph = SimpleNamespace(closed=False)

    assert owner.pin(graph, allocation=None, pool=pool) is False
    assert owner.pin(graph, allocation=allocation, pool=None) is False
    assert owner.pinned_count == 0
    assert pool.pins == []

    # Unpinning an unregistered graph never consults the missing binding.
    assert owner.unpin(graph, allocation=None, pool=None) is False
    assert pool.unpins == []


def test_graph_lifetime_pin_failure_keeps_the_registry_unchanged() -> None:
    class _FailingPool(_Pool):
        def pin(self, block_ids) -> None:
            raise RuntimeError("pool pin failed")

    owner = Qwen35GGUFGraphLifetime()
    graph = SimpleNamespace(closed=False)

    with pytest.raises(RuntimeError, match="pool pin failed"):
        owner.pin(
            graph,
            allocation=SimpleNamespace(block_ids=(4,)),
            pool=_FailingPool(),
        )

    assert owner.pinned_count == 0


def test_graph_lifetime_unpin_after_detach_raises_and_keeps_the_pin() -> None:
    owner = Qwen35GGUFGraphLifetime()
    pool = _Pool()
    allocation = SimpleNamespace(block_ids=(4, 5))
    graph = SimpleNamespace(closed=False)
    owner.pin(graph, allocation=allocation, pool=pool)

    with pytest.raises(RuntimeError, match="outlived its allocation"):
        owner.unpin(graph, allocation=None, pool=pool)

    assert owner.pinned_count == 1
    assert pool.unpins == []


def test_graph_lifetime_closes_pinned_handles_first_and_each_handle_once() -> None:
    owner = Qwen35GGUFGraphLifetime()
    pool = _Pool()
    allocation = SimpleNamespace(block_ids=(4,))
    events: list[str] = []
    shared = SimpleNamespace(closed=False)
    handle_only = SimpleNamespace(closed=False)
    already_closed = SimpleNamespace(closed=True, close=lambda: events.append("closed"))
    no_close = object()

    def close_shared() -> None:
        shared.closed = True
        events.append("shared")

    def close_handle() -> None:
        handle_only.closed = True
        events.append("handle")

    shared.close = close_shared
    handle_only.close = close_handle

    owner.pin(shared, allocation=allocation, pool=pool)
    owner.pin(handle_only, allocation=allocation, pool=pool)
    owner.decode_graphs.extend((shared, already_closed, no_close))

    assert owner.close_live_graphs() == 2
    assert events == ["shared", "handle"]
    # A handle with no close() cannot be closed by this policy; it stays visible
    # so packed-workspace teardown can still refuse to free its buffers.
    assert owner.live_graphs() == (no_close,)


def test_close_graph_handles_strict_policy_names_the_unclosable_graph() -> None:
    assert close_graph_handles([object()]) == 0

    with pytest.raises(RuntimeError, match="cannot resize the packed verify workspace"):
        close_graph_handles(
            [object()],
            require_close=True,
            error="cannot resize the packed verify workspace: no close()",
        )


def test_graph_lifetime_close_decode_graphs_keeps_the_registry_for_retry() -> None:
    owner = Qwen35GGUFGraphLifetime()
    trace: list[str] = []
    first = _SessionGraph("first", SimpleNamespace(_unpin_device_kv_graph=lambda _g: None))
    second = _SessionGraph("second", SimpleNamespace(_unpin_device_kv_graph=lambda _g: None))
    first.trace = trace
    second.trace = trace
    owner.decode_graphs.extend((first, second))

    def fail() -> None:
        raise RuntimeError("close failed")

    second.close = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="close failed"):
        owner.close_decode_graphs()

    assert trace == ["first"]
    assert owner.decode_graphs == [first, second]

    second.close = lambda: trace.append("second")  # type: ignore[method-assign]
    assert owner.close_decode_graphs() == 2
    assert trace == ["first", "second"]
    assert owner.decode_graphs == []


def test_graph_lifetime_retains_one_submission_context_per_transport() -> None:
    owner = Qwen35GGUFGraphLifetime()
    created: list[object] = []

    def factory() -> object:
        created.append(object())
        return created[-1]

    first = owner.submission_context("pm4", factory)
    second = owner.submission_context("pm4", factory)

    assert first is second
    assert len(created) == 1
    assert owner.submission_context("hipgraph", lambda: None) is None
    assert set(owner.submission_contexts) == {"pm4"}


def test_graph_lifetime_submission_context_teardown_returns_before_after_proof() -> None:
    owner = Qwen35GGUFGraphLifetime()
    context = _Context()
    owner.submission_contexts["pm4"] = context

    proofs = owner.close_submission_contexts()

    assert proofs == {
        "pm4": {"before": {"closed": False}, "after": {"closed": True}}
    }
    assert owner.submission_contexts == {}

    owner.submission_contexts["bad"] = object()
    with pytest.raises(RuntimeError, match="invalid lifecycle ABI"):
        owner.close_submission_contexts()


def test_resident_session_graph_aliases_share_one_lifetime_owner() -> None:
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    lifetime = session._graph_lifetime()

    assert session._decode_graphs is lifetime.decode_graphs
    assert session._device_kv_graph_handles is lifetime.kv_pins
    assert session._decode_graph_submission_contexts is lifetime.submission_contexts

    graph = SimpleNamespace(closed=False)
    session._decode_graphs = [graph]
    session._device_kv_graph_handles = {id(graph): graph}
    session._decode_graph_submission_contexts = {"pm4": graph}

    assert lifetime.decode_graphs == [graph]
    assert lifetime.kv_pins == {id(graph): graph}
    assert lifetime.submission_contexts == {"pm4": graph}


def test_two_sessions_keep_independent_graph_lifetimes() -> None:
    first = object.__new__(runner.Qwen35GGUFResidentSession)
    second = object.__new__(runner.Qwen35GGUFResidentSession)
    first._decode_graphs.append(SimpleNamespace(closed=False))

    assert first._graph_lifetime() is not second._graph_lifetime()
    assert len(first._decode_graphs) == 1
    assert second._decode_graphs == []


def test_two_sessions_hold_independent_pins_over_one_pool() -> None:
    pool = _Pool()
    first = _make_session(pool, block_ids=(1, 2))
    second = _make_session(pool, block_ids=(3, 4))
    first_graph = _SessionGraph("first", first)
    second_graph = _SessionGraph("second", second)
    first._decode_graphs.append(first_graph)
    second._decode_graphs.append(second_graph)

    first._pin_device_kv_graph(first_graph)
    second._pin_device_kv_graph(second_graph)
    assert pool.pins == [(1, 2), (3, 4)]

    assert first.invalidate_device_kv_graphs() == 1
    assert pool.unpins == [(1, 2)]
    assert first._device_kv_graph_handles == {}
    assert second._graph_lifetime().pinned_count == 1

    assert second.invalidate_device_kv_graphs() == 1
    assert pool.unpins == [(1, 2), (3, 4)]
    assert second._device_kv_graph_handles == {}


def test_session_invalidation_returns_each_pool_pin_once() -> None:
    pool = _Pool()
    session = _make_session(pool)
    trace: list[str] = []
    graph = _SessionGraph("graph", session, trace=trace)
    session._decode_graphs.append(graph)

    session._pin_device_kv_graph(graph)
    session._pin_device_kv_graph(graph)
    assert pool.pins == [(1, 2)]
    assert session._graph_lifetime().pinned_count == 1

    assert session.invalidate_device_kv_graphs() == 1
    assert trace == ["graph"]
    assert pool.unpins == [(1, 2)]
    assert session._device_kv_graph_handles == {}
    assert session.invalidate_device_kv_graphs() == 0


def test_unbind_waits_for_graph_pins_and_succeeds_after_invalidation(monkeypatch) -> None:
    monkeypatch.setattr(runner, "copy_host_array_to_device", lambda *args, **kwargs: None)
    pool = _Pool()
    session = _make_session(pool)
    graph = _SessionGraph("graph", session)
    session._decode_graphs.append(graph)
    session._pin_device_kv_graph(graph)

    with pytest.raises(RuntimeError, match="graphs remain pinned"):
        session.unbind_device_kv_allocation()

    assert session.invalidate_device_kv_graphs() == 1
    allocation = session.unbind_device_kv_allocation()

    assert tuple(allocation.block_ids) == (1, 2)
    assert pool.unpins == [(1, 2)]
    assert session._device_kv_allocation is None
    assert session._device_kv_pool is None


def test_packed_invalidation_closes_decode_graphs_before_pins() -> None:
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    trace: list[str] = []
    decode_graph = SimpleNamespace(
        closed=False, close=lambda: trace.append("decode")
    )
    pinned_graph = SimpleNamespace(
        closed=False, close=lambda: trace.append("pin")
    )
    session._decode_graphs = [decode_graph]
    session._device_kv_graph_handles = {id(pinned_graph): pinned_graph}

    assert session._invalidate_live_packed_decode_graphs() == 2
    assert trace == ["decode", "pin"]


def test_session_closes_retained_submission_contexts_through_the_owner() -> None:
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    context = _Context()
    session._decode_graph_submission_contexts["pm4"] = context

    assert session.close_decode_graph_submission_contexts() == {
        "pm4": {"before": {"closed": False}, "after": {"closed": True}}
    }
    assert session._decode_graph_submission_contexts == {}
