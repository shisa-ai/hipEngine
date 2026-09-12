from __future__ import annotations

import inspect
import random

import pytest

from hipengine.generation.concurrency2 import (
    BlockingOutputCollector,
    ChildPhase,
    ChildRequest,
    EngineOutput,
    OutputKind,
    ParentRequest,
    StreamingOutputCollector,
)
from hipengine.generation.concurrency2_simulator import (
    FAKE_KV_BACKEND_KINDS,
    DeterministicEngineSimulator,
    SimulatedResourceCapacityError,
    SimulatedResourceLedger,
    create_fake_kv_backend,
    independent_token_ids,
)
from hipengine.generation.registry import GenerationOutput, GenerationStreamChunk
from hipengine.kvcache import (    ClaimConfidence,
    ClaimLifetime,
    KVBackendSpec,
    KVCacheBackend,
    KVPlaneView,
    KVPoolPlan,
    KVPoolSpec,
    KVStorageView,
    ResourceChange,
    ResourceClaim,
    ResourceClaimSet,
    ResourceDelta,
)


def _child(
    request_id: int,
    *,
    prompt_length: int,
    max_new_tokens: int,
    parent_id: int | None = None,
    choice_index: int = 0,
    streaming: bool = False,
) -> ChildRequest:
    return ChildRequest(
        request_id=request_id,
        prompt_tokens=tuple(100 + ((request_id + index) % 17) for index in range(prompt_length)),
        max_new_tokens=max_new_tokens,
        parent_id=parent_id,
        choice_index=choice_index,
        streaming=streaming,
    )


def test_backend_contract_value_objects_reject_ambiguous_or_unsafe_state() -> None:
    spec = KVBackendSpec(
        topology_key="paged_dense",
        hot_codec_key="bf16",
        tier_key="device_only",
        layout_fingerprint="layout:bf16:v1",
        artifact_fingerprint="artifact:none",
        prefix_mode="immutable_pages",
        transaction_mode="journal",
        kernel_bundle_key="fake_dense_bf16",
        physical_widths=(1, 2, 4, 8),
    )
    assert spec.compatibility_key[-1] == "fake_dense_bf16"

    claims = ResourceClaimSet(
        claim_id="admit:7",
        request_id=7,
        claims=(
            ResourceClaim("kv.k", 11, ClaimLifetime.LEASE, ClaimConfidence.EXACT),
            ResourceClaim("kv.v", 11, ClaimLifetime.LEASE, ClaimConfidence.EXACT),
        ),
    )
    assert claims.units_by_pool() == {"kv.k": 11, "kv.v": 11}

    plan = KVPoolPlan(
        backend_fingerprint=spec.fingerprint,
        generation=1,
        pools=(
            KVPoolSpec("kv.k", 128, unit="cells", plane_role="k_payload"),
            KVPoolSpec("kv.v", 128, unit="cells", plane_role="v_payload"),
        ),
    )
    assert plan.pool("kv.k").capacity == 128

    storage = KVStorageView(
        layout_key="paged_dense+bf16",
        generation=1,
        planes=(
            KVPlaneView("k_payload", "bf16", 0x1000, (128,), (1,)),
            KVPlaneView("v_payload", "bf16", 0x2000, (128,), (1,)),
        ),
        metadata_descriptor_ptr=0x3000,
        metadata_descriptor_bytes=64,
        artifact_fingerprint=spec.artifact_fingerprint,
    )
    assert storage.plane("v_payload").ptr == 0x2000

    delta = ResourceDelta(
        operation_id="grow:7:1",
        lease_id="lease:7",
        request_id=7,
        changes=(
            ResourceChange("kv.k", 1, ClaimLifetime.LEASE),
            ResourceChange("kv.v", 1, ClaimLifetime.LEASE),
        ),
    )
    assert delta.units_by_pool() == {"kv.k": 1, "kv.v": 1}

    with pytest.raises(ValueError, match="non-negative"):
        ResourceClaimSet.from_mapping("negative", {"kv.k": -1})
    with pytest.raises(ValueError, match="duplicate resource claim"):
        ResourceClaimSet(
            claim_id="bad",
            claims=(
                ResourceClaim("kv.k", 1),
                ResourceClaim("kv.k", 2),
            ),
        )
    with pytest.raises(ValueError, match="duplicate pool_id"):
        KVPoolPlan(
            backend_fingerprint=spec.fingerprint,
            generation=1,
            pools=(KVPoolSpec("kv.k", 1), KVPoolSpec("kv.k", 2)),
        )
    with pytest.raises(ValueError, match="duplicate plane role"):
        KVStorageView(
            layout_key="bad",
            generation=1,
            planes=(
                KVPlaneView("payload", "bf16", 1, (1,), (1,)),
                KVPlaneView("payload", "bf16", 2, (1,), (1,)),
            ),
            artifact_fingerprint="artifact:none",
        )


def test_output_contract_carries_real_stream_and_terminal_payloads() -> None:
    collector = StreamingOutputCollector(max_output_tokens=1, max_chunks=1)
    collector.bind(99)
    chunk = GenerationStreamChunk(text="hello", generated_token_ids=(123,))
    assert collector.publish(
        EngineOutput(
            kind=OutputKind.TOKEN,
            request_id=99,
            token_id=123,
            token_index=0,
            stream_chunk=chunk,
        )
    )
    final = GenerationOutput(text="hello", generated_token_ids=(123,))
    assert collector.publish(
        EngineOutput(
            kind=OutputKind.TERMINAL,
            request_id=99,
            generated_token_ids=(123,),
            finish_reason="length",
            generation_output=final,
        )
    )

    events = collector.drain()
    assert events[0].stream_chunk is chunk
    assert collector.result is not None
    assert collector.result.generation_output is final


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_fake_ledger_reservation_failure_is_atomic_at_every_pool_boundary(kind: str) -> None:
    plan = create_fake_kv_backend(kind, capacity_tokens=64).plan_pools(None)

    for rejected_pool in plan.pools:
        ledger = SimulatedResourceLedger(plan)
        claims = ResourceClaimSet(
            claim_id=f"inject:{rejected_pool.pool_id}",
            claims=tuple(
                ResourceClaim(
                    pool.pool_id,
                    pool.capacity + 1 if pool.pool_id == rejected_pool.pool_id else 1,
                    pool.lifetimes[0],
                )
                for pool in plan.pools
            ),
        )
        before = ledger.snapshot()

        with pytest.raises(SimulatedResourceCapacityError) as raised:
            ledger.reserve("injected-owner", claims)

        assert raised.value.pool_id == rejected_pool.pool_id
        assert ledger.snapshot() == before
        assert not ledger.has_owner("injected-owner")
        ledger.assert_conserved()


def test_parent_is_aggregation_only_and_requires_unique_child_choices() -> None:
    children = (
        _child(10, prompt_length=2, max_new_tokens=1, parent_id=5, choice_index=0),
        _child(11, prompt_length=2, max_new_tokens=1, parent_id=5, choice_index=1),
    )
    parent = ParentRequest.from_children(5, children)

    assert parent.child_request_ids == (10, 11)
    assert not hasattr(parent, "kv_lease")
    assert not hasattr(parent, "resident_slot")

    duplicate_choice = _child(
        12,
        prompt_length=2,
        max_new_tokens=1,
        parent_id=5,
        choice_index=0,
    )
    with pytest.raises(ValueError, match="choice_index"):
        ParentRequest.from_children(5, (children[0], duplicate_choice))


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_backend_swap_uses_one_engine_scheduler_and_frontend_contract(kind: str) -> None:
    backend = create_fake_kv_backend(kind, capacity_tokens=4096)
    engine = DeterministicEngineSimulator(backend, resident_capacity=4)
    blocking = BlockingOutputCollector(max_output_tokens=4)
    streaming = StreamingOutputCollector(max_output_tokens=4, max_chunks=4)

    assert isinstance(backend, KVCacheBackend)
    assert type(engine) is DeterministicEngineSimulator
    assert type(blocking) is BlockingOutputCollector
    assert type(streaming) is StreamingOutputCollector
    assert engine.command_queue_type is type(engine.command_queue)
    engine_source = inspect.getsource(DeterministicEngineSimulator)
    assert all(backend_kind not in engine_source for backend_kind in FAKE_KV_BACKEND_KINDS)
    assert engine.backend.plan_pools(None).backend_fingerprint == engine.backend.spec.fingerprint

    engine.submit(_child(1, prompt_length=3, max_new_tokens=2), blocking)
    engine.submit(_child(2, prompt_length=3, max_new_tokens=2, streaming=True), streaming)
    engine.run_until_idle()

    assert blocking.result is not None
    assert streaming.result is not None
    assert blocking.result.generated_token_ids == streaming.result.generated_token_ids
    streamed = streaming.drain()
    assert streamed[-1].kind is OutputKind.TERMINAL
    assert streaming.wait_for_event(timeout=0.0) is False
    engine.assert_invariants()


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_short_child_reclaims_and_refills_while_old_sibling_is_decoding(kind: str) -> None:
    backend = create_fake_kv_backend(kind, capacity_tokens=4096)
    engine = DeterministicEngineSimulator(backend, resident_capacity=2)
    short = _child(10, prompt_length=3, max_new_tokens=1, parent_id=1, choice_index=0)
    long = _child(11, prompt_length=3, max_new_tokens=5, parent_id=1, choice_index=1)
    refill = _child(12, prompt_length=2, max_new_tokens=1)
    short_collector = BlockingOutputCollector(max_output_tokens=1)
    long_collector = BlockingOutputCollector(max_output_tokens=5)
    refill_collector = BlockingOutputCollector(max_output_tokens=1)

    engine.submit(short, short_collector)
    engine.submit(long, long_collector)
    engine.submit(refill, refill_collector)
    engine.step()

    assert short_collector.result is not None
    assert short_collector.result.finish_reason == "length"
    assert engine.phase(10) is ChildPhase.TERMINAL
    assert engine.phase(11) is ChildPhase.DECODE
    assert engine.phase(12) is ChildPhase.DECODE
    assert set(engine.resident_request_ids) == {11, 12}
    assert not engine.ledger.has_owner("lease:10")
    assert long_collector.result is None
    assert engine.snapshot()["counters"]["queued"] == 0

    engine.step()
    assert refill_collector.result is not None
    assert long_collector.result is None
    assert engine.phase(11) is ChildPhase.DECODE
    engine.run_until_idle()

    assert long_collector.result is not None
    assert short_collector.result.generated_token_ids == independent_token_ids(short)
    assert long_collector.result.generated_token_ids == independent_token_ids(long)
    assert refill_collector.result.generated_token_ids == independent_token_ids(refill)
    engine.assert_invariants()


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_blocking_and_streaming_children_share_order_cancel_and_reclaim(kind: str) -> None:
    backend = create_fake_kv_backend(kind, capacity_tokens=4096)
    engine = DeterministicEngineSimulator(backend, resident_capacity=2)
    blocking_request = _child(20, prompt_length=5, max_new_tokens=5)
    streaming_request = _child(21, prompt_length=5, max_new_tokens=5, streaming=True)
    blocking = BlockingOutputCollector(max_output_tokens=5)
    streaming = StreamingOutputCollector(max_output_tokens=5, max_chunks=2)

    engine.submit(blocking_request, blocking)
    engine.submit(streaming_request, streaming)
    engine.step()
    stream_events = streaming.drain()
    assert tuple(event.kind for event in stream_events) == (OutputKind.TOKEN,)
    assert tuple(event.token_id for event in stream_events) == blocking.generated_token_ids

    assert engine.cancel(20, reason="client_cancel") is True
    assert engine.cancel(21, reason="client_cancel") is True
    assert blocking.result is not None and streaming.result is not None
    assert blocking.result.finish_reason == streaming.result.finish_reason == "client_cancel"
    assert blocking.result.generated_token_ids == streaming.result.generated_token_ids
    assert engine.terminal_order == (20, 21)
    assert engine.resident_count == 0
    assert all(pool["used"] == 0 for pool in engine.snapshot()["pools"].values())
    engine.assert_invariants()


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_slow_stream_consumer_cancels_only_itself(kind: str) -> None:
    backend = create_fake_kv_backend(kind, capacity_tokens=4096)
    engine = DeterministicEngineSimulator(backend, resident_capacity=2)
    slow_request = _child(30, prompt_length=2, max_new_tokens=5, streaming=True)
    neighbor_request = _child(31, prompt_length=2, max_new_tokens=5)
    slow = StreamingOutputCollector(max_output_tokens=5, max_chunks=1)
    neighbor = BlockingOutputCollector(max_output_tokens=5)

    engine.submit(slow_request, slow)
    engine.submit(neighbor_request, neighbor)
    engine.run_until_idle()

    assert slow.result is not None
    assert slow.result.finish_reason == "client_backpressure"
    assert len(slow.result.generated_token_ids) == 1
    assert neighbor.result is not None
    assert neighbor.result.finish_reason == "length"
    assert neighbor.result.generated_token_ids == independent_token_ids(neighbor_request)
    engine.assert_invariants()


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_random_c1_c32_lifecycle_conserves_every_fake_backend_pool(kind: str) -> None:
    rng = random.Random(20260816)
    aggregate_operations: dict[str, int] = {}

    for logical_c in range(1, 33):
        backend = create_fake_kv_backend(kind, capacity_tokens=200_000)
        engine = DeterministicEngineSimulator(backend, resident_capacity=logical_c)
        requests: dict[int, ChildRequest] = {}
        collectors: dict[int, BlockingOutputCollector | StreamingOutputCollector] = {}
        total = logical_c + 3
        request_base = logical_c * 1000

        for index in range(total):
            request_id = request_base + index
            request = _child(
                request_id,
                prompt_length=rng.randint(1, 9),
                max_new_tokens=rng.randint(1, 9),
                streaming=index % 3 == 0,
            )
            collector = (
                StreamingOutputCollector(max_output_tokens=9, max_chunks=3)
                if request.streaming
                else BlockingOutputCollector(max_output_tokens=9)
            )
            requests[request_id] = request
            collectors[request_id] = collector
            if index < logical_c + 1:
                engine.submit(request, collector)

        queued_cancel_id = request_base + logical_c
        assert engine.cancel(queued_cancel_id, reason="cancel_before_admission") is True

        step_index = 0
        active_cancel_id: int | None = None
        while not engine.idle:
            engine.step()
            step_index += 1
            if step_index == 1:
                first_groups = engine.snapshot()["counters"]["last_physical_groups"]
                assert sum(first_groups) == logical_c
                assert all(group in (1, 2, 4, 8) for group in first_groups)
                for delayed_index in range(logical_c + 1, total):
                    delayed_id = request_base + delayed_index
                    engine.submit(requests[delayed_id], collectors[delayed_id])
            engine.assert_invariants()
            slots = tuple(engine.resident_slots)
            assert len(slots) == len(set(slots))
            assert len(engine.resident_request_ids) == len(set(engine.resident_request_ids))
            for collector in collectors.values():
                if isinstance(collector, StreamingOutputCollector):
                    collector.drain()
            if step_index == 1:
                active_cancel_id = next(
                    (
                        request_id
                        for request_id in engine.resident_request_ids
                        if requests[request_id].max_new_tokens > 1
                    ),
                    None,
                )
            elif step_index == 2 and active_cancel_id is not None:
                engine.cancel(active_cancel_id, reason="cancel_during_decode")
            if step_index % 2 == 0:
                engine.compact()
            assert step_index < 64

        for operation, count in engine.operation_counts.items():
            aggregate_operations[operation] = aggregate_operations.get(operation, 0) + count

        for request_id, request in requests.items():
            result = collectors[request_id].result
            assert result is not None
            oracle = independent_token_ids(request)
            if result.finish_reason == "length":
                assert result.generated_token_ids == oracle
            else:
                assert result.generated_token_ids == oracle[: len(result.generated_token_ids)]
            engine.take_result(request_id)

        snapshot = engine.snapshot()
        assert snapshot["counters"]["queued"] == 0
        assert snapshot["counters"]["resident"] == 0
        assert snapshot["counters"]["completion_records"] == 0
        assert all(pool["used"] == 0 for pool in snapshot["pools"].values())
        assert sum(snapshot["counters"]["physical_widths"].values()) > 0
        engine.assert_invariants()

    if kind == "mixed_bf16_packed":
        assert aggregate_operations.get("demote", 0) > 0
    if kind == "dms_variable":
        assert aggregate_operations.get("compact", 0) > 0
