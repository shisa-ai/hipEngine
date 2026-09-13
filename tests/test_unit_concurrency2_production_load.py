from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine.generation import (
    EngineLoopConfig,
    GeneratedToken,
    LoadArrival,
    ResidentEngineLoop,
    poisson_arrivals,
    run_load_scenario,
)
from hipengine.kvcache import (
    BackendRadixCache,
    DenseKVAdmissionManager,
    DenseKVResidentRunnerAdapter,
    GraphReplayBindingRegistry,
    KVPageState,
    PrefixCompatibilityKey,
    create_dense_bf16_backend,
)


def _request(request_id: int, tokens: tuple[int, ...], *, max_new_tokens: int = 1):
    return SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tokens,
        max_new_tokens=max_new_tokens,
    )


def _reserve(backend, request_id: int, tokens: tuple[int, ...]):
    request = _request(request_id, tokens)
    return backend.reserve(backend.estimate(request, None, {"kind": "admission"}))


def _scope(backend) -> PrefixCompatibilityKey:
    return PrefixCompatibilityKey.for_backend(
        backend.spec,
        model_artifact_fingerprint="model:graph-load",
        model_revision="revision:test",
        hardware_backend="hip_gfx1100",
        model_key="qwen-test",
        weight_quant="q4_k_m",
        rope_fingerprint="rope:default",
    )


def _cache(backend) -> BackendRadixCache:
    return BackendRadixCache(
        spec=backend.spec,
        generation=backend.generation,
        block_size=backend.block_size,
        pool=backend.pool,
        ledger=backend.ledger,
        page_pool_ids=backend.page_pool_ids,
        max_cached_pages=backend.page_capacity,
    )


def test_graph_replay_accepts_changing_page_ids_and_slot_reuse_with_stable_planes() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=8,
        block_size=2,
        backend_fingerprint="dense-bf16:graph-replay",
    )
    first = _reserve(backend, 1, (1, 2))
    first_view = backend.prepare(SimpleNamespace(request_ids=(1,), context_lengths=(2,)))
    graphs = GraphReplayBindingRegistry(backend.pool)
    signature = graphs.capture("decode:c1", first_view)
    first_binding = graphs.bind_replay(
        "decode:c1",
        first_view,
        request_ids=(1,),
        lease_ids=(first.lease_id,),
        slot_ids=(0,),
    )
    first_pages = first_binding.page_ids
    backend.reclaim(first)
    assert all(
        backend.pool.page(page_id).state is KVPageState.IN_FLIGHT
        for page_id in first_pages
    )

    second = _reserve(backend, 2, (3, 4))
    second_view = backend.prepare(SimpleNamespace(request_ids=(2,), context_lengths=(2,)))
    graphs.retire(first_binding)
    second_binding = graphs.bind_replay(
        "decode:c1",
        second_view,
        request_ids=(2,),
        lease_ids=(second.lease_id,),
        slot_ids=(0,),
    )
    assert second_binding.page_ids != first_pages
    assert second_view.storage_view == first_view.storage_view
    assert signature.planes == GraphReplayBindingRegistry(backend.pool).capture(
        "control",
        second_view,
    ).planes
    assert graphs.snapshot()["slot_reuse_count"] == 1

    backend.reclaim(second)
    graphs.retire(second_binding)
    assert backend.pool.free_pages == backend.pool.page_capacity
    graphs.assert_conserved()
    backend.ledger.assert_conserved()


def test_graph_replay_survives_prefix_eviction_while_shared_pages_are_in_flight() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=8,
        block_size=2,
        backend_fingerprint="dense-bf16:graph-prefix",
    )
    cache = _cache(backend)
    scope = _scope(backend)
    source = _reserve(backend, 1, (1, 2, 3, 4))
    snapshot = cache.publish(scope, source, (1, 2, 3, 4))
    assert snapshot is not None
    backend.reclaim(source)

    hit = cache.lookup(scope, (1, 2, 3, 4, 5))
    request = _request(2, (1, 2, 3, 4, 5))
    child = backend.reserve(
        backend.estimate(request, hit.snapshot, {"kind": "admission"})
    )
    view = backend.prepare(SimpleNamespace(request_ids=(2,), context_lengths=(5,)))
    graphs = GraphReplayBindingRegistry(backend.pool)
    graphs.capture("decode:prefix", view)
    binding = graphs.bind_replay(
        "decode:prefix",
        view,
        request_ids=(2,),
        lease_ids=(child.lease_id,),
        slot_ids=(0,),
    )

    cache.evict(snapshot, reason="graph-pressure")
    backend.reclaim(child)
    assert all(
        backend.pool.page(page_id).state is KVPageState.IN_FLIGHT
        for page_id in binding.page_ids
    )
    graphs.retire(binding)
    assert backend.pool.free_pages == backend.pool.page_capacity
    cache.assert_conserved()
    graphs.assert_conserved()


def test_graph_registry_rejects_stale_storage_and_inflight_invalidation() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=4,
        block_size=2,
        backend_fingerprint="dense-bf16:graph-stale",
    )
    lease = _reserve(backend, 1, (1, 2))
    view = backend.prepare(SimpleNamespace(request_ids=(1,), context_lengths=(2,)))
    graphs = GraphReplayBindingRegistry(backend.pool)
    graphs.capture("decode:c1", view)
    binding = graphs.bind_replay(
        "decode:c1",
        view,
        request_ids=(1,),
        lease_ids=(lease.lease_id,),
        slot_ids=(0,),
    )
    with pytest.raises(RuntimeError, match="in flight"):
        graphs.invalidate_generation(backend.generation)

    stale_view = SimpleNamespace(
        storage_view=SimpleNamespace(
            layout_key=view.storage_view.layout_key,
            generation=view.storage_view.generation + 1,
            artifact_fingerprint=view.storage_view.artifact_fingerprint,
            planes=view.storage_view.planes,
            metadata_descriptor_ptr=view.storage_view.metadata_descriptor_ptr,
            metadata_descriptor_bytes=view.storage_view.metadata_descriptor_bytes,
        )
    )
    with pytest.raises(ValueError, match="signature changed"):
        graphs.bind_replay(
            "decode:c1",
            stale_view,
            request_ids=(1,),
            lease_ids=(lease.lease_id,),
            slot_ids=(0,),
        )
    backend.reclaim(lease)
    graphs.retire(binding)
    assert graphs.invalidate_generation(backend.generation) == 1


@dataclass
class _LoadRunner:
    capacity: int
    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = True

    def prefill_batch(self, work, *, commit):
        assert commit is True

    def decode_batch(self, work, *, commit):
        assert commit is True
        return tuple(
            GeneratedToken(request_id, 500 + request_id)
            for request_id in work.request_ids
        )

    def compact_batch(self, moves):
        del moves

    def reclaim(self, completed):
        del completed


def _load_loop(*, capacity: int, max_pending_requests: int | None = None) -> ResidentEngineLoop:
    return ResidentEngineLoop(
        _LoadRunner(capacity),
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=capacity,
            max_prefill_chunk_tokens=4,
            round_prefill_token_budget=max(4, capacity * 4),
            round_decode_row_budget=capacity,
            max_pending_requests=max_pending_requests,
        ),
    )


@pytest.mark.parametrize(
    "arrivals",
    (
        tuple(
            LoadArrival(f"fixed:{index}", index, (1, 2, 3), 2)
            for index in range(12)
        ),
        tuple(
            LoadArrival(f"ragged:{index}", index % 3, tuple(range(1, 2 + index % 7)), 1 + index % 3)
            for index in range(16)
        ),
        tuple(
            LoadArrival(f"burst:{index}", 0, (index + 1,), 2)
            for index in range(32)
        ),
        poisson_arrivals(
            count=24,
            rate_per_tick=2.5,
            seed=1234,
            prompt_tokens=(1, 2, 3, 4),
            max_new_tokens=2,
        ),
    ),
)
def test_fixed_ragged_burst_and_poisson_loads_drain_through_c32(arrivals) -> None:
    loop = _load_loop(capacity=32)
    result = run_load_scenario(loop, arrivals, max_ticks=256)
    assert result.drained is True
    assert result.submitted == result.offered
    assert result.completed == result.offered
    assert result.max_active <= 32
    assert result.occupancy_history[-1] == 0
    assert all(reason in {"length", "stop"} for _arrival, reason in result.finish_reasons)


def test_overload_retries_then_recovers_and_disconnects_reclaim_exactly() -> None:
    loop = _load_loop(capacity=4, max_pending_requests=2)
    arrivals = tuple(
        LoadArrival(
            f"overload:{index}",
            0,
            (index + 1, index + 2, index + 3),
            3,
            disconnect_after_ticks=(2 if index in {1, 7} else None),
        )
        for index in range(16)
    )
    result = run_load_scenario(loop, arrivals, max_ticks=256, retry_rejected=True)
    assert result.drained is True
    assert result.submitted == 16
    assert result.completed == 16
    assert result.disconnected == 2
    assert result.retryable_rejections > 0
    reasons = dict(result.finish_reasons)
    assert reasons["overload:1"] == "disconnect"
    assert reasons["overload:7"] == "disconnect"
    assert loop.active_count == 0
    assert loop.pending_count == 0


def test_long_context_mixed_membership_4k_16k_32k_drains_global_pool() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=220,
        block_size=256,
        backend_fingerprint="dense-bf16:long-context-mixed",
    )
    admission = DenseKVAdmissionManager(backend)

    class LongRunner:
        capacity = 3
        kv_supports_masked_rows = True

        def __init__(self) -> None:
            self.kv_kernel_bundle_key = backend.spec.kernel_bundle_key
            self.kv_storage_layout_keys = (backend.storage_view().layout_key,)

        def prefill_batch_with_kv(self, work, *, kv_batch_view, commit):
            raise AssertionError("physical prefill path required")

        def decode_batch_with_kv(self, work, *, kv_batch_view, commit):
            raise AssertionError("physical decode path required")

        def prefill_physical_group_with_kv(
            self,
            work,
            *,
            physical_group,
            kv_batch_view,
            commit,
        ):
            assert commit is True
            assert kv_batch_view.storage_view is backend.storage_view()

        def decode_physical_group_with_kv(
            self,
            work,
            *,
            physical_group,
            kv_batch_view,
            commit,
        ):
            assert commit is True
            return tuple(
                GeneratedToken(request_id, 900 + request_id)
                for request_id in physical_group.request_ids
            )

        def compact_batch(self, moves):
            del moves

        def reclaim(self, completed):
            del completed

    graphs = GraphReplayBindingRegistry(backend.pool)
    loop = ResidentEngineLoop(
        DenseKVResidentRunnerAdapter(
            LongRunner(),
            admission,
            graph_registry=graphs,
        ),
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=3,
            max_prefill_chunk_tokens=1024,
            round_prefill_token_budget=4096,
            round_decode_row_budget=3,
        ),
    )
    arrivals = (
        LoadArrival("4k", 0, tuple(range(4096)), 2),
        LoadArrival("16k", 0, tuple(range(16_384)), 2),
        LoadArrival("32k", 0, tuple(range(32_768)), 2),
    )
    result = run_load_scenario(loop, arrivals, max_ticks=128)
    assert result.drained is True
    assert result.completed == 3
    assert result.max_active == 3
    assert backend.pool.free_pages == backend.pool.page_capacity
    assert all(
        pool["used"] == 0
        for pool in backend.ledger.snapshot()["pools"].values()
    )
    backend.pool.assert_conserved()
    backend.ledger.assert_conserved()
    assert graphs.snapshot()["replay_count"] > 0
    assert graphs.snapshot()["in_flight"] == 0
    graphs.assert_conserved()


def test_c1_and_c32_have_the_same_decode_round_count_without_response_cliff() -> None:
    single = run_load_scenario(
        _load_loop(capacity=1),
        (LoadArrival("single", 0, (1,), 4),),
        max_ticks=16,
    )
    wide = run_load_scenario(
        _load_loop(capacity=32),
        tuple(LoadArrival(f"wide:{index}", 0, (index + 1,), 4) for index in range(32)),
        max_ticks=16,
    )
    assert single.drained is True and wide.drained is True
    assert single.ticks == wide.ticks == 4
    assert single.completed == 1
    assert wide.completed == 32
