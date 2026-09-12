from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.dispatch import RequestState
from hipengine.generation import GeneratedToken, ResidentBatchScheduler, ResidentEngineLoop
from hipengine.kvcache import (
    BackendRadixCache,
    ClaimLifetime,
    DenseKVAdmissionManager,
    DenseKVResidentRunnerAdapter,
    KVPageState,
    PrefixCompatibilityKey,
    RadixCache,
    create_dense_bf16_backend,
)


def _backend(*, pages: int = 12, block_size: int = 2, suffix: str = "prefix"):
    return create_dense_bf16_backend(
        page_capacity=pages,
        block_size=block_size,
        backend_fingerprint=f"dense-bf16:{suffix}",
    )


def _scope(backend, **overrides) -> PrefixCompatibilityKey:
    values = {
        "model_artifact_fingerprint": "model:abc123",
        "model_revision": "revision:main",
        "adapter_identity": "none",
        "hardware_backend": "hip_gfx1100",
        "model_key": "qwen-test",
        "weight_quant": "q4_k_m",
        "backend_fingerprint": backend.spec.fingerprint,
        "backend_artifact_fingerprint": backend.spec.artifact_fingerprint,
        "rope_fingerprint": "rope:default",
        "multimodal_input_hash": "none",
    }
    values.update(overrides)
    return PrefixCompatibilityKey(**values)


def _cache(
    backend,
    *,
    max_cached_pages: int = 8,
    ttl_seconds: float | None = None,
    tenant_page_quotas=None,
    clock=None,
) -> BackendRadixCache:
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return BackendRadixCache(
        spec=backend.spec,
        generation=backend.generation,
        block_size=backend.block_size,
        pool=backend.pool,
        ledger=backend.ledger,
        page_pool_ids=backend.page_pool_ids,
        max_cached_pages=max_cached_pages,
        ttl_seconds=ttl_seconds,
        tenant_page_quotas=tenant_page_quotas,
        **kwargs,
    )


def _reserve(backend, request_id: int, tokens: tuple[int, ...], *, prefix=None):
    request = SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tokens,
        max_new_tokens=1,
    )
    claims = backend.estimate(request, prefix, {"kind": "admission"})
    return request, claims, backend.reserve(claims)


def _publish_and_reclaim(cache, scope, backend, request_id: int, tokens: tuple[int, ...], *, tenant="default"):
    request, _claims, lease = _reserve(backend, request_id, tokens)
    snapshot = cache.publish(scope, lease, tokens, tenant_id=tenant)
    backend.reclaim(lease)
    return request, snapshot


def test_backend_radix_snapshot_retains_complete_pages_and_full_compatibility_key() -> None:
    backend = _backend()
    scope = _scope(backend)
    cache = _cache(backend)
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (10, 11, 12, 13, 14),
    )
    assert snapshot is not None
    assert snapshot.matched_tokens == (10, 11, 12, 13)
    assert len(snapshot.page_ids) == 2
    assert all(
        backend.pool.page(page_id).state is KVPageState.CACHED_EVICTABLE
        for page_id in snapshot.page_ids
    )
    ledger = backend.ledger.snapshot()
    assert ledger["pools"]["kv.k_payload.pages"]["used_by_lifetime"] == {
        ClaimLifetime.CACHE.value: 2
    }

    hit = cache.lookup(scope, (10, 11, 12, 13, 99))
    assert hit.hit is True
    assert hit.snapshot is not None
    assert hit.snapshot.snapshot_id == snapshot.snapshot_id
    assert hit.remaining_tokens == (99,)

    incompatible = _scope(backend, model_revision="revision:other")
    miss = cache.lookup(incompatible, (10, 11, 12, 13, 99))
    assert miss.hit is False
    assert miss.miss_reason == "scope_miss"
    cache.assert_conserved()


def test_snapshot_hit_shares_full_pages_and_eviction_transfers_live_ownership() -> None:
    backend = _backend(pages=8)
    scope = _scope(backend)
    cache = _cache(backend)
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
    )
    assert snapshot is not None
    hit = cache.lookup(scope, (1, 2, 3, 4, 5))
    request, claims, lease = _reserve(
        backend,
        2,
        (1, 2, 3, 4, 5),
        prefix=hit.snapshot,
    )
    assert request.request_id == 2
    assert claims.metadata_dict()["private_pages"] == 1
    page_lease = backend.page_lease(lease)
    assert page_lease.shared_page_ids == snapshot.page_ids
    assert len(page_lease.private_page_ids) == 1
    assert all(
        backend.pool.page(page_id).state is KVPageState.ACTIVE_SHARED
        for page_id in snapshot.page_ids
    )

    assert cache.evict(snapshot, reason="test") is True
    assert cache.snapshot()["cached_pages"] == 0
    assert not backend.ledger.has_owner(f"prefix-cache:{backend.spec.fingerprint}")
    owner_units = backend.ledger.owner_claims(lease.lease_id).units_by_pool()
    assert owner_units["kv.k_payload.pages"] == 4
    assert owner_units["kv.v_payload.pages"] == 4

    republished = cache.publish(scope, lease, request.prompt_tokens)
    assert republished is not None
    backend.reclaim(lease)
    assert cache.snapshot()["cached_pages"] == 2
    cache.evict(republished)
    assert backend.pool.free_pages == backend.pool.page_capacity
    cache.assert_conserved()


def test_nested_snapshots_reference_each_page_once_in_the_resource_ledger() -> None:
    backend = _backend(pages=10)
    scope = _scope(backend)
    cache = _cache(backend)
    _request, shorter = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
    )
    assert shorter is not None

    hit = cache.lookup(scope, (1, 2, 3, 4, 5, 6))
    _request, _claims, lease = _reserve(
        backend,
        2,
        (1, 2, 3, 4, 5, 6),
        prefix=hit.snapshot,
    )
    longer = cache.publish(scope, lease, (1, 2, 3, 4, 5, 6))
    assert longer is not None
    backend.reclaim(lease)

    assert cache.snapshot()["cached_pages"] == 3
    assert backend.ledger.owner_claims(
        f"prefix-cache:{backend.spec.fingerprint}"
    ).units_by_pool()["kv.k_payload.pages"] == 3
    assert all(backend.pool.page(page_id).cache_references == 2 for page_id in shorter.page_ids)

    cache.evict(shorter)
    assert cache.snapshot()["cached_pages"] == 3
    assert all(backend.pool.page(page_id).cache_references == 1 for page_id in shorter.page_ids)
    cache.evict(longer)
    assert backend.pool.free_pages == backend.pool.page_capacity
    cache.assert_conserved()


def test_snapshot_publish_failure_rolls_back_pool_and_cross_lifetime_transfer(
    monkeypatch,
) -> None:
    backend = _backend(pages=8, suffix="publish-rollback")
    scope = _scope(backend)
    cache = _cache(backend)
    request, _claims, lease = _reserve(backend, 1, (1, 2, 3, 4))
    before = backend.ledger.owner_claims(lease.lease_id).units_by_pool()

    def fail_insert(self, request_id, tokens, block_ids):
        raise RuntimeError("injected radix insertion failure")

    monkeypatch.setattr(RadixCache, "insert", fail_insert)
    with pytest.raises(RuntimeError, match="injected"):
        cache.publish(scope, lease, request.prompt_tokens)

    assert backend.ledger.owner_claims(lease.lease_id).units_by_pool() == before
    assert not backend.ledger.has_owner(f"prefix-cache:{backend.spec.fingerprint}")
    assert all(
        backend.pool.page(page_id).cache_references == 0
        for page_id in backend.page_lease(lease).private_page_ids
    )
    backend.reclaim(lease)
    cache.assert_conserved()


def test_ttl_and_generation_staleness_are_misses_and_reclaim_cache_ownership() -> None:
    now = [100.0]
    backend = _backend(pages=8, suffix="ttl")
    scope = _scope(backend)
    cache = _cache(backend, ttl_seconds=5.0, clock=lambda: now[0])
    _request, first = _publish_and_reclaim(cache, scope, backend, 1, (1, 2, 3, 4))
    assert first is not None
    now[0] = 106.0
    expired = cache.lookup(scope, (1, 2, 3, 4))
    assert expired.hit is False
    assert cache.snapshot()["expired_evictions"] == 1
    assert backend.pool.free_pages == backend.pool.page_capacity

    _request, second = _publish_and_reclaim(cache, scope, backend, 2, (5, 6, 7, 8))
    assert second is not None
    backend.pool.generation += 1
    stale = cache.lookup(scope, (5, 6, 7, 8))
    assert stale.hit is False
    assert stale.miss_reason == "stale_generation"
    assert cache.snapshot()["stale_misses"] == 1
    backend.pool.generation -= 1
    cache.assert_conserved()


def test_global_and_tenant_lru_quotas_evict_oldest_snapshot() -> None:
    now = [1.0]
    backend = _backend(pages=10, suffix="quota")
    scope = _scope(backend)
    cache = _cache(
        backend,
        max_cached_pages=3,
        tenant_page_quotas={"tenant-a": 2},
        clock=lambda: now[0],
    )
    _request, first = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
        tenant="tenant-a",
    )
    now[0] = 2.0
    _request, second = _publish_and_reclaim(
        cache,
        scope,
        backend,
        2,
        (5, 6, 7, 8),
        tenant="tenant-a",
    )
    assert first is not None and second is not None
    assert cache.lookup(scope, (1, 2, 3, 4)).hit is False
    assert cache.lookup(scope, (5, 6, 7, 8)).hit is True
    snapshot = cache.snapshot()
    assert snapshot["cached_pages"] == 2
    assert snapshot["tenant_pages"] == {"tenant-a": 2}
    assert snapshot["quota_evictions"] == 1
    cache.assert_conserved()


def test_pressure_evicts_cache_before_fit_aware_admission_rejects_live_work() -> None:
    backend = _backend(pages=4, suffix="pressure")
    scope = _scope(backend)
    cache = _cache(backend, max_cached_pages=4)
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
    )
    assert snapshot is not None
    assert backend.pool.free_pages == 2

    manager = DenseKVAdmissionManager(
        backend,
        prefix_cache=cache,
        prefix_scope=scope,
        reuse_eligibility=lambda request: True,
    )
    pending = SimpleNamespace(
        request_id=2,
        prompt_tokens=(9, 10, 11, 12),
        max_new_tokens=1,
    )
    assert manager.plan_admission((pending,), max_items=1) == (2,)
    assert cache.snapshot()["pressure_evictions"] == 1
    assert cache.snapshot()["cached_pages"] == 0
    assert backend.has_request(2)
    manager.rollback_admission(pending)
    assert backend.pool.free_pages == backend.pool.page_capacity
    cache.assert_conserved()


def test_prefix_admission_advances_only_generic_prefill_cursor_and_saves_work() -> None:
    backend = _backend(pages=10, suffix="cursor")
    scope = _scope(backend)
    cache = _cache(backend)
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
    )
    assert snapshot is not None
    manager = DenseKVAdmissionManager(
        backend,
        prefix_cache=cache,
        prefix_scope=scope,
        reuse_eligibility=lambda request: True,
    )
    scheduler = ResidentBatchScheduler(capacity=2)
    request_id = scheduler.submit((1, 2, 3, 4, 9), max_new_tokens=1)
    pending = scheduler.pending_requests
    assert manager.plan_admission(pending, max_items=1) == (request_id,)
    admitted = scheduler.admit_pending(
        request_ids=(request_id,),
        reserve_callback=manager.reserve_admission,
        rollback_callback=manager.rollback_admission,
    )
    assert admitted == (request_id,)
    active = scheduler.active_batch.requests[request_id]
    assert active.next_prompt_index == 4
    work = scheduler.next_prefill_work(chunk_size=8)
    assert work is not None
    assert work.token_rows == ((9,),)

    miss_request = RequestState.from_tokens(99, (1, 2, 3, 4, 9), max_new_tokens=1)
    miss_claims = backend.estimate(miss_request, None, {"kind": "admission"})
    hit_claims = backend.ledger.owner_claims(f"lease:{request_id}")
    assert miss_claims.units_by_pool()["kv.k_payload.pages"] == 4
    assert hit_claims.units_by_pool()["kv.k_payload.pages"] == 2

    manager.rollback_admission(active)
    cache.assert_conserved()


@pytest.mark.parametrize("prefix_tokens", (256, 2048, 8192))
def test_complete_prefix_correctness_and_page_economics_at_serving_shapes(
    prefix_tokens: int,
) -> None:
    block_size = 256
    pages = prefix_tokens // block_size + 4
    backend = _backend(
        pages=pages,
        block_size=block_size,
        suffix=f"economics-{prefix_tokens}",
    )
    scope = _scope(backend)
    cache = _cache(backend, max_cached_pages=pages - 2)
    base_tokens = tuple(range(prefix_tokens))
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        base_tokens,
    )
    assert snapshot is not None
    prompt = (*base_tokens, prefix_tokens + 7)
    manager = DenseKVAdmissionManager(
        backend,
        prefix_cache=cache,
        prefix_scope=scope,
        reuse_eligibility=lambda request: True,
    )
    scheduler = ResidentBatchScheduler(capacity=1)
    request_id = scheduler.submit(prompt, max_new_tokens=1)
    assert manager.plan_admission(scheduler.pending_requests, max_items=1) == (
        request_id,
    )
    scheduler.admit_pending(
        request_ids=(request_id,),
        reserve_callback=manager.reserve_admission,
        rollback_callback=manager.rollback_admission,
    )
    active = scheduler.active_batch.requests[request_id]
    work = scheduler.next_prefill_work(chunk_size=block_size)
    assert work is not None
    assert active.prompt_tokens[: active.next_prompt_index] + work.token_rows[0] == prompt
    assert len(work.token_rows[0]) == 1

    owner = backend.ledger.owner_claims(f"lease:{request_id}").units_by_pool()
    uncached_pages = len(prompt) // block_size + 2
    assert owner["kv.k_payload.pages"] == 2
    assert owner["kv.k_payload.pages"] < uncached_pages
    manager.rollback_admission(active)
    cache.assert_conserved()


def test_resident_engine_prefix_hit_preserves_output_and_reclaims_independently() -> None:
    backend = _backend(pages=10, suffix="resident-output")
    scope = _scope(backend)
    cache = _cache(backend)
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
    )
    assert snapshot is not None
    admission = DenseKVAdmissionManager(
        backend,
        prefix_cache=cache,
        prefix_scope=scope,
        reuse_eligibility=lambda request: True,
    )

    class Runner:
        capacity = 1

        def __init__(self) -> None:
            self.kv_kernel_bundle_key = backend.spec.kernel_bundle_key
            self.kv_storage_layout_keys = (backend.storage_view().layout_key,)
            self.prefill_rows = []

        def prefill_batch_with_kv(self, work, *, kv_batch_view, commit):
            assert kv_batch_view.storage_view is backend.storage_view()
            assert commit is True
            self.prefill_rows.extend(work.token_rows)

        def decode_batch_with_kv(self, work, *, kv_batch_view, commit):
            assert kv_batch_view.storage_view is backend.storage_view()
            assert commit is True
            return tuple(
                GeneratedToken(request_id, 777, finished=True)
                for request_id in work.request_ids
            )

        def compact_batch(self, moves):
            del moves

        def reclaim(self, completed):
            del completed

    kernel_runner = Runner()
    loop = ResidentEngineLoop(
        DenseKVResidentRunnerAdapter(kernel_runner, admission),
        capacity=1,
        prefill_chunk_size=8,
    )
    request_id = loop.submit((1, 2, 3, 4, 9), max_new_tokens=1)
    loop.tick()
    loop.tick()
    completed = loop.completed.get(request_id)
    assert completed is not None
    assert completed.generated_tokens == (777,)
    assert kernel_runner.prefill_rows == [(9,)]
    assert loop.active_count == 0
    assert cache.lookup(scope, (1, 2, 3, 4, 9)).matched_token_count == 4
    cache.assert_conserved()


def test_sampled_or_unqualified_reuse_stays_disabled_until_explicit_policy() -> None:
    backend = _backend(pages=10, suffix="sampling-gate")
    scope = _scope(backend)
    cache = _cache(backend)
    _request, snapshot = _publish_and_reclaim(
        cache,
        scope,
        backend,
        1,
        (1, 2, 3, 4),
    )
    assert snapshot is not None
    with pytest.raises(ValueError, match="eligibility policy"):
        DenseKVAdmissionManager(
            backend,
            prefix_cache=cache,
            prefix_scope=scope,
        )

    manager = DenseKVAdmissionManager(
        backend,
        prefix_cache=cache,
        prefix_scope=scope,
        reuse_eligibility=lambda request: False,
    )
    sampled = SimpleNamespace(
        request_id=2,
        prompt_tokens=(1, 2, 3, 4, 9),
        max_new_tokens=1,
    )
    assert manager.plan_admission((sampled,), max_items=1) == (2,)
    claims = backend.ledger.owner_claims("lease:2").units_by_pool()
    assert claims["kv.k_payload.pages"] == 4
    manager.rollback_admission(sampled)
    assert cache.lookup(scope, sampled.prompt_tokens).hit is True
    cache.assert_conserved()


def test_incompatible_or_malformed_snapshot_never_casts_or_reinterprets_pages() -> None:
    backend = _backend(pages=8, suffix="incompatible")
    scope = _scope(backend)
    cache = _cache(backend)
    request, snapshot = _publish_and_reclaim(cache, scope, backend, 1, (1, 2, 3, 4))
    assert snapshot is not None
    stale = replace(snapshot, generation=snapshot.generation + 1)
    with pytest.raises(ValueError, match="incompatible or stale"):
        backend.estimate(request, stale, {"kind": "admission"})
    wrong_artifact = replace(snapshot, artifact_fingerprint="another-codec")
    with pytest.raises(ValueError, match="incompatible or stale"):
        backend.estimate(request, wrong_artifact, {"kind": "admission"})
    cache.assert_conserved()
