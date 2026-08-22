from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation import GeneratedToken, ResidentEngineLoop
from hipengine.kvcache import (
    ClaimLifetime,
    DenseKVAdmissionManager,
    DenseKVArtifactQualification,
    DenseKVCacheBackend,
    DenseKVResidentRunnerAdapter,
    GlobalKVPoolSet,
    KVCacheBackend,
    KVPageState,
    ResourceLedger,
    ResourceUnavailable,
    create_dense_bf16_backend,
    create_dense_int8_backend,
)


def _qualification(suffix: str) -> DenseKVArtifactQualification:
    return DenseKVArtifactQualification(
        artifact_fingerprint=f"qwen35-int8-kv:{suffix}",
        kl_divergence=0.01,
        top1_agreement=0.95,
        no_bf16_mirror=True,
        evidence_source="tests/fixtures/qwen35-int8-kv.json",
    )


def _dense_backend(codec: str, *, page_capacity: int, block_size: int, suffix: str):
    if codec == "bf16":
        return create_dense_bf16_backend(
            page_capacity=page_capacity,
            block_size=block_size,
            backend_fingerprint=f"dense-bf16:{suffix}",
        )
    return create_dense_int8_backend(
        page_capacity=page_capacity,
        block_size=block_size,
        qualification=_qualification(suffix),
    )


def _pool() -> GlobalKVPoolSet:
    return GlobalKVPoolSet(
        backend_fingerprint="dense:test",
        generation=3,
        plane_page_pointers={
            "k_payload": (0x1000, 0x1100, 0x9000, 0x9100, 0x15000),
            "v_payload": (0x2000, 0x2100, 0xA000, 0xA100, 0x16000),
        },
        pointer_table_pointers={"k_payload": 0x500, "v_payload": 0x600},
    )


def test_global_pool_uses_stable_arbitrary_page_indirection_across_segments() -> None:
    pool = _pool()
    before = pool.storage_view()
    lease = pool.allocate("lease:1", private_pages=3, growth_credit_pages=1)

    assert lease.private_page_ids == (0, 1, 2)
    assert lease.growth_credit_page_ids == (3,)
    assert pool.page_pointer("k_payload", 2) == 0x9000
    assert pool.page_pointer("v_payload", 2) == 0xA000
    assert pool.storage_view() == before
    assert {plane.role for plane in before.planes} == {
        "k_payload.page_table",
        "v_payload.page_table",
    }
    assert before.layout_key == "global-arbitrary-pages:g3"
    assert before.metadata_descriptor_ptr > 0
    assert before.metadata_descriptor_bytes == 256
    assert pool.snapshot()["counts"] == {"active_private": 3, "reserved_credit": 1, "free": 1}
    pool.assert_conserved()


def test_global_pool_cross_plane_fragmentation_reuses_noncontiguous_pages() -> None:
    pool = _pool()
    pool.allocate("first", private_pages=2, growth_credit_pages=0)
    pool.allocate("middle", private_pages=2, growth_credit_pages=0)
    pool.release("first")

    fragmented = pool.allocate("fragmented", private_pages=3, growth_credit_pages=0)
    assert fragmented.private_page_ids == (0, 1, 4)
    assert tuple(
        pool.page_pointer("k_payload", page_id)
        for page_id in fragmented.private_page_ids
    ) == (0x1000, 0x1100, 0x15000)
    assert tuple(
        pool.page_pointer("v_payload", page_id)
        for page_id in fragmented.private_page_ids
    ) == (0x2000, 0x2100, 0x16000)
    pool.release("fragmented")
    pool.release("middle")
    pool.assert_conserved()


def test_global_pool_growth_credit_cow_inflight_and_reclaim_are_safe() -> None:
    pool = _pool()
    parent = pool.allocate("parent", private_pages=2, growth_credit_pages=1)
    appended_page = pool.consume_growth_credit("parent")
    assert appended_page == 2
    assert pool.lease("parent").growth_credit_page_ids == ()

    pool.allocate(
        "child",
        private_pages=0,
        growth_credit_pages=1,
        shared_page_ids=(parent.private_page_ids[0],),
    )
    assert pool.page(parent.private_page_ids[0]).state is KVPageState.ACTIVE_SHARED
    cow_page = pool.copy_on_write("child", parent.private_page_ids[0])
    assert cow_page == 3
    assert pool.page(parent.private_page_ids[0]).state is KVPageState.ACTIVE_PRIVATE

    pool.mark_in_flight("parent", (parent.private_page_ids[0],), epoch=11)
    assert pool.page(parent.private_page_ids[0]).state is KVPageState.IN_FLIGHT
    pool.release("parent")
    assert pool.page(parent.private_page_ids[0]).state is KVPageState.IN_FLIGHT
    pool.retire_epoch(11)
    assert pool.page(parent.private_page_ids[0]).state is KVPageState.FREE

    pool.release("child")
    assert pool.free_pages == pool.page_capacity
    pool.assert_conserved()


def test_global_pool_rejects_release_while_session_pinned_until_unpinned() -> None:
    pool = _pool()
    lease = pool.allocate("session", private_pages=1, growth_credit_pages=0)
    page_id = lease.private_page_ids[0]
    pool.pin_session("session", (page_id,))
    pool.release("session")
    assert pool.page(page_id).state is KVPageState.PINNED_SESSION
    pool.unpin_session((page_id,))
    assert pool.page(page_id).state is KVPageState.FREE
    pool.assert_conserved()


def test_dense_bf16_backend_reserve_growth_view_and_reclaim_share_one_lifecycle() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=16,
        block_size=4,
        backend_fingerprint="dense-bf16:test-artifact",
    )
    assert isinstance(backend, DenseKVCacheBackend)
    assert isinstance(backend, KVCacheBackend)
    plan = backend.plan_pools(None)
    assert plan is backend.plan_pools(None)
    request = SimpleNamespace(request_id=5, prompt_tokens=tuple(range(6)), max_new_tokens=7)
    claims = backend.estimate(request, None, {"kind": "admission"})
    assert claims.metadata_dict() == {
        "growth_credit_pages": 1,
        "private_pages": 2,
        "shared_page_ids": "",
    }
    assert claims.units_by_pool() == {
        "kv.k_payload.pages": 3,
        "kv.request_rows": 1,
        "kv.v_payload.pages": 3,
    }

    lease = backend.reserve(claims)
    details = backend.page_lease(lease)
    assert len(details.private_page_ids) == 2
    assert len(details.growth_credit_page_ids) == 1
    view = backend.storage_view(lease)
    assert {plane.role for plane in view.planes} == {
        "k_payload.page_table",
        "v_payload.page_table",
    }
    batch_view = backend.prepare(
        SimpleNamespace(request_ids=(5,), context_lengths=(6,))
    )
    assert batch_view.storage_view is view
    assert batch_view.live_spans.live_counts.shape[0] == 1
    assert batch_view.live_spans.storage_dtype.value == "bf16"

    backend.append_page(lease)
    assert len(backend.page_lease(lease).private_page_ids) == 3
    assert backend.page_lease(lease).growth_credit_page_ids == ()
    backend.renew_growth_credit(lease, pages=2)
    assert len(backend.page_lease(lease).growth_credit_page_ids) == 2
    assert backend.ledger.snapshot()["pools"]["kv.k_payload.pages"]["used"] == 5

    delta = backend.reclaim(lease)
    assert delta is not None
    assert all(change.units <= 0 for change in delta.changes)
    assert all(pool["used"] == 0 for pool in backend.ledger.snapshot()["pools"].values())
    assert backend.pool.free_pages == backend.pool.page_capacity
    backend.ledger.assert_conserved()
    backend.pool.assert_conserved()


@pytest.mark.parametrize(
    ("codec", "expected_codec"),
    (("bf16", "bf16"), ("int8", "int8_per_token_head")),
)
def test_dense_admission_manager_drives_resident_lifecycle_and_exact_reclaim(
    codec: str,
    expected_codec: str,
) -> None:
    backend = _dense_backend(
        codec,
        page_capacity=8,
        block_size=2,
        suffix="resident",
    )
    admission = DenseKVAdmissionManager(backend, lookahead=4, max_bypasses=2)

    class Runner:
        capacity = 4

        def __init__(self) -> None:
            self.views = []
            self.kv_kernel_bundle_key = backend.spec.kernel_bundle_key
            self.kv_storage_layout_keys = (backend.storage_view().layout_key,)

        def prefill_batch_with_kv(self, work, *, kv_batch_view, commit: bool):
            assert commit is True
            self.views.append(kv_batch_view)

        def decode_batch_with_kv(self, work, *, kv_batch_view, commit: bool):
            assert commit is True
            self.views.append(kv_batch_view)
            return tuple(
                GeneratedToken(request_id, 1000 + request_id, finished=True)
                for request_id in work.request_ids
            )

        def compact_batch(self, moves):
            del moves

        def reclaim(self, completed):
            del completed

    kernel_runner = Runner()
    runner = DenseKVResidentRunnerAdapter(kernel_runner, admission)
    loop = ResidentEngineLoop(runner, capacity=4, prefill_chunk_size=8)
    first = loop.submit([1, 2, 3], max_new_tokens=1)
    second = loop.submit([4, 5, 6], max_new_tokens=1)
    events = loop.tick()
    assert [event.request_id for event in events if event.kind == "admitted"] == [
        first,
        second,
    ]
    assert backend.pool.free_pages == 2
    assert backend.ledger.snapshot()["pools"]["kv.k_payload.pages"]["used"] == 6
    assert kernel_runner.views
    stable_storage_view = backend.storage_view()
    assert all(view.storage_view is stable_storage_view for view in kernel_runner.views)

    for _ in range(4):
        if loop.active_count == 0:
            break
        loop.tick()
    assert loop.active_count == 0
    assert backend.pool.free_pages == 8
    assert all(
        pool["used"] == 0
        for pool in backend.ledger.snapshot()["pools"].values()
    )
    resources = loop.observability_snapshot()["resources"]
    assert resources["backend"]["codec"] == expected_codec
    assert resources["admission"]["admitted_total"] == 2
    assert all(view.storage_view is stable_storage_view for view in kernel_runner.views)

    cancelled = loop.submit([7, 8], max_new_tokens=2)
    loop.tick()
    assert backend.has_request(cancelled)
    loop.cancel(cancelled)
    assert not backend.has_request(cancelled)
    assert backend.pool.free_pages == 8
    assert all(
        pool["used"] == 0
        for pool in backend.ledger.snapshot()["pools"].values()
    )
    backend.pool.assert_conserved()
    backend.ledger.assert_conserved()


def test_dense_runner_adapter_fails_closed_on_unregistered_bundle_or_layout() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=4,
        block_size=2,
        backend_fingerprint="dense-bf16:binding-rejection",
    )
    admission = DenseKVAdmissionManager(backend)
    bad_runner = SimpleNamespace(
        capacity=2,
        kv_kernel_bundle_key="wrong-bundle",
        kv_storage_layout_keys=(backend.storage_view().layout_key,),
        prefill_batch_with_kv=lambda *args, **kwargs: None,
        decode_batch_with_kv=lambda *args, **kwargs: (),
    )
    with pytest.raises(ValueError, match="kernel bundle"):
        DenseKVResidentRunnerAdapter(bad_runner, admission)

    bad_runner.kv_kernel_bundle_key = backend.spec.kernel_bundle_key
    bad_runner.kv_storage_layout_keys = ("wrong-layout",)
    with pytest.raises(ValueError, match="storage layout"):
        DenseKVResidentRunnerAdapter(bad_runner, admission)


def test_dense_manager_pressure_waits_then_refills_and_rejects_impossible() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=4,
        block_size=2,
        backend_fingerprint="dense-bf16:pressure",
    )
    admission = DenseKVAdmissionManager(backend, lookahead=4, max_bypasses=2)
    first = SimpleNamespace(
        request_id=1,
        prompt_tokens=(1, 2, 3),
        max_new_tokens=1,
    )
    second = SimpleNamespace(
        request_id=2,
        prompt_tokens=(4, 5, 6),
        max_new_tokens=1,
    )

    assert admission.plan_admission((first, second), max_items=2) == (1,)
    assert backend.pool.free_pages == 1
    assert admission.plan_admission((second,), max_items=1) == ()
    assert admission.controller.pending_state(2).blocking_resources == (
        "kv.k_payload.pages",
        "kv.v_payload.pages",
    )
    admission.reclaim_request(first)
    assert admission.plan_admission((second,), max_items=1) == (2,)
    admission.reclaim_request(second)

    impossible = SimpleNamespace(
        request_id=3,
        prompt_tokens=tuple(range(9)),
        max_new_tokens=1,
    )
    with pytest.raises(ResourceUnavailable) as error:
        admission.plan_admission((impossible,), max_items=1)
    assert error.value.impossible is True
    assert error.value.resource == "kv.k_payload.pages"
    assert backend.pool.free_pages == 4
    backend.pool.assert_conserved()
    backend.ledger.assert_conserved()


def test_dense_int8_backend_is_artifact_qualified_and_has_no_bf16_mirror() -> None:
    with pytest.raises(ValueError, match="qualification"):
        create_dense_int8_backend(
            page_capacity=8,
            block_size=4,
            qualification=None,  # type: ignore[arg-type]
        )
    qualification = _qualification("abc123")
    backend = create_dense_int8_backend(
        page_capacity=8,
        block_size=4,
        qualification=qualification,
    )
    assert backend.spec.hot_codec_key == "int8_per_token_head"
    assert backend.spec.artifact_fingerprint == "qwen35-int8-kv:abc123"
    pools = {pool.pool_id: pool for pool in backend.plan_pools(None).pools}
    assert set(pools) == {
        "kv.k_payload.pages",
        "kv.k_scale.pages",
        "kv.request_rows",
        "kv.v_payload.pages",
        "kv.v_scale.pages",
    }
    assert all("bf16" not in pool_id and "mirror" not in pool_id for pool_id in pools)
    assert ClaimLifetime.LEASE in pools["kv.k_scale.pages"].lifetimes

    request = SimpleNamespace(request_id=9, prompt_tokens=(1, 2, 3), max_new_tokens=2)
    claims = backend.estimate(request, None, {"kind": "admission"})
    lease = backend.reserve(claims)
    roles = {plane.role for plane in backend.storage_view(lease).planes}
    assert roles == {
        "k_payload.page_table",
        "k_scale.page_table",
        "v_payload.page_table",
        "v_scale.page_table",
    }
    backend.reclaim(lease)
    backend.ledger.assert_conserved()


def test_dense_backend_uses_atomic_ledger_rollback_when_physical_allocation_fails(
    monkeypatch,
) -> None:
    backend = create_dense_bf16_backend(
        page_capacity=4,
        block_size=4,
        backend_fingerprint="dense-bf16:rollback",
    )
    request = SimpleNamespace(request_id=3, prompt_tokens=(1,), max_new_tokens=1)
    claims = backend.estimate(request, None, {"kind": "admission"})
    before = backend.ledger.snapshot()

    def fail_allocate(*args, **kwargs):
        raise MemoryError("injected physical allocation failure")

    monkeypatch.setattr(backend.pool, "allocate", fail_allocate)
    with pytest.raises(MemoryError, match="injected"):
        backend.reserve(claims)

    after = backend.ledger.snapshot()
    assert after["owners"] == before["owners"]
    assert after["provisional_reservations"] == 0
    assert all(pool["used"] == 0 for pool in after["pools"].values())
    backend.ledger.assert_conserved()


def test_dense_backend_repeated_lease_cycles_leave_no_reservation_or_page_history() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=8,
        block_size=4,
        backend_fingerprint="dense-bf16:cycle-soak",
    )
    for request_id in range(64):
        request = SimpleNamespace(
            request_id=request_id,
            prompt_tokens=(1, 2, 3),
            max_new_tokens=1,
        )
        lease = backend.reserve(
            backend.estimate(request, None, {"kind": "admission"})
        )
        backend.reclaim(lease)

    ledger = backend.ledger.snapshot()
    assert ledger["active_reservations"] == 0
    assert ledger["owners"] == {}
    assert backend.pool.snapshot()["lease_count"] == 0
    assert backend.pool.free_pages == backend.pool.page_capacity
    backend.ledger.assert_conserved()
    backend.pool.assert_conserved()


def test_dense_backend_plan_can_initialize_an_external_generic_ledger() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=6,
        block_size=2,
        backend_fingerprint="dense-bf16:external-ledger",
    )
    ledger = ResourceLedger(backend.plan_pools(None))
    request = SimpleNamespace(request_id=2, prompt_tokens=(1, 2), max_new_tokens=1)
    claims = backend.estimate(request, None, {"kind": "admission"})
    reservation = ledger.reserve_provisional(claims)
    ledger.commit(reservation, owner_id="external:2")
    ledger.release("external:2")
    ledger.assert_conserved()
