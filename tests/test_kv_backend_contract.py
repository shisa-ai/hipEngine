from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache.backend import (
    ClaimConfidence,
    ClaimLifetime,
    KVBackendSpec,
    KVBatchView,
    KVCacheBackend,
    KVLease,
    KVPoolPlan,
    KVPoolSpec,
    KVStoragePlane,
    KVStorageView,
    LeaseState,
    PrefixMode,
    ResourceClaim,
    ResourceClaimSet,
    ResourceChange,
    ResourceDelta,
    TransactionMode,
)
from hipengine.kvcache.spans import KVLiveSpans


def _spec(codec: str = "bf16", *, layout: str | None = None) -> KVBackendSpec:
    return KVBackendSpec(
        topology_key="paged_dense",
        hot_codec_key=codec,
        tier_key="device_only",
        layout_fingerprint=layout or f"layout:{codec}",
        artifact_fingerprint=f"artifact:{codec}",
        prefix_mode=PrefixMode.IMMUTABLE_PAGES,
        transaction_mode=TransactionMode.JOURNAL,
        kernel_bundle_key=f"bundle:{codec}",
    )


def test_backend_spec_is_immutable_and_identity_covers_every_compatibility_axis() -> None:
    baseline = _spec()
    same = _spec()
    changed = _spec(layout="layout:other")

    assert baseline.identity_fingerprint == same.identity_fingerprint
    assert baseline.execution_compatibility_prefix == (
        "paged_dense",
        "bf16",
        "device_only",
        "layout:bf16",
        "artifact:bf16",
        "bundle:bf16",
    )
    assert changed.identity_fingerprint != baseline.identity_fingerprint
    with pytest.raises(FrozenInstanceError):
        baseline.hot_codec_key = "int8_per_token_head"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("topology_key", ""),
        ("hot_codec_key", " "),
        ("tier_key", ""),
        ("layout_fingerprint", ""),
        ("artifact_fingerprint", ""),
        ("kernel_bundle_key", ""),
    ),
)
def test_backend_spec_rejects_empty_identity_fields(field: str, value: str) -> None:
    values = {
        "topology_key": "paged_dense",
        "hot_codec_key": "bf16",
        "tier_key": "device_only",
        "layout_fingerprint": "layout",
        "artifact_fingerprint": "artifact",
        "prefix_mode": PrefixMode.IMMUTABLE_PAGES,
        "transaction_mode": TransactionMode.JOURNAL,
        "kernel_bundle_key": "bundle",
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        KVBackendSpec(**values)


def test_pool_plan_and_claim_set_are_named_format_neutral_vectors() -> None:
    spec = _spec("int8_per_token_head")
    plan = KVPoolPlan(
        backend_fingerprint=spec.identity_fingerprint,
        pools=(
            KVPoolSpec("payload", 4096, unit="bytes"),
            KVPoolSpec("scales", 128, unit="bytes"),
            KVPoolSpec("resident_rows", 32, unit="rows"),
        ),
        generation=7,
    )
    claims = ResourceClaimSet(
        stage="prefill",
        claims=(
            ResourceClaim("payload", 1024, unit="bytes", lifetime=ClaimLifetime.LEASE),
            ResourceClaim("scales", 32, unit="bytes", lifetime=ClaimLifetime.LEASE),
            ResourceClaim(
                "resident_rows",
                1,
                unit="rows",
                lifetime=ClaimLifetime.LEASE,
                confidence=ClaimConfidence.BOUNDED,
            ),
        ),
    )

    assert plan.pool("scales").capacity == 128
    assert claims.amount("payload") == 1024
    assert claims.pool_ids == ("payload", "scales", "resident_rows")
    shared_pool_claims = ResourceClaimSet(
        stage="decode",
        claims=(
            ResourceClaim("payload", 4, lifetime=ClaimLifetime.LEASE),
            ResourceClaim("payload", 2, lifetime=ClaimLifetime.WORK_ITEM),
        ),
    )
    assert shared_pool_claims.pool_ids == ("payload",)
    assert shared_pool_claims.amount("payload") == 6
    assert shared_pool_claims.amount(
        "payload", lifetime=ClaimLifetime.WORK_ITEM
    ) == 2
    with pytest.raises(ValueError, match="duplicate pool/lifetime"):
        ResourceClaimSet(
            stage="decode",
            claims=(
                ResourceClaim("payload", 1),
                ResourceClaim("payload", 2),
            ),
        )
    with pytest.raises(ValueError, match="duplicate pool_id"):
        KVPoolPlan(
            backend_fingerprint=spec.identity_fingerprint,
            pools=(KVPoolSpec("payload", 1), KVPoolSpec("payload", 2)),
        )
    with pytest.raises(ValueError, match="amount must be an integer"):
        ResourceClaim("payload", 1.5)  # type: ignore[arg-type]


def test_storage_view_carries_stable_planes_and_batch_liveness_separately() -> None:
    device = Device("hip", 0)
    spans = KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(0x1000, (2, 4), "int32", device),
        live_counts=Tensor.from_handle(0x2000, (2,), "int64", device),
        max_live_count=1024,
        storage_dtype="bf16",
    )
    storage = KVStorageView(
        layout_key="paged_dense/bf16/v1",
        generation=3,
        planes=(
            KVStoragePlane(
                role="key_payload",
                ptr=0x3000,
                dtype="bf16",
                shape=(16, 256, 4, 256),
                strides=(262144, 1024, 256, 1),
            ),
            KVStoragePlane(
                role="value_payload",
                ptr=0x4000,
                dtype="bf16",
                shape=(16, 256, 4, 256),
                strides=(262144, 1024, 256, 1),
            ),
        ),
        device_metadata_ptr=0x5000,
        device_metadata_nbytes=256,
        artifact_fingerprint="artifact:bf16",
    )
    batch = KVBatchView(
        spans=spans,
        storage=storage,
        kernel_bundle_key="bundle:bf16",
        execution_compatibility_key=("model", "gfx1100", "decode", "c2"),
    )

    assert batch.spans is spans
    assert batch.storage.plane("key_payload").ptr == 0x3000
    assert batch.storage.generation == 3
    with pytest.raises(ValueError, match="duplicate storage plane role"):
        KVStorageView(
            layout_key="bad",
            generation=0,
            planes=(storage.planes[0], storage.planes[0]),
            artifact_fingerprint="artifact",
        )


def test_lease_and_delta_are_operation_scoped_immutable_records() -> None:
    claims = ResourceClaimSet(
        stage="decode",
        claims=(ResourceClaim("workspace", 2048, lifetime=ClaimLifetime.WORK_ITEM),),
    )
    lease = KVLease(
        lease_id="lease-1",
        request_id=9,
        backend_fingerprint="backend",
        claims=claims,
        generation=2,
        state=LeaseState.PROVISIONAL,
    )
    delta = ResourceDelta(
        operation_id=lease.lease_id,
        changes=(
            ResourceChange(
                "workspace",
                -1024,
                lifetime=ClaimLifetime.WORK_ITEM,
                reason="unused split workspace",
            ),
        ),
    )

    assert delta.changes[0].amount == -1024
    with pytest.raises(ValueError, match="operation_id"):
        ResourceDelta(operation_id="", changes=delta.changes)
    with pytest.raises(ValueError, match="non-zero"):
        ResourceChange("workspace", 0)


class _FakeBackend:
    def __init__(self, spec: KVBackendSpec, pools: tuple[KVPoolSpec, ...]) -> None:
        self.spec = spec
        self._plan = KVPoolPlan(
            backend_fingerprint=spec.identity_fingerprint,
            pools=pools,
        )

    def plan_pools(self, load_plan):
        return self._plan

    def estimate(self, request, prefix, stage):
        return ResourceClaimSet(
            stage=str(stage),
            claims=tuple(
                ResourceClaim(pool.pool_id, 1, unit=pool.unit)
                for pool in self._plan.pools
            ),
        )

    def reserve(self, claims):
        return claims

    def prepare(self, work_item):
        return work_item

    def begin_transaction(self, rows, draft):
        return (rows, draft)

    def commit(self, operation, result):
        return (operation, result)

    def rollback(self, operation):
        return operation

    def reclaim(self, lease):
        return lease

    def prefix_lookup(self, tokens):
        return tuple(tokens)

    def maintenance(self, budget):
        return []


def test_distinct_fake_storage_backends_implement_one_scheduler_facing_protocol() -> None:
    bf16 = _FakeBackend(
        _spec("bf16"),
        (KVPoolSpec("key_payload", 4096), KVPoolSpec("value_payload", 4096)),
    )
    int8 = _FakeBackend(
        _spec("int8_per_token_head"),
        (
            KVPoolSpec("key_payload", 2048),
            KVPoolSpec("value_payload", 2048),
            KVPoolSpec("key_scales", 128),
            KVPoolSpec("value_scales", 128),
        ),
    )

    assert isinstance(bf16, KVCacheBackend)
    assert isinstance(int8, KVCacheBackend)
    assert type(bf16) is type(int8)
    assert bf16.plan_pools(None).pool_ids == ("key_payload", "value_payload")
    assert int8.plan_pools(None).pool_ids == (
        "key_payload",
        "value_payload",
        "key_scales",
        "value_scales",
    )
