from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kvcache import (
    ColdObjectKey,
    ColdTierStore,
    DenseKVCacheBackend,
    KVTCColdCodec,
    TieredKVCacheBackend,
    evaluate_restore_economics,
)


def _hot_backend(*, capacity: int = 64) -> DenseKVCacheBackend:
    return DenseKVCacheBackend(
        codec="bf16",
        page_capacity=capacity,
        block_size=4,
        artifact_fingerprint="artifact:hot-fixture",
    )


def _request(request_id: int, *, tokens: int = 8, max_new: int = 2):
    return SimpleNamespace(
        request_id=int(request_id),
        prompt_tokens=tuple(range(int(tokens))),
        max_new_tokens=int(max_new),
    )


def _lease(backend: DenseKVCacheBackend, request_id: int):
    request = _request(request_id)
    return backend.reserve(backend.estimate(request, None, {}))


def _key(backend: DenseKVCacheBackend, *, token: int = 1, state: str = "state-a"):
    return ColdObjectKey.from_tokens(
        hot_backend_fingerprint=backend.spec.fingerprint,
        artifact_fingerprint=backend.spec.artifact_fingerprint,
        hot_generation=backend.generation,
        hot_codec=backend.spec.hot_codec_key,
        cold_codec="kvtc_zlib_v1",
        token_ids=(token, token + 1, token + 2),
        request_scope=f"request:{token}",
        state_fingerprint=state,
    )


def test_kvtc_codec_is_deterministic_and_validates_complete_key() -> None:
    backend = _hot_backend()
    key = _key(backend)
    codec = KVTCColdCodec(level=6)
    payload = (b"kv-state-" * 128) + bytes(range(64))

    first = codec.encode(key, payload)
    second = codec.encode(key, payload)

    assert first == second
    assert first.encoded_bytes < first.original_bytes
    assert codec.decode(key, first.encoded) == payload
    with pytest.raises(ValueError, match="key fingerprint"):
        codec.decode(_key(backend, state="other"), first.encoded)
    corrupted = bytearray(first.encoded)
    corrupted[-1] ^= 0xFF
    with pytest.raises((ValueError, zlib_error())):
        codec.decode(key, bytes(corrupted))


def zlib_error():
    import zlib

    return zlib.error


def test_cold_store_spills_to_nvme_enforces_quota_and_drains(tmp_path) -> None:
    backend = _hot_backend()
    codec = KVTCColdCodec()
    first_key = _key(backend, token=1)
    second_key = _key(backend, token=10)
    first = codec.encode(first_key, b"A" * 4096)
    second = codec.encode(second_key, b"B" * 4096)
    store = ColdTierStore(
        host_capacity_bytes=first.encoded_bytes,
        nvme_capacity_bytes=second.encoded_bytes * 2,
        nvme_directory=tmp_path / "nvme",
        tenant_quota_bytes={"tenant-a": first.encoded_bytes + second.encoded_bytes},
    )

    host = store.put(
        first_key,
        first.encoded,
        original_bytes=first.original_bytes,
        tenant_id="tenant-a",
    )
    nvme = store.put(
        second_key,
        second.encoded,
        original_bytes=second.original_bytes,
        tenant_id="tenant-a",
    )

    assert host.tier == "host"
    assert nvme.tier == "nvme"
    assert nvme.path is not None
    assert codec.decode(second_key, store.get(second_key)[1]) == b"B" * 4096
    with pytest.raises(MemoryError, match="quota"):
        store.put(
            _key(backend, token=20),
            second.encoded,
            original_bytes=second.original_bytes,
            tenant_id="tenant-a",
        )
    store.pin(first_key)
    evicted = store.evict_lru(required_bytes=1)
    assert [record.key for record in evicted] == [second_key]
    assert store.contains(first_key)
    store.pin(first_key, pinned=False)
    assert store.delete(first_key).tier == "host"
    store.assert_conserved()
    assert store.snapshot()["object_count"] == 0
    assert not list((tmp_path / "nvme").glob("*.kvtc"))


def test_tiered_backend_offloads_restores_and_keeps_attention_hot(tmp_path) -> None:
    hot = _hot_backend()
    store = ColdTierStore(
        host_capacity_bytes=1 << 20,
        nvme_capacity_bytes=1 << 20,
        nvme_directory=tmp_path,
    )
    tiered = TieredKVCacheBackend(
        hot,
        store=store,
        transfer_workspace_bytes=1 << 20,
        maintenance_budget_bytes=1 << 20,
    )
    original_lease = _lease(hot, 1)
    key = tiered.cold_key_for_tokens(
        token_ids=(1, 2, 3),
        request_scope="request:1",
        state_fingerprint="hybrid-state-v1",
    )
    payload = b"serialized-hot-kv" * 512

    tiered.enqueue_offload(
        key=key,
        lease=original_lease,
        hot_payload=payload,
        tenant_id="tenant-a",
    )
    offload = tiered.maintenance()

    assert len(offload) == 1 and offload[0].passed is True
    assert hot.has_request(1) is False
    assert store.contains(key)
    assert tiered.tier_ledger.snapshot()["stats"]["commits"] == 1
    restored: list[tuple[object, bytes]] = []
    tiered.enqueue_restore(
        key=key,
        request=_request(2),
        restore_callback=lambda lease, data: restored.append((lease, data)),
    )
    restore = tiered.maintenance()

    assert len(restore) == 1 and restore[0].passed is True
    assert restore[0].lease is not None and restore[0].lease.request_id == 2
    assert restored == [(restore[0].lease, payload)]
    assert store.contains(key) is False
    assert hot.has_request(2) is True
    hot_view = tiered.prepare(SimpleNamespace(request_ids=(2,), context_lengths=(8,)))
    assert hot_view.storage_view.layout_key.startswith("global-arbitrary-pages")
    assert tiered.spec.hot_codec_key == hot.spec.hot_codec_key
    assert tiered.spec.tier_key == "optional_kvtc_zlib_v1"
    hot.reclaim(restore[0].lease)
    tiered.assert_conserved()
    assert tiered.tier_ledger.snapshot()["owners"] == {}
    assert tiered.tier_ledger.snapshot()["pools"]["tier.transfer_workspace_bytes"]["high_water"] == len(payload)


def test_failed_restore_preserves_cold_object_and_reclaims_hot_lease(tmp_path) -> None:
    hot = _hot_backend()
    tiered = TieredKVCacheBackend(
        hot,
        store=ColdTierStore(
            host_capacity_bytes=1 << 20,
            nvme_capacity_bytes=0,
        ),
        transfer_workspace_bytes=1 << 20,
    )
    lease = _lease(hot, 3)
    key = tiered.cold_key_for_tokens(
        token_ids=(3,),
        request_scope="request:3",
        state_fingerprint="state",
    )
    tiered.enqueue_offload(key=key, lease=lease, hot_payload=b"payload" * 100)
    assert tiered.maintenance()[0].passed is True

    def fail_restore(_lease, _payload):
        raise RuntimeError("restore injection")

    tiered.enqueue_restore(
        key=key,
        request=_request(4),
        restore_callback=fail_restore,
    )
    result = tiered.maintenance()[0]

    assert result.passed is False
    assert "restore injection" in str(result.error)
    assert tiered.store.contains(key)
    assert hot.has_request(4) is False
    tiered.drain()
    assert tiered.store.snapshot()["object_count"] == 0


def test_cold_lru_eviction_releases_tier_ledger_ownership() -> None:
    hot = _hot_backend()
    tiered = TieredKVCacheBackend(
        hot,
        store=ColdTierStore(host_capacity_bytes=1 << 20),
    )
    keys = []
    for request_id in (1, 2):
        lease = _lease(hot, request_id)
        key = tiered.cold_key_for_tokens(
            token_ids=(request_id,),
            request_scope=f"request:{request_id}",
            state_fingerprint="state",
        )
        keys.append(key)
        tiered.enqueue_offload(
            key=key,
            lease=lease,
            hot_payload=bytes([request_id]) * 1024,
        )
        assert tiered.maintenance()[0].passed is True
    tiered.store.get(keys[0])
    tiered.enqueue_evict(required_bytes=1)

    eviction = tiered.maintenance()[0]

    assert eviction.passed is True
    assert eviction.evicted_keys == (keys[1].fingerprint,)
    assert tiered.store.contains(keys[0])
    assert not tiered.store.contains(keys[1])
    assert len(tiered.tier_ledger.snapshot()["owners"]) == 1
    tiered.drain()


def test_cold_key_rejects_backend_artifact_generation_and_codec_mismatch() -> None:
    hot = _hot_backend()
    tiered = TieredKVCacheBackend(
        hot,
        store=ColdTierStore(host_capacity_bytes=4096),
    )
    valid = tiered.cold_key_for_tokens(
        token_ids=(1,),
        request_scope="request:1",
        state_fingerprint="state",
    )
    mutations = (
        {"hot_backend_fingerprint": "other"},
        {"artifact_fingerprint": "other"},
        {"hot_generation": 2},
        {"hot_codec": "int8"},
        {"cold_codec": "other"},
    )
    from dataclasses import replace

    for mutation in mutations:
        with pytest.raises(ValueError, match="mismatch"):
            tiered.enqueue_restore(
                key=replace(valid, **mutation),
                request=_request(2),
                restore_callback=lambda _lease, _payload: None,
            )


def test_restore_economics_selects_only_measured_ttft_savings() -> None:
    win = evaluate_restore_economics(
        restore_seconds=0.2,
        recompute_seconds=0.8,
        minimum_savings_seconds=0.1,
    )
    loss = evaluate_restore_economics(
        restore_seconds=0.7,
        recompute_seconds=0.8,
        minimum_savings_seconds=0.2,
    )

    assert win.use_restore is True
    assert win.savings_seconds == pytest.approx(0.6)
    assert loss.use_restore is False


def test_tier_backend_gate_measures_restore_and_drains() -> None:
    from scripts import tier_backend_gate

    args = tier_backend_gate.build_parser().parse_args(
        [
            "--payload-bytes",
            "4096",
            "--repetitions",
            "2",
            "--recompute-rounds",
            "2",
        ]
    )
    payload = tier_backend_gate.run(args)

    assert payload["status"] == "accepted_host_tier"
    assert payload["passed"] is True
    assert payload["hot_backend"]["attention_storage_view_remains_hot"] is True
    assert payload["final"]["store"]["object_count"] == 0
    assert payload["final"]["ledger"]["owners"] == {}
    assert payload["economics"]["restore_median_seconds"] >= 0.0
    assert payload["economics"]["recompute_median_seconds"] >= 0.0


def test_tier_drain_cleans_pending_and_cold_ownership() -> None:
    hot = _hot_backend()
    tiered = TieredKVCacheBackend(
        hot,
        store=ColdTierStore(host_capacity_bytes=1 << 20),
    )
    lease = _lease(hot, 1)
    key = tiered.cold_key_for_tokens(
        token_ids=(1,),
        request_scope="request:1",
        state_fingerprint="state",
    )
    tiered.enqueue_offload(key=key, lease=lease, hot_payload=b"payload" * 32)

    tiered.drain()

    assert not hot.has_request(1)
    assert tiered.store.snapshot()["object_count"] == 0
    assert tiered.observability_snapshot()["tier"]["pending_maintenance"] == 0
    tiered.assert_conserved()
