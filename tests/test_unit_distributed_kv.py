"""CPU-only tests for per-rank KV pool geometry and aggregate claims.

No HIP/ROCm is touched: geometry is arithmetic and the claim path drives a
caller-supplied allocator.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.distributed.kv import (
    SCHEMA_VERSION,
    KvClaim,
    KvClaimSet,
    KvPoolError,
    KvPoolGeometry,
    KvSpansLayout,
    build_group_kv_plan,
    claim_all,
    full_attention_layer_count,
    local_kv_head_counts,
    plane_pool_id,
    reserve_group_kv,
    resolve_kv_claims,
    resolve_kv_geometry,
    spans_pool_id,
)
from hipengine.distributed.plan import DistributedPlan
from hipengine.kvcache.backend import KVPoolPlan
from hipengine.kvcache.ledger import ResourceLedger

Q4_LAYER_TYPES = tuple(
    "full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)
)


def _config(**overrides):
    base = {
        "layer_types": Q4_LAYER_TYPES,
        "head_count_kv": 4,
        "key_length": 256,
        "value_length": 256,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _dense_spans() -> KvSpansLayout:
    """The dense-policy metadata layout the retained capacity figures used."""

    return KvSpansLayout.dense_policy()


def _plan(world_size: int) -> DistributedPlan:
    return DistributedPlan.resolve(list(range(world_size)), hidden_size=5120)


def test_local_kv_head_counts_matches_the_weight_partition() -> None:
    assert local_kv_head_counts(4, 1) == (4,)
    assert local_kv_head_counts(4, 2) == (2, 2)
    assert local_kv_head_counts(4, 4) == (1, 1, 1, 1)
    # More ranks than heads replicates rather than handing a rank nothing.
    assert local_kv_head_counts(2, 4) == (1, 1, 1, 1)
    with pytest.raises(KvPoolError):
        local_kv_head_counts(4, 3)
    with pytest.raises(KvPoolError):
        local_kv_head_counts(0, 2)
    with pytest.raises(KvPoolError):
        local_kv_head_counts(4, 0)


def test_full_attention_layer_count_requires_layer_types() -> None:
    assert full_attention_layer_count(_config()) == 16
    assert full_attention_layer_count(_config(layer_types=("full_attention",) * 3)) == 3
    with pytest.raises(KvPoolError):
        full_attention_layer_count(SimpleNamespace())


def test_geometry_arithmetic_for_qwen38_at_n2() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    assert geometry.full_attention_layers == 16
    assert geometry.kv_heads_per_rank == (2, 2)
    # 16 layers x (K+V) x 2 heads x 256 dim x 2 bytes = 32 KiB per token.
    assert geometry.bytes_per_token(0) == 32768
    assert geometry.plane_bytes(0) == 32768 * 8192
    assert geometry.spans_bytes(0) == 16 * 16 * 8192
    assert geometry.total_bytes(0) == geometry.plane_bytes(0) + geometry.spans_bytes(0)
    # Halving the group halves each rank's plane, and N=1 holds the whole pool.
    n1 = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=1, context_tokens=8192)
    assert n1.plane_bytes(0) == 2 * geometry.plane_bytes(0)


def test_geometry_scales_linearly_with_context() -> None:
    short = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=4096)
    long = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=16384)
    assert long.plane_bytes(0) == 4 * short.plane_bytes(0)
    assert long.spans_bytes(0) == 4 * short.spans_bytes(0)


def test_geometry_rejects_impossible_requests() -> None:
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=0)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(
        _config(), spans=_dense_spans(), world_size=2, context_tokens=8192, kv_dtype_bytes=0
    )
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(
            _config(value_length=0), spans=_dense_spans(), world_size=2, context_tokens=8192
        )
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(head_count_kv=0), spans=_dense_spans(), world_size=2, context_tokens=8192)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(key_length=0), spans=_dense_spans(), world_size=2, context_tokens=8192)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(layer_types=("linear_attention",) * 4), spans=_dense_spans(), world_size=2, context_tokens=8192)
    with pytest.raises(KvPoolError):
        KvPoolGeometry(full_attention_layers=1, kv_heads_per_rank=(0,), key_head_dim=256, context_tokens=128)
    with pytest.raises(KvPoolError):
        KvPoolGeometry(
            full_attention_layers=1,
            kv_heads_per_rank=(1,),
            key_head_dim=256,
            context_tokens=128,
            schema_version=99,
        )


def test_qwen38_degree_refusal_matches_the_weight_planner() -> None:
    """N=3 fails for the same reason the shard planner refuses it."""

    with pytest.raises(KvPoolError) as error:
        resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=3, context_tokens=8192)
    assert "4 groups" in str(error.value)
    assert "does not divide evenly across 3 ranks" in str(error.value)


def test_resolve_kv_claims_binds_geometry_to_the_plan() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    claims = resolve_kv_claims(_plan(2), geometry)
    assert claims.world_size == 2
    assert [claim.rank for claim in claims.claims] == [0, 1]
    assert claims.claims[0].device == "hip:0"
    assert claims.claims[1].device == "hip:1"
    assert claims.total_bytes == sum(claim.total_bytes for claim in claims.claims)
    assert claims.largest_claim_bytes == max(claim.total_bytes for claim in claims.claims)
    with pytest.raises(KvPoolError):
        resolve_kv_claims(_plan(1), geometry)


def test_resolve_kv_claims_checks_free_memory_before_allocating() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    needed = geometry.total_bytes(0)
    # Exactly enough on both ranks passes; one byte short on rank 1 fails.
    resolve_kv_claims(_plan(2), geometry, free_bytes_per_rank=[needed, needed])
    with pytest.raises(KvPoolError) as error:
        resolve_kv_claims(_plan(2), geometry, free_bytes_per_rank=[needed, needed - 1])
    assert "rank 1" in str(error.value)
    with pytest.raises(KvPoolError):
        resolve_kv_claims(_plan(2), geometry, free_bytes_per_rank=[needed])
    # A reserve is subtracted from what is available.
    with pytest.raises(KvPoolError):
        resolve_kv_claims(_plan(2), geometry, free_bytes_per_rank=[needed, needed], reserve_bytes=1)


def test_claim_set_rejects_incomplete_or_misordered_ranks() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=1024)
    good = KvClaim(rank=0, device="hip:0", plane_bytes=1, spans_bytes=1)
    with pytest.raises(KvPoolError):
        KvClaimSet(claims=(good,), geometry=geometry)
    with pytest.raises(KvPoolError):
        KvClaimSet(
            claims=(KvClaim(rank=1, device="hip:1", plane_bytes=1, spans_bytes=1), good), geometry=geometry
        )
    with pytest.raises(KvPoolError):
        KvClaim(rank=-1, device="hip:0", plane_bytes=1, spans_bytes=1)
    with pytest.raises(KvPoolError):
        KvClaim(rank=0, device="hip:0", plane_bytes=-1, spans_bytes=1)


def test_claim_all_is_all_or_nothing() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    claims = resolve_kv_claims(_plan(2), geometry)

    allocated: list[int] = []
    released: list[str] = []

    def allocate(claim: KvClaim) -> str:
        allocated.append(claim.rank)
        if claim.rank == 1:
            raise RuntimeError("rank 1 out of memory")
        return f"buffer-{claim.rank}"

    with pytest.raises(KvPoolError) as error:
        claim_all(claims, allocate, released.append)
    assert allocated == [0, 1]
    assert released == ["buffer-0"], "rank 0's claim must be released when rank 1 fails"
    assert "1 of 2 ranks" in str(error.value)
    assert "rank 1 out of memory" in str(error.value)


def test_claim_all_failure_on_the_first_rank_releases_nothing() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    claims = resolve_kv_claims(_plan(2), geometry)
    released: list[str] = []

    def allocate(claim: KvClaim) -> str:
        raise RuntimeError(f"rank {claim.rank} out of memory")

    with pytest.raises(KvPoolError):
        claim_all(claims, allocate, released.append)
    assert released == []


def test_claim_all_rollback_survives_a_failing_release() -> None:
    # 4 KV heads cannot be split three ways, so use a geometry that can.
    geometry = resolve_kv_geometry(_config(head_count_kv=6), spans=_dense_spans(), world_size=3, context_tokens=1024)
    assert geometry.kv_heads_per_rank == (2, 2, 2)
    claims = resolve_kv_claims(_plan(3), geometry)
    released: list[str] = []

    def allocate(claim: KvClaim) -> str:
        if claim.rank == 2:
            raise RuntimeError("rank 2 out of memory")
        return f"buffer-{claim.rank}"

    def release(buffer: str) -> None:
        released.append(buffer)
        if buffer == "buffer-1":
            raise RuntimeError("release failed")

    with pytest.raises(KvPoolError) as error:
        claim_all(claims, allocate, release)
    # Reverse order, and a failing release does not hide the original error.
    assert released == ["buffer-1", "buffer-0"]
    assert "released 1 buffer(s)" in str(error.value)
    assert "rank 2 out of memory" in str(error.value)


def test_claim_all_success_returns_one_buffer_per_rank() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    claims = resolve_kv_claims(_plan(2), geometry)
    buffers = claim_all(claims, lambda claim: claim.total_bytes, lambda buffer: None)
    assert buffers == [claims.claims[0].total_bytes, claims.claims[1].total_bytes]


def test_claim_set_is_json_serializable() -> None:
    geometry = resolve_kv_geometry(_config(), spans=_dense_spans(), world_size=2, context_tokens=8192)
    payload = json.loads(json.dumps(resolve_kv_claims(_plan(2), geometry).to_dict()))
    assert payload["world_size"] == 2
    assert payload["geometry"]["kv_heads_per_rank"] == [2, 2]
    assert payload["claims"][1]["rank"] == 1
    assert payload["claims"][0]["total_bytes"] == geometry.total_bytes(0)


# -- backend-owned geometry ---------------------------------------------------


def test_key_and_value_widths_are_independent_config_fields() -> None:
    """The plane term sums the two declared widths instead of doubling one."""

    asymmetric = resolve_kv_geometry(
        _config(key_length=256, value_length=128),
        spans=_dense_spans(),
        world_size=2,
        context_tokens=8192,
    )
    assert asymmetric.key_head_dim == 256
    assert asymmetric.value_dim == 128
    assert asymmetric.value_dim_assumed_equal is False
    # 16 layers x 2 heads x (256 + 128) x 2 bytes = 24 KiB per token.
    assert asymmetric.bytes_per_token(0) == 16 * 2 * (256 + 128) * 2
    assert asymmetric.plane_bytes(0) == 16 * 2 * 384 * 2 * 8192

    symmetric = resolve_kv_geometry(
        _config(key_length=256, value_length=256),
        spans=_dense_spans(),
        world_size=2,
        context_tokens=8192,
    )
    assert symmetric.value_dim == 256
    # The difference is the value plane alone: 16 layers x 2 heads x 128 x 2 B.
    assert symmetric.bytes_per_token(0) == asymmetric.bytes_per_token(0) + 16 * 2 * 128 * 2


def test_geometry_records_an_inferred_value_width() -> None:
    """A config without value_length gets equal widths and says so."""

    geometry = resolve_kv_geometry(
        SimpleNamespace(
            layer_types=Q4_LAYER_TYPES, head_count_kv=4, key_length=256
        ),
        spans=_dense_spans(),
        world_size=2,
        context_tokens=8192,
    )
    assert geometry.value_dim_assumed_equal is True
    assert geometry.value_dim == geometry.key_head_dim
    payload = geometry.to_dict()
    assert payload["value_head_dim"] == 256
    assert payload["value_head_dim_assumed_equal"] is True
    assert payload["schema_version"] == SCHEMA_VERSION == 2


def test_spans_footprint_follows_the_declared_mode() -> None:
    """Each KVLiveSpans mode has its own tensor set, so its own footprint."""

    # Fixed-page: int32 block table per layer per sequence plus int32 live counts,
    # and an int64 row_positions per sequence. No per-token metadata.
    paged = KvSpansLayout.paged_uniform(max_pages_per_sequence=64)
    assert paged.mode == "uniform"
    assert paged.bytes_per_token_per_layer == 0
    assert paged.bytes_per_sequence_per_layer == 4 * 64 + 4
    assert paged.bytes_for(layers=16, context_tokens=8192) == 16 * (4 * 64 + 4) + 8

    # Per-head-variable: int32 base_offsets and live_counts per head per layer.
    per_head = KvSpansLayout.per_head_variable(kv_heads=2)
    assert per_head.mode == "per_head_variable"
    assert per_head.bytes_per_sequence_per_layer == 8 * 2
    assert per_head.bytes_for(layers=16, context_tokens=8192) == 16 * 16

    # Sliding ring: int64 token_positions, one mask byte and one int32 base offset
    # per capacity slot, so it is per token per layer.
    ring = KvSpansLayout.sliding_ring(slots_per_sequence=2048)
    assert ring.mode == "sliding_ring"
    assert ring.bytes_per_token_per_layer == 13
    assert ring.bytes_for(layers=16, context_tokens=2048) == 13 * 16 * 2048 + 8

    # The dense-policy layout is the historical 16 B/token/layer accounting.
    dense = KvSpansLayout.dense_policy()
    assert dense.mode == "dense_policy"
    assert dense.bytes_for(layers=16, context_tokens=8192) == 16 * 16 * 8192


def test_spans_layout_refuses_an_unknown_mode_or_impossible_terms() -> None:
    with pytest.raises(KvPoolError):
        KvSpansLayout("dms", 4)
    with pytest.raises(KvPoolError):
        KvSpansLayout("uniform", -1)
    with pytest.raises(KvPoolError):
        KvSpansLayout("uniform", 4, per_sequence_count=0)
    with pytest.raises(KvPoolError):
        KvSpansLayout.paged_uniform(max_pages_per_sequence=0)
    with pytest.raises(KvPoolError):
        KvSpansLayout.per_head_variable(kv_heads=0)
    with pytest.raises(KvPoolError):
        KvSpansLayout.sliding_ring(slots_per_sequence=0)


def test_a_paged_geometry_scales_with_the_page_table_not_the_context() -> None:
    """The uniform mode's metadata does not grow with context tokens."""

    layout = KvSpansLayout.paged_uniform(max_pages_per_sequence=256)
    short = resolve_kv_geometry(
        _config(), spans=layout, world_size=2, context_tokens=4096
    )
    long = resolve_kv_geometry(
        _config(), spans=layout, world_size=2, context_tokens=16384
    )
    assert long.plane_bytes(0) == 4 * short.plane_bytes(0)
    assert long.spans_bytes(0) == short.spans_bytes(0), (
        "a fixed-page block table does not grow with context tokens"
    )


def test_spans_layout_is_recorded_in_the_geometry_payload() -> None:
    geometry = resolve_kv_geometry(
        _config(),
        spans=KvSpansLayout.sliding_ring(slots_per_sequence=4096),
        world_size=2,
        context_tokens=4096,
    )
    payload = json.loads(json.dumps(geometry.to_dict()))
    assert payload["spans"]["mode"] == "sliding_ring"
    assert payload["spans"]["bytes_per_token_per_layer"] == 13
    assert payload["spans_bytes_per_rank"][0] == geometry.spans_bytes(0)


# -- scheduler-owned reservations ---------------------------------------------


def _rank_plans(world_size: int = 2, *, context_tokens: int = 8192):
    geometry = resolve_kv_geometry(
        _config(), spans=_dense_spans(), world_size=world_size, context_tokens=context_tokens
    )
    claims = resolve_kv_claims(_plan(world_size), geometry)
    return build_group_kv_plan(
        claims, backend_fingerprint="qwen38-q4km-tp2", generation=1
    )


def test_group_plan_declares_per_rank_pools_and_claims() -> None:
    plans = _rank_plans()
    assert [plan.rank for plan in plans] == [0, 1]
    assert plans[0].pool_ids == (plane_pool_id(0), spans_pool_id(0))
    assert plans[0].claims.units_by_pool() == {
        plane_pool_id(0): plans[0].geometry_bytes - 16 * 16 * 8192,
        spans_pool_id(0): 16 * 16 * 8192,
    }
    # Capacity equals the rank's own region, so the claim fits exactly.
    for plan in plans:
        capacities = {pool.pool_id: pool.capacity for pool in plan.plan.pools}
        assert capacities[plane_pool_id(plan.rank)] == plan.claims.units_by_pool()[plane_pool_id(plan.rank)]
        assert capacities[spans_pool_id(plan.rank)] == plan.claims.units_by_pool()[spans_pool_id(plan.rank)]
    payload = json.loads(json.dumps(plans[0].to_dict()))
    assert payload["generation"] == 1
    assert payload["pool_ids"][0] == plane_pool_id(0)


def test_reserve_group_kv_holds_every_rank_then_commits() -> None:
    plans = _rank_plans()
    ledgers = [ResourceLedger(plan.plan) for plan in plans]
    reservation = reserve_group_kv(plans, ledgers=ledgers, group_id="g1")
    assert reservation.world_size == 2
    assert reservation.settled is None
    # Provisional holds are visible to the ledger before any commit.
    for ledger in reservation.ledgers:
        snapshot = ledger.snapshot()
        assert snapshot["provisional_reservations"] == 1
        assert snapshot["stats"].get("commits", 0) == 0
        ledger.assert_conserved()
    reservation.commit()
    assert reservation.settled == "committed"
    for index, ledger in enumerate(reservation.ledgers):
        snapshot = ledger.snapshot()
        assert snapshot["stats"].get("commits", 0) == 1
        assert ledger.has_owner(f"{reservation.owner_id}:rank{index}")
        ledger.assert_conserved()
    with pytest.raises(KvPoolError):
        reservation.commit()
    with pytest.raises(KvPoolError):
        reservation.rollback()


def test_a_second_group_cannot_claim_the_same_rank_pools() -> None:
    """The pools are the rank's KV budget, so ownership is exclusive."""

    plans = _rank_plans()
    ledgers = [ResourceLedger(plan.plan) for plan in plans]
    first = reserve_group_kv(plans, ledgers=ledgers).commit()
    with pytest.raises(KvPoolError) as error:
        reserve_group_kv(plans, ledgers=ledgers)
    assert "rank 0" in str(error.value)
    assert "cannot reserve its KV share" in str(error.value)
    assert plane_pool_id(0) in str(error.value)
    # The committed group is untouched by the refused second group.
    assert first.settled == "committed"
    for index, ledger in enumerate(ledgers):
        assert ledger.has_owner(f"{first.owner_id}:rank{index}")
        ledger.assert_conserved()


def test_reserve_group_kv_rolls_back_when_a_later_rank_fails() -> None:
    plans = _rank_plans()
    # Shrink rank 1's plane pool below its claim: its reservation cannot fit.
    broken = list(plans)
    rank_one = broken[1]
    shrunk_pool = replace(rank_one.plan.pools[0], capacity=rank_one.plan.pools[0].capacity - 1)
    broken[1] = replace(
        rank_one,
        plan=KVPoolPlan(
            backend_fingerprint=rank_one.plan.backend_fingerprint,
            generation=rank_one.plan.generation,
            pools=(shrunk_pool, rank_one.plan.pools[1]),
        ),
    )
    with pytest.raises(KvPoolError) as error:
        reserve_group_kv(broken)
    assert "rank 1" in str(error.value)
    # Rank 0's provisional hold was released, so the original plan still fits.
    reserve_group_kv(plans).commit()


def test_rollback_releases_every_rank_hold() -> None:
    plans = _rank_plans()
    ledgers = [ResourceLedger(plan.plan) for plan in plans]
    reservation = reserve_group_kv(plans, ledgers=ledgers)
    reservation.rollback()
    assert reservation.settled == "rolled_back"
    for ledger in reservation.ledgers:
        snapshot = ledger.snapshot()
        assert snapshot["stats"].get("rollbacks", 0) == 1
        assert all(pool["used"] == 0 for pool in snapshot["pools"].values())
        ledger.assert_conserved()
    # The pools are free again, so a fresh group can take them.
    reserve_group_kv(plans, ledgers=ledgers).commit()


def test_group_plan_refuses_a_device_budget_below_the_geometry() -> None:
    geometry = resolve_kv_geometry(
        _config(), spans=_dense_spans(), world_size=2, context_tokens=8192
    )
    claims = resolve_kv_claims(_plan(2), geometry)
    needed = [geometry.total_bytes(0), geometry.total_bytes(1)]
    build_group_kv_plan(
        claims,
        backend_fingerprint="fp",
        generation=1,
        device_budget_bytes=needed,
    )
    with pytest.raises(KvPoolError) as error:
        build_group_kv_plan(
            claims,
            backend_fingerprint="fp",
            generation=1,
            device_budget_bytes=[needed[0], needed[1] - 1],
        )
    assert "rank 1" in str(error.value)
    with pytest.raises(KvPoolError):
        build_group_kv_plan(
            claims, backend_fingerprint="fp", generation=1, device_budget_bytes=[needed[0]]
        )
    with pytest.raises(KvPoolError):
        build_group_kv_plan(claims, backend_fingerprint="fp", generation=0)


def test_ledger_owned_claims_drive_the_device_allocator_after_commit() -> None:
    """claim_all is the buffer path, and it runs after the ledger commits."""

    plans = _rank_plans()
    reservation = reserve_group_kv(plans).commit()
    claims = KvClaimSet(
        claims=tuple(
            KvClaim(
                rank=plan.rank,
                device=plan.device,
                plane_bytes=plan.claims.units_by_pool()[plane_pool_id(plan.rank)],
                spans_bytes=plan.claims.units_by_pool()[spans_pool_id(plan.rank)],
            )
            for plan in plans
        ),
        geometry=resolve_kv_geometry(
            _config(), spans=_dense_spans(), world_size=2, context_tokens=8192
        ),
    )
    released: list[str] = []
    buffers = claim_all(claims, lambda claim: f"buffer-{claim.rank}", released.append)
    assert buffers == ["buffer-0", "buffer-1"]
    assert released == []
    assert reservation.settled == "committed"
