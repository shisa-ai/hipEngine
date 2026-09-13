"""CPU-only tests for per-rank KV pool geometry and aggregate claims.

No HIP/ROCm is touched: geometry is arithmetic and the claim path drives a
caller-supplied allocator.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hipengine.distributed.kv import (
    KvClaim,
    KvClaimSet,
    KvPoolError,
    KvPoolGeometry,
    claim_all,
    full_attention_layer_count,
    local_kv_head_counts,
    resolve_kv_claims,
    resolve_kv_geometry,
)
from hipengine.distributed.plan import DistributedPlan

Q4_LAYER_TYPES = tuple(
    "full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)
)


def _config(**overrides):
    base = {"layer_types": Q4_LAYER_TYPES, "head_count_kv": 4, "key_length": 256}
    base.update(overrides)
    return SimpleNamespace(**base)


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
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
    assert geometry.full_attention_layers == 16
    assert geometry.kv_heads_per_rank == (2, 2)
    # 16 layers x (K+V) x 2 heads x 256 dim x 2 bytes = 32 KiB per token.
    assert geometry.bytes_per_token(0) == 32768
    assert geometry.plane_bytes(0) == 32768 * 8192
    assert geometry.spans_bytes(0) == 16 * 16 * 8192
    assert geometry.total_bytes(0) == geometry.plane_bytes(0) + geometry.spans_bytes(0)
    # Halving the group halves each rank's plane, and N=1 holds the whole pool.
    n1 = resolve_kv_geometry(_config(), world_size=1, context_tokens=8192)
    assert n1.plane_bytes(0) == 2 * geometry.plane_bytes(0)


def test_geometry_scales_linearly_with_context() -> None:
    short = resolve_kv_geometry(_config(), world_size=2, context_tokens=4096)
    long = resolve_kv_geometry(_config(), world_size=2, context_tokens=16384)
    assert long.plane_bytes(0) == 4 * short.plane_bytes(0)
    assert long.spans_bytes(0) == 4 * short.spans_bytes(0)


def test_geometry_rejects_impossible_requests() -> None:
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(), world_size=2, context_tokens=0)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(), world_size=2, context_tokens=8192, kv_dtype_bytes=0)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(), world_size=2, context_tokens=8192, spans_bytes_per_token_per_layer=-1)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(head_count_kv=0), world_size=2, context_tokens=8192)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(key_length=0), world_size=2, context_tokens=8192)
    with pytest.raises(KvPoolError):
        resolve_kv_geometry(_config(layer_types=("linear_attention",) * 4), world_size=2, context_tokens=8192)
    with pytest.raises(KvPoolError):
        KvPoolGeometry(full_attention_layers=1, kv_heads_per_rank=(0,), head_dim=256, context_tokens=128)
    with pytest.raises(KvPoolError):
        KvPoolGeometry(
            full_attention_layers=1, kv_heads_per_rank=(1,), head_dim=256, context_tokens=128, schema_version=99
        )


def test_qwen38_degree_refusal_matches_the_weight_planner() -> None:
    """N=3 fails for the same reason the shard planner refuses it."""

    with pytest.raises(KvPoolError) as error:
        resolve_kv_geometry(_config(), world_size=3, context_tokens=8192)
    assert "4 groups" in str(error.value)
    assert "does not divide evenly across 3 ranks" in str(error.value)


def test_resolve_kv_claims_binds_geometry_to_the_plan() -> None:
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
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
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
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
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=1024)
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
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
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
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
    claims = resolve_kv_claims(_plan(2), geometry)
    released: list[str] = []

    def allocate(claim: KvClaim) -> str:
        raise RuntimeError(f"rank {claim.rank} out of memory")

    with pytest.raises(KvPoolError):
        claim_all(claims, allocate, released.append)
    assert released == []


def test_claim_all_rollback_survives_a_failing_release() -> None:
    # 4 KV heads cannot be split three ways, so use a geometry that can.
    geometry = resolve_kv_geometry(_config(head_count_kv=6), world_size=3, context_tokens=1024)
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
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
    claims = resolve_kv_claims(_plan(2), geometry)
    buffers = claim_all(claims, lambda claim: claim.total_bytes, lambda buffer: None)
    assert buffers == [claims.claims[0].total_bytes, claims.claims[1].total_bytes]


def test_claim_set_is_json_serializable() -> None:
    geometry = resolve_kv_geometry(_config(), world_size=2, context_tokens=8192)
    payload = json.loads(json.dumps(resolve_kv_claims(_plan(2), geometry).to_dict()))
    assert payload["world_size"] == 2
    assert payload["geometry"]["kv_heads_per_rank"] == [2, 2]
    assert payload["claims"][1]["rank"] == 1
    assert payload["claims"][0]["total_bytes"] == geometry.total_bytes(0)
