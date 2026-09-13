"""Per-rank KV pool geometry and the aggregate claim set.

A TP group does not share one KV pool: every rank owns the KV planes for its own
attention heads and its own ``KVLiveSpans`` metadata, and a request is only
admitted when *every* rank can claim its share. This module resolves that
geometry against the same head partition the weight planner uses, and provides
the all-or-nothing claim path so a partial admission cannot leave one rank
holding memory another rank never got.

Design rules (docs/QWEN38-27B-GFX1100-TP2.md):
  * the KV head partition is the weight planner's partition, not a second
    opinion - a rank's KV heads are exactly the heads its weights serve;
  * claim order is the rank order and rollback releases in reverse;
  * nothing here allocates: it resolves bytes and drives a caller-supplied
    allocator, so it is testable without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hipengine.distributed.plan import DistributedPlan, PlanError

SCHEMA_VERSION = 1

#: ``KVLiveSpans`` metadata is ``(base_offsets, live_counts, token_positions,
#: evict_mask)`` per layer per request; four 32-bit fields per token is the
#: dense-policy size and the figure the capacity accounting uses.
DEFAULT_SPANS_BYTES_PER_TOKEN_PER_LAYER = 16

#: KV plane dtype used by the production path.
DEFAULT_KV_DTYPE_BYTES = 2


class KvPoolError(ValueError):
    """Raised when KV pool geometry cannot be served by the requested group."""


def local_kv_head_counts(head_count_kv: int, world_size: int) -> tuple[int, ...]:
    """Split ``head_count_kv`` across ``world_size`` ranks the way weights are split.

    ``partition_groups`` is the weight planner's partition; reusing it here is
    what keeps the pool geometry and the shard plan from disagreeing. When the
    group is larger than the head count every rank still gets at least one head
    (replicated), because a rank with no KV heads cannot run the layer.
    """

    from hipengine.loading.qwen35_gguf_shards import AxisSegment, partition_groups

    if int(head_count_kv) <= 0:
        raise KvPoolError("head_count_kv must be positive")
    if int(world_size) <= 0:
        raise KvPoolError("world_size must be positive")
    try:
        ranges = partition_groups(AxisSegment(start=0, stop=int(head_count_kv), group=1), int(world_size))
    except Exception as error:  # noqa: BLE001 - re-raise with KV context
        raise KvPoolError(f"cannot split {head_count_kv} KV heads across {world_size} ranks: {error}") from error
    return tuple(int(stop) - int(start) for start, stop in ranges)


def full_attention_layer_count(config: Any) -> int:
    """Count the layers that carry a paged KV cache for this model config."""

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if not layer_types:
        raise KvPoolError("model config does not declare layer_types")
    return sum(1 for layer_type in layer_types if str(layer_type) == "full_attention")


@dataclass(frozen=True)
class KvPoolGeometry:
    """Per-rank KV pool geometry for one TP group and one context length."""

    full_attention_layers: int
    kv_heads_per_rank: tuple[int, ...]
    head_dim: int
    context_tokens: int
    kv_dtype_bytes: int = DEFAULT_KV_DTYPE_BYTES
    spans_bytes_per_token_per_layer: int = DEFAULT_SPANS_BYTES_PER_TOKEN_PER_LAYER
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.full_attention_layers) <= 0:
            raise KvPoolError("a TP group needs at least one full-attention layer")
        if not self.kv_heads_per_rank:
            raise KvPoolError("kv_heads_per_rank must cover every rank")
        if any(int(heads) <= 0 for heads in self.kv_heads_per_rank):
            raise KvPoolError("every rank must own at least one KV head")
        if int(self.head_dim) <= 0:
            raise KvPoolError("head_dim must be positive")
        if int(self.context_tokens) <= 0:
            raise KvPoolError("context_tokens must be positive")
        if int(self.kv_dtype_bytes) <= 0:
            raise KvPoolError("kv_dtype_bytes must be positive")
        if int(self.spans_bytes_per_token_per_layer) < 0:
            raise KvPoolError("spans_bytes_per_token_per_layer must be non-negative")
        if int(self.schema_version) != SCHEMA_VERSION:
            raise KvPoolError(f"unsupported KV geometry schema version {self.schema_version}")

    @property
    def world_size(self) -> int:
        return len(self.kv_heads_per_rank)

    def bytes_per_token(self, rank: int) -> int:
        """K and V plane bytes this rank reads/writes per token."""

        heads = int(self.kv_heads_per_rank[int(rank)])
        return 2 * int(self.full_attention_layers) * heads * int(self.head_dim) * int(self.kv_dtype_bytes)

    def plane_bytes(self, rank: int) -> int:
        return self.bytes_per_token(rank) * int(self.context_tokens)

    def spans_bytes(self, rank: int) -> int:
        return (
            int(self.full_attention_layers)
            * int(self.spans_bytes_per_token_per_layer)
            * int(self.context_tokens)
        )

    def total_bytes(self, rank: int) -> int:
        return self.plane_bytes(rank) + self.spans_bytes(rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_attention_layers": int(self.full_attention_layers),
            "kv_heads_per_rank": [int(heads) for heads in self.kv_heads_per_rank],
            "head_dim": int(self.head_dim),
            "context_tokens": int(self.context_tokens),
            "kv_dtype_bytes": int(self.kv_dtype_bytes),
            "spans_bytes_per_token_per_layer": int(self.spans_bytes_per_token_per_layer),
            "schema_version": int(self.schema_version),
            "plane_bytes_per_rank": [self.plane_bytes(rank) for rank in range(self.world_size)],
            "total_bytes_per_rank": [self.total_bytes(rank) for rank in range(self.world_size)],
        }


def resolve_kv_geometry(
    config: Any,
    *,
    world_size: int,
    context_tokens: int,
    kv_dtype_bytes: int = DEFAULT_KV_DTYPE_BYTES,
    spans_bytes_per_token_per_layer: int = DEFAULT_SPANS_BYTES_PER_TOKEN_PER_LAYER,
) -> KvPoolGeometry:
    """Resolve KV geometry for ``world_size`` ranks from a model config."""

    head_count_kv = int(getattr(config, "head_count_kv", 0) or 0)
    head_dim = int(getattr(config, "key_length", 0) or 0)
    if head_count_kv <= 0:
        raise KvPoolError("model config does not declare head_count_kv")
    if head_dim <= 0:
        raise KvPoolError("model config does not declare key_length")
    return KvPoolGeometry(
        full_attention_layers=full_attention_layer_count(config),
        kv_heads_per_rank=local_kv_head_counts(head_count_kv, int(world_size)),
        head_dim=head_dim,
        context_tokens=int(context_tokens),
        kv_dtype_bytes=int(kv_dtype_bytes),
        spans_bytes_per_token_per_layer=int(spans_bytes_per_token_per_layer),
    )


@dataclass(frozen=True)
class KvClaim:
    """One rank's KV admission request."""

    rank: int
    device: str
    plane_bytes: int
    spans_bytes: int

    def __post_init__(self) -> None:
        if int(self.rank) < 0:
            raise KvPoolError("claim rank must be non-negative")
        if int(self.plane_bytes) < 0 or int(self.spans_bytes) < 0:
            raise KvPoolError("claim byte counts must be non-negative")

    @property
    def total_bytes(self) -> int:
        return int(self.plane_bytes) + int(self.spans_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "device": self.device,
            "plane_bytes": int(self.plane_bytes),
            "spans_bytes": int(self.spans_bytes),
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class KvClaimSet:
    """The whole group's claims for one context length."""

    claims: tuple[KvClaim, ...]
    geometry: KvPoolGeometry

    def __post_init__(self) -> None:
        if [claim.rank for claim in self.claims] != list(range(len(self.claims))):
            raise KvPoolError("claims must be contiguous and ordered from rank 0")
        if len(self.claims) != self.geometry.world_size:
            raise KvPoolError(
                f"{len(self.claims)} claims do not cover {self.geometry.world_size} ranks"
            )

    @property
    def world_size(self) -> int:
        return len(self.claims)

    @property
    def total_bytes(self) -> int:
        return sum(claim.total_bytes for claim in self.claims)

    @property
    def largest_claim_bytes(self) -> int:
        return max(claim.total_bytes for claim in self.claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "total_bytes": self.total_bytes,
            "largest_claim_bytes": self.largest_claim_bytes,
            "claims": [claim.to_dict() for claim in self.claims],
            "geometry": self.geometry.to_dict(),
        }


def resolve_kv_claims(
    plan: DistributedPlan,
    geometry: KvPoolGeometry,
    *,
    free_bytes_per_rank: Sequence[int] | None = None,
    reserve_bytes: int = 0,
) -> KvClaimSet:
    """Bind KV geometry to a plan, refusing a group the devices cannot hold.

    ``free_bytes_per_rank`` is optional; when given, every rank must have room
    for its claim plus ``reserve_bytes``. The check happens before any
    allocation, so an over-committed group never starts claiming.
    """

    if int(plan.world_size) != int(geometry.world_size):
        raise KvPoolError(
            f"plan has {plan.world_size} ranks but KV geometry covers {geometry.world_size}"
        )
    if free_bytes_per_rank is not None:
        if len(free_bytes_per_rank) != int(plan.world_size):
            raise KvPoolError("free_bytes_per_rank must cover every rank")
        for rank in range(int(plan.world_size)):
            available = int(free_bytes_per_rank[rank]) - int(reserve_bytes)
            needed = geometry.total_bytes(rank)
            if needed > available:
                raise KvPoolError(
                    f"rank {rank} needs {needed} bytes for its KV share plus {int(reserve_bytes)} reserved, "
                    f"but only {int(free_bytes_per_rank[rank])} are free"
                )
    claims = tuple(
        KvClaim(
            rank=rank,
            device=str(plan.rank_spec(rank).device),
            plane_bytes=geometry.plane_bytes(rank),
            spans_bytes=geometry.spans_bytes(rank),
        )
        for rank in range(int(plan.world_size))
    )
    return KvClaimSet(claims=claims, geometry=geometry)


def claim_all(
    claim_set: KvClaimSet,
    allocate: Callable[[KvClaim], Any],
    release: Callable[[Any], None],
) -> list[Any]:
    """Claim every rank's KV share, releasing everything on any failure.

    This is the aggregate rollback rule: a request is admitted only when all
    ranks succeed. Allocation follows rank order and rollback runs in reverse,
    so the release order mirrors the acquire order.
    """

    buffers: list[Any] = []
    try:
        for claim in claim_set.claims:
            buffers.append(allocate(claim))
    except Exception as error:  # noqa: BLE001 - re-raised as an aggregate failure
        released = 0
        for buffer in reversed(buffers):
            try:
                release(buffer)
                released += 1
            except Exception:  # noqa: BLE001 - rollback must not mask the original failure
                continue
        raise KvPoolError(
            f"KV claim failed after {len(buffers)} of {claim_set.world_size} ranks; "
            f"released {released} buffer(s): {error!r}"
        ) from error
    if len(buffers) != claim_set.world_size:
        raise PlanError(
            f"claim produced {len(buffers)} buffers for {claim_set.world_size} ranks"
        )
    return buffers
