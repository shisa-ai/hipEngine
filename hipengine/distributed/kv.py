"""Per-rank KV pool geometry and the aggregate claim set.

A TP group does not share one KV pool: every rank owns the KV planes for its own
attention heads and its own ``KVLiveSpans`` metadata, and a request is only
admitted when *every* rank can claim its share. This module resolves that
geometry from the model config and the declared spans mode, and drives the
scheduler's own ``ResourceLedger`` so a group's admission is an atomic
scheduler-owned reservation rather than a private callback loop.

Design rules (docs/QWEN38-27B-GFX1100-TP2.md):
  * the KV head partition is the weight planner's partition, not a second
    opinion - a rank's KV heads are exactly the heads its weights serve;
  * geometry comes from the backend: K and V head widths are separate config
    fields and the ``KVLiveSpans`` footprint follows the declared spans mode, so
    no equal-dimension or fixed-metadata assumption is baked in;
  * claim order is the rank order and rollback releases in reverse;
  * nothing here allocates device memory: it resolves bytes, reserves them in
    the ledger, and drives a caller-supplied allocator, so it is testable
    without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hipengine.distributed.plan import DistributedPlan, PlanError
from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVPoolPlan,
    KVPoolSpec,
    ResourceClaim,
    ResourceClaimSet,
)
from hipengine.kvcache.ledger import ResourceLedger, ResourceReservation, ResourceUnavailable

SCHEMA_VERSION = 2

#: KV plane dtype used by the production path.
DEFAULT_KV_DTYPE_BYTES = 2

#: Pool-id suffixes for the per-rank byte pools a KV claim draws on.
PLANE_POOL_SUFFIX = "kv.plane.bytes"
SPANS_POOL_SUFFIX = "kv.spans.bytes"

#: ``KVLiveSpans`` metadata is mode-specific. The dense-policy figure the
#: capacity accounting historically used - four 32-bit fields per token per
#: layer - is one declared layout among several, not a default.
DENSE_POLICY_SPANS_BYTES_PER_TOKEN_PER_LAYER = 16

SPANS_MODES = ("dense_policy", "uniform", "per_head_variable", "sliding_ring")


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
class KvSpansLayout:
    """Per-rank ``KVLiveSpans`` metadata footprint for one declared spans mode.

    ``KVLiveSpans`` carries ``base_offsets`` and ``live_counts`` always, and adds
    ``token_positions``/``evict_mask``/``row_positions`` per mode. The tensors are
    not all per-token: ``row_positions`` and ``request_ids`` are per sequence, and
    the fixed-page block table is per sequence per layer, so a single
    bytes-per-token figure cannot describe every mode.

    ``bytes_per_token_per_layer`` and ``bytes_per_sequence_per_layer`` are scaled
    by layers and context tokens; ``bytes_per_sequence`` is added once per
    sequence. ``per_sequence_count`` is how many concurrent sequences the rank
    reserves metadata for.
    """

    mode: str
    bytes_per_token_per_layer: int
    bytes_per_sequence_per_layer: int = 0
    bytes_per_sequence: int = 0
    per_sequence_count: int = 1

    def __post_init__(self) -> None:
        if str(self.mode) not in SPANS_MODES:
            raise KvPoolError(
                f"spans mode must be one of {SPANS_MODES}, got {self.mode!r}"
            )
        for name in (
            "bytes_per_token_per_layer",
            "bytes_per_sequence_per_layer",
            "bytes_per_sequence",
        ):
            if int(getattr(self, name)) < 0:
                raise KvPoolError(f"{name} must be non-negative")
        if int(self.per_sequence_count) <= 0:
            raise KvPoolError("per_sequence_count must be positive")

    @classmethod
    def dense_policy(cls) -> "KvSpansLayout":
        """Four 32-bit fields per token per layer, the historical accounting."""

        return cls("dense_policy", DENSE_POLICY_SPANS_BYTES_PER_TOKEN_PER_LAYER)

    @classmethod
    def paged_uniform(
        cls, *, max_pages_per_sequence: int, per_sequence_count: int = 1
    ) -> "KvSpansLayout":
        """Fixed-page spans: an int32 block table plus int32 live counts.

        ``base_offsets`` is ``[rows, layers, max_pages]`` int32 and
        ``live_counts`` is ``[rows, layers]`` int32, so the metadata scales with
        the page table, not with context tokens. ``row_positions`` is int64 per
        sequence row. ``token_positions`` and ``evict_mask`` are absent.
        """

        pages = int(max_pages_per_sequence)
        if pages <= 0:
            raise KvPoolError("max_pages_per_sequence must be positive")
        return cls(
            "uniform",
            bytes_per_token_per_layer=0,
            bytes_per_sequence_per_layer=4 * pages + 4,
            bytes_per_sequence=8,
            per_sequence_count=int(per_sequence_count),
        )

    @classmethod
    def per_head_variable(
        cls,
        *,
        kv_heads: int,
        bytes_per_token_per_layer: int = 0,
        per_sequence_count: int = 1,
    ) -> "KvSpansLayout":
        """DMS/H2O spans: int32 ``base_offsets`` and ``live_counts`` per head.

        Both tensors are ``[rows, layers, heads]`` int32, so the per-sequence
        term scales with the rank's KV head count. The optional
        ``token_positions``/``evict_mask`` shapes are not fixed by the ABI, so
        their per-token cost must be stated by the caller rather than assumed.
        """

        heads = int(kv_heads)
        if heads <= 0:
            raise KvPoolError("kv_heads must be positive")
        return cls(
            "per_head_variable",
            bytes_per_token_per_layer=int(bytes_per_token_per_layer),
            bytes_per_sequence_per_layer=8 * heads,
            per_sequence_count=int(per_sequence_count),
        )

    @classmethod
    def sliding_ring(cls, *, slots_per_sequence: int) -> "KvSpansLayout":
        """Token-granular ring spans: per-slot positions, mask, and base offset.

        ``token_positions`` is int64 and ``evict_mask`` and ``base_offsets`` are
        one entry per capacity slot, so the metadata is per token per layer;
        ``row_positions`` is one int64 scalar per sequence.
        """

        slots = int(slots_per_sequence)
        if slots <= 0:
            raise KvPoolError("slots_per_sequence must be positive")
        return cls(
            "sliding_ring",
            bytes_per_token_per_layer=8 + 1 + 4,
            bytes_per_sequence=8,
        )

    def bytes_for(self, *, layers: int, context_tokens: int) -> int:
        return (
            int(self.bytes_per_token_per_layer) * int(layers) * int(context_tokens)
            + (
                int(self.bytes_per_sequence_per_layer) * int(layers)
                + int(self.bytes_per_sequence)
            )
            * int(self.per_sequence_count)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "bytes_per_token_per_layer": int(self.bytes_per_token_per_layer),
            "bytes_per_sequence_per_layer": int(self.bytes_per_sequence_per_layer),
            "bytes_per_sequence": int(self.bytes_per_sequence),
            "per_sequence_count": int(self.per_sequence_count),
        }


@dataclass(frozen=True)
class KvPoolGeometry:
    """Per-rank KV pool geometry for one TP group and one context length."""

    full_attention_layers: int
    kv_heads_per_rank: tuple[int, ...]
    key_head_dim: int
    context_tokens: int
    value_head_dim: int | None = None
    kv_dtype_bytes: int = DEFAULT_KV_DTYPE_BYTES
    spans: KvSpansLayout = KvSpansLayout("dense_policy", DENSE_POLICY_SPANS_BYTES_PER_TOKEN_PER_LAYER)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.full_attention_layers) <= 0:
            raise KvPoolError("a TP group needs at least one full-attention layer")
        if not self.kv_heads_per_rank:
            raise KvPoolError("kv_heads_per_rank must cover every rank")
        if any(int(heads) <= 0 for heads in self.kv_heads_per_rank):
            raise KvPoolError("every rank must own at least one KV head")
        if int(self.key_head_dim) <= 0:
            raise KvPoolError("key_head_dim must be positive")
        if self.value_head_dim is not None and int(self.value_head_dim) <= 0:
            raise KvPoolError("value_head_dim must be positive when set")
        if int(self.context_tokens) <= 0:
            raise KvPoolError("context_tokens must be positive")
        if int(self.kv_dtype_bytes) <= 0:
            raise KvPoolError("kv_dtype_bytes must be positive")
        if int(self.schema_version) != SCHEMA_VERSION:
            raise KvPoolError(f"unsupported KV geometry schema version {self.schema_version}")

    @property
    def world_size(self) -> int:
        return len(self.kv_heads_per_rank)

    @property
    def head_dim(self) -> int:
        """Key head width; equal to the value width unless one is stated."""

        return int(self.key_head_dim)

    @property
    def value_dim(self) -> int:
        return int(self.key_head_dim if self.value_head_dim is None else self.value_head_dim)

    @property
    def value_dim_assumed_equal(self) -> bool:
        """True when the value width was inferred rather than declared."""

        return self.value_head_dim is None

    def bytes_per_token(self, rank: int) -> int:
        """K and V plane bytes this rank reads/writes per token."""

        heads = int(self.kv_heads_per_rank[int(rank)])
        widths = int(self.key_head_dim) + self.value_dim
        return int(self.full_attention_layers) * heads * widths * int(self.kv_dtype_bytes)

    def plane_bytes(self, rank: int) -> int:
        return self.bytes_per_token(rank) * int(self.context_tokens)

    def spans_bytes(self, rank: int) -> int:
        del rank  # the declared layout is the same on every rank
        return self.spans.bytes_for(
            layers=int(self.full_attention_layers), context_tokens=int(self.context_tokens)
        )

    def total_bytes(self, rank: int) -> int:
        return self.plane_bytes(rank) + self.spans_bytes(rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_attention_layers": int(self.full_attention_layers),
            "kv_heads_per_rank": [int(heads) for heads in self.kv_heads_per_rank],
            "key_head_dim": int(self.key_head_dim),
            "value_head_dim": self.value_dim,
            "value_head_dim_assumed_equal": bool(self.value_dim_assumed_equal),
            "context_tokens": int(self.context_tokens),
            "kv_dtype_bytes": int(self.kv_dtype_bytes),
            "spans": self.spans.to_dict(),
            "schema_version": int(self.schema_version),
            "plane_bytes_per_rank": [self.plane_bytes(rank) for rank in range(self.world_size)],
            "spans_bytes_per_rank": [self.spans_bytes(rank) for rank in range(self.world_size)],
            "total_bytes_per_rank": [self.total_bytes(rank) for rank in range(self.world_size)],
        }


def resolve_kv_geometry(
    config: Any,
    *,
    world_size: int,
    context_tokens: int,
    spans: KvSpansLayout,
    kv_dtype_bytes: int = DEFAULT_KV_DTYPE_BYTES,
) -> KvPoolGeometry:
    """Resolve KV geometry for ``world_size`` ranks from a model config.

    K and V head widths are separate config fields (``key_length`` and
    ``value_length``). A config that omits ``value_length`` gets equal widths and
    the geometry records that the value width was inferred rather than declared.
    The spans footprint must be stated by the caller, because the metadata is
    mode-specific and no single default describes every policy.
    """

    head_count_kv = int(getattr(config, "head_count_kv", 0) or 0)
    key_head_dim = int(getattr(config, "key_length", 0) or 0)
    raw_value = getattr(config, "value_length", None)
    value_head_dim = None if raw_value is None else int(raw_value)
    if head_count_kv <= 0:
        raise KvPoolError("model config does not declare head_count_kv")
    if key_head_dim <= 0:
        raise KvPoolError("model config does not declare key_length")
    if value_head_dim is not None and value_head_dim <= 0:
        raise KvPoolError("model config declares a non-positive value_length")
    return KvPoolGeometry(
        full_attention_layers=full_attention_layer_count(config),
        kv_heads_per_rank=local_kv_head_counts(head_count_kv, int(world_size)),
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        context_tokens=int(context_tokens),
        kv_dtype_bytes=int(kv_dtype_bytes),
        spans=spans,
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
    so the release order mirrors the acquire order. Call it *after*
    :func:`reserve_group_kv` commits, so the ledger owns the accounting and this
    function owns the device buffers.
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


@dataclass(frozen=True)
class KvRankPlan:
    """One rank's stable pool plan and the claim this group takes from it."""

    rank: int
    device: str
    plan: KVPoolPlan
    claims: ResourceClaimSet
    geometry_bytes: int

    @property
    def pool_ids(self) -> tuple[str, ...]:
        return tuple(pool.pool_id for pool in self.plan.pools)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "device": self.device,
            "backend_fingerprint": self.plan.backend_fingerprint,
            "generation": int(self.plan.generation),
            "pool_ids": list(self.pool_ids),
            "claim_id": self.claims.claim_id,
            "geometry_bytes": int(self.geometry_bytes),
            "units_by_pool": self.claims.units_by_pool(),
        }


def plane_pool_id(rank: int) -> str:
    return f"kv.plane.rank{int(rank)}.{PLANE_POOL_SUFFIX}"


def spans_pool_id(rank: int) -> str:
    return f"kv.spans.rank{int(rank)}.{SPANS_POOL_SUFFIX}"


def build_group_kv_plan(
    claim_set: KvClaimSet,
    *,
    backend_fingerprint: str,
    generation: int,
    device_budget_bytes: Sequence[int] | None = None,
) -> tuple[KvRankPlan, ...]:
    """Declare each rank's stable KV pools and the claim this group takes.

    The pools are the rank's KV byte budgets, so a second group cannot also
    claim the same region: the ledger's capacity check refuses it. Claims use
    ``LEASE`` lifetime because a TP group holds its KV for the life of the
    request set, and ``claim_all`` releases the buffers when the group ends.
    """

    if int(generation) <= 0:
        raise KvPoolError("pool generation must be positive")
    if device_budget_bytes is not None and len(device_budget_bytes) != claim_set.world_size:
        raise KvPoolError("device_budget_bytes must cover every rank")
    plans: list[KvRankPlan] = []
    for claim in claim_set.claims:
        rank = int(claim.rank)
        plane_capacity = int(claim.plane_bytes)
        spans_capacity = max(int(claim.spans_bytes), 1)
        if device_budget_bytes is not None:
            budget = int(device_budget_bytes[rank])
            if claim.total_bytes > budget:
                raise KvPoolError(
                    f"rank {rank} needs {claim.total_bytes} bytes for its KV share "
                    f"but its device budget is {budget}"
                )
            plane_capacity = min(plane_capacity, budget)
            spans_capacity = max(min(spans_capacity, budget), 1)
        pools = (
            KVPoolSpec(
                pool_id=plane_pool_id(rank),
                capacity=plane_capacity,
                unit="bytes",
                plane_role="k_payload",
                lifetimes=(ClaimLifetime.LEASE,),
            ),
            KVPoolSpec(
                pool_id=spans_pool_id(rank),
                capacity=spans_capacity,
                unit="bytes",
                plane_role="metadata",
                lifetimes=(ClaimLifetime.LEASE,),
            ),
        )
        plan = KVPoolPlan(
            backend_fingerprint=str(backend_fingerprint),
            generation=int(generation),
            pools=pools,
        )
        changes = []
        if int(claim.plane_bytes) > 0:
            changes.append(
                ResourceClaim(plane_pool_id(rank), int(claim.plane_bytes), ClaimLifetime.LEASE)
            )
        if int(claim.spans_bytes) > 0:
            changes.append(
                ResourceClaim(spans_pool_id(rank), int(claim.spans_bytes), ClaimLifetime.LEASE)
            )
        plans.append(
            KvRankPlan(
                rank=rank,
                device=str(claim.device),
                plan=plan,
                claims=ResourceClaimSet(
                    claim_id=f"tp2-kv:rank{rank}:generation{int(generation)}",
                    claims=tuple(changes),
                    metadata=(
                        ("rank", rank),
                        ("spans_mode", str(claim_set.geometry.spans.mode)),
                    ),
                ),
                geometry_bytes=int(claim.total_bytes),
            )
        )
    return tuple(plans)


@dataclass
class KvGroupReservation:
    """A provisional reservation of every rank's KV share, held by the ledgers.

    The group is admitted only when every rank reserves successfully. Commit and
    rollback both apply to all ranks, so no rank can be left holding a
    reservation the others never took.
    """

    rank_plans: tuple[KvRankPlan, ...]
    ledgers: tuple[ResourceLedger, ...]
    reservations: tuple[ResourceReservation, ...]
    _settled: str | None = None

    @property
    def world_size(self) -> int:
        return len(self.rank_plans)

    @property
    def owner_id(self) -> str:
        return f"tp2-kv-group:{self.rank_plans[0].plan.backend_fingerprint}:{self.rank_plans[0].plan.generation}"

    @property
    def settled(self) -> str | None:
        return self._settled

    def commit(self) -> "KvGroupReservation":
        if self._settled is not None:
            raise KvPoolError(f"KV group reservation is already {self._settled}")
        owner = self.owner_id
        for index, (ledger, reservation) in enumerate(zip(self.ledgers, self.reservations)):
            try:
                ledger.commit(reservation, owner_id=f"{owner}:rank{index}")
            except Exception as error:  # noqa: BLE001 - aggregate failure
                self.rollback()
                raise KvPoolError(
                    f"KV group commit failed on rank {index}: {error!r}"
                ) from error
        self._settled = "committed"
        return self

    def rollback(self) -> "KvGroupReservation":
        if self._settled == "committed":
            raise KvPoolError("cannot roll back a committed KV group reservation")
        for ledger, reservation in zip(self.ledgers, self.reservations):
            try:
                ledger.rollback(reservation)
            except Exception:  # noqa: BLE001 - rollback must not mask the original failure
                continue
        self._settled = "rolled_back"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "owner_id": self.owner_id,
            "settled": self.settled,
            "total_bytes": sum(plan.geometry_bytes for plan in self.rank_plans),
            "ranks": [plan.to_dict() for plan in self.rank_plans],
        }


def reserve_group_kv(
    rank_plans: Sequence[KvRankPlan],
    *,
    ledgers: Sequence[ResourceLedger] | None = None,
    group_id: str | None = None,
) -> KvGroupReservation:
    """Reserve every rank's KV share in that rank's own resource ledger.

    Each rank owns its device memory, so each rank gets its own ledger over its
    own pools. The group rule is all-or-nothing: a failure on any rank rolls back
    the reservations already taken and raises, naming the rank that could not fit.

    ``ledgers`` lets the caller pass the scheduler's existing per-rank ledgers so
    a second group on the same rank is refused by the same accounting; with no
    ledgers given, fresh ones are created for this call and ownership ends with
    the reservation. ``group_id`` names the reservations for diagnostics; when it
    is omitted the ledger names them, so a repeated call is refused by pool
    capacity rather than by a reused reservation name.
    """

    if not rank_plans:
        raise KvPoolError("a KV group needs at least one rank plan")
    if ledgers is not None:
        if len(ledgers) != len(rank_plans):
            raise KvPoolError("ledgers must cover every rank plan")
        for plan, ledger in zip(rank_plans, ledgers):
            if ledger.plan.pools != plan.plan.pools:
                raise KvPoolError(
                    f"rank {plan.rank} ledger does not match the rank's pool plan"
                )
    active_ledgers: list[ResourceLedger] = []
    reservations: list[ResourceReservation] = []
    for index, plan in enumerate(rank_plans):
        ledger = (
            ResourceLedger(plan.plan)
            if ledgers is None
            else ledgers[index]
        )
        active_ledgers.append(ledger)
        try:
            reservations.append(
                ledger.reserve_provisional(
                    plan.claims,
                    reservation_id=(
                        None
                        if group_id is None
                        else f"tp2-kv:{str(group_id)}:rank{plan.rank}"
                    ),
                )
            )
        except ResourceUnavailable as error:
            for previous_ledger, previous in zip(active_ledgers, reservations):
                try:
                    previous_ledger.rollback(previous)
                except Exception:  # noqa: BLE001 - rollback must not mask the failure
                    continue
            raise KvPoolError(
                f"rank {plan.rank} cannot reserve its KV share: {error}"
            ) from error
        except Exception as error:  # noqa: BLE001 - aggregate failure
            for previous_ledger, previous in zip(active_ledgers, reservations):
                try:
                    previous_ledger.rollback(previous)
                except Exception:  # noqa: BLE001 - rollback must not mask the failure
                    continue
            raise KvPoolError(
                f"KV group reservation failed on rank {plan.rank}: {error!r}"
            ) from error
    return KvGroupReservation(
        rank_plans=tuple(rank_plans),
        ledgers=tuple(active_ledgers),
        reservations=tuple(reservations),
    )
