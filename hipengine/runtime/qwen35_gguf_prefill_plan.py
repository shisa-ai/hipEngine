"""Pure prefill scratch layout planning for the hybrid GGUF family.

``_GGUFFullAttentionPrefillScratch.allocate`` needs one answer to "which device
owners does this prefill scratch take, and where does each field point?". This
module computes that from explicit geometry and already-resolved policies, with
no runner, environment lookup, device allocation or host array.

``owner_sizes`` is ordered exactly as the allocator calls ``malloc``: bulk
owners, then dedicated metadata owners, then the optional KV caches, so
``total_nbytes`` is the exact footprint. ``views`` maps every field to
``(owner, relative offset, nbytes)``; ``None`` means the field has no storage:
disabled by the route policy, zero-sized like an unpriced ``q6_f16_*`` plane, or
an unallocated ``key_cache``/``value_cache``. Disabled and inactive planes keep
their computed size in ``field_sizes`` but are absent from ``owner_sizes``.

``disabled_fields=None`` selects the dedicated one-owner-per-field layout; a
frozenset selects the lifetime-aliased arena layout together with the offsets,
lifetimes, groups, in-place aliases and subranges that describe it. Callers
resolve backend capability and the GDN route outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from hipengine.core.dtype import DType

__all__ = [
    "Qwen35GGUFPrefillScratchPlan",
    "ScratchLifetime",
    "ScratchLifetimes",
    "ScratchSubrange",
    "build_prefill_scratch_plan",
    "plan_prefill_scratch_liveness_arena",
    "plan_prefill_scratch_liveness_offsets",
    "plan_prefill_scratch_liveness_owner_slots",
    "_GGUF_PREFILL_SCRATCH_ARENA_GROUPINGS",
    "_GGUF_PREFILL_SCRATCH_COLOR_BYTES",
    "_GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS",
    "_GGUF_PREFILL_SCRATCH_DENSE_DISABLED_FIELDS",
    "_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES",
    "_GGUF_PREFILL_SCRATCH_LIFETIMES",
    "_GGUF_PREFILL_SCRATCH_OPTIONAL_FIELDS",
    "_GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS",
    "_plan_prefill_scratch_liveness_arena",
    "_plan_prefill_scratch_liveness_offsets",
    "_plan_prefill_scratch_liveness_owner_slots",
    "_prefill_scratch_allocations_conflict",
    "_prefill_scratch_field_subranges",
    "_prefill_scratch_lifetimes_overlap",
]

ScratchLifetime = tuple[str, int, int]
ScratchLifetimes = tuple[ScratchLifetime, ...]
ScratchSubrange = tuple[int, int, ScratchLifetimes]
ScratchSubranges = tuple[ScratchSubrange, ...]

# Byte constants of the prefill scratch layout. ``_PREFILL_SCRATCH_ALIGNMENT``,
# ``_PREFILL_SCRATCH_Q8_1_BLOCK`` and ``_PREFILL_SCRATCH_Q8_1_BLOCK_BYTES``
# mirror ``qwen35_gguf_planning``; they are repeated here so this module stays
# free of that module's KV/device import chain.
_PREFILL_SCRATCH_ALIGNMENT = 256
_PREFILL_SCRATCH_BLOCK_SIZE = 256
_PREFILL_SCRATCH_Q8_1_BLOCK = 32
_PREFILL_SCRATCH_Q8_1_BLOCK_BYTES = 36
_PREFILL_SCRATCH_FULL_ATTN_SPLIT_BATCH_ROWS = 16
_PREFILL_SCRATCH_FULL_ATTN_SPLIT_QUERY_ROWS = 1
_PREFILL_SCRATCH_WMMA_ROW_PAD = 16
_PREFILL_SCRATCH_Q6_F16_FIELDS = ("q6_f16_x", "q6_f16_weight", "q6_f16_out")

_PREFILL_SCRATCH_KV_FIELD_ORDER = ("key_cache", "value_cache")
# Fields typed ``object | None`` on the scratch: an absent view means ``None``,
# not the shared empty buffer the other unstored planes use.
_GGUF_PREFILL_SCRATCH_OPTIONAL_FIELDS = frozenset(_PREFILL_SCRATCH_KV_FIELD_ORDER)


def _align_prefill_scratch(value: int, alignment: int = _PREFILL_SCRATCH_ALIGNMENT) -> int:
    """Round ``value`` up to the scratch alignment."""

    return (int(value) + alignment - 1) // alignment * alignment


_GGUF_PREFILL_SCRATCH_COLOR_BYTES = 64 * 1024
_GGUF_PREFILL_SCRATCH_ARENA_GROUPINGS = frozenset({"single", "owner_slots"})

_GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS = frozenset(
    {
        "linear_z_f32",
        "linear_alpha_f32",
        "linear_beta_f32",
        "prefill_query",
        "prefill_key",
        "prefill_value",
        "linear_conv_state_tmp",
        "linear_recurrent_state_tmp",
        "post_norm_f32",
        "ffn_intermediate_f32",
        "moe_down_out_f32",
        "moe_shared_out_f32",
    }
)
_GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS = frozenset(
    _GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS
    - {"prefill_query", "prefill_key", "prefill_value"}
)


def _both_prefill_routes(start: int, end: int) -> ScratchLifetimes:
    return (("linear", start, end), ("full", start, end))


# Logical per-layer lifetimes for the production Qwen3.6 MoE bulk-prefill route.
# Linear- and full-attention temporaries live in mutually-exclusive routes;
# attention outputs hand off to one common post-attention MoE chain. Intervals
# are conservative unions across the compact-WMMA and unfused MoE fallbacks.
_GGUF_PREFILL_SCRATCH_LIFETIMES: Mapping[str, ScratchLifetimes] = MappingProxyType(
    {
        "norm": _both_prefill_routes(0, 1),
        "linear_qkv": (("linear", 0, 2),),
        "linear_qkv_f32": (("linear", 1, 3),),
        "linear_z": (("linear", 0, 5),),
        "linear_alpha": (("linear", 0, 4),),
        "linear_beta": (("linear", 0, 4),),
        # The convolution sequence is read throughout the recurrence, so
        # conv_out stays live to stage 5 for every GDN mode.
        "conv_out": (("linear", 2, 5),),
        # Normalized peer GDN materializes Q/K/V after convolution and keeps
        # them live only through the recurrent output handoff.
        "prefill_query": (("linear", 3, 5),),
        "prefill_key": (("linear", 3, 5),),
        "prefill_value": (("linear", 3, 5),),
        # Optional changed-arithmetic Q4/Q5/Q6 routes consume these during
        # initial projections, recurrent output, and dense FFN down. Production
        # lifetimes let their transient planes reuse stage-disjoint scratch
        # instead of growing a persistent weight sidecar.
        "q6_f16_x": (
            _both_prefill_routes(0, 1)
            + (("full", 5, 6),)
            + _both_prefill_routes(12, 13)
        ),
        "q6_f16_weight": (
            _both_prefill_routes(0, 1)
            + _both_prefill_routes(5, 6)
            + _both_prefill_routes(12, 13)
        ),
        "q6_f16_out": (
            _both_prefill_routes(0, 1)
            + _both_prefill_routes(5, 6)
            + _both_prefill_routes(12, 13)
        ),
        "prefill_beta": (("linear", 3, 5),),
        "prefill_decay": (("linear", 3, 5),),
        "prefill_query_scale": (("linear", 3, 5),),
        "prefill_key_scale": (("linear", 3, 5),),
        "recurrent_out": (("linear", 4, 5),),
        "recurrent_bf16": (("linear", 5, 6),),
        "full_q": (("full", 0, 2),),
        "full_k": (("full", 0, 3),),
        "full_v": (("full", 0, 4),),
        "full_query_raw": (("full", 1, 3),),
        "full_key_raw": (("full", 1, 3),),
        "full_query": (("full", 2, 4),),
        "full_key": (("full", 2, 4),),
        "full_query_bf16": (("full", 3, 5),),
        "full_gate": (("full", 1, 5),),
        "full_attn_bf16": (("full", 4, 5),),
        "full_gated": (("full", 5, 6),),
        "attn_out": _both_prefill_routes(5, 7),
        "post_norm": _both_prefill_routes(7, 15),
        "residual": _both_prefill_routes(7, 17),
        "ffn_gate_up": _both_prefill_routes(9, 15),
        "ffn_intermediate": _both_prefill_routes(10, 12),
        "ffn_down": _both_prefill_routes(12, 17),
        # q8_1 is optional in the current default but remains provisioned for
        # replacement layouts. Its union covers dense and selected routes.
        "moe_q8_1": _both_prefill_routes(0, 17),
        "moe_router_logits": _both_prefill_routes(7, 8),
        "moe_shared_gate_logits": _both_prefill_routes(7, 17),
        "moe_selected_experts": _both_prefill_routes(7, 12),
        "moe_routing_weights": _both_prefill_routes(7, 17),
        "moe_down_out": _both_prefill_routes(8, 17),
        "moe_group_counts": _both_prefill_routes(8, 12),
        "moe_padded_counts": _both_prefill_routes(8, 12),
        "moe_scatter_offsets": _both_prefill_routes(8, 12),
        "moe_expert_start_compact": _both_prefill_routes(8, 12),
        "moe_expert_start_wmma": _both_prefill_routes(8, 12),
        "moe_total_compact": _both_prefill_routes(8, 12),
        "moe_wmma_total": _both_prefill_routes(8, 12),
        "moe_tile_expert": _both_prefill_routes(8, 12),
        "moe_sorted_lanes": _both_prefill_routes(8, 13),
        "moe_sorted_experts": _both_prefill_routes(8, 13),
        "moe_sorted_weights": _both_prefill_routes(8, 13),
        "moe_lane_to_row": _both_prefill_routes(8, 13),
        "moe_shared_gate": _both_prefill_routes(13, 15),
        "moe_shared_up": _both_prefill_routes(13, 15),
        "moe_shared_intermediate": _both_prefill_routes(14, 16),
        "moe_shared_out": _both_prefill_routes(15, 17),
    }
)
_GGUF_PREFILL_SCRATCH_DENSE_DISABLED_FIELDS = frozenset(
    name for name in _GGUF_PREFILL_SCRATCH_LIFETIMES if name.startswith("moe_")
)
_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES: Mapping[str, ScratchLifetimes] = MappingProxyType(
    {
        **_GGUF_PREFILL_SCRATCH_LIFETIMES,
        # Dense gate/up is dead after SiLU. Dense down reads all intermediate
        # columns through stage 13, so only intermediate/output stay disjoint.
        "ffn_gate_up": _both_prefill_routes(9, 12),
        "ffn_intermediate": _both_prefill_routes(10, 13),
    }
)


def _prefill_scratch_lifetimes_overlap(
    lhs: ScratchLifetimes,
    rhs: ScratchLifetimes,
) -> bool:
    """True when two lifetime sets are live at the same route stage."""

    return any(
        lhs_route == rhs_route and lhs_start < rhs_end and rhs_start < lhs_end
        for lhs_route, lhs_start, lhs_end in lhs
        for rhs_route, rhs_start, rhs_end in rhs
    )


def _prefill_scratch_field_subranges(
    name: str,
    size: int,
    *,
    lifetimes: Mapping[str, ScratchLifetimes],
    allocation_subranges: Mapping[str, ScratchSubranges],
) -> ScratchSubranges:
    """Resolve one field's allocatable subranges, defaulting to its whole plane."""

    subranges = allocation_subranges.get(name)
    if subranges is None:
        return ((0, int(size), lifetimes[name]),)
    for relative_offset, subrange_size, _subrange_lifetimes in subranges:
        if (
            int(relative_offset) < 0
            or int(subrange_size) <= 0
            or int(relative_offset) + int(subrange_size) > int(size)
        ):
            raise ValueError(f"invalid bulk-prefill allocation subrange for {name}")
    return subranges


def _prefill_scratch_allocations_conflict(
    name_a: str,
    offset_a: int,
    size_a: int,
    name_b: str,
    offset_b: int,
    size_b: int,
    *,
    lifetimes: Mapping[str, ScratchLifetimes],
    allocation_subranges: Mapping[str, ScratchSubranges],
) -> bool:
    """True when two placed fields overlap in bytes *and* in lifetime."""

    subranges_a = _prefill_scratch_field_subranges(
        name_a,
        size_a,
        lifetimes=lifetimes,
        allocation_subranges=allocation_subranges,
    )
    subranges_b = _prefill_scratch_field_subranges(
        name_b,
        size_b,
        lifetimes=lifetimes,
        allocation_subranges=allocation_subranges,
    )
    return any(
        int(offset_a) + int(relative_a)
        < int(offset_b) + int(relative_b) + int(subrange_size_b)
        and int(offset_b) + int(relative_b)
        < int(offset_a) + int(relative_a) + int(subrange_size_a)
        and _prefill_scratch_lifetimes_overlap(lifetimes_a, lifetimes_b)
        for relative_a, subrange_size_a, lifetimes_a in subranges_a
        for relative_b, subrange_size_b, lifetimes_b in subranges_b
    )


def _plan_prefill_scratch_liveness_offsets(
    sizes: Mapping[str, int],
    order: tuple[str, ...],
    *,
    lifetimes: Mapping[str, ScratchLifetimes],
    allocation_subranges: Mapping[str, ScratchSubranges],
) -> tuple[int, Mapping[str, tuple[int, int]]]:
    """Place ``order`` greedily at the lowest color-separated legal offset."""

    offsets: dict[str, tuple[int, int]] = {}
    for name in order:
        size = int(sizes[name])
        if size <= 0:
            raise ValueError(f"bulk-prefill liveness field {name} must be positive")
        candidates = {0}
        # Leave one 64-KiB color between simultaneously-live subranges to
        # avoid the measured RDNA3 L2/TLB collision from exact power-of-two
        # weight/activation separation.
        current_subranges = _prefill_scratch_field_subranges(
            name,
            size,
            lifetimes=lifetimes,
            allocation_subranges=allocation_subranges,
        )
        for other, (other_offset, other_size) in offsets.items():
            other_subranges = _prefill_scratch_field_subranges(
                other,
                other_size,
                lifetimes=lifetimes,
                allocation_subranges=allocation_subranges,
            )
            candidates.update(
                _align_prefill_scratch(
                    other_offset
                    + other_relative
                    + other_subrange_size
                    + _GGUF_PREFILL_SCRATCH_COLOR_BYTES
                    - current_relative
                )
                for current_relative, _current_size, _current_lifetimes in current_subranges
                for other_relative, other_subrange_size, _other_lifetimes in other_subranges
                if (
                    other_offset
                    + other_relative
                    + other_subrange_size
                    + _GGUF_PREFILL_SCRATCH_COLOR_BYTES
                    >= current_relative
                )
            )
        for candidate in sorted(candidates):
            if not any(
                _prefill_scratch_allocations_conflict(
                    name,
                    candidate,
                    size,
                    other,
                    other_offset,
                    other_size,
                    lifetimes=lifetimes,
                    allocation_subranges=allocation_subranges,
                )
                for other, (other_offset, other_size) in offsets.items()
            ):
                offsets[name] = (candidate, size)
                break
        else:  # pragma: no cover - candidate set always includes the current end
            raise RuntimeError(f"failed to place bulk-prefill liveness field {name}")
    arena_nbytes = _align_prefill_scratch(
        max(offset + size for offset, size in offsets.values())
    )
    return arena_nbytes, MappingProxyType(offsets)


def _plan_prefill_scratch_liveness_arena(
    sizes: Mapping[str, int],
    *,
    lifetimes: Mapping[str, ScratchLifetimes] = _GGUF_PREFILL_SCRATCH_LIFETIMES,
    allocation_subranges: Mapping[str, ScratchSubranges] = MappingProxyType({}),
    priority_min_live_stages: int | None = None,
) -> tuple[int, Mapping[str, tuple[int, int]]]:
    """Size and place the single-arena layout without allocating it."""

    missing = sorted(set(sizes) - set(lifetimes))
    if missing:
        raise ValueError(f"bulk-prefill liveness plan is missing fields: {missing}")
    size_order = tuple(sorted(sizes, key=lambda name: (-int(sizes[name]), name)))
    arena_nbytes, offsets = _plan_prefill_scratch_liveness_offsets(
        sizes,
        size_order,
        lifetimes=lifetimes,
        allocation_subranges=allocation_subranges,
    )
    if priority_min_live_stages is not None:
        duration = {
            name: sum(end - start for _route, start, end in lifetimes[name])
            for name in sizes
        }
        priority = tuple(
            sorted(
                (
                    name
                    for name in sizes
                    if duration[name] >= int(priority_min_live_stages)
                ),
                key=lambda name: (-duration[name], -int(sizes[name]), name),
            )
        )
        priority_set = frozenset(priority)
        candidate_order = priority + tuple(
            name for name in size_order if name not in priority_set
        )
        candidate_nbytes, candidate_offsets = _plan_prefill_scratch_liveness_offsets(
            sizes,
            candidate_order,
            lifetimes=lifetimes,
            allocation_subranges=allocation_subranges,
        )
        # Keep established addresses on ties and never admit a larger plan.
        if candidate_nbytes < arena_nbytes:
            arena_nbytes = candidate_nbytes
            offsets = candidate_offsets
    return arena_nbytes, offsets


def _plan_prefill_scratch_liveness_owner_slots(
    sizes: Mapping[str, int],
    *,
    lifetimes: Mapping[str, ScratchLifetimes] = _GGUF_PREFILL_SCRATCH_LIFETIMES,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[int, ...],
    Mapping[str, tuple[int, int]],
    Mapping[str, str],
]:
    """Group fields into lifetime-disjoint owner slots and place them virtually."""

    missing = sorted(set(sizes) - set(lifetimes))
    if missing:
        raise ValueError(f"bulk-prefill liveness plan is missing fields: {missing}")

    slot_members: list[list[str]] = []
    slot_sizes: list[int] = []
    # Descending size order makes each slot's first field its largest, so later
    # disjoint fields can reuse the owner without resizing or address churn.
    for name, raw_size in sorted(sizes.items(), key=lambda item: (-int(item[1]), item[0])):
        size = int(raw_size)
        if size <= 0:
            raise ValueError(f"bulk-prefill liveness field {name} must be positive")
        for members in slot_members:
            if all(
                not _prefill_scratch_lifetimes_overlap(
                    lifetimes[name],
                    lifetimes[other],
                )
                for other in members
            ):
                members.append(name)
                break
        else:
            slot_members.append([name])
            slot_sizes.append(size)

    virtual_offsets: dict[str, tuple[int, int]] = {}
    field_groups: dict[str, str] = {}
    virtual_base = 0
    for slot_index, (slot_size, members) in enumerate(
        zip(slot_sizes, slot_members, strict=True)
    ):
        group = f"owner_slot_{slot_index:02d}"
        owner_nbytes = _align_prefill_scratch(slot_size)
        for name in members:
            size = int(sizes[name])
            virtual_offsets[name] = (virtual_base, size)
            field_groups[name] = group
        virtual_base = _align_prefill_scratch(virtual_base + owner_nbytes)
    return (
        tuple(tuple(members) for members in slot_members),
        tuple(_align_prefill_scratch(size) for size in slot_sizes),
        MappingProxyType(virtual_offsets),
        MappingProxyType(field_groups),
    )


def _plan_prefill_scratch_liveness_arenas(
    sizes: Mapping[str, int],
    *,
    grouping: str,
    lifetimes: Mapping[str, ScratchLifetimes] = _GGUF_PREFILL_SCRATCH_LIFETIMES,
    allocation_subranges: Mapping[str, ScratchSubranges] = MappingProxyType({}),
    priority_min_live_stages: int | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    Mapping[str, tuple[int, int]],
    Mapping[str, tuple[int, int]],
    Mapping[str, str],
    Mapping[str, str],
]:
    """Resolve arena owners, logical offsets, physical view offsets and groups.

    Logical offsets are what ``allocation_offsets`` reports (virtual cumulative
    offsets for owner slots). Physical offsets are the byte offsets a view adds
    to its owner pointer; owner-slot members all start at their slot pointer.
    """

    if grouping == "owner_slots":
        if allocation_subranges:
            raise ValueError("owner-slot prefill arenas do not support allocation subranges")
        slot_members, slot_sizes, virtual_offsets, field_groups = (
            _plan_prefill_scratch_liveness_owner_slots(sizes, lifetimes=lifetimes)
        )
        owner_names = tuple(
            f"owner_slot_{slot_index:02d}" for slot_index in range(len(slot_sizes))
        )
        field_owners = MappingProxyType(
            {
                name: owner_names[slot_index]
                for slot_index, members in enumerate(slot_members)
                for name in members
            }
        )
        physical_offsets = MappingProxyType(
            {name: (0, int(size)) for name, size in sizes.items()}
        )
        return (
            owner_names,
            slot_sizes,
            virtual_offsets,
            physical_offsets,
            field_groups,
            field_owners,
        )
    if grouping != "single":  # pragma: no cover - validated by the plan builder
        raise ValueError(f"unsupported prefill scratch arena grouping {grouping!r}")
    arena_nbytes, offsets = _plan_prefill_scratch_liveness_arena(
        sizes,
        lifetimes=lifetimes,
        allocation_subranges=allocation_subranges,
        priority_min_live_stages=priority_min_live_stages,
    )
    return (
        ("arena_bulk",),
        (arena_nbytes,),
        offsets,
        offsets,
        MappingProxyType({name: "single" for name in sizes}),
        MappingProxyType({name: "arena_bulk" for name in sizes}),
    )


# Public aliases for the pure liveness helpers.
plan_prefill_scratch_liveness_offsets = _plan_prefill_scratch_liveness_offsets
plan_prefill_scratch_liveness_arena = _plan_prefill_scratch_liveness_arena
plan_prefill_scratch_liveness_owner_slots = _plan_prefill_scratch_liveness_owner_slots


def _prefill_scratch_f16_shape_tiles(
    f16_policy: Mapping[str, object] | None,
) -> tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], ...]:
    """Normalize a resolved source-F16 policy to ``(shape, ((rows, tile), ...))``.

    The policy's bookkeeping keys (``linear_variant_intervals_by_quant``,
    ``max_rows_by_quant_shape``, ``pair_only_second_operand_policies``) describe
    kernel selection, not scratch geometry, and are skipped exactly as the
    allocator skips them.
    """

    if f16_policy is None:
        return ()
    return tuple(
        (
            (int(shape[0]), int(shape[1])),
            tuple(sorted((int(rows), int(tile)) for rows, tile in row_policy.items())),
        )
        for quant, quant_policy in f16_policy.items()
        if quant
        not in {
            "linear_variant_intervals_by_quant",
            "max_rows_by_quant_shape",
            "pair_only_second_operand_policies",
        }
        for shape, row_policy in quant_policy.items()
    )


@dataclass(frozen=True)
class Qwen35GGUFPrefillScratchPlan:
    """Immutable prefill scratch layout for one row / capacity / segment bound.

    ``owner_sizes`` is ordered exactly as the allocator calls ``malloc``:
    bulk owners, then dedicated metadata owners, then the optional KV caches.
    """

    # Explicit geometry and bounds.
    rows: int
    capacity: int
    segments: int
    hidden_size: int
    ffn_size: int
    q_width: int
    kv_width: int
    linear_qkv_width: int
    ssm_value_dim: int
    fp16_recurrent_state: bool
    allocate_kv_cache: bool
    block_size: int
    blocks: int
    max_positions: int
    block_rounded_positions: int
    # Resolved policies supplied by the caller.
    normalized_qk_heads: int
    f16_policy_applied: bool
    f16_shape_tiles: tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], ...]
    disabled_fields: frozenset[str] | None
    arena_grouping: str
    priority_min_live_stages: int | None
    # Sizes and layout.
    field_sizes: Mapping[str, int]
    dedicated_sizes: Mapping[str, int]
    owner_sizes: Mapping[str, int]
    views: Mapping[str, tuple[str, int, int] | None]
    allocation_mode: str
    allocation_offsets: Mapping[str, tuple[int, int]]
    allocation_lifetimes: Mapping[str, ScratchLifetimes]
    allocation_groups: Mapping[str, str]
    allocation_inplace_aliases: Mapping[str, str]
    allocation_subranges: Mapping[str, ScratchSubranges]
    # Split-K and MoE geometry the scratch exposes to callers.
    full_attn_split_count: int
    full_attn_split_batch_rows: int
    full_attn_split_capacity_rows: int
    full_attn_split_partial_row_bytes: int
    full_attn_split_stat_row_bytes: int
    moe_experts: int
    moe_top_k: int
    moe_selected_rows_capacity: int
    moe_wmma_rows_capacity: int

    @property
    def total_nbytes(self) -> int:
        """Exact device bytes the plan's owner list asks ``malloc`` for."""

        return int(sum(self.owner_sizes.values()))

    def storage(self, name: str) -> tuple[str, int, int]:
        """Return ``(owner, relative offset, nbytes)`` for a field with storage."""

        view = self.views[name]
        if view is None:
            raise KeyError(f"prefill scratch field {name!r} has no storage")
        return view


def build_prefill_scratch_plan(
    cfg: object,
    *,
    hidden_size: int,
    ffn_size: int,
    q_width: int,
    kv_width: int,
    linear_qkv_width: int,
    ssm_value_dim: int,
    fp16_recurrent_state: bool = False,
    rows: int,
    capacity: int | None = None,
    segments: int = 1,
    allocate_kv_cache: bool = True,
    normalized_qk_heads: int,
    f16_policy: Mapping[str, object] | None = None,
    disabled_fields: frozenset[str] | None = None,
    arena_grouping: str = "single",
    priority_min_live_stages: int | None = None,
) -> Qwen35GGUFPrefillScratchPlan:
    """Build the immutable prefill scratch layout plan.

    ``disabled_fields=None`` selects the dedicated (one owner per field) layout;
    a ``frozenset`` - even an empty one - selects the liveness-aliased layout
    with exactly those bulk fields removed. ``arena_grouping``,
    ``normalized_qk_heads``, ``f16_policy`` and ``priority_min_live_stages`` are
    resolved by the caller from backend capability, GDN route and model
    identity; this function only applies them.
    """

    if cfg is None:
        raise ValueError("prefill scratch plan requires a model configuration")
    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    segments = int(segments)
    if segments <= 0:
        raise ValueError("segments must be positive")
    capacity = rows if capacity is None else int(capacity)
    if capacity < rows:
        raise ValueError(f"capacity {capacity} must be >= rows {rows}")
    grouping = str(arena_grouping).strip().lower()
    if grouping not in _GGUF_PREFILL_SCRATCH_ARENA_GROUPINGS:
        choices = ", ".join(sorted(_GGUF_PREFILL_SCRATCH_ARENA_GROUPINGS))
        raise ValueError(f"prefill scratch arena grouping must be one of: {choices}")

    hidden_size = int(hidden_size)
    ffn_size = int(ffn_size)
    q_width = int(q_width)
    kv_width = int(kv_width)
    linear_qkv_width = int(linear_qkv_width)
    ssm_value_dim = int(ssm_value_dim)
    normalized_qk_heads = int(normalized_qk_heads)
    fp16_recurrent_state = bool(fp16_recurrent_state)
    allocate_kv_cache = bool(allocate_kv_cache)
    disabled = frozenset(disabled_fields) if disabled_fields is not None else None

    is_moe = bool(cfg.is_moe)
    block_size = _PREFILL_SCRATCH_BLOCK_SIZE
    blocks = (capacity + block_size - 1) // block_size
    # ``max_positions`` is the declared capacity, exactly as the scratch reports
    # it; only the KV cache uses the block-rounded position count.
    max_positions = capacity
    block_rounded_positions = blocks * block_size

    hidden_bytes = rows * hidden_size * DType.BF16.itemsize
    hidden_f32_bytes = rows * hidden_size * DType.FP32.itemsize
    q_proj_bytes = rows * 2 * q_width * DType.BF16.itemsize
    kv_bf16_bytes = rows * kv_width * DType.BF16.itemsize
    q_f32_bytes = rows * q_width * DType.FP32.itemsize
    kv_f32_bytes = rows * kv_width * DType.FP32.itemsize
    ffn_bytes = rows * ffn_size * DType.BF16.itemsize
    moe_lane_count = max(1, int(cfg.expert_used_count)) if is_moe else 1
    moe_top_k = max(1, int(cfg.expert_used_count))
    moe_experts = max(1, int(cfg.expert_count))
    moe_selected_rows_capacity = rows * moe_top_k
    moe_wmma_rows_capacity = moe_selected_rows_capacity + (
        _PREFILL_SCRATCH_WMMA_ROW_PAD * moe_experts
    )
    moe_tile_capacity = max(
        1, (moe_wmma_rows_capacity + _PREFILL_SCRATCH_WMMA_ROW_PAD - 1) // _PREFILL_SCRATCH_WMMA_ROW_PAD
    )
    moe_shared_ffn = max(
        1, int(cfg.expert_shared_feed_forward_length or ffn_size or 1)
    )
    q8_1_gate_blocks = rows * (
        (hidden_size + _PREFILL_SCRATCH_Q8_1_BLOCK - 1) // _PREFILL_SCRATCH_Q8_1_BLOCK
    )
    q8_1_down_blocks = moe_selected_rows_capacity * (
        (ffn_size + _PREFILL_SCRATCH_Q8_1_BLOCK - 1) // _PREFILL_SCRATCH_Q8_1_BLOCK
    )
    q8_1_moe_bytes = (
        max(q8_1_gate_blocks, q8_1_down_blocks) * _PREFILL_SCRATCH_Q8_1_BLOCK_BYTES
    )
    linear_qkv_bf16_bytes = rows * linear_qkv_width * DType.BF16.itemsize
    linear_qkv_f32_bytes = rows * linear_qkv_width * DType.FP32.itemsize
    linear_z_bytes = rows * cfg.ssm_inner_size * DType.BF16.itemsize
    linear_z_f32_bytes = rows * cfg.ssm_inner_size * DType.FP32.itemsize
    linear_ab_bytes = rows * cfg.ssm_time_step_rank * DType.BF16.itemsize
    linear_ab_f32_bytes = rows * cfg.ssm_time_step_rank * DType.FP32.itemsize
    recurrent_f32_bytes = rows * cfg.ssm_inner_size * DType.FP32.itemsize
    normalized_qk_f32_bytes = (
        rows
        * normalized_qk_heads
        * cfg.ssm_state_size
        * DType.FP32.itemsize
    )
    f16_shape_tiles = _prefill_scratch_f16_shape_tiles(f16_policy)
    q6_f16_x_bytes = (
        rows * hidden_size * DType.FP16.itemsize if f16_policy is not None else 0
    )
    q6_f16_weight_bytes = (
        max(
            int(tile) * int(shape[0])
            for shape, row_tiles in f16_shape_tiles
            for _rows, tile in row_tiles
        )
        * DType.FP16.itemsize
        if f16_shape_tiles
        else 0
    )
    q6_f16_out_bytes = (
        rows
        * max(
            int(tile)
            for _shape, row_tiles in f16_shape_tiles
            for _rows, tile in row_tiles
        )
        * DType.FP16.itemsize
        if f16_shape_tiles
        else 0
    )
    conv_state_bytes = (
        linear_qkv_width * cfg.ssm_conv_kernel * DType.FP32.itemsize
    )
    recurrent_state_bytes = (
        cfg.ssm_time_step_rank
        * cfg.ssm_state_size
        * ssm_value_dim
        * (
            DType.FP16.itemsize
            if fp16_recurrent_state
            else DType.FP32.itemsize
        )
    )
    prefill_scalar_bytes = rows * cfg.ssm_time_step_rank * DType.FP32.itemsize
    cache_nbytes = (
        block_rounded_positions
        * cfg.head_count_kv
        * cfg.key_length
        * DType.BF16.itemsize
        if allocate_kv_cache
        else 0
    )
    full_attn_split_count = (capacity + block_size - 1) // block_size
    full_attn_split_batch_rows = min(
        rows, _PREFILL_SCRATCH_FULL_ATTN_SPLIT_BATCH_ROWS
    )
    full_attn_split_query_rows = _PREFILL_SCRATCH_FULL_ATTN_SPLIT_QUERY_ROWS
    full_attn_split_partial_bytes = (
        full_attn_split_query_rows
        * q_width
        * full_attn_split_count
        * DType.FP32.itemsize
    )
    full_attn_split_stat_bytes = (
        full_attn_split_query_rows
        * cfg.head_count
        * full_attn_split_count
        * DType.FP32.itemsize
    )
    block_table_bytes = rows * blocks * DType.INT32.itemsize
    position_bytes = rows * DType.INT64.itemsize
    # ``_rope_tables`` builds a (rows, 2 * (rotary_dim // 2)) table, so an odd
    # rotary width rounds down. Match that shape exactly: the upload uses the
    # table's own nbytes and must never overrun the host buffer.
    rope_width = 2 * (int(cfg.rope_dimension_count) // 2)
    rope_table_bytes = rows * rope_width * DType.FP32.itemsize
    cu_seqlens_bytes = (segments + 1) * DType.INT32.itemsize
    atomic_bytes = DType.INT32.itemsize

    field_sizes = {
        "norm": hidden_bytes,
        "full_q": q_proj_bytes,
        "full_k": kv_bf16_bytes,
        "full_v": kv_bf16_bytes,
        "linear_qkv": linear_qkv_bf16_bytes,
        "linear_qkv_f32": linear_qkv_f32_bytes,
        "linear_z": linear_z_bytes,
        "linear_z_f32": linear_z_f32_bytes,
        "linear_alpha": linear_ab_bytes,
        "linear_alpha_f32": linear_ab_f32_bytes,
        "linear_beta": linear_ab_bytes,
        "linear_beta_f32": linear_ab_f32_bytes,
        "conv_out": linear_qkv_f32_bytes,
        "prefill_query": normalized_qk_f32_bytes,
        "prefill_key": normalized_qk_f32_bytes,
        "prefill_value": recurrent_f32_bytes,
        "q6_f16_x": q6_f16_x_bytes,
        "q6_f16_weight": q6_f16_weight_bytes,
        "q6_f16_out": q6_f16_out_bytes,
        "prefill_beta": prefill_scalar_bytes,
        "prefill_decay": prefill_scalar_bytes,
        "prefill_query_scale": prefill_scalar_bytes,
        "prefill_key_scale": prefill_scalar_bytes,
        "recurrent_out": recurrent_f32_bytes,
        "recurrent_bf16": linear_z_bytes,
        "linear_conv_state_tmp": conv_state_bytes,
        "linear_recurrent_state_tmp": recurrent_state_bytes,
        "full_query_raw": q_f32_bytes,
        "full_key_raw": kv_f32_bytes,
        "full_query": q_f32_bytes,
        "full_key": kv_f32_bytes,
        "full_query_bf16": rows * q_width * DType.BF16.itemsize,
        "full_gate": rows * q_width * DType.BF16.itemsize,
        "full_attn_bf16": rows * q_width * DType.BF16.itemsize,
        "full_gated": rows * q_width * DType.BF16.itemsize,
        "attn_out": hidden_bytes,
        "post_norm": hidden_bytes,
        "post_norm_f32": hidden_f32_bytes,
        "residual": hidden_bytes,
        "ffn_gate_up": 2 * ffn_bytes * moe_lane_count,
        "ffn_intermediate": ffn_bytes * moe_lane_count,
        "ffn_intermediate_f32": (
            moe_selected_rows_capacity * ffn_size * DType.FP32.itemsize
        ),
        "ffn_down": hidden_bytes,
        "moe_q8_1": q8_1_moe_bytes,
        "moe_router_logits": rows * moe_experts * DType.FP32.itemsize,
        "moe_shared_gate_logits": rows * DType.FP32.itemsize,
        "moe_selected_experts": rows * moe_top_k * DType.INT64.itemsize,
        "moe_routing_weights": rows * moe_top_k * DType.FP32.itemsize,
        "moe_down_out": moe_top_k * hidden_bytes,
        "moe_down_out_f32": (
            moe_selected_rows_capacity * hidden_size * DType.FP32.itemsize
        ),
        "moe_group_counts": moe_experts * DType.INT32.itemsize,
        "moe_padded_counts": moe_experts * DType.INT32.itemsize,
        "moe_scatter_offsets": moe_experts * DType.INT32.itemsize,
        "moe_expert_start_compact": (moe_experts + 1) * DType.INT64.itemsize,
        "moe_expert_start_wmma": (moe_experts + 1) * DType.INT64.itemsize,
        "moe_total_compact": DType.INT64.itemsize,
        "moe_wmma_total": DType.INT64.itemsize,
        "moe_tile_expert": moe_tile_capacity * DType.INT64.itemsize,
        "moe_sorted_lanes": moe_selected_rows_capacity * DType.INT64.itemsize,
        "moe_sorted_experts": moe_selected_rows_capacity * DType.INT64.itemsize,
        "moe_sorted_weights": moe_selected_rows_capacity * DType.FP32.itemsize,
        "moe_lane_to_row": moe_selected_rows_capacity * DType.INT64.itemsize,
        "moe_shared_gate": rows * moe_shared_ffn * DType.BF16.itemsize,
        "moe_shared_up": rows * moe_shared_ffn * DType.BF16.itemsize,
        "moe_shared_intermediate": rows * moe_shared_ffn * DType.BF16.itemsize,
        "moe_shared_out": hidden_bytes,
        "moe_shared_out_f32": hidden_f32_bytes,
    }
    field_sizes = MappingProxyType(field_sizes)
    if disabled is not None:
        # The route policy may name planes that this geometry does not have;
        # the allocator ignores those, and so does the plan. A name that is not
        # a bulk scratch field at all is a caller error.
        unknown = sorted(disabled - set(field_sizes))
        if unknown:
            raise ValueError(f"unknown bulk-prefill scratch fields: {unknown}")
    inactive_fields = frozenset(
        name
        for name in _PREFILL_SCRATCH_Q6_F16_FIELDS
        if int(field_sizes[name]) == 0
    )

    dedicated_sizes: dict[str, int] = {
        "gdn_cu_seqlens": cu_seqlens_bytes,
        "gdn_state_indices": segments * DType.INT64.itemsize,
        "block_table": block_table_bytes,
        "positions": position_bytes,
        "context_counts": position_bytes,
        "cos_table": rope_table_bytes,
        "sin_table": rope_table_bytes,
        "cu_q": cu_seqlens_bytes,
        "cu_k": cu_seqlens_bytes,
        "softmax_lse": cfg.head_count * rows * DType.FP32.itemsize,
        "full_attn_split_partial": full_attn_split_partial_bytes,
        "full_attn_split_m": full_attn_split_stat_bytes,
        "full_attn_split_l": full_attn_split_stat_bytes,
        "atomic": atomic_bytes,
        "moe_router_counter": atomic_bytes,
    }

    owner_sizes: dict[str, int] = {}
    views: dict[str, tuple[str, int, int] | None] = {
        name: None for name in field_sizes
    }
    allocation_offsets: dict[str, tuple[int, int]] = {}
    allocation_lifetimes: dict[str, ScratchLifetimes] = {}
    allocation_groups: dict[str, str] = {}
    allocation_inplace_aliases: dict[str, str] = {}
    allocation_subranges: dict[str, ScratchSubranges] = {}

    if disabled is None:
        allocation_mode = "dedicated"
        for name, nbytes in field_sizes.items():
            if name in inactive_fields:
                continue
            size = int(nbytes)
            owner_sizes[name] = size
            views[name] = (name, 0, size)
    else:
        allocation_mode = "liveness_aliased"
        base_lifetimes = (
            _GGUF_PREFILL_SCRATCH_LIFETIMES
            if is_moe
            else _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES
        )
        # Dense SiLU consumes gate/up elementwise and may replace the dead gate
        # plane in place. Keep this intentional source/output alias separate
        # from ordinary stage-disjoint arena coloring. Owner-slot arenas cannot
        # describe subranges and retain the ordinary layout.
        inplace_aliases = (
            {"ffn_intermediate": "ffn_gate_up"}
            if not is_moe and grouping == "single"
            else {}
        )
        active_names = tuple(
            name
            for name in field_sizes
            if name not in disabled and name not in inactive_fields
        )
        active_sizes = {
            name: int(field_sizes[name])
            for name in active_names
            if name not in inplace_aliases
        }
        plan_lifetimes = dict(base_lifetimes)
        planned_subranges: dict[str, ScratchSubranges] = {}
        for alias, source in inplace_aliases.items():
            source_lifetimes = base_lifetimes[source]
            alias_lifetimes = base_lifetimes[alias]
            alias_nbytes = int(field_sizes[alias])
            source_nbytes = int(field_sizes[source])
            combined_lifetimes = tuple(
                dict.fromkeys((*source_lifetimes, *alias_lifetimes))
            )
            plan_lifetimes[source] = combined_lifetimes
            source_subranges = [(0, alias_nbytes, combined_lifetimes)]
            if source_nbytes > alias_nbytes:
                source_subranges.append(
                    (alias_nbytes, source_nbytes - alias_nbytes, source_lifetimes)
                )
            planned_subranges[source] = tuple(source_subranges)
        (
            owner_names,
            owner_nbytes,
            arena_offsets,
            physical_offsets,
            arena_groups,
            field_owners,
        ) = _plan_prefill_scratch_liveness_arenas(
            active_sizes,
            grouping=grouping,
            lifetimes=plan_lifetimes,
            allocation_subranges=planned_subranges,
            priority_min_live_stages=priority_min_live_stages,
        )
        for owner_name, nbytes in zip(owner_names, owner_nbytes, strict=True):
            owner_sizes[owner_name] = int(nbytes)
        allocation_offsets = dict(arena_offsets)
        allocation_groups = dict(arena_groups)
        for name in active_names:
            if name in inplace_aliases:
                continue
            offset, size = physical_offsets[name]
            views[name] = (field_owners[name], int(offset), int(size))
        for alias, source in inplace_aliases.items():
            alias_nbytes = int(field_sizes[alias])
            source_view = views[source]
            if source_view is None or alias_nbytes > int(source_view[2]):
                raise RuntimeError(
                    f"invalid dense prefill in-place alias {alias} -> {source}"
                )
            source_owner, source_offset, _source_nbytes = source_view
            views[alias] = (source_owner, int(source_offset), alias_nbytes)
            allocation_offsets[alias] = (int(source_offset), alias_nbytes)
            allocation_groups[alias] = allocation_groups[source]
        allocation_lifetimes = {
            name: (
                base_lifetimes[name] if name in inplace_aliases else plan_lifetimes[name]
            )
            for name in active_names
        }
        allocation_inplace_aliases = dict(inplace_aliases)
        allocation_subranges = planned_subranges

    if allocate_kv_cache:
        for name in _PREFILL_SCRATCH_KV_FIELD_ORDER:
            dedicated_sizes[name] = int(cache_nbytes)
    for name, nbytes in dedicated_sizes.items():
        size = int(nbytes)
        owner_sizes[name] = size
        views[name] = (name, 0, size)
    if not allocate_kv_cache:
        for name in _PREFILL_SCRATCH_KV_FIELD_ORDER:
            views[name] = None

    return Qwen35GGUFPrefillScratchPlan(
        rows=rows,
        capacity=capacity,
        segments=segments,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        q_width=q_width,
        kv_width=kv_width,
        linear_qkv_width=linear_qkv_width,
        ssm_value_dim=ssm_value_dim,
        fp16_recurrent_state=fp16_recurrent_state,
        allocate_kv_cache=allocate_kv_cache,
        block_size=block_size,
        blocks=blocks,
        max_positions=max_positions,
        block_rounded_positions=block_rounded_positions,
        normalized_qk_heads=normalized_qk_heads,
        f16_policy_applied=f16_policy is not None,
        f16_shape_tiles=f16_shape_tiles,
        disabled_fields=disabled,
        arena_grouping=grouping,
        priority_min_live_stages=(
            None if priority_min_live_stages is None else int(priority_min_live_stages)
        ),
        field_sizes=field_sizes,
        dedicated_sizes=MappingProxyType(dedicated_sizes),
        owner_sizes=MappingProxyType(owner_sizes),
        views=MappingProxyType(views),
        allocation_mode=allocation_mode,
        allocation_offsets=MappingProxyType(allocation_offsets),
        allocation_lifetimes=MappingProxyType(allocation_lifetimes),
        allocation_groups=MappingProxyType(allocation_groups),
        allocation_inplace_aliases=MappingProxyType(allocation_inplace_aliases),
        allocation_subranges=MappingProxyType(allocation_subranges),
        full_attn_split_count=full_attn_split_count,
        full_attn_split_batch_rows=full_attn_split_batch_rows,
        full_attn_split_capacity_rows=full_attn_split_query_rows,
        full_attn_split_partial_row_bytes=(
            q_width * full_attn_split_count * DType.FP32.itemsize
        ),
        full_attn_split_stat_row_bytes=(
            cfg.head_count * full_attn_split_count * DType.FP32.itemsize
        ),
        moe_experts=moe_experts,
        moe_top_k=moe_top_k,
        moe_selected_rows_capacity=moe_selected_rows_capacity,
        moe_wmma_rows_capacity=moe_wmma_rows_capacity,
    )
