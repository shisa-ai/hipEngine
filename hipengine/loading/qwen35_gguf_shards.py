"""Byte-preserving GGUF shard planning for the Qwen3.5-family TP architecture.

This module answers one question before any allocation happens: *given an
ordered rank list and a GGUF model, which byte ranges of which tensors does
each rank own, and what are the local shapes?*

Design rules (docs/QWEN38-27B-GFX1100-TP2.md "Weight and state ownership"):

  * Column-parallel splits cut the output axis (logical axis 0), which is
    byte-contiguous per row for GGML block layouts.
  * Row-parallel splits cut the input axis (logical axis 1). That axis is
    block-quantized, so the cut must land on a quant block boundary; each local
    row is a contiguous byte segment and the local row stride shrinks.
  * Group splits (attention heads, GDN value-head groups) cut axis 0 but only
    on complete groups. Segments are described explicitly so a fused tensor
    such as ``attn_qkv`` can split q, k and v independently.
  * Replication and single-owner tensors are declared, never inferred.
  * Quant blocks/scales are copied verbatim: no dequantize/requantize step.

The manifest is immutable, serializable, and hashed so two processes cannot
disagree about shard identity. Materialization is a separate, explicit step so
metadata-only preflight and per-device memory accounting can reject a plan
before a single byte is copied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from hipengine.quant.gguf import quant_layout

COLUMN = "column"
ROW = "row"
GROUP = "group"
REPLICATED = "replicated"
OWNER = "owner"

SPLIT_KINDS = (COLUMN, ROW, GROUP, REPLICATED, OWNER)
AXIS0_KINDS = (COLUMN, GROUP)

SHARD_SCHEMA_VERSION = 1


class ShardPlanError(ValueError):
    """Raised when a tensor layout cannot be partitioned safely."""


@dataclass(frozen=True)
class AxisSegment:
    """One contiguous axis-0 range that must be split on complete groups."""

    start: int
    stop: int
    group: int = 1

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ShardPlanError(f"invalid axis segment [{self.start}, {self.stop})")
        if self.group < 1:
            raise ShardPlanError("axis segment group must be positive")

    @property
    def length(self) -> int:
        return self.stop - self.start

    @property
    def groups(self) -> int:
        return self.length // self.group


@dataclass(frozen=True)
class TensorShardRule:
    """How one tensor is partitioned across ranks."""

    kind: str
    axis: int | None = None
    segments: tuple[AxisSegment, ...] = ()
    replicated: bool = False
    owner_rank: int = 0

    def __post_init__(self) -> None:
        if self.kind not in SPLIT_KINDS:
            raise ShardPlanError(f"unknown shard kind {self.kind!r}")
        if self.kind in AXIS0_KINDS and not self.segments:
            raise ShardPlanError(f"{self.kind} rule requires axis-0 segments")
        if self.kind == ROW and self.axis != 1:
            raise ShardPlanError("row rule must split logical axis 1")
        if self.kind == OWNER and int(self.owner_rank) < 0:
            raise ShardPlanError("owner_rank must be non-negative")


@dataclass(frozen=True)
class ShardSegment:
    """One contiguous byte range inside the source tensor payload."""

    source_offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.source_offset < 0 or self.nbytes < 0:
            raise ShardPlanError("shard segment offsets/sizes must be non-negative")


@dataclass(frozen=True)
class RowSplitLayout:
    """Compact description of a row-parallel split.

    Every source row is cut into the same block-aligned column ranges, so the
    per-row byte segments are generated on demand instead of being stored.

    ``byte_ranges`` are the physical ranges the loader reads, derived from the
    logical ``ranges`` as ``column // block_size * type_size``. They are stored
    rather than recomputed on every row iteration, so the constructor checks
    that each pair actually corresponds - a layout whose physical ranges
    disagree with its own logical ranges would materialize bytes it does not
    describe. ``block_size`` and ``type_size`` are the quant geometry that
    makes that correspondence checkable.
    """

    source_row_bytes: int
    local_row_bytes: int
    ranges: tuple[tuple[int, int], ...]
    byte_ranges: tuple[tuple[int, int], ...]
    block_size: int
    type_size: int

    def __post_init__(self) -> None:
        if self.source_row_bytes <= 0 or self.local_row_bytes <= 0:
            raise ShardPlanError("row split byte widths must be positive")
        if len(self.ranges) != len(self.byte_ranges):
            raise ShardPlanError("row split ranges and byte ranges must pair up")
        if self.block_size <= 0 or self.type_size <= 0:
            raise ShardPlanError("row split quant geometry must be positive")
        for (start, stop), (byte_start, byte_stop) in zip(self.ranges, self.byte_ranges):
            expected = (
                int(start) // self.block_size * self.type_size,
                int(stop) // self.block_size * self.type_size,
            )
            if (int(byte_start), int(byte_stop)) != expected:
                raise ShardPlanError(
                    f"row split byte range [{byte_start}, {byte_stop}) does not correspond to "
                    f"column range [{start}, {stop}) under block size {self.block_size} and "
                    f"type size {self.type_size}; expected [{expected[0]}, {expected[1]})"
                )

    def iter_segments(self, rows: int) -> Iterator[ShardSegment]:
        for row in range(int(rows)):
            base = row * self.source_row_bytes
            for byte_start, byte_stop in self.byte_ranges:
                yield ShardSegment(source_offset=base + byte_start, nbytes=byte_stop - byte_start)


@dataclass(frozen=True)
class TensorShardSlice:
    """One rank's ownership of one tensor.

    ``axis_ranges`` are the logical ranges on the split axis, in ascending
    order. Most tensors have exactly one; fused tensors such as ``attn_qkv``
    have one per segment family (q, k, v). Axis-0 splits store their byte
    segments explicitly (they are few); row splits store a compact
    :class:`RowSplitLayout` because they have one segment per row.
    """

    rank: int
    axis_ranges: tuple[tuple[int, int], ...]
    local_shape: tuple[int, ...]
    local_nbytes: int
    segments: tuple[ShardSegment, ...] = ()
    row_split: RowSplitLayout | None = None
    replicated: bool = False

    @property
    def axis_start(self) -> int:
        return int(self.axis_ranges[0][0]) if self.axis_ranges else 0

    @property
    def axis_stop(self) -> int:
        return int(self.axis_ranges[-1][1]) if self.axis_ranges else 0

    @property
    def rows(self) -> int:
        return int(self.local_shape[0]) if self.local_shape else 0

    def iter_segments(self) -> Iterator[ShardSegment]:
        if self.row_split is not None:
            yield from self.row_split.iter_segments(self.rows)
            return
        yield from self.segments

    def segment_bytes(self) -> int:
        if self.row_split is not None:
            return self.rows * self.row_split.local_row_bytes
        return sum(int(segment.nbytes) for segment in self.segments)


@dataclass(frozen=True)
class TensorShardPlan:
    """Full partition plan for one tensor."""

    name: str
    kind: str
    axis: int | None
    source_shape: tuple[int, ...]
    source_nbytes: int
    quant_type: str
    block_size: int
    type_size: int
    slices: tuple[TensorShardSlice, ...]
    replicated: bool = False

    @property
    def source_row_bytes(self) -> int:
        if not self.source_shape:
            return int(self.source_nbytes)
        if len(self.source_shape) == 1:
            return int(self.source_nbytes)
        return int(self.source_nbytes) // int(self.source_shape[0])

    def slice_for(self, rank: int) -> TensorShardSlice:
        if not 0 <= int(rank) < len(self.slices):
            raise ShardPlanError(f"rank {rank} outside tensor plan {self.name!r}")
        return self.slices[int(rank)]


#: Tensors whose split axis carries no head semantics, so an uneven split is a
#: geometry change rather than a wrong result. ``ffn_gate``/``ffn_up`` split
#: independent output rows; ``ffn_down`` splits the input-feature axis whose
#: partials the exchange already sums.
COUPLED_MLP_LEAVES = ("ffn_gate", "ffn_up", "ffn_down")

#: The only leaves an uneven split may name. Everything else in the model is
#: head-structured (attention, GDN) or single-owner, and splitting those
#: unevenly would be silently wrong rather than an error - so the policy
#: refuses them here instead of relying on a caller to know better.
SAFE_UNEVEN_LEAVES = COUPLED_MLP_LEAVES


@dataclass(frozen=True)
class UnevenSplitPolicy:
    """Per-rank shares for the tensors whose split axis carries no head semantics.

    Only the MLP projections qualify. Their split is *coupled*: the rank that
    owns intermediate rows ``[b, b')`` of ``ffn_gate``/``ffn_up`` is the one
    that must reduce over exactly those columns of ``ffn_down``, so all three
    tensors share one boundary. Because ``ffn_down`` requires its boundary to
    land on a quant block, the shared boundary is rounded on ``alignment``
    elements and applied to every eligible tensor of the layer - rounding each
    tensor independently would move the boundary and silently break the
    coupling.

    Attention and GDN tensors are deliberately out of scope:
    :func:`partition_groups` refuses uneven splits for head-structured axes
    because query-head ownership and KV-head loading have to correspond.
    """

    fractions: tuple[float, ...]
    leaves: tuple[str, ...] = ("ffn_gate", "ffn_up", "ffn_down")
    alignment: int = 256

    def __post_init__(self) -> None:
        if len(self.fractions) < 2:
            raise ShardPlanError("an uneven split needs at least two ranks")
        if any(float(value) <= 0.0 for value in self.fractions):
            raise ShardPlanError("every rank's share must be positive")
        total = sum(float(value) for value in self.fractions)
        if abs(total - 1.0) > 1e-9:
            raise ShardPlanError(f"shares must sum to 1.0, not {total!r}")
        if int(self.alignment) < 1:
            raise ShardPlanError("alignment must be a positive element count")
        if not self.leaves:
            raise ShardPlanError("an uneven split must name at least one tensor")
        unknown = sorted(set(self.leaves) - set(SAFE_UNEVEN_LEAVES))
        if unknown:
            raise ShardPlanError(
                f"an uneven split cannot name {unknown}: only {list(SAFE_UNEVEN_LEAVES)} "
                "split an axis without head semantics. The attention and GDN tensors "
                "are head-structured, and moving their boundary would silently "
                "break query-head ownership against KV coverage."
            )
        missing = sorted(set(COUPLED_MLP_LEAVES) - set(self.leaves))
        if missing:
            raise ShardPlanError(
                f"the MLP projections are coupled and must be named together; "
                f"{missing} is missing from {list(self.leaves)}. The rank that owns "
                "intermediate rows of ffn_gate/ffn_up must reduce over exactly those "
                "columns of ffn_down."
            )

    def applies_to(self, name: str) -> bool:
        return _tensor_leaf(name) in self.leaves

    def ranges(self, axis_length: int) -> tuple[tuple[int, int], ...]:
        """Per-rank ``[start, stop)`` boundaries on a shared axis.

        Whole ``alignment`` units are assigned by largest remainder, then every
        rank is left at least one unit, so the result always tiles the axis.
        """

        length = int(axis_length)
        unit = int(self.alignment)
        if length % unit:
            raise ShardPlanError(
                f"axis {length} is not a whole number of {unit}-element units; "
                "the coupled MLP boundary cannot be placed"
            )
        units = length // unit
        ranks = len(self.fractions)
        if units < ranks:
            raise ShardPlanError(
                f"axis {length} holds {units} units, fewer than the {ranks} ranks"
            )
        exact = [units * float(value) for value in self.fractions]
        counts = [int(value) for value in exact]
        for index in sorted(
            range(ranks), key=lambda i: exact[i] - counts[i], reverse=True
        )[: units - sum(counts)]:
            counts[index] += 1
        for index in range(ranks):
            if counts[index] < 1:
                donor = max(range(ranks), key=lambda i: counts[i])
                if counts[donor] <= 1:
                    raise ShardPlanError(
                        "the shares leave a rank no unit; no split can honor them"
                    )
                counts[donor] -= 1
                counts[index] = 1
        if sum(counts) != units:
            raise ShardPlanError("unit counts do not tile the axis")
        ranges: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            stop = start + count * unit
            ranges.append((start, stop))
            start = stop
        return tuple(ranges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractions": [float(value) for value in self.fractions],
            "leaves": list(self.leaves),
            "alignment": int(self.alignment),
        }


@dataclass(frozen=True)
class ShardManifest:
    """Immutable rank-qualified shard manifest for one model and degree."""

    model_hash: str
    world_size: int
    hidden_size: int
    tensors: tuple[TensorShardPlan, ...]
    schema_version: int = SHARD_SCHEMA_VERSION
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.world_size) < 1:
            raise ShardPlanError("world_size must be positive")
        if int(self.hidden_size) <= 0:
            raise ShardPlanError("hidden_size must be positive")
        names = [plan.name for plan in self.tensors]
        if len(set(names)) != len(names):
            raise ShardPlanError("manifest contains duplicate tensor names")
        for plan in self.tensors:
            if len(plan.slices) != int(self.world_size):
                raise ShardPlanError(
                    f"tensor {plan.name!r} has {len(plan.slices)} slices for world size {self.world_size}"
                )

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(plan.name for plan in self.tensors)

    def plan_for(self, name: str) -> TensorShardPlan:
        for plan in self.tensors:
            if plan.name == name:
                return plan
        raise ShardPlanError(f"tensor {name!r} is not in the shard manifest")

    def rank_bytes(self, rank: int) -> int:
        return sum(plan.slice_for(rank).local_nbytes for plan in self.tensors)

    def rank_summary(self, rank: int) -> dict[str, int]:
        totals = {kind: 0 for kind in SPLIT_KINDS}
        for plan in self.tensors:
            totals[plan.kind] += plan.slice_for(rank).local_nbytes
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "model_hash": self.model_hash,
            "world_size": int(self.world_size),
            "hidden_size": int(self.hidden_size),
            "notes": list(self.notes),
            "rank_bytes": [self.rank_bytes(rank) for rank in range(self.world_size)],
            "tensors": [
                {
                    "name": plan.name,
                    "kind": plan.kind,
                    "axis": plan.axis,
                    "replicated": bool(plan.replicated),
                    "source_shape": list(plan.source_shape),
                    "source_nbytes": int(plan.source_nbytes),
                    "quant_type": plan.quant_type,
                    "block_size": int(plan.block_size),
                    "type_size": int(plan.type_size),
                    "slices": [
                        {
                            "rank": slice_.rank,
                            "axis_start": slice_.axis_start,
                            "axis_stop": slice_.axis_stop,
                            "local_shape": list(slice_.local_shape),
                            "local_nbytes": int(slice_.local_nbytes),
                            "replicated": bool(slice_.replicated),
                            "axis_ranges": [list(pair) for pair in slice_.axis_ranges],
                            "local_row_bytes": (
                                None if slice_.row_split is None else slice_.row_split.local_row_bytes
                            ),
                            "source_row_bytes": (
                                None if slice_.row_split is None else slice_.row_split.source_row_bytes
                            ),
                            # The loader reads these physical ranges, so the
                            # manifest identity must cover them: two layouts with
                            # the same logical ranges and different physical
                            # ranges otherwise hash identically.
                            "byte_ranges": (
                                None
                                if slice_.row_split is None
                                else [list(pair) for pair in slice_.row_split.byte_ranges]
                            ),
                            "segments": [
                                {"source_offset": seg.source_offset, "nbytes": seg.nbytes}
                                for seg in slice_.segments
                            ],
                        }
                        for slice_ in plan.slices
                    ],
                }
                for plan in self.tensors
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Rule resolution
# ---------------------------------------------------------------------------


def _gdn_geometry(config: Any) -> tuple[int, int, int, int, int]:
    """Return ``(k_heads, head_k_dim, v_heads, head_v_dim, v_tiles)``.

    The GDN kernel addresses the fused q/k/v projection as q at
    ``head * head_k_dim`` (``ssm_group_count`` heads), k at
    ``key_width + head * head_k_dim``, and v at
    ``2 * key_width + head * head_v_dim`` (``ssm_time_step_rank`` heads), where
    ``key_width`` is ``ssm_group_count * ssm_state_size``.
    """

    k_heads = int(config.ssm_group_count)
    head_k = int(config.ssm_state_size)
    v_heads = int(config.ssm_time_step_rank)
    inner = int(config.ssm_inner_size)
    if k_heads < 1 or head_k < 1:
        raise ShardPlanError("GDN key-head geometry must be positive")
    if v_heads < 1 or inner % v_heads:
        raise ShardPlanError(f"ssm_inner_size {inner} is not divisible by ssm_time_step_rank {v_heads}")
    if v_heads % k_heads:
        raise ShardPlanError(f"ssm_time_step_rank {v_heads} is not a multiple of ssm_group_count {k_heads}")
    return k_heads, head_k, v_heads, inner // v_heads, v_heads // k_heads


def _gdn_qkv_segments(config: Any) -> tuple[AxisSegment, ...]:
    """Segments of the fused GDN q/k/v projection in its on-device layout.

    The row order is q, k, then the wider value block. In GGUF weights the value
    block is in llama.cpp's *tiled* order (``k_head = v_head % k_heads``), so it
    decomposes into one segment per tile of ``k_heads`` value heads. Splitting
    k heads contiguously therefore cuts the value block into whole per-tile runs
    rather than one contiguous range; :func:`_gdn_value_segments` is that same
    decomposition for the value-head axis of the other tensors.
    """

    k_heads, head_k, _v_heads, head_v, tiles = _gdn_geometry(config)
    key_width = k_heads * head_k
    segments = [
        AxisSegment(0, key_width, head_k),
        AxisSegment(key_width, 2 * key_width, head_k),
    ]
    base = 2 * key_width
    for tile in range(tiles):
        segments.append(
            AxisSegment(base + tile * k_heads * head_v, base + (tile + 1) * k_heads * head_v, head_v)
        )
    return tuple(segments)


def _gdn_value_segments(config: Any, *, group: int) -> tuple[AxisSegment, ...]:
    """Segments of an axis laid out in GGUF tiled value-head order.

    ``group`` is ``ssm_state_size`` for value-head axes that carry one block per
    value head, or 1 for the per-head scalar axes (alpha, beta, A_log, dt bias).
    """

    k_heads, _head_k, _v_heads, head_v, tiles = _gdn_geometry(config)
    if group == head_v:
        tile_width = k_heads * head_v
    elif group == 1:
        tile_width = k_heads
    else:
        raise ShardPlanError(f"value-axis group must be 1 or {head_v}, got {group}")
    return tuple(
        AxisSegment(tile * tile_width, (tile + 1) * tile_width, group) for tile in range(tiles)
    )


def shard_rule_for_tensor(name: str, *, config: Any) -> TensorShardRule:
    """Return the declared TP ownership rule for one Qwen3.5-family tensor."""

    if name == "token_embd.weight":
        return TensorShardRule(kind=OWNER)
    if name == "output_norm.weight":
        return TensorShardRule(kind=REPLICATED, replicated=True)
    if name == "output.weight":
        return TensorShardRule(kind=OWNER)
    if not name.startswith("blk."):
        raise ShardPlanError(f"no TP ownership rule for root tensor {name!r}")

    suffix = name.split(".", 2)[2]
    if suffix in {"attn_norm.weight", "post_attention_norm.weight"}:
        return TensorShardRule(kind=REPLICATED, replicated=True)
    if suffix in {"attn_q_norm.weight", "attn_k_norm.weight"}:
        return TensorShardRule(kind=REPLICATED, replicated=True)
    if suffix == "ssm_norm.weight":
        return TensorShardRule(kind=REPLICATED, replicated=True)

    if suffix == "attn_q.weight":
        # Rows are [q_head(head_dim), gate_head(head_dim)] interleaved, so one
        # indivisible group is 2 * key_length elements and a head split is a
        # contiguous row range (kernels/qwen35_split_qgate_bf16 contract).
        group = 2 * int(config.key_length)
        return TensorShardRule(kind=GROUP, axis=0, segments=(AxisSegment(0, 2 * int(config.head_count) * int(config.key_length), group),))
    if suffix in {"attn_k.weight", "attn_v.weight"}:
        group = int(config.key_length if suffix == "attn_k.weight" else config.value_length)
        return TensorShardRule(kind=GROUP, axis=0, segments=(AxisSegment(0, int(config.head_count_kv) * group, group),))
    if suffix == "attn_output.weight":
        head = int(config.key_length)
        return TensorShardRule(
            kind=ROW,
            axis=1,
            segments=(AxisSegment(0, int(config.head_count) * head, head),),
        )

    if suffix == "attn_gate.weight":
        # linear_z gate: one row block per value head, in tiled value-head order.
        head_v = _gdn_geometry(config)[3]
        return TensorShardRule(kind=GROUP, axis=0, segments=_gdn_value_segments(config, group=head_v))
    if suffix in {"attn_qkv.weight", "ssm_conv1d.weight"}:
        return TensorShardRule(kind=GROUP, axis=0, segments=_gdn_qkv_segments(config))
    if suffix in {"ssm_alpha.weight", "ssm_beta.weight"}:
        # One scalar per value head (ssm_time_step_rank rows).
        return TensorShardRule(kind=GROUP, axis=0, segments=_gdn_value_segments(config, group=1))
    if suffix in {"ssm_a", "ssm_dt.bias"}:
        return TensorShardRule(kind=GROUP, axis=0, segments=_gdn_value_segments(config, group=1))
    if suffix == "ssm_out.weight":
        head_v = _gdn_geometry(config)[3]
        return TensorShardRule(kind=ROW, axis=1, segments=_gdn_value_segments(config, group=head_v))

    if suffix in {"ffn_gate.weight", "ffn_up.weight"}:
        return TensorShardRule(kind=COLUMN, axis=0, segments=(AxisSegment(0, int(config.feed_forward_length), 1),))
    if suffix == "ffn_down.weight":
        return TensorShardRule(kind=ROW, axis=1)

    if suffix.startswith("nextn."):
        # MTP/NextN blocks are not part of the AR shard set (Packet 5).
        return TensorShardRule(kind=OWNER)

    raise ShardPlanError(f"no TP ownership rule for tensor {name!r}")


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def _tensor_leaf(name: str) -> str:
    """The role name of a tensor (``blk.3.ffn_gate.weight`` -> ``ffn_gate``)."""

    return str(name).split(".", 2)[-1].removesuffix(".weight")


def _validate_explicit_tiling(
    name: str,
    element_ranges: Sequence[tuple[int, int]],
    segments: Sequence[AxisSegment],
) -> None:
    """Assert that explicit per-rank ranges tile the declared segments exactly."""

    ranges = sorted((int(start), int(stop)) for start, stop in element_ranges)
    for start, stop in ranges:
        if start >= stop:
            raise ShardPlanError(f"{name}: explicit range [{start}, {stop}) is empty or inverted")
    expected = min(segment.start for segment in segments)
    for start, stop in ranges:
        if start != expected:
            raise ShardPlanError(
                f"{name}: explicit ranges must tile the axis from {expected}; "
                f"the next range starts at {start}"
            )
        expected = stop
    if expected != max(segment.stop for segment in segments):
        raise ShardPlanError(
            f"{name}: explicit ranges stop at {expected}, short of "
            f"{max(segment.stop for segment in segments)}"
        )


def partition_groups(
    segment: AxisSegment,
    world_size: int,
    *,
    rank_groups: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Split one axis segment into ``world_size`` contiguous complete groups.

    Groups divide evenly whenever the segment holds at least one group per rank.
    A segment with fewer groups than ranks (the KV-head case) is replicated so
    that every rank still owns whole groups; the caller's coverage check
    accepts that only when the replication is uniform. Uneven splits are
    rejected because they would give ranks different local geometry.

    Replication is *block* replication - consecutive ranks share a group - not
    round robin. The axis being split here is a consumer axis whose grouping is
    fixed by a sibling axis: rank ``r`` owns query heads
    ``[r * q_per_rank, (r + 1) * q_per_rank)``, and the KV heads covering those
    queries are the ones it must load. Round-robin assignment would hand rank 1
    KV head 1 while its queries attend to KV head 0, which is silent wrong
    attention rather than a refusal.
    """

    if int(world_size) < 1:
        raise ShardPlanError("world_size must be positive")
    groups = segment.groups
    if rank_groups is not None:
        # An explicit per-rank group count. Ranges stay contiguous runs, which
        # is what keeps a head split's query ownership and KV coverage
        # corresponding; only the run lengths change.
        counts = [int(value) for value in rank_groups]
        if len(counts) != int(world_size):
            raise ShardPlanError(
                f"{len(counts)} group counts for world size {world_size}"
            )
        if any(count < 1 for count in counts):
            raise ShardPlanError("every rank must own at least one whole group")
        if sum(counts) != groups:
            raise ShardPlanError(
                f"group counts {counts} do not sum to the segment's {groups} groups"
            )
        ranges = []
        start = segment.start
        for count in counts:
            stop = start + count * segment.group
            ranges.append((start, stop))
            start = stop
        return ranges
    if groups < 1:
        raise ShardPlanError(f"segment [{segment.start}, {segment.stop}) has no complete groups")
    if groups < int(world_size):
        if int(world_size) % groups:
            raise ShardPlanError(
                f"segment [{segment.start}, {segment.stop}) holds {groups} groups, which cannot be "
                f"replicated uniformly across {world_size} ranks; every rank must own whole groups "
                f"and share its group with the same number of neighbours"
            )
        ranks_per_group = int(world_size) // groups
        ranges: list[tuple[int, int]] = []
        for rank in range(int(world_size)):
            group = rank // ranks_per_group
            start = segment.start + group * segment.group
            ranges.append((start, start + segment.group))
        return ranges
    if groups % int(world_size):
        raise ShardPlanError(
            f"segment [{segment.start}, {segment.stop}) holds {groups} groups, which does not "
            f"divide evenly across {world_size} ranks"
        )
    per_rank = groups // int(world_size)
    ranges = []
    for rank in range(int(world_size)):
        start = segment.start + rank * per_rank * segment.group
        ranges.append((start, start + per_rank * segment.group))
    return ranges


def _row_split_bounds(dim: int, world_size: int, block_size: int) -> list[tuple[int, int]]:
    if dim % int(world_size) != 0:
        raise ShardPlanError(f"input dimension {dim} is not divisible by world size {world_size}")
    per_rank = dim // int(world_size)
    if per_rank % block_size != 0:
        raise ShardPlanError(
            f"row split of dimension {dim} into {world_size} ranks gives {per_rank} columns, "
            f"which is not a multiple of the {block_size}-element quant block"
        )
    return [(rank * per_rank, (rank + 1) * per_rank) for rank in range(int(world_size))]


@dataclass(frozen=True)
class GDNHeadMap:
    """Global-to-local GDN head mapping for one rank.

    GGUF value heads are in llama.cpp's tiled order, ``k_head = v_head %
    k_heads``. A rank owns a contiguous run of key heads and, for each value
    tile, the value heads of those key heads. The local value-head order is
    tile-major, which is the order of the materialized value rows/columns.
    """

    rank: int
    world_size: int
    k_heads: int
    v_heads: int
    head_k_dim: int
    head_v_dim: int
    tiles: int

    @property
    def local_k_heads(self) -> int:
        return self.k_heads // self.world_size

    @property
    def local_v_heads(self) -> int:
        return self.v_heads // self.world_size

    def k_heads_for(self) -> tuple[int, ...]:
        start = self.rank * self.local_k_heads
        return tuple(range(start, start + self.local_k_heads))

    def v_heads_for(self) -> tuple[int, ...]:
        """Global value heads in local (tile-major) order."""

        owned = self.k_heads_for()
        return tuple(tile * self.k_heads + k for tile in range(self.tiles) for k in owned)

    def local_k_head(self, local_v_head: int) -> int:
        """Global key head paired with a local value head."""

        if not 0 <= int(local_v_head) < self.local_v_heads:
            raise ShardPlanError(f"local value head {local_v_head} outside [0, {self.local_v_heads})")
        within_tile = int(local_v_head) % self.local_k_heads
        return self.rank * self.local_k_heads + within_tile


def gdn_head_map(config: Any, rank: int, world_size: int) -> GDNHeadMap:
    """Head mapping implied by the shard plan for one rank."""

    k_heads, head_k, v_heads, head_v, tiles = _gdn_geometry(config)
    if int(world_size) < 1 or k_heads % int(world_size):
        raise ShardPlanError(f"{k_heads} GDN key heads do not divide across {world_size} ranks")
    if not 0 <= int(rank) < int(world_size):
        raise ShardPlanError(f"rank {rank} outside world size {world_size}")
    return GDNHeadMap(
        rank=int(rank),
        world_size=int(world_size),
        k_heads=k_heads,
        v_heads=v_heads,
        head_k_dim=head_k,
        head_v_dim=head_v,
        tiles=tiles,
    )


def build_tensor_shard_plan(
    *,
    name: str,
    shape: Sequence[int],
    nbytes: int,
    quant_type_id: int,
    quant_type_name: str,
    rule: TensorShardRule,
    world_size: int,
    element_ranges: Sequence[tuple[int, int]] | None = None,
) -> TensorShardPlan:
    """Resolve one tensor's rule into per-rank byte segments and local shapes.

    ``element_ranges`` overrides the even split with explicit per-rank ranges
    along the split axis. It is how a coupled uneven split is placed: every
    tensor that shares the axis receives the same boundaries.
    """

    plan = _resolve_tensor_shard_plan(
        name=name,
        shape=shape,
        nbytes=nbytes,
        quant_type_id=quant_type_id,
        quant_type_name=quant_type_name,
        rule=rule,
        world_size=world_size,
        element_ranges=element_ranges,
    )
    validate_plan_coverage(plan)
    return plan


def _resolve_tensor_shard_plan(
    *,
    name: str,
    shape: Sequence[int],
    nbytes: int,
    quant_type_id: int,
    quant_type_name: str,
    rule: TensorShardRule,
    world_size: int,
    element_ranges: Sequence[tuple[int, int]] | None = None,
) -> TensorShardPlan:
    """Resolve one tensor's rule into per-rank byte segments and local shapes."""

    layout = quant_layout(int(quant_type_id))
    block_size = int(layout.block_size)
    type_size = int(layout.type_size)
    shape_tuple = tuple(int(dim) for dim in shape)
    if int(nbytes) <= 0:
        raise ShardPlanError(f"tensor {name!r} has no payload")
    if element_ranges is not None:
        if len(element_ranges) != int(world_size):
            raise ShardPlanError(
                f"{name}: {len(element_ranges)} explicit ranges for world size {world_size}"
            )
        if len(rule.segments) > 1:
            raise ShardPlanError(
                f"{name}: an explicit uneven split is only defined for single-segment rules, "
                f"but this tensor declares {len(rule.segments)} segments"
            )

    if rule.kind in AXIS0_KINDS:
        slices: list[TensorShardSlice] = []
        row_bytes = int(nbytes) // shape_tuple[0]
        for rank in range(int(world_size)):
            ranges: list[tuple[int, int]] = []
            for segment in rule.segments:
                if element_ranges is None:
                    ranges.append(partition_groups(segment, int(world_size))[rank])
                    continue
                start, stop = (int(value) for value in element_ranges[rank])
                if start < segment.start or stop > segment.stop or start >= stop:
                    raise ShardPlanError(
                        f"{name}: explicit range [{start}, {stop}) leaves segment "
                        f"[{segment.start}, {segment.stop})"
                    )
                if (start - segment.start) % segment.group or (stop - segment.start) % segment.group:
                    raise ShardPlanError(
                        f"{name}: explicit range [{start}, {stop}) is not a whole number of "
                        f"{segment.group}-element groups"
                    )
                ranges.append((start, stop))
            segments: list[ShardSegment] = []
            local_rows = 0
            for start, stop in ranges:
                segments.append(ShardSegment(source_offset=start * row_bytes, nbytes=(stop - start) * row_bytes))
                local_rows += stop - start
            local_shape = (local_rows, *shape_tuple[1:])
            slices.append(
                TensorShardSlice(
                    rank=rank,
                    axis_ranges=tuple(ranges),
                    local_shape=local_shape,
                    local_nbytes=sum(seg.nbytes for seg in segments),
                    segments=tuple(segments),
                )
            )
        if element_ranges is not None:
            _validate_explicit_tiling(name, element_ranges, rule.segments)
        return TensorShardPlan(
            name=name,
            kind=rule.kind,
            axis=0,
            source_shape=shape_tuple,
            source_nbytes=int(nbytes),
            quant_type=quant_type_name,
            block_size=block_size,
            type_size=type_size,
            slices=tuple(slices),
        )

    if rule.kind == ROW:
        if len(shape_tuple) < 2:
            raise ShardPlanError(f"tensor {name!r} has no input axis to split")
        source_row_bytes = int(nbytes) // shape_tuple[0]
        segments = rule.segments or (AxisSegment(0, shape_tuple[1], 1),)
        if element_ranges is not None:
            _validate_explicit_tiling(name, element_ranges, segments)
        # Per-rank column ranges, one list per declared segment family.
        per_rank_ranges: list[list[tuple[int, int]]] = []
        for segment in segments:
            if element_ranges is not None:
                per_rank_ranges.append(
                    [tuple(int(value) for value in element_ranges[rank]) for rank in range(int(world_size))]
                )
                continue
            if segment.length % int(world_size):
                raise ShardPlanError(
                    f"input block [{segment.start}, {segment.stop}) is not divisible by world size {world_size}"
                )
            per_rank = segment.length // int(world_size)
            if per_rank % block_size:
                raise ShardPlanError(
                    f"row split of [{segment.start}, {segment.stop}) gives {per_rank} columns per rank, "
                    f"which is not a multiple of the {block_size}-element quant block"
                )
            if segment.group > 1 and per_rank % segment.group:
                raise ShardPlanError(
                    f"row split gives {per_rank} columns per rank, which is not a whole number of "
                    f"{segment.group}-element groups"
                )
            per_rank_ranges.append(
                [
                    (segment.start + rank * per_rank, segment.start + (rank + 1) * per_rank)
                    for rank in range(int(world_size))
                ]
            )
        slices = []
        for rank in range(int(world_size)):
            ranges = tuple(entry[rank] for entry in per_rank_ranges)
            for segment, (start, stop) in zip(segments, ranges):
                if (start - segment.start) % block_size or (stop - segment.start) % block_size:
                    raise ShardPlanError(
                        f"{name}: rank {rank} range [{start}, {stop}) is not a whole number of "
                        f"{block_size}-element quant blocks"
                    )
                if segment.group > 1 and (
                    (start - segment.start) % segment.group or (stop - segment.start) % segment.group
                ):
                    raise ShardPlanError(
                        f"{name}: rank {rank} range [{start}, {stop}) is not a whole number of "
                        f"{segment.group}-element groups"
                    )
            byte_ranges = tuple(
                (start // block_size * type_size, stop // block_size * type_size) for start, stop in ranges
            )
            local_row_bytes = sum(stop - start for start, stop in byte_ranges)
            local_cols = sum(stop - start for start, stop in ranges)
            slices.append(
                TensorShardSlice(
                    rank=rank,
                    axis_ranges=ranges,
                    local_shape=(shape_tuple[0], local_cols),
                    local_nbytes=shape_tuple[0] * local_row_bytes,
                    row_split=RowSplitLayout(
                        source_row_bytes=source_row_bytes,
                        local_row_bytes=local_row_bytes,
                        ranges=ranges,
                        byte_ranges=byte_ranges,
                        block_size=block_size,
                        type_size=type_size,
                    ),
                )
            )
        return TensorShardPlan(
            name=name,
            kind=ROW,
            axis=1,
            source_shape=shape_tuple,
            source_nbytes=int(nbytes),
            quant_type=quant_type_name,
            block_size=block_size,
            type_size=type_size,
            slices=tuple(slices),
        )

    if rule.kind == REPLICATED:
        segments = (ShardSegment(source_offset=0, nbytes=int(nbytes)),)
        slices = tuple(
            TensorShardSlice(
                rank=rank,
                axis_ranges=((0, shape_tuple[0] if shape_tuple else 1),),
                local_shape=shape_tuple,
                local_nbytes=int(nbytes),
                segments=segments,
                replicated=True,
            )
            for rank in range(int(world_size))
        )
        return TensorShardPlan(
            name=name,
            kind=REPLICATED,
            axis=None,
            source_shape=shape_tuple,
            source_nbytes=int(nbytes),
            quant_type=quant_type_name,
            block_size=block_size,
            type_size=type_size,
            slices=slices,
            replicated=True,
        )

    if rule.kind == OWNER:
        owner = int(rule.owner_rank)
        if owner >= int(world_size):
            raise ShardPlanError(f"owner rank {owner} is outside world size {world_size}")
        slices = []
        for rank in range(int(world_size)):
            owned = rank == owner
            slices.append(
                TensorShardSlice(
                    rank=rank,
                    axis_ranges=((0, shape_tuple[0] if shape_tuple else 1),),
                    local_shape=shape_tuple if owned else tuple(0 for _ in shape_tuple),
                    local_nbytes=int(nbytes) if owned else 0,
                    segments=(ShardSegment(source_offset=0, nbytes=int(nbytes)),) if owned else (),
                    replicated=not owned,
                )
            )
        return TensorShardPlan(
            name=name,
            kind=OWNER,
            axis=None,
            source_shape=shape_tuple,
            source_nbytes=int(nbytes),
            quant_type=quant_type_name,
            block_size=block_size,
            type_size=type_size,
            slices=tuple(slices),
        )

    raise ShardPlanError(f"unsupported shard kind {rule.kind!r}")  # pragma: no cover


def _expected_local_nbytes(local_shape: Sequence[int], block_size: int, type_size: int) -> int:
    """Bytes implied by a declared local shape under the GGML row layout."""

    if not local_shape:
        return 0
    last = int(local_shape[-1])
    if last % int(block_size) != 0:
        raise ShardPlanError(f"local dimension {last} is not a multiple of quant block {block_size}")
    row_bytes = last // int(block_size) * int(type_size)
    rows = 1
    for dim in local_shape[:-1]:
        rows *= int(dim)
    return rows * row_bytes


def validate_plan_coverage(plan: TensorShardPlan) -> None:
    """Assert that a tensor plan tiles its source payload exactly once.

    Every non-replicated byte must be owned by exactly one rank; replicated
    tensors must appear whole on every rank; single-owner tensors must be whole
    on exactly one rank and absent elsewhere. Declared local shapes must agree
    with the byte counts.
    """

    intervals: list[tuple[int, int]] = []
    owners = 0
    for shard_slice in plan.slices:
        declared = int(shard_slice.segment_bytes())
        if declared != int(shard_slice.local_nbytes):
            raise ShardPlanError(
                f"{plan.name}: rank {shard_slice.rank} declares {shard_slice.local_nbytes} bytes "
                f"but its segments total {declared}"
            )
        if shard_slice.local_nbytes:
            expected = _expected_local_nbytes(shard_slice.local_shape, plan.block_size, plan.type_size)
            if expected != int(shard_slice.local_nbytes):
                raise ShardPlanError(
                    f"{plan.name}: rank {shard_slice.rank} local shape {tuple(shard_slice.local_shape)} "
                    f"implies {expected} bytes, descriptor says {shard_slice.local_nbytes}"
                )
        if plan.kind == REPLICATED:
            if int(shard_slice.local_nbytes) != int(plan.source_nbytes):
                raise ShardPlanError(f"{plan.name}: replicated slice on rank {shard_slice.rank} is partial")
            continue
        if plan.kind == OWNER:
            if shard_slice.local_nbytes:
                owners += 1
            elif declared:
                raise ShardPlanError(f"{plan.name}: rank {shard_slice.rank} has segments but no bytes")
            continue
        if shard_slice.row_split is not None:
            # Row splits repeat the same column ranges on every row, so check
            # the ranges once instead of expanding rows * ranges intervals.
            for byte_start, byte_stop in shard_slice.row_split.byte_ranges:
                if byte_start < 0 or byte_stop > shard_slice.row_split.source_row_bytes:
                    raise ShardPlanError(
                        f"{plan.name}: rank {shard_slice.rank} row range [{byte_start}, {byte_stop}) "
                        "falls outside the source row"
                    )
                intervals.append((byte_start, byte_stop))
            continue
        for segment in shard_slice.iter_segments():
            start = int(segment.source_offset)
            intervals.append((start, start + int(segment.nbytes)))

    if plan.kind == REPLICATED:
        return
    if plan.kind == OWNER:
        if owners != 1:
            raise ShardPlanError(f"{plan.name}: expected exactly one owning rank, found {owners}")
        owned = [s for s in plan.slices if s.local_nbytes]
        if int(owned[0].local_nbytes) != int(plan.source_nbytes):
            raise ShardPlanError(f"{plan.name}: owning rank does not hold the whole tensor")
        return

    intervals.sort()
    if not intervals or intervals[0][0] != 0:
        raise ShardPlanError(f"{plan.name}: shard coverage does not start at byte 0")
    # Sweep the coverage so uniform replication is accepted while gaps, ragged
    # partial replication, and overlaps beyond the declared multiplicity are not.
    events: list[tuple[int, int]] = []
    for start, stop in intervals:
        events.append((start, 1))
        events.append((stop, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    coverage = 0
    multiplicities: set[int] = set()
    cursor: int | None = None
    for position, delta in events:
        if cursor is not None and position > cursor:
            multiplicities.add(coverage)
        coverage += delta
        cursor = position
    if coverage != 0:
        raise ShardPlanError(f"{plan.name}: shard coverage does not close (residual {coverage})")
    expected_end = int(plan.source_nbytes)
    if plan.kind == ROW and plan.slices and plan.slices[0].row_split is not None:
        expected_end = int(plan.slices[0].row_split.source_row_bytes)
    if cursor != expected_end:
        raise ShardPlanError(
            f"{plan.name}: shard coverage ends at byte {cursor}, expected {expected_end}"
        )
    if not multiplicities or min(multiplicities) < 1:
        raise ShardPlanError(f"{plan.name}: shard coverage has a gap")
    if len(multiplicities) > 1:
        raise ShardPlanError(
            f"{plan.name}: non-uniform shard coverage {sorted(multiplicities)}; "
            "replication must apply to every byte equally"
        )


def build_shard_manifest(
    info: Any,
    *,
    world_size: int,
    model_hash: str = "",
    exclude_prefixes: Sequence[str] = (),
    owner_rank: int = 0,
    uneven_split: UnevenSplitPolicy | None = None,
) -> ShardManifest:
    """Build the full manifest for one GGUF model and TP degree.

    ``info`` is a :class:`hipengine.loading.gguf.GGUFModelInfo`. Blocks the AR
    config already classified as non-autoregressive (the MTP/NextN block, via
    ``config.ignored_block_ids``) are excluded from the AR shard set and
    recorded in ``notes``; ``exclude_prefixes`` adds further exclusions.

    ``uneven_split`` gives named tensors an explicit per-rank boundary instead
    of an even one. The boundary is computed once per split axis and applied to
    every eligible tensor of the layer, because the MLP projections are
    coupled: the rank owning intermediate rows of ``ffn_gate``/``ffn_up`` must
    reduce over exactly those columns of ``ffn_down``.
    """

    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata

    config = qwen35_gguf_config_from_metadata(info)
    if bool(getattr(config, "is_moe", False)):
        raise ShardPlanError("MoE tensor parallelism is out of scope for this plan")
    notes: list[str] = []
    if int(owner_rank) != 0:
        notes.append(f"single-owner tensors (embedding/lm_head) live on rank {int(owner_rank)}")
    if uneven_split is not None and len(uneven_split.fractions) != int(world_size):
        raise ShardPlanError(
            f"the uneven split names {len(uneven_split.fractions)} shares for world size {world_size}"
        )
    prefixes = tuple(exclude_prefixes) + tuple(
        f"blk.{int(block_id)}." for block_id in getattr(config, "ignored_block_ids", ()) or ()
    )
    # One boundary per split axis, so every coupled tensor of a layer gets the
    # same one regardless of the order the tensors are visited in.
    boundary_cache: dict[int, tuple[tuple[int, int], ...]] = {}
    axis_lengths: dict[str, int] = {}
    plans: list[TensorShardPlan] = []
    excluded: list[str] = []
    for tensor in info.tensors:
        if any(tensor.name.startswith(prefix) for prefix in prefixes):
            excluded.append(tensor.name)
            continue
        rule = shard_rule_for_tensor(tensor.name, config=config)
        if rule.kind == OWNER:
            rule = replace(rule, owner_rank=int(owner_rank))
        element_ranges: Sequence[tuple[int, int]] | None = None
        if uneven_split is not None and uneven_split.applies_to(tensor.name):
            axis = 0 if rule.kind in AXIS0_KINDS else 1
            if len(tensor.shape) <= axis:
                raise ShardPlanError(
                    f"{tensor.name}: an uneven split needs an axis {axis} to split"
                )
            axis_length = int(tensor.shape[axis])
            if axis_length not in boundary_cache:
                boundary_cache[axis_length] = uneven_split.ranges(axis_length)
            element_ranges = boundary_cache[axis_length]
            axis_lengths[tensor.name] = axis_length
        try:
            plans.append(
                build_tensor_shard_plan(
                    name=tensor.name,
                    shape=tensor.shape,
                    nbytes=tensor.nbytes,
                    quant_type_id=tensor.ggml_type,
                    quant_type_name=tensor.ggml_type_name,
                    rule=rule,
                    world_size=int(world_size),
                    element_ranges=element_ranges,
                )
            )
        except ShardPlanError as error:
            raise ShardPlanError(f"{tensor.name}: {error}") from error
    if uneven_split is not None:
        named = {
            _tensor_leaf(tensor.name)
            for tensor in info.tensors
            if uneven_split.applies_to(tensor.name)
        }
        absent = sorted(set(uneven_split.leaves) - named)
        if absent:
            raise ShardPlanError(
                f"the uneven split names {absent}, which this model has no tensor for"
            )
        _validate_coupled_split(plans, uneven_split)
        shares = "/".join(f"{float(value):.6f}" for value in uneven_split.fractions)
        leaves = ",".join(uneven_split.leaves)
        notes.append(
            f"uneven split shares {shares} on {leaves} "
            f"at {int(uneven_split.alignment)}-element alignment"
        )
    if excluded:
        notes.append(f"excluded {len(excluded)} MTP/NextN tensors from the AR shard set")
    notes.append(f"architecture={config.architecture} blocks={config.block_count}")
    return ShardManifest(
        model_hash=str(model_hash),
        world_size=int(world_size),
        hidden_size=int(config.hidden_size),
        tensors=tuple(plans),
        notes=tuple(notes),
    )


def _validate_coupled_split(
    plans: Sequence[TensorShardPlan], uneven_split: UnevenSplitPolicy
) -> None:
    """Assert that every coupled tensor of a layer shares one boundary.

    ``ffn_gate``/``ffn_up`` own intermediate rows and ``ffn_down`` reduces over
    the same columns. If their boundaries ever disagree the MLP result is
    silently wrong rather than an error, so the coupling is checked here as
    well as being shared by construction.
    """

    by_layer: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {}
    for plan in plans:
        if not uneven_split.applies_to(plan.name):
            continue
        layer = plan.name.split(".", 2)[1] if plan.name.startswith("blk.") else ""
        by_layer.setdefault(layer, {})[_tensor_leaf(plan.name)] = tuple(
            tuple(int(value) for value in shard.axis_ranges[0])
            for shard in plan.slices
        )
    for layer, roles in by_layer.items():
        reference_role = sorted(roles)[0]
        reference = roles[reference_role]
        for role, ranges in sorted(roles.items()):
            if ranges != reference:
                raise ShardPlanError(
                    f"layer {layer}: {role} splits at {ranges} but {reference_role} splits at "
                    f"{reference}; the coupled MLP projections must share one boundary"
                )


# ---------------------------------------------------------------------------
# Byte-preserving materialization
# ---------------------------------------------------------------------------


def source_payload(path: Any, *, data_offset: int, nbytes: int) -> np.ndarray:
    """Read-only byte view of one tensor payload (no copy, no dequantization)."""

    return np.memmap(path, mode="r", dtype=np.uint8, offset=int(data_offset), shape=(int(nbytes),))


def materialize_slice(source: np.ndarray, shard_slice: TensorShardSlice, *, allocator: Any = None) -> np.ndarray:
    """Copy one rank's byte segments into a compact local payload.

    The returned array is exactly ``shard_slice.local_nbytes`` bytes. For axis-0
    splits it is a verbatim contiguous copy; for row splits it concatenates each
    row's block-aligned segment, which is what changes the local row stride.
    No byte is decoded, requantized, or reordered inside a quant block.

    ``allocator`` defaults to :func:`numpy.empty` and exists so a caller (or a
    test) can observe the exact bytes this path allocates.
    """

    allocate = np.empty if allocator is None else allocator
    destination = allocate(int(shard_slice.local_nbytes), dtype=np.uint8)
    cursor = 0
    for segment in shard_slice.iter_segments():
        start = int(segment.source_offset)
        stop = start + int(segment.nbytes)
        if stop > source.size:
            raise ShardPlanError(
                f"shard segment [{start}, {stop}) exceeds source payload size {source.size}"
            )
        destination[cursor : cursor + segment.nbytes] = source[start:stop]
        cursor += int(segment.nbytes)
    if cursor != destination.size:
        raise ShardPlanError("materialized shard size does not match its descriptor")
    return destination


def materialize_manifest(
    reader: Any,
    manifest: ShardManifest,
    *,
    tensor_filter: Iterable[str] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Materialize every planned tensor for every rank (host-side, test-scale).

    This is the reconstruction oracle path: it returns the exact bytes each
    rank would upload. It intentionally does not allocate device memory.
    """

    selected = None if tensor_filter is None else set(tensor_filter)
    payloads: dict[int, dict[str, np.ndarray]] = {rank: {} for rank in range(manifest.world_size)}
    for plan in manifest.tensors:
        if selected is not None and plan.name not in selected:
            continue
        tensor = reader.tensor_info(plan.name)
        source = source_payload(reader.path, data_offset=tensor.data_offset, nbytes=tensor.nbytes)
        for rank in range(manifest.world_size):
            payloads[rank][plan.name] = materialize_slice(source, plan.slice_for(rank))
    return payloads


def iter_rank_payloads(
    reader: Any,
    manifest: ShardManifest,
    *,
    rank: int,
    tensor_filter: Iterable[str] | None = None,
) -> Iterator[tuple[TensorShardPlan, np.ndarray]]:
    """Yield ``(plan, payload)`` for one rank, one tensor at a time.

    This is the loader-facing path: a rank uploads each payload and drops it
    before the next one is built, so peak host memory is the largest single
    local tensor rather than a full model copy. The source is a read-only
    memmap, so the tensor payload is never resident twice.
    """

    if int(rank) < 0 or int(rank) >= int(manifest.world_size):
        raise ShardPlanError(f"rank {rank} is outside manifest world size {manifest.world_size}")
    selected = None if tensor_filter is None else set(tensor_filter)
    for plan in manifest.tensors:
        if selected is not None and plan.name not in selected:
            continue
        tensor = reader.tensor_info(plan.name)
        source = source_payload(reader.path, data_offset=tensor.data_offset, nbytes=tensor.nbytes)
        yield plan, materialize_slice(source, plan.slice_for(int(rank)))
        del source


def streaming_memory_report(
    reader: Any,
    manifest: ShardManifest,
    *,
    rank: int,
    tensor_filter: Iterable[str] | None = None,
    sampler: Any = None,
) -> dict[str, Any]:
    """Stream one rank's shards and report the peak memory the path needs.

    ``sampler`` is called after every tensor and returns the current anonymous
    and resident set size in bytes; the default reads
    ``/proc/self/smaps_rollup``. Anonymous bytes are the ones this path
    allocates - resident bytes also include reclaimable file-backed page cache
    from the source memmap.
    """

    observe = _sample_process_memory if sampler is None else sampler
    tensors = 0
    local_bytes = 0
    largest = 0
    peak_anonymous = 0
    peak_rss = 0
    baseline = observe()
    for plan, payload in iter_rank_payloads(reader, manifest, rank=rank, tensor_filter=tensor_filter):
        tensors += 1
        local_bytes += int(payload.size)
        largest = max(largest, int(payload.size))
        sample = observe()
        peak_anonymous = max(peak_anonymous, int(sample[0]))
        peak_rss = max(peak_rss, int(sample[1]))
    return {
        "rank": int(rank),
        "world_size": int(manifest.world_size),
        "tensors": tensors,
        "local_bytes": local_bytes,
        "largest_tensor_bytes": largest,
        "baseline_anonymous_bytes": int(baseline[0]),
        "peak_anonymous_bytes": peak_anonymous,
        "peak_rss_bytes": peak_rss,
        "anonymous_growth_bytes": peak_anonymous - int(baseline[0]),
        "full_model_copy_bytes": manifest.rank_bytes(int(rank)),
        "streaming": True,
    }


def _sample_process_memory() -> tuple[int, int]:
    """Return ``(anonymous_bytes, rss_bytes)`` for this process."""

    anonymous = 0
    rss = 0
    try:
        with open("/proc/self/smaps_rollup", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("Anonymous:"):
                    anonymous = int(line.split()[1]) * 1024
                elif line.startswith("Rss:"):
                    rss = int(line.split()[1]) * 1024
    except OSError:
        pass
    return anonymous, rss


def verify_shard_bytes(
    reader: Any,
    manifest: ShardManifest,
    *,
    tensor_filter: Iterable[str] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Stream the whole manifest and check every shard set is bit-identical.

    Peak memory is one tensor's payload times the world size, so this is safe
    to run against a full-size model. Returns a serializable report.
    """

    selected = None if tensor_filter is None else set(tensor_filter)
    checked = 0
    total_bytes = 0
    mismatches: list[str] = []
    kind_bytes: dict[str, int] = {}
    for plan in manifest.tensors:
        if selected is not None and plan.name not in selected:
            continue
        tensor = reader.tensor_info(plan.name)
        source = source_payload(reader.path, data_offset=tensor.data_offset, nbytes=tensor.nbytes)
        payloads = {
            rank: materialize_slice(source, plan.slice_for(rank)) for rank in range(manifest.world_size)
        }
        rebuilt = reconstruct_tensor(manifest, plan.name, payloads)
        identical = bool(np.array_equal(rebuilt, np.asarray(source)))
        if not identical:
            mismatches.append(plan.name)
        checked += 1
        total_bytes += int(plan.source_nbytes)
        kind_bytes[plan.kind] = kind_bytes.get(plan.kind, 0) + int(plan.source_nbytes)
        if progress is not None:
            progress(plan, identical)
    return {
        "world_size": int(manifest.world_size),
        "manifest_hash": manifest.manifest_hash(),
        "tensors_checked": checked,
        "source_bytes": total_bytes,
        "bytes_by_kind": kind_bytes,
        "rank_bytes": [manifest.rank_bytes(rank) for rank in range(manifest.world_size)],
        "bit_exact": not mismatches,
        "mismatches": mismatches,
    }


def reconstruct_tensor(manifest: ShardManifest, name: str, payloads: Mapping[int, np.ndarray]) -> np.ndarray:
    """Reassemble the original tensor payload from all rank shards.

    This is the byte-preservation oracle: the result must be bit-identical to
    the source tensor payload. Replication is resolved by preferring the lowest
    rank that owns a byte range, and every byte must be covered exactly once.
    """

    plan = manifest.plan_for(name)
    destination = np.zeros(int(plan.source_nbytes), dtype=np.uint8)
    covered = np.zeros(int(plan.source_nbytes), dtype=np.bool_)
    for rank in range(manifest.world_size):
        payload = np.asarray(payloads[rank], dtype=np.uint8)
        shard_slice = plan.slice_for(rank)
        if payload.size != int(shard_slice.local_nbytes):
            raise ShardPlanError(
                f"rank {rank} payload for {name!r} has {payload.size} bytes, expected {shard_slice.local_nbytes}"
            )
        cursor = 0
        for segment in shard_slice.iter_segments():
            start = int(segment.source_offset)
            stop = start + int(segment.nbytes)
            chunk = payload[cursor : cursor + segment.nbytes]
            if not covered[start:stop].all():
                fresh = ~covered[start:stop]
                destination[start:stop][fresh] = chunk[fresh]
                covered[start:stop] = True
            cursor += int(segment.nbytes)
    if not covered.all():
        missing = int((~covered).sum())
        raise ShardPlanError(f"reconstruction of {name!r} is missing {missing} bytes")
    return destination
