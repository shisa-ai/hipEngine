"""Layer-0 dense-MLP capture descriptors plus an independent FP32 reference.

Diagnostics only, CPU-only, no GPU and no heavy runtime imports.

Why this module exists
----------------------

After the GGUF dispatch-context fix, every layer-0 *attention* producer is
bit-identical between the resident TP1 bulk teacher and the opt-in TP2 bulk
candidate (``scripts/tp2_bulk_vs_resident_layer0.py``). The remaining layer-0
difference therefore has to come from the MLP half, where the two routes run
genuinely different schedules:

    resident TP1   add_rmsnorm -> fused gate/up+SiLU -> fused down+residual
    TP2 bulk       add_rmsnorm -> per-rank gate, up, SiLU
                   -> per-rank *bf16* down partial
                   -> staged f32 reduce -> bf16 cast -> bf16 residual add

The teacher never materializes a gate/up intermediate (its pair+SiLU kernel
writes only the activation) and never materializes an f32 down output (its
down+residual kernel rounds once, at the residual add). So the comparable
boundaries are the activation rows and the layer output; the rank gate/up/act
and partials are captured anyway so the bf16-partial boundary can be *measured*
against an independent FP32 reference instead of asserted.

Two things are deliberately derived rather than assumed:

* the capture descriptors (dtype/width/producer/lifetime) come from the
  producer call sites and the allocator's own lifetime table, exactly like
  ``scripts/tp2_layer0_capture``;
* the reference arithmetic below is an independent numpy/float32
  implementation of *both* schedules, so "the sharded route adds a bf16
  rounding the teacher does not" is a measurable quantity, not a claim.

The module is free of ``hipengine`` imports so the schedule arithmetic and the
descriptor logic can be unit tested without a GPU.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from scripts.tp2_layer0_capture import (
    BF16,
    F32,
    BufferDescriptor,
    CaptureError,
    Producer,
    build_capture_plan,
    descriptor,
    require_positive_int,
)

#: Partial dtypes the staged exchange can reduce from. ``f32`` is the
#: schedule that introduces no extra rounding; ``bf16`` is what the shipped
#: TP2 bulk group uses.
PARTIAL_DTYPES = (F32, BF16)


# -- bf16 arithmetic --------------------------------------------------------


def bf16_round_bits(values: object) -> np.ndarray:
    """Round float32 values to bfloat16 bits, round-to-nearest-even.

    This is the same idiom the repo's device cast is checked against
    (``(bits + 0x7FFF + ((bits >> 16) & 1)) >> 16``): the low 16 mantissa bits
    are discarded with a round-half-to-even tie-break. NaN payloads are not
    preserved, which is fine for the reference here.
    """

    arr = np.ascontiguousarray(values, dtype=np.float32)
    bits = arr.view(np.uint32).astype(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype(np.uint16)


def bf16_round(values: object) -> np.ndarray:
    """Round float32 values to bfloat16 and widen back to float32."""

    bits = bf16_round_bits(values).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def bf16_ulp(value: np.float32) -> np.float32:
    """The bf16 spacing at ``value`` (one unit in the last place)."""

    magnitude = np.float32(abs(float(value)))
    if magnitude == 0.0 or not np.isfinite(magnitude):
        return np.float32(0.0)
    exponent = np.frexp(magnitude)[1]
    return np.ldexp(np.float32(1.0), exponent - 8)


def silu(values: object) -> np.ndarray:
    """``x * sigmoid(x)`` in float32, computed the way the fused kernel does."""

    arr = np.asarray(values, dtype=np.float32)
    return (arr / (np.float32(1.0) + np.exp(-arr))).astype(np.float32)


# -- shard ownership --------------------------------------------------------


def shard_slices(
    total: int, *, ranks: int, what: str = "axis"
) -> tuple[tuple[int, int], ...]:
    """Split ``total`` into ``ranks`` contiguous, exhaustive, disjoint ranges."""

    total_i = require_positive_int(total, what=what)
    ranks_i = require_positive_int(ranks, what="ranks")
    if total_i % ranks_i:
        raise CaptureError(f"{what} {total_i} does not split across {ranks_i} ranks")
    step = total_i // ranks_i
    return tuple((rank * step, (rank + 1) * step) for rank in range(ranks_i))


def verify_shard_ownership(
    ranges: Sequence[tuple[int, int]], *, total: int, what: str = "axis"
) -> None:
    """Fail closed unless ``ranges`` tile ``[0, total)`` exactly once.

    This is the exact ownership contract the reduction depends on: a gap would
    drop part of the full-width sum and an overlap would count it twice, and
    neither is a rounding effect.
    """

    total_i = require_positive_int(total, what=what)
    cursor = 0
    for start, stop in ranges:
        if not isinstance(start, int) or not isinstance(stop, int):
            raise CaptureError(f"{what} range {(start, stop)!r} must be ints")
        if start != cursor:
            raise CaptureError(
                f"{what} ranges are not contiguous: expected start {cursor}, got {start}"
            )
        if stop <= start:
            raise CaptureError(f"{what} range {(start, stop)!r} is empty or reversed")
        cursor = stop
    if cursor != total_i:
        raise CaptureError(f"{what} ranges cover {cursor} of {total_i}")


# -- the reference schedules ------------------------------------------------


def _matmul(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``x @ weight.T`` in float32 (numpy's BLAS sgemm is the reference)."""

    return np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T


def reference_full_width_mlp(
    x: object, gate: object, up: object, down: object
) -> dict[str, np.ndarray]:
    """The resident TP1 schedule in float32: fused pair+SiLU then down.

    ``x`` is ``(rows, hidden)``, ``gate``/``up`` are ``(ffn, hidden)`` and
    ``down`` is ``(hidden, ffn)``. Both the gate/up accumulators and the
    activation are rounded to bf16 before the down projection, mirroring the
    resident kernel contract ("both accumulators are bf16-rounded before the
    SiLU, exactly like the unfused pair").
    """

    x_f = np.asarray(x, dtype=np.float32)
    rows = require_positive_int(int(x_f.shape[0]), what="x rows")
    gate_f = np.asarray(gate, dtype=np.float32)
    up_f = np.asarray(up, dtype=np.float32)
    down_f = np.asarray(down, dtype=np.float32)
    if gate_f.shape != up_f.shape:
        raise CaptureError(f"gate {gate_f.shape} and up {up_f.shape} differ")
    if int(gate_f.shape[1]) != int(x_f.shape[1]):
        raise CaptureError("gate in_features does not match x hidden")
    if tuple(down_f.shape) != (int(x_f.shape[1]), int(gate_f.shape[0])):
        raise CaptureError("down shape does not match (hidden, ffn)")

    gate_out = bf16_round(_matmul(x_f, gate_f))
    up_out = bf16_round(_matmul(x_f, up_f))
    intermediate = bf16_round(silu(gate_out) * up_out)
    down_f32 = _matmul(intermediate, down_f)
    return {
        "rows": rows,
        "gate": gate_out,
        "up": up_out,
        "intermediate": intermediate,
        "down_f32": down_f32,
        "down": bf16_round(down_f32),
    }


def reference_exact_partials(
    intermediate: object, down: object, *, ranks: int
) -> list[np.ndarray]:
    """Per-rank float32 down partials for a *measured* activation.

    ``intermediate`` is ``(rows, ffn)`` and ``down`` is ``(hidden, ffn)``. The
    ``ffn`` axis is row-parallel, so rank *i* multiplies its own contiguous
    ``ffn`` slice of the activation by the matching ``ffn`` slice of ``down``:
    both operands are sliced on the same range. Slicing only ``down`` would
    contract a full-width activation against a shard-width weight and is the
    defect this helper exists to make unrepeatable.
    """

    x_f = np.asarray(intermediate, dtype=np.float32)
    down_f = np.asarray(down, dtype=np.float32)
    if down_f.ndim != 2:
        raise CaptureError(f"down must be 2-D, got shape {down_f.shape}")
    if x_f.ndim != 2:
        raise CaptureError(f"intermediate must be 2-D, got shape {x_f.shape}")
    ffn = require_positive_int(int(x_f.shape[1]), what="ffn")
    if int(down_f.shape[1]) != ffn:
        raise CaptureError(
            f"down {down_f.shape} does not match the {ffn}-wide activation"
        )
    ranges = shard_slices(ffn, ranks=ranks, what="ffn")
    return [
        _matmul(x_f[:, start:stop], down_f[:, start:stop])
        for (start, stop) in ranges
    ]


def reference_sharded_mlp(
    x: object,
    gate: object,
    up: object,
    down: object,
    *,
    ranks: int,
    partial_dtype: str = BF16,
) -> dict[str, object]:
    """The TP2 bulk schedule in float32: per-rank chain, partials, f32 reduce.

    ``gate``/``up`` are split on the **output**-feature axis (column parallel,
    each rank owning a contiguous slice of the nonlinearity) and ``down`` is
    split on the **input**-feature axis (row parallel), matching
    ``hipengine.distributed.shard_weights``. Each rank's down output is rounded
    to ``partial_dtype`` *before* the staged exchange sums the ranks in f32 -
    that rounding is the whole point of the comparison.
    """

    if partial_dtype not in PARTIAL_DTYPES:
        raise CaptureError(
            f"partial_dtype must be one of {PARTIAL_DTYPES}, got {partial_dtype!r}"
        )
    x_f = np.asarray(x, dtype=np.float32)
    rows = require_positive_int(int(x_f.shape[0]), what="x rows")
    gate_f = np.asarray(gate, dtype=np.float32)
    up_f = np.asarray(up, dtype=np.float32)
    down_f = np.asarray(down, dtype=np.float32)
    ffn = require_positive_int(int(gate_f.shape[0]), what="ffn")
    if gate_f.shape != up_f.shape:
        raise CaptureError(f"gate {gate_f.shape} and up {up_f.shape} differ")
    if int(gate_f.shape[1]) != int(x_f.shape[1]):
        raise CaptureError("gate in_features does not match x hidden")
    if tuple(down_f.shape) != (int(x_f.shape[1]), ffn):
        raise CaptureError("down shape does not match (hidden, ffn)")
    ranks_i = require_positive_int(ranks, what="ranks")
    output_ranges = shard_slices(ffn, ranks=ranks_i, what="ffn")
    input_ranges = shard_slices(ffn, ranks=ranks_i, what="ffn input")

    intermediates: list[np.ndarray] = []
    for (start, stop) in output_ranges:
        gate_shard = bf16_round(_matmul(x_f, gate_f[start:stop]))
        up_shard = bf16_round(_matmul(x_f, up_f[start:stop]))
        intermediates.append(bf16_round(silu(gate_shard) * up_shard))
    partials = reference_exact_partials(
        np.concatenate(intermediates, axis=1), down_f, ranks=ranks_i
    )
    if partial_dtype != F32:
        partials = [bf16_round(partial) for partial in partials]

    reduced = np.zeros((rows, int(x_f.shape[1])), dtype=np.float32)
    for partial in partials:
        reduced = reduced + partial.astype(np.float32)
    return {
        "rows": rows,
        "ranks": ranks_i,
        "partial_dtype": partial_dtype,
        "output_ranges": output_ranges,
        "input_ranges": input_ranges,
        "intermediates": intermediates,
        "partials": partials,
        "reduced_f32": reduced,
        "reduced_bf16": bf16_round(reduced),
    }


def concatenate_rank_rows(arrays: Sequence[object], *, ranks: int) -> np.ndarray:
    """Concatenate per-rank activation shards along the feature axis."""

    ranks_i = require_positive_int(ranks, what="ranks")
    if len(arrays) != ranks_i:
        raise CaptureError(f"expected {ranks_i} rank arrays, got {len(arrays)}")
    parts = [np.asarray(item, dtype=np.float32) for item in arrays]
    return np.concatenate(parts, axis=1)


# -- capture descriptors ----------------------------------------------------

#: Fields the resident rows>1 dense-MLP schedule actually materializes. The
#: bulk down+residual fusion fails closed for T16 layouts (it is registered
#: only for ``dense_bf16`` weights), so the resident layer runs the *unfused*
#: chain and does write an ``ffn_down`` plane before its own bf16 residual add.
RESIDENT_MLP_FIELDS: tuple[str, ...] = (
    "post_norm",
    "residual",
    "ffn_intermediate",
    "ffn_down",
    "out",
)

#: Per-rank fields the TP2 bulk shard schedule materializes.
SHARD_MLP_FIELDS: tuple[str, ...] = (
    "post_norm",
    "residual",
    "gate",
    "up",
    "act",
    "down_partial",
    "reduced",
    "cast",
    "out",
)


def resident_mlp_producers() -> tuple[Producer, ...]:
    """The resident TP1 layer-0 MLP producer order, read off the helper body.

    At ``rows > 1`` the gate/up pair is fused with the SiLU
    (``launch_gguf_linear_pair_silu``) so only the activation is materialized,
    but the down+residual fusion is registered for ``dense_bf16`` weights only
    and fails closed for the T16 layouts this model loads, so the down
    projection writes its own bf16 plane and the residual is a separate add.
    """

    return (
        Producer("post_norm_residual", "unresolved", ("post_norm", "residual")),
        Producer("gate_up_silu", "unresolved", ("ffn_intermediate",)),
        Producer("down", "unresolved", ("ffn_down",)),
        Producer("residual_add", "unresolved", ("out",)),
    )


def shard_mlp_producers(devices: Sequence[int]) -> tuple[Producer, ...]:
    """The TP2 bulk layer-0 MLP producer order, one set of names per rank.

    ``post_norm``/``residual``/``out`` are per-rank too: each rank runs its own
    post-attention norm and its own residual add. The gate/up/SiLU/down chain
    runs per rank, the staged exchange reduces once, and the bf16 cast is a
    separate producer from the reduce so the f32 boundary is observable.
    """

    rank_ids = tuple(int(device) for device in devices)
    if not rank_ids:
        raise CaptureError("shard_mlp_producers needs at least one rank")
    if len(set(rank_ids)) != len(rank_ids):
        raise CaptureError(f"duplicate ranks {rank_ids!r}")
    producers: list[Producer] = [
        Producer(
            "post_norm_residual",
            "unresolved",
            tuple(f"post_norm@{d}" for d in rank_ids)
            + tuple(f"residual@{d}" for d in rank_ids),
        ),
        Producer("shard_gate", "unresolved", tuple(f"gate@{d}" for d in rank_ids)),
        Producer("shard_up", "unresolved", tuple(f"up@{d}" for d in rank_ids)),
        Producer("shard_silu", "unresolved", tuple(f"act@{d}" for d in rank_ids)),
        Producer(
            "shard_down", "unresolved", tuple(f"down_partial@{d}" for d in rank_ids)
        ),
        Producer("staged_reduce", "unresolved", tuple(f"reduced@{d}" for d in rank_ids)),
        Producer("cast_reduced", "unresolved", tuple(f"cast@{d}" for d in rank_ids)),
        Producer("residual_add", "unresolved", tuple(f"out@{d}" for d in rank_ids)),
    ]
    return tuple(producers)


def shard_mlp_producers_single_rank() -> tuple[Producer, ...]:
    """The shard MLP producer order for a *single* rank's own capture.

    ``scripts/tp2_bulk_vs_resident_layer0`` captures one rank per recorder, so
    that recorder's field names are unsuffixed; this is the same producer
    order as :func:`shard_mlp_producers` without the ``@<device>`` suffix.
    """

    return (
        Producer("post_norm_residual", "unresolved", ("post_norm", "residual")),
        Producer("shard_gate", "unresolved", ("gate",)),
        Producer("shard_up", "unresolved", ("up",)),
        Producer("shard_silu", "unresolved", ("act",)),
        Producer("shard_down", "unresolved", ("down_partial",)),
        Producer("staged_reduce", "unresolved", ("reduced",)),
        Producer("cast_reduced", "unresolved", ("cast",)),
        Producer("residual_add", "unresolved", ("out",)),
    )


def resident_mlp_widths(*, hidden_size: int, ffn_size: int) -> Mapping[str, tuple[str, int]]:
    """``(dtype, width)`` for the resident layer-0 MLP capture fields."""

    hidden = require_positive_int(hidden_size, what="hidden_size")
    ffn = require_positive_int(ffn_size, what="ffn_size")
    return {
        "post_norm": (BF16, hidden),
        "residual": (BF16, hidden),
        "ffn_intermediate": (BF16, ffn),
        "ffn_down": (BF16, hidden),
        "out": (BF16, hidden),
    }


def shard_mlp_widths(
    *, hidden_size: int, per_rank_ffn: int, partial_dtype: str = BF16
) -> Mapping[str, tuple[str, int]]:
    """``(dtype, width)`` for the TP2 bulk layer-0 MLP capture fields.

    ``down_partial`` is the rank's local down output in the *staging* dtype,
    ``reduced`` is the exchange's f32 published payload, and ``cast`` is the
    bf16 boundary buffer the residual add consumes.
    """

    hidden = require_positive_int(hidden_size, what="hidden_size")
    shard = require_positive_int(per_rank_ffn, what="per_rank_ffn")
    if partial_dtype not in PARTIAL_DTYPES:
        raise CaptureError(
            f"partial_dtype must be one of {PARTIAL_DTYPES}, got {partial_dtype!r}"
        )
    return {
        "post_norm": (BF16, hidden),
        "residual": (BF16, hidden),
        "gate": (BF16, shard),
        "up": (BF16, shard),
        "act": (BF16, shard),
        "down_partial": (partial_dtype, hidden),
        "reduced": (F32, hidden),
        "cast": (BF16, hidden),
        "out": (BF16, hidden),
    }


def _descriptors_for(
    *,
    fields: Mapping[str, tuple[str, int]],
    producer_of: Mapping[str, str],
    buffers: Mapping[str, object],
    rows: int,
    lifetimes: Mapping[str, tuple],
    what: str,
) -> tuple[BufferDescriptor, ...]:
    rows_i = require_positive_int(rows, what=f"{what} rows")
    out: list[BufferDescriptor] = []
    for name, (dtype, width) in fields.items():
        buffer = buffers.get(name)
        if buffer is None:
            raise CaptureError(f"{what} has no {name!r} buffer to capture")
        if name not in producer_of:
            raise CaptureError(f"{what} has no declared producer for {name!r}")
        if name not in lifetimes:
            raise CaptureError(f"{what} has no arena lifetime for {name!r}")
        out.append(
            descriptor(
                name=name,
                producer=producer_of[name],
                ptr=int(getattr(buffer, "ptr")),
                rows=rows_i,
                width=width,
                dtype=dtype,
                allocated_nbytes=int(getattr(buffer, "nbytes")),
                lifetime=lifetimes[name],
            )
        )
    return tuple(out)


#: Per-rank buffers that are allocated once and never arena-aliased: the shard
#: rank's gate/up/act planes and down partial, the exchange's published f32
#: payload, the bf16 boundary buffer, and the layer output. A single
#: never-reused slot models them; they cannot be clobbered by a later producer.
PERSISTENT_LIFETIME: tuple = (("persistent", 0, 1),)


def shard_mlp_lifetimes(
    *, scratch_lifetimes: Mapping[str, tuple], hidden_fields: Sequence[str] = ("post_norm", "residual")
) -> Mapping[str, tuple]:
    """One rank's per-field arena lifetimes for the shard MLP capture fields.

    ``post_norm``/``residual`` are prefill-scratch fields and take their real
    liveness intervals from the allocator's table; everything else on the shard
    route is a persistent rank-local allocation, so it gets
    :data:`PERSISTENT_LIFETIME` and can never be aliased.
    """

    names = set(shard_mlp_widths(hidden_size=1, per_rank_ffn=1))
    for name in names:
        if name in hidden_fields and name not in scratch_lifetimes:
            raise CaptureError(f"no arena lifetime for shard scratch field {name!r}")
    return {
        name: (scratch_lifetimes[name] if name in hidden_fields else PERSISTENT_LIFETIME)
        for name in sorted(names)
    }


def describe_resident_mlp(
    *,
    buffers: Mapping[str, object],
    rows: int,
    hidden_size: int,
    ffn_size: int,
    lifetimes: Mapping[str, tuple],
    producers: Sequence[Producer] | None = None,
) -> tuple[BufferDescriptor, ...]:
    """Build the resident layer-0 MLP descriptors from live buffers."""

    order = tuple(producers or resident_mlp_producers())
    producer_of = {
        name: producer.name for producer in order for name in producer.buffers
    }
    return _descriptors_for(
        fields=resident_mlp_widths(hidden_size=hidden_size, ffn_size=ffn_size),
        producer_of=producer_of,
        buffers=buffers,
        rows=rows,
        lifetimes=lifetimes,
        what="resident MLP",
    )


def describe_shard_mlp(
    *,
    buffers_by_rank: Mapping[int, Mapping[str, object]],
    rows: int,
    hidden_size: int,
    per_rank_ffn: int,
    partial_dtype: str = BF16,
    lifetimes_by_rank: Mapping[int, Mapping[str, tuple]] | None = None,
    default_lifetimes: Mapping[str, tuple] | None = None,
    producers: Sequence[Producer] | None = None,
) -> tuple[BufferDescriptor, ...]:
    """Build the TP2 bulk layer-0 MLP descriptors for every rank.

    Rank-local names are suffixed ``@<device>`` so one capture plan covers the
    whole group; ``buffers_by_rank`` and ``lifetimes_by_rank`` are keyed by the
    same device ids.
    """

    ranks = tuple(sorted(int(device) for device in buffers_by_rank))
    order = tuple(producers or shard_mlp_producers(ranks))
    producer_of = {
        name: producer.name for producer in order for name in producer.buffers
    }
    widths = shard_mlp_widths(
        hidden_size=hidden_size, per_rank_ffn=per_rank_ffn, partial_dtype=partial_dtype
    )
    out: list[BufferDescriptor] = []
    for rank in ranks:
        buffers = buffers_by_rank[rank]
        rank_lifetimes = (
            lifetimes_by_rank[rank]
            if lifetimes_by_rank is not None and rank in lifetimes_by_rank
            else (default_lifetimes or {})
        )
        out.extend(
            _descriptors_for(
                fields={f"{name}@{rank}": spec for name, spec in widths.items()},
                producer_of=producer_of,
                buffers={f"{name}@{rank}": buffers.get(name) for name in widths},
                rows=rows,
                lifetimes={
                    f"{name}@{rank}": rank_lifetimes[name] for name in widths
                },
                what=f"shard MLP rank {rank}",
            )
        )
    return tuple(out)


def build_mlp_capture_plan(
    descriptors: Sequence[BufferDescriptor],
    producers: Sequence[Producer],
) -> tuple:
    """Order the MLP captures by producer and verify each read precedes reuse."""

    return build_capture_plan(descriptors, producers)


__all__ = [
    "BF16",
    "F32",
    "PARTIAL_DTYPES",
    "PERSISTENT_LIFETIME",
    "RESIDENT_MLP_FIELDS",
    "SHARD_MLP_FIELDS",
    "bf16_round",
    "bf16_round_bits",
    "bf16_ulp",
    "build_mlp_capture_plan",
    "concatenate_rank_rows",
    "describe_resident_mlp",
    "describe_shard_mlp",
    "reference_full_width_mlp",
    "reference_sharded_mlp",
    "resident_mlp_producers",
    "resident_mlp_widths",
    "shard_mlp_lifetimes",
    "shard_mlp_producers",
    "shard_mlp_producers_single_rank",
    "shard_mlp_widths",
    "shard_slices",
    "silu",
    "verify_shard_ownership",
]
