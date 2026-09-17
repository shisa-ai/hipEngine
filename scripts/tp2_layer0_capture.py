"""Producer-derived capture descriptors for the layer-0 GDN primitive chain.

Diagnostics only, CPU-only, no GPU and no heavy runtime imports.

Why this module exists
----------------------

The resident and TP2 bulk prefill scratch stacks allocate their per-layer
workspace with **liveness aliasing**: ``_GGUFFullAttentionPrefillScratch.allocate``
feeds ``_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES`` (or the MoE table) to
``_allocate_prefill_scratch_liveness_arenas`` and packs fields whose
``(route, start, end)`` intervals do not overlap into the same arena bytes.
For the layer-0 linear-attention route that means, for example::

    norm           (linear, 0, 1)
    linear_qkv     (linear, 0, 2)
    linear_qkv_f32 (linear, 1, 3)
    conv_out       (linear, 2, 5)
    prefill_query  (linear, 3, 5)
    recurrent_out  (linear, 4, 5)
    recurrent_bf16 (linear, 5, 6)
    attn_out       (linear, 5, 7)

``norm``/``linear_qkv``/``linear_qkv_f32``/``conv_out``/``recurrent_out`` are
pairwise stage-disjoint, so the arena is free to reuse their bytes.  Reading
all of them once at the end of the layer therefore returns whichever later
producer last wrote those bytes -- not the value the producer computed.  A
capture is only meaningful when it happens **immediately after its own
producer and before any aliasing writer runs**.

The descriptors here are derived from the producer (its call site, the
runner's config-derived widths, and the allocator's own lifetime table), never
from a comparison result.  ``validate_capture_order`` fails closed when a
requested capture point sits after an aliasing writer, which is exactly the
bug the end-of-layer bulk read has.

The module is intentionally free of ``hipengine`` imports so the descriptor
and lifetime/alias logic can be unit tested without a GPU; callers pass the
lifetime mapping and the producer order in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

BF16 = "bf16"
F32 = "f32"

ITEMSIZE: Mapping[str, int] = {BF16: 2, F32: 4}

Lifetime = tuple[tuple[str, int, int], ...]


class CaptureError(ValueError):
    """Raised when a capture descriptor or capture point is not well formed."""


class CapturePlanError(CaptureError):
    """Raised when a capture point would read an aliased (overwritten) buffer."""


def require_positive_int(value: object, *, what: str) -> int:
    """Return ``value`` as an ``int`` or raise ``CaptureError``.

    ``bool`` is rejected even though it is an ``int`` subclass: a flag passed
    where a row count belongs is a caller bug, not ``1``.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureError(f"{what} must be a positive int, got {value!r}")
    if value <= 0:
        raise CaptureError(f"{what} must be positive, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Producer:
    """One producer call site in the layer-0 primitive chain.

    ``name`` is the logical operation (``attn_norm``, ``conv_prefill``, ...),
    ``kernel`` is the concrete selected kernel/route recorded at run time, and
    ``buffers`` are the scratch fields this producer is the final writer of.
    """

    name: str
    kernel: str
    buffers: tuple[str, ...]


@dataclass(frozen=True)
class BufferDescriptor:
    """A capture descriptor whose layout comes from the producer call site."""

    name: str
    producer: str
    ptr: int
    rows: int
    width: int
    dtype: str
    allocated_nbytes: int
    lifetime: Lifetime

    @property
    def itemsize(self) -> int:
        return ITEMSIZE[self.dtype]

    @property
    def nbytes(self) -> int:
        """Bytes the producer writes: ``rows * width * itemsize``."""

        return self.rows * self.width * self.itemsize

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.width)

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "producer": self.producer,
            "ptr": int(self.ptr),
            "rows": int(self.rows),
            "width": int(self.width),
            "dtype": self.dtype,
            "shape": [int(self.rows), int(self.width)],
            "itemsize": self.itemsize,
            "nbytes": self.nbytes,
            "allocated_nbytes": int(self.allocated_nbytes),
            "lifetime": [list(interval) for interval in self.lifetime],
        }


def descriptor(
    *,
    name: str,
    producer: str,
    ptr: object,
    rows: object,
    width: object,
    dtype: str,
    allocated_nbytes: object,
    lifetime: Lifetime,
) -> BufferDescriptor:
    """Build and validate a capture descriptor from its producer call site."""

    rows_i = require_positive_int(rows, what=f"{name}.rows")
    width_i = require_positive_int(width, what=f"{name}.width")
    if dtype not in ITEMSIZE:
        raise CaptureError(
            f"{name}.dtype must be one of {sorted(ITEMSIZE)}, got {dtype!r}"
        )
    if isinstance(ptr, bool) or not isinstance(ptr, int) or ptr <= 0:
        raise CaptureError(f"{name}.ptr must be a positive int, got {ptr!r}")
    allocated = require_positive_int(allocated_nbytes, what=f"{name}.allocated_nbytes")
    nbytes = rows_i * width_i * ITEMSIZE[dtype]
    if nbytes > allocated:
        raise CaptureError(
            f"{name} capture would read {nbytes} B but the allocation is only "
            f"{allocated} B"
        )
    if not lifetime:
        raise CaptureError(f"{name} has no arena lifetime interval")
    for route, start, end in lifetime:
        if not isinstance(route, str) or not route:
            raise CaptureError(f"{name} lifetime route must be a non-empty str")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            raise CaptureError(
                f"{name} lifetime interval {(route, start, end)!r} must satisfy start < end"
            )
    return BufferDescriptor(
        name=name,
        producer=producer,
        ptr=int(ptr),
        rows=rows_i,
        width=width_i,
        dtype=dtype,
        allocated_nbytes=allocated,
        lifetime=lifetime,
    )


def lifetimes_overlap(lhs: Lifetime, rhs: Lifetime) -> bool:
    """True when two arena lifetimes can be live at the same time."""

    return any(
        lhs_route == rhs_route and lhs_start < rhs_end and rhs_start < lhs_end
        for lhs_route, lhs_start, lhs_end in lhs
        for rhs_route, rhs_start, rhs_end in rhs
    )


def aliasing_pairs(
    descriptors: Sequence[BufferDescriptor], *, route: str | None = None
) -> tuple[tuple[str, str], ...]:
    """Pairs whose arena bytes may be reused (disjoint lifetimes on ``route``).

    The comparison is restricted to a single route when ``route`` is given,
    because the linear- and full-attention fields never share arena pages.
    """

    def on_route(item: BufferDescriptor) -> bool:
        return route is None or any(r == route for r, _, _ in item.lifetime)

    pairs: list[tuple[str, str]] = []
    for i, lhs in enumerate(descriptors):
        if not on_route(lhs):
            continue
        for rhs in descriptors[i + 1 :]:
            if not on_route(rhs):
                continue
            if not lifetimes_overlap(lhs.lifetime, rhs.lifetime):
                pairs.append((lhs.name, rhs.name))
    return tuple(pairs)


def validate_capture_order(
    descriptors: Sequence[BufferDescriptor],
    *,
    capture_index: Mapping[str, int],
    write_index: Mapping[str, int],
    route: str | None = None,
) -> None:
    """Fail closed when a capture would read bytes an aliasing writer replaced.

    ``write_index[name]`` is the position of the buffer's own producer in the
    layer execution order; ``capture_index[name]`` is where the diagnostic
    actually reads it.  For every aliasing pair ``(a, b)`` the earlier writer's
    bytes are clobbered by the later writer, so a read of the earlier buffer at
    or after the later writer returns the wrong tensor.
    """

    names = {item.name for item in descriptors}
    for mapping, what in (
        (capture_index, "capture_index"),
        (write_index, "write_index"),
    ):
        missing = sorted(names - set(mapping))
        if missing:
            raise CapturePlanError(f"{what} is missing descriptors: {missing}")

    for lhs, rhs in aliasing_pairs(descriptors, route=route):
        first, second = sorted((lhs, rhs), key=lambda name: write_index[name])
        if capture_index[first] >= write_index[second]:
            raise CapturePlanError(
                f"capture of {first!r} at step {capture_index[first]} is not before "
                f"the aliasing writer {second!r} at step {write_index[second]}; "
                f"{first!r} and {second!r} share arena bytes (disjoint lifetimes)"
            )


@dataclass(frozen=True)
class CaptureStep:
    """One producer boundary: capture these buffers right after it runs."""

    producer: str
    kernel: str
    buffers: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "producer": self.producer,
            "kernel": self.kernel,
            "buffers": list(self.buffers),
        }


def build_capture_plan(
    descriptors: Sequence[BufferDescriptor],
    producers: Sequence[Producer],
    *,
    route: str | None = None,
) -> tuple[CaptureStep, ...]:
    """Order the captures by producer and verify each one is read before reuse.

    Raises ``CapturePlanError`` when a descriptor names an unknown producer, a
    producer is declared twice, or the resulting plan would read an aliased
    buffer after its bytes were reused.
    """

    seen: set[str] = set()
    for item in descriptors:
        if item.name in seen:
            raise CapturePlanError(f"duplicate capture descriptor {item.name!r}")
        seen.add(item.name)

    by_name: dict[str, Producer] = {}
    order: list[Producer] = []
    for producer in producers:
        if producer.name in by_name:
            raise CapturePlanError(f"duplicate producer {producer.name!r}")
        by_name[producer.name] = producer
        order.append(producer)

    write_index: dict[str, int] = {}
    for index, producer in enumerate(order):
        for buffer in producer.buffers:
            write_index[buffer] = index

    capture_index: dict[str, int] = {}
    for item in descriptors:
        producer = by_name.get(item.producer)
        if producer is None:
            raise CapturePlanError(
                f"{item.name!r} names unknown producer {item.producer!r}"
            )
        if item.name not in producer.buffers:
            raise CapturePlanError(
                f"{item.name!r} is not declared as a buffer of producer "
                f"{producer.name!r}"
            )
        capture_index[item.name] = write_index[item.name]

    validate_capture_order(
        descriptors,
        capture_index=capture_index,
        write_index=write_index,
        route=route,
    )

    steps: list[CaptureStep] = []
    for producer in order:
        names = tuple(
            item.name for item in descriptors if item.producer == producer.name
        )
        if names:
            steps.append(
                CaptureStep(
                    producer=producer.name, kernel=producer.kernel, buffers=names
                )
            )
    return tuple(steps)


def linear_layer0_producers() -> tuple[Producer, ...]:
    """The layer-0 linear-attention producer order, read off the helper body.

    ``_run_linear_attention_prefill_attn_rows`` runs, for
    ``linear_state_rows is None`` (the route both the resident TP1 bulk teacher
    and the TP2 bulk candidate take):

    ``_run_attention_norm_rows`` -> QKV/gate projections ->
    ``_run_linear_attention_alpha_beta_rows`` -> ``bf16_to_f32`` ->
    conv prefill -> ``_run_gdn_prefill`` (prepare, recurrent, rmsnorm_gate) ->
    ``ssm_out``.
    """

    return (
        Producer("attn_norm", "unresolved", ("norm",)),
        Producer("qkv_gate", "unresolved", ("linear_qkv", "linear_z")),
        Producer("alpha_beta", "unresolved", ("linear_alpha", "linear_beta")),
        Producer("qkv_bf16_to_f32", "unresolved", ("linear_qkv_f32",)),
        Producer("conv_prefill", "unresolved", ("conv_out",)),
        Producer(
            "gdn_prepare",
            "unresolved",
            (
                "prefill_query",
                "prefill_key",
                "prefill_value",
                "prefill_beta",
                "prefill_decay",
            ),
        ),
        Producer("gdn_recurrent", "unresolved", ("recurrent_out",)),
        Producer("gdn_rmsnorm_gate", "unresolved", ("recurrent_bf16",)),
        Producer("ssm_out", "unresolved", ("attn_out",)),
    )


def linear_layer0_widths(
    *,
    hidden_size: int,
    linear_qkv_width: int,
    ssm_inner_size: int,
    ssm_group_count: int,
    ssm_state_size: int,
    ssm_time_step_rank: int,
) -> Mapping[str, tuple[str, int]]:
    """``(dtype, width)`` for each layer-0 linear-attention capture field.

    Widths come from the producer call sites: the QKV projection writes
    ``linear_qkv_width`` columns of bf16, the convolution kernel writes
    ``linear_qkv_width`` columns of f32, the normalized peer prepare writes
    ``ssm_group_count * ssm_state_size`` Q/K columns, the recurrent kernel and
    the RMSNorm-gate write ``ssm_inner_size`` columns, and the output
    projection writes ``hidden_size`` columns.
    """

    hidden = require_positive_int(hidden_size, what="hidden_size")
    qkv = require_positive_int(linear_qkv_width, what="linear_qkv_width")
    inner = require_positive_int(ssm_inner_size, what="ssm_inner_size")
    groups = require_positive_int(ssm_group_count, what="ssm_group_count")
    state = require_positive_int(ssm_state_size, what="ssm_state_size")
    heads = require_positive_int(ssm_time_step_rank, what="ssm_time_step_rank")
    return {
        "norm": (BF16, hidden),
        "linear_qkv": (BF16, qkv),
        "linear_z": (BF16, inner),
        "linear_alpha": (BF16, heads),
        "linear_beta": (BF16, heads),
        "linear_qkv_f32": (F32, qkv),
        "conv_out": (F32, qkv),
        "prefill_query": (F32, groups * state),
        "prefill_key": (F32, groups * state),
        "prefill_value": (F32, inner),
        "prefill_beta": (F32, heads),
        "prefill_decay": (F32, heads),
        "recurrent_out": (F32, inner),
        "recurrent_bf16": (BF16, inner),
        "attn_out": (BF16, hidden),
    }


def describe_layer0_linear(
    *,
    scratch,
    rows: int,
    hidden_size: int,
    linear_qkv_width: int,
    ssm_inner_size: int,
    ssm_group_count: int,
    ssm_state_size: int,
    ssm_time_step_rank: int,
    lifetimes: Mapping[str, Lifetime],
    producers: Sequence[Producer] | None = None,
) -> tuple[BufferDescriptor, ...]:
    """Build the layer-0 linear-attention descriptors from a live scratch.

    ``rows`` is the producer's actual row count (the helper's ``rows``
    argument), never the scratch capacity and never a count inferred from the
    data.  Every field's allocation size is read from the live ``DeviceBuffer``
    so an over-read fails closed here instead of silently capturing a
    neighbouring arena field.
    """

    rows_i = require_positive_int(rows, what="rows")
    widths = linear_layer0_widths(
        hidden_size=hidden_size,
        linear_qkv_width=linear_qkv_width,
        ssm_inner_size=ssm_inner_size,
        ssm_group_count=ssm_group_count,
        ssm_state_size=ssm_state_size,
        ssm_time_step_rank=ssm_time_step_rank,
    )
    producer_of = {
        buffer: producer.name
        for producer in (producers or linear_layer0_producers())
        for buffer in producer.buffers
    }
    out: list[BufferDescriptor] = []
    for name, (dtype, width) in widths.items():
        buffer = getattr(scratch, name, None)
        if buffer is None:
            raise CaptureError(f"scratch has no {name!r} field to capture")
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


def bulk_end_of_layer_capture_index(
    descriptors: Sequence[BufferDescriptor],
) -> Mapping[str, int]:
    """The (invalid) end-of-layer read position for every descriptor.

    Kept as an explicit helper so the diagnostic and its tests can prove that
    the historical "read every scratch field once after the helper returns"
    capture is rejected by :func:`validate_capture_order`.
    """

    last = len(descriptors)
    return {item.name: last for item in descriptors}


__all__ = [
    "BF16",
    "F32",
    "ITEMSIZE",
    "BufferDescriptor",
    "CaptureError",
    "CapturePlanError",
    "CaptureStep",
    "Producer",
    "aliasing_pairs",
    "build_capture_plan",
    "bulk_end_of_layer_capture_index",
    "describe_layer0_linear",
    "descriptor",
    "lifetimes_overlap",
    "linear_layer0_producers",
    "linear_layer0_widths",
    "require_positive_int",
    "validate_capture_order",
]
