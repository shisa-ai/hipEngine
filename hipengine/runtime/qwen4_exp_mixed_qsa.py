"""Depth-wave attention and index execution over one mixed target row layout.

:mod:`hipengine.runtime.qwen4_exp_mixed_layout` names one shared target pass:
which physical row, slot, owner token, position and token every request
contributes, and where its accepted prefix must be committed.
:mod:`hipengine.runtime.qwen4_exp_batch_qsa` runs the N-row QSA stages for rows
that are *independent* requests. This module is the adapter between the two: it
executes one attention/index layer of a mixed pass by splitting the layout into
depth waves, because the batch attention and index owners admit one row per
unique slot and therefore cannot be handed a chain as independent rows.

This is one layer's state stage, not mixed serving
--------------------------------------------------

Nothing here admits a request, owns a runner, drives GDN/conv/PLE state, samples,
or decides acceptance. A caller must already have computed the shared input
projections over every physical row, and must own the speculative verification
that produces the accepted-prefix counts. What this module adds is the wave
orchestration, its validation, and the append/commit/abort transaction around
one layer's attention state.

The shared stage is ``run_qwen4_exp_batch_qsa_input_projections``, which reads
only ``runtime``, ``rows``, ``q_width`` and ``kv_width`` from the plan it is
given, so a caller drives it over the pass with a plan whose ``rows`` is
``layout.total_rows`` — the row set it projects never has to be admitted on an
owner. This adapter then gathers those full-row slabs per wave, and checks they
are sized for the pass's row count before the first write.

Depth waves
-----------

Wave ``d`` is every physical row whose ``depth`` is ``d``, in ascending physical
row order, so a wave's row set is a subset of the layout's rows and never a
reinterpretation of them. Every request contributes exactly one root row, so
wave ``0`` holds every request and each later wave holds strictly fewer: the
widest wave is ``physical_request_width``, and the wave count is
``max_chain_rows``. Within one wave each slot appears once, which is the
uniqueness the owners require.

The chain is carried by the positions, not by the row set. Wave ``d`` publishes
``start_position + d`` for a request, so:

* wave ``0`` appends and attends the request's root row at its committed append
  position, reading the committed prefix of its slot;
* wave ``d > 0`` appends and attends one candidate row at the next position of
  the same slot, so its context length ``start_position + d + 1`` covers the
  prefix plus the earlier rows of its own chain, which earlier waves wrote.

A wave therefore never writes before its request's start position: appends are
the suffix ``[start_position, start_position + depth]`` and nothing else. A
request whose chain ends early simply stops appearing; its slot keeps the rows
it published, and no other slot is touched.

Per-wave stages, in order
-------------------------

1. Gather the wave's physical rows out of the caller's shared projections into
   the wave scratch (``hidden``, ``attn_q``, ``attn_k``, ``attn_v``, ``index_k``).
2. Publish the wave's slots and positions on the attention owner, and its slots
   on the index owner.
3. Admit the call with ``validate_qwen4_exp_batch_qsa_mixed_mixer``, which
   re-derives the dense/sparse split, re-checks cursor equality, and bounds every
   geometry before the first dispatch.
4. ``run_qwen4_exp_batch_qsa_attention``: index-key append, split norm/RoPE, KV
   write, sparse selection for the wave's sparse rows, rows attention, gate.
5. Scatter the wave's gated attention rows back to their physical positions in
   one full-row gated buffer.

After the last wave, one ``run_qwen4_exp_batch_qsa_output_projection`` call over
**every** physical row turns that buffer into the pass result. The projection is
per row, so projecting once after the scatter is the same arithmetic as
projecting each wave and scattering its output, with one GEMM instead of
``max_chain_rows`` and one extra ``[total_rows, q_width]`` slab. The output
projection reads its two slabs from ``scratch.output_scratch()`` because the
wave scratch's own ``gated`` is the per-wave staging buffer the gate stage
writes.

No attention, selection, normalization, gate, or projection formula is
reimplemented here: stages 4 and the pass-wide projection are the committed
mixed-mixer stages, called with a wave-sized scratch that exposes the
``Qwen4ExpQSAScratch`` attribute contract they read.

Costs, stated rather than hidden
--------------------------------

Per wave this module pays, on top of the mixer's own stages:

* five row gathers (``hidden``, ``attn_q``, ``attn_k``, ``attn_v``, ``index_k``)
  as one device-to-device copy per row per slab, so ``5 * total_rows`` small
  copies for a pass;
* one scatter of the wave's gated rows, so ``total_rows`` copies;
* one index-query projection per wave whose wave contains a sparse row. The
  mixer projects ``index_q`` from the gathered hidden for the whole wave
  whenever any row of that wave is past the dense-equivalence bound, so a wave
  with one sparse row and many dense rows pays a full-wave ``index_q`` GEMM.
  That is the mixer's own rule, reported here rather than replaced.

Across the pass: one ``attn_output`` projection over all rows, and one extra
``[total_rows, q_width]`` device slab to hold the scattered gate output.

The alternative — one shared attention pass — needs attention and index kernels
that take a repeated slot per row set, which the owners do not have.

Transaction: append, commit, abort
----------------------------------

``run_qwen4_exp_mixed_qsa_attention`` captures one
``Qwen4ExpBatchQsaIndexState.begin_rollback()`` token before wave ``0`` and
returns it on the run. Two outcomes consume it:

* :func:`commit_qwen4_exp_mixed_qsa_attention` publishes exactly each request's
  chosen depth. A request's index cursor becomes ``start_position + accepted + 1``
  — its root row plus the accepted candidates, and no rejected suffix — and its
  pooled-prefix counter is clamped to that cursor's complete blocks so a later
  step re-pools anything the rejection withdrew. Existing prefixes
  (``< start_position``) and slots outside the layout are untouched.
* :func:`abort_qwen4_exp_mixed_qsa_attention` applies the token, rewinding every
  layout slot's cursor and pooled counter to its pre-pass value, which is what
  restores logical visibility after a cancelled or failed verification.

A failure inside the run rolls the token back before re-raising, so a failed
pass never leaves a chain's keys visible. Device bytes are not restored, by the
same argument the index owner uses: every read is bounded by the counters, a
speculated append wrote strictly beyond the committed cursor, and the next step
rewrites those positions.

Not restored, and deliberately so: the attention owner's KV bytes past the
committed cursor (unreachable and rewritten next step) and both owners' active
row sets, which name the last wave of a pass that no longer exists. The caller
re-prepares the owners for its next step, exactly as the batch runner does.

Validation before mutation
--------------------------

:func:`validate_qwen4_exp_mixed_qsa_pass` refuses a pass before any device write
on: owner tokens (through the caller's ``owner_of`` callback, matched exactly
against each request), layout and slot bounds, cursor equality at every request's
start position, index capacity, runtime identity across the owners, head/index
geometry and projection shapes, shared-projection slab sizes, wave scratch
geometry, and the per-wave row sets (through the owners' own pure
``validate_rows``). The commit re-checks ownership and the plan before it writes
a counter, so a slot re-admitted between the pass and the commit is refused
instead of being committed.

Two things this module deliberately does not do: it takes no flag, and it
dispatches nothing by backend or quant — every device call is one of the
committed public wrappers the mixed mixer already uses. MRoPE position pointers
are not plumbed through the wave gather; a pass that needs them keeps its rows on
the existing single-row mixer.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hipengine.core.hip import HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    free,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaMixerPlan,
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_qsa_attention,
    run_qwen4_exp_batch_qsa_output_projection,
    validate_qwen4_exp_batch_qsa_mixed_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import (
    Qwen4ExpBatchQsaIndexRollback,
    Qwen4ExpBatchQsaIndexState,
)
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedLayout
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpQSAMixerDeviceWeights
from hipengine.speculative.interfaces import TargetCommitPlan

__all__ = [
    "Qwen4ExpMixedQsaOutputScratch",
    "Qwen4ExpMixedQsaPassPlan",
    "Qwen4ExpMixedQsaRun",
    "Qwen4ExpMixedQsaScratch",
    "Qwen4ExpMixedQsaWave",
    "abort_qwen4_exp_mixed_qsa_attention",
    "commit_qwen4_exp_mixed_qsa_attention",
    "plan_qwen4_exp_mixed_qsa_waves",
    "run_qwen4_exp_mixed_qsa_attention",
    "validate_qwen4_exp_mixed_qsa_pass",
]

# Same pre-write bounds the mixed mixer applies; a wave that breaks either would
# write KV bytes the decode could never read.
_DENSE_DECODE_BLOCK_SIZE = 256
_DENSE_DECODE_MAX_HEAD_DIM = 256
_MIXER_PROJECTIONS = ("attn_q", "attn_k", "attn_v", "attn_output", "index_q", "index_k")

# Wave scratch slabs, in allocation order. The first twelve names are exactly the
# attributes ``Qwen4ExpQSAScratch`` exposes to the mixer's stages, so the wave
# scratch is a drop-in for them; ``hidden_rows`` is this adapter's gather source,
# ``gated_rows`` collects every wave's gate output in physical row order, and
# ``output`` is the pass result the single full-row projection writes.
_WAVE_SLABS = (
    "hidden_rows",
    "q_projected",
    "key_projected",
    "value_projected",
    "index_q_projected",
    "index_k_projected",
    "index_query",
    "query",
    "key",
    "gate",
    "context",
    "gated",
    "gated_rows",
    "output",
)


# -- wave structure ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedQsaWave:
    """One depth wave: every physical row at ``depth``, in physical row order.

    ``slot_ids``, ``positions``, ``physical_rows`` and ``request_ids`` are
    parallel and in the wave's own row order, which is the order the attention
    and index owners are published with and the order the gather/scatter uses.
    """

    depth: int
    request_ids: tuple[int, ...]
    physical_rows: tuple[int, ...]
    slot_ids: tuple[int, ...]
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        depth = _non_negative(self.depth, "depth")
        object.__setattr__(self, "depth", depth)
        widths = {
            len(self.request_ids),
            len(self.physical_rows),
            len(self.slot_ids),
            len(self.positions),
        }
        if len(widths) != 1:
            raise ValueError("a wave needs one entry per row in every column")
        if not self.request_ids:
            raise ValueError(f"wave {depth} holds no rows")
        object.__setattr__(
            self, "request_ids", tuple(_non_negative(v, "request_id") for v in self.request_ids)
        )
        object.__setattr__(
            self,
            "physical_rows",
            tuple(_non_negative(v, "physical_row") for v in self.physical_rows),
        )
        object.__setattr__(
            self, "slot_ids", tuple(_non_negative(v, "slot") for v in self.slot_ids)
        )
        object.__setattr__(
            self, "positions", tuple(_non_negative(v, "position") for v in self.positions)
        )
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError(
                f"wave {depth} names a slot twice; the owners admit one row per slot"
            )
        if list(self.physical_rows) != sorted(self.physical_rows):
            raise ValueError(
                f"wave {depth} is not in physical row order; a wave is a subset of "
                "the layout's physical rows, not a reordering of them"
            )

    @property
    def rows(self) -> int:
        return len(self.physical_rows)


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedQsaPassPlan:
    """One admitted mixed pass: its waves plus the geometry they were checked at."""

    runtime: HipRuntime
    waves: tuple[Qwen4ExpMixedQsaWave, ...]
    owners: tuple[int, ...]
    q_width: int
    kv_width: int
    hidden: int
    index_heads: int
    index_dim: int
    stride: int

    @property
    def rows(self) -> int:
        """Physical rows in the pass."""

        return sum(wave.rows for wave in self.waves)

    @property
    def max_wave_rows(self) -> int:
        """Rows in the widest wave, which sizes the wave scratch."""

        return max(wave.rows for wave in self.waves)

    @property
    def root_rows(self) -> tuple[int, ...]:
        """Physical rows of the root wave, in wave order."""

        return self.waves[0].physical_rows

    def wave(self, depth: int) -> Qwen4ExpMixedQsaWave:
        parsed = _non_negative(depth, "depth")
        for wave in self.waves:
            if wave.depth == parsed:
                return wave
        raise KeyError(f"this pass has no wave {parsed}")

    def output_plan(self) -> Qwen4ExpBatchQsaMixerPlan:
        """The one full-row plan the pass-wide output projection runs under.

        ``run_qwen4_exp_batch_qsa_output_projection`` reads ``runtime``, ``rows``
        and ``q_width`` from the plan and its two slabs from the scratch, so the
        row set named here is never admitted on an owner and the row-kind and
        position fields are unused. ``rows`` is every physical row of the pass.
        """

        return Qwen4ExpBatchQsaMixerPlan(
            self.runtime,
            self.rows,
            self.q_width,
            self.kv_width,
            (),
            (),
            0,
        )


def plan_qwen4_exp_mixed_qsa_waves(
    layout: Qwen4ExpMixedLayout,
) -> tuple[Qwen4ExpMixedQsaWave, ...]:
    """Split a layout into depth waves without touching any state.

    Wave ``d`` is the layout's rows at depth ``d`` in ascending physical row
    order. Every request contributes a root row, so wave ``0`` is the full
    request width and each later wave is a strict subset of the one before it.
    """

    if not isinstance(layout, Qwen4ExpMixedLayout):
        raise TypeError("plan_qwen4_exp_mixed_qsa_waves requires a Qwen4ExpMixedLayout")
    waves: list[Qwen4ExpMixedQsaWave] = []
    for depth in range(layout.max_chain_rows):
        rows = tuple(row for row in layout.rows if row.depth == depth)
        waves.append(
            Qwen4ExpMixedQsaWave(
                depth=depth,
                request_ids=tuple(row.request_id for row in rows),
                physical_rows=tuple(row.physical_row for row in rows),
                slot_ids=tuple(row.slot for row in rows),
                positions=tuple(row.position for row in rows),
            )
        )
    return tuple(waves)


# -- wave scratch -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedQsaOutputScratch:
    """The two slabs ``run_qwen4_exp_batch_qsa_output_projection`` reads.

    The pass-wide projection is one call over every physical row, so it consumes
    the scattered full-row gate output and writes the pass result. The wave
    scratch's own ``gated`` is the per-wave staging buffer the gate stage writes,
    which cannot also be the scatter target, so this view names the two slabs the
    helper actually reads.
    """

    gated: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime


@dataclass
class Qwen4ExpMixedQsaScratch:
    """Wave-sized mixer slabs plus the two full-row pass buffers.

    ``q_projected`` through ``gated`` are the twelve slabs whose names and shapes
    ``Qwen4ExpQSAScratch`` gives the mixer's stages, so the committed stage
    functions run against this object unchanged. They are sized for the widest
    wave (one row per request), not for the pass's row count, because a wave
    never holds two rows of one request.

    ``hidden_rows`` is the gather target for the caller's shared input hidden, and
    the pointer the mixer projects wave index queries from. ``gated_rows`` is
    ``[total_rows, q_width]`` and collects each wave's gate output at its rows'
    physical positions; ``output`` is ``[total_rows, hidden]``, the pass result
    written by the single full-row output projection after the last wave.
    ``output_scratch()`` names those two for that projection.
    """

    hidden_rows: DeviceBuffer
    q_projected: DeviceBuffer
    key_projected: DeviceBuffer
    value_projected: DeviceBuffer
    index_q_projected: DeviceBuffer
    index_k_projected: DeviceBuffer
    index_query: DeviceBuffer
    query: DeviceBuffer
    key: DeviceBuffer
    gate: DeviceBuffer
    context: DeviceBuffer
    gated: DeviceBuffer
    gated_rows: DeviceBuffer
    output: DeviceBuffer
    selection: Qwen4ExpBatchQsaSelectionScratch
    rows: int
    total_rows: int
    hidden: int
    q_width: int
    kv_width: int
    index_heads: int
    index_dim: int
    stride: int
    runtime: HipRuntime
    closed: bool = False
    closing: bool = False
    _pending_free: list[DeviceBuffer] = field(default_factory=list)

    def output_scratch(self) -> Qwen4ExpMixedQsaOutputScratch:
        """The full-row slabs one pass-wide output projection reads and writes."""

        self._require_open()
        return Qwen4ExpMixedQsaOutputScratch(
            gated=self.gated_rows,
            output=self.output,
            runtime=self.runtime,
        )

    @classmethod
    def allocate(
        cls,
        *,
        layout: Qwen4ExpMixedLayout,
        attention_state: Qwen4ExpBatchAttentionState,
        index_state: Qwen4ExpBatchQsaIndexState,
        hidden: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpMixedQsaScratch":
        """Allocate the wave slabs and the per-wave selection scratch.

        ``rows`` is the layout's physical request width — the widest wave — and
        ``total_rows`` is the layout's row count, which sizes the result buffer.
        Every geometry check and every byte product runs before the first
        allocation, and a failed allocation frees what it took.
        """

        if not isinstance(layout, Qwen4ExpMixedLayout):
            raise TypeError("Qwen4ExpMixedQsaScratch.allocate requires a Qwen4ExpMixedLayout")
        _require_open_owners(attention_state, index_state)
        if index_state.attention is not attention_state:
            raise ValueError(
                "QSA index state must mirror the batch attention state passed to the "
                "mixed QSA adapter"
            )
        active_runtime = runtime or attention_state.runtime
        if (
            active_runtime is not attention_state.runtime
            or active_runtime is not index_state.runtime
        ):
            raise ValueError(
                "runtime must match the mixed QSA attention and index owners"
            )
        rows = layout.physical_request_width
        total_rows = layout.total_rows
        parsed_hidden = _positive(hidden, "hidden")
        parsed_query_heads = _positive(query_heads, "query_heads")
        parsed_kv_heads = _positive(kv_heads, "kv_heads")
        parsed_head_dim = _positive(head_dim, "head_dim")
        if parsed_query_heads % parsed_kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if parsed_head_dim > _DENSE_DECODE_MAX_HEAD_DIM:
            raise ValueError(
                f"head_dim {parsed_head_dim} exceeds the dense decode bound "
                f"{_DENSE_DECODE_MAX_HEAD_DIM}"
            )
        if (
            parsed_kv_heads != attention_state.kv_heads
            or parsed_head_dim != attention_state.head_dim
        ):
            raise ValueError(
                "QSA head geometry must match the attention state cache "
                f"({attention_state.kv_heads} kv heads x {attention_state.head_dim})"
            )
        q_width = parsed_query_heads * parsed_head_dim
        kv_width = parsed_kv_heads * parsed_head_dim
        index_heads = int(index_state.index_heads)
        index_dim = int(index_state.index_dim)
        stride = int(index_state.output_stride)
        elements = {
            "hidden_rows": rows * parsed_hidden,
            "q_projected": rows * q_width * 2,
            "key_projected": rows * kv_width,
            "value_projected": rows * kv_width,
            "index_q_projected": rows * index_heads * index_dim,
            "index_k_projected": rows * index_dim,
            "index_query": rows * index_heads * index_dim,
            "query": rows * q_width,
            "key": rows * kv_width,
            "gate": rows * q_width,
            "context": rows * q_width,
            "gated": rows * q_width,
            "gated_rows": total_rows * q_width,
            "output": total_rows * parsed_hidden,
        }
        _require_addressable(elements)
        buffers: list[DeviceBuffer] = []
        try:
            for name in _WAVE_SLABS:
                buffers.append(malloc(elements[name] * 4, runtime=active_runtime))
        except BaseException as error:
            _free_after_failure(buffers, active_runtime, error, "wave scratch allocation")
            raise
        try:
            selection = Qwen4ExpBatchQsaSelectionScratch.allocate(
                rows=rows,
                stride=stride,
                index_heads=index_heads,
                index_dim=index_dim,
                runtime=active_runtime,
            )
        except BaseException as error:
            _free_after_failure(buffers, active_runtime, error, "wave scratch allocation")
            raise
        slabs = dict(zip(_WAVE_SLABS, buffers, strict=True))
        return cls(
            **slabs,
            selection=selection,
            rows=rows,
            total_rows=total_rows,
            hidden=parsed_hidden,
            q_width=q_width,
            kv_width=kv_width,
            index_heads=index_heads,
            index_dim=index_dim,
            stride=stride,
            runtime=active_runtime,
            _pending_free=list(buffers),
        )

    def close(self) -> None:
        """Release the selection scratch, then the wave slabs, newest first.

        Retry-safe: the selection owner pops each of its buffers only after its
        free returns, and this owner pops each slab only after its free returns,
        so a close that fails part-way releases the remainder exactly once on the
        next attempt instead of leaking or double-freeing it. ``closing`` is set
        before the first free, so a partially closed scratch refuses to be used
        as a mixer scratch instead of handing out a released address.
        """

        if self.closed:
            return
        self.closing = True
        if not self.selection.closed:
            self.selection.close()
        pending = self._pending_free
        if not pending:
            pending.extend(getattr(self, name) for name in _WAVE_SLABS)
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpMixedQsaScratch":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp mixed QSA wave scratch is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp mixed QSA wave scratch is closing; its buffers may already "
                "be released"
            )


# -- validation -------------------------------------------------------------


def validate_qwen4_exp_mixed_qsa_pass(
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    layout: Qwen4ExpMixedLayout,
    shared: Any,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    scratch: Qwen4ExpMixedQsaScratch,
    owner_of: Callable[[int], int],
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    attention_context_limit: int | None = None,
    eps: float = 1e-6,
    runtime: HipRuntime | None = None,
) -> Qwen4ExpMixedQsaPassPlan:
    """Admit one mixed pass before any wave writes a byte.

    ``shared`` is the caller's full-row input-projection scratch (the
    ``Qwen4ExpQSAScratch`` it ran
    ``run_qwen4_exp_batch_qsa_input_projections`` into): this adapter gathers
    ``q_projected``, ``key_projected``, ``value_projected`` and
    ``index_k_projected`` rows out of it, so each slab must cover every physical
    row. ``owner_of(slot)`` must return the slot's current owner token; a layout
    naming a stale admission is refused here, before wave ``0``.

    Nothing in this function writes a host mirror, a device buffer, or a counter:
    the per-wave row sets are checked through the owners' own pure
    ``validate_rows``, and the cursor check reads the index owner's host mirror.
    """

    if not isinstance(layout, Qwen4ExpMixedLayout):
        raise TypeError("a mixed QSA pass requires a Qwen4ExpMixedLayout")
    if not isinstance(scratch, Qwen4ExpMixedQsaScratch):
        raise TypeError("a mixed QSA pass requires a Qwen4ExpMixedQsaScratch")
    scratch._require_open()
    _require_open_owners(attention_state, index_state)
    if index_state.attention is not attention_state:
        raise ValueError(
            "QSA index state must mirror the batch attention state passed to the "
            "mixed QSA adapter"
        )
    if index_state.pending_rollback is not None:
        raise RuntimeError(
            "the QSA index owner already holds a pending rollback token; roll it "
            "back, discard it, or reset the affected slot before running a pass"
        )
    active_runtime = runtime or scratch.runtime
    if (
        active_runtime is not scratch.runtime
        or active_runtime is not attention_state.runtime
        or active_runtime is not index_state.runtime
    ):
        raise ValueError(
            "runtime must match the mixed QSA wave scratch and both state owners"
        )

    parsed_hidden = _positive(hidden, "hidden")
    parsed_query_heads = _positive(query_heads, "query_heads")
    parsed_kv_heads = _positive(kv_heads, "kv_heads")
    parsed_head_dim = _positive(head_dim, "head_dim")
    parsed_rotary = _positive(rotary_dim, "rotary_dim")
    if parsed_rotary > parsed_head_dim or parsed_rotary % 2:
        raise ValueError("rotary_dim must be even and <= head_dim")
    if parsed_query_heads % parsed_kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if parsed_head_dim > _DENSE_DECODE_MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim {parsed_head_dim} exceeds the dense decode bound "
            f"{_DENSE_DECODE_MAX_HEAD_DIM}"
        )
    if not _is_finite_positive(theta, "theta"):
        raise ValueError("theta must be a finite positive number")
    if not _is_finite_positive(eps, "eps"):
        raise ValueError("eps must be a finite positive number")
    if (
        parsed_kv_heads != attention_state.kv_heads
        or parsed_head_dim != attention_state.head_dim
    ):
        raise ValueError(
            "QSA head geometry must match the attention state cache "
            f"({attention_state.kv_heads} kv heads x {attention_state.head_dim})"
        )
    if attention_state.block_size != _DENSE_DECODE_BLOCK_SIZE:
        raise ValueError(
            "dense QSA decode requires block_size "
            f"{_DENSE_DECODE_BLOCK_SIZE}, state has {attention_state.block_size}"
        )
    q_width = parsed_query_heads * parsed_head_dim
    kv_width = parsed_kv_heads * parsed_head_dim

    parsed_index_heads = _positive(index_heads, "index_heads")
    parsed_index_dim = _positive(index_dim, "index_dim")
    parsed_index_rotary = _positive(index_rotary_dim, "index_rotary_dim")
    if parsed_index_rotary > parsed_index_dim or parsed_index_rotary % 2:
        raise ValueError("index_rotary_dim must be even and <= index_dim")
    if (
        parsed_index_heads != index_state.index_heads
        or parsed_index_dim != index_state.index_dim
    ):
        raise ValueError(
            "QSA index geometry must match its state owner "
            f"({index_state.index_heads} heads x {index_state.index_dim})"
        )

    _require_weight_geometry(
        weights,
        hidden=parsed_hidden,
        q_width=q_width,
        kv_width=kv_width,
        index_heads=parsed_index_heads,
        index_dim=parsed_index_dim,
    )
    _positive(mixed_ptr, "mixed_ptr")
    _require_shared_projections(
        shared,
        runtime=active_runtime,
        total_rows=layout.total_rows,
        q_width=q_width,
        kv_width=kv_width,
        index_dim=parsed_index_dim,
    )

    if scratch.total_rows != layout.total_rows:
        raise ValueError(
            f"wave scratch holds {scratch.total_rows} result rows but the layout has "
            f"{layout.total_rows}"
        )
    for name, expected in (
        ("hidden", parsed_hidden),
        ("q_width", q_width),
        ("kv_width", kv_width),
        ("index_heads", parsed_index_heads),
        ("index_dim", parsed_index_dim),
        ("stride", int(index_state.output_stride)),
    ):
        actual = getattr(scratch, name)
        if actual != expected:
            raise ValueError(
                f"wave scratch {name} is {actual}, this pass needs {expected}; "
                "allocate the scratch for this head and index geometry"
            )
    if scratch.rows < layout.physical_request_width:
        raise ValueError(
            f"wave scratch holds {scratch.rows} rows, the widest wave needs "
            f"{layout.physical_request_width}"
        )

    waves = plan_qwen4_exp_mixed_qsa_waves(layout)
    max_position = max(row.position for row in layout.rows)
    if max_position >= attention_state.max_positions:
        raise ValueError(
            f"row position {max_position} is outside the attention capacity "
            f"[0, {attention_state.max_positions})"
        )
    if attention_context_limit is not None:
        limit = _positive(attention_context_limit, "attention_context_limit")
        if limit < max_position + 1:
            raise ValueError(
                f"attention context limit {limit} does not cover the pass's widest "
                f"context {max_position + 1}"
            )
        if limit > attention_state.max_positions:
            raise ValueError(
                f"attention context limit {limit} exceeds the attention capacity "
                f"{attention_state.max_positions}"
            )

    owners = _require_owner_tokens(layout, owner_of)
    _require_start_cursors(layout, index_state)
    for wave in waves:
        # Both owners expose a pure row-plan validator, so the whole pass is
        # checked for bounds, duplicate slots and block capacity without
        # publishing a row set.
        attention_state.validate_rows(wave.slot_ids, wave.positions)
        index_state.validate_rows(wave.slot_ids)
        deepest = max(wave.positions) + 1
        if deepest > index_state.capacity:
            raise ValueError(
                f"wave {wave.depth} appends at position {deepest - 1}, past the "
                f"per-request index capacity {index_state.capacity}"
            )

    return Qwen4ExpMixedQsaPassPlan(
        runtime=active_runtime,
        waves=waves,
        owners=owners,
        q_width=q_width,
        kv_width=kv_width,
        hidden=parsed_hidden,
        index_heads=parsed_index_heads,
        index_dim=parsed_index_dim,
        stride=int(index_state.output_stride),
    )


# -- the pass ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedQsaRun:
    """One completed mixed pass, holding the rollback token its outcome consumes.

    ``output`` is the pass-wide ``[total_rows, hidden]`` attention result in
    physical row order, written by the one output projection that runs after the
    last wave. Its contents are only meaningful while the run is pending: a
    commit or an abort ends the pass, and a caller that discards the run without
    either leaves the token pending (the index owner refuses a second one until
    it is resolved).
    """

    layout: Qwen4ExpMixedLayout
    plan: Qwen4ExpMixedQsaPassPlan
    rollback: Qwen4ExpBatchQsaIndexRollback
    output: DeviceBuffer
    wave_rows: tuple[int, ...]
    index_query_rows: tuple[int, ...]

    @property
    def waves(self) -> int:
        return len(self.plan.waves)

    @property
    def rows(self) -> int:
        return sum(self.wave_rows)

    @property
    def index_query_total(self) -> int:
        """Rows whose wave projected index queries (the mixer's own cost rule)."""

        return sum(self.index_query_rows)


def run_qwen4_exp_mixed_qsa_attention(
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    layout: Qwen4ExpMixedLayout,
    shared: Any,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    scratch: Qwen4ExpMixedQsaScratch,
    owner_of: Callable[[int], int],
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    attention_context_limit: int | None = None,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> Qwen4ExpMixedQsaRun:
    """Run one layer's attention and index stages over a mixed layout.

    The caller must already have run the shared input projections over **every**
    physical row into ``shared``. This call then walks the layout's depth waves in
    ascending depth, gathering each wave's rows, publishing that wave's unique
    slots and positions, running the committed mixed-mixer attention stage, and
    scattering the wave's gated rows to their physical positions. After the last
    wave it runs one output projection over every physical row.

    Returns a run holding the pass-wide output and the index rollback token
    captured before wave ``0``; exactly one of
    :func:`commit_qwen4_exp_mixed_qsa_attention` and
    :func:`abort_qwen4_exp_mixed_qsa_attention` consumes it. A failure anywhere
    in the pass, including the final projection, rolls the token back before
    re-raising, so no rejected row stays visible.
    """

    plan = validate_qwen4_exp_mixed_qsa_pass(
        mixed_ptr,
        weights,
        layout=layout,
        shared=shared,
        attention_state=attention_state,
        index_state=index_state,
        scratch=scratch,
        owner_of=owner_of,
        hidden=hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        attention_context_limit=attention_context_limit,
        eps=eps,
        runtime=runtime,
    )
    token = index_state.begin_rollback()
    wave_rows: list[int] = []
    index_query_rows: list[int] = []
    try:
        for wave in plan.waves:
            wave_plan = _run_wave(
                wave,
                plan=plan,
                weights=weights,
                mixed_ptr=mixed_ptr,
                shared=shared,
                attention_state=attention_state,
                index_state=index_state,
                scratch=scratch,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                theta=theta,
                index_heads=index_heads,
                index_dim=index_dim,
                index_rotary_dim=index_rotary_dim,
                attention_context_limit=attention_context_limit,
                eps=eps,
                stream=stream,
            )
            wave_rows.append(wave_plan.rows)
            index_query_rows.append(
                wave_plan.rows if wave_plan.sparse_rows else 0
            )
        # One output projection for the whole pass: every wave's gate output is
        # already in physical row order, so this is the same arithmetic as one
        # projection per wave with one launch instead of ``max_chain_rows``.
        run_qwen4_exp_batch_qsa_output_projection(
            plan.output_plan(),
            weights,
            scratch=scratch.output_scratch(),
            hidden=plan.hidden,
            stream=stream,
        )
    except BaseException as error:
        try:
            index_state.rollback(token)
        except Exception as cleanup_error:  # pragma: no cover - defensive
            error.add_note(
                "rolling back the failed mixed pass also failed, so the index "
                f"cursors may still cover the rejected rows: {cleanup_error!r}"
            )
        raise
    return Qwen4ExpMixedQsaRun(
        layout=layout,
        plan=plan,
        rollback=token,
        output=scratch.output,
        wave_rows=tuple(wave_rows),
        index_query_rows=tuple(index_query_rows),
    )


def commit_qwen4_exp_mixed_qsa_attention(
    run: Qwen4ExpMixedQsaRun,
    accepted_counts: Sequence[int],
    *,
    index_state: Qwen4ExpBatchQsaIndexState,
    owner_of: Callable[[int], int],
    next_tokens: Sequence[int | None] | None = None,
    transaction_id: int = 0,
) -> TargetCommitPlan:
    """Publish exactly each request's accepted depth and withdraw the rest.

    The layout owns the destination arithmetic
    (``Qwen4ExpMixedLayout.commit_destinations``), so this function adds only the
    index-owner transaction: it re-checks every slot's owner token, checks that
    each cursor still sits where the pass left it, validates the whole plan, and
    only then rewrites the layout's cursors and pooled-prefix counters. A
    request's cursor becomes ``start_position + accepted + 1``, so its committed
    prefix is visible and its rejected suffix is not; prefixes before
    ``start_position`` and slots outside the layout are not touched.

    Nothing is written if any check refuses, and the pending rollback token stays
    usable until the counters are validated, so a refused commit can still be
    aborted.
    """

    _require_run(run)
    _require_commit_owner(run, index_state)
    commit = run.layout.commit_destinations(
        accepted_counts,
        transaction_id=transaction_id,
        next_tokens=next_tokens,
    )
    owners = _require_owner_tokens(run.layout, owner_of)
    if owners != run.plan.owners:
        raise ValueError(
            "slot ownership changed between the mixed pass and its commit: "
            f"{run.plan.owners} became {owners}"
        )
    slots, cursors, pooled = _commit_counters(run, index_state, commit)
    if not index_state.discard_rollback():
        raise RuntimeError(
            "the mixed pass's rollback token was already resolved; refusing to "
            "commit a pass whose index cursors another caller may own"
        )
    index_state.cursor_host[slots] = cursors
    index_state.pooled_count_host[slots] = pooled
    index_state.active = None
    return commit


def abort_qwen4_exp_mixed_qsa_attention(
    run: Qwen4ExpMixedQsaRun,
    *,
    index_state: Qwen4ExpBatchQsaIndexState,
) -> None:
    """Rewind the pass: restore every layout slot's index cursors and visibility.

    Applies the run's rollback token, which rewinds the cursor and the
    pooled-prefix counter of every slot the pass appended to, so the whole
    speculative suffix becomes unreadable again. Device bytes are left in place;
    every read is bounded by the counters and the next step rewrites those
    positions. The attention owner's active row set is not restored — it is
    metadata naming a pass that no longer exists, and the caller re-prepares it.
    """

    _require_run(run)
    _require_commit_owner(run, index_state)
    index_state.rollback(run.rollback)


# -- internals --------------------------------------------------------------


def _run_wave(
    wave: Qwen4ExpMixedQsaWave,
    *,
    plan: Qwen4ExpMixedQsaPassPlan,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    mixed_ptr: int,
    shared: Any,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    scratch: Qwen4ExpMixedQsaScratch,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    attention_context_limit: int | None,
    eps: float,
    stream: int,
) -> Qwen4ExpBatchQsaMixerPlan:
    """Gather, publish, attend, and scatter one depth wave."""

    rows = wave.rows
    hidden_bytes = plan.hidden * 4
    _gather_rows(
        plan.runtime,
        source_ptr=mixed_ptr,
        destination=scratch.hidden_rows,
        physical_rows=wave.physical_rows,
        row_bytes=hidden_bytes,
    )
    for source, destination, row_bytes in (
        (shared.q_projected, scratch.q_projected, plan.q_width * 2 * 4),
        (shared.key_projected, scratch.key_projected, plan.kv_width * 4),
        (shared.value_projected, scratch.value_projected, plan.kv_width * 4),
        (shared.index_k_projected, scratch.index_k_projected, plan.index_dim * 4),
    ):
        _gather_rows(
            plan.runtime,
            source_ptr=source.ptr,
            destination=destination,
            physical_rows=wave.physical_rows,
            row_bytes=row_bytes,
        )
    # Publishing the wave is what makes the chain work: the position is the
    # request's own next append position, so this row's context covers the prefix
    # and the earlier rows of its chain, which earlier waves wrote.
    attention_state.prepare(wave.slot_ids, wave.positions)
    index_state.prepare(wave.slot_ids)
    wave_plan = validate_qwen4_exp_batch_qsa_mixed_mixer(
        weights,
        attention_state=attention_state,
        index_state=index_state,
        selection=scratch.selection,
        scratch=scratch,
        rows=rows,
        hidden=plan.hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        attention_context_limit=attention_context_limit,
        eps=eps,
        runtime=plan.runtime,
    )
    run_qwen4_exp_batch_qsa_attention(
        wave_plan,
        scratch.hidden_rows.ptr,
        weights,
        attention_state=attention_state,
        index_state=index_state,
        selection=scratch.selection,
        scratch=scratch,
        hidden=plan.hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        eps=eps,
        stream=stream,
    )
    # The gate output is per row, so it is scattered to its physical positions
    # here and projected once for the whole pass after the last wave.
    _scatter_rows(
        plan.runtime,
        source=scratch.gated,
        destination=scratch.gated_rows,
        physical_rows=wave.physical_rows,
        row_bytes=plan.q_width * 4,
    )
    return wave_plan


def _gather_rows(
    runtime: HipRuntime,
    *,
    source_ptr: int,
    destination: DeviceBuffer,
    physical_rows: Sequence[int],
    row_bytes: int,
) -> None:
    """Copy ``physical_rows`` out of a full-row slab into wave row order."""

    for index, physical in enumerate(physical_rows):
        runtime.memcpy(
            destination.ptr + index * row_bytes,
            source_ptr + physical * row_bytes,
            row_bytes,
            MemcpyKind.DEVICE_TO_DEVICE,
        )


def _scatter_rows(
    runtime: HipRuntime,
    *,
    source: DeviceBuffer,
    destination: DeviceBuffer,
    physical_rows: Sequence[int],
    row_bytes: int,
) -> None:
    """Copy wave rows to their physical positions in a full-row slab."""

    for index, physical in enumerate(physical_rows):
        runtime.memcpy(
            destination.ptr + physical * row_bytes,
            source.ptr + index * row_bytes,
            row_bytes,
            MemcpyKind.DEVICE_TO_DEVICE,
        )


def _commit_counters(
    run: Qwen4ExpMixedQsaRun,
    index_state: Qwen4ExpBatchQsaIndexState,
    commit: TargetCommitPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the committed cursors and pooled counters, validating all of them.

    Every value is computed and checked into fresh arrays before the caller
    writes anything, so a refusal cannot leave half a commit behind. A cursor
    that is not exactly where the pass left it means another caller appended to
    or reset that slot, and the run is stale rather than committable.
    """

    layout = run.layout
    requests = layout.requests
    accepted = commit.accepted_counts
    if len(accepted) != layout.physical_request_width:
        raise ValueError(
            f"commit plan holds {len(accepted)} accepted counts for "
            f"{layout.physical_request_width} request(s)"
        )
    ratio = int(index_state.compression_ratio)
    slots = np.asarray([request.slot for request in requests], dtype=np.int64)
    current = index_state.cursor_host[slots]
    pooled = index_state.pooled_count_host[slots]
    cursors = np.empty(len(requests), dtype=np.int64)
    counters = np.empty(len(requests), dtype=np.int64)
    for index, (request, keep) in enumerate(zip(requests, accepted, strict=True)):
        cursor = int(current[index])
        expected = request.start_position + request.depth + 1
        if cursor != expected:
            raise ValueError(
                f"slot {request.slot} index cursor {cursor} is not where the mixed "
                f"pass left it ({expected}); the run is stale because the slot was "
                "appended to or reset since"
            )
        destination = request.start_position + int(keep) + 1
        if not request.start_position < destination <= cursor:
            raise ValueError(
                f"request {request.request_id} accepted {keep} of {request.depth} "
                f"candidate(s), which commits cursor {destination} outside "
                f"({request.start_position}, {cursor}]"
            )
        if commit.commit_positions[index] != destination - 1:
            raise ValueError(
                f"commit plan position {commit.commit_positions[index]} does not "
                f"match request {request.request_id}'s accepted depth {keep}"
            )
        cursors[index] = destination
        # A pooled block whose members run past the committed cursor was pooled
        # from a rejected row, so it must be pooled again by a later step.
        counters[index] = min(int(pooled[index]), destination // ratio)
    return slots, cursors, counters


def _require_run(run: object) -> Qwen4ExpMixedQsaRun:
    if not isinstance(run, Qwen4ExpMixedQsaRun):
        raise TypeError("this call requires a Qwen4ExpMixedQsaRun")
    return run


def _require_commit_owner(
    run: Qwen4ExpMixedQsaRun, index_state: Qwen4ExpBatchQsaIndexState
) -> None:
    """Refuse a commit or abort aimed at an owner that did not run the pass."""

    if not isinstance(index_state, Qwen4ExpBatchQsaIndexState):
        raise TypeError("this call requires the Qwen4ExpBatchQsaIndexState that ran it")
    if run.rollback.owner != index_state.identity:
        raise ValueError(
            "the mixed pass was run on a different index owner: the token names "
            f"owner {run.rollback.owner}, this owner is {index_state.identity}"
        )
    if index_state.pending_rollback is not run.rollback:
        raise RuntimeError(
            "the mixed pass's rollback token is no longer pending; the pass was "
            "already committed or aborted"
        )


def _require_owner_tokens(
    layout: Qwen4ExpMixedLayout, owner_of: Callable[[int], int]
) -> tuple[int, ...]:
    """Check every request's admission token through the caller's lookup.

    The token is compared exactly, without coercion: a callback returning a
    bool, a numpy scalar, or a different admission is refused, because a layout
    that names a stale admission would append into another request's chain.
    """

    if not callable(owner_of):
        raise TypeError("owner_of must be a callable returning a slot's owner token")
    tokens: list[int] = []
    for request in layout.requests:
        token = owner_of(request.slot)
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError(
                f"owner_of({request.slot}) returned {type(token).__name__}; an owner "
                "token is an int"
            )
        if token != request.owner_token:
            raise ValueError(
                f"slot {request.slot} is owned by token {token} but request "
                f"{request.request_id} names owner token {request.owner_token}; the "
                "layout is stale for this slot"
            )
        tokens.append(int(token))
    return tuple(tokens)


def _require_start_cursors(
    layout: Qwen4ExpMixedLayout, index_state: Qwen4ExpBatchQsaIndexState
) -> None:
    """Require every slot's index cursor to sit at its request's start position.

    A pass appends the suffix at and after that cursor, so a cursor that is past
    it would overwrite a committed key and a cursor that is behind it would leave
    a hole the chain's context reads as garbage. Both are refused here, before
    wave ``0``.
    """

    slots = np.asarray([request.slot for request in layout.requests], dtype=np.int64)
    cursors = index_state.cursor_host[slots]
    for index, request in enumerate(layout.requests):
        cursor = int(cursors[index])
        if cursor != request.start_position:
            raise ValueError(
                f"slot {request.slot} index cursor {cursor} does not match request "
                f"{request.request_id}'s start position {request.start_position}; a "
                "mixed pass appends the suffix at and after that cursor"
            )


def _require_shared_projections(
    shared: Any,
    *,
    runtime: HipRuntime,
    total_rows: int,
    q_width: int,
    kv_width: int,
    index_dim: int,
) -> None:
    """Refuse a shared-projection owner that cannot cover every physical row."""

    if getattr(shared, "closed", False):
        raise RuntimeError("the shared projection scratch is closed")
    if getattr(shared, "closing", False):
        raise RuntimeError(
            "the shared projection scratch is closing; its buffers may already be "
            "released"
        )
    shared_runtime = getattr(shared, "runtime", None)
    if shared_runtime is not None and shared_runtime is not runtime:
        raise ValueError(
            "the shared projection scratch was allocated on a different runtime"
        )
    required = {
        "q_projected": total_rows * q_width * 2 * 4,
        "key_projected": total_rows * kv_width * 4,
        "value_projected": total_rows * kv_width * 4,
        "index_k_projected": total_rows * index_dim * 4,
    }
    for name, needed in required.items():
        buffer = getattr(shared, name, None)
        if buffer is None:
            raise TypeError(
                f"the shared projection scratch has no {name}; run the shared input "
                "projections over every physical row first"
            )
        if int(buffer.nbytes) < needed:
            raise ValueError(
                f"shared projection scratch {name} holds {int(buffer.nbytes)} bytes, "
                f"needs {needed} for {total_rows} physical rows"
            )


def _require_weight_geometry(
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    hidden: int,
    q_width: int,
    kv_width: int,
    index_heads: int,
    index_dim: int,
) -> None:
    """Refuse missing weights, a zero norm pointer, or a mis-shaped projection.

    The mixer repeats these checks per wave; doing them once for the pass means a
    malformed layer is refused before wave ``0`` publishes anything.
    """

    projections = getattr(weights, "projections", None)
    if projections is None:
        raise TypeError("mixed QSA weights must expose a projections mapping")
    missing = sorted(set(_MIXER_PROJECTIONS) - set(projections))
    if missing:
        raise ValueError("missing Qwen4Exp QSA weights: " + ", ".join(missing))
    for name in ("q_norm_weight_ptr", "k_norm_weight_ptr"):
        if int(getattr(weights, name, 0)) == 0:
            raise ValueError("Qwen4Exp QSA Q/K norm weights are required")
    for name in ("index_q_norm_weight_ptr", "index_k_norm_weight_ptr"):
        if int(getattr(weights, name, 0)) == 0:
            raise ValueError("Qwen4Exp QSA index Q/K norm weights are required")
    expected = {
        "attn_q": (q_width * 2, hidden),
        "attn_k": (kv_width, hidden),
        "attn_v": (kv_width, hidden),
        "attn_output": (hidden, q_width),
        "index_q": (index_heads * index_dim, hidden),
        "index_k": (index_dim, hidden),
    }
    for slot, shape in expected.items():
        spec = getattr(projections[slot], "spec", None)
        source_shape = getattr(getattr(spec, "source", None), "shape", None)
        if source_shape is None:
            continue
        parsed = tuple(int(dimension) for dimension in source_shape)
        if parsed != shape:
            raise ValueError(
                f"{slot} weight shape {parsed} does not match the expected "
                f"{shape} projection geometry"
            )


def _require_open_owners(
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
) -> None:
    if attention_state.closed or index_state.closed:
        raise RuntimeError("Qwen4Exp mixed QSA owners are closed")
    if attention_state.closing or index_state.closing:
        raise RuntimeError(
            "Qwen4Exp mixed QSA owners are closing; their buffers may already be "
            "released"
        )


def _require_addressable(elements: dict[str, int]) -> None:
    limit = np.iinfo(np.int64).max
    for name, count in elements.items():
        nbytes = count * 4
        if nbytes <= 0 or nbytes > limit:
            raise ValueError(f"wave scratch {name} byte size {nbytes} is not addressable")


def _free_pending(pending: list[DeviceBuffer], runtime: HipRuntime) -> None:
    """Free ``pending`` newest-first, popping only after each free returns."""

    while pending:
        free(pending[-1], runtime=runtime)
        pending.pop()


def _free_after_failure(
    buffers: list[DeviceBuffer],
    runtime: HipRuntime,
    error: BaseException,
    context: str,
) -> None:
    """Release partially allocated slabs, keeping the original failure primary."""

    try:
        _free_pending(buffers, runtime)
    except Exception as cleanup_error:  # pragma: no cover - defensive
        error.add_note(
            f"cleanup after the failed {context} also failed: {cleanup_error!r}"
        )


def _non_negative(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a bool")
    try:
        parsed = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(parsed)


def _positive(value: object, name: str) -> int:
    parsed = _non_negative(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _is_finite_positive(value: object, name: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        parsed = float(value)
    except OverflowError:
        return False
    return math.isfinite(parsed) and parsed > 0.0
