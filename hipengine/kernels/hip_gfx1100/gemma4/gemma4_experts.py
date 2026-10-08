"""Orchestrated Gemma 4 routed-expert forward on gfx1100.

Assembles the expert block from kernels that are each tested on their own:

    compact lanes per expert   qwen35_moe_group_compact_active (parallel)
    gather hidden rows         qwen35_moe_gather_packed_hidden_lowp
    per-expert gate_up GEMV    dense_gemv_out_bf16
    GeGLU                      gemma4_gelu_tanh_mul_bf16
    per-expert down GEMV       dense_gemv_out_bf16
    weighted accumulate        gemma4_moe_weighted_accumulate_bf16

The per-expert GEMVs are one launch per non-empty expert with a pointer offset
into the stacked expert weights, not one launch per token-lane. Compaction makes
each expert's rows contiguous, which is what makes that possible.

One host synchronisation is unavoidable here: the per-expert row counts live in
``expert_start`` on the device, and the launch count depends on them. The whole
prefix is read back in one copy (``num_experts + 1`` int64 values) rather than
read per expert. A device-resident dispatch would remove the sync but is a
separate optimisation; this path is correctness-first.

Quantised GGUF expert weights resolve a kernel from the registry by quant key.
Decode and small lane counts use the selected-GEMV family, which launches one
block per (out_col, lane). A prefill block has enough compact rows per expert
that re-reading the expert's weight matrix once per lane dominates: measured at
84.8% of the 512-token prefill device time on gfx1151 (Q4_K gate/up 53.4%,
Q5_1 down 31.4%). Above one lane per expert the projection therefore prefers a
grouped-prefill family, where one launch per (expert, out_col) reads that
expert's weight row once and reuses it across the expert's rows. That is a
capability the quant key either declares or does not: where no grouped family is
registered the projection keeps the arithmetic it had before.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from hipengine.core.memory import DeviceBuffer, free as hip_free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
    gemma4_gelu_tanh_mul_bf16,
    gemma4_moe_lane_to_row_i32,
    gemma4_moe_weighted_accumulate_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_types import Gemma4Projection
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_bf16
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_gather_packed_hidden_lowp,
    qwen35_moe_group_compact_active,
)

_BF16_BYTES = 2
_I32_BYTES = 4
_I64_BYTES = 8
# Kahan-bound multiplier for the iu8 risk criterion. The screened floor on actual
# weights is between 0.5 and 1 (below it, BF16 flips escape the repair); 4.0
# keeps a >=4x margin and is the value the Qwen route measured at 1.50-1.82x
# operation-complete.
_IU8_RISK_MULTIPLIER = 4.0
_F32_BYTES = 4
_DS4_BLOCK_VALUES = 128


# WMMA prefill owners address padded 16-row tiles rather than compact rows, so
# they need a routing-independent upper bound to size their plan from shape
# alone. Assigning one row to each potentially active expert costs one tile per
# expert, and every further tile needs 16 more rows; unused tiles are written as
# expert -1 and rejected by the kernels. Same bound as
# ``_compact_wmma_static_upper_bound`` in the Qwen35 runner.
def _wmma_tile_upper_bound(selected_rows: int, num_experts: int) -> tuple[int, int]:
    active_experts = min(int(selected_rows), int(num_experts))
    upper_tiles = active_experts + (int(selected_rows) - active_experts) // 16
    return upper_tiles * 16, upper_tiles

# Activation planes the MMQ gate/up route packs and consumes. The pack uses the
# passes for error feedback - each pass re-quantizes the previous pass's
# residual - so this is an accuracy knob as well as a size knob: one plane costs
# 0.717% relative activation error on Gemma 4's activations, two cost 0.0028% and
# three cost 0.0004%. Must match the ACTIVATION_PASSES the leaf is instantiated
# with, which is the ds4x3 symbol.
#
# Set to 3 on 2026-09-28 once the leaf's per-plane min correction was fixed. Two
# defects had to clear first, and the second one is why the earlier attempt at
# three planes measured as a no-op:
#
# 1. The fused leaf hardcoded ACTIVATION_PASSES = 1 and the launcher did not
#    forward it, so a three-pass instantiation ran as one pass. The kernel is now
#    templated on the pass count and the launcher forwards it.
# 2. The body applied the Q4_K min-offset correction ``-dmin * xsum`` once per
#    activation plane. That term is a property of the activation, and only the
#    first plane stores the activation's own sum; the later planes store their
#    residuals' sums. Accumulating it per plane over-counted it by about the size
#    of the error the extra planes remove. It is now taken from the first plane
#    only, which makes three planes deliver the accuracy the pack was built for.
#
# Held at 1 because raising it changes the arithmetic of a production default,
# which needs its execution-profile gate, and this route has no plane policy for
# that gate to resolve: scripts/execution_profile_q8_mmq_plane_gate.py drives
# Qwen4Exp's Q8MMQPrefillPolicy.planes, not this constant. Everything the
# promotion needs is measured and recorded in
# worklog/entries/20260927T185444.658791Z-lhl-gemma4-mmq-activation-planes-a58366.md:
# at 64 ids on the first layer against the fp32 grouped arm, one plane is 0.655
# percent relative and three planes are 0.030 percent, 22x; prefill costs 10.0
# percent at 512 prompt tokens (270 -> 244 tok/s) and 11.2 percent at 1024
# (249 -> 224 tok/s), with decode unchanged. To lift this, route the plane count
# through the variant policy so the plane gate can resolve it, then run that gate.
_MMQ_ACTIVATION_PASSES = 1

# DS4 activation block strides. ``block_q8_1_mmq_ds4`` is ``uint16_t ds4[8]``
# plus ``int8_t qs[128]``, so its fp16 layout -- for a hidden-state input -- is
# 16 + 128 bytes; the fp32 one is for a post-SiLU input; see
# gemma4_project_experts_down_mmq.
_DS4_BLOCK_BYTES = 16 + 128
_DS4_F32_BLOCK_BYTES = 32 + 128


def _ds4_block_count(width: int) -> int:
    """DS4 blocks for ``width``, rounding up for a partial trailing block."""

    return (width + 127) // 128

# Grouped-prefill family: one launch covers every expert, reading each expert's
# weight matrix once and reusing it across that expert's contiguous row slice.
# The ABI is (input, expert_start, weights, out, compact_rows, num_experts,
# in_features, out_features), and the registry decides by quant key which quants
# ship it; a quant without one falls through to the per-expert row-slice route.
#
# These are the same reduction at different fetch strategies, in preference
# order: every variant here produces bit-identical output to the selected GEMV
# for the quant that registers it (asserted per quant in
# tests/test_unit_gemma4_expert_route.py), so preferring the later ones is a cost
# decision, not an accuracy one. A quant that registers only the first name
# still runs; the list is a preference, not a requirement.
_GROUPED_PREFILL_VARIANTS = (
    "selected_grouped_prefill_staged_out4_bf16_bf16_out",
    "selected_grouped_prefill_staged_out8_bf16_bf16_out",
    "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
    # The Q8_0 expert family registers only the plain compact variant, and one
    # MoE layer of this artifact carries Q8_0 expert weights. Without it here
    # that layer's projection falls through to the per-row gather, which costs
    # about eight times a grouped layer.
    "selected_grouped_prefill_compact_bf16_bf16_out",
)

# The grouped row4 GEMV is the only grouped kernel Q5_K has. Its launch ABI
# differs from the grouped prefill family above (it takes a lane map and both a
# source and a destination row count), so it is a separate dispatch step rather
# than another entry in the preference list.
_GROUPED_ROW4_VARIANT = "selected_grouped_row4_gemv_bf16_bf16_out"

# Prefill prefers a grouped family once there is at least one compact lane per
# expert. Below that most experts are empty, so a grouped launch's per-expert
# grid would spend its blocks on nothing and the selected GEMV's per-lane grid
# is cheaper.
_PREFILL_MIN_LANES_PER_EXPERT = 1

# Diagnostic: how many expert projections ran through each route since import.
# Tests and probes read this to confirm the intended path ran rather than
# inferring it from a timing.
_MOE_ROUTE_COUNTS: dict[str, int] = {}

# Diagnostic: how many grouped projections resolved to each registered variant.
# The route name alone cannot show which of the grouped family's fetch strategies
# ran, and they differ in cost rather than in output.
_GROUPED_VARIANT_COUNTS: dict[str, int] = {}

# The fused MMQ gate/up route is the default path.
#
# The lead's decision on 2026-09-27 made it the default after the teacher-forced
# check passed at worst-case KL 0.0013 and 0.00077 against the 0.05 bar with no
# top-1 changes. This branch reaches the same conclusion by measurement: on
# realistic prose (2004 ids) the route matches the fp32 arm for ten consecutive
# tokens and the first divergence is llama.cpp alone, and on a truncated Python
# snippet it matches for all 23. Where the arms do part they part from llama.cpp
# together rather than from each other. The route's raw-logit perturbation is
# three times smaller in distribution than out (3.46 max absolute difference on
# prose against 9.76 on random ids), which is the mechanism: the int8
# activation-quantisation step is small enough not to move a decision on input the
# model was trained for, and random ids leave the logits flat enough that it is.
#
# The kl_max breach this flag was originally held for is 0.0651 on 2 of 1022
# near-certain repeated-context rows (kl_mean 1.44e-4, kl_p95 1.3e-5, kl_p99
# 1.5e-4 and top-1 100% all pass with margin), and the shipped 4-slice attention
# split breaches the same bar on the same chain at the same rows (0.055589, 1
# row). The flag's recorded cause - a Python snippet collapsing into a repeated
# token - did not reproduce against llama.cpp on a comparable prompt.
#
# ``HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0`` is the rollback lever and restores the
# fp32 grouped route. Removal condition: once a teacher-forced gate against the
# campaign's frozen evaluator has been recorded for this default, delete the flag.
# See ``docs/REFACTOR.md``.
_GEMMA4_MOE_GATE_UP_MMQ_ENV = "HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ"
_GEMMA4_MOE_DOWN_MMQ_ENV = "HIPENGINE_GEMMA4_MOE_DOWN_MMQ"

# Values that turn the route off. Anything else, including unset, leaves it on, so
# a typo cannot silently downgrade the default path.
_MMQ_DISABLING_VALUES = frozenset(("0", "false", "no", "off", "disable", "disabled"))


def gemma4_moe_down_mmq_enabled() -> bool:
    """Whether the Q5_1 DS4 MMQ down-projection route may be selected.

    On by default. The down projection is the largest single route in gfx1151
    prefill and has no MMQ path without this one, so it runs the fp32 grouped
    family at 8.6 GB/s where the Q4_K gate/up MMQ on the same layer runs at
    67 GB/s. ``HIPENGINE_GEMMA4_MOE_DOWN_MMQ`` set to a falsy value is the
    rollback lever and restores the grouped route.
    """

    import os

    return os.environ.get(_GEMMA4_MOE_DOWN_MMQ_ENV, "").strip().lower() not in (
        _MMQ_DISABLING_VALUES
    )


def gemma4_moe_gate_up_mmq_enabled() -> bool:
    """Whether the fused MMQ gate/up route may be selected.

    On by default: it is the measured faster path (1.27x on the 512/128 prefill)
    and the numerical decision that held it back has cleared.
    ``HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ`` set to a falsy value is the rollback
    lever and restores the fp32 grouped route.
    """

    import os

    return os.environ.get(_GEMMA4_MOE_GATE_UP_MMQ_ENV, "").strip().lower() not in (
        _MMQ_DISABLING_VALUES
    )


def gemma4_moe_expert_route_counts() -> dict[str, int]:
    """Return how many expert projections used each dispatch route."""

    return dict(_MOE_ROUTE_COUNTS)


def gemma4_moe_grouped_variant_counts() -> dict[str, int]:
    """Return how many grouped projections resolved to each registered variant.

    The route name reports the family, and every variant in that family produces
    the same bits; this reports which fetch strategy actually ran, which is what
    a cost change needs to confirm.
    """

    return dict(_GROUPED_VARIANT_COUNTS)


def _record_moe_route(route: str) -> None:
    _MOE_ROUTE_COUNTS[route] = _MOE_ROUTE_COUNTS.get(route, 0) + 1


def gemma4_moe_prefill_route_enabled(*, lanes: int, num_experts: int) -> bool:
    """Whether an expert projection should take the row-reusing prefill route.

    The selected-GEMV family launches one block per (out_col, lane), so a block
    with many lanes per expert re-reads each expert's weight matrix once per
    lane. The row-reusing route reads each weight element once per expert and
    reuses it across that expert's rows. The crossover is one lane per expert:
    below it most experts are empty, so a grouped launch's per-expert grid would
    spend its blocks on nothing and the per-lane GEMV grid is cheaper.

    ``lanes`` is the width the *runner* was built for, not the width of the call
    in flight. The two routes are different arithmetic -- an int8-dp4a
    accumulation against an fp32 one -- so choosing between them by live row
    count would make a token's output depend on how many tokens happened to
    share its call, and that difference compounds through the KV cache. Passing
    the declared capacity instead makes the route a property of the runner, so a
    runner's prefill and its single-token decodes take the same path. Callers
    that genuinely have no declared width, such as the grouped family's own
    internal dispatcher, may pass their live row count.
    """

    if int(num_experts) <= 0:
        raise ValueError("num_experts must be positive")
    return int(lanes) >= _PREFILL_MIN_LANES_PER_EXPERT * int(num_experts)


@dataclass
class Gemma4ExpertScratch:
    """Reusable device scratch for one expert-forward shape.

    Sized once from ``(tokens, top_k, hidden_size, intermediate, num_experts)``
    and reused across layers and decode steps, which is why it is a separate
    object rather than allocated per call.
    """

    tokens: int
    top_k: int
    hidden_size: int
    intermediate: int
    num_experts: int
    _buffers: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _by_name: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)
    # The MMQ tile walk's compact-to-source map is the identity over the scratch
    # capacity, so it is written once per scratch object rather than per layer.
    mmq_identity_ready: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("tokens", self.tokens),
            ("top_k", self.top_k),
            ("hidden_size", self.hidden_size),
            ("intermediate", self.intermediate),
            ("num_experts", self.num_experts),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def total_lanes(self) -> int:
        return self.tokens * self.top_k

    def buffer(self, name: str) -> DeviceBuffer:
        """Allocate ``name`` on first use, then reuse it."""

        existing = self._by_name.get(name)
        if existing is not None:
            return existing
        buf = malloc(self._size_of(name))
        self._by_name[name] = buf
        self._buffers.append(buf)
        if name == "compact_to_source":
            # The MMQ leaf dereferences this unconditionally, so it has to hold
            # a real map rather than a null. Gemma4's compact buffer is already
            # in source order, which makes it the identity: fill it once here
            # instead of paying a kernel launch on every prefill.
            import numpy as np

            from hipengine.core.memory import copy_host_to_device, host_array_ptr

            iota = np.arange(self.total_lanes, dtype=np.int64)
            copy_host_to_device(buf, host_array_ptr(iota))
        return buf

    def free(self) -> None:
        for buf in self._buffers:
            hip_free(buf)
        self._buffers.clear()
        self._by_name.clear()

    def _sizes(self) -> dict[str, int]:
        """Byte size of every buffer this scratch can hold.

        One table shared by ``_size_of`` and ``resident_bytes`` so the
        allocator and the memory planner cannot drift apart.
        """
        lanes = self.total_lanes
        # Widths follow the group-scatter ABI exactly: `counts` is int32 and
        # `sorted_weights` is float, but `selected_experts`, `expert_start`,
        # `active_experts`, `active_count`, `sorted_lanes`, and
        # `sorted_experts` are all int64. Narrowing any of them makes the
        # kernels read past the end of the buffer.
        sizes = {
            "counts": self.num_experts * _I32_BYTES,
            "expert_start": (self.num_experts + 1) * _I64_BYTES,
            "active_experts": self.num_experts * _I64_BYTES,
            "active_count": _I64_BYTES,
            "sorted_lanes": lanes * _I64_BYTES,
            "sorted_experts": lanes * _I64_BYTES,
            "sorted_weights": lanes * _F32_BYTES,
            "lane_to_row": lanes * _I32_BYTES,
            "packed_hidden": lanes * self.hidden_size * _BF16_BYTES,
            "gate_up_out": lanes * 2 * self.intermediate * _BF16_BYTES,
            "activated": lanes * self.intermediate * _BF16_BYTES,
            "expert_out": lanes * self.hidden_size * _BF16_BYTES,
            # DS4 Q8_1 activation planes: one 144-byte block per 128 input
            # elements per row, times the plane count the MMQ32 route consumes.
            # The pack uses extra planes for error feedback, which is the whole
            # point of paying for them; _MMQ_ACTIVATION_PASSES records why that
            # count is 1 today and what would lift it.
            "mmq_workspace": lanes
            * max(
                _ds4_block_count(self.hidden_size) * _DS4_BLOCK_BYTES,
                _ds4_block_count(self.intermediate) * _DS4_F32_BLOCK_BYTES,
            )
            * _MMQ_ACTIVATION_PASSES,
            "mmq_identity": lanes * _I64_BYTES,
            # One 32-row tile per entry; a tile-per-expert bound is exact when
            # every expert has at least one row, and adding the leftover rows
            # covers the fragmented case.
            "mmq_expert_start": (lanes // 32 + self.num_experts + 1) * _I64_BYTES,
            "mmq_tile_expert": (lanes // 32 + self.num_experts + 1) * _I64_BYTES,
            "mmq_total": _I64_BYTES,
            # The iu8-WMMA risk route queues the compact-row and output indices
            # whose activation quantization could move a result, then repairs
            # exactly those. Capacity is the worst case - every output at risk -
            # so the repair can never silently drop a queued index.
            "mmq_risk_count": _I32_BYTES,
            "mmq_risk_indices": lanes * 2 * self.intermediate * _I32_BYTES,
            # WMMA prefill tile plan. ``expert_start`` counts compact rows and
            # the WMMA owners address padded 16-row tiles instead, so they need
            # their own per-expert start, one expert id per tile, and the total
            # padded row count. The WMMA tile walk packs 16 rows per tile
            # against the MMQ32 route's 32, so it needs its own buffers rather
            # than the mmq_* pair above.
            "wmma_expert_start": (self.num_experts + 1) * _I64_BYTES,
            # Sized from ``lanes`` rather than from the routing-independent
            # bound ``_wmma_tile_upper_bound`` returns: that bound
            # (``min(L, E) + (L - min(L, E)) // 16``) is what both tile-plan
            # builders pass as ``tile_capacity``, while the MMQ32 builder's
            # returned row count launches a grid of ``(L + 31 * E) / 32`` tiles
            # against the same buffer, which the 16-row bound does not cover for
            # every ``(L, E)``. This allocation is at least both of those, so
            # neither writer can run past it.
            "wmma_tile_expert": (lanes // 16 + self.num_experts + 1) * _I64_BYTES,
            "wmma_total": _I64_BYTES,
            # Grouped int8 MMQ prefill. ``ds4_q8`` holds the compact activations
            # packed as llama.cpp-style DS4 ``block_q8_1_mmq`` blocks, and
            # ``compact_to_source`` is the row map the MMQ leaf dereferences. The
            # MMQ32 tile ABI is the same 16-row plan the WMMA owners build, so
            # that plan is reused and no second one is allocated here.
            # ``ds4_q8`` is sized across both widths that pack into it -- the
            # gate/up route's ``hidden_size`` and the down route's
            # ``intermediate`` -- the same way ``mmq_workspace`` above is, so
            # neither pack can run past it. ``_ds4_block_count`` rounds a
            # partial trailing block up, which a plain ``// 128`` does not.
            "ds4_q8": lanes
            * max(
                _ds4_block_count(self.hidden_size),
                _ds4_block_count(self.intermediate),
            )
            * _DS4_BLOCK_BYTES,
            "compact_to_source": lanes * _I64_BYTES,
        }
        return sizes

    def _size_of(self, name: str) -> int:
        try:
            return self._sizes()[name]
        except KeyError:
            raise KeyError(f"unknown expert scratch buffer {name!r}") from None

    def resident_bytes(self) -> int:
        """Upper bound on every device buffer this scratch can take."""
        return sum(self._sizes().values())


def gemma4_experts_forward_bf16(
    hidden_ptr: int,
    selected_experts_ptr: int,
    routing_weights_ptr: int,
    gate_up_proj: Gemma4Projection,
    down_proj: Gemma4Projection,
    out_ptr: int,
    *,
    scratch: Gemma4ExpertScratch,
    rows: int | None = None,
    stream: int = 0,
    library: object | None = None,
    runtime: object | None = None,
) -> None:
    """Run the routed-expert FFN for one block of tokens.

    ``hidden`` is ``(tokens, hidden_size)`` BF16. ``selected_experts`` is
    ``(tokens, top_k)`` **int64** and ``routing_weights`` is ``(tokens, top_k)``
    F32, both token-major — the int64 width is the group-scatter ABI, not a
    choice. ``gate_up_proj`` is ``(num_experts, 2 * intermediate, hidden_size)``
    BF16 with the gate in the first half, ``down_proj`` is ``(num_experts,
    hidden_size, intermediate)`` BF16, and ``out`` is ``(tokens, hidden_size)``
    BF16, overwritten rather than accumulated into.

    All buffers are raw device pointers so this composes with the paged-KV and
    attention paths without a host-side tensor round trip.
    """

    # ``scratch`` is a capacity, not an identity: the caller sizes it for the
    # widest block it will run and then runs narrower blocks through it. The
    # remaining shape parameters must match exactly, since they describe weights.
    tokens = scratch.tokens if rows is None else int(rows)
    if tokens <= 0:
        raise ValueError("rows must be positive")
    if tokens > scratch.tokens:
        raise ValueError(f"rows={tokens} exceeds scratch capacity {scratch.tokens}")
    top_k = scratch.top_k
    hidden_size = scratch.hidden_size
    intermediate = scratch.intermediate
    num_experts = scratch.num_experts
    # ``lanes`` is the live lane count for *this* call, not the buffer's
    # capacity. The group-scatter kernels iterate it over ``sorted_lanes`` and
    # ``sorted_experts``, so passing the capacity makes them walk lanes that
    # were never written and index with uninitialized expert ids and offsets.
    # That reads far outside every expert buffer, which the device reports as an
    # SQ privilege fault and never recovers from - the fault is silent and the
    # next synchronizing call simply blocks forever.
    lanes = tokens * top_k
    kwargs = {"stream": stream}
    if library is not None:
        kwargs["library"] = library
    if runtime is not None:
        kwargs["runtime"] = runtime

    expert_start = scratch.buffer("expert_start")
    active_experts = scratch.buffer("active_experts")
    active_count = scratch.buffer("active_count")
    sorted_lanes = scratch.buffer("sorted_lanes")
    sorted_experts = scratch.buffer("sorted_experts")
    sorted_weights = scratch.buffer("sorted_weights")
    lane_to_row = scratch.buffer("lane_to_row")
    packed_hidden = scratch.buffer("packed_hidden")
    gate_up_out = scratch.buffer("gate_up_out")
    activated = scratch.buffer("activated")
    expert_out = scratch.buffer("expert_out")

    # 1. Group the lanes by expert. The compact-active kernel issues its own
    #    count, prefix and scatter stages internally -- both the serial and the
    #    parallel variants do -- so the caller-side group_count /
    #    group_prefix_active passes are redundant: they added two launches per
    #    MoE block and nothing reads `counts` afterwards.
    #
    # Which compactor runs is a registered backend capability, not a correctness
    # question. The serial kernel launches one block of 256 threads (dim3(1)) to
    # compact every lane and was the largest single piece of this block's glue
    # at 12.354 ms of a 656 ms prefill (worklog/entries/20260929T103000); the
    # parallel sibling launches one block per expert and is bit-identical to it
    # at tokens 1/512/4096/777 with top_k 8/8/8/4
    # (scripts/gemma4_group_compact_equivalence.py). Both hip_gfx1100 and
    # hip_gfx1151 declare "parallel", so resolving the capability turns it on
    # without a backend branch here; "serial" stays the rollback and needs no
    # new flag. The accessor still carries the name of the model whose MoE
    # group-scatter this machinery was first built for.
    from hipengine.runtime.laguna_moe import resolve_laguna_group_compact_mode

    # `Gemma4Projection` is `int | Gemma4GGUFDeviceWeight`, and a raw pointer
    # carries no backend to resolve a capability against. That is not a silent
    # downgrade: it is the same answer the resolver gives a backend that
    # declares nothing, and the mode is still the profile's to set by name. The
    # GGUF path, which is what production passes, always has the weight object.
    compact_backend = getattr(gate_up_proj, "backend", None)
    compact_parallel = compact_backend is not None and (
        resolve_laguna_group_compact_mode(compact_backend) == "parallel"
    )
    qwen35_moe_group_compact_active(
        selected_experts_ptr,
        routing_weights_ptr,
        expert_start.ptr,
        active_experts.ptr,
        active_count.ptr,
        sorted_lanes.ptr,
        sorted_experts.ptr,
        sorted_weights.ptr,
        lanes,
        num_experts,
        parallel=compact_parallel,
        **kwargs,
    )

    # 2. Pull the hidden rows into compact order so each expert's rows are a
    #    contiguous slice of `packed_hidden`. This happens after the route flags
    #    below rather than here, because the MMQ gate_up gathers straight out of
    #    the source rows and leaves `packed_hidden` untouched.

    # 3. One gate_up projection for every compact row, then GeGLU over the whole
    #    compact buffer in a single launch.
    #
    # The unpinned default first tries the registered fused-slab T16 WMMA
    # owner for wide calls. Other layouts retain the int8 MMQ gate/up and
    # fallback ladder in :func:`gemma4_project_experts_rows`. A pinned
    # HIPENGINE_GEMMA4_MOE_PREFILL mode selects one of the other owners
    #    instead: the split-weight grouped int8 leaf, the WMMA owners, or the
    #    exact grouped/selected arms. A pin is a selection rather than an
    #    addition, so pinning an exact owner cannot leave an MMQ leaf running
    #    underneath it. The WMMA owners need a padded tile plan, so it is built
    #    once per call and only when a projection will actually use it.
    fused = 2 * intermediate
    mode = _prefill_mode()
    gate_up_wmma, compensated, use_mmq = _prefill_route_flags(mode)
    pinned = mode != "auto"
    fused_wmma_leaf = None if pinned else _fused_wmma_owner(
        gate_up_proj, lanes, hidden_size, intermediate, num_experts
    )
    # Tiles-only layouts retain their MMQ route when no fused WMMA owner
    # supports this shape; neither route reconstructs raw expert weights.
    if not pinned and getattr(getattr(gate_up_proj, "spec", None), "layout", None) == "gguf_q4_k_t16_v1":
        use_mmq = True
    # The MMQ gate reads the runner's declared width rather than this call's lane
    # count: the two routes are different arithmetic, so selecting by live width
    # would make a token's output depend on the batch it arrived in.
    route_width = scratch.tokens
    route_enabled = gemma4_moe_prefill_route_enabled(
        lanes=route_width, num_experts=num_experts
    )
    # The fused-stack MMQ leaf is the default path's route, so its env lever is
    # what rolls the default back. A pinned ``mmq`` mode asks for the int8 family
    # by name, and there the pin wins over that lever: the leaf stays available
    # as the split leaf's fallback, so the request cannot degrade to the grouped
    # family while an MMQ leaf can still serve it.
    fused_mmq_enabled = route_enabled and (
        use_mmq or (not pinned and gemma4_moe_gate_up_mmq_enabled())
    )
    # The MMQ down leaf is ~8x slower than the WMMA owner at Gemma's down
    # geometry -- profiled at 947 ms against 116 ms over the same 116 launches,
    # 4.0 against 32.5 TFLOP/s -- and the MMQ gate_up already writes bf16, which
    # is exactly what the WMMA down reads. So the down keeps the WMMA owner on
    # this route and only the gate_up uses the int8 leaf.
    down_wmma = gate_up_wmma or use_mmq
    mmq_rows = 0
    if use_mmq and fused_wmma_leaf is None:
        # The grouped int8 MMQ leaf reads Gemma's fused ``ffn_gate_up_exps``
        # stack directly through an explicit expert stride, addressing the up
        # half one half into each expert's block, so no split layout is
        # required and no guard on one is needed here.
        if lanes >= _WMMA_PREFILL_MIN_LANES_PER_EXPERT * num_experts:
            mmq_rows = _build_mmq_tile_plan(
                scratch, expert_start.ptr, lanes, stream=stream, runtime=runtime
            )
    wmma_rows = 0
    # Only the WMMA gate_up needs the plan here. The down's plan is built later,
    # just before the down itself, because on the MMQ route it has to come after
    # the gate_up has finished with the 32-row plan that shares these buffers.
    # Building it here under ``down_wmma`` alone would overwrite that plan before
    # the gate_up ran and fault the MMQ leaf.
    if (
        (gate_up_wmma or fused_wmma_leaf is not None)
        and lanes >= _WMMA_PREFILL_MIN_LANES_PER_EXPERT * num_experts
    ):
        wmma_rows = _build_wmma_tile_plan(
            scratch, expert_start.ptr, lanes, stream=stream, runtime=runtime
        )
    # 2'. Gather the hidden rows into compact order, unless the MMQ gate_up
    #     below is taking the route -- it gathers and packs in one kernel, and
    #     on that route `packed_hidden` has no other reader, so staging it here
    #     would write and immediately re-read `lanes * hidden_size * 2` bytes.
    if not (use_mmq and mmq_rows) or not _mmq_dual_route(
        gate_up_proj, lanes, hidden_size, intermediate, num_experts
    ):
        qwen35_moe_gather_packed_hidden_lowp(
            hidden_ptr,
            sorted_lanes.ptr,
            packed_hidden.ptr,
            lanes * hidden_size,
            tokens,
            top_k,
            hidden_size,
            **kwargs,
        )
    # ``selected`` pins the per-lane GEMV, so it is handled first and in full:
    # the ladder's first rung is a grouped owner, and reaching it would run an
    # owner the pin excludes. The per-expert offset walk is its only fallback.
    if mode == "selected":
        if not gemma4_project_experts_selected(
            gate_up_proj,
            packed_hidden.ptr,
            sorted_experts.ptr,
            gate_up_out.ptr,
            lanes,
            lanes,
            num_experts,
            hidden_size,
            fused,
            **kwargs,
        ):
            gemma4_project_experts_by_offset(
                gate_up_proj,
                packed_hidden.ptr,
                gate_up_out.ptr,
                expert_start,
                num_experts,
                hidden_size,
                fused,
                **kwargs,
            )
    elif fused_wmma_leaf is not None:
        fused_wmma_leaf(
            packed_hidden.ptr, expert_start.ptr,
            scratch.buffer("wmma_expert_start").ptr,
            scratch.buffer("wmma_tile_expert").ptr,
            gate_up_proj.allocation().buffer.ptr, gate_up_out.ptr,
            lanes, hidden_size, intermediate, intermediate, num_experts,
            wmma_rows, stream=stream, runtime=runtime,
        )
        _record_moe_route("gate_up_fused_t16")
    elif use_mmq and mmq_rows and gemma4_project_experts_mmq_dual(
        gate_up_proj,
        hidden_ptr,
        sorted_lanes.ptr,
        expert_start.ptr,
        scratch,
        gate_up_out.ptr,
        lanes,
        num_experts,
        hidden_size,
        intermediate,
        fused,
        mmq_rows,
        tokens=tokens,
        top_k=top_k,
        stream=stream,
        runtime=runtime,
    ):
        pass
    elif gate_up_wmma and wmma_rows and gemma4_project_experts_wmma_dual(
        gate_up_proj,
        packed_hidden.ptr,
        expert_start.ptr,
        scratch.buffer("wmma_expert_start").ptr,
        scratch.buffer("wmma_tile_expert").ptr,
        gate_up_out.ptr,
        lanes,
        num_experts,
        hidden_size,
        intermediate,
        wmma_rows,
        compensated=compensated,
        stream=stream,
        runtime=runtime,
    ):
        pass
    # The grouped rungs are the owners a pinned mode names. The unpinned default
    # reaches them only through the ladder in :func:`gemma4_project_experts_rows`,
    # so neither preference list can shadow the other.
    elif pinned and gemma4_project_experts_grouped_dual(
        gate_up_proj,
        packed_hidden.ptr,
        expert_start.ptr,
        gate_up_out.ptr,
        lanes,
        num_experts,
        hidden_size,
        intermediate,
        fused,
        stream=stream,
        runtime=runtime,
    ):
        pass
    elif pinned and gemma4_project_experts_grouped(
        gate_up_proj,
        packed_hidden.ptr,
        expert_start.ptr,
        gate_up_out.ptr,
        lanes,
        num_experts,
        hidden_size,
        fused,
        stream=stream,
        runtime=runtime,
    ):
        pass
    elif fused_mmq_enabled and gemma4_project_experts_gate_up_mmq(
        gate_up_proj,
        packed_hidden.ptr,
        gate_up_out.ptr,
        expert_start,
        lanes,
        num_experts,
        hidden_size,
        intermediate,
        scratch=scratch,
        **kwargs,
    ):
        pass
    else:
        _record_moe_route(
            gemma4_project_experts_rows(
                gate_up_proj,
                packed_hidden.ptr,
                gate_up_out.ptr,
                expert_start,
                sorted_experts.ptr,
                lanes,
                num_experts,
                hidden_size,
                fused,
                **kwargs,
            )
        )
    # The MMQ-family wrapper records the leaf that served the projection, so the
    # ladder is the only gate/up route left for the caller to name. Naming the
    # family here as well counted a Q4T16 tile launch twice and named a Q5_K iu8
    # launch as a route it never ran.
    gemma4_gelu_tanh_mul_bf16(gate_up_out.ptr, activated.ptr, lanes, intermediate, **kwargs)

    # 4. The down projection, over the same compact rows. The mode split is the
    #    gate_up's: the unpinned default runs the DS4 MMQ down route where the
    #    weight qualifies and the ladder otherwise, and a pinned mode runs the
    #    owners it names.
    #
    # The gate_up consumed the 32-row MMQ plan and the WMMA owner tiles 16 rows,
    # so the plan is rebuilt at the WMMA width here. Both plans share the same
    # buffers, which is why this has to happen after the gate_up rather than
    # alongside the MMQ plan above.
    if (
        use_mmq
        and down_wmma
        and not wmma_rows
        and lanes >= _WMMA_PREFILL_MIN_LANES_PER_EXPERT * num_experts
    ):
        wmma_rows = _build_wmma_tile_plan(
            scratch, expert_start.ptr, lanes, stream=stream, runtime=runtime
        )
    # The down's own MMQ lever, under the same rule as the gate_up's: it rolls
    # the unpinned default back, and a pinned ``mmq`` keeps the leaf available as
    # the fallback for the owners that pin prefers.
    fused_down_mmq_enabled = route_enabled and (
        use_mmq or (not pinned and gemma4_moe_down_mmq_enabled())
    )
    if mode == "selected":
        if not gemma4_project_experts_selected(
            down_proj,
            activated.ptr,
            sorted_experts.ptr,
            expert_out.ptr,
            lanes,
            lanes,
            num_experts,
            intermediate,
            hidden_size,
            **kwargs,
        ):
            gemma4_project_experts_by_offset(
                down_proj,
                activated.ptr,
                expert_out.ptr,
                expert_start,
                num_experts,
                intermediate,
                hidden_size,
                **kwargs,
            )
    elif down_wmma and wmma_rows and (
        (not compensated and _gemma4_project_experts_down_wmma_t16(
            down_proj,
            activated.ptr,
            expert_start.ptr,
            scratch.buffer("wmma_expert_start").ptr,
            scratch.buffer("wmma_tile_expert").ptr,
            expert_out.ptr,
            lanes,
            num_experts,
            intermediate,
            hidden_size,
            wmma_rows,
            stream=stream,
            runtime=runtime,
        ))
        or gemma4_project_experts_wmma(
        down_proj,
        activated.ptr,
        expert_start.ptr,
        scratch.buffer("wmma_expert_start").ptr,
        scratch.buffer("wmma_tile_expert").ptr,
        expert_out.ptr,
        lanes,
        num_experts,
        intermediate,
        hidden_size,
        wmma_rows,
        compensated=compensated,
        stream=stream,
        runtime=runtime,
    )
    ):
        pass
    elif use_mmq and gemma4_project_experts_mmq(
        down_proj,
        activated.ptr,
        expert_start.ptr,
        scratch,
        expert_out.ptr,
        lanes,
        num_experts,
        intermediate,
        hidden_size,
        stream=stream,
        runtime=runtime,
    ):
        pass
    elif fused_down_mmq_enabled and gemma4_project_experts_down_mmq(
        down_proj,
        activated.ptr,
        expert_out.ptr,
        expert_start,
        lanes,
        num_experts,
        intermediate,
        hidden_size,
        scratch=scratch,
        **kwargs,
    ):
        _record_moe_route("down_mmq32")
    elif pinned and gemma4_project_experts_grouped(
        down_proj,
        activated.ptr,
        expert_start.ptr,
        expert_out.ptr,
        lanes,
        num_experts,
        intermediate,
        hidden_size,
        stream=stream,
        runtime=runtime,
    ):
        pass
    else:
        _record_moe_route(
            gemma4_project_experts_rows(
                down_proj,
                activated.ptr,
                expert_out.ptr,
                expert_start,
                sorted_experts.ptr,
                lanes,
                num_experts,
                intermediate,
                hidden_size,
                **kwargs,
            )
        )

    # 5. Accumulate the compacted expert outputs back onto their tokens.
    gemma4_moe_lane_to_row_i32(sorted_lanes.ptr, lane_to_row.ptr, lanes, **kwargs)
    gemma4_moe_weighted_accumulate_bf16(
        expert_out.ptr,
        lane_to_row.ptr,
        sorted_weights.ptr,
        out_ptr,
        tokens,
        hidden_size,
        top_k,
        **kwargs,
    )


def gemma4_project_expert(
    weight: Gemma4Projection,
    expert: int,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
) -> None:
    """Run one expert's projection out of a stacked expert tensor.

    The stacked ``(num_experts, out, in)`` tensor is one allocation, and an
    expert is selected by offsetting into it. For a bf16 weight the stride is
    ``out * in * 2``; for a quantized weight a row is a whole number of blocks,
    so the stride comes from the tensor's byte count instead of being recomputed
    here.
    """

    expert = int(expert)
    if expert < 0:
        raise ValueError(f"expert index must be non-negative, got {expert}")
    if isinstance(weight, int):
        stride = out_features * in_features * _BF16_BYTES
        dense_gemv_out_bf16(
            x_ptr,
            weight + expert * stride,
            out_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
        )
        return
    from hipengine.loading.gemma4_gguf_device import LAYOUT_RAW_GGUF

    if weight.spec.layout != LAYOUT_RAW_GGUF:
        raise ValueError(
            f"{weight.spec.slot_path}: the per-expert raw launch reads raw GGUF "
            f"blocks, but this weight is resident as {weight.spec.layout!r}; "
            "the layout-aware selected owner must serve it"
        )
    # Imported here rather than at module scope: the quantized dispatch lives in
    # the runtime layer and the kernel package does not depend on it otherwise.
    from hipengine.runtime.gguf_linear import launch_gguf_linear_raw_ptr

    allocation = weight.allocation()
    if expert >= int(weight.spec.source.shape[0]):
        raise ValueError(
            f"expert {expert} is out of range for {weight.spec.slot_path} "
            f"with {int(weight.spec.source.shape[0])} experts"
        )
    launch_gguf_linear_raw_ptr(
        weight,
        allocation.buffer.ptr + expert * weight.expert_stride_bytes,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
    )


# Every GGUF quant type this artifact uses registers a selected-expert GEMV under
# this one variant name, so the expert forward resolves it from the registry by
# quant key rather than branching on the type.
_SELECTED_VARIANT = "selected_gemv_bf16_bf16_out"
# D11: the pack8 selected leaf shares the raw selected GEMV's launch ABI and
# sits registered under this sibling variant for quants whose pack-of-8
# blocks exist (q8_0, q5_k); the chain below prefers it only where
# out_features meets its out % 8 launch contract.
#
# It is a bit-exact sibling of ``_SELECTED_VARIANT`` (see
# ``gemma4_project_experts_selected``): the k walk, the per-element dequant and
# the reduction tree are properties of an output, not of the block, and the
# pack8 leaf reproduces all three per output. It is preferred where it is
# registered because one block then reads the x row once and pays the block
# reduction once for eight outputs instead of one.
_SELECTED_PACK8_VARIANT = "selected_pack8_gemv_bf16_bf16_out"

# Grouped prefill owners that keep one CTA per (expert, output column) and reuse
# each loaded weight row across ``row_batch`` compact rows, instead of the
# selected GEMV's one CTA per (row, output column). Not every quant registers
# one, so this is a probe the caller falls back from, exactly like
# ``_SELECTED_VARIANT``. The variant name is the ABI, not the quant: a quant that
# registers it is served, and one that does not keeps the selected path.
_GROUPED_PREFILL_VARIANT = "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out"

# The Qwen35-era paired/folded owner, preferred ahead of both row-batch variants
# where it registers. It is the same arithmetic -- same thread-to-column map,
# same 256-thread tree per output -- and measured bit-identical to the amortized
# owner on Gemma's down geometry (8192 compact rows, 128 experts, in 704, out
# 2816), where it is also 1.84x faster: 21.8 ms against 40.2 ms, with the
# plain pair2 form at 28.6 and the two row-batch forms at 45.0 and 51.2. Only
# Q5_1 registers it, and only for in_features at or below its fold limit, so a
# quant or width it does not serve falls through to the variants below.
_GROUPED_FOLD128_PREFILL_VARIANT = (
    "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"
)

# The paired Q5_1 forms reject wider inputs outright rather than declining, so
# the probe only offers them inside the width their wrapper accepts.
_GROUPED_FOLD128_MAX_IN_FEATURES = 4096

# The same owner with the loop nest swapped so one CTA covers four output
# columns and reuses the input row batch across them. Only quants that register
# it are served by it; the probe prefers it and falls back to the row-batch
# variant above, which is why both are listed in that order.
_GROUPED_AMORTIZED_PREFILL_VARIANT = (
    "selected_grouped_prefill_compact_rowbatch8_out4_amortized_bf16_bf16_out"
)

# The fused-``gate_up`` form of the same owner. It takes the fused expert stride
# and writes both halves into one row, which is why the single-output variant
# above cannot serve Gemma's fused tensor.
_GROUPED_DUAL_PREFILL_VARIANT = (
    "selected_dual_grouped_rowbatch8_bf16_bf16_out"
)

# The same owner with the loop nest swapped so one CTA covers four output
# columns and reuses the input row batch across them. The association of every
# output is unchanged -- same thread-to-column map, same 128-thread tree -- so
# this is the bit-identical route and is preferred whenever the width fits.
#
# The ``_bundle`` sibling was measured and rejected: it publishes all ROW_BATCH
# rows after one barrier per output column instead of one barrier per row per
# output half (8 barriers per row down to 1), but it also collapses the final
# reduction onto 2 * ROW_BATCH = 16 of the 128 threads. On Gemma's fused gate_up
# geometry it ran 21900 us against 7960 us, 1.5 TF/s against 4.1 -- 2.75x slower.
# Barriers are not this kernel's bottleneck; parallelism in the reduction is.
_GROUPED_DUAL_AMORTIZED_PREFILL_VARIANT = (
    "selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out"
)

# The amortized owner's block metadata lives in one shared slab sized for 16
# Q4_K blocks, so a wider input has no slab and keeps the row-batch owner.
_GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES = 4096

# Lanes, not rows: a compact row is one (token, top-k) pair, and the grouped
# grid is ``out_features * num_experts`` CTAs however few rows are live. Below
# this many lanes the selected GEMV's smaller grid wins, because most grouped
# CTAs would find their expert empty. ``num_experts`` divides out at four rows
# per expert on average, which is where weight reuse starts to pay for the
# wider grid.
_GROUPED_PREFILL_MIN_LANES_PER_EXPERT = 4

# WMMA prefill owners keep the whole tile in registers and read each weight
# block once per 16-row tile, so they need more live rows than the grouped GEMV
# before the wider grid pays. Same reasoning as the grouped gate, higher bar.
_WMMA_PREFILL_VARIANT = "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
# Grouped int8 MMQ prefill. The leaf is Q4_K-specific and consumes DS4-packed
# activations, so it is a route of its own rather than a variant of the grouped
# owners, which take BF16 activations and dequantize the weights.
# The mmq32 prefill leaf is registered per (backend, layer, quant), so the
# route asks a capability question -- can this quant resolve the leaf? --
# rather than matching a list of quant names. Q4_K and Q5_K both have owners;
# admitting a further quant means registering an owner for it, never editing a
# branch here. See AGENTS.md "Never key admission on identity".
_MMQ32_PREFILL_VARIANT = (
    "selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
)
# The down projection is a single Q5_1 matrix, so its MMQ leaf needs no split.
_MMQ_DOWN_QUANT_KEY = "gguf_q5_1"
# The grouped int8 MMQ leaf reads the fused ``ffn_gate_up_exps`` stack directly
# through an explicit expert stride, with the up half addressed one half into
# each expert's block. No split layout is materialized.
_WMMA_PREFILL_MIN_LANES_PER_EXPERT = 16

# Compensated twins of the two WMMA owners. The plain owners round every
# dequantised weight to fp16 (~2^-11 relative), which is accurate enough for the
# kernels' own parity contract but leaves a measurable tail divergence against
# the strict f32 dequant baseline. The compensated owners carry each weight as
# an fp16 high part plus an fp16 residual and issue a second WMMA per k-tile,
# which costs one extra op per weight on a path that runs far below WMMA issue
# rate. Both are registered per quant and probed exactly like the plain ones.
_WMMA_PREFILL_COMP_VARIANT = (
    "selected_grouped_wmma_prefill_compact_comp_bf16_bf16_out"
)

# A fused ``gate_up`` expert tensor stores both halves in one allocation, so the
# dual WMMA owner needs the fused stride rather than each half's own width. The
# variant name is the ABI for the fused form; the two-tensor form registers
# under the same name without the stride.
_WMMA_DUAL_PREFILL_VARIANT = "selected_dual_wmma_prefill_compact_bf16_bf16_out"
_WMMA_DUAL_PREFILL_COMP_VARIANT = (
    "selected_dual_wmma_prefill_compact_comp_bf16_bf16_out"
)

# Prefill expert-route selector. A pinned mode picks one owner family by name;
# ``auto`` pins nothing and leaves the default path to its own default-on levers
# (see ``_prefill_route_flags``). Unrecognised values fall back to ``auto``.
#
# ``wmma`` selects the compensated WMMA owners instead. They are 2.7x faster on
# a 1024-token prefill but breach the campaign's absolute ``kl_max`` bar on 1 of
# 1023 rows (0.0609 against 0.05) while passing every aggregate bar with 10-200x
# margin and leaving top-1 unchanged on all 1023 rows. The breach is reduction
# association, not a defect: compensating the fp16 weight rounding moves it from
# 0.182 to 0.061, and a two-way accumulator split moves it to 0.082. Treating an
# absolute ``kl_max`` as inapplicable to a reordering-class change is an open
# lead decision recorded in docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md, so the
# arm stays off the default path until that is ruled on.
#
# ``wmma_plain`` is the uncompensated form, kept as the diagnostic that isolates
# the fp16 weight-rounding term. ``grouped`` and ``selected`` pin the exact arms.
_PREFILL_MODE_ENV = "HIPENGINE_GEMMA4_MOE_PREFILL"
_PREFILL_MODES = frozenset(
    {"auto", "wmma", "wmma_plain", "mmq", "grouped", "selected"}
)


def _prefill_mode() -> str:
    """Return the pinned prefill route, or ``auto`` for the production policy."""

    import os

    raw = os.environ.get(_PREFILL_MODE_ENV, "").strip().lower()
    return raw if raw in _PREFILL_MODES else "auto"


def _prefill_route_flags(mode: str) -> tuple[bool, bool, bool]:
    """Return ``(use_wmma, compensated, use_mmq)`` for a prefill selector.

    ``auto`` pins no owner: the unpinned default resolves through the route's own
    default-on levers, which are the fused-stack int8 MMQ gate/up and the DS4 MMQ
    down. A pinned mode selects one of the owners below instead, and the forward
    pass runs that owner's chain rather than stacking it on the default one.

    ``mmq`` pins the grouped int8 MMQ owner: the split-weight gate/up leaf with
    the WMMA owner for the down, measured at 1393 tok/s against the exact grouped
    route's 675 at ``--prompt 1024``, with its logits gate 37x inside the
    calibrated envelope (``kl_max`` 0.001341 against a 0.05 bar, 0 of 1023 top-1
    flips). The split-weight leaf needs gate and up resident separately, so this
    is also the mode the loader materializes that layout for.

    ``grouped`` and ``selected`` pin the exact routes. ``wmma`` probes the WMMA
    owners in their compensated form, ``wmma_plain`` in their uncompensated form.
    """

    if mode == "auto":
        # No owner is pinned: the forward pass resolves the default path through
        # the route's own default-on levers -- the fused-stack int8 MMQ gate/up
        # and the DS4 MMQ down, each with its own env rollback lever and width
        # policy. Those levers, not this table, are what the unpinned default
        # runs, which is why ``auto`` pins nothing.
        return False, False, False
    if mode == "wmma":
        return True, True, False
    if mode == "wmma_plain":
        return True, False, False
    if mode == "mmq":
        return False, False, True
    return False, False, False


def _fused_wmma_owner(
    weight: Gemma4Projection, compact_rows: int, in_features: int,
    out_features: int, num_experts: int,
):
    """Exact registered fused-slab consumer, or None for unsupported shapes.

    The consumer preserves the split WMMA parent's arithmetic and reads the
    primary resident slab. Narrow calls keep their existing selected path.
    """
    if (isinstance(weight, int) or in_features % 256 or out_features % 16 or
            compact_rows < _WMMA_PREFILL_MIN_LANES_PER_EXPERT * num_experts):
        return None
    from hipengine.kernels.registry import KernelKey, is_registered, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    key = KernelKey(weight.backend, "moe_linear", weight.spec.quant_key,
                    "selected_dual_wmma_prefill_fused_bf16_bf16_out")
    _ensure_linear_kernel_registered(key)
    if not is_registered(key):
        return None
    return resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                   variant=key.variant)


def gemma4_project_experts_wmma_dual(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    runtime: object | None = None,
    compensated: bool = False,
) -> bool:
    """Run a fused gate+up projection through the dual WMMA prefill owner.

    ``out_features`` is one half's width: the weight tensor holds ``2 *
    out_features`` rows per expert with the gate first, and the output row is
    ``2 * out_features`` wide with the gate in the first half. The owner indexes
    each half from its own start, so the up half is the same allocation offset
    by one half's bytes and the expert stride is the full ``2 * out_features``.

    Returns ``False`` when no dual owner serves this weight.
    """

    if isinstance(weight, int):
        return False
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    variant = (
        _WMMA_DUAL_PREFILL_COMP_VARIANT if compensated else _WMMA_DUAL_PREFILL_VARIANT
    )
    key = KernelKey(weight.backend, "moe_linear", weight.spec.quant_key, variant)
    _ensure_linear_kernel_registered(key)
    try:
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
    except MissingKernelError:
        return False
    base = weight.allocation().buffer.ptr
    # One half's byte length, not a row stride: a Q4_K row is a whole number of
    # 256-value blocks, so the half boundary lands on a block boundary too.
    half_bytes = weight.expert_stride_bytes // 2
    fn(
        x_ptr,
        expert_start_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        base,
        base + half_bytes,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        out_features,
        num_experts,
        wmma_total_rows,
        expert_stride_rows=2 * out_features,
        stream=stream,
        runtime=runtime,
    )
    return True


def gemma4_project_experts_wmma(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    runtime: object | None = None,
    compensated: bool = False,
) -> bool:
    """Run one compact-row projection through a WMMA prefill owner.

    ``expert_start_wmma_ptr`` and ``tile_expert_ptr`` are the padded tile plan
    built by :func:`qwen35_moe_wmma_tile_map`; ``wmma_total_rows`` is its row
    count, which the caller takes from the same upper bound the plan was built
    against so no device-to-host read is needed to size the grid.

    Returns ``False`` when no WMMA owner serves this weight.
    """

    if isinstance(weight, int):
        return False
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    variant = _WMMA_PREFILL_COMP_VARIANT if compensated else _WMMA_PREFILL_VARIANT
    key = KernelKey(weight.backend, "moe_linear", weight.spec.quant_key, variant)
    _ensure_linear_kernel_registered(key)
    try:
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
    except MissingKernelError:
        return False
    fn(
        x_ptr,
        expert_start_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        weight.allocation().buffer.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        wmma_total_rows,
        stream=stream,
        runtime=runtime,
    )
    return True


def _build_wmma_tile_plan(
    scratch: Gemma4ExpertScratch,
    expert_start_ptr: int,
    lanes: int,
    *,
    stream: int,
    runtime: object | None,
) -> int:
    """Fill ``scratch``'s WMMA tile plan and return its padded row count."""

    from hipengine.kernels.hip_gfx1100.moe.group_scatter import qwen35_moe_wmma_tile_map

    upper_rows, upper_tiles = _wmma_tile_upper_bound(lanes, scratch.num_experts)
    qwen35_moe_wmma_tile_map(
        expert_start_ptr,
        scratch.buffer("wmma_expert_start").ptr,
        scratch.buffer("wmma_tile_expert").ptr,
        scratch.buffer("wmma_total").ptr,
        scratch.num_experts,
        tile_capacity=upper_tiles,
        stream=stream,
        runtime=runtime,
    )
    return upper_rows


def _build_mmq_tile_plan(
    scratch: Gemma4ExpertScratch,
    expert_start_ptr: int,
    lanes: int,
    *,
    stream: int,
    runtime: object | None,
) -> int:
    """Fill ``scratch``'s 32-row MMQ tile plan and return its padded row count.

    The MMQ32 leaf tiles 32 rows at a time where the WMMA owners tile 16, so it
    needs its own map -- the two are not interchangeable, and feeding a 16-row
    plan to the MMQ32 leaf walks off the end of the activation buffer.

    Returns a routing-independent upper bound instead of the total the map
    actually wrote, so no device readback and no stream synchronize is needed per
    call.

    The bound is ``lanes + 31 * num_experts``, from
    ``sum(ceil(c_e / 32) * 32) <= sum(c_e + 31)``. It has to be computed for 32-row
    tiling specifically: 32-row padding is *larger* than 16-row padding
    (``ceil(33/32)*32 = 64`` against ``ceil(33/16)*16 = 48``), so the 16-row bound
    ``_wmma_tile_upper_bound`` returns is not a ceiling for this plan and passing
    it drops real tiles -- measured 6304 real rows against a 6016 bound. The
    sentinel fill makes the extra tiles harmless: the map writes ``-1`` across the
    whole capacity and the leaf returns early on a negative expert, and the
    16-row capacity always exceeds ``bound / 32`` tiles, so the fill covers the
    grid this launches.
    """

    from hipengine.kernels.hip_gfx1100.moe.group_scatter import qwen35_moe_mmq32_tile_map

    _, upper_tiles = _wmma_tile_upper_bound(lanes, scratch.num_experts)
    qwen35_moe_mmq32_tile_map(
        expert_start_ptr,
        scratch.buffer("wmma_expert_start").ptr,
        scratch.buffer("wmma_tile_expert").ptr,
        scratch.buffer("wmma_total").ptr,
        scratch.num_experts,
        tile_capacity=upper_tiles,
        stream=stream,
        runtime=runtime,
    )
    return lanes + 31 * scratch.num_experts


def _mmq32_leaf_owner(
    weight: Gemma4Projection | int,
):
    """Resolve the mmq32 prefill leaf for ``weight``'s quant, or ``None``.

    Capability, not identity: the registry is keyed ``(backend, layer, quant,
    variant)``, so a quant is admitted exactly when an owner for it exists.
    Returns ``None`` for an absent owner so callers fall back to the grouped
    route instead of raising mid-prefill.

    ``is_registered`` first, then this leaf's own registrar, because two things
    conspire against a bare ``resolve``: registration is import-time and lazy,
    and ``tests/conftest.py`` restores a collection-time baseline after every
    test, so an owner registered during one test is gone by the next. The
    shared ``gguf_linear`` battery does not cover this variant either. Probing
    first keeps a test's deliberate fixture registration intact.
    """
    from hipengine.kernels.registry import (
        KernelKey,
        MissingKernelError,
        is_registered,
        resolve,
    )

    key = KernelKey(
        weight.backend, "moe_linear", weight.spec.quant_key, _MMQ32_PREFILL_VARIANT
    )
    if not is_registered(key):
        from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
            register_gguf_q4_k_q8_1_selected_prefill_kernels,
        )

        register_gguf_q4_k_q8_1_selected_prefill_kernels()
    try:
        return resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
    except MissingKernelError:
        return None


def _mmq_dual_route(
    weight: Gemma4Projection | int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
) -> bool:
    """True when :func:`gemma4_project_experts_mmq_dual` will take the route.

    Pure: the same guards the projection itself returns ``False`` on, lifted so
    the caller can decide *before* the gather whether ``packed_hidden`` will have
    a reader at all. Keeping them in one place is what makes that decision safe.
    """

    if isinstance(weight, int):
        return False
    if not _mmq32_leaf_owner(weight):
        return False
    if in_features % _DS4_BLOCK_VALUES or out_features % 32:
        return False
    if compact_rows < _GROUPED_PREFILL_MIN_LANES_PER_EXPERT * num_experts:
        return False
    # The leaf sizes its 32-row plan as ``compact_rows + 31 * num_experts`` (see
    # the bound computed above) and ``_check_mmq32_common`` raises on a total
    # that is not a multiple of 32. Without this guard the route is selected
    # anyway and the projection raises ``ValueError: mmq_total_rows must be a
    # multiple of 32`` mid-prefill, which surfaces through ``LLM.generate()`` as
    # a GenerationExecutionFailed. For a 128-expert stack the 31 * 128 term is
    # 3968, itself a multiple of 32, so the condition reduces to
    # ``compact_rows % 32`` -- and with top_k = 8 routed lanes per token that is
    # ``tokens % 4``. Refusing here is what the docstring promises: the grouped
    # and selected owners below serve any row count.
    if (compact_rows + 31 * num_experts) % 32:
        return False
    return True


def gemma4_project_experts_mmq_dual(
    weight: Gemma4Projection,
    hidden_ptr: int,
    sorted_lanes_ptr: int,
    expert_start_ptr: int,
    scratch: Gemma4ExpertScratch,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    fused_width: int,
    mmq_total_rows: int,
    *,
    tokens: int,
    top_k: int,
    stream: int = 0,
    library: object | None = None,
    runtime: object | None = None,
) -> bool:
    """Run a fused ``gate_up`` projection through the grouped int8 MMQ owner.

    Packs the compact activations to llama.cpp-style DS4 ``block_q8_1_mmq`` on
    the GPU and then runs the 32x32 packed-dot leaf, which multiplies int8
    activations against int8 weights. That is the structural difference from the
    grouped and WMMA owners, which dequantize each weight to BF16 and then do
    BF16 FMAs: at the Gemma4 MoE shape the int8 leaf measures 2.07x the BF16
    WMMA owner, and 2.00x even paying this packing cost.

    ``mmq_total_rows`` is the padded row count of the 32-row MMQ tile plan
    already held in ``scratch``. The leaf reads that plan's per-expert start and
    tile->expert map, which the WMMA owners' 16-row plan cannot substitute for.

    ``fused_width`` is the resident tensor's row width per expert (``2 *
    out_features`` for Gemma's ``gate | up`` stack). It is passed to the leaf as
    its expert stride with ``up`` addressed one half into the same allocation,
    which is how the fused layout is read without a second resident copy.

    Takes ``hidden_ptr`` (the un-gathered ``(tokens, hidden_size)`` rows) plus
    ``sorted_lanes``, and gathers them to DS4 ``block_q8_1_mmq`` in a single
    kernel rather than staging a BF16 ``packed_hidden`` row in between: on this
    route that buffer has no other reader, so folding the two drops a write and
    a read of ``compact_rows * in_features * 2`` bytes per call. The result is
    byte-identical to gathering first, which
    ``test_q8_1_mmq_gather_ds4_pack_is_byte_exact_to_gather_then_pack`` pins.

    Returns ``False`` when the weight is not a Q4_K expert stack or the shape is
    not one the leaf serves, which leaves the grouped and selected owners to
    handle it.
    """

    if not _mmq_dual_route(
        weight, compact_rows, in_features, out_features, num_experts
    ):
        return False

    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        build_gguf_q4_k_q8_1_selected_prefill,
        gguf_q8_1_mmq_gather_ds4_pack_bf16,
    )

    # Resolved by capability rather than by name: whichever quant owns the
    # mmq32 leaf runs it, so Q4_K and Q5_K share this path with no branch on
    # quant_key. ``_mmq_dual_route`` already checked the owner exists.
    leaf = _mmq32_leaf_owner(weight)
    if leaf is None:
        return False

    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    ds4 = scratch.buffer("ds4_q8")
    gguf_q8_1_mmq_gather_ds4_pack_bf16(
        hidden_ptr,
        sorted_lanes_ptr,
        ds4.ptr,
        compact_rows,
        in_features,
        tokens,
        top_k,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    base_ptr = weight.allocation().buffer.ptr
    # One half's byte length, not a row stride: a Q4_K row is a whole number of
    # 256-value blocks, so the half boundary lands on a block boundary too. The
    # fused stack holds each expert's gate rows then its up rows, so the up half
    # starts one half into that expert's block and the leaf strides experts by
    # the fused width.
    half_bytes = weight.expert_stride_bytes // 2
    leaf(
        ds4.ptr,
        scratch.buffer("compact_to_source").ptr,
        expert_start_ptr,
        scratch.buffer("wmma_expert_start").ptr,
        scratch.buffer("wmma_tile_expert").ptr,
        base_ptr,
        base_ptr + half_bytes,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        out_features,
        num_experts,
        mmq_total_rows,
        expert_stride_rows=fused_width,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    return True


def gemma4_project_experts_mmq(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    scratch: Gemma4ExpertScratch,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: object | None = None,
    runtime: object | None = None,
) -> bool:
    """Run one projection through the grouped int8 Q5_1 MMQ owner.

    The down projection is a single matrix, so this leaf reads the raw GGUF Q5_1
    layout directly: no split is needed, and unlike the Q4_K dual leaf it takes
    the compact ``expert_start`` rather than a padded tile plan.

    Returns ``False`` when the weight is not a Q5_1 expert stack or the shape is
    not one the leaf serves, which leaves the grouped and selected owners to
    handle it.
    """

    if isinstance(weight, int):
        return False
    if weight.spec.quant_key != _MMQ_DOWN_QUANT_KEY:
        return False
    # No in_features divisibility gate: the DS4 pack zero-fills a partial final
    # block, so the down projection's K=704 (5.5 blocks of 128) is served.
    # Q5_1 still needs in_features % 32 == 0 for the weight row, which the
    # block_q5_1 layout itself guarantees.
    if compact_rows < _GROUPED_PREFILL_MIN_LANES_PER_EXPERT * num_experts:
        return False

    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        # d4x3 = residual_passes 3: the consumer runs with planes=3, so it
        # reads all three DS4 planes. The plain symbol is <1> and writes only
        # plane 0, leaving planes 1-2 uninitialized -- the K=704 gate used to
        # keep this path from ever running, which hid the mismatch.
        gguf_q8_1_mmq_ds4_pack_bf16_d4x3 as gguf_q8_1_mmq_ds4_pack_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
        build_gguf_q5_1_mmq_selected_prefill,
        gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out,
    )

    library = library or build_gguf_q5_1_mmq_selected_prefill(load=True)
    # The down input is the post-GeGLU activation, which is narrower than the
    # gate_up input, so it reuses the same DS4 workspace the dual leaf packs
    # into: the two projections run back to back and never hold it at once.
    ds4 = scratch.buffer("ds4_q8")
    gguf_q8_1_mmq_ds4_pack_bf16(
        x_ptr,
        ds4.ptr,
        compact_rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out(
        ds4.ptr,
        expert_start_ptr,
        weight.allocation().buffer.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        runtime=runtime,
        library=library,
    )
    return True


def gemma4_project_experts_grouped_dual(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    fused_width: int,
    *,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run a fused ``gate_up`` projection through a grouped dual owner.

    Prefers the amortized owner when ``in_features`` fits its metadata slab,
    which is the same arithmetic with the input row batch reused across four
    output columns instead of re-read once per column.

    The owner keeps one CTA per (expert, output column) and walks that expert's
    compact rows in batches, so each weight row is loaded once per batch instead
    of once per row. It writes both halves of the fused row: gate columns at
    ``[0, out_features)`` and up columns at ``[out_features, fused_width)``.

    ``fused_width`` is the fused row width (``2 * out_features`` for Gemma's
    ``gate_up``). Both the weight expert stride and the output row stride are
    that width, because both halves live in one allocation while each still
    indexes from its own origin.

    Returns ``False`` when no grouped dual owner serves this weight, which
    leaves the selected GEMV as the only path for quants without one.
    """

    if isinstance(weight, int):
        return False
    if compact_rows < _GROUPED_PREFILL_MIN_LANES_PER_EXPERT * num_experts:
        return False
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    variants = [_GROUPED_DUAL_PREFILL_VARIANT]
    if in_features <= _GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES:
        variants.insert(0, _GROUPED_DUAL_AMORTIZED_PREFILL_VARIANT)
    fn = None
    for variant in variants:
        key = KernelKey(weight.backend, "moe_linear", weight.spec.quant_key, variant)
        _ensure_linear_kernel_registered(key)
        try:
            fn = resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
            break
        except MissingKernelError:
            continue
    if fn is None:
        return False
    base_ptr = weight.allocation().buffer.ptr
    # One half's byte length, not a row stride: a Q4_K row is a whole number of
    # 256-value blocks, so the half boundary lands on a block boundary too. The
    # owner indexes each side from its own origin while striding experts by the
    # fused width, so the up half starts one half into the allocation.
    half_bytes = weight.expert_stride_bytes // 2
    fn(
        x_ptr,
        expert_start_ptr,
        base_ptr,
        base_ptr + half_bytes,
        out_ptr,
        out_ptr + out_features * 2,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        output_row_stride=fused_width,
        expert_stride_rows=fused_width,
        stream=stream,
        runtime=runtime,
    )
    return True

def gemma4_project_experts_grouped(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run one grouped projection over the compact expert rows.

    ``expert_start_ptr`` is ``int64`` with ``num_experts + 1`` ascending row
    offsets -- the inclusive-end convention the expert scratch already builds.
    Each CTA owns one output column of one expert and walks that expert's
    compact rows in batches, so a weight row is loaded once per batch instead of
    once per row. Where the quant registers one, the amortized owner is preferred:
    it is the same arithmetic with the input row batch reused across four output
    columns instead of re-read once per column.

    Returns ``False`` when no grouped owner serves this weight, which leaves the
    selected GEMV as the only path for quants without a prefill owner.
    """

    if isinstance(weight, int):
        return False
    if compact_rows < _GROUPED_PREFILL_MIN_LANES_PER_EXPERT * num_experts:
        return False
    # Same registration caveat as the selected path: a lazily imported family can
    # be missing because a registry test cleared global registrations, so the
    # lookup goes through the dispatch's own ensure helper.
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    variants = [
        _GROUPED_AMORTIZED_PREFILL_VARIANT,
        _GROUPED_PREFILL_VARIANT,
    ]
    if in_features <= _GROUPED_FOLD128_MAX_IN_FEATURES:
        variants.insert(0, _GROUPED_FOLD128_PREFILL_VARIANT)
    fn = None
    for variant in variants:
        key = KernelKey(weight.backend, "moe_linear", weight.spec.quant_key, variant)
        _ensure_linear_kernel_registered(key)
        try:
            fn = resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
            break
        except MissingKernelError:
            continue
    if fn is None:
        return False
    fn(
        x_ptr,
        expert_start_ptr,
        weight.allocation().buffer.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        runtime=runtime,
    )
    return True


# Memo for the selected-expert route: fingerprint -> (generation, fn,
# allocation_name). Routing is pure in its fingerprint plus registry state,
# so a decode projection pays a dict lookup instead of the candidate walk --
# the walk measured +0.84 ms/step of host time on the d8 attribution probe
# when paid per call. Any registry register/clear bumps registry.generation()
# and every entry carries the generation it was computed under, so tests that
# clear or re-register invalidate themselves without touching this table.
_SELECTED_ROUTE_CACHE: dict[
    tuple[str, str, str | None, bool, int], tuple[int, object | None, str | None]
] = {}


def _selected_route(
    weight: Gemma4Projection, out_features: int
) -> tuple[object | None, str | None]:
    """Resolve the chain to ``(fn, allocation_name)``, memoized per fingerprint.

    The candidate walk itself is unchanged: registration-existence and
    geometry only, exact ``is_registered`` gating, with one restore-and-retry
    pass when nothing resolves (registry plan tests clear global registrations
    and pytest restores a collection-time baseline, so a lazy import can be a
    no-op and a lookup for a kernel that exists can still report it missing;
    ``_ensure_linear_kernel_registered`` is the repo's answer to exactly that,
    and is what the GGUF runtime dispatch uses).

    Allocation *pointers* stay outside the fingerprint on purpose: the caller
    reads them off the weight at launch, so two weights with equal fingerprint
    share only the resolved function, never a buffer address.
    """

    from hipengine.kernels.registry import (
        KernelKey,
        MissingKernelError,
        generation,
        is_registered,
        resolve,
    )
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    tiles_quant = getattr(weight.spec, "tiles_quant_key", None)
    has_tiles = bool(
        callable(getattr(weight, "has_allocation", None))
        and weight.has_allocation("tiles")
    )
    fingerprint = (
        weight.backend,
        weight.spec.quant_key,
        tiles_quant,
        has_tiles,
        out_features,
    )
    gen = generation()
    entry = _SELECTED_ROUTE_CACHE.get(fingerprint)
    if entry is not None and entry[0] == gen:
        return entry[1], entry[2]

    candidates: list[tuple[str, str, str | None]] = []
    if tiles_quant and has_tiles:
        candidates.append((tiles_quant, _SELECTED_VARIANT, "tiles"))
    if out_features % 8 == 0:
        candidates.append(
            (weight.spec.quant_key, _SELECTED_PACK8_VARIANT, None)
        )
    candidates.append((weight.spec.quant_key, _SELECTED_VARIANT, None))

    resolved_fn: object | None = None
    resolved_alloc: str | None = None
    for attempt in range(2):
        for quant, variant, allocation in candidates:
            key = KernelKey(weight.backend, "linear", quant, variant)
            # Exact registration only. resolve() walks generic fallbacks
            # (same layer without variant, fp16 on the same backend, then the
            # cpu_reference backend), so resolving an *unpromoted*
            # candidate key would hand back another layer's kernel -- e.g.
            # cpu_reference's ``linear`` -- and crash at launch. A promoted
            # candidate therefore has to be registered under its own key or
            # it is skipped; the incumbent keeps exact-registered semantics
            # too, because its old fallback answers were never callable here
            # (they reject the stream keyword).
            if not is_registered(key):
                continue
            try:
                fn = resolve(
                    backend=key.backend,
                    layer=key.layer,
                    quant=key.quant,
                    variant=key.variant,
                )
            except MissingKernelError:
                continue
            resolved_fn, resolved_alloc = fn, allocation
            break
        if resolved_fn is not None:
            break
        if attempt == 0:
            _ensure_linear_kernel_registered(
                KernelKey(
                    weight.backend, "linear", weight.spec.quant_key, _SELECTED_VARIANT
                )
            )
        else:
            break

    # Stamp with the generation *after* the walk: the restore pass inside it
    # may itself have registered the family, and that state is what the
    # resolved function came from.
    _SELECTED_ROUTE_CACHE[fingerprint] = (
        generation(),
        resolved_fn,
        resolved_alloc,
    )
    return resolved_fn, resolved_alloc


def gemma4_project_experts_selected(
    weight: Gemma4Projection,
    x_ptr: int,
    selected_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
) -> bool:
    """Run one projection per compact row, each row using its own expert.

    ``selected_ptr`` is ``int64`` with one expert index per compact row -- the
    expert scratch already builds exactly this as ``sorted_experts``.

    Returns ``False`` when no selected kernel serves this weight, which is the
    bf16 case: the selected family is a GGUF quantized-block concept and a bf16
    weight is not a GGUF quant, so the caller uses the per-expert offset path.
    That is a check on the weight's storage form, not on a quant name.

    Candidates are tried in order, gated on registration and geometry only --
    never on identity:

    1. the T16 tiles sibling under ``spec.tiles_quant_key`` when the weight
       carries the ``tiles`` allocation the planner shipped for it (D11
       screen: bit-exact with the raw incumbent and 4.35x faster at rows 8,
       0.1830 -> 0.0421 ms at the layer-29 gate_up geometry);
    2. the pack8 selected leaf under the same quant when ``out_features``
       meets the ``out % 8`` contract ``_validate(require_pack8=True)``
       enforces at launch (D11: 3.05x bit-exact at rows 8 for the layer-29
       down);
    3. the registered ``selected_gemv`` incumbent -- every quant that
       registers neither sibling lands here, and D3's q5_1 logical-t64
       owner is registered *as* this variant, so the chain never reorders
       it.

    A candidate whose key is not registered falls through to the next, and
    the weight pointer follows the candidate: the tiles sibling reads the
    tiles allocation, every raw candidate reads the primary.
    """

    if isinstance(weight, int):
        return False
    fn, allocation = _selected_route(weight, out_features)
    if fn is None:
        return False
    fn(
        x_ptr,
        selected_ptr,
        weight.allocation(allocation).buffer.ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
    )
    return True


def gemma4_project_experts_by_offset(
    weight: Gemma4Projection,
    x_ptr: int,
    out_ptr: int,
    expert_start,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
) -> None:
    """Project each non-empty expert's contiguous row slice in its own launch.

    The fallback for weights with no selected kernel. It reads ``expert_start``
    back to the host, which is a device-to-host synchronisation, so the selected
    path is preferred wherever one exists.
    """

    from hipengine.loading.gemma4_gguf_device import LAYOUT_RAW_GGUF

    if not isinstance(weight, int) and weight.spec.layout != LAYOUT_RAW_GGUF:
        raise ValueError(
            f"{weight.spec.slot_path}: the by-offset expert launch reads raw "
            f"GGUF blocks, but this weight is resident as {weight.spec.layout!r}; "
            "the layout-aware selected owner must serve it"
        )

    # The compact-prefix producer may run on the nonblocking MoE stream.
    # A synchronous default-stream D2H copy does not wait for that producer.
    # Drain its stream before reading offsets used to size each expert launch.
    if stream != 0:
        from hipengine.core.hip import get_hip_runtime

        get_hip_runtime().stream_synchronize(stream)
    starts = _read_int64(expert_start, num_experts + 1)
    for expert in range(num_experts):
        start = int(starts[expert])
        rows = int(starts[expert + 1]) - start
        if rows <= 0:
            continue
        gemma4_project_expert(
            weight,
            expert,
            x_ptr + start * in_features * _BF16_BYTES,
            out_ptr + start * out_features * _BF16_BYTES,
            rows,
            in_features,
            out_features,
            stream=stream,
        )


def gemma4_project_experts_grouped_prefill(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
) -> bool:
    """Run one grouped launch that reuses each expert's weights across its rows.

    Returns ``False`` when this weight has no registered grouped-prefill family,
    which is a property of the quant key, not of a model or an artifact.
    """

    if isinstance(weight, int):
        return False
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    for variant in _GROUPED_PREFILL_VARIANTS:
        key = KernelKey(weight.backend, "moe_linear", weight.spec.quant_key, variant)
        _ensure_linear_kernel_registered(key)
        try:
            fn = resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
        except MissingKernelError:
            continue
        _GROUPED_VARIANT_COUNTS[variant] = _GROUPED_VARIANT_COUNTS.get(variant, 0) + 1
        fn(
            x_ptr,
            expert_start_ptr,
            weight.allocation("raw").buffer.ptr,
            out_ptr,
            compact_rows,
            num_experts,
            in_features,
            out_features,
            stream=stream,
        )
        return True
    return False


def gemma4_project_experts_down_mmq(
    weight: Gemma4Projection,
    x_ptr: int,
    out_ptr: int,
    expert_start,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    scratch: Gemma4ExpertScratch,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run the DS4 DP4A MMQ route over a Q5_1 or Q8_0 expert down projection.

    The down projection is the largest single route in gfx1151 prefill. Without
    this route it has no MMQ path at all -- :func:`gemma4_project_experts_gate_up_mmq`
    accepts only ``gguf_q4_k`` and ``gguf_q5_k`` -- so it runs the fp32 grouped
    family at 8.6 GB/s where the Q4_K gate/up MMQ on the same layer runs at
    67 GB/s, 11.8x per FLOP at this artifact's shapes.

    Returns ``False`` when the weight or the geometry does not qualify, which is
    a property of the quant key and the shapes rather than of a model or an
    artifact; the caller then falls back to the grouped route.

    This route quantizes the block's activations to DS4 Q8_1 over
    ``_MMQ_ACTIVATION_PASSES`` activation planes, so its arithmetic differs from
    the fp32 grouped route: the error is bounded by the activation quantization
    step, not by reassociation. It is the same envelope the Q4_K gate/up route
    already runs at, measured in ``tests/test_unit_gemma4_expert_route.py``.

    Both consumers tile ``in_features`` in 128-wide DS4 blocks and read one
    32-wide weight block per sub-block, so a width that is not a multiple of 128
    is a partial trailing block rather than a refusal -- which is what lets Gemma
    4 26B-A4B's 704-wide expert down projection use this route at all. Q5_1 and
    Q8_0 share the DS4 activation pack and differ only in the weight decode:
    Q5_1 carries a min and a fifth bit, Q8_0 a single scale and 32 signed bytes.
    """

    if isinstance(weight, int):
        return False
    # The two consumers are the same DP4A structure over the same DS4 activation
    # pack and differ only in the weight decode, so the quant key selects the
    # consumer rather than deciding whether the route runs at all. An artifact
    # that quantizes a layer's expert down differently from its siblings -- Gemma
    # 4 26B-A4B UD-Q4_K_XL ships 29 Q5_1 layers and one Q8_0 -- would otherwise
    # send that one layer to the fp32 grouped family at 8x the cost.
    # The down projection's input is a GeGLU output, so it uses the range-safe
    # fp32 DS4 activation layout rather than the fp16 one the gate/up route
    # packs from a hidden state. An fp16 scale tops out at 65504 and the scale
    # is amax/127, so the fp16 layout runs out at amax of about 8.3e6 -- and a
    # small model's GeGLU output reaches past that. The consumer then evaluates
    # d * inf * dot, which is inf where the dot is nonzero and NaN where it is
    # zero, and a NaN here flows through the residual stream into the next
    # layer's router.
    quant_key = getattr(weight.spec, "quant_key", None)
    if quant_key == "gguf_q5_1":
        from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
            build_gguf_q5_1_mmq_selected_prefill as build_consumer,
            gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out as mmq_down,
        )
    elif quant_key == "gguf_q8_0":
        from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_selected_prefill import (
            build_gguf_q8_0_mmq_selected_prefill as build_consumer,
            gguf_q8_0_mmq_ds4_selected_prefill_bf16_bf16_out as mmq_down,
        )
    else:
        return False
    if in_features <= 0 or in_features % 32:
        return False
    if out_features <= 0 or compact_rows <= 0 or num_experts <= 0:
        return False

    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        build_gguf_q4_k_q8_1_selected_prefill,
        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3 as pack_activations,
    )

    workspace = scratch.buffer("mmq_workspace")
    needed = compact_rows * _ds4_block_count(in_features) * _DS4_F32_BLOCK_BYTES
    if workspace.nbytes < needed:
        return False

    kwargs = {"stream": stream}
    if runtime is not None:
        kwargs["runtime"] = runtime
    pack_activations(
        x_ptr,
        workspace.ptr,
        compact_rows,
        in_features,
        residual_passes=_MMQ_ACTIVATION_PASSES,
        library=build_gguf_q4_k_q8_1_selected_prefill(load=True),
        **kwargs,
    )
    mmq_down(
        workspace.ptr,
        expert_start.ptr,
        weight.allocation("raw").buffer.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        _MMQ_ACTIVATION_PASSES,
        library=build_consumer(load=True),
        f32_scales=True,
        **kwargs,
    )
    return True


def gemma4_project_experts_grouped_row4(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
) -> bool:
    """Run the grouped row4 GEMV when the quant has no grouped prefill family.

    This is the fallback for a quant whose expert weights have a grouped kernel
    but no grouped *prefill* kernel. It reuses an expert's weight rows across
    four rows instead of one, which measured 3.05x the per-row gather at the
    fused gate/up geometry and is bit-exact against it.

    Returns ``False`` when this quant has no such kernel, which is a property of
    the quant key and not of a model or an artifact.
    """

    if isinstance(weight, int):
        return False
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    key = KernelKey(
        weight.backend, "moe_linear", weight.spec.quant_key, _GROUPED_ROW4_VARIANT
    )
    _ensure_linear_kernel_registered(key)
    try:
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
    except MissingKernelError:
        return False
    _GROUPED_VARIANT_COUNTS[key.variant] = _GROUPED_VARIANT_COUNTS.get(key.variant, 0) + 1
    # The compact layout puts lane i's activation in row i, so the lane map is
    # the identity and the launcher takes a null pointer for it.
    fn(
        x_ptr,
        expert_start_ptr,
        None,
        weight.allocation("raw").buffer.ptr,
        out_ptr,
        compact_rows,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
    )
    return True


def gemma4_project_experts_gate_up_mmq(
    weight: Gemma4Projection,
    x_ptr: int,
    out_ptr: int,
    expert_start,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    intermediate: int,
    *,
    scratch: Gemma4ExpertScratch,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run the int8-dp4a MMQ gate+up route over the fused expert weight.

    ``weight`` holds Gemma 4's fused ``ffn_gate_up_exps`` layout - per expert the
    gate rows then the up rows - so one launch produces both halves and the
    output is already the ``gate | up`` row block that
    :func:`gemma4_gelu_tanh_mul_bf16` consumes. Returns ``False`` when the weight
    or the geometry does not qualify, which is a property of the quant key and
    the shapes rather than of a model or an artifact; the caller then falls back
    to the fp32 grouped route.

    This route quantizes the block's activations to DS4 Q8_1 over
    ``_MMQ_ACTIVATION_PASSES`` activation planes, so its arithmetic differs from
    the fp32 route: the error is bounded by the activation quantization step, not
    by reassociation, and is measured against the strict owner in
    ``tests/test_unit_gemma4_expert_route.py``. The pack spends the extra planes
    on error feedback and the leaf now reads them correctly, so the three-plane
    route is 0.030 percent relative against 0.655 percent at one plane, a 22x
    reduction; ``_MMQ_ACTIVATION_PASSES`` records why the route still runs one.

    This is a family, not one leaf: the fused MMQ32 launch below, the Q4T16 tile
    leaf, and the Q5_K iu8 leaf all arrive through here. Each served projection
    is counted under the name of the leaf that ran it -- ``gate_up_mmq32``,
    ``gate_up_t16``, ``gate_up_iu8`` -- exactly once, from this function, which
    is the only place that knows which leaf ran. A caller therefore records the
    ladder route and nothing else; naming the family on top of a leaf's own name
    counted a tile-route launch twice and named an iu8 launch as MMQ32.
    """

    if isinstance(weight, int):
        return False
    quant_key = getattr(weight.spec, "quant_key", None)
    if quant_key == "gguf_q5_k":
        if _gemma4_project_experts_gate_up_wmma_iu8(
            weight,
            x_ptr,
            out_ptr,
            expert_start,
            compact_rows,
            num_experts,
            in_features,
            intermediate,
            scratch=scratch,
            stream=stream,
            runtime=runtime,
        ):
            _record_moe_route("gate_up_iu8")
            return True
        return False
    if quant_key != "gguf_q4_k":
        return False
    if _gemma4_t16_tiles_ready(weight):
        if _gemma4_project_experts_gate_up_wmma_t16(
            weight,
            x_ptr,
            out_ptr,
            expert_start,
            compact_rows,
            num_experts,
            in_features,
            intermediate,
            scratch=scratch,
            stream=stream,
            runtime=runtime,
        ):
            _record_moe_route("gate_up_t16")
            return True
    if in_features % 128 or (2 * intermediate) % 32 or intermediate % 32:
        return False
    if compact_rows <= 0 or num_experts <= 0:
        return False

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        host_array_ptr,
    )
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_mmq32_tile_map,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        build_gguf_q4_k_q8_1_selected_prefill,
        gguf_q4_k_selected_dual_q8_1_ds4_mmq32_fused_prefill_compact32_bf16_bf16_out as mmq_gate_up,
        gguf_q8_1_mmq_ds4_pack_bf16 as pack_activations,
    )

    import numpy as np

    workspace = scratch.buffer("mmq_workspace")
    identity = scratch.buffer("mmq_identity")
    if not scratch.mmq_identity_ready:
        # The pack reads compact rows in compact order, so the tile walk's
        # compact-to-source map is the identity over the scratch capacity. It is
        # written once per scratch object, not once per layer.
        arange = np.ascontiguousarray(np.arange(scratch.total_lanes, dtype=np.int64))
        copy_host_array_to_device(identity, arange, runtime=runtime)
        scratch.mmq_identity_ready = True
    mmq_starts = scratch.buffer("mmq_expert_start")
    tile_expert = scratch.buffer("mmq_tile_expert")
    mmq_total = scratch.buffer("mmq_total")

    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    kwargs = {"stream": stream}
    if runtime is not None:
        kwargs["runtime"] = runtime
    pack_activations(
        x_ptr, workspace.ptr, compact_rows, in_features, library=library, **kwargs
    )
    tile_capacity = scratch.buffer("mmq_tile_expert").nbytes // 8
    qwen35_moe_mmq32_tile_map(
        expert_start.ptr,
        mmq_starts.ptr,
        tile_expert.ptr,
        mmq_total.ptr,
        num_experts,
        tile_capacity=tile_capacity,
        **kwargs,
    )
    if stream:
        from hipengine.core.hip import get_hip_runtime

        (runtime or get_hip_runtime()).stream_synchronize(stream)
    total_host = np.empty(1, dtype=np.int64)
    copy_device_to_host(
        host_array_ptr(total_host),
        DeviceBuffer(ptr=mmq_total.ptr, nbytes=8),
        8,
        runtime=runtime,
    )
    total_rows = int(total_host[0])
    if total_rows <= 0 or total_rows > tile_capacity * 32:
        raise RuntimeError(
            f"gemma4 MMQ gate/up tile row count {total_rows} is outside "
            f"capacity {tile_capacity * 32}"
        )
    weight_ptr = weight.allocation("raw").buffer.ptr
    row_bytes = (in_features // 256) * 144
    mmq_gate_up(
        workspace.ptr,
        identity.ptr,
        expert_start.ptr,
        mmq_starts.ptr,
        tile_expert.ptr,
        weight_ptr,
        weight_ptr + intermediate * row_bytes,
        out_ptr,
        compact_rows,
        in_features,
        intermediate,
        intermediate,
        num_experts,
        total_rows,
        library=library,
        **kwargs,
    )
    _record_moe_route("gate_up_mmq32")
    return True


def _gemma4_t16_tiles_ready(weight: object) -> bool:
    """Whether ``weight`` carries the Q4T16 gate and up tile allocations.

    A weight that predates the tile repack -- a dense Q4_K matrix, or a test stub
    that models only the raw allocation -- has neither, and the route declines it
    rather than falling through to an AttributeError.
    """

    allocations = getattr(weight, "allocations", None)
    if not isinstance(allocations, Mapping):
        return False
    return "t16_gate" in allocations and "t16_up" in allocations


def _expert_tile_row_bound(
    compact_rows: int, num_experts: int, tile_rows: int, tile_capacity: int,
) -> int:
    """Bound padded rows without copying the device tile total to the host.

    Each nonempty expert pads at most tile_rows-1 rows. Unused map entries
    are initialized to -1 and the consumers return before reading weights.
    """
    active = min(compact_rows, num_experts)
    tiles = (compact_rows + (tile_rows - 1) * active) // tile_rows
    if tiles <= 0 or tiles > tile_capacity:
        raise RuntimeError(
            f"gemma4 tile bound {tiles} is outside capacity {tile_capacity}"
        )
    return tiles * tile_rows


def _gemma4_project_experts_gate_up_wmma_t16(
    weight: Gemma4Projection,
    x_ptr: int,
    out_ptr: int,
    expert_start,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    intermediate: int,
    *,
    scratch: Gemma4ExpertScratch,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run the Q4_K Q4T16-WMMA gate+up route.

    Q4_K twin of the DS4 MMQ32 route above and structurally the Q5_K iu8 route's
    simpler sibling: same fused ``ffn_gate_up_exps`` layout, so one launch still
    produces both halves and the output is already the ``gate | up`` row block
    that :func:`gemma4_gelu_tanh_mul_bf16` consumes.

    Two things make it faster than the MMQ32 route at the same geometry. It
    reads the block's activations as they arrive -- there is no BF16 -> DS4 Q8_1
    packing pass over ``compact_rows x in_features`` -- and its weight operand is
    the Q4T16 tile layout, which is one 16-column tile per 256-wide K block
    rather than a column-strided walk of raw Q4_K rows. Measured at Gemma 4
    26B-A4B's expert geometry (128 experts, 32 rows per expert) it is 1.264x the
    MMQ32 route, 5.040 ms against 6.369 ms per layer.

    The tiles cost 2.78 percent more than the raw blocks they are built from and
    are produced once at materialize time, so the route adds no per-call work.
    They are a second resident layout rather than a replacement: the raw blocks
    stay, so a tensor carrying both holds 2.02778 times its raw bytes.
    """

    if intermediate % 16:
        return False
    if not _gemma4_t16_tiles_ready(weight):
        return False

    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_wmma_tile_map,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
        build_gguf_q4_k_t16_selected_prefill,
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_column_major_bf16_bf16_out
        as wmma_gate_up,
    )

    wmma_starts = scratch.buffer("wmma_expert_start")
    tile_expert = scratch.buffer("wmma_tile_expert")
    wmma_total = scratch.buffer("wmma_total")

    library = build_gguf_q4_k_t16_selected_prefill(load=True)
    kwargs = {"stream": stream}
    if runtime is not None:
        kwargs["runtime"] = runtime

    tile_capacity = tile_expert.nbytes // 8
    qwen35_moe_wmma_tile_map(
        expert_start.ptr,
        wmma_starts.ptr,
        tile_expert.ptr,
        wmma_total.ptr,
        num_experts,
        tile_capacity=tile_capacity,
        **kwargs,
    )
    total_rows = _expert_tile_row_bound(compact_rows, num_experts, 16, tile_capacity)

    wmma_gate_up(
        x_ptr,
        expert_start.ptr,
        wmma_starts.ptr,
        tile_expert.ptr,
        weight.allocation("t16_gate").buffer.ptr,
        weight.allocation("t16_up").buffer.ptr,
        out_ptr,
        compact_rows,
        in_features,
        intermediate,
        intermediate,
        num_experts,
        total_rows,
        library=library,
        **kwargs,
    )
    return True


def _gemma4_down_t16_tiles_ready(weight: object) -> bool:
    """Whether ``weight`` carries the Q5_1T16 down tile allocation."""

    allocations = getattr(weight, "allocations", None)
    if not isinstance(allocations, Mapping):
        return False
    return "t16_down" in allocations


def _gemma4_project_experts_down_wmma_t16(
    weight: Gemma4Projection,
    x_ptr: int,
    expert_start_ptr: int,
    wmma_starts_ptr: int,
    tile_expert_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run the Q5_1 expert down projection through the Q5_1T16 tile layout.

    The raw-block grouped WMMA down kernel decodes every weight fragment from
    sixteen scalar global byte loads; the tile layout pre-combines each
    value's low nibble and high bit into one byte and stages the current K
    block's 32-byte per-column runs in LDS, so a fragment decode reads four
    uint32 words from shared memory instead. The decoded weights, the
    accumulation order, and the output are bit-identical to the raw-block
    kernel; the tiles are produced once at materialize time and the raw blocks
    stay resident for the selected GEMV decode path. Measured at the Gemma 4
    26B-A4B expert geometry the leaf is 1.64x the raw-block WMMA kernel.

    Returns ``False`` when the weight does not carry the tile allocation or
    its geometry is outside the leaf's contract; the caller falls through to
    the raw-block WMMA owner.
    """

    import os

    quant_key = getattr(getattr(weight, "spec", None), "quant_key", None)
    if quant_key != "gguf_q5_1":
        return False
    if os.environ.get("HIPENGINE_GEMMA4_Q5_1_DOWN_T16", "1") in {"", "0", "false", "False"}:
        return False
    if in_features % 32 or out_features % 16:
        return False
    if not _gemma4_down_t16_tiles_ready(weight):
        return False
    if isinstance(weight, int):
        return False

    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_t16_selected_grouped_prefill_bf16_bf16_out as wmma_down_t16,
    )

    library = build_qwen4_exp_q5_1(load=True)
    kwargs = {"stream": stream}
    if runtime is not None:
        kwargs["runtime"] = runtime
    wmma_down_t16(
        x_ptr,
        expert_start_ptr,
        wmma_starts_ptr,
        tile_expert_ptr,
        weight.allocation("t16_down").buffer.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        wmma_total_rows,
        library=library,
        **kwargs,
    )
    _record_moe_route("down_t16")
    return True


def _gemma4_project_experts_gate_up_wmma_iu8(
    weight: Gemma4Projection,
    x_ptr: int,
    out_ptr: int,
    expert_start,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    intermediate: int,
    *,
    scratch: Gemma4ExpertScratch,
    stream: int = 0,
    runtime: object | None = None,
) -> bool:
    """Run the Q5_K iu8-WMMA risk+repair gate+up route.

    Q5_K twin of the DS4 MMQ32 route above: same fused ``ffn_gate_up_exps``
    layout, so one launch still produces both halves and the output is already
    the ``gate | up`` row block that :func:`gemma4_gelu_tanh_mul_bf16` consumes.

    Where the Q4_K route quantizes activations to DS4 Q8_1, this one uses a
    3-plane residual int8 under the Kahan-bounded risk criterion, queues the
    outputs whose rounding could move a result, and repairs exactly those
    against the strict row4 owner. The repaired result is bit-identical to that
    owner rather than merely close to it, which is the property
    ``tests/test_unit_gemma4_expert_route.py`` checks.
    """

    # The leaf's column block is 128 wide but it resolves the gate/up half per
    # *column*, so a block straddling the seam at ``intermediate`` already reads
    # the right weight for each of its columns and a half does not have to be a
    # multiple of 128. Gemma 4 26B-A4B's 704 (5 x 128 + 64) therefore runs; the
    # fused width 1408 is 11 x 128, so the grid has no partial block either.
    # What the leaf does need is the fused expert stride, because this artifact
    # stores one ``ffn_gate_up_exps`` tensor per layer with the gate rows first
    # *per expert* rather than two per-expert-contiguous halves.
    if intermediate % 16:
        return False

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_wmma_tile_map,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
        build_gguf_q5_k_q8_1_selected_prefill,
        gguf_q5_k_selected_dual_sparse_exact_repair_bf16 as sparse_exact_repair,
        gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out as iu8_risk_gate_up,
    )

    wmma_starts = scratch.buffer("wmma_expert_start")
    tile_expert = scratch.buffer("wmma_tile_expert")
    wmma_total = scratch.buffer("wmma_total")
    risk_count = scratch.buffer("mmq_risk_count")
    risk_indices = scratch.buffer("mmq_risk_indices")

    library = build_gguf_q5_k_q8_1_selected_prefill(load=True)
    kwargs = {"stream": stream}
    if runtime is not None:
        kwargs["runtime"] = runtime

    tile_capacity = tile_expert.nbytes // 8
    qwen35_moe_wmma_tile_map(
        expert_start.ptr,
        wmma_starts.ptr,
        tile_expert.ptr,
        wmma_total.ptr,
        num_experts,
        tile_capacity=tile_capacity,
        **kwargs,
    )
    total_rows = _expert_tile_row_bound(compact_rows, num_experts, 16, tile_capacity)

    risk_capacity = compact_rows * 2 * intermediate
    if risk_indices.nbytes < risk_capacity * _I32_BYTES:
        raise RuntimeError(
            f"gemma4 iu8 risk queue holds {risk_indices.nbytes // _I32_BYTES} "
            f"indices but {risk_capacity} are needed"
        )
    (runtime or get_hip_runtime()).memset_async(risk_count.ptr, 0, _I32_BYTES, stream)

    weight_ptr = weight.allocation("raw").buffer.ptr
    # Q5_K stores 176 bytes per 256-element block against Q4_K's 144, so the up
    # half starts at a different offset than in the route above.
    row_bytes = (in_features // 256) * 176
    qweight_a = weight_ptr
    qweight_b = weight_ptr + intermediate * row_bytes
    iu8_risk_gate_up(
        x_ptr,
        expert_start.ptr,
        wmma_starts.ptr,
        tile_expert.ptr,
        qweight_a,
        qweight_b,
        out_ptr,
        risk_count.ptr,
        risk_indices.ptr,
        risk_capacity,
        _IU8_RISK_MULTIPLIER,
        compact_rows,
        in_features,
        intermediate,
        intermediate,
        num_experts,
        total_rows,
        expert_stride=2 * intermediate,
        library=library,
        **kwargs,
    )
    sparse_exact_repair(
        x_ptr,
        expert_start.ptr,
        qweight_a,
        qweight_b,
        out_ptr,
        risk_count.ptr,
        risk_indices.ptr,
        risk_capacity,
        compact_rows,
        in_features,
        intermediate,
        intermediate,
        num_experts,
        expert_stride=2 * intermediate,
        library=library,
        **kwargs,
    )
    return True


def gemma4_project_experts_rows(
    weight: Gemma4Projection,
    x_ptr: int,
    out_ptr: int,
    expert_start,
    selected_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
) -> str:
    """Project one block of compact rows and report the route that ran.

    Preference order, highest first:

    1. ``grouped_prefill`` -- one launch per (expert, out_col) that reads the
       expert's weight row once and reuses it across that expert's rows. Only
       when the block has at least one row per expert *and* the weight's quant
       key registers a grouped family.
    2. ``selected_gemv`` -- one block per (out_col, lane), one weight read per
       lane, over the raw blocks.
    3. ``per_expert_offset`` -- the fallback for a weight with no selected
       kernel, which costs a device-to-host read of the row counts.

    The ladder used to carry a ``pack8_selected`` rung ahead of ``selected_gemv``,
    for a weight the planner had given the packed arrays. The packed layout was
    removed on 2026-09-28: it was measured never ahead (prefill 343.2 against
    340.0 tok/s, decode 21.24 against 7.78) and the planner gave it out only below
    the one-row-per-expert threshold, so the rung was unreachable. The settlement
    is in ``docs/REFACTOR.md``.

    Routes 1 and 2 are bit-exact against each other wherever both are registered
    (pinned by ``tests/test_unit_gemma4_expert_route.py``). Route 3 is reached
    only where route 2 is absent, which is the bf16 case.
    """

    if gemma4_moe_prefill_route_enabled(
        lanes=compact_rows, num_experts=num_experts
    ) and gemma4_project_experts_grouped_prefill(
        weight,
        x_ptr,
        expert_start.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
    ):
        return "grouped_prefill"
    if gemma4_project_experts_grouped_row4(
        weight,
        x_ptr,
        expert_start.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
    ):
        return "grouped_row4"
    if gemma4_project_experts_selected(
        weight,
        x_ptr,
        selected_ptr,
        out_ptr,
        compact_rows,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
    ):
        return "selected_gemv"
    gemma4_project_experts_by_offset(
        weight,
        x_ptr,
        out_ptr,
        expert_start,
        num_experts,
        in_features,
        out_features,
        stream=stream,
    )
    return "per_expert_offset"


def _zero(buffer: DeviceBuffer, **kwargs: object) -> None:
    """Zero a device buffer through the HIP runtime.

    ``qwen35_moe_group_count`` accumulates into ``counts``, so the caller must
    supply a zeroed buffer; doing it here keeps that requirement next to the
    kernel that has it.
    """

    from hipengine.core.hip import get_hip_runtime

    runtime = kwargs.get("runtime") or get_hip_runtime()
    runtime.memset(buffer.ptr, 0, buffer.nbytes)


def _read_int64(buffer: DeviceBuffer, count: int):
    import numpy as np

    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    out = np.zeros(count, dtype=np.int64)
    copy_device_to_host(host_array_ptr(out), buffer, count * _I64_BYTES)
    return out


__all__ = ["Gemma4ExpertScratch", "gemma4_experts_forward_bf16"]
