"""Orchestrated Gemma 4 routed-expert forward on gfx1100.

Assembles the expert block from kernels that are each tested on their own:

    compact lanes per expert   qwen35_moe_group_count / _prefix_active / _compact_active
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
    qwen35_moe_group_count,
    qwen35_moe_group_prefix_active,
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

# DS4 activation block strides. The fp16 layout is for a hidden-state input and
# the fp32 one for a post-SiLU input; see gemma4_project_experts_down_mmq.
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
        return buf

    def free(self) -> None:
        for buf in self._buffers:
            hip_free(buf)
        self._buffers.clear()
        self._by_name.clear()

    def _size_of(self, name: str) -> int:
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
            # The WMMA tile walk packs 16 rows per tile against the MMQ32 route's
            # 32, so it needs its own buffers rather than the mmq_* pair above.
            "wmma_expert_start": (self.num_experts + 1) * _I64_BYTES,
            "wmma_tile_expert": (lanes // 16 + self.num_experts + 1) * _I64_BYTES,
            "wmma_total": _I64_BYTES,
        }
        try:
            return sizes[name]
        except KeyError:
            raise KeyError(f"unknown expert scratch buffer {name!r}") from None


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

    counts = scratch.buffer("counts")
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

    # 1. Group the lanes by expert. `counts` must be zeroed by the caller.
    _zero(counts, **kwargs)
    qwen35_moe_group_count(selected_experts_ptr, counts.ptr, lanes, num_experts, **kwargs)
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
    qwen35_moe_group_prefix_active(
        counts.ptr,
        expert_start.ptr,
        active_experts.ptr,
        active_count.ptr,
        num_experts,
        **kwargs,
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
    #    contiguous slice of `packed_hidden`.
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

    # 3. One gate_up projection for every compact row, then GeGLU over the whole
    #    compact buffer in a single launch. The int8-dp4a MMQ route produces both
    #    halves in one launch and returns False when the weight or the geometry
    #    does not qualify, in which case the fp32 grouped route runs instead.
    fused = 2 * intermediate
    # The MMQ gate reads the runner's declared width rather than this call's lane
    # count: the two routes are different arithmetic, so selecting by live width
    # would make a token's output depend on the batch it arrived in.
    route_width = scratch.tokens
    if not (
        gemma4_moe_gate_up_mmq_enabled()
        and gemma4_moe_prefill_route_enabled(lanes=route_width, num_experts=num_experts)
        and gemma4_project_experts_gate_up_mmq(
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
        )
    ):
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
    else:
        _record_moe_route("gate_up_mmq32")
    gemma4_gelu_tanh_mul_bf16(gate_up_out.ptr, activated.ptr, lanes, intermediate, **kwargs)

    # 4. The down projection, over the same compact rows. The Q5_1 DS4 MMQ
    # route runs first where the weight qualifies; the grouped family is the
    # fallback, and it is also what every non-Q5_1 down weight takes.
    if not (
        gemma4_moe_down_mmq_enabled()
        and gemma4_moe_prefill_route_enabled(lanes=route_width, num_experts=num_experts)
        and gemma4_project_experts_down_mmq(
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
        )
    ):
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
    else:
        _record_moe_route("down_mmq32")

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
    # Imported here rather than at module scope: the quantized dispatch lives in
    # the runtime layer and the kernel package does not depend on it otherwise.
    from hipengine.runtime.gguf_linear import launch_gguf_linear_raw_ptr

    allocation = weight.allocation("raw")
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

# The same per-lane selected shape with eight output columns per block. It is a
# bit-exact sibling of ``_SELECTED_VARIANT`` (see
# ``gemma4_project_experts_selected``) and is preferred where it is registered,
# because one block then reads the x row once and pays the block reduction once
# for eight outputs instead of one.
_SELECTED_PACK8_VARIANT = "selected_pack8_gemv_bf16_bf16_out"


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
    """

    if isinstance(weight, int):
        return False
    # Registration is not guaranteed to have survived: registry plan tests clear
    # global registrations and pytest restores a collection-time baseline, so a
    # lazy import can be a no-op and a lookup for a kernel that exists can still
    # report it missing. `_ensure_linear_kernel_registered` is the repo's answer
    # to exactly that, and is what the GGUF runtime dispatch uses.
    from hipengine.kernels.registry import (
        KernelKey,
        MissingKernelError,
        is_registered,
        resolve,
    )
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    def resolve_variant(variant: str):
        key = KernelKey(weight.backend, "linear", weight.spec.quant_key, variant)
        _ensure_linear_kernel_registered(key)
        # The resolver broadens a miss to the variant-less key and then to the
        # cpu_reference backend. That is right for dense dispatch, where a
        # variant-less registration is a legitimate substitute, and wrong here:
        # the CPU reference's ``linear`` takes array objects, not device
        # pointers, so a broadened hit is a NumPy kernel called with ints. The
        # selected family is a per-variant ABI, so require the exact key.
        if not is_registered(key):
            return None
        try:
            return resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
        except MissingKernelError:
            return None

    # Eight output columns per block is the same arithmetic as one. The k walk
    # (`k = tid; k += blockDim.x`), the per-element dequant and the reduction
    # tree are properties of an output, not of the block, and the pack8 sibling
    # reproduces all three per output -- same shuffle offsets, same per-warp
    # partial published by lane 0, same sequential sum over warps. So this is a
    # bit-exact sibling rather than a numerical candidate, and it is preferred
    # where it is registered and the width admits it, because it reads the x row
    # once per eight outputs and pays the block reduction once per eight outputs.
    fn = None
    if int(out_features) % 8 == 0:
        fn = resolve_variant(_SELECTED_PACK8_VARIANT)
    if fn is None:
        fn = resolve_variant(_SELECTED_VARIANT)
    if fn is None:
        return False
    fn(
        x_ptr,
        selected_ptr,
        weight.allocation("raw").buffer.ptr,
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
    """

    if isinstance(weight, int):
        return False
    quant_key = getattr(weight.spec, "quant_key", None)
    if quant_key == "gguf_q5_k":
        return _gemma4_project_experts_gate_up_wmma_iu8(
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
        )
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
    are produced once at materialize time, so the route adds no per-call work and
    no second copy of the tensor.
    """

    if intermediate % 16:
        return False
    if not _gemma4_t16_tiles_ready(weight):
        return False

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        host_array_ptr,
    )
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_wmma_tile_map,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
        build_gguf_q4_k_t16_selected_prefill,
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out as wmma_gate_up,
    )

    import numpy as np

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
    total_host = np.empty(1, dtype=np.int64)
    copy_device_to_host(
        host_array_ptr(total_host),
        DeviceBuffer(ptr=wmma_total.ptr, nbytes=8),
        8,
        runtime=runtime,
    )
    total_rows = int(total_host[0])
    # The T16 tile map pads to 16 rows per tile, where the MMQ32 map pads to 32.
    if total_rows <= 0 or total_rows > tile_capacity * 16:
        raise RuntimeError(
            f"gemma4 T16 gate/up tile row count {total_rows} is outside "
            f"capacity {tile_capacity * 16}"
        )

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
    _record_moe_route("gate_up_t16")
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
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        host_array_ptr,
    )
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_wmma_tile_map,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
        build_gguf_q5_k_q8_1_selected_prefill,
        gguf_q5_k_selected_dual_sparse_exact_repair_bf16 as sparse_exact_repair,
        gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out as iu8_risk_gate_up,
    )

    import numpy as np

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
    total_host = np.empty(1, dtype=np.int64)
    copy_device_to_host(
        host_array_ptr(total_host),
        DeviceBuffer(ptr=wmma_total.ptr, nbytes=8),
        8,
        runtime=runtime,
    )
    total_rows = int(total_host[0])
    if total_rows <= 0 or total_rows > tile_capacity * 16:
        raise RuntimeError(
            f"gemma4 iu8 gate/up tile row count {total_rows} is outside "
            f"capacity {tile_capacity * 16}"
        )

    risk_capacity = compact_rows * 2 * intermediate
    if risk_indices.nbytes < risk_capacity * _I32_BYTES:
        raise RuntimeError(
            f"gemma4 iu8 risk queue holds {risk_indices.nbytes // _I32_BYTES} "
            f"indices but {risk_capacity} are needed"
        )
    get_hip_runtime().memset(risk_count.ptr, 0, _I32_BYTES)

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
