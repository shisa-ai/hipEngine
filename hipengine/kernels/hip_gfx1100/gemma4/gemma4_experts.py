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

Weights are BF16 here. Quantised GGUF expert weights go through the separate
``gguf_expert_pack8_gemv`` path and are not wired into this function yet.
"""

from __future__ import annotations

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
_F32_BYTES = 4
# ``block_q8_1_mmq_ds4`` is ``uint16_t ds4[8]`` plus ``int8_t qs[128]``.
_DS4_BLOCK_BYTES = 144
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
            # WMMA prefill tile plan. ``expert_start`` counts compact rows; the
            # WMMA owners address padded 16-row tiles instead, so they need
            # their own per-expert start, one expert id per tile, and the total
            # padded row count. Sizes come from the routing-independent upper
            # bound in ``_wmma_tile_upper_bound``, which is why they can be
            # allocated once from shape alone.
            "wmma_expert_start": (self.num_experts + 1) * _I64_BYTES,
            "wmma_tile_expert": _wmma_tile_upper_bound(lanes, self.num_experts)[1] * _I64_BYTES,
            "wmma_total": _I64_BYTES,
            # Grouped int8 MMQ prefill. ``ds4_q8`` holds the compact activations
            # packed as llama.cpp-style DS4 ``block_q8_1_mmq`` blocks, and
            # ``compact_to_source`` is the row map the MMQ leaf dereferences. The
            # MMQ32 tile ABI is the same 16-row plan the WMMA owners build, so
            # that plan is reused and no second one is allocated here.
            "ds4_q8": lanes * (-(-self.hidden_size // 128)) * _DS4_BLOCK_BYTES,
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

    # 1. Group the lanes by expert. The parallel compaction issues its own
    #    count, prefix and scatter stages internally, so the caller-side
    #    group_count / group_prefix_active passes are redundant -- they added
    #    two launches per MoE block and nothing reads `counts` afterwards.
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
        parallel=True,
        **kwargs,
    )

    # 2. Pull the hidden rows into compact order so each expert's rows are a
    #    contiguous slice of `packed_hidden`. This happens after the route flags
    #    below rather than here, because the MMQ gate_up gathers straight out of
    #    the source rows and leaves `packed_hidden` untouched.

    # 3. One gate_up projection for every compact row, then GeGLU over the whole
    #    compact buffer in a single launch. The WMMA owners need a padded tile
    #    plan, so it is built once per call and only when a projection will
    #    actually use it.
    fused = 2 * intermediate
    mode = _prefill_mode()
    gate_up_wmma, compensated, use_mmq = _prefill_route_flags(mode)
    # The MMQ down leaf is ~8x slower than the WMMA owner at Gemma's down
    # geometry -- profiled at 947 ms against 116 ms over the same 116 launches,
    # 4.0 against 32.5 TFLOP/s -- and the MMQ gate_up already writes bf16, which
    # is exactly what the WMMA down reads. So the down keeps the WMMA owner on
    # this route and only the gate_up uses the int8 leaf.
    down_wmma = gate_up_wmma or use_mmq
    mmq_rows = 0
    if use_mmq:
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
        gate_up_wmma
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
    if use_mmq and mmq_rows and gemma4_project_experts_mmq_dual(
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
    elif mode != "selected" and gemma4_project_experts_grouped_dual(
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
    elif mode != "selected" and gemma4_project_experts_grouped(
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
    elif not gemma4_project_experts_selected(
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
    gemma4_gelu_tanh_mul_bf16(gate_up_out.ptr, activated.ptr, lanes, intermediate, **kwargs)

    # 4. The down projection, over the same compact rows.
    #
    # The gate_up consumed the 32-row MMQ plan and the WMMA owner tiles 16 rows,
    # so the plan is rebuilt at the WMMA width here. Both plans share the same
    # buffers, which is why this has to happen after the gate_up rather than
    # alongside the MMQ plan above.
    if (
        use_mmq
        and down_wmma
        and lanes >= _WMMA_PREFILL_MIN_LANES_PER_EXPERT * num_experts
    ):
        wmma_rows = _build_wmma_tile_plan(
            scratch, expert_start.ptr, lanes, stream=stream, runtime=runtime
        )
    if down_wmma and wmma_rows and gemma4_project_experts_wmma(
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
    elif mode != "selected" and gemma4_project_experts_grouped(
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
    elif not gemma4_project_experts_selected(
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

# Prefill expert-route selector. ``auto`` runs the exact routes only: the
# grouped owner where a quant registers one and the selected GEMV otherwise.
# Both measured bit-identical to the strict reference, which is what the
# campaign's logits gate requires.
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

    ``auto`` is the production route: the grouped int8 MMQ owner for the gate_up
    with the WMMA owner for the down. It is the fastest correct path measured at
    the campaign recipe (1393 tok/s against the exact grouped route's 675 at
    ``--prompt 1024``), and its logits gate sits 37x inside the calibrated
    envelope (``kl_max`` 0.001341 against a 0.05 bar, 0 of 1023 top-1 flips), so
    it is a production path rather than a bit-exact one.

    ``grouped`` and ``selected`` remain the exact routes and are the rollback
    levers if the production route ever has to be withdrawn. ``wmma`` probes the
    WMMA owners in their compensated form, ``wmma_plain`` in their uncompensated
    form, and ``mmq`` pins the production route explicitly.
    """

    if mode == "auto":
        return False, False, True
    if mode == "wmma":
        return True, True, False
    if mode == "wmma_plain":
        return True, False, False
    if mode == "mmq":
        return False, False, True
    return False, False, False


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
    base = weight.allocation("raw").buffer.ptr
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
        weight.allocation("raw").buffer.ptr,
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
    base_ptr = weight.allocation("raw").buffer.ptr
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
        weight.allocation("raw").buffer.ptr,
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
    base_ptr = weight.allocation("raw").buffer.ptr
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
        weight.allocation("raw").buffer.ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        runtime=runtime,
    )
    return True


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
    from hipengine.kernels.registry import KernelKey, MissingKernelError, resolve
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    key = KernelKey(weight.backend, "linear", weight.spec.quant_key, _SELECTED_VARIANT)
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


def _read_int64(buffer: DeviceBuffer, count: int):
    import numpy as np

    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    out = np.zeros(count, dtype=np.int64)
    copy_device_to_host(host_array_ptr(out), buffer, count * _I64_BYTES)
    return out


__all__ = ["Gemma4ExpertScratch", "gemma4_experts_forward_bf16"]
