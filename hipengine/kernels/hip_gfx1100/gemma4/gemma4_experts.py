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
    qwen35_moe_group_count,
    qwen35_moe_group_prefix_active,
)

_BF16_BYTES = 2
_I32_BYTES = 4
_I64_BYTES = 8
_F32_BYTES = 4


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
    lanes = scratch.total_lanes
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

    # 3. One fused gate_up GEMV per non-empty expert, then GeGLU over the whole
    #    compact buffer in a single launch.
    starts = _read_int64(expert_start, num_experts + 1)
    fused = 2 * intermediate
    for expert in range(num_experts):
        start = int(starts[expert])
        rows = int(starts[expert + 1]) - start
        if rows <= 0:
            continue
        gemma4_project_expert(
            gate_up_proj,
            expert,
            packed_hidden.ptr + start * hidden_size * _BF16_BYTES,
            gate_up_out.ptr + start * fused * _BF16_BYTES,
            rows,
            hidden_size,
            fused,
            **kwargs,
        )
    gemma4_gelu_tanh_mul_bf16(gate_up_out.ptr, activated.ptr, lanes, intermediate, **kwargs)

    # 4. One down GEMV per non-empty expert.
    for expert in range(num_experts):
        start = int(starts[expert])
        rows = int(starts[expert + 1]) - start
        if rows <= 0:
            continue
        gemma4_project_expert(
            down_proj,
            expert,
            activated.ptr + start * intermediate * _BF16_BYTES,
            expert_out.ptr + start * hidden_size * _BF16_BYTES,
            rows,
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
