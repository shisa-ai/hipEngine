"""Gemma 4 decoder layer forward on gfx1100.

Mirrors ``gemma4_decoder_layer_forward`` in the CPU reference stage for stage.
The structure that is easiest to get wrong, and that the reference pins:

    residual = hidden
    attended = attention(rmsnorm(hidden, input_layernorm))
    hidden = residual + rmsnorm(attended, post_attention_layernorm)

    # Both feed-forward branches read the SAME pre-MLP residual and run in
    # parallel; they are not stacked.
    residual = hidden
    dense = rmsnorm(down(gelu(gate(x)) * up(x)), post_feedforward_layernorm_1)
        where x = rmsnorm(residual, pre_feedforward_layernorm)
    experts = rmsnorm(moe(rmsnorm(residual, pre_feedforward_layernorm_2)),
                      post_feedforward_layernorm_2)

    out = (residual + rmsnorm(dense + experts, post_feedforward_layernorm)) * layer_scalar

Three Gemma 4 specifics, each a silent-corruption risk if assumed otherwise:

* **The norm weights apply as-is.** Gemma 4's RMSNorm computes ``normed * weight``;
  the Qwen3.5 kernels next door compute ``normed * (1 + weight)``. This module
  uses the Gemma 4 family throughout for that reason.
* **V gets a weightless norm and is never rotated.** On global (full attention)
  layers ``attention_k_eq_v`` removes ``v_proj`` entirely, so V is the *raw* K
  projection; the k_norm and the rotation apply to K only.
* **The attention scale is 1.0.** Gemma 4 folds the softmax scaling into the query
  norm weight, so this passes ``geometry.scale`` rather than ``head_dim**-0.5``.

Attention here is the dense prefill form: it attends over the block it is given
using a caller-supplied keep mask. Paged-KV decoding is the runner's concern and
plugs in at the attention step without touching the rest of the layer.

A block may also be **physically packed** from several requests: the dense and
MoE stages run once over every row, while the rotation, the cache append and the
attention launch are repeated per request through a
:class:`Gemma4PackedSegment`. That is what makes a multi-request verify one
forward pass instead of a loop of full forwards -- the projections are shared,
the KV is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, free as hip_free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    DECODE_ATTENTION_FLASH,
    DECODE_ATTENTION_FLASH_LONG,
    DECODE_ATTENTION_LAYER,
    DECODE_ATTENTION_PLAIN,
    DECODE_ATTENTION_QUANT,
    DECODE_ATTENTION_STAGED,
    EXACT_PREFILL_ATTENTION_LAYER,
    EXACT_PREFILL_ATTENTION_PLAIN,
    EXACT_PREFILL_ATTENTION_QUANT,
    EXACT_PREFILL_ATTENTION_STAGED,
    Gemma4AttentionScratch,
    aotriton_prefill_admits,
    FLASH_DECODE_PREFERRED_MIN_KEYS,
    FLASH_MULTI_ROW_MAX_TOKENS,
    flash_admits,
    flash_decode_short_row,
    gemma4_attention_prefill_aotriton,
    gemma4_attention_prefill_bf16,
    gemma4_attention_resident_overflows,
    select_prefill_attention,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import (
    Gemma4AttentionTiledUnsupported,
    gemma4_attention_prefill_tiled,
    gemma4_attention_tiled_admits,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    Gemma4ExpertScratch,
    gemma4_experts_forward_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
    gemma4_gelu_tanh_mul_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_add_rmsnorm_scale_bf16,
    gemma4_dense_combine_rmsnorm_scale_bf16,
    gemma4_head_rmsnorm_f32w_bf16,
    gemma4_qkv_split_bf16,
    gemma4_rmsnorm_f32w_bf16,
    gemma4_rmsnorm_weightless_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
    gemma4_partial_rotary_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_types import Gemma4Projection
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_router import (
    Gemma4RouterScratch,
    gemma4_router_topk_bf16,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_bf16

_BF16_BYTES = 2
_I64_BYTES = 8
_F32_BYTES = 4


@dataclass
class Gemma4LayerPointers:
    """Weights for one layer.

    A projection field is either a **bf16 device pointer** (``int``) or a
    **GGUF device weight** carrying quantized blocks. Both are real resident
    layouts -- the bf16 form is what a bf16 artifact and the layer's own
    synthetic fixtures use, the GGUF form is what a quantized artifact uses --
    so :func:`gemma4_project` dispatches on the value's storage form rather than
    on a flag. The layer's arithmetic is identical either way; only how the
    ``(rows, in) x (in, out)`` product is computed differs.

    ``v_proj`` is ``0`` on ``attention_k_eq_v`` layers, where the reference takes
    the raw K projection as V and there is no ``v_proj`` tensor in the artifact.
    ``layer_scalar`` is ``0`` to mean 1.0, which is what the kernel does with a
    null pointer. The norm and router fields stay raw pointers: they are f32
    weight vectors in every artifact, never quantized blocks.
    """

    input_layernorm: int
    q_proj: Gemma4Projection
    k_proj: Gemma4Projection
    o_proj: Gemma4Projection
    q_norm: int
    k_norm: int
    post_attention_layernorm: int
    pre_feedforward_layernorm: int
    mlp_gate_up_proj: Gemma4Projection
    mlp_down_proj: Gemma4Projection
    post_feedforward_layernorm_1: int
    router_scale: int
    router_proj: int
    router_per_expert_scale: int
    pre_feedforward_layernorm_2: int
    experts_gate_up_proj: Gemma4Projection
    experts_down_proj: Gemma4Projection
    post_feedforward_layernorm_2: int
    post_feedforward_layernorm: int
    v_proj: Gemma4Projection = 0
    layer_scalar: int = 0
    # One resident q|k|v weight, set only when the artifact's storage let the
    # loader fuse all three and the layer is not k_eq_v. ``0`` means "project
    # the three separately", which is the unfused path every artifact could
    # always take; the layer picks the path from this value rather than from a
    # flag, so what actually runs is decided by what was loaded.
    qkv_proj: Gemma4Projection = 0


def gemma4_project(
    x_ptr: int,
    weight: Gemma4Projection,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    use_wmma_prefill: bool = True,
) -> None:
    """Run one ``(rows, in_features) x (in_features, out_features)`` projection.

    ``int`` is a bf16 device pointer; anything else is a GGUF device weight and
    goes through the quantized linear dispatch, which picks its own kernel from
    the weight's quant key and the row count.

    Keep WMMA prefill eligible for every dense projection, including GeGLU
    down. T16 Q8 kernels repair non-finite WMMA accumulators in FP32 from the
    original activation rather than disabling prefill for the whole projection.

    ``use_wmma_prefill=False`` keeps a multi-row block on the row-exact dense
    schedule, whose every row is bit-identical to the one-row decode kernel. A
    speculative verify block asks for it when its rows have to equal one-token
    decode.
    """

    if isinstance(weight, int):
        dense_gemv_out_bf16(x_ptr, weight, out_ptr, rows, in_features, out_features, stream=stream)
        return
    # Imported here rather than at module scope: the quantized dispatch lives in
    # the runtime layer and the kernel package does not depend on it otherwise.
    from hipengine.runtime.gguf_linear import launch_gguf_linear

    launch_gguf_linear(
        weight,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        use_wmma_prefill=use_wmma_prefill,
    )


@dataclass(frozen=True)
class Gemma4RowDecodeAttention:
    """What each row of a block would have been handed as a one-token step.

    ``key_begins`` and ``windows`` are per row, cache-relative, exactly the
    ``key_begin`` and ``window`` the caller passes for a one-row block at that
    row's position; the two variant requests are the ones a decode step carries.
    """

    key_begins: tuple[int, ...]
    windows: tuple[int, ...]
    prefill_attention_variants: tuple[str, ...] | None = None
    decode_attention_variants: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Gemma4PackedSegment:
    """One request's rows inside a physically packed block.

    A packed block concatenates several requests' rows into one physical block
    so the dense projections and the MoE stage run once over all of them.
    Everything that is a property of the *request* rather than of the row stays
    per segment: its own KV cache and write offset, its own sliding-window base
    and keep mask, its own RoPE tables (positions differ), and its own
    attention-variant requests.

    The row ranges must tile ``[0, total)`` in order with no gap and no overlap;
    :func:`gemma4_packed_segment_rows` is the check, and the layer forward runs
    it before any device work.

    ``key_begin`` and ``window`` are cache-relative, exactly as the one-request
    call's arguments of the same name: ``key_begin`` is the first key the
    request's rows read (already translated by its sliding base), and ``window``
    bounds each row's walk past that. ``keep_mask_ptr`` is the request's own
    mask, whose column 0 is key ``key_begin`` in absolute terms -- the same
    relationship the one-request path documents.

    ``row_decode_attention`` is that request's per-row decode ranges, for a
    speculative verify block whose rows must each repeat the one-token step; a
    ``None`` attends the segment's rows in one launch, and a one-row segment is
    attended row-exactly either way.

    ``row_exact`` says the request's rows must each repeat the one-token step.
    A one-row segment carries no ``row_decode_attention`` -- one row already is
    that computation -- so this flag is what still tells the block-wide router
    loop that four one-row requests are four decode steps and not one four-row
    projection, whose reduction order differs in the last place.
    """

    row_start: int
    rows: int
    kv: Gemma4LayerKV
    cos_ptr: int
    sin_ptr: int
    keep_mask_ptr: int
    key_begin: int = 0
    window: int = 0
    row_exact: bool = False
    row_decode_attention: Gemma4RowDecodeAttention | None = None
    prefill_attention_variants: tuple[str, ...] | None = None
    decode_attention_variants: tuple[str, ...] | None = None
    exact_prefill_attention_variants: tuple[str, ...] | None = None


def gemma4_packed_segment_rows(segments: Sequence[Gemma4PackedSegment]) -> int:
    """Validate a packed block's segmentation and return its total row count.

    The whole point of packing is that a row lands in exactly one request's row
    range, so a gap, an overlap or a segment with no cache is a caller bug that
    would otherwise write one request's keys into another's cache. This runs
    before any launch, so a malformed packing costs no device work.
    """

    if not segments:
        raise ValueError("a packed block needs at least one segment")
    total = 0
    for index, segment in enumerate(segments):
        if segment.rows <= 0:
            raise ValueError(f"packed segment {index} covers {segment.rows} rows")
        if int(segment.row_start) != total:
            raise ValueError(
                f"packed segment {index} starts at row {segment.row_start} but the "
                f"segments before it cover rows 0..{total - 1}"
            )
        if segment.kv is None:
            raise ValueError(f"packed segment {index} carries no KV cache")
        if int(segment.key_begin) < 0:
            raise ValueError(
                f"packed segment {index} has key_begin {segment.key_begin}; it must "
                "be non-negative and cache-relative"
            )
        if int(segment.window) < 0:
            raise ValueError(f"packed segment {index} has window {segment.window}")
        if not int(segment.keep_mask_ptr):
            raise ValueError(f"packed segment {index} carries no keep mask")
        if not int(segment.cos_ptr) or not int(segment.sin_ptr):
            raise ValueError(f"packed segment {index} carries no rope tables")
        if segment.row_decode_attention is not None and len(
            segment.row_decode_attention.key_begins
        ) != int(segment.rows):
            raise ValueError(
                f"packed segment {index} carries "
                f"{len(segment.row_decode_attention.key_begins)} row ranges for "
                f"{segment.rows} rows"
            )
        total += int(segment.rows)
    return total


@dataclass
class Gemma4LayerGeometry:
    """The attention geometry fields the layer forward needs."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float = 1.0
    k_eq_v: bool = False
    sliding_window: int | None = None


class Gemma4SharedLayerBuffers:
    """One set of layer temporaries that a runner's layers take in turn.

    Every named buffer is transient within one layer forward: the layer writes
    its result to the caller's ``hidden_ptr``, and the next layer's launches are
    ordered behind this one's on the same streams. One buffer per name, sized
    for the widest layer, therefore serves every layer. Per-layer owners
    multiplied these temporaries by the layer count in each runner, and a
    resident server builds several runners.

    Sizes are fixed when the pool is built and never grow, so a captured decode
    graph can hold the pointers.

    ``PER_LAYER`` names stay with each layer: the decode graph recognizes a
    layer's KV-append memcpy by its source, the layer's own rotated-K and V
    buffers, so those pointers must differ between layers.
    """

    PER_LAYER = frozenset({"k_rot", "v"})

    def __init__(self, sizes: dict[str, int]) -> None:
        self._sizes = dict(sizes)
        self._buffers: list[DeviceBuffer] = []
        self._by_name: dict[str, DeviceBuffer] = {}

    @classmethod
    def for_layers(cls, layers: "Sequence[Gemma4LayerScratch]") -> "Gemma4SharedLayerBuffers":
        sizes: dict[str, int] = {}
        for layer in layers:
            for name, nbytes in layer._sizes().items():
                if name not in cls.PER_LAYER:
                    sizes[name] = max(sizes.get(name, 0), int(nbytes))
        return cls(sizes)

    def buffer(self, name: str) -> DeviceBuffer:
        existing = self._by_name.get(name)
        if existing is not None:
            return existing
        try:
            nbytes = self._sizes[name]
        except KeyError:
            raise ValueError(f"unknown scratch buffer {name!r}") from None
        buf = malloc(nbytes)
        self._by_name[name] = buf
        self._buffers.append(buf)
        return buf

    def resident_bytes(self) -> int:
        return sum(self._sizes.values())

    def free(self) -> None:
        for buf in self._buffers:
            hip_free(buf)
        self._buffers.clear()
        self._by_name.clear()


@dataclass
class Gemma4LayerScratch:
    """Reusable device scratch for one layer shape.

    Owns the expert and router scratches too, so a caller allocates once per
    shape and reuses across layers and decode steps.
    """

    tokens: int
    hidden_size: int
    dense_intermediate: int
    geometry: Gemma4LayerGeometry
    num_experts: int
    top_k: int
    expert_intermediate: int
    _buffers: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _by_name: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)
    experts: Gemma4ExpertScratch | None = field(default=None, repr=False)
    router: Gemma4RouterScratch | None = field(default=None, repr=False)
    attention: Gemma4AttentionScratch = field(default_factory=Gemma4AttentionScratch, repr=False)
    # A runner's layers can share one temporaries pool; see
    # ``Gemma4SharedLayerBuffers``. Without one, this scratch owns its own.
    shared_buffers: Gemma4SharedLayerBuffers | None = field(default=None, repr=False)
    # Pair of HIP events gating the parallel MoE branch: one recorded on the
    # main stream after attention so the branch cannot read `hidden` early, one
    # recorded on the branch stream so the combine cannot run ahead of it.
    # Created on first use, because a per-forward event would cost a
    # hipEventCreate per layer per step.
    _moe_entry_event: int | None = field(default=None, repr=False)
    _moe_exit_event: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("tokens", self.tokens),
            ("hidden_size", self.hidden_size),
            ("dense_intermediate", self.dense_intermediate),
            ("num_experts", self.num_experts),
            ("top_k", self.top_k),
            ("expert_intermediate", self.expert_intermediate),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        # A runner hands every layer one shared expert owner, as it does the
        # attention owner; a standalone layer scratch builds its own.
        self._owns_experts = self.experts is None
        if self.experts is None:
            self.experts = Gemma4ExpertScratch(
                tokens=self.tokens,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate=self.expert_intermediate,
                num_experts=self.num_experts,
            )
        self.router = Gemma4RouterScratch(
            tokens=self.tokens,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

    @property
    def num_heads(self) -> int:
        return self.geometry.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.geometry.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.geometry.head_dim

    def buffer(self, name: str) -> DeviceBuffer:
        """Allocate ``name`` on first use, then reuse it."""

        shared = self.shared_buffers
        if shared is not None and name not in shared.PER_LAYER:
            return shared.buffer(name)
        existing = self._by_name.get(name)
        if existing is not None:
            return existing
        buf = malloc(self._size_of(name))
        self._by_name[name] = buf
        self._buffers.append(buf)
        return buf

    def free(self) -> None:
        # Acquiring the HIP runtime is itself a device action: it loads
        # libamdhip64.so. An unused scratch owns no events, no buffers and no
        # sub-scratch buffers, so releasing it must be a pure host operation.
        # Only an event actually created by the parallel MoE branch needs the
        # runtime, and it is created with one, so its presence implies the
        # runtime was already acquired.
        if self._moe_entry_event is not None or self._moe_exit_event is not None:
            from hipengine.core.hip import get_hip_runtime

            runtime = get_hip_runtime()
            for event in (self._moe_entry_event, self._moe_exit_event):
                if event is not None:
                    runtime.event_destroy(event)
        self._moe_entry_event = None
        self._moe_exit_event = None
        self.attention.close()
        if self.shared_buffers is not None:
            self.shared_buffers.free()
        if self.experts is not None:
            self.experts.free()
        if self.router is not None:
            self.router.free()
        for buf in self._buffers:
            hip_free(buf)
        self._buffers.clear()
        self._by_name.clear()

    def _sizes(self) -> dict[str, int]:
        """Byte size of every buffer this scratch can hold, at ``tokens`` rows.

        Built as one table so the allocator and the memory planner cannot
        drift apart: ``_size_of`` looks a name up here, and ``resident_bytes``
        sums it.
        """
        rows = self.tokens
        hidden = self.hidden_size
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        sizes = {
            "normalized": rows * hidden * _BF16_BYTES,
            "q": rows * q_width * _BF16_BYTES,
            # Transient fused projection output. The fused weight writes one
            # [rows, q + 2*kv] buffer and the split copies it out to q/k/v, so
            # this is what the extra launch costs in memory: it doubles the
            # q/k/v region. In-place rearrangement does not avoid it -- writing
            # the packed blocks clobbers source rows not yet read -- which the
            # P6 split investigation checked directly.
            "qkv": rows * (q_width + 2 * kv_width) * _BF16_BYTES,
            "k": rows * kv_width * _BF16_BYTES,
            "k_normed": rows * kv_width * _BF16_BYTES,
            "v": rows * kv_width * _BF16_BYTES,
            "q_rot": rows * q_width * _BF16_BYTES,
            "k_rot": rows * kv_width * _BF16_BYTES,
            "context": rows * q_width * _BF16_BYTES,
            "attn_out": rows * hidden * _BF16_BYTES,
            "hidden": rows * hidden * _BF16_BYTES,
            # One buffer holds both halves: the fused weight projects into
            # (rows, 2 * intermediate) in a single launch, and the fused GeGLU
            # reads it the way the expert path reads its stacked gate/up.
            "dense_gate_up": rows * 2 * self.dense_intermediate * _BF16_BYTES,
            "dense_act": rows * self.dense_intermediate * _BF16_BYTES,
            "dense": rows * hidden * _BF16_BYTES,
            # The dense and MoE branches run on separate streams, so the MoE
            # branch's pre-FFN normalisation gets its own buffer rather than
            # sharing `normalized` with the dense branch. Sharing would race:
            # both write it from `hidden` and read it for their projections.
            "moe_normalized": rows * hidden * _BF16_BYTES,
            "experts": rows * hidden * _BF16_BYTES,
            "selected": rows * self.top_k * _I64_BYTES,
            "routing": rows * self.top_k * _F32_BYTES,
            # "branch_sum" existed for the pre-D6 chain
            # (branch_add -> add_rmsnorm_scale). The tail fold computes the sum
            # in registers, so the buffer has no consumer and no allocation.
        }
        return sizes

    def _size_of(self, name: str) -> int:
        try:
            return self._sizes()[name]
        except KeyError:
            raise ValueError(f"unknown scratch buffer {name!r}") from None

    def resident_bytes(self) -> int:
        """Upper bound on every device buffer this scratch can take.

        Deliberately a sum over the whole table rather than over the names one
        forward path happens to touch: the memory planner must not assume a
        path, and over-estimating only costs a smaller block.
        """
        # Shared owners are charged once by whoever holds them, not per layer.
        shared = self.shared_buffers
        total = sum(
            nbytes
            for name, nbytes in self._sizes().items()
            if shared is None or name in shared.PER_LAYER
        )
        if self.experts is not None and self._owns_experts:
            total += self.experts.resident_bytes()
        if self.router is not None:
            total += self.router.resident_bytes()
        return total


@dataclass(frozen=True)
class Gemma4LayerKV:
    """Where this layer reads and appends key/value state.

    ``key_cache`` and ``value_cache`` are each ``(capacity, num_kv_heads,
    head_dim)`` BF16, allocated by the runner and shared across steps. This block
    of ``rows`` tokens appends at ``write_offset`` and then attends over
    ``write_offset + rows`` cached positions.
    """

    key_cache: int
    value_cache: int
    capacity: int
    write_offset: int


def _append_kv(
    cache_ptr: int, src_ptr: int, element_offset: int, elements: int, stream: int
) -> None:
    """Copy ``elements`` BF16 values from ``src_ptr`` into the cache.

    A plain device-to-device move: both sides are contiguous over
    ``(position, num_kv_heads, head_dim)``, so appending is an offset, not a
    scatter.
    """

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime

    if elements <= 0:
        return
    runtime = get_hip_runtime()
    runtime.memcpy_async(
        cache_ptr + element_offset * _BF16_BYTES,
        src_ptr,
        elements * _BF16_BYTES,
        HipMemcpyKind.DEVICE_TO_DEVICE,
        stream,
    )


_last_prefill_route: str | None = None

# The writer/consumer keys the most recent INT8 layer forward selected. The
# registry is the selection mechanism; this records what it returned so a
# caller can confirm the intended kernels ran rather than inferring it from
# finite output. Diagnostic only; nothing reads it to decide anything.
_last_int8_route: dict[str, str] | None = None

_INT8_ATTENTION_QUANT = "int8_per_token_head"
_INT8_ATTENTION_VARIANT = "gemma4_direct_spans"


def last_int8_kv_route() -> dict[str, str] | None:
    """The registered writer/consumer keys the last INT8 layer forward selected."""

    return None if _last_int8_route is None else dict(_last_int8_route)


def _select_int8_kv_kernels(int8_kv, rows: int):
    """Resolve the INT8 writer and attention consumer from the registry.

    The writer route comes from ``plan_paged_kv_write`` on the writer spans
    (BF16 source, per-token/head granularity); the consumer is the Gemma 4
    direct-spans variant registered by ``gemma4_attention_int8``. Selection is
    by declared capability and launch shape, never by model or artifact
    identity.
    """

    from hipengine.dispatch.kv import PagedKVWriteKind, plan_paged_kv_write
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    # Importing the writer module registers its kernels; nothing else on the
    # Gemma runtime path imports it.
    from hipengine.kernels.hip_gfx1100.attention import paged_kv_write  # noqa: F401
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_int8 import (
        register_gemma4_int8_attention_kernels,
    )

    backend = str(int8_kv.backend)
    kind = PagedKVWriteKind.PROMPT if int(rows) > 1 else PagedKVWriteKind.DECODE
    write_selection = plan_paged_kv_write(
        int8_kv.writer_spans, kind=kind, source_dtype=DType.BF16
    )
    writer = resolve(
        backend=backend,
        layer=write_selection.layer,
        quant=write_selection.quant,
        variant=write_selection.variant,
    )
    consumer_layer = "paged_attn_prefill" if int(rows) > 1 else "paged_attn_decode"
    consumer_key = KernelKey(
        backend, consumer_layer, _INT8_ATTENTION_QUANT, _INT8_ATTENTION_VARIANT
    )
    if not is_registered(consumer_key):
        register_gemma4_int8_attention_kernels()
    consumer = resolve(
        backend=backend,
        layer=consumer_layer,
        quant=_INT8_ATTENTION_QUANT,
        variant=_INT8_ATTENTION_VARIANT,
    )
    return write_selection, writer, consumer_layer, consumer


def _run_int8_attention(
    int8_kv,
    *,
    query_bf16: int,
    key_bf16: int,
    value_bf16: int,
    context_bf16: int,
    rows: int,
    geometry: Gemma4LayerGeometry,
    stream: int,
) -> None:
    """Append this block to the INT8 cache and attend over it.

    The consumers take FP32 queries and emit FP32 rows, so the BF16 query is
    widened into the owner's FP32 scratch and the FP32 context is narrowed back
    into the layer's BF16 ``context`` buffer. The reconstruction is the
    writer's own representation -- ``float32(int8) * float32(scale)`` -- never a
    BF16 rounding of it.
    """

    from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16

    global _last_int8_route

    num_heads = geometry.num_heads
    num_kv_heads = geometry.num_kv_heads
    head_dim = geometry.head_dim
    q_width = num_heads * head_dim
    write_selection, writer, consumer_layer, consumer = _select_int8_kv_kernels(
        int8_kv, rows
    )
    _last_int8_route = {
        "writer": f"{write_selection.quant}/{write_selection.variant}",
        "consumer": f"{consumer_layer}/{_INT8_ATTENTION_QUANT}/{_INT8_ATTENTION_VARIANT}",
    }

    bf16_to_f32(query_bf16, int8_kv.query_f32, rows * q_width, stream=stream)
    if rows > 1:
        writer(
            key_bf16,
            value_bf16,
            int8_kv.key_cache,
            int8_kv.value_cache,
            int8_kv.k_scale,
            int8_kv.v_scale,
            int8_kv.writer_spans,
            rows,
            int8_kv.block_size,
            num_kv_heads,
            head_dim,
            stream=stream,
        )
    else:
        writer(
            key_bf16,
            value_bf16,
            int8_kv.key_cache,
            int8_kv.value_cache,
            int8_kv.k_scale,
            int8_kv.v_scale,
            int8_kv.writer_spans,
            int8_kv.block_size,
            num_kv_heads,
            head_dim,
            stream=stream,
        )
    # The multi-row prefill consumer takes the row count between its spans and
    # its context bound; the decode consumer has no row argument because it is
    # always one row. Passing the wrong arity is a launch-time metadata refusal,
    # not a wrong answer, so the two shapes are built explicitly.
    consumer_args = [
        int8_kv.query_f32,
        int8_kv.key_cache,
        int8_kv.value_cache,
        int8_kv.context_f32,
        int8_kv.consumer_spans,
    ]
    if rows > 1:
        consumer_args.append(rows)
    consumer_args.extend(
        [
            int8_kv.max_context_len,
            int8_kv.block_size,
            num_heads,
            num_kv_heads,
            head_dim,
            geometry.scale,
            geometry.sliding_window,
        ]
    )
    consumer(*consumer_args, stream=stream)
    f32_to_bf16(int8_kv.context_f32, context_bf16, rows * q_width, stream=stream)

# The second stream the parallel MoE branch runs on, created once per process.
# Deliberately module-level rather than per-runner: HIP streams are cheap and
# this mirrors the single default stream the rest of the layer already assumes.
_moe_stream_cache: int | None = None


def _moe_stream() -> int:
    global _moe_stream_cache
    if _moe_stream_cache is None:
        from hipengine.core.hip import get_hip_runtime

        _moe_stream_cache = get_hip_runtime().stream_create(nonblocking=True)
    return _moe_stream_cache


def last_prefill_attention_route() -> str | None:
    """Which prefill kernel the most recent layer forward selected.

    Routing is a per-block capability decision, so the only honest way to
    confirm which path executed is to read what was selected: finite output
    proves the layer ran, not that it ran the kernel you intended. This is
    diagnostic state -- nothing reads it to decide anything -- and ``None``
    before the first call.
    """

    return _last_prefill_route


def _select_prefill_route(
    *,
    rows: int,
    keys: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
    mask_is_causal: bool,
) -> str:
    """Pick a prefill kernel by what the request *is*, never by where it came from.

    Three routes, each admitted by arithmetic on the dimensions themselves:

    ``tiled``
        head_dim 512 with a key count that is a multiple of 128 and head counts
        that tile evenly -- Gemma 4's five global layers. A flash formulation
        with an online softmax, ported from llama.cpp's fattn-tile.
    ``aotriton``
        head_dim 256 with a mask the flash kernel can re-derive itself. The
        vendored image set covers the sliding layers only.
    ``exact``
        everything else: the correctness-first three-pass block kernel, which
        accepts any shape.

    The two specialized sets are disjoint today (256 vs 512), so priority never
    has to choose between them, but the order is still stated: a working path
    takes the cheaper formulation, and a shape no specialized kernel can execute
    falls through to ``exact`` instead of being refused. Nothing here consults a
    model name, a file, a hash, or whether a combination has been benchmarked.
    """

    try:
        gemma4_attention_tiled_admits(
            tokens=rows,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            keys=keys,
        )
    except Gemma4AttentionTiledUnsupported:
        pass
    else:
        return "tiled"

    if aotriton_prefill_admits(
        rows=rows,
        keys=keys,
        head_dim=head_dim,
        sliding_window=sliding_window,
        mask_is_causal=mask_is_causal,
    ):
        return "aotriton"
    return "exact"


# The strict native/exact entry's own decode kernel holds one FP32 logit per
# live key in shared memory. Past 15,616 keys at head_dim 512 it keeps the same
# arithmetic and moves the logit row to owned global scratch (the launcher's
# ``global_class`` variant), which is correct but slow. The staged singleton
# candidate computes the identical arithmetic in bounded shared memory, so a
# production profile may prefer it for exactly that shape. The preference is
# scoped here, at the one call site the exact route owns, and never touches the
# tiled, aotriton, or profile-selected prefill routes.
_EXACT_DECODE_STAGED = DECODE_ATTENTION_STAGED


def _resolve_decode_attention_launcher(*, backend: str, variant: str):
    """Resolve one decode-attention variant from the kernel registry.

    The registration is executable, not decorative: the layer runs whatever
    callable the registry holds for this key, so a replaced entry is what runs.
    A missing entry is a named :class:`MissingKernelError` rather than a silent
    fall back to a hard-coded wrapper.
    """

    from hipengine.kernels.registry import resolve

    return resolve(
        backend=backend,
        layer=DECODE_ATTENTION_LAYER,
        quant=DECODE_ATTENTION_QUANT,
        variant=variant,
    )


def _select_exact_decode_attention(
    *,
    decode_attention_variants: tuple[str, ...] | None,
    backend: str | None = None,
    tokens: int,
    keys: int | None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
    key_begin: int = 0,
):
    """Resolve the launcher for the native/exact attention call site.

    Returns ``(launcher, route)`` where ``route`` is what the diagnostic
    reporter should show. The staged candidate is chosen only where it was
    verified against the native ``global_class`` route bit for bit:

    * one query token (``tokens == 1``) -- a decode step, not a prefill block or
      a speculative verify block, which arrive multi-row;
    * ``head_dim`` 512 with a GQA ratio of 8 -- Gemma 4's five global layers;
    * a key count past the resident LDS bound, so the native path is the slow
      owned-global-logits route this candidate replaces;
    * the untrimmed global scope: no sliding window on the layer
      (``sliding_window is None``) and no caller-dropped prefix
      (``key_begin == 0``). A sliding layer's decode is windowed and a trimmed
      global call skips keys the candidate was not verified against, so both
      keep the native kernel.

    Both launchers come from the registry under ``DECODE_ATTENTION_LAYER``: the
    registered strict fallback (``gemma4_plain``) for every call that keeps the
    native arithmetic, and the staged candidate (``gemma4_staged``) for the
    scoped shape. A caller that requests no decode scope keeps the strict entry
    point directly, because nothing asked the registry for the scope.
    """

    if not decode_attention_variants:
        return gemma4_attention_prefill_bf16, "exact"
    resolved_backend = str(backend) if backend is not None else "hip_gfx1100"
    plain = _resolve_decode_attention_launcher(
        backend=resolved_backend, variant=DECODE_ATTENTION_PLAIN
    )

    def keep_native():
        return plain, "exact"

    # The long-row variant is the same route with a different slice count, so
    # every scope test below treats the two alike; the plan's request decides
    # which registered launcher a flash decision resolves to.
    flash_variant = (
        DECODE_ATTENTION_FLASH_LONG
        if DECODE_ATTENTION_FLASH_LONG in decode_attention_variants
        else DECODE_ATTENTION_FLASH
        if DECODE_ATTENTION_FLASH in decode_attention_variants
        else None
    )
    if (
        flash_variant is None
        and DECODE_ATTENTION_STAGED not in decode_attention_variants
    ):
        return keep_native()
    if int(tokens) != 1:
        # A multi-row block under a decode scope is a speculative verify block.
        # The flash variants serve it in one launch on either packing once the
        # block reaches the preferred entry; a sliding layer's trimmed range is
        # only a key offset, carried by the K/V pointers and each row's mask.
        # Anything else -- a block too wide, a geometry without a flash
        # instantiation, a short range -- keeps the native multi-row kernel.
        if (
            flash_variant is not None
            and keys is not None
            and int(keys) >= FLASH_DECODE_PREFERRED_MIN_KEYS
            and flash_admits(
                tokens=int(tokens), head_dim=int(head_dim), num_heads=int(num_heads),
                num_kv_heads=int(num_kv_heads), max_tokens=FLASH_MULTI_ROW_MAX_TOKENS,
            )
        ):
            flash = _resolve_decode_attention_launcher(
                backend=resolved_backend, variant=flash_variant
            )
            return flash, flash_variant
        return keep_native()
    if (
        flash_variant is not None
        and keys is not None
        and int(key_begin) == 0
        and flash_admits(
            tokens=1, head_dim=int(head_dim), num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
        )
        and not gemma4_attention_resident_overflows(head_dim=int(head_dim), keys=int(keys))
        and flash_decode_short_row(keys=int(keys), head_dim=int(head_dim))
    ):
        # The short-row arm of the same preference, for either packing: below
        # the split policy's 1024 keys the native launch is the single class
        # kernel, whose serial key walk the key-sliced route overtakes from
        # ``FLASH_DECODE_PREFERRED_MIN_KEYS`` keys. From 1024 keys the native launch
        # already asks for flash, so this changes nothing there. A sliding
        # layer has not started trimming at these counts (``key_begin == 0``).
        flash = _resolve_decode_attention_launcher(
            backend=resolved_backend, variant=flash_variant
        )
        return flash, flash_variant
    if int(head_dim) != 512:
        return keep_native()
    if int(num_kv_heads) <= 0 or int(num_heads) % int(num_kv_heads):
        return keep_native()
    if int(num_heads) // int(num_kv_heads) != 8:
        return keep_native()
    if keys is None or int(keys) <= 0:
        return keep_native()
    if not gemma4_attention_resident_overflows(head_dim=512, keys=int(keys)):
        return keep_native()
    if sliding_window is not None:
        return keep_native()
    if int(key_begin) != 0:
        return keep_native()
    if flash_variant is not None:
        # The flash candidate is the same scoped shape as the staged one and
        # replaces it in the production plan: one token, head_dim 512, GQA
        # ratio 8, untrimmed global, past the resident bound. It is resolved
        # after the guards rather than before them, so every shape outside that
        # scope -- a prefill block, a sliding layer, a trimmed call -- keeps the
        # native kernel. The staged entry stays registered as the exact chain
        # strict and batch-invariant keep, and as this route's rollback lever.
        flash = _resolve_decode_attention_launcher(
            backend=resolved_backend, variant=flash_variant
        )
        return flash, flash_variant
    staged = _resolve_decode_attention_launcher(
        backend=resolved_backend, variant=DECODE_ATTENTION_STAGED
    )
    return staged, DECODE_ATTENTION_STAGED


def _resolve_exact_prefill_attention_launcher(*, backend: str, variant: str):
    """Resolve one exact-prefill attention variant from the kernel registry.

    Mirrors :func:`_resolve_decode_attention_launcher`: the layer runs whatever
    callable the registry holds for this key, so a replaced entry is what runs,
    and a missing entry is a named :class:`MissingKernelError` rather than a
    silent fall back to a hard-coded wrapper.
    """

    from hipengine.kernels.registry import resolve

    return resolve(
        backend=backend,
        layer=EXACT_PREFILL_ATTENTION_LAYER,
        quant=EXACT_PREFILL_ATTENTION_QUANT,
        variant=variant,
    )


def _select_exact_prefill_attention(
    *,
    exact_prefill_attention_variants: tuple[str, ...] | None,
    backend: str | None = None,
    tokens: int,
    keys: int | None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
):
    """Resolve the launcher for the native/exact prefill call site.

    Returns ``(launcher, route)`` when the exact-prefill scope applies to this
    call: ``(registered_plain, "exact")`` when it keeps the native arithmetic and
    ``(staged, "gemma4_staged")`` when it selects the candidate. It returns
    ``(None, None)`` when the call is outside the scope, so the caller keeps
    whatever launcher the decode scope already resolved -- a global block and a
    one-row block are both outside it, and overriding their resolved callable
    would route a call the scope never claimed.

    The scope is the native/exact sliding multi-row geometry: the block the
    tiled and AOTriton routes refused, with ``tokens > 1`` and a sliding layer.
    Within it the staged candidate is chosen only for a shape
    :func:`staged_plan` accepts -- its warp-key score tree covers ``head_dim``
    256/512, the GQA ratio must be integral, and the launch grids and workspace
    must be representable. A shape it cannot execute keeps the registered strict
    fallback, so the prefill scope's own registration is the callable that runs
    for the strict arm too.

    Both launchers come from the registry under ``EXACT_PREFILL_ATTENTION_LAYER``:
    the registered strict fallback (``gemma4_plain``) for every call that keeps
    the native arithmetic, and the staged candidate (``gemma4_staged``) for the
    scoped shape.
    """

    if not exact_prefill_attention_variants:
        return None, None
    if int(tokens) <= 1:
        return None, None
    if sliding_window is None:
        return None, None
    if keys is None or int(keys) <= 0:
        return None, None

    resolved_backend = str(backend) if backend is not None else "hip_gfx1100"

    def keep_native():
        return (
            _resolve_exact_prefill_attention_launcher(
                backend=resolved_backend, variant=EXACT_PREFILL_ATTENTION_PLAIN
            ),
            "exact",
        )

    if EXACT_PREFILL_ATTENTION_STAGED not in exact_prefill_attention_variants:
        return keep_native()

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_staged import (
        staged_plan,
    )

    try:
        staged_plan(
            tokens=int(tokens),
            keys=int(keys),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
        )
    except (ValueError, NotImplementedError):
        return keep_native()

    staged = _resolve_exact_prefill_attention_launcher(
        backend=resolved_backend, variant=EXACT_PREFILL_ATTENTION_STAGED
    )
    return staged, EXACT_PREFILL_ATTENTION_STAGED


def _layer_backend(layer: Gemma4LayerPointers) -> str | None:
    """The backend of whichever projection carries resident quantized weights.

    ``router_proj`` is always a raw f32 pointer, so the router cannot read a
    backend off its own weight. Any quantized projection in the same layer is on
    the same backend, and a bf16 artifact has none -- the caller falls back to
    the kernel tree's own backend, which is what ``gguf_embedding`` does.
    """

    for name in (
        "mlp_gate_up_proj",
        "mlp_gate_proj",
        "experts_gate_up_proj",
        "q_proj",
    ):
        projection = getattr(layer, name, None)
        value = getattr(projection, "backend", None)
        if value is not None:
            return str(value)
    return None


def gemma4_layer_forward_bf16(
    hidden_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    keep_mask_ptr: int,
    layer: Gemma4LayerPointers,
    *,
    scratch: Gemma4LayerScratch,
    kv: Gemma4LayerKV | None = None,
    int8_kv: object | None = None,
    rows: int | None = None,
    eps: float = 1e-6,
    rotary_dim: int | None = None,
    key_begin: int = 0,
    attention_mask_is_causal: bool = False,
    window: int = 0,
    prefill_attention_variants: tuple[str, ...] | None = None,
    # The decode-attention variants the execution profile requests, or ``None``
    # for the strict native kernel. Consumed only at the native/exact call site
    # below; a profile that selects the staged candidate keeps every other route
    # -- the tiled and aotriton prefills and the profile-selected prefill
    # variants -- unchanged.
    decode_attention_variants: tuple[str, ...] | None = None,
    # The exact-prefill attention variants the execution profile requests, or
    # ``None`` for the strict native kernel. Also consumed only at the
    # native/exact call site below, and only for a multi-row sliding block the
    # tiled and aotriton routes refused; every other route is unchanged.
    exact_prefill_attention_variants: tuple[str, ...] | None = None,
    # False keeps every dense projection of a multi-row block on the row-exact
    # schedule instead of the WMMA prefill kernels, so each row is bit-identical
    # to a one-row decode step. A one-row block is unaffected either way.
    dense_wmma_prefill: bool = True,
    # Per-row decode ranges for a multi-row block whose attention must repeat
    # the one-token step row by row; ``None`` attends the block in one launch.
    row_decode_attention: Gemma4RowDecodeAttention | None = None,
    # A physically packed multi-request block: the dense and MoE stages run over
    # every row of ``hidden_ptr`` once, while rotation, cache append and attention
    # run once per segment against that segment's own cache, positions, sliding
    # base and mask. Mutually exclusive with ``kv``, ``int8_kv``, the block-wide
    # rope tables, keep mask, ``key_begin``, ``window``, ``row_decode_attention``
    # and the block-wide attention-variant requests: every one of those is a
    # property of a request, and a packed call carries one per segment.
    packed_segments: Sequence[Gemma4PackedSegment] | None = None,
    stream: int = 0,
    # -1 (the default) selects a second stream so the MoE branch overlaps the
    # dense branch. Passing ``stream_moe=stream`` restores the single-stream
    # order, which is both the rollback lever and the A/B control.
    stream_moe: int = -1,
) -> int:
    """Run one Gemma 4 decoder layer over a block of tokens, in place.

    ``hidden_ptr`` is ``(tokens, hidden_size)`` BF16 and is overwritten with the
    layer output. ``cos_ptr``/``sin_ptr`` are the rope tables for this block's
    positions and ``keep_mask_ptr`` is a ``(rows, keys)`` uint8 keep-mask
    covering exactly those positions — including the sliding-window bound on
    sliding layers, which this function does not re-derive.

    ``key_begin`` drops cached keys the caller knows are masked out, by moving
    the key and value pointers forward and shortening ``keys``. The keep-mask
    must already start at ``key_begin``: the kernel indexes it as
    ``keep_mask + token * keys`` with ``keys`` shortened by the same amount, so a
    mask whose column 0 is key ``key_begin`` is addressed correctly for every
    row without the wrapper shifting it. It changes no arithmetic: a masked key
    contributes zero to both reductions, so the remaining terms keep their order
    and the result is bit-identical.

    ``window`` extends that skip to a per-row bound. Row ``t`` of the block sits
    at cache position ``kv.write_offset + t``, so with ``window > 0`` its walk
    starts at column ``max(0, (kv.write_offset - key_begin) + t - window + 1)``.
    ``key_begin`` drops the keys every row of the block masks; this drops the
    rest of each row's own prefix, which only a multi-row block has. Sound under
    the same condition as ``key_begin`` -- the mask must be zero below that
    column, which the sliding causal mask above is and an arbitrary caller-built
    mask need not be -- so it defaults to 0, meaning no promise and a walk from
    column 0. Like ``key_begin`` it changes no arithmetic.

    ``attention_mask_is_causal`` asserts that ``keep_mask_ptr`` holds nothing but
    ``key <= query``, which is what lets a sliding layer take the flash path: a
    sliding-window mask is exactly causal while the window is at least as wide as
    the attended range. It defaults to False because this function cannot verify
    the assertion, and a mask carrying eviction or window bounds that the flash
    kernel does not read would silently change the result.

    ``int8_kv`` selects the INT8 per-token/head storage route instead of the BF16
    cache: the block's BF16 K/V is quantized by the registered BF16-source writer
    and attended by the registered direct INT8 consumer. The consumer takes FP32
    queries and emits FP32 rows, so the query is widened into the owner's FP32
    scratch and the context narrowed back into the layer's BF16 buffer. When
    ``int8_kv`` is set, ``kv`` must be None; the two routes are mutually
    exclusive. ``keep_mask_ptr`` and ``key_begin`` are unused on the INT8 route:
    it derives the mask from the consumer's own sliding-window and position
    semantics, so it can honour exactly the causal/window contract and requires
    ``attention_mask_is_causal=True``. A caller asking for any other mask
    semantics is refused before device work rather than having its extra
    exclusions silently dropped.

    ``packed_segments`` runs several requests through one physical block. The
    norm, projection, MoE and combine stages see ``rows`` contiguous rows and do
    not know the requests exist; the rotation, the KV append and the attention
    launch are repeated per segment with that segment's own pointers and
    positions. A request's rows therefore depend only on that request's cache
    and position, which is the property a multi-request verify needs -- and the
    reason this is not a loop of one-request forwards: the projections and the
    MoE stage are launched once, not once per request. The INT8 route has no
    per-request layout (it derives one block-wide write span), so a packed block
    refuses it by name.

    Returns ``hidden_ptr`` so the call reads as a pipeline stage.
    """

    global _last_prefill_route

    # The block size is an argument, not a property of the scratch. A scratch is
    # sized for the widest block its owner will run, so inferring the row count
    # from the allocation would make a narrow block (a one-token decode step)
    # run as though it were wide, reading past its inputs and writing past its
    # cache slot. ``scratch.tokens`` remains the default for the dense case.
    rows = scratch.tokens if rows is None else int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows > scratch.tokens:
        raise ValueError(f"rows={rows} exceeds scratch capacity {scratch.tokens}")
    if int8_kv is not None:
        if kv is not None:
            raise ValueError("a layer forward takes either a BF16 or an INT8 KV cache, not both")
        if not attention_mask_is_causal:
            # The INT8 consumer derives its own mask from causal order plus the
            # layer's sliding window and never reads keep_mask_ptr. It can
            # therefore honour exactly the causal/window contract and nothing
            # else; an arbitrary exclusion mask would be silently dropped. This
            # is a contract refusal, so it happens before any device work.
            raise ValueError(
                "the INT8 KV route derives its mask from causal order plus the "
                "layer's sliding window and cannot honour an arbitrary exclusion "
                "mask, so it requires attention_mask_is_causal=True"
            )
    if packed_segments is not None:
        # Every one of these arguments is a property of a *request*, and a packed
        # block carries one per segment. Refusing them by name is what stops a
        # caller from believing a block-wide mask or rope table applies to rows
        # that sit at different positions in different caches.
        if int8_kv is not None:
            raise ValueError(
                "packed_segments runs the BF16 KV route; the INT8 per-token/head "
                "owner derives one block-wide write span and has no per-request "
                "cache layout"
            )
        if kv is not None:
            raise ValueError(
                "packed_segments carries each request's own KV cache in its "
                "segment; pass kv=None"
            )
        if int(cos_ptr) or int(sin_ptr) or int(keep_mask_ptr):
            raise ValueError(
                "packed_segments carries each request's own rope tables and keep "
                "mask in its segment; pass 0 for the block-wide cos/sin/mask"
            )
        if int(key_begin) or int(window) or row_decode_attention is not None:
            raise ValueError(
                "packed_segments carries each request's own key_begin, window and "
                "row_decode_attention in its segment; leave the block-wide ones "
                "unset"
            )
        if (
            prefill_attention_variants
            or decode_attention_variants
            or exact_prefill_attention_variants
        ):
            raise ValueError(
                "packed_segments carries each request's own attention-variant "
                "requests in its segment; leave the block-wide ones unset"
            )
        packed_rows = gemma4_packed_segment_rows(packed_segments)
        if rows is not None and int(rows) != packed_rows:
            raise ValueError(
                f"rows={int(rows)} does not match the packed segments' "
                f"{packed_rows} rows"
            )
        rows = packed_rows
        if rows > scratch.tokens:
            raise ValueError(f"rows={rows} exceeds scratch capacity {scratch.tokens}")
    hidden_size = scratch.hidden_size
    geometry = scratch.geometry
    num_heads = geometry.num_heads
    num_kv_heads = geometry.num_kv_heads
    head_dim = geometry.head_dim
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    kwargs = {"stream": stream}
    project_kwargs = {"stream": stream, "use_wmma_prefill": bool(dense_wmma_prefill)}
    # P11: the shared-expert MLP and the MoE experts are independent -- both
    # read `hidden` and write their own buffer, and neither consumes the
    # other's result until the combine -- so they overlap on two streams.
    moe_stream = _moe_stream() if int(stream_moe) < 0 else int(stream_moe)
    moe_kwargs = {"stream": moe_stream}
    parallel = moe_stream != stream

    def buf(name: str) -> int:
        return scratch.buffer(name).ptr

    # --- attention ---------------------------------------------------------
    gemma4_rmsnorm_f32w_bf16(
        hidden_ptr, layer.input_layernorm, buf("normalized"), rows, hidden_size, eps, **kwargs
    )

    if layer.qkv_proj:
        # P6: one launch projects q|k|v into the transient fused buffer, then
        # the split separates them into the buffers every consumer below
        # already reads -- so no consumer's ABI changes. Non-k_eq_v only: the
        # loader never sets this on a k_eq_v layer, where it would not reduce
        # the launch count (2 projections become projection + split = 2).
        gemma4_project(
            buf("normalized"),
            layer.qkv_proj,
            buf("qkv"),
            rows,
            hidden_size,
            q_width + 2 * kv_width,
            **project_kwargs,
        )
        gemma4_qkv_split_bf16(
            buf("qkv"),
            buf("q"),
            buf("k"),
            buf("v"),
            rows,
            q_width,
            kv_width,
            3,
            **kwargs,
        )
    else:
        gemma4_project(buf("normalized"), layer.q_proj, buf("q"), rows, hidden_size, q_width, **project_kwargs)
        gemma4_project(buf("normalized"), layer.k_proj, buf("k"), rows, hidden_size, kv_width, **project_kwargs)

    # `attention_k_eq_v` layers have no v_proj: the reference binds V to the *raw*
    # K projection and normalises a separate `key` array. So K is never normed in
    # place — the k_norm writes to its own buffer and the raw K stays intact for V.
    if geometry.k_eq_v:
        # V = weightless_norm(raw K). Note the norm is weightless here, unlike
        # the k_norm applied below.
        gemma4_rmsnorm_weightless_bf16(
            buf("k"), buf("v"), rows * num_kv_heads, head_dim, eps, **kwargs
        )
    else:
        if not layer.qkv_proj:
            # On the fused path the split already wrote buf("v"); only the
            # unfused path needs its own V projection.
            gemma4_project(
                buf("normalized"), layer.v_proj, buf("v"), rows, hidden_size, kv_width, **project_kwargs
            )
        # V is normalised weightlessly and never rotated.
        gemma4_rmsnorm_weightless_bf16(
            buf("v"), buf("v"), rows * num_kv_heads, head_dim, eps, **kwargs
        )

    # The q/k norms are per-head weighted norms over head_dim, applied to the
    # unrotated projections. The k_norm targets `k_normed` so that raw K survives.
    gemma4_head_rmsnorm_f32w_bf16(
        buf("q"), layer.q_norm, buf("q"), rows * num_heads, head_dim, eps, **kwargs
    )
    gemma4_head_rmsnorm_f32w_bf16(
        buf("k"), layer.k_norm, buf("k_normed"), rows * num_kv_heads, head_dim, eps, **kwargs
    )

    if packed_segments is None:
        gemma4_partial_rotary_bf16(
            buf("q"),
            buf("k_normed"),
            cos_ptr,
            sin_ptr,
            buf("q_rot"),
            buf("k_rot"),
            rows,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim=rotary_dim,
            **kwargs,
        )
    else:
        # Rotation is a function of the row's absolute position, so each
        # request's rows are rotated with that request's own tables. The dense
        # stages above already ran over the whole packed block; only this stage
        # and the attention below need to know where one request's rows end.
        for segment in packed_segments:
            gemma4_partial_rotary_bf16(
                buf("q") + segment.row_start * q_width * _BF16_BYTES,
                buf("k_normed") + segment.row_start * kv_width * _BF16_BYTES,
                segment.cos_ptr,
                segment.sin_ptr,
                buf("q_rot") + segment.row_start * q_width * _BF16_BYTES,
                buf("k_rot") + segment.row_start * kv_width * _BF16_BYTES,
                segment.rows,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim=rotary_dim,
                **kwargs,
            )

    # With a cache, this block's rotated K and normalized V are appended at
    # ``write_offset`` and attention reads the cache instead of the block. Both
    # appends are a contiguous move: the cache is laid out (capacity, kv_heads,
    # head_dim) and the block (rows, kv_heads, head_dim), so position p of the
    # block lands at position write_offset + p of the cache with no stride
    # change. The mask is already (rows, write_offset + rows), so the stale slots
    # past the live context are masked out rather than read.
    #
    # INT8 storage takes the writer/consumer route instead: the block's BF16 K/V
    # is quantized into the owned INT8 cache and the direct INT8 consumer
    # reconstructs it in FP32. The two routes are mutually exclusive; the caller
    # check above refuses a caller that supplies both, and this branch owns the
    # whole attention block so neither route's mask semantics leak into the
    # other.
    if int8_kv is not None:
        _run_int8_attention(
            int8_kv,
            query_bf16=buf("q_rot"),
            key_bf16=buf("k_rot"),
            value_bf16=buf("v"),
            context_bf16=buf("context"),
            rows=rows,
            geometry=geometry,
            stream=stream,
        )
    else:
        def attend(
            rows: int,
            key_begin: int,
            keep_mask_ptr: int,
            window: int,
            write_offset: int,
            query_ptr: int,
            out_ptr: int,
            prefill_attention_variants: tuple[str, ...] | None,
            decode_attention_variants: tuple[str, ...] | None,
            exact_prefill_attention_variants: tuple[str, ...] | None,
            kv: Gemma4LayerKV | None,
        ) -> None:
            """Attend ``rows`` queries whose first row sits at ``write_offset``.

            ``kv`` is the cache those rows were appended to, or ``None`` to
            attend the block's own K/V. It is a parameter rather than the
            enclosing block's cache because a packed block attends each
            request over that request's own cache.
            """

            global _last_prefill_route
            key_count = rows if kv is None else write_offset + rows - key_begin
            if rows > 1 and decode_attention_variants is not None and kv is not None:
                # Only a speculative verify block reaches a layer as several
                # rows with the decode scope attached. The decode selector serves
                # it when its flash variant admits the block; otherwise the
                # block keeps the prefill request below.
                launcher, route = _select_exact_decode_attention(
                    decode_attention_variants=decode_attention_variants,
                    backend=_layer_backend(layer),
                    tokens=rows,
                    keys=key_count,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    sliding_window=geometry.sliding_window,
                    key_begin=key_begin,
                )
                if route != "exact":
                    _last_prefill_route = route
                    launcher(
                        query_ptr,
                        kv.key_cache + key_begin * kv_width * _BF16_BYTES,
                        kv.value_cache + key_begin * kv_width * _BF16_BYTES,
                        keep_mask_ptr,
                        out_ptr,
                        tokens=rows,
                        keys=key_count,
                        scratch=scratch.attention,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        scale=geometry.scale,
                        **kwargs,
                    )
                    return
            route = _select_prefill_route(
                rows=rows,
                keys=key_count,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                sliding_window=geometry.sliding_window,
                mask_is_causal=attention_mask_is_causal,
            )
            _last_prefill_route = route
            if prefill_attention_variants is not None:
                # `rows` goes with it because a one-token block is a decode step, and
                # the variants are multi-row prefill kernels. Before this was passed,
                # decode ran the prefill tiling for all 30 layers and never reached
                # the decode kernel the strict path routes to: 30 launches at
                # tokens=1 per step, 0 to `_launch_prefill`, and 10 percent off
                # decode throughput.
                #
                # `keys` goes with `rows` because which of the two a one-token block
                # takes is a measured crossover, not a fixed rule: the strict decode
                # kernel is faster below `_STRICT_DECODE_WINS_UNTIL_KEYS` and the
                # variants' decode shape is faster above it. A block past that
                # crossover takes the variants' online-softmax walk, which is also
                # the only path that runs once the strict kernel's own shared-memory
                # bound is passed -- and what makes a context longer than that bound
                # reachable at all.
                attention = select_prefill_attention(
                    requested_variant=prefill_attention_variants,
                    num_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, tokens=rows, keys=key_count,
                )
                _last_prefill_route = attention.variant
                attention.launcher(
                    query_ptr,
                    (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None else buf("k_rot"),
                    (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None else buf("v"),
                    keep_mask_ptr,
                    out_ptr, tokens=rows, keys=key_count,
                    window=window,
                    row_offset=(write_offset if kv is not None else 0) - key_begin,
                    scratch=scratch.attention, num_heads=num_heads,
                    num_kv_heads=num_kv_heads, head_dim=head_dim,
                    scale=geometry.scale, **kwargs,
                )
            elif route == "tiled":
                # head_dim 512: Gemma 4's five global layers. The tiled kernel stages
                # its own dtype buffers inside the HIP source, so unlike the other two
                # routes it takes no scratch arena.
                gemma4_attention_prefill_tiled(
                    query_ptr,
                    (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None
                    else buf("k_rot"),
                    (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None
                    else buf("v"),
                    keep_mask_ptr,
                    out_ptr,
                    tokens=rows,
                    keys=key_count,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    scale=geometry.scale,
                    **kwargs,
                )
            elif route == "aotriton":
                gemma4_attention_prefill_aotriton(
                    query_ptr,
                    (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None
                    else buf("k_rot"),
                    (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None
                    else buf("v"),
                    out_ptr,
                    tokens=rows,
                    keys=key_count,
                    scratch=scratch.attention,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    scale=geometry.scale,
                    **kwargs,
                )
            else:
                # The native/exact route. Its BF16 entry serves the multi-row prefill
                # fallback and the one-token global decode step; a production profile
                # may route the latter to the staged singleton candidate, which
                # reproduces this entry's ``global_class`` arithmetic in bounded
                # shared memory. The reporter shows the inner variant that ran.
                launcher, _last_prefill_route = _select_exact_decode_attention(
                    decode_attention_variants=decode_attention_variants,
                    backend=_layer_backend(layer),
                    tokens=rows,
                    keys=key_count,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    # The scope the candidate was verified on: a windowless global
                    # layer and a caller that dropped no prefix. A sliding layer's
                    # decode is windowed, and a trimmed global call skips keys the
                    # candidate was not checked against; both keep the native
                    # kernel.
                    sliding_window=geometry.sliding_window,
                    key_begin=key_begin,
                )
                # The staged multi-row sliding prefill is a second, disjoint scope
                # on the same call site. It is consulted only after the decode
                # selector kept the native route, and it overrides that launcher
                # only when the scope actually applies to this call (a multi-row
                # sliding block): an out-of-scope call -- a global block or a
                # one-row block -- keeps the decode selector's resolved callable,
                # so a replaced decode fallback still runs.
                staged_prefill = False
                if _last_prefill_route == "exact" and exact_prefill_attention_variants:
                    prefill_launcher, prefill_route = _select_exact_prefill_attention(
                        exact_prefill_attention_variants=exact_prefill_attention_variants,
                        backend=_layer_backend(layer),
                        tokens=rows,
                        keys=key_count,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        sliding_window=geometry.sliding_window,
                    )
                    if prefill_route is not None:
                        launcher = prefill_launcher
                        _last_prefill_route = prefill_route
                        staged_prefill = prefill_route == EXACT_PREFILL_ATTENTION_STAGED
                # ``window``/``row_offset`` are forwarded only to the staged
                # multi-row prefill, the one launcher whose per-row trim the layer's
                # window licenses. The strict native entry and the registered decode
                # fallback keep the exact keyword set they have always been called
                # with, so their call is byte-for-byte unchanged.
                if staged_prefill:
                    trim_kwargs = {
                        "window": window,
                        "row_offset": (write_offset if kv is not None else 0) - key_begin,
                    }
                else:
                    trim_kwargs = {}
                launcher(
                    query_ptr,
                    (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None
                    else buf("k_rot"),
                    (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                    if kv is not None
                    else buf("v"),
                    keep_mask_ptr,
                    out_ptr,
                    tokens=rows,
                    keys=key_count,
                    **trim_kwargs,
                    scratch=scratch.attention,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    scale=geometry.scale,
                    **kwargs,
                )

        def attend_one(
            *,
            row_start: int,
            rows: int,
            kv: Gemma4LayerKV | None,
            key_begin: int,
            keep_mask_ptr: int,
            window: int,
            prefill_attention_variants: tuple[str, ...] | None,
            decode_attention_variants: tuple[str, ...] | None,
            exact_prefill_attention_variants: tuple[str, ...] | None,
            row_decode_attention: Gemma4RowDecodeAttention | None,
        ) -> None:
            """Attend one request's rows, which start at packed row ``row_start``.

            This is the whole attention step for one request: one launch when
            the request's rows may be attended together, and one decode-shaped
            launch per row when they must each repeat the one-token step. The
            one-request path calls it with ``row_start=0`` and the block's own
            cache; a packed block calls it once per segment with that segment's
            cache, mask and positions, so both paths share one definition of
            what a request's attention is.
            """

            q_base = buf("q_rot") + row_start * q_width * _BF16_BYTES
            out_base = buf("context") + row_start * q_width * _BF16_BYTES
            if row_decode_attention is None or rows == 1:
                attend(
                    rows, key_begin, keep_mask_ptr, window,
                    kv.write_offset if kv is not None else 0,
                    q_base, out_base,
                    prefill_attention_variants, decode_attention_variants,
                    exact_prefill_attention_variants, kv,
                )
                return
            # One decode-shaped launch per row. A multi-row launch reduces each
            # row over the block's key range, and the decode kernel partitions
            # its reduction by the key count, so only a launch that carries the
            # row's own range repeats the one-token step's arithmetic.
            if kv is None:
                raise ValueError("row_decode_attention requires a KV cache")
            if len(row_decode_attention.key_begins) != rows:
                raise ValueError("row_decode_attention must describe every row")
            block_keys = kv.write_offset + rows - key_begin
            row_bytes = q_width * _BF16_BYTES
            for row, (row_begin, row_window) in enumerate(
                zip(row_decode_attention.key_begins, row_decode_attention.windows)
            ):
                if row_begin < key_begin:
                    raise ValueError("a row cannot read keys before its block's first key")
                attend(
                    1, int(row_begin),
                    keep_mask_ptr + row * block_keys + (int(row_begin) - key_begin),
                    int(row_window), kv.write_offset + row,
                    q_base + row * row_bytes, out_base + row * row_bytes,
                    row_decode_attention.prefill_attention_variants,
                    row_decode_attention.decode_attention_variants,
                    None,
                    kv,
                )

        if packed_segments is None:
            if kv is not None:
                _append_kv(
                    kv.key_cache,
                    buf("k_rot"),
                    kv.write_offset * kv_width,
                    rows * kv_width,
                    stream,
                )
                _append_kv(
                    kv.value_cache,
                    buf("v"),
                    kv.write_offset * kv_width,
                    rows * kv_width,
                    stream,
                )

            key_begin = int(key_begin)
            if key_begin < 0:
                raise ValueError(f"key_begin must be non-negative, got {key_begin}")
            if key_begin and kv is None:
                raise ValueError("key_begin requires a cache to skip into")
            attend_one(
                row_start=0,
                rows=rows,
                kv=kv,
                key_begin=key_begin,
                keep_mask_ptr=keep_mask_ptr,
                window=window,
                prefill_attention_variants=prefill_attention_variants,
                decode_attention_variants=decode_attention_variants,
                exact_prefill_attention_variants=exact_prefill_attention_variants,
                row_decode_attention=row_decode_attention,
            )
        else:
            # One append and one attention step per request, over the packed
            # block's rows for that request only. The append is the same
            # contiguous move as above with the segment's own source offset, and
            # the attention reads that segment's cache, mask and sliding base,
            # so no request can see another's keys.
            for segment in packed_segments:
                _append_kv(
                    segment.kv.key_cache,
                    buf("k_rot") + segment.row_start * kv_width * _BF16_BYTES,
                    segment.kv.write_offset * kv_width,
                    segment.rows * kv_width,
                    stream,
                )
                _append_kv(
                    segment.kv.value_cache,
                    buf("v") + segment.row_start * kv_width * _BF16_BYTES,
                    segment.kv.write_offset * kv_width,
                    segment.rows * kv_width,
                    stream,
                )
                attend_one(
                    row_start=segment.row_start,
                    rows=segment.rows,
                    kv=segment.kv,
                    key_begin=int(segment.key_begin),
                    keep_mask_ptr=segment.keep_mask_ptr,
                    window=int(segment.window),
                    prefill_attention_variants=segment.prefill_attention_variants,
                    decode_attention_variants=segment.decode_attention_variants,
                    exact_prefill_attention_variants=(
                        segment.exact_prefill_attention_variants
                    ),
                    row_decode_attention=segment.row_decode_attention,
                )
    gemma4_project(
        buf("context"), layer.o_proj, buf("attn_out"), rows, q_width, hidden_size, **project_kwargs
    )

    # residual + norm(attended), with a null layer scalar meaning 1.0.
    gemma4_add_rmsnorm_scale_bf16(
        buf("attn_out"),
        hidden_ptr,
        layer.post_attention_layernorm,
        0,
        buf("hidden"),
        rows,
        hidden_size,
        eps,
        **kwargs,
    )

    # The MoE branch may run concurrently with the dense branch below, so it
    # needs an entry barrier: record on the main stream *before* the dense
    # branch is dispatched (recording after would make the branch wait for the
    # dense work too, deleting the overlap this exists to create).
    if parallel:
        from hipengine.core.hip import get_hip_runtime

        runtime = get_hip_runtime()
        if scratch._moe_entry_event is None:
            scratch._moe_entry_event = runtime.event_create()
            scratch._moe_exit_event = runtime.event_create()
        runtime.event_record(scratch._moe_entry_event, stream)
        runtime.stream_wait_event(moe_stream, scratch._moe_entry_event)

    # --- dense branch (parallel with the MoE branch) ------------------------
    gemma4_rmsnorm_f32w_bf16(
        buf("hidden"),
        layer.pre_feedforward_layernorm,
        buf("normalized"),
        rows,
        hidden_size,
        eps,
        **kwargs,
    )
    gemma4_project(
        buf("normalized"),
        layer.mlp_gate_up_proj,
        buf("dense_gate_up"),
        rows,
        hidden_size,
        2 * scratch.dense_intermediate,
        **project_kwargs,
    )
    # Both halves live in one resident weight, so the fused GeGLU reads them
    # from a single buffer -- the form the expert path already uses.
    gemma4_gelu_tanh_mul_bf16(
        buf("dense_gate_up"),
        buf("dense_act"),
        rows,
        scratch.dense_intermediate,
        **kwargs,
    )
    gemma4_project(
        buf("dense_act"),
        layer.mlp_down_proj,
        buf("dense"),
        rows,
        scratch.dense_intermediate,
        hidden_size,
        **project_kwargs,
    )
    # D6: post_feedforward_layernorm_1 is NOT applied here anymore. The fold
    # below consumes the RAW dense output, so this stage moved into the combine
    # kernel -- one launch instead of three, bit-identical to the chain.

    # --- MoE branch ---------------------------------------------------------
    # A row-exact block routes each row through the one-token launch: the
    # fused decode router and the multi-row chain reduce the projection in a
    # different order, so their weights can differ in the last place. A packed
    # block is row-exact as soon as any request in it asked for a row-exact
    # step, because the router launch is block-wide and cannot be mixed. The
    # request's own flag is what carries that: a one-row segment needs no
    # per-row decode ranges, so the ranges alone would let four one-row requests
    # route as one four-row block.
    router_row_exact = row_decode_attention is not None or (
        packed_segments is not None
        and any(segment.row_exact for segment in packed_segments)
    )
    router_rows = 1 if router_row_exact else rows
    for router_row in range(0, rows, router_rows):
        gemma4_router_topk_bf16(
            buf("hidden") + router_row * hidden_size * _BF16_BYTES,
            layer.router_scale,
            layer.router_proj,
            layer.router_per_expert_scale,
            buf("selected") + router_row * scratch.top_k * _I64_BYTES,
            buf("routing") + router_row * scratch.top_k * _F32_BYTES,
            tokens=router_rows,
            hidden_size=hidden_size,
            num_experts=scratch.num_experts,
            top_k=scratch.top_k,
            scratch=scratch.router,
            eps=eps,
            stream=moe_stream,
            backend=_layer_backend(layer),
        )
    gemma4_rmsnorm_f32w_bf16(
        buf("hidden"),
        layer.pre_feedforward_layernorm_2,
        buf("moe_normalized"),
        rows,
        hidden_size,
        eps,
        **moe_kwargs,
    )
    gemma4_experts_forward_bf16(
        buf("moe_normalized"),
        buf("selected"),
        buf("routing"),
        layer.experts_gate_up_proj,
        layer.experts_down_proj,
        buf("experts"),
        scratch=scratch.experts,
        rows=rows,
        row_exact=bool(packed_segments) and all(segment.row_exact for segment in packed_segments),
        stream=moe_stream,
    )
    gemma4_rmsnorm_f32w_bf16(
        buf("experts"),
        layer.post_feedforward_layernorm_2,
        buf("experts"),
        rows,
        hidden_size,
        eps,
        **moe_kwargs,
    )

    # Exit barrier: the combine below reads `experts` from the main stream.
    if parallel:
        runtime.event_record(scratch._moe_exit_event, moe_stream)
        runtime.stream_wait_event(stream, scratch._moe_exit_event)

    # --- combine ------------------------------------------------------------
    # D6 tail fold: post_ffw_norm_1 + branch_add + add_rmsnorm_scale in one
    # launch (bit-identical to the chain it replaced; see the kernel
    # docstring). It must sit after the exit barrier because `experts` is an
    # input, and it reads the raw dense output -- which is why the first stage
    # no longer runs right after down_proj.
    gemma4_dense_combine_rmsnorm_scale_bf16(
        buf("dense"),
        buf("experts"),
        buf("hidden"),
        layer.post_feedforward_layernorm_1,
        layer.post_feedforward_layernorm,
        layer.layer_scalar,
        hidden_ptr,
        rows,
        hidden_size,
        eps,
        **kwargs,
    )
    return hidden_ptr
