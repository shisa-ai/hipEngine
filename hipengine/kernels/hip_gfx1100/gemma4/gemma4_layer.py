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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, free as hip_free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    Gemma4AttentionScratch,
    aotriton_prefill_admits,
    gemma4_attention_prefill_aotriton,
    gemma4_attention_prefill_bf16,
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
) -> None:
    """Run one ``(rows, in_features) x (in_features, out_features)`` projection.

    ``int`` is a bf16 device pointer; anything else is a GGUF device weight and
    goes through the quantized linear dispatch, which picks its own kernel from
    the weight's quant key and the row count.
    """

    if isinstance(weight, int):
        dense_gemv_out_bf16(x_ptr, weight, out_ptr, rows, in_features, out_features, stream=stream)
        return
    # Imported here rather than at module scope: the quantized dispatch lives in
    # the runtime layer and the kernel package does not depend on it otherwise.
    from hipengine.runtime.gguf_linear import launch_gguf_linear

    # ``use_wmma_prefill=True`` is what both shipping GGUF call sites pass
    # literally (generation/qwen35_gguf.py, runtime/qwen35_gguf_nextn.py), and
    # ENVS.md records it as the public generator's behaviour: the env var is only
    # the low-level session default. Gemma's dense projections were taking the
    # fallback schedule instead -- 888.9 us against 281.5 us on the (512, 2816,
    # 2112) q/k/v shape, a 3.16x difference measured with the unaffected MoE
    # gate_up flat across the same runs. The dispatch only rewrites shapes that
    # have a registered WMMA prefill kernel (currently gguf_q8_0 and raw
    # gguf_q4_k), so every other quant and row count keeps its existing schedule.
    launch_gguf_linear(
        weight,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        use_wmma_prefill=True,
    )


@dataclass
class Gemma4LayerGeometry:
    """The attention geometry fields the layer forward needs."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float = 1.0
    k_eq_v: bool = False
    sliding_window: int | None = None


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
        total = sum(self._sizes().values())
        if self.experts is not None:
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
    the key, value and mask pointers forward together and shortening ``keys``.
    It is only valid for a one-row block, where the mask has a single row to
    offset; the caller owns that restriction (see ``_sliding_read_range``). It
    changes no arithmetic: a masked key contributes zero to both reductions, so
    the remaining terms keep their order and the result is bit-identical.

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
    hidden_size = scratch.hidden_size
    geometry = scratch.geometry
    num_heads = geometry.num_heads
    num_kv_heads = geometry.num_kv_heads
    head_dim = geometry.head_dim
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    kwargs = {"stream": stream}
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
            **kwargs,
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
        gemma4_project(buf("normalized"), layer.q_proj, buf("q"), rows, hidden_size, q_width, **kwargs)
        gemma4_project(buf("normalized"), layer.k_proj, buf("k"), rows, hidden_size, kv_width, **kwargs)

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
                buf("normalized"), layer.v_proj, buf("v"), rows, hidden_size, kv_width, **kwargs
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

        key_count = rows if kv is None else kv.write_offset + rows - key_begin
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
        if route == "tiled":
            # head_dim 512: Gemma 4's five global layers. The tiled kernel stages
            # its own dtype buffers inside the HIP source, so unlike the other two
            # routes it takes no scratch arena.
            gemma4_attention_prefill_tiled(
                buf("q_rot"),
                (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                if kv is not None
                else buf("k_rot"),
                (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                if kv is not None
                else buf("v"),
                keep_mask_ptr + key_begin,
                buf("context"),
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
                buf("q_rot"),
                (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                if kv is not None
                else buf("k_rot"),
                (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                if kv is not None
                else buf("v"),
                buf("context"),
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
            gemma4_attention_prefill_bf16(
                buf("q_rot"),
                (kv.key_cache + key_begin * kv_width * _BF16_BYTES)
                if kv is not None
                else buf("k_rot"),
                (kv.value_cache + key_begin * kv_width * _BF16_BYTES)
                if kv is not None
                else buf("v"),
                keep_mask_ptr + key_begin,
                buf("context"),
                tokens=rows,
                keys=key_count,
                scratch=scratch.attention,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scale=geometry.scale,
                **kwargs,
            )
    gemma4_project(
        buf("context"), layer.o_proj, buf("attn_out"), rows, q_width, hidden_size, **kwargs
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
        **kwargs,
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
        **kwargs,
    )
    # D6: post_feedforward_layernorm_1 is NOT applied here anymore. The fold
    # below consumes the RAW dense output, so this stage moved into the combine
    # kernel -- one launch instead of three, bit-identical to the chain.

    # --- MoE branch ---------------------------------------------------------
    gemma4_router_topk_bf16(
        buf("hidden"),
        layer.router_scale,
        layer.router_proj,
        layer.router_per_expert_scale,
        buf("selected"),
        buf("routing"),
        tokens=rows,
        hidden_size=hidden_size,
        num_experts=scratch.num_experts,
        top_k=scratch.top_k,
        scratch=scratch.router,
        eps=eps,
        stream=moe_stream,
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
