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

from hipengine.core.memory import DeviceBuffer, free as hip_free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    Gemma4AttentionScratch,
    gemma4_attention_prefill_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    Gemma4ExpertScratch,
    gemma4_experts_forward_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
    gemma4_gelu_tanh_mul_split_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_add_rmsnorm_scale_bf16,
    gemma4_branch_add_bf16,
    gemma4_head_rmsnorm_f32w_bf16,
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
    mlp_gate_proj: Gemma4Projection
    mlp_up_proj: Gemma4Projection
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
        self.attention.close()
        if self.experts is not None:
            self.experts.free()
        if self.router is not None:
            self.router.free()
        for buf in self._buffers:
            hip_free(buf)
        self._buffers.clear()
        self._by_name.clear()

    def _size_of(self, name: str) -> int:
        rows = self.tokens
        hidden = self.hidden_size
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        sizes = {
            "normalized": rows * hidden * _BF16_BYTES,
            "q": rows * q_width * _BF16_BYTES,
            "k": rows * kv_width * _BF16_BYTES,
            "k_normed": rows * kv_width * _BF16_BYTES,
            "v": rows * kv_width * _BF16_BYTES,
            "q_rot": rows * q_width * _BF16_BYTES,
            "k_rot": rows * kv_width * _BF16_BYTES,
            "context": rows * q_width * _BF16_BYTES,
            "attn_out": rows * hidden * _BF16_BYTES,
            "hidden": rows * hidden * _BF16_BYTES,
            "dense_gate": rows * self.dense_intermediate * _BF16_BYTES,
            "dense_up": rows * self.dense_intermediate * _BF16_BYTES,
            "dense_act": rows * self.dense_intermediate * _BF16_BYTES,
            "dense": rows * hidden * _BF16_BYTES,
            "experts": rows * hidden * _BF16_BYTES,
            "selected": rows * self.top_k * _I64_BYTES,
            "routing": rows * self.top_k * _F32_BYTES,
            "branch_sum": rows * hidden * _BF16_BYTES,
        }
        try:
            return sizes[name]
        except KeyError:
            raise ValueError(f"unknown scratch buffer {name!r}") from None


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


def gemma4_layer_forward_bf16(
    hidden_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    keep_mask_ptr: int,
    layer: Gemma4LayerPointers,
    *,
    scratch: Gemma4LayerScratch,
    kv: Gemma4LayerKV | None = None,
    rows: int | None = None,
    eps: float = 1e-6,
    rotary_dim: int | None = None,
    key_begin: int = 0,
    stream: int = 0,
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

    Returns ``hidden_ptr`` so the call reads as a pipeline stage.
    """

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
    hidden_size = scratch.hidden_size
    geometry = scratch.geometry
    num_heads = geometry.num_heads
    num_kv_heads = geometry.num_kv_heads
    head_dim = geometry.head_dim
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    kwargs = {"stream": stream}

    def buf(name: str) -> int:
        return scratch.buffer(name).ptr

    # --- attention ---------------------------------------------------------
    gemma4_rmsnorm_f32w_bf16(
        hidden_ptr, layer.input_layernorm, buf("normalized"), rows, hidden_size, eps, **kwargs
    )

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
        keys=None if kv is None else kv.write_offset + rows - key_begin,
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
        layer.mlp_gate_proj,
        buf("dense_gate"),
        rows,
        hidden_size,
        scratch.dense_intermediate,
        **kwargs,
    )
    gemma4_project(
        buf("normalized"),
        layer.mlp_up_proj,
        buf("dense_up"),
        rows,
        hidden_size,
        scratch.dense_intermediate,
        **kwargs,
    )
    # The dense MLP's gate and up are separate tensors in the artifact, so this
    # uses the split GeGLU rather than the fused expert-path kernel.
    gemma4_gelu_tanh_mul_split_bf16(
        buf("dense_gate"),
        buf("dense_up"),
        buf("dense_act"),
        rows * scratch.dense_intermediate,
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
    gemma4_rmsnorm_f32w_bf16(
        buf("dense"),
        layer.post_feedforward_layernorm_1,
        buf("dense"),
        rows,
        hidden_size,
        eps,
        **kwargs,
    )

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
        stream=stream,
    )
    gemma4_rmsnorm_f32w_bf16(
        buf("hidden"),
        layer.pre_feedforward_layernorm_2,
        buf("normalized"),
        rows,
        hidden_size,
        eps,
        **kwargs,
    )
    gemma4_experts_forward_bf16(
        buf("normalized"),
        buf("selected"),
        buf("routing"),
        layer.experts_gate_up_proj,
        layer.experts_down_proj,
        buf("experts"),
        scratch=scratch.experts,
        rows=rows,
        stream=stream,
    )
    gemma4_rmsnorm_f32w_bf16(
        buf("experts"),
        layer.post_feedforward_layernorm_2,
        buf("experts"),
        rows,
        hidden_size,
        eps,
        **kwargs,
    )

    # --- combine ------------------------------------------------------------
    gemma4_branch_add_bf16(
        buf("dense"), buf("experts"), buf("branch_sum"), rows * hidden_size, **kwargs
    )
    gemma4_add_rmsnorm_scale_bf16(
        buf("branch_sum"),
        buf("hidden"),
        layer.post_feedforward_layernorm,
        layer.layer_scalar,
        hidden_ptr,
        rows,
        hidden_size,
        eps,
        **kwargs,
    )
    return hidden_ptr
