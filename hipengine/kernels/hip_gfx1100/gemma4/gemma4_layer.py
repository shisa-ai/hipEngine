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
    """Device pointers for one layer's weights.

    ``v_proj`` is ``0`` on ``attention_k_eq_v`` layers, where the reference takes
    the raw K projection as V and there is no ``v_proj`` tensor in the artifact.
    ``layer_scalar`` is ``0`` to mean 1.0, which is what the kernel does with a
    null pointer.
    """

    input_layernorm: int
    q_proj: int
    k_proj: int
    o_proj: int
    q_norm: int
    k_norm: int
    post_attention_layernorm: int
    pre_feedforward_layernorm: int
    mlp_gate_proj: int
    mlp_up_proj: int
    mlp_down_proj: int
    post_feedforward_layernorm_1: int
    router_scale: int
    router_proj: int
    router_per_expert_scale: int
    pre_feedforward_layernorm_2: int
    experts_gate_up_proj: int
    experts_down_proj: int
    post_feedforward_layernorm_2: int
    post_feedforward_layernorm: int
    v_proj: int = 0
    layer_scalar: int = 0


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


def gemma4_layer_forward_bf16(
    hidden_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    keep_mask_ptr: int,
    layer: Gemma4LayerPointers,
    *,
    scratch: Gemma4LayerScratch,
    eps: float = 1e-6,
    rotary_dim: int | None = None,
    stream: int = 0,
) -> int:
    """Run one Gemma 4 decoder layer over a block of tokens, in place.

    ``hidden_ptr`` is ``(tokens, hidden_size)`` BF16 and is overwritten with the
    layer output. ``cos_ptr``/``sin_ptr`` are the rope tables for this block's
    positions and ``keep_mask_ptr`` is a ``(tokens, tokens)`` uint8 keep-mask
    covering exactly those positions — including the sliding-window bound on
    sliding layers, which this function does not re-derive.

    Returns ``hidden_ptr`` so the call reads as a pipeline stage.
    """

    rows = scratch.tokens
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

    dense_gemv_out_bf16(
        buf("normalized"), layer.q_proj, buf("q"), rows, hidden_size, q_width, **kwargs
    )
    dense_gemv_out_bf16(
        buf("normalized"), layer.k_proj, buf("k"), rows, hidden_size, kv_width, **kwargs
    )

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
        dense_gemv_out_bf16(
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

    gemma4_attention_prefill_bf16(
        buf("q_rot"),
        buf("k_rot"),
        buf("v"),
        keep_mask_ptr,
        buf("context"),
        tokens=rows,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=geometry.scale,
        **kwargs,
    )
    dense_gemv_out_bf16(
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
    dense_gemv_out_bf16(
        buf("normalized"),
        layer.mlp_gate_proj,
        buf("dense_gate"),
        rows,
        hidden_size,
        scratch.dense_intermediate,
        **kwargs,
    )
    dense_gemv_out_bf16(
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
    dense_gemv_out_bf16(
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
