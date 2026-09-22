"""Torch-free NumPy reference forward for the Gemma 4 text decoder.

Gemma 4 is not a configuration variant of an existing hipEngine model. Its text
tower differs from every ported architecture in ways that change the arithmetic,
so this module is the independent oracle the GPU path is gated against:

* **Dual attention geometry.** Sliding layers use ``head_dim`` with
  ``num_key_value_heads``; global layers use ``global_head_dim`` with
  ``num_global_key_value_heads``. Both appear in one model, so the KV cache and
  the attention kernel take per-layer geometry.
* **Proportional partial RoPE.** Global layers rotate only the first
  ``partial_rotary_factor * head_dim / 2`` pairs, and those pairs use the
  exponent scale of the *full* head dimension. The remaining pairs rotate by
  angle zero.
* **K = V on global layers.** ``attention_k_eq_v`` removes ``v_proj``; V is the
  raw K projection passed through a *weightless* RMS norm and no rotation.
* **Parallel dense MLP and routed experts.** The dense MLP and the MoE branch
  both run, each with its own post-norm, and are summed before the block
  post-norm. This is not a shared-expert topology: the routed branch reads the
  pre-MLP residual, not the dense MLP output.
* **GELU, not SiLU.** ``gelu_pytorch_tanh`` is used in the dense MLP and in the
  routed experts.
* **Weightless router norm plus two scales.** The router normalizes without a
  weight, multiplies by a learned ``scale`` and ``hidden_size**-0.5``, then
  rescales the renormalized top-k weights by ``per_expert_scale``.
* **Per-layer ``layer_scalar`` and ``sqrt(hidden_size)`` embedding scale.**

Everything here is plain NumPy float32 and deliberately unoptimized. It is an
oracle, not a runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.kernels.cpu_reference.ops import linear, rmsnorm, rotate
from hipengine.kernels.registry import KernelKey, register

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

ArrayLike = Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemma4RopeConfig:
    """Per-layer-type rotary contract.

    ``rope_angles`` counts the rotated *pairs*. Global layers use proportional
    RoPE, where the rotated pairs keep the exponent scale of ``head_dim`` rather
    than of the rotated width, and every remaining pair rotates by angle zero.
    """

    rope_theta: float
    head_dim: int
    rope_angles: int
    rope_type: str = "default"

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if not 0 <= self.rope_angles <= self.head_dim // 2:
            raise ValueError("rope_angles must be within [0, head_dim // 2]")

    @property
    def inverse_frequencies(self) -> np.ndarray:
        """Return ``head_dim // 2`` inverse frequencies, zero past the rotated span."""

        half = self.head_dim // 2
        index = np.arange(self.rope_angles, dtype=np.float64)
        rotated = 1.0 / np.power(
            float(self.rope_theta),
            (2.0 * index) / float(self.head_dim),
            dtype=np.float64,
        )
        if self.rope_angles == half:
            return rotated.astype(np.float32)
        return np.concatenate(
            (rotated, np.zeros(half - self.rope_angles, dtype=np.float64))
        ).astype(np.float32)


@dataclass(frozen=True)
class Gemma4AttentionGeometry:
    """Resolved attention geometry for one decoder layer."""

    layer_type: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope: Gemma4RopeConfig
    sliding_window: int | None = None
    k_eq_v: bool = False

    def __post_init__(self) -> None:
        if self.layer_type not in (SLIDING_ATTENTION, FULL_ATTENTION):
            raise ValueError(f"unsupported layer type {self.layer_type!r}")
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.rope.head_dim != self.head_dim:
            raise ValueError("rope head_dim must match attention head_dim")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when set")

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def scale(self) -> float:
        """Gemma 4 attention scaling. Always 1.0, never ``head_dim**-0.5``."""

        return 1.0


@dataclass(frozen=True)
class Gemma4TextConfig:
    """Normalized Gemma 4 text configuration."""

    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    top_k_experts: int
    rms_norm_eps: float
    attention: tuple[Gemma4AttentionGeometry, ...]
    final_logit_softcapping: float | None = None
    tie_word_embeddings: bool = True
    vocab_size: int | None = None
    hidden_activation: str = "gelu_pytorch_tanh"

    def __post_init__(self) -> None:
        if not self.attention:
            raise ValueError("attention geometry must cover at least one layer")
        if self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma 4 text reference implements gelu_pytorch_tanh only, got "
                f"{self.hidden_activation!r}"
            )
        if self.num_experts <= 0 or self.top_k_experts <= 0:
            raise ValueError("expert counts must be positive")
        if self.top_k_experts > self.num_experts:
            raise ValueError("top_k_experts must not exceed num_experts")

    @property
    def num_hidden_layers(self) -> int:
        return len(self.attention)

    @property
    def embed_scale(self) -> float:
        return math.sqrt(float(self.hidden_size))

    @property
    def router_scalar_root_size(self) -> float:
        return float(self.hidden_size) ** -0.5

    def geometry(self, layer: int) -> Gemma4AttentionGeometry:
        return self.attention[layer]


def gemma4_text_config_from_hf(config: Mapping[str, Any]) -> Gemma4TextConfig:
    """Normalize an HF ``gemma4_text`` config mapping.

    ``config`` may be either the text config itself or a multimodal wrapper that
    carries it under ``text_config``.
    """

    source = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config

    hidden_size = int(source["hidden_size"])
    num_heads = int(source["num_attention_heads"])
    num_kv_heads = int(source["num_key_value_heads"])
    head_dim = int(source.get("head_dim") or hidden_size // num_heads)
    global_head_dim = int(source.get("global_head_dim") or head_dim)
    num_global_kv_heads = int(source.get("num_global_key_value_heads") or num_kv_heads)
    sliding_window = source.get("sliding_window")
    sliding_window = None if sliding_window is None else int(sliding_window)
    k_eq_v = bool(source.get("attention_k_eq_v", False))

    layer_types = source.get("layer_types")
    if layer_types is None:
        pattern = int(source.get("sliding_window_pattern", 6))
        layer_types = [
            SLIDING_ATTENTION if (index + 1) % pattern else FULL_ATTENTION
            for index in range(int(source["num_hidden_layers"]))
        ]
    layer_types = [str(value) for value in layer_types]

    rope_parameters = source.get("rope_parameters") or {}
    global_rope = rope_parameters.get(FULL_ATTENTION, {})
    sliding_rope = rope_parameters.get(SLIDING_ATTENTION, {})

    global_partial = float(global_rope.get("partial_rotary_factor", 1.0))
    global_rope_angles = int(global_partial * global_head_dim // 2)
    sliding_rope_angles = head_dim // 2

    global_geometry = Gemma4AttentionGeometry(
        layer_type=FULL_ATTENTION,
        num_heads=num_heads,
        num_kv_heads=num_global_kv_heads,
        head_dim=global_head_dim,
        rope=Gemma4RopeConfig(
            rope_theta=float(global_rope.get("rope_theta", 1_000_000.0)),
            head_dim=global_head_dim,
            rope_angles=global_rope_angles,
            rope_type=str(global_rope.get("rope_type", "proportional")),
        ),
        sliding_window=None,
        k_eq_v=k_eq_v,
    )
    sliding_geometry = Gemma4AttentionGeometry(
        layer_type=SLIDING_ATTENTION,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope=Gemma4RopeConfig(
            rope_theta=float(sliding_rope.get("rope_theta", 10_000.0)),
            head_dim=head_dim,
            rope_angles=sliding_rope_angles,
            rope_type=str(sliding_rope.get("rope_type", "default")),
        ),
        sliding_window=sliding_window,
        k_eq_v=False,
    )

    return Gemma4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=int(source["intermediate_size"]),
        moe_intermediate_size=int(
            source.get("moe_intermediate_size") or source["intermediate_size"]
        ),
        num_experts=int(source.get("num_experts") or 1),
        top_k_experts=int(source.get("top_k_experts") or 1),
        rms_norm_eps=float(source.get("rms_norm_eps", 1e-6)),
        attention=tuple(
            global_geometry if layer_type == FULL_ATTENTION else sliding_geometry
            for layer_type in layer_types
        ),
        final_logit_softcapping=(
            None
            if source.get("final_logit_softcapping") is None
            else float(source["final_logit_softcapping"])
        ),
        tie_word_embeddings=bool(source.get("tie_word_embeddings", True)),
        vocab_size=(
            None if source.get("vocab_size") is None else int(source["vocab_size"])
        ),
        hidden_activation=str(source.get("hidden_activation", "gelu_pytorch_tanh")),
    )


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def gemma4_gelu_tanh(x: ArrayLike) -> np.ndarray:
    """``gelu_pytorch_tanh``: ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))``."""

    value = np.asarray(x, dtype=np.float32)
    inner = np.float32(math.sqrt(2.0 / math.pi)) * (
        value + np.float32(0.044715) * value * value * value
    )
    return (np.float32(0.5) * value * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def gemma4_rmsnorm(
    x: ArrayLike,
    weight: ArrayLike | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Gemma 4 RMS norm, optionally weightless.

    ``Gemma4RMSNorm`` computes the norm in float32 and multiplies by a float32
    weight, then casts back. ``with_scale=False`` omits the weight entirely,
    which is how ``v_norm`` and the router norm are built.
    """

    value = np.asarray(x, dtype=np.float32)
    variance = np.mean(value * value, axis=-1, keepdims=True, dtype=np.float32)
    normalized = value * np.reciprocal(np.sqrt(variance + np.float32(eps))).astype(np.float32)
    if weight is None:
        return normalized.astype(np.float32)
    return (normalized * np.asarray(weight, dtype=np.float32)).astype(np.float32)


def gemma4_rope_tables(
    rope: Gemma4RopeConfig,
    positions: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos, sin)`` tables of shape ``(len(positions), head_dim // 2)``."""

    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    angles = np.outer(pos.astype(np.float64), rope.inverse_frequencies.astype(np.float64))
    return (
        np.cos(angles).astype(np.float32),
        np.sin(angles).astype(np.float32),
    )


def gemma4_router_topk(
    hidden: ArrayLike,
    *,
    norm_weight: ArrayLike | None,
    scale: ArrayLike,
    proj_weight: ArrayLike,
    per_expert_scale: ArrayLike,
    top_k: int,
    scalar_root_size: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(router_probabilities, top_k_weights, top_k_index)``.

    The router normalizes without a weight, applies the learned ``scale`` and
    ``hidden_size**-0.5``, projects to expert logits, softmaxes over all
    experts, keeps the top ``k``, renormalizes those weights to sum to one, and
    finally multiplies by ``per_expert_scale`` of the selected experts.
    """

    normalized = gemma4_rmsnorm(hidden, norm_weight, eps)
    scaled = (
        normalized
        * np.asarray(scale, dtype=np.float32)
        * np.float32(scalar_root_size)
    ).astype(np.float32)
    logits = linear(scaled, proj_weight)
    probabilities = _softmax(logits, axis=-1)

    k = int(top_k)
    # Descending order matches torch.topk. Ties are broken by lower index so the
    # selection is deterministic instead of depending on the sort implementation.
    order = np.argsort(-probabilities, axis=-1, kind="stable")[..., :k]
    weights = np.take_along_axis(probabilities, order, axis=-1)
    totals = weights.sum(axis=-1, keepdims=True, dtype=np.float32)
    weights = (weights / totals).astype(np.float32)
    expert_scale = np.asarray(per_expert_scale, dtype=np.float32)
    weights = (weights * expert_scale[order]).astype(np.float32)
    return probabilities, weights, order


def gemma4_attention_mask(
    geometry: Gemma4AttentionGeometry,
    query_positions: ArrayLike,
    key_positions: ArrayLike,
) -> np.ndarray:
    """Return a boolean ``(len(query), len(key))`` keep-mask."""

    query = np.asarray(query_positions, dtype=np.int64).reshape(-1)[:, None]
    key = np.asarray(key_positions, dtype=np.int64).reshape(-1)[None, :]
    keep = key <= query
    if geometry.sliding_window is not None:
        keep = keep & (key > query - int(geometry.sliding_window))
    return keep


def gemma4_attention_forward(
    hidden: ArrayLike,
    geometry: Gemma4AttentionGeometry,
    *,
    q_proj: ArrayLike,
    k_proj: ArrayLike,
    v_proj: ArrayLike | None,
    o_proj: ArrayLike,
    q_norm: ArrayLike,
    k_norm: ArrayLike,
    cos: ArrayLike,
    sin: ArrayLike,
    positions: ArrayLike,
    mask: ArrayLike | None,
    eps: float,
) -> np.ndarray:
    """Single-layer attention for a prefill block of tokens."""

    value = np.asarray(hidden, dtype=np.float32)
    tokens = value.shape[0]
    heads = geometry.num_heads
    kv_heads = geometry.num_kv_heads
    head_dim = geometry.head_dim

    query = linear(value, q_proj).reshape(tokens, heads, head_dim)
    query = gemma4_rmsnorm(query, q_norm, eps)
    query = _apply_rope(query, cos, sin, head_dim)

    key_raw = linear(value, k_proj).reshape(tokens, kv_heads, head_dim)
    # attention_k_eq_v removes v_proj, so V is the raw K projection. The k_norm
    # and the rotation apply to K only; V gets its own weightless norm and is
    # never rotated.
    if v_proj is None:
        value_states = key_raw
    else:
        value_states = linear(value, v_proj).reshape(tokens, kv_heads, head_dim)
    key = gemma4_rmsnorm(key_raw, k_norm, eps)
    key = _apply_rope(key, cos, sin, head_dim)
    value_states = gemma4_rmsnorm(value_states, None, eps)

    context = _grouped_attention(
        query,
        key,
        value_states,
        groups=geometry.num_kv_groups,
        scale=geometry.scale,
        mask=mask,
    )
    return linear(context.reshape(tokens, heads * head_dim), o_proj)


def gemma4_dense_mlp_forward(
    hidden: ArrayLike,
    *,
    gate_proj: ArrayLike,
    up_proj: ArrayLike,
    down_proj: ArrayLike,
) -> np.ndarray:
    """Dense SwiGLU-shaped MLP with the Gemma 4 tanh-GELU activation."""

    value = np.asarray(hidden, dtype=np.float32)
    activated = gemma4_gelu_tanh(linear(value, gate_proj)) * linear(value, up_proj)
    return linear(activated, down_proj)


def gemma4_experts_forward(
    hidden: ArrayLike,
    top_k_index: ArrayLike,
    top_k_weights: ArrayLike,
    *,
    gate_up_proj: ArrayLike,
    down_proj: ArrayLike,
) -> np.ndarray:
    """Routed expert FFN over 3-D stacked expert weights.

    ``gate_up_proj`` has shape ``(num_experts, 2 * moe_intermediate_size,
    hidden_size)`` and ``down_proj`` has shape ``(num_experts, hidden_size,
    moe_intermediate_size)``. The first half of the fused projection is the gate.
    """

    value = np.asarray(hidden, dtype=np.float32)
    index = np.asarray(top_k_index, dtype=np.int64)
    weights = np.asarray(top_k_weights, dtype=np.float32)
    gate_up = np.asarray(gate_up_proj, dtype=np.float32)
    down = np.asarray(down_proj, dtype=np.float32)

    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("expert weights must be 3-D stacked tensors")
    fused = gate_up.shape[1]
    if fused % 2:
        raise ValueError("fused gate_up expert width must be even")

    out = np.zeros_like(value)
    for token in range(value.shape[0]):
        row = value[token]
        for slot in range(index.shape[-1]):
            expert = int(index[token, slot])
            projection = gate_up[expert] @ row
            gate, up = np.split(projection, 2)
            activated = gemma4_gelu_tanh(gate) * up
            out[token] += (down[expert] @ activated) * weights[token, slot]
    return out


def gemma4_decoder_layer_forward(
    hidden: ArrayLike,
    layer: "Gemma4LayerWeights",
    geometry: Gemma4AttentionGeometry,
    config: Gemma4TextConfig,
    *,
    positions: ArrayLike,
) -> np.ndarray:
    """One Gemma 4 decoder layer, prefill over a block of tokens."""

    value = np.asarray(hidden, dtype=np.float32)
    eps = config.rms_norm_eps

    cos, sin = gemma4_rope_tables(geometry.rope, positions)
    cos_expanded = cos[:, None, :]
    sin_expanded = sin[:, None, :]
    mask = gemma4_attention_mask(geometry, positions, positions)

    residual = value
    normalized = gemma4_rmsnorm(value, layer.input_layernorm, eps)
    attended = gemma4_attention_forward(
        normalized,
        geometry,
        q_proj=layer.q_proj,
        k_proj=layer.k_proj,
        v_proj=layer.v_proj,
        o_proj=layer.o_proj,
        q_norm=layer.q_norm,
        k_norm=layer.k_norm,
        cos=cos_expanded,
        sin=sin_expanded,
        positions=positions,
        mask=mask,
        eps=eps,
    )
    hidden_after_attention = residual + gemma4_rmsnorm(attended, layer.post_attention_layernorm, eps)

    # Both feed-forward branches run in parallel off the pre-MLP residual.
    residual = hidden_after_attention
    dense_input = gemma4_rmsnorm(hidden_after_attention, layer.pre_feedforward_layernorm, eps)
    dense = gemma4_dense_mlp_forward(
        dense_input,
        gate_proj=layer.mlp_gate_proj,
        up_proj=layer.mlp_up_proj,
        down_proj=layer.mlp_down_proj,
    )
    dense = gemma4_rmsnorm(dense, layer.post_feedforward_layernorm_1, eps)

    _, top_k_weights, top_k_index = gemma4_router_topk(
        residual,
        norm_weight=None,
        scale=layer.router_scale,
        proj_weight=layer.router_proj,
        per_expert_scale=layer.router_per_expert_scale,
        top_k=config.top_k_experts,
        scalar_root_size=config.router_scalar_root_size,
        eps=eps,
    )
    expert_input = gemma4_rmsnorm(residual, layer.pre_feedforward_layernorm_2, eps)
    experts = gemma4_experts_forward(
        expert_input,
        top_k_index,
        top_k_weights,
        gate_up_proj=layer.experts_gate_up_proj,
        down_proj=layer.experts_down_proj,
    )
    experts = gemma4_rmsnorm(experts, layer.post_feedforward_layernorm_2, eps)

    combined = gemma4_rmsnorm(dense + experts, layer.post_feedforward_layernorm, eps)
    out = residual + combined
    return (out * np.asarray(layer.layer_scalar, dtype=np.float32).reshape(())).astype(np.float32)


def gemma4_text_forward(
    weights: "Gemma4TextWeights",
    config: Gemma4TextConfig,
    token_ids: ArrayLike,
    *,
    positions: ArrayLike | None = None,
    capture_hidden_states: bool = False,
) -> "Gemma4ForwardResult":
    """Full Gemma 4 text forward for one prompt, returning fp32 logits."""

    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if positions is None:
        positions = np.arange(ids.shape[0], dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.shape[0] != ids.shape[0]:
        raise ValueError("positions must have one entry per token")

    hidden = (np.asarray(weights.embed_tokens, dtype=np.float32)[ids] * np.float32(config.embed_scale)).astype(
        np.float32
    )
    captured: list[np.ndarray] = []
    if capture_hidden_states:
        captured.append(hidden.copy())

    for layer_index in range(config.num_hidden_layers):
        hidden = gemma4_decoder_layer_forward(
            hidden,
            weights.layers[layer_index],
            config.geometry(layer_index),
            config,
            positions=positions,
        )
        if capture_hidden_states:
            captured.append(hidden.copy())

    hidden = gemma4_rmsnorm(hidden, weights.final_norm, config.rms_norm_eps)
    head = weights.lm_head if weights.lm_head is not None else weights.embed_tokens
    logits = linear(hidden, head)
    if config.final_logit_softcapping is not None:
        cap = np.float32(config.final_logit_softcapping)
        logits = (np.tanh(logits / cap) * cap).astype(np.float32)
    return Gemma4ForwardResult(
        logits=logits,
        hidden_states=tuple(captured) if capture_hidden_states else (),
    )


# ---------------------------------------------------------------------------
# Weight containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemma4LayerWeights:
    """One decoder layer's reference weights."""

    input_layernorm: ArrayLike
    post_attention_layernorm: ArrayLike
    pre_feedforward_layernorm: ArrayLike
    post_feedforward_layernorm: ArrayLike
    post_feedforward_layernorm_1: ArrayLike
    post_feedforward_layernorm_2: ArrayLike
    pre_feedforward_layernorm_2: ArrayLike
    q_proj: ArrayLike
    k_proj: ArrayLike
    o_proj: ArrayLike
    q_norm: ArrayLike
    k_norm: ArrayLike
    mlp_gate_proj: ArrayLike
    mlp_up_proj: ArrayLike
    mlp_down_proj: ArrayLike
    router_scale: ArrayLike
    router_proj: ArrayLike
    router_per_expert_scale: ArrayLike
    experts_gate_up_proj: ArrayLike
    experts_down_proj: ArrayLike
    layer_scalar: ArrayLike
    v_proj: ArrayLike | None = None


@dataclass(frozen=True)
class Gemma4TextWeights:
    """Reference weights for the whole Gemma 4 text tower."""

    embed_tokens: ArrayLike
    final_norm: ArrayLike
    layers: tuple[Gemma4LayerWeights, ...]
    lm_head: ArrayLike | None = None


@dataclass(frozen=True)
class Gemma4ForwardResult:
    logits: np.ndarray
    hidden_states: tuple[np.ndarray, ...] = ()


def gemma4_text_weights_from_hf(
    state_dict: Mapping[str, ArrayLike],
    config: Gemma4TextConfig,
    *,
    prefix: str = "model.",
) -> Gemma4TextWeights:
    """Adapt an HF ``Gemma4ForCausalLM`` state dict into reference weights.

    ``attention_k_eq_v`` layers carry no ``v_proj``; the adapter accepts its
    absence and rejects a present one so a mismatched checkpoint cannot silently
    take the wrong path.
    """

    def get(name: str) -> np.ndarray:
        return np.asarray(state_dict[f"{prefix}{name}"], dtype=np.float32)

    def maybe(name: str) -> np.ndarray | None:
        key = f"{prefix}{name}"
        return None if key not in state_dict else np.asarray(state_dict[key], dtype=np.float32)

    layers: list[Gemma4LayerWeights] = []
    for index in range(config.num_hidden_layers):
        base = f"layers.{index}"
        geometry = config.geometry(index)
        v_proj = maybe(f"{base}.self_attn.v_proj.weight")
        if geometry.k_eq_v and v_proj is not None:
            raise ValueError(f"layer {index} declares attention_k_eq_v but carries v_proj")
        layers.append(
            Gemma4LayerWeights(
                input_layernorm=get(f"{base}.input_layernorm.weight"),
                post_attention_layernorm=get(f"{base}.post_attention_layernorm.weight"),
                pre_feedforward_layernorm=get(f"{base}.pre_feedforward_layernorm.weight"),
                post_feedforward_layernorm=get(f"{base}.post_feedforward_layernorm.weight"),
                post_feedforward_layernorm_1=get(f"{base}.post_feedforward_layernorm_1.weight"),
                post_feedforward_layernorm_2=get(f"{base}.post_feedforward_layernorm_2.weight"),
                pre_feedforward_layernorm_2=get(f"{base}.pre_feedforward_layernorm_2.weight"),
                q_proj=get(f"{base}.self_attn.q_proj.weight"),
                k_proj=get(f"{base}.self_attn.k_proj.weight"),
                v_proj=v_proj,
                o_proj=get(f"{base}.self_attn.o_proj.weight"),
                q_norm=get(f"{base}.self_attn.q_norm.weight"),
                k_norm=get(f"{base}.self_attn.k_norm.weight"),
                mlp_gate_proj=get(f"{base}.mlp.gate_proj.weight"),
                mlp_up_proj=get(f"{base}.mlp.up_proj.weight"),
                mlp_down_proj=get(f"{base}.mlp.down_proj.weight"),
                router_scale=get(f"{base}.router.scale"),
                router_proj=get(f"{base}.router.proj.weight"),
                router_per_expert_scale=get(f"{base}.router.per_expert_scale"),
                experts_gate_up_proj=get(f"{base}.experts.gate_up_proj"),
                experts_down_proj=get(f"{base}.experts.down_proj"),
                layer_scalar=get(f"{base}.layer_scalar"),
            )
        )

    return Gemma4TextWeights(
        embed_tokens=get("embed_tokens.weight"),
        final_norm=get("norm.weight"),
        layers=tuple(layers),
        lm_head=maybe("lm_head.weight"),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray, head_dim: int) -> np.ndarray:
    """Split-half rotation over the full head width with a half-width table.

    Gemma 4 rotates ``x[..., i]`` against ``x[..., i + head_dim // 2]`` for every
    pair ``i``. Pairs past the rotated span carry an inverse frequency of zero,
    so their cosine is one and their sine is zero, which leaves them untouched.
    """

    cos_arr = np.asarray(cos, dtype=np.float32)
    sin_arr = np.asarray(sin, dtype=np.float32)
    if cos_arr.shape[-1] != head_dim // 2 or sin_arr.shape[-1] != head_dim // 2:
        raise ValueError("rotary tables must carry head_dim // 2 angles")
    return rotate(value, cos_arr, sin_arr, rotary_dim=head_dim)


def _grouped_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    groups: int,
    scale: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Repeat-KV grouped attention returning ``(tokens, heads, head_dim)``."""

    tokens, kv_heads, head_dim = key.shape
    if groups > 1:
        key = np.repeat(key, groups, axis=1)
        value = np.repeat(value, groups, axis=1)

    scores = np.einsum("thd,shd->hts", query.astype(np.float32), key.astype(np.float32))
    scores = (scores * np.float32(scale)).astype(np.float32)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (tokens, key.shape[0]):
            raise ValueError("attention mask shape must be (query, key)")
        scores = np.where(keep[None, :, :], scores, np.float32(-np.inf))
    weights = _softmax(scores, axis=-1)
    context = np.einsum("hts,shd->thd", weights.astype(np.float32), value.astype(np.float32))
    return context.astype(np.float32)


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    shifted = (x - np.max(x, axis=axis, keepdims=True)).astype(np.float32)
    numerator = np.exp(shifted).astype(np.float32)
    total = numerator.sum(axis=axis, keepdims=True, dtype=np.float32)
    return (numerator / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_gemma4_cpu_reference_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 reference entry points under the CPU backend."""

    register(
        KernelKey("cpu_reference", "gemma4_text", "fp32", "reference_forward"),
        gemma4_text_forward,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "gemma4_text", "fp32", "router_topk"),
        gemma4_router_topk,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "gemma4_text", "fp32", "gelu_tanh"),
        gemma4_gelu_tanh,
        replace=replace,
    )


__all__ = [
    "FULL_ATTENTION",
    "SLIDING_ATTENTION",
    "Gemma4AttentionGeometry",
    "Gemma4ForwardResult",
    "Gemma4LayerWeights",
    "Gemma4RopeConfig",
    "Gemma4TextConfig",
    "Gemma4TextWeights",
    "gemma4_attention_forward",
    "gemma4_attention_mask",
    "gemma4_decoder_layer_forward",
    "gemma4_dense_mlp_forward",
    "gemma4_experts_forward",
    "gemma4_gelu_tanh",
    "gemma4_rmsnorm",
    "gemma4_rope_tables",
    "gemma4_router_topk",
    "gemma4_text_config_from_hf",
    "gemma4_text_forward",
    "gemma4_text_weights_from_hf",
    "register_gemma4_cpu_reference_kernels",
]
