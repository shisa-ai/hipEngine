"""Torch-free NumPy reference primitives for Laguna."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hipengine.kernels.cpu_reference.ops import linear, rmsnorm, rotate
from hipengine.kernels.registry import KernelKey, register

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class LagunaRoutingResult:
    """Intermediate and selected values from Laguna sigmoid MoE routing."""

    router_logits: np.ndarray
    routing_scores: np.ndarray
    selection_scores: np.ndarray
    selected_experts: np.ndarray
    routing_weights: np.ndarray
    scaled_routing_weights: np.ndarray


@dataclass(frozen=True)
class LagunaRopeConfig:
    """Per-layer rotary contract after GGUF metadata normalization.

    ``yarn_attn_factor`` is the GGUF multiplier. For YaRN, ggml additionally
    applies ``1 + 0.1 * log(scaling_factor)`` to cosine and sine magnitudes,
    matching the default Hugging Face YaRN attention scaling.
    """

    rope_type: str
    rotary_dim: int
    freq_base: float
    scaling_factor: float = 1.0
    original_context_length: int = 0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0


@dataclass(frozen=True)
class LagunaAttentionConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope: LagunaRopeConfig
    sliding_window: int | None = None


@dataclass(frozen=True)
class LagunaAttentionWeights:
    input_norm: ArrayLike
    q_proj: ArrayLike
    k_proj: ArrayLike
    v_proj: ArrayLike
    gate_proj: ArrayLike
    q_norm: ArrayLike
    k_norm: ArrayLike
    o_proj: ArrayLike


@dataclass(frozen=True)
class LagunaDenseFFNWeights:
    gate_proj: ArrayLike
    up_proj: ArrayLike
    down_proj: ArrayLike


@dataclass(frozen=True)
class LagunaSparseFFNWeights:
    router: ArrayLike
    correction_bias: ArrayLike
    expert_gate: ArrayLike
    expert_up: ArrayLike
    expert_down: ArrayLike
    shared_gate: ArrayLike
    shared_up: ArrayLike
    shared_down: ArrayLike
    experts_used: int
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    router_logit_softcapping: float = 0.0


@dataclass(frozen=True)
class LagunaLayerWeights:
    attention: LagunaAttentionWeights
    ffn_norm: ArrayLike
    mlp: LagunaDenseFFNWeights | LagunaSparseFFNWeights


@dataclass(frozen=True)
class LagunaReferenceLayer:
    config: LagunaAttentionConfig
    weights: LagunaLayerWeights


@dataclass(frozen=True)
class LagunaAttentionResult:
    normalized: np.ndarray
    query_normalized: np.ndarray
    key_normalized: np.ndarray
    value: np.ndarray
    gate_logits: np.ndarray
    context: np.ndarray
    gated_context: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class LagunaSparseMoEResult:
    routing: LagunaRoutingResult
    routed_output_unscaled: np.ndarray
    routed_output: np.ndarray
    shared_output: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class LagunaLayerResult:
    attention: LagunaAttentionResult
    post_attention: np.ndarray
    ffn_normalized: np.ndarray
    ffn_output: np.ndarray
    hidden: np.ndarray
    sparse_moe: LagunaSparseMoEResult | None = None


@dataclass(frozen=True)
class LagunaModelResult:
    hidden_states: tuple[np.ndarray, ...]
    layers: tuple[LagunaLayerResult, ...]
    final_hidden: np.ndarray
    logits: np.ndarray


def laguna_rope_tables(
    positions: ArrayLike,
    config: LagunaRopeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build split-half FP32 RoPE tables for Laguna full or SWA layers.

    The YaRN branch follows both Transformers' ``_compute_yarn_parameters`` and
    ggml's ``rope_yarn`` equations. The returned tables have the full rotary
    width: each half-frequency vector is repeated for split-half rotation.
    """

    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 1:
        raise ValueError("positions must have shape [tokens]")
    if np.any(pos < 0):
        raise ValueError("positions must be non-negative")
    rope_type = str(config.rope_type)
    if rope_type not in {"default", "yarn"}:
        raise ValueError("Laguna rope_type must be 'default' or 'yarn'")
    rotary_dim = int(config.rotary_dim)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    freq_base = float(config.freq_base)
    if not math.isfinite(freq_base) or freq_base <= 0.0:
        raise ValueError("freq_base must be finite and positive")

    # torch.pow computes this scalar-base expression at double precision before
    # narrowing to FP32. Preserve that rounding: a one-ULP inverse-frequency
    # difference becomes a visible phase error at position 262143.
    exponent = np.arange(0, rotary_dim, 2, dtype=np.float64) / float(rotary_dim)
    pos_freqs = np.power(freq_base, exponent).astype(np.float32)
    inv_freq = np.reciprocal(pos_freqs).astype(np.float32)
    attention_scale = np.float32(1.0)

    if rope_type == "yarn":
        factor = float(config.scaling_factor)
        original_context = int(config.original_context_length)
        beta_fast = float(config.yarn_beta_fast)
        beta_slow = float(config.yarn_beta_slow)
        attn_factor = float(config.yarn_attn_factor)
        if not math.isfinite(factor) or factor < 1.0:
            raise ValueError("YaRN scaling_factor must be finite and at least one")
        if original_context <= 0:
            raise ValueError("YaRN original_context_length must be positive")
        if beta_fast <= 0.0 or beta_slow <= 0.0 or beta_fast < beta_slow:
            raise ValueError("YaRN beta values must satisfy beta_fast >= beta_slow > 0")
        if not math.isfinite(attn_factor) or attn_factor <= 0.0:
            raise ValueError("YaRN yarn_attn_factor must be finite and positive")

        low = max(
            0.0,
            math.floor(
                rotary_dim
                * math.log(original_context / (beta_fast * 2.0 * math.pi))
                / (2.0 * math.log(freq_base))
            ),
        )
        high = min(
            float(rotary_dim - 1),
            math.ceil(
                rotary_dim
                * math.log(original_context / (beta_slow * 2.0 * math.pi))
                / (2.0 * math.log(freq_base))
            ),
        )
        ramp_denom = max(0.001, high - low)
        ramp = np.clip(
            (np.arange(rotary_dim // 2, dtype=np.float32) - np.float32(low))
            / np.float32(ramp_denom),
            np.float32(0.0),
            np.float32(1.0),
        )
        extrapolation_mix = (np.float32(1.0) - ramp).astype(np.float32)
        interpolated = np.reciprocal(np.float32(factor) * pos_freqs).astype(np.float32)
        inv_freq = (
            interpolated * (np.float32(1.0) - extrapolation_mix) + inv_freq * extrapolation_mix
        ).astype(np.float32)
        default_mscale = 1.0 if factor <= 1.0 else 1.0 + 0.1 * math.log(factor)
        attention_scale = np.float32(attn_factor * default_mscale)

    frequencies = (pos.astype(np.float32)[:, None] * inv_freq[None, :]).astype(np.float32)
    cos_half = (np.cos(frequencies) * attention_scale).astype(np.float32)
    sin_half = (np.sin(frequencies) * attention_scale).astype(np.float32)
    return (
        np.concatenate((cos_half, cos_half), axis=-1),
        np.concatenate((sin_half, sin_half), axis=-1),
    )


def laguna_apply_rope(
    values: ArrayLike,
    cos: ArrayLike,
    sin: ArrayLike,
    *,
    rotary_dim: int,
) -> np.ndarray:
    """Apply Laguna's split-half RoPE while preserving non-rotary channels."""

    return rotate(values, cos, sin, rotary_dim=rotary_dim).astype(np.float32)


def laguna_causal_mask(
    query_positions: ArrayLike,
    key_positions: ArrayLike,
    *,
    sliding_window: int | None = None,
) -> np.ndarray:
    """Return visibility from absolute positions, independent of physical slots.

    A 512-token window follows Transformers' strict lower boundary:
    ``query - 512 < key <= query``. Thus query positions 510/511/512 expose
    511/512/512 keys for a sequence beginning at zero.
    """

    query = np.asarray(query_positions, dtype=np.int64)
    key = np.asarray(key_positions, dtype=np.int64)
    if query.ndim != 1 or key.ndim != 1:
        raise ValueError("query_positions and key_positions must be rank one")
    if np.any(query < 0) or np.any(key < 0):
        raise ValueError("query_positions and key_positions must be non-negative")
    mask = key[None, :] <= query[:, None]
    if sliding_window is not None:
        window = int(sliding_window)
        if window <= 0:
            raise ValueError("sliding_window must be positive")
        mask &= key[None, :] > query[:, None] - window
    return mask


def laguna_head_rmsnorm(
    values: ArrayLike,
    weight: ArrayLike,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Apply FP32 RMSNorm independently over every Laguna head or hidden row."""

    value = np.asarray(values, dtype=np.float32)
    norm_weight = np.asarray(weight, dtype=np.float32)
    if value.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if norm_weight.shape != (value.shape[-1],):
        raise ValueError("weight must match values.shape[-1]")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not np.isfinite(value).all() or not np.isfinite(norm_weight).all():
        raise ValueError("values and weight must be finite")
    return rmsnorm(value, norm_weight, eps=float(eps)).astype(np.float32)


def laguna_attention_forward(
    hidden: ArrayLike,
    weights: LagunaAttentionWeights,
    config: LagunaAttentionConfig,
    *,
    positions: ArrayLike,
    eps: float = 1.0e-6,
) -> LagunaAttentionResult:
    """Run Laguna pre-norm GQA, per-head gate, and output projection in FP32."""

    x = np.asarray(hidden, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    if pos.shape != (tokens,):
        raise ValueError("positions must have shape [tokens]")
    heads = int(config.num_heads)
    kv_heads = int(config.num_kv_heads)
    head_dim = int(config.head_dim)
    if heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if heads % kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if config.rope.rotary_dim > head_dim:
        raise ValueError("rotary_dim cannot exceed head_dim")

    input_norm = np.asarray(weights.input_norm, dtype=np.float32)
    q_proj = np.asarray(weights.q_proj, dtype=np.float32)
    k_proj = np.asarray(weights.k_proj, dtype=np.float32)
    v_proj = np.asarray(weights.v_proj, dtype=np.float32)
    gate_proj = np.asarray(weights.gate_proj, dtype=np.float32)
    q_norm = np.asarray(weights.q_norm, dtype=np.float32)
    k_norm = np.asarray(weights.k_norm, dtype=np.float32)
    o_proj = np.asarray(weights.o_proj, dtype=np.float32)
    expected_shapes = {
        "input_norm": (hidden_size,),
        "q_proj": (heads * head_dim, hidden_size),
        "k_proj": (kv_heads * head_dim, hidden_size),
        "v_proj": (kv_heads * head_dim, hidden_size),
        "gate_proj": (heads, hidden_size),
        "q_norm": (head_dim,),
        "k_norm": (head_dim,),
        "o_proj": (hidden_size, heads * head_dim),
    }
    actual_arrays = {
        "input_norm": input_norm,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "gate_proj": gate_proj,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "o_proj": o_proj,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_arrays[name].shape}"
            )

    normalized = laguna_head_rmsnorm(x, input_norm, eps=eps)
    query = linear(normalized, q_proj).reshape(tokens, heads, head_dim)
    key = linear(normalized, k_proj).reshape(tokens, kv_heads, head_dim)
    value = linear(normalized, v_proj).reshape(tokens, kv_heads, head_dim)
    gate_logits = linear(normalized, gate_proj).astype(np.float32)
    query_normalized = laguna_head_rmsnorm(query, q_norm, eps=eps)
    key_normalized = laguna_head_rmsnorm(key, k_norm, eps=eps)
    cos, sin = laguna_rope_tables(pos, config.rope)
    query_rotated = laguna_apply_rope(
        query_normalized,
        cos[:, None, :],
        sin[:, None, :],
        rotary_dim=config.rope.rotary_dim,
    )
    key_rotated = laguna_apply_rope(
        key_normalized,
        cos[:, None, :],
        sin[:, None, :],
        rotary_dim=config.rope.rotary_dim,
    )

    visible = laguna_causal_mask(
        pos,
        pos,
        sliding_window=config.sliding_window,
    )
    head_group = heads // kv_heads
    attention = np.empty((tokens, heads, head_dim), dtype=np.float32)
    scale = np.float32(head_dim**-0.5)
    for token in range(tokens):
        token_keys = np.flatnonzero(visible[token])
        if token_keys.size == 0:
            raise ValueError("causal mask left no visible key positions")
        for head in range(heads):
            kv_head = head // head_group
            logits = (
                np.matmul(
                    key_rotated[token_keys, kv_head],
                    query_rotated[token, head],
                )
                * scale
            ).astype(np.float32)
            probabilities = _softmax(logits)
            attention[token, head] = np.matmul(
                probabilities,
                value[token_keys, kv_head],
            ).astype(np.float32)

    gated = laguna_softplus_head_gate(attention, gate_logits)
    output = linear(gated.reshape(tokens, heads * head_dim), o_proj).astype(np.float32)
    return LagunaAttentionResult(
        normalized=normalized,
        query_normalized=query_normalized,
        key_normalized=key_normalized,
        value=value,
        gate_logits=gate_logits,
        context=attention,
        gated_context=gated,
        output=output,
    )


def laguna_dense_ffn_forward(
    hidden: ArrayLike,
    weights: LagunaDenseFFNWeights,
) -> np.ndarray:
    """Run the Laguna parallel SwiGLU dense FFN without its residual."""

    x = np.asarray(hidden, dtype=np.float32)
    gate = np.asarray(weights.gate_proj, dtype=np.float32)
    up = np.asarray(weights.up_proj, dtype=np.float32)
    down = np.asarray(weights.down_proj, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    if gate.ndim != 2 or up.shape != gate.shape or gate.shape[1] != x.shape[1]:
        raise ValueError("dense gate/up weights must have matching [intermediate, hidden] shape")
    if down.shape != (x.shape[1], gate.shape[0]):
        raise ValueError("dense down weight must have shape [hidden, intermediate]")
    activated = (_silu(linear(x, gate)) * linear(x, up)).astype(np.float32)
    return linear(activated, down).astype(np.float32)


def laguna_sparse_moe_forward(
    hidden: ArrayLike,
    weights: LagunaSparseFFNWeights,
) -> LagunaSparseMoEResult:
    """Run routed and always-on shared Laguna experts without the residual."""

    x = np.asarray(hidden, dtype=np.float32)
    expert_gate = np.asarray(weights.expert_gate, dtype=np.float32)
    expert_up = np.asarray(weights.expert_up, dtype=np.float32)
    expert_down = np.asarray(weights.expert_down, dtype=np.float32)
    shared_gate = np.asarray(weights.shared_gate, dtype=np.float32)
    shared_up = np.asarray(weights.shared_up, dtype=np.float32)
    shared_down = np.asarray(weights.shared_down, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    experts = expert_gate.shape[0] if expert_gate.ndim == 3 else 0
    intermediate = expert_gate.shape[1] if experts else 0
    hidden_size = x.shape[1]
    if experts <= 0 or intermediate <= 0 or expert_gate.shape[2] != hidden_size:
        raise ValueError("expert_gate must have shape [experts, intermediate, hidden]")
    if expert_up.shape != expert_gate.shape:
        raise ValueError("expert_up must match expert_gate")
    if expert_down.shape != (experts, hidden_size, intermediate):
        raise ValueError("expert_down must have shape [experts, hidden, intermediate]")
    if shared_gate.ndim != 2 or shared_up.shape != shared_gate.shape:
        raise ValueError("shared gate/up must have matching [intermediate, hidden] shape")
    if shared_gate.shape[1] != hidden_size or shared_down.shape != (
        hidden_size,
        shared_gate.shape[0],
    ):
        raise ValueError("shared expert shapes are inconsistent with hidden size")

    routing = laguna_sigmoid_correction_topk(
        x,
        weights.router,
        weights.correction_bias,
        experts_used=weights.experts_used,
        routed_scaling_factor=weights.routed_scaling_factor,
        norm_topk_prob=weights.norm_topk_prob,
        router_logit_softcapping=weights.router_logit_softcapping,
    )
    if np.max(routing.selected_experts, initial=-1) >= experts:
        raise ValueError("router selected an expert outside expert weight storage")

    routed_unscaled = np.zeros_like(x)
    for token in range(x.shape[0]):
        token_input = x[token : token + 1]
        for route in range(routing.selected_experts.shape[1]):
            expert = int(routing.selected_experts[token, route])
            gate = linear(token_input, expert_gate[expert])
            up = linear(token_input, expert_up[expert])
            down = linear(
                (_silu(gate) * up).astype(np.float32),
                expert_down[expert],
            )
            routed_unscaled[token] += (down[0] * routing.routing_weights[token, route]).astype(
                np.float32
            )
    routed = (routed_unscaled * np.float32(weights.routed_scaling_factor)).astype(np.float32)
    shared = linear(
        (_silu(linear(x, shared_gate)) * linear(x, shared_up)).astype(np.float32),
        shared_down,
    ).astype(np.float32)
    output = (routed + shared).astype(np.float32)
    return LagunaSparseMoEResult(
        routing=routing,
        routed_output_unscaled=routed_unscaled,
        routed_output=routed,
        shared_output=shared,
        output=output,
    )


def laguna_layer_forward(
    hidden: ArrayLike,
    layer: LagunaReferenceLayer,
    *,
    positions: ArrayLike,
    eps: float = 1.0e-6,
) -> LagunaLayerResult:
    """Run one exact unfused Laguna layer in architecture residual order."""

    x = np.asarray(hidden, dtype=np.float32)
    attention = laguna_attention_forward(
        x,
        layer.weights.attention,
        layer.config,
        positions=positions,
        eps=eps,
    )
    post_attention = (x + attention.output).astype(np.float32)
    ffn_normalized = laguna_head_rmsnorm(
        post_attention,
        layer.weights.ffn_norm,
        eps=eps,
    )
    sparse_result: LagunaSparseMoEResult | None = None
    if isinstance(layer.weights.mlp, LagunaDenseFFNWeights):
        ffn_output = laguna_dense_ffn_forward(ffn_normalized, layer.weights.mlp)
    elif isinstance(layer.weights.mlp, LagunaSparseFFNWeights):
        sparse_result = laguna_sparse_moe_forward(ffn_normalized, layer.weights.mlp)
        ffn_output = sparse_result.output
    else:
        raise TypeError("Laguna layer MLP weights must be dense or sparse")
    output = (post_attention + ffn_output).astype(np.float32)
    return LagunaLayerResult(
        attention=attention,
        post_attention=post_attention,
        ffn_normalized=ffn_normalized,
        ffn_output=ffn_output,
        hidden=output,
        sparse_moe=sparse_result,
    )


def laguna_model_forward(
    input_ids: ArrayLike,
    embedding_weight: ArrayLike,
    layers: tuple[LagunaReferenceLayer, ...] | list[LagunaReferenceLayer],
    final_norm_weight: ArrayLike,
    output_weight: ArrayLike | None,
    *,
    positions: ArrayLike | None = None,
    eps: float = 1.0e-6,
) -> LagunaModelResult:
    """Run a compact Laguna model and explicit untied LM head in FP32."""

    token_ids = np.asarray(input_ids, dtype=np.int64)
    embedding = np.asarray(embedding_weight, dtype=np.float32)
    if token_ids.ndim != 1:
        raise ValueError("input_ids must have shape [tokens]")
    if embedding.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    if np.any(token_ids < 0) or np.any(token_ids >= embedding.shape[0]):
        raise ValueError("input_ids contain an out-of-range token")
    if output_weight is None:
        raise ValueError("Laguna reference requires an explicit untied output_weight")
    output = np.asarray(output_weight, dtype=np.float32)
    if output.ndim != 2 or output.shape[1] != embedding.shape[1]:
        raise ValueError("output_weight must have shape [vocab, hidden]")
    position_array = (
        np.arange(token_ids.size, dtype=np.int64)
        if positions is None
        else np.asarray(positions, dtype=np.int64)
    )
    if position_array.shape != token_ids.shape:
        raise ValueError("positions must match input_ids")

    hidden = np.ascontiguousarray(embedding[token_ids])
    hidden_states: list[np.ndarray] = [hidden.copy()]
    layer_results: list[LagunaLayerResult] = []
    for layer in layers:
        result = laguna_layer_forward(
            hidden,
            layer,
            positions=position_array,
            eps=eps,
        )
        layer_results.append(result)
        hidden = result.hidden
        hidden_states.append(hidden.copy())
    final_hidden = laguna_head_rmsnorm(hidden, final_norm_weight, eps=eps)
    logits = linear(final_hidden, output).astype(np.float32)
    return LagunaModelResult(
        hidden_states=tuple(hidden_states),
        layers=tuple(layer_results),
        final_hidden=final_hidden,
        logits=logits,
    )


def laguna_softplus_head_gate(
    attention: ArrayLike,
    gate_logits: ArrayLike,
) -> np.ndarray:
    """Apply Laguna's FP32 softplus per-head attention gate.

    ``attention`` is shaped ``[..., heads, head_dim]`` and ``gate_logits`` is
    shaped ``[..., heads]``. One scalar is broadcast across each head's final
    dimension before the output projection.
    """

    attn = np.asarray(attention, dtype=np.float32)
    gate = np.asarray(gate_logits, dtype=np.float32)
    if attn.ndim < 2:
        raise ValueError("attention must have shape [..., heads, head_dim]")
    if gate.shape != attn.shape[:-1]:
        raise ValueError("gate_logits must match attention leading dimensions [..., heads]")
    if not np.isfinite(attn).all() or not np.isfinite(gate).all():
        raise ValueError("attention and gate_logits must be finite")
    scale = np.logaddexp(np.float32(0.0), gate).astype(np.float32)
    return (attn * scale[..., None]).astype(np.float32)


def laguna_sigmoid_correction_topk(
    hidden: ArrayLike,
    router_weight: ArrayLike,
    correction_bias: ArrayLike,
    *,
    experts_used: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
    router_logit_softcapping: float = 0.0,
) -> LagunaRoutingResult:
    """Reference Laguna sigmoid routing with selection-only correction bias.

    Selection uses ``sigmoid(logits) + correction_bias``. Returned routing
    weights are gathered from the uncorrected sigmoid scores, optionally
    sum-normalized, then exposed both before and after routed-output scaling.
    Stable descending selection keeps the lower expert ID on exact ties.
    """

    x = np.asarray(hidden, dtype=np.float32)
    router = np.asarray(router_weight, dtype=np.float32)
    bias = np.asarray(correction_bias, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    if router.ndim != 2 or router.shape[1] != x.shape[1]:
        raise ValueError("router_weight must have shape [experts, hidden]")
    if bias.shape != (router.shape[0],):
        raise ValueError("correction_bias must have shape [experts]")
    if not np.isfinite(x).all() or not np.isfinite(router).all() or not np.isfinite(bias).all():
        raise ValueError("hidden, router_weight, and correction_bias must be finite")

    top_k = int(experts_used)
    if top_k <= 0 or top_k > router.shape[0]:
        raise ValueError("experts_used must be within [1, number of experts]")
    softcap = float(router_logit_softcapping)
    if softcap < 0.0:
        raise ValueError("router_logit_softcapping must be non-negative")

    logits = np.matmul(x, router.T).astype(np.float32)
    if softcap > 0.0:
        logits = (np.tanh(logits / np.float32(softcap)) * np.float32(softcap)).astype(np.float32)
    return laguna_sigmoid_correction_topk_from_logits(
        logits,
        bias,
        experts_used=top_k,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
    )


def laguna_sigmoid_correction_topk_from_logits(
    router_logits: ArrayLike,
    correction_bias: ArrayLike,
    *,
    experts_used: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> LagunaRoutingResult:
    """Apply Laguna's selection-only correction and routing to FP32 logits.

    This is the direct oracle for the native post-projection router kernel. The
    logits are assumed to be after any model-specific softcap. Stable descending
    selection keeps the lower expert ID on exact ties.
    """

    logits = np.asarray(router_logits, dtype=np.float32)
    bias = np.asarray(correction_bias, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if bias.shape != (logits.shape[1],):
        raise ValueError("correction_bias must have shape [experts]")
    if not np.isfinite(logits).all() or not np.isfinite(bias).all():
        raise ValueError("router_logits and correction_bias must be finite")
    top_k = int(experts_used)
    if top_k <= 0 or top_k > logits.shape[1]:
        raise ValueError("experts_used must be within [1, number of experts]")
    scale = float(routed_scaling_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("routed_scaling_factor must be finite and positive")

    routing_scores = _stable_sigmoid(logits)
    selection_scores = (routing_scores + bias[None, :]).astype(np.float32)
    selected = np.argsort(-selection_scores, axis=-1, kind="stable")[:, :top_k].astype(np.int64)
    weights = np.take_along_axis(routing_scores, selected, axis=-1).astype(np.float32)
    if norm_topk_prob:
        denominator = np.maximum(
            weights.sum(axis=-1, keepdims=True, dtype=np.float32),
            np.float32(np.finfo(np.float32).tiny),
        )
        weights = (weights / denominator).astype(np.float32)
    scaled = (weights * np.float32(scale)).astype(np.float32)
    return LagunaRoutingResult(
        router_logits=logits,
        routing_scores=routing_scores,
        selection_scores=selection_scores,
        selected_experts=selected,
        routing_weights=weights,
        scaled_routing_weights=scaled,
    )


def register_laguna_cpu_reference_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "cpu_reference",
            "softplus_head_gate",
            "fp32",
            "laguna_per_head",
        ),
        laguna_softplus_head_gate,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "laguna_sigmoid_router_topk",
            "gguf_f32",
            "correction_bias",
        ),
        laguna_sigmoid_correction_topk,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "laguna_rope_tables",
            "fp32",
            "yarn_or_default",
        ),
        laguna_rope_tables,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "laguna_model",
            "fp32",
            "two_layer_reference",
        ),
        laguna_model_forward,
        replace=replace,
    )


def _softmax(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    shifted = (x - np.max(x)).astype(np.float32)
    numerator = np.exp(shifted).astype(np.float32)
    return (numerator / numerator.sum(dtype=np.float32)).astype(np.float32)


def _silu(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return (x * _stable_sigmoid(x)).astype(np.float32)


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-x[positive]).astype(np.float32))
    negative_exp = np.exp(x[~positive]).astype(np.float32)
    out[~positive] = negative_exp / (np.float32(1.0) + negative_exp)
    return out


__all__ = [
    "LagunaAttentionConfig",
    "LagunaAttentionResult",
    "LagunaAttentionWeights",
    "LagunaDenseFFNWeights",
    "LagunaLayerResult",
    "LagunaLayerWeights",
    "LagunaModelResult",
    "LagunaReferenceLayer",
    "LagunaRopeConfig",
    "LagunaRoutingResult",
    "LagunaSparseFFNWeights",
    "LagunaSparseMoEResult",
    "laguna_apply_rope",
    "laguna_attention_forward",
    "laguna_causal_mask",
    "laguna_dense_ffn_forward",
    "laguna_head_rmsnorm",
    "laguna_layer_forward",
    "laguna_model_forward",
    "laguna_rope_tables",
    "laguna_sigmoid_correction_topk",
    "laguna_sigmoid_correction_topk_from_logits",
    "laguna_softplus_head_gate",
    "laguna_sparse_moe_forward",
    "register_laguna_cpu_reference_kernels",
]
