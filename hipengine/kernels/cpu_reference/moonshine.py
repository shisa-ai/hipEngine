"""Torch-free NumPy correctness oracles for Moonshine FP16 inference.

The functions in this module model explicit FP16 activation boundaries with
FP32 reductions/accumulation.  They are intentionally simple and independent
of any HIP/CUDA implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hipengine.kernels.registry import KernelKey, register

ArrayLike = Any


def _finite(name: str, value: np.ndarray) -> None:
    if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _fp16(name: str, value: ArrayLike) -> np.ndarray:
    result = np.asarray(value, dtype=np.float16)
    _finite(name, result)
    return result


def _fp32(name: str, value: ArrayLike) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    _finite(name, result)
    return result


def _linear_fp16(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
) -> np.ndarray:
    x_arr = _fp16("x", x)
    weight_arr = _fp16("weight", weight)
    if x_arr.ndim < 1 or weight_arr.ndim != 2:
        raise ValueError("x must have a final feature dimension and weight must have shape [out, in]")
    if x_arr.shape[-1] != weight_arr.shape[1]:
        raise ValueError("x final dimension must match weight input dimension")
    output = np.matmul(x_arr.astype(np.float32), weight_arr.astype(np.float32).T)
    if bias is not None:
        bias_arr = _fp16("bias", bias)
        if bias_arr.shape != (weight_arr.shape[0],):
            raise ValueError("bias must have shape [out]")
        output = (output + bias_arr.astype(np.float32)).astype(np.float32)
    result = output.astype(np.float16)
    _finite("linear output", result)
    return result


def moonshine_layernorm(
    x: ArrayLike,
    weight: ArrayLike,
    eps: float = 1.0e-5,
) -> np.ndarray:
    """Bias-free LayerNorm with FP32 mean/variance and FP16 output."""

    x_arr = _fp16("x", x)
    weight_arr = _fp16("weight", weight)
    if x_arr.ndim < 1 or weight_arr.shape != (x_arr.shape[-1],):
        raise ValueError("weight must have shape [hidden]")
    eps_value = np.float32(eps)
    if not np.isfinite(eps_value) or eps_value <= 0:
        raise ValueError("eps must be positive and finite")
    x32 = x_arr.astype(np.float32)
    mean = np.mean(x32, axis=-1, keepdims=True, dtype=np.float32)
    centered = (x32 - mean).astype(np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    output = (
        centered
        * np.reciprocal(np.sqrt((variance + eps_value).astype(np.float32)))
        * weight_arr.astype(np.float32)
    ).astype(np.float16)
    _finite("layernorm output", output)
    return output


def moonshine_rope_tables(
    max_positions: int,
    *,
    rotary_dim: int = 32,
    theta: float = 10_000.0,
    dtype: str | np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pair-frequency tables for interleaved partial RoPE."""

    if isinstance(max_positions, bool) or not isinstance(max_positions, int) or max_positions <= 0:
        raise ValueError("max_positions must be a positive integer")
    if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int) or rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    theta_value = float(theta)
    if not np.isfinite(theta_value) or theta_value <= 0:
        raise ValueError("theta must be positive and finite")
    output_dtype = np.dtype(dtype)
    if output_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
        raise ValueError("RoPE table dtype must be float16 or float32")
    dimensions = np.arange(0, rotary_dim, 2, dtype=np.float32)
    inverse_frequency = np.reciprocal(
        np.power(np.float32(theta_value), dimensions / np.float32(rotary_dim))
    ).astype(np.float32)
    positions = np.arange(max_positions, dtype=np.float32)
    angles = np.multiply.outer(positions, inverse_frequency).astype(np.float32)
    return np.cos(angles).astype(output_dtype), np.sin(angles).astype(output_dtype)


def _selected_rope_values(
    table: ArrayLike,
    *,
    position_ids: np.ndarray | None,
    batch: int,
    sequence: int,
    pairs: int,
    name: str,
) -> np.ndarray:
    values = _fp16(name, table)
    if position_ids is not None:
        if values.ndim != 2 or values.shape[1] != pairs:
            raise ValueError(f"{name} table must have shape [positions, rotary_dim / 2]")
        positions = np.asarray(position_ids, dtype=np.int64)
        if positions.ndim == 1:
            positions = np.broadcast_to(positions[None, :], (batch, sequence))
        if positions.shape != (batch, sequence):
            raise ValueError("position_ids must have shape [batch, sequence] or [sequence]")
        if positions.size and (int(positions.min()) < 0 or int(positions.max()) >= values.shape[0]):
            raise ValueError("position_ids are outside the RoPE table")
        selected = values[positions]
    elif values.ndim == 1 and values.shape == (pairs,):
        selected = np.broadcast_to(values, (batch, sequence, pairs))
    elif values.ndim == 2 and values.shape == (sequence, pairs):
        selected = np.broadcast_to(values[None, :, :], (batch, sequence, pairs))
    else:
        raise ValueError(
            f"{name} must be [pairs] or [sequence, pairs] without position_ids"
        )
    return np.repeat(selected, 2, axis=-1)[:, None, :, :].astype(np.float32)


def _interleaved_quarter_turn(value: np.ndarray) -> np.ndarray:
    pairs = value.reshape(*value.shape[:-1], value.shape[-1] // 2, 2)
    output = np.empty_like(pairs, dtype=np.float32)
    output[..., 0] = -pairs[..., 1]
    output[..., 1] = pairs[..., 0]
    return output.reshape(value.shape)


def moonshine_apply_partial_rope(
    query: ArrayLike,
    key: ArrayLike,
    cos: ArrayLike,
    sin: ArrayLike,
    *,
    position_ids: ArrayLike | None = None,
    rotary_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pair-interleaved RoPE to the leading dimensions of Q and K."""

    query_arr = _fp16("query", query)
    key_arr = _fp16("key", key)
    if query_arr.ndim != 4 or key_arr.ndim != 4:
        raise ValueError("query and key must have shape [batch, heads, sequence, head_dim]")
    if (
        query_arr.shape[0] != key_arr.shape[0]
        or query_arr.shape[2] != key_arr.shape[2]
        or query_arr.shape[3] != key_arr.shape[3]
    ):
        raise ValueError("query and key batch, sequence, and head dimensions must match")
    head_dim = query_arr.shape[-1]
    if rotary_dim is None:
        cos_arr = np.asarray(cos)
        if cos_arr.ndim == 0:
            raise ValueError("cannot infer rotary_dim from scalar cos")
        rotary_dim = int(cos_arr.shape[-1]) * 2
    if (
        isinstance(rotary_dim, bool)
        or not isinstance(rotary_dim, int)
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2
    ):
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    positions = None if position_ids is None else np.asarray(position_ids, dtype=np.int64)
    batch, _, sequence, _ = query_arr.shape
    cos_values = _selected_rope_values(
        cos,
        position_ids=positions,
        batch=batch,
        sequence=sequence,
        pairs=rotary_dim // 2,
        name="cos",
    )
    sin_values = _selected_rope_values(
        sin,
        position_ids=positions,
        batch=batch,
        sequence=sequence,
        pairs=rotary_dim // 2,
        name="sin",
    )

    def rotate(value: np.ndarray) -> np.ndarray:
        leading = value[..., :rotary_dim].astype(np.float32)
        rotated = (
            leading * cos_values + _interleaved_quarter_turn(leading) * sin_values
        ).astype(np.float16)
        return np.concatenate([rotated, value[..., rotary_dim:]], axis=-1)

    query_output = rotate(query_arr)
    key_output = rotate(key_arr)
    _finite("rotated query", query_output)
    _finite("rotated key", key_output)
    return query_output, key_output


def _validate_cache_pair(
    key_cache: np.ndarray,
    value_cache: np.ndarray,
) -> tuple[int, int, int, int]:
    if key_cache.dtype != np.float16 or value_cache.dtype != np.float16:
        raise ValueError("fixed K/V caches must use float16 storage")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("key_cache and value_cache must share [batch, heads, capacity, head_dim]")
    _finite("key_cache", key_cache)
    _finite("value_cache", value_cache)
    return key_cache.shape


def moonshine_fixed_cache_read(
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    *,
    visible_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only the initialized fixed-cache prefix, including empty prefix 0."""

    _, _, capacity, _ = _validate_cache_pair(key_cache, value_cache)
    if (
        isinstance(visible_length, bool)
        or not isinstance(visible_length, int)
        or visible_length < 0
        or visible_length > capacity
    ):
        raise ValueError("visible_length must be in 0..capacity")
    return key_cache[:, :, :visible_length, :], value_cache[:, :, :visible_length, :]


def moonshine_fixed_cache_write(
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    key: ArrayLike,
    value: ArrayLike,
    *,
    position: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Write one logical token to a fixed FP16 cache and return its visible prefix."""

    batch, heads, capacity, head_dim = _validate_cache_pair(key_cache, value_cache)
    if isinstance(position, bool) or not isinstance(position, int) or position < 0 or position >= capacity:
        raise ValueError("position must be in 0..capacity-1")
    key_arr = _fp16("key", key)
    value_arr = _fp16("value", value)
    expected = (batch, heads, 1, head_dim)
    if key_arr.ndim == 3:
        key_arr = key_arr[:, :, None, :]
    if value_arr.ndim == 3:
        value_arr = value_arr[:, :, None, :]
    if key_arr.shape != expected or value_arr.shape != expected:
        raise ValueError("key and value must have shape [batch, heads, 1, head_dim]")
    key_cache[:, :, position : position + 1, :] = key_arr
    value_cache[:, :, position : position + 1, :] = value_arr
    return moonshine_fixed_cache_read(
        key_cache,
        value_cache,
        visible_length=position + 1,
    )


def _broadcast_attention_mask(mask: ArrayLike, shape: tuple[int, int, int, int]) -> np.ndarray:
    batch, _, query_length, key_length = shape
    value = np.asarray(mask)
    if value.shape == (batch, key_length):
        value = value[:, None, None, :]
    elif value.shape == (query_length, key_length):
        value = value[None, None, :, :]
    elif value.shape == (batch, query_length, key_length):
        value = value[:, None, :, :]
    try:
        return np.broadcast_to(value, shape)
    except ValueError as error:
        raise ValueError("attention mask is not broadcastable to [batch, heads, query, key]") from error


def moonshine_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    scale: float | None = None,
    causal: bool = False,
    query_positions: ArrayLike | None = None,
) -> np.ndarray:
    """FP32 softmax attention over logical (unpadded) head dimensions."""

    query_arr = _fp16("query", query)
    key_arr = _fp16("key", key)
    value_arr = _fp16("value", value)
    if query_arr.ndim != 4 or key_arr.ndim != 4 or value_arr.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, sequence, dim]")
    batch, query_heads, query_length, head_dim = query_arr.shape
    if key_arr.shape[0] != batch or value_arr.shape[:3] != key_arr.shape[:3]:
        raise ValueError("key/value batch, head, and sequence dimensions must match")
    key_heads, key_length, key_dim = key_arr.shape[1:]
    if key_length <= 0:
        raise ValueError("key length must be positive")
    if key_dim != head_dim:
        raise ValueError("query and key head dimensions must match")
    if query_heads % key_heads:
        raise ValueError("query head count must be divisible by key/value head count")
    if query_length <= 0:
        raise ValueError("query length must be positive")
    repeats = query_heads // key_heads
    if repeats != 1:
        key_arr = np.repeat(key_arr, repeats, axis=1)
        value_arr = np.repeat(value_arr, repeats, axis=1)
    scale_value = head_dim**-0.5 if scale is None else float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("scale must be positive and finite")
    logits = (
        np.matmul(
            query_arr.astype(np.float32),
            np.swapaxes(key_arr.astype(np.float32), -1, -2),
        )
        * np.float32(scale_value)
    ).astype(np.float32)

    valid = np.ones(logits.shape, dtype=bool)
    if causal:
        if query_positions is None:
            if query_length > key_length:
                raise ValueError("causal query length cannot exceed key length")
            positions = np.arange(key_length - query_length, key_length, dtype=np.int64)
            positions = np.broadcast_to(positions[None, :], (batch, query_length))
        else:
            positions = np.asarray(query_positions, dtype=np.int64)
            if positions.ndim == 1:
                positions = np.broadcast_to(positions[None, :], (batch, query_length))
            if positions.shape != (batch, query_length):
                raise ValueError("query_positions must have shape [batch, query]")
        causal_valid = np.arange(key_length, dtype=np.int64)[None, None, None, :] <= positions[
            :, None, :, None
        ]
        valid &= np.broadcast_to(causal_valid, logits.shape)
    if mask is not None:
        broadcast_mask = _broadcast_attention_mask(mask, logits.shape)
        if np.issubdtype(broadcast_mask.dtype, np.bool_) or np.issubdtype(
            broadcast_mask.dtype, np.integer
        ):
            valid &= broadcast_mask.astype(bool)
        else:
            additive = np.asarray(broadcast_mask, dtype=np.float32)
            if np.isnan(additive).any():
                raise ValueError("additive attention mask must not contain NaN")
            logits = (logits + additive).astype(np.float32)
    logits = np.where(valid, logits, -np.inf)
    maximum = np.max(logits, axis=-1, keepdims=True)
    if not bool(np.isfinite(maximum).all()):
        raise ValueError("every attention query must have at least one finite visible key")
    exponential = np.exp((logits - maximum).astype(np.float32)).astype(np.float32)
    denominator = np.sum(exponential, axis=-1, keepdims=True, dtype=np.float32)
    probabilities = (exponential / denominator).astype(np.float32)
    output = np.matmul(probabilities, value_arr.astype(np.float32)).astype(np.float16)
    _finite("attention output", output)
    return output


def moonshine_projection(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
) -> np.ndarray:
    """One FP16 projection with FP32 accumulation and an FP16 output boundary."""

    return _linear_fp16(x, weight, bias)


def moonshine_triple_projection(
    x: ArrayLike,
    q_weight: ArrayLike,
    k_weight: ArrayLike,
    v_weight: ArrayLike,
    q_bias: ArrayLike | None = None,
    k_bias: ArrayLike | None = None,
    v_bias: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent Q/K/V projections sharing one logical FP16 input."""

    return (
        _linear_fp16(x, q_weight, q_bias),
        _linear_fp16(x, k_weight, k_bias),
        _linear_fp16(x, v_weight, v_bias),
    )


def moonshine_decoder_mlp(
    x: ArrayLike,
    fc1_weight: ArrayLike,
    fc1_bias: ArrayLike,
    fc2_weight: ArrayLike,
    fc2_bias: ArrayLike,
) -> np.ndarray:
    """Moonshine decoder ``fc1 -> split(value, gate) -> SiLU(gate)*value -> fc2``."""

    x_arr = _fp16("x", x)
    fc1 = _linear_fp16(x_arr, fc1_weight, fc1_bias)
    if fc1.shape[-1] <= 0 or fc1.shape[-1] % 2:
        raise ValueError("fc1 output width must be twice the decoder intermediate size")
    value, gate = np.split(fc1, 2, axis=-1)
    gate32 = gate.astype(np.float32)
    silu = (gate32 / (np.float32(1.0) + np.exp(-gate32).astype(np.float32))).astype(
        np.float32
    )
    activated = (value.astype(np.float32) * silu).astype(np.float16)
    output = _linear_fp16(activated, fc2_weight, fc2_bias)
    if output.shape != x_arr.shape:
        raise ValueError("fc2 output shape must match the decoder hidden input")
    return output


def moonshine_residual(hidden: ArrayLike, residual: ArrayLike) -> np.ndarray:
    """Add two hidden states and round at the FP16 residual boundary."""

    hidden_arr = _fp16("hidden", hidden)
    residual_arr = _fp16("residual", residual)
    if hidden_arr.shape != residual_arr.shape:
        raise ValueError("hidden and residual shapes must match")
    output = np.add(hidden_arr, residual_arr, dtype=np.float16)
    _finite("residual output", output)
    return output


def moonshine_tied_lm_logits(
    hidden: ArrayLike,
    tied_embedding_weight: ArrayLike,
) -> np.ndarray:
    """Project through the single embedding-owned ``[vocab, hidden]`` FP16 matrix."""

    return _linear_fp16(hidden, tied_embedding_weight)


def moonshine_stable_argmax(logits: ArrayLike, *, axis: int = -1) -> np.ndarray:
    """Return the lowest index on ties and reject non-finite logits."""

    values = np.asarray(logits)
    if values.ndim == 0 or values.shape[axis] <= 0:
        raise ValueError("logits must have a non-empty argmax axis")
    _finite("logits", values)
    return np.asarray(np.argmax(values, axis=axis), dtype=np.int64)


def moonshine_lm_head_argmax(
    hidden: ArrayLike,
    tied_embedding_weight: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    logits = moonshine_tied_lm_logits(hidden, tied_embedding_weight)
    return logits, moonshine_stable_argmax(logits)


def register_moonshine_cpu_reference_kernels(*, replace: bool = True) -> None:
    """Register Moonshine primitive oracles under explicit four-axis keys."""

    kernels = {
        ("moonshine_layernorm", "fp32_stats"): moonshine_layernorm,
        ("moonshine_partial_rope", "interleaved"): moonshine_apply_partial_rope,
        ("moonshine_self_cache", "fixed"): moonshine_fixed_cache_write,
        ("moonshine_attention", "logical_head_dim"): moonshine_attention,
        ("moonshine_projection", "fp32_accum"): moonshine_projection,
        ("moonshine_qkv_proj", "triple"): moonshine_triple_projection,
        ("moonshine_decoder_mlp", "gated_silu"): moonshine_decoder_mlp,
        ("moonshine_residual", "rounded"): moonshine_residual,
        ("moonshine_lm_head", "tied"): moonshine_tied_lm_logits,
        ("moonshine_argmax", "lowest_id"): moonshine_stable_argmax,
    }
    for (layer, variant), kernel in kernels.items():
        register(
            KernelKey("cpu_reference", layer, "fp16", variant),
            kernel,
            replace=replace,
        )
