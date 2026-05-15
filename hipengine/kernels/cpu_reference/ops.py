"""Torch-free NumPy CPU-reference kernels.

These functions are small correctness oracles for the first registry and fixture tests. They
are intentionally plain NumPy, not optimized, and not a substitute for HIP kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hipengine.kernels.registry import KernelKey, register

ArrayLike = Any


def embed(token_ids: ArrayLike, table: ArrayLike) -> np.ndarray:
    token_ids_arr = np.asarray(token_ids, dtype=np.int64)
    table_arr = np.asarray(table)
    return table_arr[token_ids_arr]


def rmsnorm(x: ArrayLike, weight: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    weight_arr = np.asarray(weight, dtype=np.float32)
    variance = np.mean(x_arr * x_arr, axis=-1, keepdims=True)
    return (x_arr * np.reciprocal(np.sqrt(variance + eps))) * weight_arr


def linear(x: ArrayLike, weight: ArrayLike, bias: ArrayLike | None = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    weight_arr = np.asarray(weight, dtype=np.float32)
    out = np.matmul(x_arr, np.swapaxes(weight_arr, -1, -2))
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float32)
    return out


def qkv_proj(x: ArrayLike, weight: ArrayLike, bias: ArrayLike | None = None) -> np.ndarray:
    return linear(x, weight, bias)


def o_proj(x: ArrayLike, weight: ArrayLike, bias: ArrayLike | None = None) -> np.ndarray:
    return linear(x, weight, bias)


def lm_head(x: ArrayLike, weight: ArrayLike, bias: ArrayLike | None = None) -> np.ndarray:
    return linear(x, weight, bias)


def rotate(
    x: ArrayLike,
    cos: ArrayLike,
    sin: ArrayLike,
    rotary_dim: int | None = None,
) -> np.ndarray:
    """Apply split-half rotary embedding to the last dimension of ``x``.

    ``cos`` and ``sin`` may have the half-rotary dimension or the full rotary dimension as
    their last axis. The implementation follows the common split-half form:
    ``[x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]``.
    """

    x_arr = np.asarray(x, dtype=np.float32)
    dim = x_arr.shape[-1]
    rotary_dim = dim if rotary_dim is None else int(rotary_dim)
    if rotary_dim < 0 or rotary_dim > dim:
        raise ValueError("rotary_dim must be between 0 and x.shape[-1]")
    if rotary_dim % 2:
        raise ValueError("rotary_dim must be even")
    if rotary_dim == 0:
        return x_arr.copy()

    half = rotary_dim // 2
    x_rot = x_arr[..., :rotary_dim]
    x_pass = x_arr[..., rotary_dim:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]

    cos_arr = _half_rotary_table(cos, half, "cos")
    sin_arr = _half_rotary_table(sin, half, "sin")
    rotated = np.concatenate((x1 * cos_arr - x2 * sin_arr, x1 * sin_arr + x2 * cos_arr), axis=-1)
    if x_pass.shape[-1] == 0:
        return rotated
    return np.concatenate((rotated, x_pass), axis=-1)


def attention_decode(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    mask: ArrayLike | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Reference scaled dot-product attention for decode-shaped inputs."""

    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    scale = (q.shape[-1] ** -0.5) if scale is None else float(scale)
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        logits = np.where(mask_arr, logits, -np.inf)
    weights = _softmax(logits, axis=-1)
    return np.matmul(weights, v)


def linear_attn_conv_prefill_segments(
    hidden_states: ArrayLike,
    conv_state: ArrayLike,
    conv_weight: ArrayLike,
    cu_seqlens: ArrayLike,
    state_indices: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment-aware linear-attention convolution prefill reference.

    ``hidden_states`` is packed ``[T_total, channels]``. ``conv_state`` is a
    mutable-state snapshot shaped ``[state_slots, channels, kernel_size]`` and
    ``state_indices[segment]`` selects the slot committed by each segment.
    Short segments are valid: their tail state is the old shifted state plus the
    segment rows, without reading rows from neighboring segments.
    """

    hidden = np.asarray(hidden_states, dtype=np.float32)
    state = np.asarray(conv_state, dtype=np.float32).copy()
    weight = np.asarray(conv_weight, dtype=np.float32)
    cu = np.asarray(cu_seqlens, dtype=np.int64)
    slots = np.asarray(state_indices, dtype=np.int64)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must have shape [T_total, channels]")
    if state.ndim != 3:
        raise ValueError("conv_state must have shape [state_slots, channels, kernel_size]")
    if weight.shape != state.shape[1:]:
        raise ValueError("conv_weight must have shape [channels, kernel_size]")
    _validate_segments(cu, slots, hidden.shape[0], state.shape[0])
    channels = hidden.shape[1]
    kernel_size = state.shape[2]
    out = np.empty_like(hidden, dtype=np.float32)
    for segment, slot in enumerate(slots):
        start = int(cu[segment])
        end = int(cu[segment + 1])
        tokens = end - start
        for local_token, row in enumerate(range(start, end)):
            for channel in range(channels):
                acc = np.float32(0.0)
                for k in range(kernel_size):
                    padded = local_token + k
                    if padded < kernel_size - 1:
                        value = state[slot, channel, padded + 1]
                    else:
                        value = hidden[start + padded - (kernel_size - 1), channel]
                    acc = np.float32(acc + np.float32(value * weight[channel, k]))
                out[row, channel] = _silu(acc)
        if tokens >= kernel_size:
            state[slot] = hidden[end - kernel_size : end].T
        else:
            kept = kernel_size - tokens
            state[slot, :, :kept] = state[slot, :, tokens:]
            state[slot, :, kept:] = hidden[start:end].T
    return out, state


def gdn_prefill_recurrent_segments(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    beta: ArrayLike,
    decay: ArrayLike,
    recurrent_state: ArrayLike,
    cu_seqlens: ArrayLike,
    state_indices: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment-aware GDN recurrent prefill reference over packed prompt rows."""

    q = np.asarray(query, dtype=np.float32)
    k_arr = np.asarray(key, dtype=np.float32)
    v_arr = np.asarray(value, dtype=np.float32)
    beta_arr = np.asarray(beta, dtype=np.float32)
    decay_arr = np.asarray(decay, dtype=np.float32)
    state = np.asarray(recurrent_state, dtype=np.float32).copy()
    cu = np.asarray(cu_seqlens, dtype=np.int64)
    slots = np.asarray(state_indices, dtype=np.int64)
    if q.ndim != 3:
        raise ValueError("query must have shape [T_total, num_v_heads, head_k_dim]")
    if k_arr.shape != q.shape:
        raise ValueError("key must match query shape")
    if v_arr.ndim != 3 or v_arr.shape[:2] != q.shape[:2]:
        raise ValueError("value must have shape [T_total, num_v_heads, head_v_dim]")
    if beta_arr.shape != q.shape[:2] or decay_arr.shape != q.shape[:2]:
        raise ValueError("beta and decay must have shape [T_total, num_v_heads]")
    if state.ndim != 4 or state.shape[1:] != (q.shape[1], q.shape[2], v_arr.shape[2]):
        raise ValueError("recurrent_state must have shape [state_slots, num_v_heads, head_k_dim, head_v_dim]")
    _validate_segments(cu, slots, q.shape[0], state.shape[0])
    out = np.empty_like(v_arr, dtype=np.float32)
    for segment, slot in enumerate(slots):
        start = int(cu[segment])
        end = int(cu[segment + 1])
        for row in range(start, end):
            for v_head in range(q.shape[1]):
                for value_idx in range(v_arr.shape[2]):
                    state_vec = state[slot, v_head, :, value_idx]
                    state_vec = np.asarray(state_vec * decay_arr[row, v_head], dtype=np.float32)
                    kv_mem = np.sum(k_arr[row, v_head] * state_vec, dtype=np.float32)
                    delta = np.float32((v_arr[row, v_head, value_idx] - kv_mem) * beta_arr[row, v_head])
                    state_vec = np.asarray(state_vec + k_arr[row, v_head] * delta, dtype=np.float32)
                    state[slot, v_head, :, value_idx] = state_vec
                    out[row, v_head, value_idx] = np.sum(q[row, v_head] * state_vec, dtype=np.float32)
    return out, state


def full_attn_prefill(
    query: ArrayLike,
    gate: ArrayLike,
    key_cache: ArrayLike,
    value_cache: ArrayLike,
    positions: ArrayLike,
    *,
    context_counts: ArrayLike | None = None,
    block_table: ArrayLike | None = None,
    block_size: int | None = None,
    scale: float | None = None,
    output_dtype: str | np.dtype | type | None = np.float16,
) -> np.ndarray:
    """Reference append-then-attend causal GQA prefill with sigmoid gate.

    ``key_cache`` and ``value_cache`` may be dense ``[S, Hkv, D]`` arrays or
    paged ``[B, block, Hkv, D]`` arrays. Paged caches may be BF16-bit ``uint16``;
    other dtypes are interpreted numerically as floats. ``positions`` are the
    absolute cache positions for the T query rows and ``context_counts`` are the
    1-based visible lengths for each row.
    """

    q = np.asarray(query, dtype=np.float32)
    g = np.asarray(gate, dtype=np.float32)
    positions_arr = np.asarray(positions, dtype=np.int64)
    if q.ndim != 3:
        raise ValueError("query must have shape [T, num_q_heads, head_dim]")
    if g.shape != q.shape:
        raise ValueError("gate must match query shape")
    if positions_arr.shape != (q.shape[0],):
        raise ValueError("positions must have shape [T]")
    contexts = positions_arr + 1 if context_counts is None else np.asarray(context_counts, dtype=np.int64)
    if contexts.shape != (q.shape[0],):
        raise ValueError("context_counts must have shape [T]")

    key = _cache_to_float(key_cache)
    value = _cache_to_float(value_cache)
    if key.shape != value.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if key.ndim == 3:
        dense_cache = True
        inferred_block = key.shape[0]
        num_kv_heads = key.shape[1]
        head_dim = key.shape[2]
    elif key.ndim == 4:
        dense_cache = False
        inferred_block = key.shape[1]
        num_kv_heads = key.shape[2]
        head_dim = key.shape[3]
    else:
        raise ValueError("key_cache must have shape [S, Hkv, D] or [B, block, Hkv, D]")
    if q.shape[2] != head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    num_q_heads = q.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    block = inferred_block if block_size is None else int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    table = None if block_table is None else np.asarray(block_table, dtype=np.int64).reshape(-1)
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)
    out = np.empty_like(q, dtype=np.float32)
    kv_group = num_q_heads // num_kv_heads
    for row in range(q.shape[0]):
        position = int(positions_arr[row])
        context = int(contexts[row])
        if position < 0:
            raise ValueError("positions must be non-negative")
        if context <= 0:
            raise ValueError("context_counts must be positive")
        visible_positions = [cache_pos for cache_pos in range(context) if cache_pos <= position]
        if not visible_positions:
            raise ValueError("causal mask left no visible cache positions")
        for q_head in range(num_q_heads):
            kv_head = q_head // kv_group
            keys = np.stack(
                [
                    _cache_row(
                        key,
                        cache_pos,
                        kv_head,
                        dense_cache=dense_cache,
                        block_size=block,
                        block_table=table,
                    )
                    for cache_pos in visible_positions
                ],
                axis=0,
            )
            values = np.stack(
                [
                    _cache_row(
                        value,
                        cache_pos,
                        kv_head,
                        dense_cache=dense_cache,
                        block_size=block,
                        block_table=table,
                    )
                    for cache_pos in visible_positions
                ],
                axis=0,
            )
            logits = np.matmul(keys, q[row, q_head]) * scale_value
            weights = _softmax(logits, axis=0)
            attn = np.matmul(weights, values)
            out[row, q_head] = attn * _sigmoid(g[row, q_head])
    if output_dtype is None:
        return out
    return out.astype(np.dtype(output_dtype))


def register_cpu_reference_kernels(*, replace: bool = True) -> None:
    """Register the first CPU-reference primitive set under fp16 keys."""

    kernels = {
        "embed": embed,
        "rmsnorm": rmsnorm,
        "linear": linear,
        "qkv_proj": qkv_proj,
        "rotate": rotate,
        "attention_decode": attention_decode,
        "full_attn_prefill": full_attn_prefill,
        "linear_attn_conv_prefill_segments": linear_attn_conv_prefill_segments,
        "gdn_prefill_recurrent_segments": gdn_prefill_recurrent_segments,
        "o_proj": o_proj,
        "lm_head": lm_head,
    }
    for layer, fn in kernels.items():
        register(KernelKey("cpu_reference", layer, "fp16"), fn, replace=replace)
    register(
        KernelKey("cpu_reference", "full_attn_prefill", "w4_paro", "qwen35_causal_gqa_gate_fp16"),
        full_attn_prefill,
        replace=replace,
    )


def _validate_segments(cu: np.ndarray, slots: np.ndarray, total_rows: int, state_slots: int) -> None:
    if cu.ndim != 1:
        raise ValueError("cu_seqlens must be 1D")
    if slots.ndim != 1:
        raise ValueError("state_indices must be 1D")
    if cu.shape[0] != slots.shape[0] + 1:
        raise ValueError("cu_seqlens length must be len(state_indices) + 1")
    if cu.shape[0] <= 1:
        raise ValueError("at least one segment is required")
    if int(cu[0]) != 0 or int(cu[-1]) != int(total_rows):
        raise ValueError("cu_seqlens must span all rows")
    if np.any(cu[1:] <= cu[:-1]):
        raise ValueError("cu_seqlens segments must be non-empty and increasing")
    if np.any(slots < 0) or np.any(slots >= state_slots):
        raise ValueError("state_indices reference state slot outside state")


def _half_rotary_table(value: ArrayLike, half: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] == half:
        return arr
    if arr.shape[-1] == half * 2:
        return arr[..., :half]
    raise ValueError(f"{name}.shape[-1] must be {half} or {half * 2}, got {arr.shape[-1]}")


def _cache_to_float(value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.uint16:
        return (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)


def _cache_row(
    cache: np.ndarray,
    position: int,
    kv_head: int,
    *,
    dense_cache: bool,
    block_size: int,
    block_table: np.ndarray | None,
) -> np.ndarray:
    if dense_cache:
        if position >= cache.shape[0]:
            raise ValueError("cache position exceeds dense cache length")
        return cache[position, kv_head]
    logical_block = position // block_size
    block_offset = position % block_size
    physical_block = logical_block if block_table is None else int(block_table[logical_block])
    if physical_block < 0 or physical_block >= cache.shape[0]:
        raise ValueError("block_table references cache block outside key/value cache")
    return cache[physical_block, block_offset, kv_head]


def _silu(x: np.ndarray | np.float32 | float) -> np.ndarray | np.float32:
    x_arr = np.asarray(x, dtype=np.float32)
    return x_arr / (np.float32(1.0) + np.exp(-x_arr).astype(np.float32))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x_arr))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    finite = np.isfinite(x)
    safe_x = np.where(finite, x, -np.inf)
    max_x = np.max(safe_x, axis=axis, keepdims=True)
    shifted = safe_x - max_x
    exp = np.where(finite, np.exp(shifted), 0.0)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom
