"""Torch-free NumPy CPU-reference kernels.

These functions are small correctness oracles for the first registry and fixture tests. They
are intentionally plain NumPy, not optimized, and not a substitute for HIP kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hipengine.kernels.registry import KernelKey, register
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_q4_k import GGUF_Q4_K_PACK, awq_pack8_shift_for_lane

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


def gguf_quant_gemv(
    x: ArrayLike,
    qweight: ArrayLike,
    qtype: GGMLQuantizationType,
) -> np.ndarray:
    """Reference GEMV over raw GGUF quantized weight bytes."""

    x_arr = np.asarray(x, dtype=np.float32)
    qweight_arr = np.asarray(qweight)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [rows, in_features]")
    if qweight_arr.ndim != 2:
        raise ValueError("qweight must have GGUF byte shape [out_features, bytes_per_row]")
    weight = dequantize_gguf_data(qweight_arr, qtype)
    if weight.ndim != 2:
        raise ValueError("qweight must dequantize to [out_features, in_features]")
    if x_arr.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match qweight in_features")
    return np.matmul(x_arr, weight.T).astype(np.float32)


def gguf_q8_0_gemv(x: ArrayLike, qweight: ArrayLike) -> np.ndarray:
    """Reference GEMV over raw GGUF ``block_q8_0`` weight bytes."""

    return gguf_quant_gemv(x, qweight, GGMLQuantizationType.Q8_0)


def gguf_q4_k_gemv(x: ArrayLike, qweight: ArrayLike) -> np.ndarray:
    """Reference GEMV over raw GGUF ``block_q4_K`` weight bytes."""

    return gguf_quant_gemv(x, qweight, GGMLQuantizationType.Q4_K)


def gguf_q5_k_gemv(x: ArrayLike, qweight: ArrayLike) -> np.ndarray:
    """Reference GEMV over raw GGUF ``block_q5_K`` weight bytes."""

    return gguf_quant_gemv(x, qweight, GGMLQuantizationType.Q5_K)


def gguf_q6_k_gemv(x: ArrayLike, qweight: ArrayLike) -> np.ndarray:
    """Reference GEMV over raw GGUF ``block_q6_K`` weight bytes."""

    return gguf_quant_gemv(x, qweight, GGMLQuantizationType.Q6_K)


def gguf_q6_k_embedding(token_ids: ArrayLike, qweight: ArrayLike) -> np.ndarray:
    """Reference embedding lookup over raw GGUF ``block_q6_K`` rows."""

    token_arr = np.asarray(token_ids, dtype=np.int64)
    if token_arr.ndim != 1:
        raise ValueError("token_ids must have shape [rows]")
    qweight_arr = np.asarray(qweight)
    if qweight_arr.ndim != 2:
        raise ValueError("qweight must have GGUF byte shape [vocab_size, bytes_per_row]")
    if np.any(token_arr < 0) or np.any(token_arr >= qweight_arr.shape[0]):
        raise ValueError("token_ids contain out-of-range token IDs")
    return dequantize_gguf_data(qweight_arr[token_arr], GGMLQuantizationType.Q6_K).astype(np.float32)


def gguf_q4_k_pack8_gemv(
    x: ArrayLike,
    qweight: ArrayLike,
    scales: ArrayLike,
    mins: ArrayLike,
) -> np.ndarray:
    """Reference GEMV over the lossless GGUF Q4_K pack8 layout."""

    x_arr = np.asarray(x, dtype=np.float32)
    qweight_arr = np.asarray(qweight).view(np.uint32)
    scales_arr = np.asarray(scales, dtype=np.float32)
    mins_arr = np.asarray(mins, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [rows, in_features]")
    if qweight_arr.ndim != 2:
        raise ValueError("qweight must have shape [out_features / 8, in_features]")
    if scales_arr.shape != mins_arr.shape:
        raise ValueError("scales and mins must have the same shape")
    out_packed, in_features = qweight_arr.shape
    out_features = out_packed * GGUF_Q4_K_PACK
    if x_arr.shape[1] != in_features:
        raise ValueError("x.shape[1] must match qweight in_features")
    if scales_arr.shape != (in_features // 32, out_features):
        raise ValueError("scales/mins must have shape [in_features / 32, out_features]")

    q_values = np.empty((out_features, in_features), dtype=np.float32)
    for lane in range(GGUF_Q4_K_PACK):
        out_cols = np.arange(out_packed) * GGUF_Q4_K_PACK + lane
        q_values[out_cols] = (
            (qweight_arr >> np.uint32(awq_pack8_shift_for_lane(lane))) & np.uint32(0x0F)
        ).astype(np.float32)
    group_for_k = np.arange(in_features, dtype=np.int64) // 32
    weight = q_values * scales_arr[group_for_k].T - mins_arr[group_for_k].T
    return np.matmul(x_arr, weight.T).astype(np.float32)


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


def quantize_kv_int8_per_token_head(
    key: ArrayLike,
    value: ArrayLike,
    *,
    scale_dtype: str | np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize K/V rows with separate per-token/per-KV-head INT8 scales.

    The last dimension is ``head_dim``.  Every leading token/page/head location
    gets one K scale and one V scale.  All-zero rows use scale ``0`` and store
    all-zero INT8 payload so dequantization is well-defined and never divides by
    zero.
    """

    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    if k.shape != v.shape:
        raise ValueError("key and value must have the same shape")
    if k.ndim not in {3, 4}:
        raise ValueError("key/value must have shape [tokens, Hkv, D] or [blocks, block, Hkv, D]")
    qk, ks = _quantize_int8_rows(k, scale_dtype)
    qv, vs = _quantize_int8_rows(v, scale_dtype)
    return qk, qv, ks, vs


def write_paged_kv_int8_per_token_head(
    key: ArrayLike,
    value: ArrayLike,
    positions: ArrayLike,
    block_table: ArrayLike,
    *,
    block_size: int,
    cache_blocks: int | None = None,
    scale_dtype: str | np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference paged INT8 K/V append for one request.

    ``key`` and ``value`` are row-major ``[rows, Hkv, D]`` post-RoPE rows.
    ``positions`` are logical token positions.  ``block_table`` maps logical
    blocks to physical cache blocks; this catches page-boundary and indirection
    mistakes independently from GPU kernels.
    """

    k_rows = np.asarray(key, dtype=np.float32)
    v_rows = np.asarray(value, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    table = np.asarray(block_table, dtype=np.int64).reshape(-1)
    if k_rows.shape != v_rows.shape:
        raise ValueError("key and value must have the same shape")
    if k_rows.ndim != 3:
        raise ValueError("key/value rows must have shape [rows, Hkv, D]")
    if pos.shape != (k_rows.shape[0],):
        raise ValueError("positions must have shape [rows]")
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    if table.size == 0:
        raise ValueError("block_table must not be empty")
    if np.any(table < 0):
        raise ValueError("block_table must not contain negative physical blocks")
    inferred_blocks = int(np.max(table)) + 1
    blocks = inferred_blocks if cache_blocks is None else int(cache_blocks)
    if blocks < inferred_blocks or blocks <= 0:
        raise ValueError("cache_blocks must cover the block_table physical blocks")

    qk, qv, ks, vs = quantize_kv_int8_per_token_head(k_rows, v_rows, scale_dtype=scale_dtype)
    key_cache = np.zeros((blocks, block, k_rows.shape[1], k_rows.shape[2]), dtype=np.int8)
    value_cache = np.zeros_like(key_cache)
    k_scale = np.zeros((blocks, block, k_rows.shape[1]), dtype=np.dtype(scale_dtype))
    v_scale = np.zeros_like(k_scale)
    for row, position in enumerate(pos):
        if position < 0:
            raise ValueError("positions must be non-negative")
        logical_block = int(position) // block
        block_offset = int(position) % block
        if logical_block >= table.size:
            raise ValueError("position exceeds block_table length")
        physical_block = int(table[logical_block])
        key_cache[physical_block, block_offset] = qk[row]
        value_cache[physical_block, block_offset] = qv[row]
        k_scale[physical_block, block_offset] = ks[row]
        v_scale[physical_block, block_offset] = vs[row]
    return key_cache, value_cache, k_scale, v_scale


def dequantize_kv_int8_per_token_head(
    key_cache: ArrayLike,
    value_cache: ArrayLike,
    k_scale: ArrayLike,
    v_scale: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Dequantize INT8 K/V cache using per-token/per-head K and V scales."""

    kq = np.asarray(key_cache, dtype=np.int8)
    vq = np.asarray(value_cache, dtype=np.int8)
    ks = np.asarray(k_scale, dtype=np.float32)
    vs = np.asarray(v_scale, dtype=np.float32)
    _validate_int8_kv_cache_shapes(kq, vq, ks, vs)
    return kq.astype(np.float32) * ks[..., None], vq.astype(np.float32) * vs[..., None]


def kv_dequant_int8_per_token_head(
    key_cache: ArrayLike,
    value_cache: ArrayLike,
    k_scale: ArrayLike,
    v_scale: ArrayLike,
) -> np.ndarray:
    """Fixture-friendly K/V dequantization; returns ``stack([K, V])``."""

    key, value = dequantize_kv_int8_per_token_head(key_cache, value_cache, k_scale, v_scale)
    return np.stack((key, value), axis=0)


def paged_attn_decode_int8_per_token_head(
    query: ArrayLike,
    key_cache: ArrayLike,
    value_cache: ArrayLike,
    k_scale: ArrayLike,
    v_scale: ArrayLike,
    live_counts: ArrayLike,
    *,
    block_table: ArrayLike | None = None,
    block_size: int | None = None,
    scale: float | None = None,
    output_dtype: str | np.dtype | type | None = np.float32,
) -> np.ndarray:
    """Reference paged GQA decode over INT8 K/V plus per-token/head scales."""

    key, value = dequantize_kv_int8_per_token_head(key_cache, value_cache, k_scale, v_scale)
    q = np.asarray(query, dtype=np.float32)
    squeeze_row = False
    if q.ndim == 2:
        q = q[None, ...]
        squeeze_row = True
    if q.ndim != 3:
        raise ValueError("query must have shape [Q, D] or [rows, Q, D]")
    counts = np.asarray(live_counts, dtype=np.int64).reshape(-1)
    if counts.shape != (q.shape[0],):
        raise ValueError("live_counts must have one entry per query row")
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
    kv_group = num_q_heads // num_kv_heads
    block = inferred_block if block_size is None else int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    tables = _normalize_block_tables(block_table, rows=q.shape[0])
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)
    out = np.empty_like(q, dtype=np.float32)
    for row in range(q.shape[0]):
        context = int(counts[row])
        if context <= 0:
            raise ValueError("live_counts must be positive")
        row_table = None if tables is None else tables[row]
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
                        block_table=row_table,
                    )
                    for cache_pos in range(context)
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
                        block_table=row_table,
                    )
                    for cache_pos in range(context)
                ],
                axis=0,
            )
            weights = _softmax(np.matmul(keys, q[row, q_head]) * scale_value, axis=0)
            out[row, q_head] = np.matmul(weights, values)
    if squeeze_row:
        out = out[0]
    if output_dtype is None:
        return out
    return out.astype(np.dtype(output_dtype))


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

    q = _round_to_bf16_float(np.asarray(query, dtype=np.float32))
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
            attn = _round_to_bf16_float(np.matmul(weights, values))
            out[row, q_head] = attn * _sigmoid(g[row, q_head])
    if output_dtype is None:
        return out
    return out.astype(np.dtype(output_dtype))


def full_attn_prefill_varlen(
    query: ArrayLike,
    gate: ArrayLike,
    key_cache: ArrayLike,
    value_cache: ArrayLike,
    positions: ArrayLike,
    cu_seqlens_q: ArrayLike,
    cu_seqlens_k: ArrayLike,
    *,
    context_counts: ArrayLike,
    block_tables: ArrayLike,
    block_size: int,
    scale: float | None = None,
    output_dtype: str | np.dtype | type | None = np.float16,
) -> np.ndarray:
    """Reference varlen/block-diagonal append-then-attend prefill.

    The cache remains paged; `block_tables[row]` selects the request-owned KV
    blocks for each packed query row. `cu_seqlens_q/k` define packed request
    segments and clamp each row so it cannot attend beyond the segment's K end.
    """

    q = _round_to_bf16_float(np.asarray(query, dtype=np.float32))
    g = np.asarray(gate, dtype=np.float32)
    positions_arr = np.asarray(positions, dtype=np.int64)
    contexts = np.asarray(context_counts, dtype=np.int64)
    tables = np.asarray(block_tables, dtype=np.int64)
    cu_q = np.asarray(cu_seqlens_q, dtype=np.int64)
    cu_k = np.asarray(cu_seqlens_k, dtype=np.int64)
    if q.ndim != 3:
        raise ValueError("query must have shape [T, num_q_heads, head_dim]")
    if g.shape != q.shape:
        raise ValueError("gate must match query shape")
    if positions_arr.shape != (q.shape[0],) or contexts.shape != (q.shape[0],):
        raise ValueError("positions and context_counts must have shape [T]")
    if tables.ndim != 2 or tables.shape[0] != q.shape[0]:
        raise ValueError("block_tables must have shape [T, block_table_len]")
    dummy_slots = np.zeros((cu_q.shape[0] - 1,), dtype=np.int64)
    _validate_segments(cu_q, dummy_slots, q.shape[0], 1)
    _validate_segments(cu_k, dummy_slots, q.shape[0], 1)

    key = _cache_to_float(key_cache)
    value = _cache_to_float(value_cache)
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("key_cache and value_cache must have shape [B, block, Hkv, D]")
    if key.shape[1] != block_size:
        raise ValueError("block_size must match cache shape")
    num_q_heads = q.shape[1]
    num_kv_heads = key.shape[2]
    head_dim = key.shape[3]
    if q.shape[2] != head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    kv_group = num_q_heads // num_kv_heads
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)
    out = np.empty_like(q, dtype=np.float32)
    for segment in range(cu_q.shape[0] - 1):
        q_start = int(cu_q[segment])
        q_end = int(cu_q[segment + 1])
        k_len = int(cu_k[segment + 1] - cu_k[segment])
        segment_position_start = int(positions_arr[q_start])
        segment_visible_limit = segment_position_start + k_len
        for row in range(q_start, q_end):
            visible_len = min(int(contexts[row]), int(positions_arr[row]) + 1, segment_visible_limit)
            if visible_len <= 0:
                raise ValueError("causal mask left no visible cache positions")
            for q_head in range(num_q_heads):
                kv_head = q_head // kv_group
                keys = np.stack(
                    [
                        _cache_row(
                            key,
                            cache_pos,
                            kv_head,
                            dense_cache=False,
                            block_size=block_size,
                            block_table=tables[row],
                        )
                        for cache_pos in range(visible_len)
                    ],
                    axis=0,
                )
                values = np.stack(
                    [
                        _cache_row(
                            value,
                            cache_pos,
                            kv_head,
                            dense_cache=False,
                            block_size=block_size,
                            block_table=tables[row],
                        )
                        for cache_pos in range(visible_len)
                    ],
                    axis=0,
                )
                logits = np.matmul(keys, q[row, q_head]) * scale_value
                weights = _softmax(logits, axis=0)
                attn = _round_to_bf16_float(np.matmul(weights, values))
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
        "gguf_q8_0_gemv": gguf_q8_0_gemv,
        "gguf_q4_k_gemv": gguf_q4_k_gemv,
        "gguf_q5_k_gemv": gguf_q5_k_gemv,
        "gguf_q6_k_gemv": gguf_q6_k_gemv,
        "gguf_q4_k_pack8_gemv": gguf_q4_k_pack8_gemv,
        "rotate": rotate,
        "attention_decode": attention_decode,
        "kv_dequant": kv_dequant_int8_per_token_head,
        "paged_attn_decode": paged_attn_decode_int8_per_token_head,
        "full_attn_prefill": full_attn_prefill,
        "full_attn_prefill_varlen": full_attn_prefill_varlen,
        "linear_attn_conv_prefill_segments": linear_attn_conv_prefill_segments,
        "gdn_prefill_recurrent_segments": gdn_prefill_recurrent_segments,
        "o_proj": o_proj,
        "lm_head": lm_head,
    }
    for layer, fn in kernels.items():
        quant = "int8_per_token_head" if layer in {"kv_dequant", "paged_attn_decode"} else "fp16"
        register(KernelKey("cpu_reference", layer, quant), fn, replace=replace)
    register(
        KernelKey("cpu_reference", "full_attn_prefill", "w4_paro", "qwen35_causal_gqa_gate_fp16"),
        full_attn_prefill,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "linear", "gguf_q8_0", "gemv_f32_f32_out"),
        gguf_q8_0_gemv,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "linear", "gguf_q4_k", "gemv_f32_f32_out"),
        gguf_q4_k_gemv,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "linear", "gguf_q5_k", "gemv_f32_f32_out"),
        gguf_q5_k_gemv,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "linear", "gguf_q6_k", "gemv_f32_f32_out"),
        gguf_q6_k_gemv,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "embedding", "gguf_q6_k", "lookup_f32_out"),
        gguf_q6_k_embedding,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "linear", "gguf_q4_k", "pack8_f32_f32_out"),
        gguf_q4_k_pack8_gemv,
        replace=replace,
    )


def _quantize_int8_rows(value: np.ndarray, scale_dtype: str | np.dtype | type) -> tuple[np.ndarray, np.ndarray]:
    scale_np_dtype = np.dtype(scale_dtype)
    if scale_np_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
        raise ValueError("scale_dtype must be float16 or float32")
    max_abs = np.max(np.abs(value), axis=-1)
    scale = max_abs / np.float32(127.0)
    safe_scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    quantized = np.rint(value / safe_scale[..., None])
    quantized = np.clip(quantized, -127.0, 127.0).astype(np.int8)
    quantized = np.where(scale[..., None] > 0.0, quantized, 0).astype(np.int8)
    return quantized, scale.astype(scale_np_dtype)


def _validate_int8_kv_cache_shapes(kq: np.ndarray, vq: np.ndarray, ks: np.ndarray, vs: np.ndarray) -> None:
    if kq.shape != vq.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if kq.ndim not in {3, 4}:
        raise ValueError("key_cache/value_cache must have shape [S, Hkv, D] or [B, block, Hkv, D]")
    expected_scale_shape = kq.shape[:-1]
    if ks.shape != expected_scale_shape or vs.shape != expected_scale_shape:
        raise ValueError("k_scale and v_scale must match key/value shape without head_dim")


def _normalize_block_tables(block_table: ArrayLike | None, *, rows: int) -> np.ndarray | None:
    if block_table is None:
        return None
    table = np.asarray(block_table, dtype=np.int64)
    if table.ndim == 1:
        if rows != 1:
            raise ValueError("1D block_table is only valid for one query row")
        table = table[None, :]
    if table.ndim != 2 or table.shape[0] != rows:
        raise ValueError("block_table must have shape [block_table_len] or [rows, block_table_len]")
    if table.shape[1] == 0:
        raise ValueError("block_table must not be empty")
    return table


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


def _round_to_bf16_float(value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


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
