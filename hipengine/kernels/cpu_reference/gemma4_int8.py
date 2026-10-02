"""CPU representation oracle for the Gemma 4 direct INT8 decode consumer.

This module is the independent reference for the gfx1100 Gemma 4 per-token/head
INT8 attention consumer (``gemma4_attention_int8.hip``). It works on exactly the
representation the writer stores: an INT8 payload plus a per-token/per-KV-head
FP16 or FP32 scale, reconstructed as ``float32(int8) * float32(scale)``. It does
not use a GPU kernel as its oracle and it does not reconstruct from the original
BF16 rows -- comparing against BF16 is a separate diagnostic.

Mask semantics match the Gemma 4 reference (``gemma4_attention_mask``): a key is
kept when it is not evicted, its absolute position is non-negative and no later
than the query position, and -- on a sliding layer -- strictly inside the window.
When span position/eviction tensors are absent the dense fill is uniform: slot
``j`` is at position ``j`` and the query sits at ``context_len - 1``.
"""

from __future__ import annotations

import numpy as np

from hipengine.kernels.cpu_reference.ops import dequantize_kv_int8_per_token_head

__all__ = [
    "gemma4_attention_decode_int8_per_token_head",
    "gemma4_attention_prefill_int8_per_token_head",
]


def _softmax_masked(logits: np.ndarray) -> np.ndarray:
    """Softmax over a row whose masked entries are ``-inf``.

    An all-masked row returns all zeros rather than NaN, which is the consumer's
    documented all-masked output.
    """

    finite = np.isfinite(logits)
    if not finite.any():
        return np.zeros_like(logits, dtype=np.float32)
    row_max = np.max(np.where(finite, logits, -np.inf))
    shifted = np.where(finite, logits - row_max, -np.inf)
    weights = np.where(finite, np.exp(shifted), 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros_like(weights, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _attention_head_row(
    q_head: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    keep: np.ndarray,
    scale: float,
) -> np.ndarray | None:
    """One query head's masked attention row over already-gathered K/V.

    ``keys``/``values`` are ``[context, head_dim]`` for one KV head, already
    reconstructed as ``float32(int8) * float32(scale)``. Returns ``None`` when a
    visible logit or visible reconstructed V is non-finite: a numerical failure
    that the consumers report as a NaN row, independent of the slot's softmax
    weight. A masked slot never participates.
    """

    visible_logits = (keys @ q_head).astype(np.float32) * np.float32(scale)
    if not np.all(np.isfinite(visible_logits[keep])) or not np.all(np.isfinite(values[keep])):
        return None
    logits = np.where(keep, visible_logits, -np.inf)
    weights = _softmax_masked(logits)
    # Masked slots never contribute, so a poisoned (NaN/Inf) masked V scale
    # cannot leak into the result; a visible nonfinite V does propagate.
    contrib = np.where(keep[:, None], weights[:, None] * values, 0.0)
    return contrib.sum(axis=0)


def gemma4_attention_decode_int8_per_token_head(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    block_table: np.ndarray,
    live_count: int,
    *,
    block_size: int,
    scale: float = 1.0,
    token_positions: np.ndarray | None = None,
    evict_mask: np.ndarray | None = None,
    row_position: int | None = None,
    sliding_window: int | None = None,
    output_dtype: str | np.dtype | type = np.float32,
) -> np.ndarray:
    """Reference single-row Gemma 4 INT8 decode attention over a paged cache.

    ``query`` is ``[num_q_heads, head_dim]`` in FP32. ``key_cache`` /
    ``value_cache`` are ``[blocks, block_size, num_kv_heads, head_dim]`` INT8 and
    ``k_scale`` / ``v_scale`` are ``[blocks, block_size, num_kv_heads]`` FP16 or
    FP32, exactly as the writer stores them. ``block_table`` maps logical blocks
    to physical cache blocks. ``live_count`` is the number of live logical slots.

    ``token_positions`` (absolute position per logical slot) and ``evict_mask``
    (nonzero = evicted) override the uniform dense fill when supplied.
    ``row_position`` is the query's absolute position; it defaults to
    ``live_count - 1``. ``sliding_window`` enables the Gemma 4 sliding bound.

    An empty span (``live_count == 0``) is supported and returns an all-zero
    row: there are no keys to attend to, and no stale output is left behind. A
    negative ``live_count`` is a malformed count and is refused, distinct from
    the supported empty case.

    A non-positive ``sliding_window`` is a global (causal-only) layer, matching
    the kernel. A *visible* non-finite logit (non-finite Q/K reconstruction) or a
    *visible* non-finite reconstructed V is a failure and yields a NaN row,
    independent of the slot's softmax weight; masked slots never participate, so
    a poisoned (NaN/Inf) masked K/V scale cannot leak into the result.
    """

    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("query must have shape [num_q_heads, head_dim]")
    context = int(live_count)
    if context < 0:
        raise ValueError("live_count must not be negative")
    if context == 0:
        return np.zeros(q.shape, dtype=output_dtype)
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    key, value = dequantize_kv_int8_per_token_head(
        key_cache, value_cache, k_scale, v_scale
    )
    if key.ndim != 4:
        raise ValueError("key_cache/value_cache must have shape [blocks, block, Hkv, D]")
    num_q_heads, head_dim = q.shape
    num_kv_heads = key.shape[2]
    if key.shape[3] != head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    kv_group = num_q_heads // num_kv_heads
    table = np.asarray(block_table, dtype=np.int64).reshape(-1)
    if table.size * block < context:
        raise ValueError("block_table is too short for live_count")

    positions = None if token_positions is None else np.asarray(token_positions, dtype=np.int64)
    evicted = None if evict_mask is None else np.asarray(evict_mask).astype(bool)
    query_position = context - 1 if row_position is None else int(row_position)
    window = None if sliding_window is None else int(sliding_window)
    if window is not None and window <= 0:
        window = None  # a non-positive window is a global (causal-only) layer

    physical = np.empty(context, dtype=np.int64)
    for slot in range(context):
        logical_block, offset = divmod(slot, block)
        physical[slot] = table[logical_block] * block + offset
    cache_blocks = key.shape[0]
    if np.any(physical < 0) or np.any(physical >= cache_blocks * block):
        raise ValueError("block_table physical id is out of range for the cache")

    out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for head in range(num_q_heads):
        kv_head = head // kv_group
        # Gather through the physical slot list (robust to shuffled pages).
        keys = np.stack([key[int(p) // block, int(p) % block, kv_head] for p in physical])
        values = np.stack([value[int(p) // block, int(p) % block, kv_head] for p in physical])
        keep = np.ones(context, dtype=bool)
        for slot in range(context):
            pos = slot if positions is None else int(positions[slot])
            visible = pos >= 0 and pos <= query_position
            if window is not None:
                visible = visible and pos > query_position - window
            if evicted is not None:
                visible = visible and not bool(evicted[slot])
            keep[slot] = visible
        # A visible non-finite logit or reconstructed V is a failure, not a
        # masked key: it must not be silently dropped or turned into a success
        # zero. The V check is independent of the softmax weight, so an
        # underflowed slot cannot hide a poisoned visible value.
        row = _attention_head_row(q[head], keys, values, keep, scale)
        out[head] = np.nan if row is None else row
    return out.astype(output_dtype)


def gemma4_attention_prefill_int8_per_token_head(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    block_table: np.ndarray,
    live_counts: np.ndarray,
    *,
    block_size: int,
    row_positions: np.ndarray | None = None,
    scale: float = 1.0,
    token_positions: np.ndarray | None = None,
    evict_mask: np.ndarray | None = None,
    sliding_window: int | None = None,
    output_dtype: str | np.dtype | type = np.float32,
) -> np.ndarray:
    """Reference multi-row Gemma 4 INT8 prefill attention over a shared prefix.

    Declared semantics: every query row attends over ONE SHARED LOGICAL KV PREFIX
    through a single 1-D ``block_table``. A per-row (row-major) page table is a
    separate prefix and is not modelled here. ``query`` is
    ``[rows, num_q_heads, head_dim]`` FP32 and the result has the same shape.
    ``live_counts`` holds one per-row causal prefix length and ``row_positions``
    one per-row absolute query position.

    ``key_cache``/``value_cache`` are ``[blocks, block_size, num_kv_heads,
    head_dim]`` INT8 and ``k_scale``/``v_scale`` are ``[blocks, block_size,
    num_kv_heads]`` FP16 or FP32, exactly as the writer stores them. ``scale`` is
    Gemma 4's ``geometry.scale`` (1.0).

    A zero live count is a supported empty row (zeros). A negative count is
    refused. A row whose position leaves no visible slot is all-masked and
    returns zeros. A *visible* non-finite logit or reconstructed V is a failure
    for that (row, head) and yields a NaN row there, independent of the slot's
    softmax weight; masked slots never participate.
    """

    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 3:
        raise ValueError("query must have shape [rows, num_q_heads, head_dim]")
    rows, num_q_heads, head_dim = (int(dim) for dim in q.shape)
    counts = np.asarray(live_counts, dtype=np.int64).reshape(-1)
    if counts.size != rows:
        raise ValueError("live_counts must have one entry per query row")
    if np.any(counts < 0):
        raise ValueError("live_count must not be negative")
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    key, value = dequantize_kv_int8_per_token_head(
        key_cache, value_cache, k_scale, v_scale
    )
    if key.ndim != 4:
        raise ValueError("key_cache/value_cache must have shape [blocks, block, Hkv, D]")
    num_kv_heads = int(key.shape[2])
    if int(key.shape[3]) != head_dim:
        raise ValueError("query head_dim must match cache head_dim")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    kv_group = num_q_heads // num_kv_heads
    table = np.asarray(block_table, dtype=np.int64).reshape(-1)
    max_context = int(counts.max()) if counts.size else 0
    if table.size * block < max_context:
        raise ValueError("block_table is too short for the largest live count")

    positions = None if token_positions is None else np.asarray(token_positions, dtype=np.int64)
    evicted = None if evict_mask is None else np.asarray(evict_mask).astype(bool)
    if row_positions is None:
        query_positions = counts - 1
    else:
        query_positions = np.asarray(row_positions, dtype=np.int64).reshape(-1)
        if query_positions.size != rows:
            raise ValueError("row_positions must have one entry per query row")
    window = None if sliding_window is None else int(sliding_window)
    if window is not None and window <= 0:
        window = None  # a non-positive window is a global (causal-only) layer

    cache_blocks = int(key.shape[0])
    out = np.empty((rows, num_q_heads, head_dim), dtype=np.float32)
    for row in range(rows):
        context = int(counts[row])
        if context == 0:
            out[row] = 0.0
            continue
        physical = np.empty(context, dtype=np.int64)
        for slot in range(context):
            logical_block, offset = divmod(slot, block)
            physical[slot] = table[logical_block] * block + offset
        if np.any(physical < 0) or np.any(physical >= cache_blocks * block):
            raise ValueError("block_table physical id is out of range for the cache")
        query_position = int(query_positions[row])
        for head in range(num_q_heads):
            kv_head = head // kv_group
            keys = np.stack([key[int(p) // block, int(p) % block, kv_head] for p in physical])
            values = np.stack([value[int(p) // block, int(p) % block, kv_head] for p in physical])
            keep = np.ones(context, dtype=bool)
            for slot in range(context):
                pos = slot if positions is None else int(positions[slot])
                visible = pos >= 0 and pos <= query_position
                if window is not None:
                    visible = visible and pos > query_position - window
                if evicted is not None:
                    visible = visible and not bool(evicted[slot])
                keep[slot] = visible
            head_row = _attention_head_row(q[row, head], keys, values, keep, scale)
            out[row, head] = np.nan if head_row is None else head_row
    return out.astype(output_dtype)
