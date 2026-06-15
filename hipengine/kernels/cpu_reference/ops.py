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


def qwen35_gguf_mtp_eh_proj(
    hidden_seed: ArrayLike,
    token_embedding: ArrayLike,
    eh_proj_weight: ArrayLike,
    hnorm_weight: ArrayLike,
    enorm_weight: ArrayLike,
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference for Qwen35 GGUF NextN h/e normalization + projection.

    llama.cpp normalizes the target hidden row with ``nextn.hnorm``, normalizes
    the token embedding with ``nextn.enorm``, concatenates ``[e_norm, h_norm]``,
    then applies ``nextn.eh_proj``.  This helper pins that order before the full
    draft-only NextN attention/MoE oracle lands.
    """

    hidden = np.asarray(hidden_seed, dtype=np.float32)
    embedding = np.asarray(token_embedding, dtype=np.float32)
    weight = np.asarray(eh_proj_weight, dtype=np.float32)
    hnorm = np.asarray(hnorm_weight, dtype=np.float32)
    enorm = np.asarray(enorm_weight, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden_seed must have shape [rows, hidden]")
    if embedding.shape != hidden.shape:
        raise ValueError("token_embedding must match hidden_seed shape")
    if weight.ndim != 2:
        raise ValueError("eh_proj_weight must have shape [hidden, 2 * hidden]")
    rows, hidden_size = hidden.shape
    if weight.shape != (hidden_size, hidden_size * 2):
        raise ValueError(
            "eh_proj_weight must have shape [hidden, 2 * hidden] "
            f"for hidden={hidden_size}; got {weight.shape}"
        )
    if hnorm.shape != (hidden_size,):
        raise ValueError("hnorm_weight must have shape [hidden]")
    if enorm.shape != (hidden_size,):
        raise ValueError("enorm_weight must have shape [hidden]")
    del rows
    h_norm = rmsnorm(hidden, hnorm, eps=eps)
    e_norm = rmsnorm(embedding, enorm, eps=eps)
    fused = np.concatenate([e_norm, h_norm], axis=-1)
    return np.matmul(fused, weight.T).astype(np.float32)


def qwen35_gguf_mtp_draft_topk_full_vocab_d2h(
    logits: ArrayLike,
    *,
    k: int = 1,
    vocab_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU oracle for the unfused GGUF MTP draft top-k fallback.

    The future backend ``topk_device`` variant should avoid copying the full
    vocab logits to the host. This reference deliberately models the unfused
    fallback/oracle: logits are already materialized on the host, then stable
    descending top-k is selected with lower token ids winning ties.
    """

    logits_arr = np.asarray(logits, dtype=np.float32)
    if logits_arr.ndim != 2:
        raise ValueError("logits must have shape [rows, vocab]")
    _, vocab = logits_arr.shape
    limit = vocab if vocab_limit is None else int(vocab_limit)
    if limit <= 0 or limit > vocab:
        raise ValueError("vocab_limit must be in 1..vocab")
    top_k = int(k)
    if top_k <= 0 or top_k > limit:
        raise ValueError("k must be in 1..vocab_limit")
    limited = logits_arr[:, :limit]
    token_ids = np.argsort(-limited, axis=-1, kind="stable")[:, :top_k].astype(np.int32)
    values = np.take_along_axis(limited, token_ids, axis=-1).astype(np.float32)
    return token_ids, values


def qwen35_gguf_mtp_shared_head_logits(
    nextn_hidden: ArrayLike,
    shared_head_norm_weight: ArrayLike,
    shared_head_weight: ArrayLike,
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference for Qwen35 GGUF NextN shared-head logits.

    llama.cpp applies ``nextn.shared_head_norm`` when present, otherwise the
    target ``output_norm``, exposes that row as ``h_nextn``, then applies
    ``nextn.shared_head_head`` when present, otherwise the target LM head.
    """

    hidden = np.asarray(nextn_hidden, dtype=np.float32)
    norm_weight = np.asarray(shared_head_norm_weight, dtype=np.float32)
    head_weight = np.asarray(shared_head_weight, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("nextn_hidden must have shape [rows, hidden]")
    rows, hidden_size = hidden.shape
    if norm_weight.shape != (hidden_size,):
        raise ValueError("shared_head_norm_weight must have shape [hidden]")
    if head_weight.ndim != 2 or head_weight.shape[1] != hidden_size:
        raise ValueError("shared_head_weight must have shape [vocab, hidden]")
    del rows
    normed = rmsnorm(hidden, norm_weight, eps=eps)
    return np.matmul(normed, head_weight.T).astype(np.float32)


def qwen35_gguf_mtp_boundary_logits(
    hidden_seed: ArrayLike,
    token_embedding: ArrayLike,
    eh_proj_weight: ArrayLike,
    hnorm_weight: ArrayLike,
    enorm_weight: ArrayLike,
    shared_head_norm_weight: ArrayLike,
    shared_head_weight: ArrayLike,
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference scaffold for known Qwen35 GGUF MTP boundary stages.

    This helper intentionally composes only the two llama.cpp-pinned boundary
    stages available before the full M3 oracle lands:
    ``hnorm/enorm + [e_norm, h_norm] + eh_proj`` followed by
    ``shared_head_norm + LM head``.  It does **not** implement the required
    NextN dense attention or MoE sublayers and must not be used as a parity
    oracle for final draft logits.
    """

    projected = qwen35_gguf_mtp_eh_proj(
        hidden_seed,
        token_embedding,
        eh_proj_weight,
        hnorm_weight,
        enorm_weight,
        eps=eps,
    )
    return qwen35_gguf_mtp_shared_head_logits(
        projected,
        shared_head_norm_weight,
        shared_head_weight,
        eps=eps,
    )


def qwen35_gguf_mtp_attention_sublayer(
    hidden: ArrayLike,
    attn_norm_weight: ArrayLike,
    wq_weight: ArrayLike,
    wk_weight: ArrayLike,
    wv_weight: ArrayLike,
    wo_weight: ArrayLike,
    q_norm_weight: ArrayLike,
    k_norm_weight: ArrayLike,
    *,
    num_heads: int,
    num_kv_heads: int,
    positions: ArrayLike | None = None,
    context_counts: ArrayLike | None = None,
    key_cache: ArrayLike | None = None,
    value_cache: ArrayLike | None = None,
    kv_base_offsets: ArrayLike | None = None,
    kv_live_counts: ArrayLike | None = None,
    kv_token_positions: ArrayLike | None = None,
    kv_evict_mask: ArrayLike | None = None,
    block_size: int | None = None,
    rope_cos: ArrayLike | None = None,
    rope_sin: ArrayLike | None = None,
    rotary_dim: int | None = None,
    scale: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference for the dense Qwen35 GGUF MTP attention sublayer.

    llama.cpp projects ``wq`` to an interleaved ``[Q_head, gate_head]`` layout
    per head, RMS-normalizes Q/K, applies RoPE, runs dense GQA attention over the
    MTP context's own K/V cache, multiplies the attention result by
    ``sigmoid(gate)``, applies ``wo``, then adds the pre-attention residual.
    Qwen35 GGUF attention width is not necessarily ``hidden / num_heads``; infer
    Q/K and V head widths from the GGUF norm/projection tensor shapes.

    This helper is a CPU oracle for that sublayer only.  Runtime MTP attention
    still must use the KVLiveSpans paged-KV ABI. Dense ``key_cache``/
    ``value_cache`` arguments model an already-materialized CPU cache, while the
    ``kv_base_offsets``/``kv_live_counts``/``kv_token_positions``/
    ``kv_evict_mask`` arguments provide a NumPy-only KVLiveSpans-shaped paged
    cache oracle for draft-runtime bring-up.
    """

    x = np.asarray(hidden, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    heads = int(num_heads)
    kv_heads = int(num_kv_heads)
    if heads <= 0 or kv_heads <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if heads % kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    attn_norm = np.asarray(attn_norm_weight, dtype=np.float32)
    q_norm = np.asarray(q_norm_weight, dtype=np.float32)
    k_norm = np.asarray(k_norm_weight, dtype=np.float32)
    if attn_norm.shape != (hidden_size,):
        raise ValueError("attn_norm_weight must have shape [hidden]")
    if q_norm.ndim != 1:
        raise ValueError("q_norm_weight must have shape [qk_head_dim]")
    qk_head_dim = q_norm.shape[0]
    if qk_head_dim <= 0:
        raise ValueError("qk_head_dim must be positive")
    if k_norm.shape != (qk_head_dim,):
        raise ValueError("k_norm_weight must have shape [qk_head_dim]")

    wq = np.asarray(wq_weight, dtype=np.float32)
    wk = np.asarray(wk_weight, dtype=np.float32)
    wv = np.asarray(wv_weight, dtype=np.float32)
    wo = np.asarray(wo_weight, dtype=np.float32)
    if wq.shape != (heads * 2 * qk_head_dim, hidden_size):
        raise ValueError("wq_weight must have shape [num_heads * 2 * qk_head_dim, hidden]")
    if wk.shape != (kv_heads * qk_head_dim, hidden_size):
        raise ValueError("wk_weight must have shape [num_kv_heads * qk_head_dim, hidden]")
    if wv.ndim != 2 or wv.shape[1] != hidden_size or wv.shape[0] % kv_heads != 0:
        raise ValueError("wv_weight must have shape [num_kv_heads * value_head_dim, hidden]")
    value_head_dim = wv.shape[0] // kv_heads
    if value_head_dim <= 0:
        raise ValueError("value_head_dim must be positive")
    if value_head_dim != qk_head_dim:
        raise ValueError("value_head_dim must match qk_head_dim for gated Qwen35 attention")
    if wo.shape != (hidden_size, heads * value_head_dim):
        raise ValueError("wo_weight must have shape [hidden, num_heads * value_head_dim]")

    normed = rmsnorm(x, attn_norm, eps=eps)
    q_full = linear(normed, wq)
    q_gate = q_full.reshape(tokens, heads, 2, qk_head_dim)
    query = rmsnorm(q_gate[:, :, 0, :], q_norm, eps=eps)
    gate = q_gate[:, :, 1, :]
    key_cur = rmsnorm(linear(normed, wk).reshape(tokens, kv_heads, qk_head_dim), k_norm, eps=eps)
    value_cur = linear(normed, wv).reshape(tokens, kv_heads, value_head_dim)

    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must be provided together")
    if rope_cos is not None and rope_sin is not None:
        query = rotate(query, rope_cos, rope_sin, rotary_dim=rotary_dim)
        key_cur = rotate(key_cur, rope_cos, rope_sin, rotary_dim=rotary_dim)

    pos = np.arange(tokens, dtype=np.int64) if positions is None else np.asarray(positions, dtype=np.int64)
    if pos.shape != (tokens,):
        raise ValueError("positions must have shape [tokens]")
    ctx = pos + 1 if context_counts is None else np.asarray(context_counts, dtype=np.int64)
    if ctx.shape != (tokens,):
        raise ValueError("context_counts must have shape [tokens]")

    if kv_base_offsets is None:
        if any(item is not None for item in (kv_live_counts, kv_token_positions, kv_evict_mask)):
            raise ValueError("kv_base_offsets is required when passing KVLiveSpans fields")
        key_dense = key_cur if key_cache is None else np.asarray(key_cache, dtype=np.float32)
        value_dense = value_cur if value_cache is None else np.asarray(value_cache, dtype=np.float32)
        if key_dense.ndim != 3 or value_dense.ndim != 3:
            raise ValueError("key_cache and value_cache must have shape [cache_tokens, num_kv_heads, head_dim]")
        if key_dense.shape[:2] != value_dense.shape[:2]:
            raise ValueError("key_cache and value_cache must have matching cache_tokens and num_kv_heads")
        if key_dense.shape[1:] != (kv_heads, qk_head_dim):
            raise ValueError("key_cache shape must match num_kv_heads and qk_head_dim")
        if value_dense.shape[1:] != (kv_heads, value_head_dim):
            raise ValueError("value_cache shape must match num_kv_heads and value_head_dim")
        attn = _dense_causal_gqa_attention(query, key_dense, value_dense, pos, ctx, scale=scale)
    else:
        if key_cache is None or value_cache is None:
            raise ValueError("paged KVLiveSpans attention requires key_cache and value_cache")
        if kv_live_counts is None:
            raise ValueError("kv_live_counts is required with kv_base_offsets")
        attn = _kv_live_spans_gqa_attention(
            query,
            np.asarray(key_cache, dtype=np.float32),
            np.asarray(value_cache, dtype=np.float32),
            base_offsets=kv_base_offsets,
            live_counts=kv_live_counts,
            token_positions=pos if kv_token_positions is None else kv_token_positions,
            evict_mask=kv_evict_mask,
            block_size=block_size,
            scale=scale,
        )
    gated = (attn * _sigmoid(gate)).reshape(tokens, heads * value_head_dim)
    return (x + linear(gated, wo)).astype(np.float32)


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


def qwen35_gguf_mtp_moe_routing(
    hidden: ArrayLike,
    router_weight: ArrayLike,
    *,
    experts_used: int,
    expert_weights_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference for Qwen35 GGUF MTP MoE router selection.

    llama.cpp's MTP FFN calls ``build_moe_ffn`` with ``ffn_gate_inp``,
    ``LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX``, ``norm_w=true``, and
    ``hparams.expert_weights_scale``.  This helper models the routing part only:
    dense router logits, softmax over all experts, top-k expert IDs, selected
    weight renormalization, and optional scaling.  The returned arrays feed the
    existing selected-expert FFN oracle.
    """

    x = np.asarray(hidden, dtype=np.float32)
    router = np.asarray(router_weight, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    if router.ndim != 2 or router.shape[1] != x.shape[1]:
        raise ValueError("router_weight must have shape [experts, hidden]")
    top_k = int(experts_used)
    if top_k <= 0:
        raise ValueError("experts_used must be positive")
    experts = router.shape[0]
    if top_k > experts:
        raise ValueError("experts_used must be <= number of experts")

    logits = np.matmul(x, router.T).astype(np.float32)
    probs = _softmax(logits, axis=-1).astype(np.float32)
    selected = np.argsort(-probs, axis=-1, kind="stable")[:, :top_k].astype(np.int64)
    selected_weights = np.take_along_axis(probs, selected, axis=-1).astype(np.float32)
    weight_sum = np.maximum(np.sum(selected_weights, axis=-1, keepdims=True), np.float32(6.103515625e-5))
    selected_weights = (selected_weights / weight_sum).astype(np.float32)
    scale_value = float(expert_weights_scale)
    if scale_value != 0.0 and scale_value != 1.0:
        selected_weights = (selected_weights * np.float32(scale_value)).astype(np.float32)
    return selected, selected_weights


def gguf_moe_selected_ffn(
    x: ArrayLike,
    selected_experts: ArrayLike,
    routing_weights: ArrayLike,
    gate_qweight: ArrayLike,
    up_qweight: ArrayLike,
    down_qweight: ArrayLike,
    gate_qtype: GGMLQuantizationType,
    up_qtype: GGMLQuantizationType,
    down_qtype: GGMLQuantizationType,
) -> np.ndarray:
    """Reference selected-expert MoE FFN (the megakernel-gated unit).

    For each token ``t`` and each of its ``top_k`` selected experts ``e``:
    ``gate = Wg_e @ x_t``; ``up = Wu_e @ x_t``; ``inter = silu(gate) * up``;
    ``down = Wd_e @ inter``; ``out_t += routing_weights[t, k] * down``.

    Expert weights are raw GGUF byte tensors shaped ``[E, out, row_bytes]``
    (rank-3). Each per-expert row block is dequantized with its quant type.
    This is the exact compute the B1 fused FFN megakernel reproduces, and it is
    row-invariant by construction (each token/expert pair is independent).
    """

    x_arr = np.asarray(x, dtype=np.float32)
    sel = np.asarray(selected_experts, dtype=np.int64)
    weights = np.asarray(routing_weights, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [tokens, hidden]")
    tokens, hidden = x_arr.shape
    if sel.ndim != 2 or sel.shape[0] != tokens:
        raise ValueError("selected_experts must have shape [tokens, top_k]")
    if weights.shape != sel.shape:
        raise ValueError("routing_weights must match selected_experts shape")
    top_k = sel.shape[1]
    gate_q = np.asarray(gate_qweight)
    up_q = np.asarray(up_qweight)
    down_q = np.asarray(down_qweight)
    if gate_q.ndim != 3 or up_q.ndim != 3 or down_q.ndim != 3:
        raise ValueError("expert weights must be rank-3 [E, out, row_bytes]")
    num_experts = gate_q.shape[0]
    if up_q.shape[0] != num_experts or down_q.shape[0] != num_experts:
        raise ValueError("gate/up/down must share the expert axis length")
    out = np.zeros((tokens, hidden), dtype=np.float32)
    for t in range(tokens):
        xt = x_arr[t : t + 1]
        for k in range(top_k):
            expert = int(sel[t, k])
            if expert < 0 or expert >= num_experts:
                raise ValueError("selected expert id out of range")
            gate = gguf_quant_gemv(xt, gate_q[expert], gate_qtype)
            up = gguf_quant_gemv(xt, up_q[expert], up_qtype)
            inter = (_silu(gate) * up).astype(np.float32)
            down = gguf_quant_gemv(inter, down_q[expert], down_qtype)
            out[t] += np.float32(weights[t, k]) * down[0]
    return out


def gguf_moe_ffn_block(
    x: ArrayLike,
    residual: ArrayLike,
    selected_experts: ArrayLike,
    routing_weights: ArrayLike,
    gate_qweight: ArrayLike,
    up_qweight: ArrayLike,
    down_qweight: ArrayLike,
    gate_qtype: GGMLQuantizationType,
    up_qtype: GGMLQuantizationType,
    down_qtype: GGMLQuantizationType,
    shared_gate_logit_weight: ArrayLike,
    shared_gate_qweight: ArrayLike,
    shared_up_qweight: ArrayLike,
    shared_down_qweight: ArrayLike,
    shared_qtype: GGMLQuantizationType,
) -> np.ndarray:
    """Reference full qwen35moe FFN block matching ``_run_post_attention_moe_c1``.

    ``out = residual + selected_ffn + sigmoid(shared_gate_logit) * shared_ffn``
    where ``selected_ffn`` is :func:`gguf_moe_selected_ffn`, the shared expert is
    a dense Q8_0 ``silu(gate) * up -> down`` FFN, and ``shared_gate_logit`` is the
    F32 ``ffn_gate_inp_shexp`` projection of ``x``.
    """

    x_arr = np.asarray(x, dtype=np.float32)
    res = np.asarray(residual, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [tokens, hidden]")
    if res.shape != x_arr.shape:
        raise ValueError("residual must match x shape")
    selected_out = gguf_moe_selected_ffn(
        x_arr,
        selected_experts,
        routing_weights,
        gate_qweight,
        up_qweight,
        down_qweight,
        gate_qtype,
        up_qtype,
        down_qtype,
    )
    shared_gate = gguf_quant_gemv(x_arr, np.asarray(shared_gate_qweight), shared_qtype)
    shared_up = gguf_quant_gemv(x_arr, np.asarray(shared_up_qweight), shared_qtype)
    shared_inter = (_silu(shared_gate) * shared_up).astype(np.float32)
    shared_out = gguf_quant_gemv(shared_inter, np.asarray(shared_down_qweight), shared_qtype)
    gate_vec = np.asarray(shared_gate_logit_weight, dtype=np.float32)
    if gate_vec.ndim != 1 or gate_vec.shape[0] != x_arr.shape[1]:
        raise ValueError("shared_gate_logit_weight must have shape [hidden]")
    shared_gate_logit = x_arr @ gate_vec
    gate = _sigmoid(shared_gate_logit)[:, None]
    return (res + selected_out + gate * shared_out).astype(np.float32)


def qwen35_gguf_mtp_ffn_sublayer(
    hidden: ArrayLike,
    attn_post_norm_weight: ArrayLike,
    router_weight: ArrayLike,
    gate_qweight: ArrayLike,
    up_qweight: ArrayLike,
    down_qweight: ArrayLike,
    gate_qtype: GGMLQuantizationType,
    up_qtype: GGMLQuantizationType,
    down_qtype: GGMLQuantizationType,
    shared_gate_logit_weight: ArrayLike,
    shared_gate_qweight: ArrayLike,
    shared_up_qweight: ArrayLike,
    shared_down_qweight: ArrayLike,
    shared_qtype: GGMLQuantizationType,
    *,
    experts_used: int,
    expert_weights_scale: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference for the Qwen35 GGUF MTP post-attention FFN sublayer.

    llama.cpp saves the post-attention residual, applies ``attn_post_norm``,
    routes with ``ffn_gate_inp`` using softmax top-k, runs the selected MoE FFN,
    adds the gated shared expert when present, and finally adds the saved
    residual.  This helper covers the common shared-expert Qwen35 path by
    composing :func:`qwen35_gguf_mtp_moe_routing` with
    :func:`gguf_moe_ffn_block`.
    """

    x = np.asarray(hidden, dtype=np.float32)
    norm_weight = np.asarray(attn_post_norm_weight, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    if norm_weight.shape != (x.shape[1],):
        raise ValueError("attn_post_norm_weight must have shape [hidden]")
    normed = rmsnorm(x, norm_weight, eps=eps)
    selected_experts, routing_weights = qwen35_gguf_mtp_moe_routing(
        normed,
        router_weight,
        experts_used=experts_used,
        expert_weights_scale=expert_weights_scale,
    )
    return gguf_moe_ffn_block(
        normed,
        x,
        selected_experts,
        routing_weights,
        gate_qweight,
        up_qweight,
        down_qweight,
        gate_qtype,
        up_qtype,
        down_qtype,
        shared_gate_logit_weight,
        shared_gate_qweight,
        shared_up_qweight,
        shared_down_qweight,
        shared_qtype,
    )


def qwen35_gguf_mtp_nextn_layer_logits(
    hidden_seed: ArrayLike,
    token_embedding: ArrayLike,
    eh_proj_weight: ArrayLike,
    hnorm_weight: ArrayLike,
    enorm_weight: ArrayLike,
    attn_norm_weight: ArrayLike,
    wq_weight: ArrayLike,
    wk_weight: ArrayLike,
    wv_weight: ArrayLike,
    wo_weight: ArrayLike,
    q_norm_weight: ArrayLike,
    k_norm_weight: ArrayLike,
    attn_post_norm_weight: ArrayLike,
    router_weight: ArrayLike,
    gate_qweight: ArrayLike,
    up_qweight: ArrayLike,
    down_qweight: ArrayLike,
    gate_qtype: GGMLQuantizationType,
    up_qtype: GGMLQuantizationType,
    down_qtype: GGMLQuantizationType,
    shared_gate_logit_weight: ArrayLike,
    shared_gate_qweight: ArrayLike,
    shared_up_qweight: ArrayLike,
    shared_down_qweight: ArrayLike,
    shared_qtype: GGMLQuantizationType,
    shared_head_norm_weight: ArrayLike,
    shared_head_weight: ArrayLike,
    *,
    num_heads: int,
    num_kv_heads: int,
    experts_used: int,
    positions: ArrayLike | None = None,
    context_counts: ArrayLike | None = None,
    key_cache: ArrayLike | None = None,
    value_cache: ArrayLike | None = None,
    kv_base_offsets: ArrayLike | None = None,
    kv_live_counts: ArrayLike | None = None,
    kv_token_positions: ArrayLike | None = None,
    kv_evict_mask: ArrayLike | None = None,
    block_size: int | None = None,
    rope_cos: ArrayLike | None = None,
    rope_sin: ArrayLike | None = None,
    rotary_dim: int | None = None,
    scale: float | None = None,
    expert_weights_scale: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference for one dense Qwen35 GGUF MTP NextN draft layer.

    The composition follows the llama.cpp draft-only layer order: ``hnorm`` and
    ``enorm`` into ``eh_proj``, MTP self-attention with either a dense CPU cache
    or KVLiveSpans-shaped paged CPU cache, post-attention MoE/shared-expert FFN,
    then shared-head norm plus LM head logits.  This is still a CPU oracle;
    runtime attention/KV writes must use the KVLiveSpans paged-KV ABI.
    """

    projected = qwen35_gguf_mtp_eh_proj(
        hidden_seed,
        token_embedding,
        eh_proj_weight,
        hnorm_weight,
        enorm_weight,
        eps=eps,
    )
    attended = qwen35_gguf_mtp_attention_sublayer(
        projected,
        attn_norm_weight,
        wq_weight,
        wk_weight,
        wv_weight,
        wo_weight,
        q_norm_weight,
        k_norm_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        positions=positions,
        context_counts=context_counts,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_base_offsets=kv_base_offsets,
        kv_live_counts=kv_live_counts,
        kv_token_positions=kv_token_positions,
        kv_evict_mask=kv_evict_mask,
        block_size=block_size,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        rotary_dim=rotary_dim,
        scale=scale,
        eps=eps,
    )
    ffn_out = qwen35_gguf_mtp_ffn_sublayer(
        attended,
        attn_post_norm_weight,
        router_weight,
        gate_qweight,
        up_qweight,
        down_qweight,
        gate_qtype,
        up_qtype,
        down_qtype,
        shared_gate_logit_weight,
        shared_gate_qweight,
        shared_up_qweight,
        shared_down_qweight,
        shared_qtype,
        experts_used=experts_used,
        expert_weights_scale=expert_weights_scale,
        eps=eps,
    )
    return qwen35_gguf_mtp_shared_head_logits(
        ffn_out,
        shared_head_norm_weight,
        shared_head_weight,
        eps=eps,
    )


def gguf_q4_k_moe_selected_ffn(
    x: ArrayLike,
    selected_experts: ArrayLike,
    routing_weights: ArrayLike,
    gate_qweight: ArrayLike,
    up_qweight: ArrayLike,
    down_qweight: ArrayLike,
) -> np.ndarray:
    """Q4_K/Q4_K/Q4_K specialization (the Qwen3.6-35B-A3B-UD-Q4_K_S expert path)."""

    return gguf_moe_selected_ffn(
        x,
        selected_experts,
        routing_weights,
        gate_qweight,
        up_qweight,
        down_qweight,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q4_K,
    )


def awq_pack8_shift_for_lane(lane: int) -> int:
    """Bit shift for the AWQ pack8 interleaved layout (matches the HIP kernels)."""

    packed_pos = (4 + (lane >> 1)) if (lane & 1) else (lane >> 1)
    return packed_pos * 4


def awq_pack8_dequant_transposed(
    qweight: ArrayLike,
    qzeros: ArrayLike,
    scales: ArrayLike,
    in_features: int,
    out_features: int,
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize AWQ W4 pack8 (transposed qweight layout) to dense ``[out, in]``.

    ``qweight`` is ``[out_features/8, in_features]`` int32 with 8 output channels
    packed per word; ``qzeros`` is ``[groups, out_features/8]`` int32 (same
    packing); ``scales`` is ``[groups, out_features]`` f32. Per the HIP decode
    path: ``w[out, k] = (q - z) * scale`` with ``q``/``z`` 4-bit nibbles selected
    by :func:`awq_pack8_shift_for_lane` and ``group = k // group_size``.
    """

    qw = np.asarray(qweight).view(np.uint32)
    qz = np.asarray(qzeros).view(np.uint32)
    sc = np.asarray(scales, dtype=np.float32)
    out_packed = out_features // 8
    if qw.shape != (out_packed, in_features):
        raise ValueError("qweight must have shape [out_features/8, in_features]")
    groups = in_features // group_size
    if qz.shape != (groups, out_packed) or sc.shape != (groups, out_features):
        raise ValueError("qzeros/scales shapes do not match in/out/group sizes")
    group_of_k = np.arange(in_features, dtype=np.int64) // group_size
    weight = np.empty((out_features, in_features), dtype=np.float32)
    for out_col in range(out_features):
        pack = out_col >> 3
        shift = np.uint32(awq_pack8_shift_for_lane(out_col & 7))
        q = ((qw[pack, :] >> shift) & np.uint32(0xF)).astype(np.float32)
        z = ((qz[group_of_k, pack] >> shift) & np.uint32(0xF)).astype(np.float32)
        weight[out_col] = (q - z) * sc[group_of_k, out_col]
    return weight


def awq_pack8_gemv_transposed(
    x: ArrayLike,
    qweight: ArrayLike,
    qzeros: ArrayLike,
    scales: ArrayLike,
    in_features: int,
    out_features: int,
    group_size: int = 128,
) -> np.ndarray:
    """Reference GEMV over AWQ W4 pack8 transposed weights: ``x @ dequant(W).T``."""

    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2 or x_arr.shape[1] != in_features:
        raise ValueError("x must have shape [rows, in_features]")
    weight = awq_pack8_dequant_transposed(qweight, qzeros, scales, in_features, out_features, group_size)
    return np.matmul(x_arr, weight.T).astype(np.float32)


def paro_rotate1(
    x: ArrayLike,
    pairs: ArrayLike,
    theta: ArrayLike,
    scales: ArrayLike,
    group_size: int,
    krot: int,
) -> np.ndarray:
    """Reference PARO single-output incoherence rotation (matches paro_rotate1).

    Per group of ``group_size`` channels, pre-scale by ``scales`` then apply
    ``krot`` rounds of Givens rotations on calibration ``pairs`` with ``theta``
    angles: ``(b[i], b[j]) -> (c*b[i] + s*b[j], -s*b[i] + c*b[j])``. ``pairs`` is
    ``[krot, hidden]`` int (group-local indices interleaved per pair); ``theta``
    is ``[krot, hidden/2]`` f32; ``scales`` is ``[hidden]`` f32.
    """

    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [tokens, hidden]")
    tokens, hidden = x_arr.shape
    pairs_arr = np.asarray(pairs, dtype=np.int64).reshape(krot, hidden)
    theta_arr = np.asarray(theta, dtype=np.float32).reshape(krot, hidden // 2)
    scales_arr = np.asarray(scales, dtype=np.float32).reshape(hidden)
    half = group_size // 2
    groups = hidden // group_size
    out = np.empty_like(x_arr)
    for row in range(tokens):
        for g in range(groups):
            base = g * group_size
            buf = np.empty(group_size, dtype=np.float32)
            buf[:half] = x_arr[row, base : base + half] * scales_arr[base : base + half]
            buf[half:] = x_arr[row, base + half : base + group_size] * scales_arr[base + half : base + group_size]
            for r in range(krot):
                new_buf = buf.copy()
                for lane in range(half):
                    pb = base + 2 * lane
                    i = int(pairs_arr[r, pb])
                    j = int(pairs_arr[r, pb + 1])
                    angle = float(theta_arr[r, g * half + lane])
                    s = np.float32(np.sin(angle))
                    c = np.float32(np.cos(angle))
                    bi = buf[i]
                    bj = buf[j]
                    new_buf[i] = bi * c + bj * s
                    new_buf[j] = -bi * s + bj * c
                buf = new_buf
            out[row, base : base + half] = buf[:half]
            out[row, base + half : base + group_size] = buf[half:]
    return out


def paro_moe_selected_ffn(
    x: ArrayLike,
    selected_experts: ArrayLike,
    routing_weights: ArrayLike,
    gate_awq: tuple,
    up_awq: tuple,
    down_awq: tuple,
    hidden: int,
    ffn_len: int,
    group_size: int,
    rotate1_calib: tuple,
    down_rotate_calib: tuple,
) -> np.ndarray:
    """Reference PARO selected-expert MoE FFN (the B3 megakernel-gated unit).

    Matches the deployed PARO chain: ``rotate1(x)`` [shared incoherence rotation
    over hidden] then, per (token, expert): AWQ gate/up GEMV -> ``silu(gate)*up``
    -> down-rotate [shared, over ffn_len] -> AWQ down GEMV -> routing-weighted
    combine. ``*_awq`` are ``(qweight[E,...], qzeros[E,...], scales[E,...])``
    rank-(1+...) per-expert AWQ pack8 transposed weights. ``rotate1_calib`` /
    ``down_rotate_calib`` are ``(pairs, theta, scales, krot)`` for
    :func:`paro_rotate1` over hidden / ffn_len respectively.

    Row-invariant by construction: ``rotate1`` is row-independent and each
    (token, expert) pair is processed independently.
    """

    x_arr = np.asarray(x, dtype=np.float32)
    sel = np.asarray(selected_experts, dtype=np.int64)
    weights = np.asarray(routing_weights, dtype=np.float32)
    if x_arr.ndim != 2 or x_arr.shape[1] != hidden:
        raise ValueError("x must have shape [tokens, hidden]")
    tokens = x_arr.shape[0]
    if sel.ndim != 2 or sel.shape[0] != tokens or weights.shape != sel.shape:
        raise ValueError("selected_experts/routing_weights must be [tokens, top_k]")
    top_k = sel.shape[1]
    gate_qw, gate_qz, gate_sc = gate_awq
    up_qw, up_qz, up_sc = up_awq
    down_qw, down_qz, down_sc = down_awq
    r1_pairs, r1_theta, r1_scales, r1_krot = rotate1_calib
    d_pairs, d_theta, d_scales, d_krot = down_rotate_calib

    x_rot = paro_rotate1(x_arr, r1_pairs, r1_theta, r1_scales, group_size, int(r1_krot))
    out = np.zeros((tokens, hidden), dtype=np.float32)
    for t in range(tokens):
        xt = x_rot[t : t + 1]
        for k in range(top_k):
            e = int(sel[t, k])
            gate = awq_pack8_gemv_transposed(xt, gate_qw[e], gate_qz[e], gate_sc[e], hidden, ffn_len, group_size)
            up = awq_pack8_gemv_transposed(xt, up_qw[e], up_qz[e], up_sc[e], hidden, ffn_len, group_size)
            act = (_silu(gate) * up).astype(np.float32)
            act_rot = paro_rotate1(act, d_pairs, d_theta, d_scales, group_size, int(d_krot))
            down = awq_pack8_gemv_transposed(act_rot, down_qw[e], down_qz[e], down_sc[e], ffn_len, hidden, group_size)
            out[t] += np.float32(weights[t, k]) * down[0]
    return out


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
        "qwen35_gguf_mtp_eh_proj": qwen35_gguf_mtp_eh_proj,
        "qwen35_gguf_mtp_shared_head_logits": qwen35_gguf_mtp_shared_head_logits,
        "qwen35_gguf_mtp_boundary_logits": qwen35_gguf_mtp_boundary_logits,
        "qwen35_gguf_mtp_attention_sublayer": qwen35_gguf_mtp_attention_sublayer,
        "qwen35_gguf_mtp_moe_routing": qwen35_gguf_mtp_moe_routing,
        "qwen35_gguf_mtp_ffn_sublayer": qwen35_gguf_mtp_ffn_sublayer,
        "qwen35_gguf_mtp_nextn_layer_logits": qwen35_gguf_mtp_nextn_layer_logits,
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
        KernelKey("cpu_reference", "mtp_nextn_eh_proj", "gguf_f32", "qwen35"),
        qwen35_gguf_mtp_eh_proj,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "mtp_nextn_shared_head", "gguf_f32", "qwen35"),
        qwen35_gguf_mtp_shared_head_logits,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "mtp_nextn_boundary_logits", "gguf_f32", "qwen35"),
        qwen35_gguf_mtp_boundary_logits,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h"),
        qwen35_gguf_mtp_draft_topk_full_vocab_d2h,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "mtp_nextn_attention", "gguf_f32", "qwen35_dense"),
        qwen35_gguf_mtp_attention_sublayer,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "mtp_nextn_moe_routing", "gguf_f32", "qwen35_softmax_topk"),
        qwen35_gguf_mtp_moe_routing,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "mtp_nextn_ffn", "gguf_moe", "qwen35_shared"),
        qwen35_gguf_mtp_ffn_sublayer,
        replace=replace,
    )
    for quant in ("w4_gguf", "gguf_moe"):
        register(
            KernelKey("cpu_reference", "mtp_nextn_layer", quant, "qwen35_dense_logits"),
            qwen35_gguf_mtp_nextn_layer_logits,
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
    register(
        KernelKey("cpu_reference", "moe_ffn_selected", "gguf_q4_k"),
        gguf_q4_k_moe_selected_ffn,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "linear", "w4_paro", "pack8_gemv_transposed"),
        awq_pack8_gemv_transposed,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "rotate1", "w4_paro"),
        paro_rotate1,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "moe_ffn_selected", "w4_paro"),
        paro_moe_selected_ffn,
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


def _dense_causal_gqa_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    positions: np.ndarray,
    context_counts: np.ndarray,
    *,
    scale: float | None,
) -> np.ndarray:
    if query.ndim != 3:
        raise ValueError("query must have shape [tokens, num_heads, qk_head_dim]")
    tokens, heads, qk_head_dim = query.shape
    if key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("key_cache and value_cache must have shape [cache_tokens, num_kv_heads, head_dim]")
    cache_tokens, kv_heads, key_dim = key_cache.shape
    value_tokens, value_heads, value_head_dim = value_cache.shape
    if (value_tokens, value_heads) != (cache_tokens, kv_heads):
        raise ValueError("key_cache and value_cache must have matching cache_tokens and num_kv_heads")
    if key_dim != qk_head_dim:
        raise ValueError("query qk_head_dim must match key_cache head_dim")
    if heads % kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if positions.shape != (tokens,) or context_counts.shape != (tokens,):
        raise ValueError("positions and context_counts must have shape [tokens]")
    scale_value = (qk_head_dim ** -0.5) if scale is None else float(scale)
    kv_group = heads // kv_heads
    out = np.empty((tokens, heads, value_head_dim), dtype=np.float32)
    for row in range(tokens):
        position = int(positions[row])
        context = int(context_counts[row])
        if position < 0:
            raise ValueError("positions must be non-negative")
        if context <= 0:
            raise ValueError("context_counts must be positive")
        if position >= cache_tokens or context > cache_tokens:
            raise ValueError("positions/context_counts exceed cache length")
        visible_positions = [cache_pos for cache_pos in range(context) if cache_pos <= position]
        if not visible_positions:
            raise ValueError("causal mask left no visible cache positions")
        for q_head in range(heads):
            kv_head = q_head // kv_group
            keys = key_cache[visible_positions, kv_head, :]
            values = value_cache[visible_positions, kv_head, :]
            weights = _softmax(np.matmul(keys, query[row, q_head]) * scale_value, axis=0)
            out[row, q_head] = np.matmul(weights, values)
    return out


def _kv_live_spans_gqa_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    *,
    base_offsets: ArrayLike,
    live_counts: ArrayLike,
    token_positions: ArrayLike,
    evict_mask: ArrayLike | None,
    block_size: int | None,
    scale: float | None,
) -> np.ndarray:
    if query.ndim != 3:
        raise ValueError("query must have shape [tokens, num_heads, qk_head_dim]")
    tokens, heads, qk_head_dim = query.shape
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("paged key_cache and value_cache must have shape [blocks, block, num_kv_heads, head_dim]")
    if key_cache.shape[:3] != value_cache.shape[:3]:
        raise ValueError("paged key_cache and value_cache must match through num_kv_heads")
    block = key_cache.shape[1] if block_size is None else int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    if key_cache.shape[1] != block or value_cache.shape[1] != block:
        raise ValueError("block_size must match paged key/value cache shape")
    kv_heads = key_cache.shape[2]
    key_dim = key_cache.shape[3]
    value_head_dim = value_cache.shape[3]
    if key_dim != qk_head_dim:
        raise ValueError("query qk_head_dim must match key_cache head_dim")
    if heads % kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    counts = np.asarray(live_counts, dtype=np.int64).reshape(-1)
    positions = np.asarray(token_positions, dtype=np.int64).reshape(-1)
    if counts.shape != (tokens,):
        raise ValueError("kv_live_counts must have shape [tokens]")
    if positions.shape != (tokens,):
        raise ValueError("kv_token_positions must have shape [tokens]")
    tables = _normalize_block_tables(base_offsets, rows=tokens)
    assert tables is not None
    mask = None
    if evict_mask is not None:
        mask = np.asarray(evict_mask, dtype=np.bool_)
        if mask.ndim == 1:
            if tokens != 1:
                raise ValueError("1D kv_evict_mask is only valid for one query row")
            mask = mask[None, :]
        if mask.ndim != 2 or mask.shape[0] != tokens:
            raise ValueError("kv_evict_mask must have shape [tokens, max_live_count]")

    scale_value = (qk_head_dim ** -0.5) if scale is None else float(scale)
    kv_group = heads // kv_heads
    out = np.empty((tokens, heads, value_head_dim), dtype=np.float32)
    for row in range(tokens):
        live_count = int(counts[row])
        position = int(positions[row])
        if live_count <= 0:
            raise ValueError("kv_live_counts must be positive")
        if position < 0:
            raise ValueError("kv_token_positions must be non-negative")
        row_mask = None if mask is None else mask[row]
        visible_slots = [
            slot
            for slot in range(live_count)
            if slot <= position and not (row_mask is not None and slot < row_mask.shape[0] and bool(row_mask[slot]))
        ]
        if not visible_slots:
            raise ValueError("KVLiveSpans mask left no visible cache positions")
        for q_head in range(heads):
            kv_head = q_head // kv_group
            keys = np.stack(
                [
                    _cache_row(
                        key_cache,
                        slot,
                        kv_head,
                        dense_cache=False,
                        block_size=block,
                        block_table=tables[row],
                    )
                    for slot in visible_slots
                ],
                axis=0,
            )
            values = np.stack(
                [
                    _cache_row(
                        value_cache,
                        slot,
                        kv_head,
                        dense_cache=False,
                        block_size=block,
                        block_table=tables[row],
                    )
                    for slot in visible_slots
                ],
                axis=0,
            )
            weights = _softmax(np.matmul(keys, query[row, q_head]) * scale_value, axis=0)
            out[row, q_head] = np.matmul(weights, values)
    return out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    finite = np.isfinite(x)
    safe_x = np.where(finite, x, -np.inf)
    max_x = np.max(safe_x, axis=axis, keepdims=True)
    shifted = safe_x - max_x
    exp = np.where(finite, np.exp(shifted), 0.0)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom
