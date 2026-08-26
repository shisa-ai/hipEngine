"""Torch-free NumPy correctness oracles for Qwen3.8-Flash-Next primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

ArrayLike = np.ndarray | Sequence[float] | Sequence[int]
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class GRReadResult:
    normalized: np.ndarray
    gate: np.ndarray
    mixed: np.ndarray
    inject_logits: np.ndarray


@dataclass(frozen=True)
class PLEHashState:
    tokens: tuple[int, ...]
    next_position: int


@dataclass(frozen=True)
class PLEConvState:
    history: np.ndarray
    next_position: int


@dataclass(frozen=True)
class PLEInjectionResult:
    residual: np.ndarray
    gate: np.ndarray
    gated_value: np.ndarray
    conv_output: np.ndarray
    state: PLEConvState


@dataclass(frozen=True)
class QSAPooledKeys:
    keys: np.ndarray
    block_starts: np.ndarray
    member_indices: np.ndarray
    tail_indices: np.ndarray


@dataclass(frozen=True)
class QSASelection:
    selected_block_starts: tuple[np.ndarray, ...]
    selected_positions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Qwen4ExpGRWeights:
    norm: ArrayLike
    down: ArrayLike
    up: ArrayLike
    inject: ArrayLike


@dataclass(frozen=True)
class Qwen4ExpQSAWeights:
    q: ArrayLike
    k: ArrayLike
    v: ArrayLike
    output: ArrayLike
    q_norm: ArrayLike
    k_norm: ArrayLike
    index_q: ArrayLike
    index_k: ArrayLike
    index_q_norm: ArrayLike
    index_k_norm: ArrayLike
    query_heads: int
    kv_heads: int
    head_dim: int
    index_heads: int
    index_dim: int


@dataclass(frozen=True)
class Qwen4ExpMoEWeights:
    router: ArrayLike
    expert_gate: ArrayLike
    expert_up: ArrayLike
    expert_down: ArrayLike
    shared_gate: ArrayLike
    shared_up: ArrayLike
    shared_down: ArrayLike
    shared_gate_weight: ArrayLike
    experts_used: int


@dataclass(frozen=True)
class Qwen4ExpReducedLayerWeights:
    attention_gr: Qwen4ExpGRWeights
    qsa: Qwen4ExpQSAWeights
    ffn_gr: Qwen4ExpGRWeights
    moe: Qwen4ExpMoEWeights


@dataclass(frozen=True)
class Qwen4ExpMoEResult:
    output: np.ndarray
    selected_experts: np.ndarray
    routing_weights: np.ndarray


@dataclass(frozen=True)
class Qwen4ExpReducedLayerResult:
    residual: np.ndarray
    attention_output: np.ndarray
    selection: QSASelection
    moe: Qwen4ExpMoEResult


def grouped_zero_centered_rmsnorm(
    residual: ArrayLike,
    weight: ArrayLike,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize each widened residual branch and apply converter-folded gamma."""

    value = np.asarray(residual, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("residual must have shape [tokens, branches, hidden]")
    _, branches, hidden = value.shape
    gamma = np.asarray(weight, dtype=np.float32)
    if gamma.shape == (branches * hidden,):
        gamma = gamma.reshape(branches, hidden)
    if gamma.shape != (branches, hidden):
        raise ValueError("weight must have shape [branches, hidden] or [branches * hidden]")
    variance = np.mean(value * value, axis=-1, keepdims=True, dtype=np.float32)
    normalized = value * np.reciprocal(np.sqrt(variance + np.float32(eps)))
    return (normalized * gamma[None, :, :]).astype(np.float32)


def gr_read(
    residual: ArrayLike,
    norm_weight: ArrayLike,
    down_weight: ArrayLike,
    up_weight: ArrayLike,
    inject_weight: ArrayLike,
    *,
    eps: float = 1e-6,
) -> GRReadResult:
    """Reference Qwen4Exp gated-residual grouped read and injection logits."""

    value = np.asarray(residual, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("residual must have shape [tokens, branches, hidden]")
    tokens, branches, hidden = value.shape
    residual_width = branches * hidden
    down = np.asarray(down_weight, dtype=np.float32)
    up = np.asarray(up_weight, dtype=np.float32)
    inject = np.asarray(inject_weight, dtype=np.float32)
    if down.ndim != 2 or down.shape[1] != residual_width:
        raise ValueError("down_weight must have shape [low_rank, branches * hidden]")
    low_rank = down.shape[0]
    if up.shape != (residual_width, low_rank):
        raise ValueError("up_weight must have shape [branches * hidden, low_rank]")
    if inject.shape != (branches, residual_width):
        raise ValueError("inject_weight must have shape [branches, branches * hidden]")

    normalized = grouped_zero_centered_rmsnorm(value, norm_weight, eps=eps)
    flat = normalized.reshape(tokens, residual_width)
    low = (flat @ down.T).astype(np.float32) / np.float32(branches)
    low = _silu(low)
    gate_flat = _sigmoid((low @ up.T).astype(np.float32))
    gate = gate_flat.reshape(tokens, branches, hidden)
    mixed = np.mean((normalized * gate).astype(np.float32), axis=1, dtype=np.float32)
    inject_logits = (flat @ inject.T).astype(np.float32)
    return GRReadResult(normalized, gate, mixed.astype(np.float32), inject_logits)


def gr_write(
    residual: ArrayLike,
    block_output: ArrayLike,
    inject_logits: ArrayLike,
) -> np.ndarray:
    """Scatter one block output into all widened residual branches."""

    value = np.asarray(residual, dtype=np.float32)
    block = np.asarray(block_output, dtype=np.float32)
    inject = np.asarray(inject_logits, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("residual must have shape [tokens, branches, hidden]")
    tokens, branches, hidden = value.shape
    if block.shape != (tokens, hidden):
        raise ValueError("block_output must have shape [tokens, hidden]")
    if inject.shape != (tokens, branches):
        raise ValueError("inject_logits must have shape [tokens, branches]")
    scatter = np.float32(2.0) * _sigmoid(inject / np.float32(branches))
    return (value + block[:, None, :] * scatter[:, :, None]).astype(np.float32)


def sigmoid_gated_rmsnorm(
    value: ArrayLike,
    weight: ArrayLike,
    gate_logits: ArrayLike,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Qwen4Exp GDN output norm with sigmoid, not Qwen3.5 SiLU, gating."""

    x = np.asarray(value, dtype=np.float32)
    gate = np.asarray(gate_logits, dtype=np.float32)
    gamma = np.asarray(weight, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError("value must have at least two dimensions")
    if gate.shape != x.shape:
        raise ValueError("gate_logits must match value shape")
    if gamma.shape != (x.shape[-1],):
        raise ValueError("weight must have shape [head_dim]")
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    normalized = x * np.reciprocal(np.sqrt(variance + np.float32(eps)))
    return (normalized * gamma * _sigmoid(gate)).astype(np.float32)


def ple_hash_rows(
    token_ids: ArrayLike,
    *,
    positions: ArrayLike,
    sequence_ids: ArrayLike,
    states: Mapping[int, PLEHashState],
    eos_token_id: int,
    layer_multipliers: Sequence[int],
    head_offsets: Sequence[int],
    head_vocab_sizes: Sequence[int],
    heads_per_ngram: int,
    ngram_size: int,
) -> tuple[np.ndarray, dict[int, PLEHashState]]:
    """Hash PLE bigram/trigram rows with uint64 wrap and request-local history."""

    tokens = np.asarray(token_ids, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    seq = np.asarray(sequence_ids, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError("token_ids must have shape [tokens]")
    if pos.shape != tokens.shape:
        raise ValueError("positions must have shape [tokens]")
    if seq.shape != tokens.shape:
        raise ValueError("sequence_ids must have shape [tokens]")
    if np.any(tokens < 0):
        raise ValueError("token_ids must be non-negative")
    if np.any(pos < 0):
        raise ValueError("positions must be non-negative")
    ngram = int(ngram_size)
    per_ngram = int(heads_per_ngram)
    if ngram < 2 or per_ngram <= 0:
        raise ValueError("ngram_size must be >= 2 and heads_per_ngram must be positive")
    multipliers = tuple(int(value) for value in layer_multipliers)
    offsets = tuple(int(value) for value in head_offsets)
    sizes = tuple(int(value) for value in head_vocab_sizes)
    head_count = (ngram - 1) * per_ngram
    if len(multipliers) < ngram:
        raise ValueError("layer_multipliers must cover ngram_size")
    if len(offsets) != head_count or len(sizes) != head_count:
        raise ValueError("head offsets/sizes must cover every n-gram head")
    if any(size <= 0 for size in sizes):
        raise ValueError("head_vocab_sizes must be positive")

    eos = int(eos_token_id)
    output_states = {int(key): value for key, value in states.items()}
    snapshots: dict[int, PLEHashState] = {}
    first_positions: dict[int, int] = {}
    seen_positions: dict[int, list[int]] = {}
    prior_batch_tokens: dict[tuple[int, int], int] = {}
    for index in range(tokens.size):
        sequence = int(seq[index])
        position = int(pos[index])
        prior_positions = seen_positions.setdefault(sequence, [])
        if prior_positions and position != prior_positions[-1] + 1:
            raise ValueError("positions must be contiguous within each sequence")
        prior_positions.append(position)
        if sequence in snapshots:
            continue
        state = output_states.get(
            sequence,
            PLEHashState(tokens=(eos,) * (ngram - 1), next_position=position),
        )
        if len(state.tokens) != ngram - 1:
            raise ValueError("PLEHashState tokens must have ngram_size - 1 entries")
        if state.next_position != position:
            state = PLEHashState(tokens=(eos,) * (ngram - 1), next_position=position)
        snapshots[sequence] = state
        first_positions[sequence] = position
        output_states[sequence] = state

    rows = np.empty((tokens.size, head_count), dtype=np.int64)
    for index in range(tokens.size):
        sequence = int(seq[index])
        position = int(pos[index])
        current = int(tokens[index])
        snapshot = snapshots[sequence]
        first_position = first_positions[sequence]

        def predecessor(distance: int) -> int:
            target = position - distance
            batch_value = prior_batch_tokens.get((sequence, target))
            if batch_value is not None:
                return batch_value
            history_start = first_position - len(snapshot.tokens)
            history_index = target - history_start
            if 0 <= history_index < len(snapshot.tokens) and target >= 0:
                return int(snapshot.tokens[history_index])
            return eos

        context = [current]
        cut = False
        for distance in range(1, ngram):
            previous = eos if cut else predecessor(distance)
            context.append(previous)
            if previous == eos:
                cut = True

        for order in range(2, ngram + 1):
            mixed = (context[0] * multipliers[0]) & _UINT64_MASK
            for context_index in range(1, order):
                mixed ^= (context[context_index] * multipliers[context_index]) & _UINT64_MASK
            base = (order - 2) * per_ngram
            for head in range(per_ngram):
                head_index = base + head
                rows[index, head_index] = mixed % sizes[head_index] + offsets[head_index]

        prior_batch_tokens[(sequence, position)] = current
        previous_state = output_states[sequence]
        history = (*previous_state.tokens, current)[-(ngram - 1) :]
        output_states[sequence] = PLEHashState(
            tokens=tuple(int(value) for value in history),
            next_position=position + 1,
        )
    return rows, output_states


def ple_signed_sqrt_gate(scores: ArrayLike) -> np.ndarray:
    """Apply PLE's signed-square-root sigmoid gate."""

    value = np.asarray(scores, dtype=np.float32)
    transformed = np.sign(value) * np.sqrt(
        np.maximum(np.abs(value), np.float32(1e-6))
    )
    return _sigmoid(transformed.astype(np.float32))


def dilated_depthwise_conv(
    values: ArrayLike,
    kernel: ArrayLike,
    *,
    dilation: int,
    positions: ArrayLike,
    state: PLEConvState | None,
) -> tuple[np.ndarray, PLEConvState]:
    """Causal per-channel convolution with explicit immutable history."""

    x = np.asarray(values, dtype=np.float32)
    weights = np.asarray(kernel, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("values must have shape [tokens, channels]")
    tokens, channels = x.shape
    if tokens == 0:
        raise ValueError("values must contain at least one token")
    if weights.ndim != 2 or weights.shape[0] != channels:
        raise ValueError("kernel must have shape [channels, kernel_size]")
    kernel_size = weights.shape[1]
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    dil = int(dilation)
    if dil <= 0:
        raise ValueError("dilation must be positive")
    if pos.shape != (tokens,):
        raise ValueError("positions must have shape [tokens]")
    if np.any(pos < 0) or np.any(np.diff(pos) != 1):
        raise ValueError("positions must be non-negative and contiguous")

    history_rows = (kernel_size - 1) * dil
    if state is None or state.next_position != int(pos[0]):
        history = np.zeros((history_rows, channels), dtype=np.float32)
    else:
        history = np.asarray(state.history, dtype=np.float32)
        if history.shape != (history_rows, channels):
            raise ValueError("state history shape does not match kernel/dilation/channels")
        history = history.copy()
    padded = np.concatenate((history, x), axis=0)
    output = np.zeros_like(x, dtype=np.float32)
    for tap in range(kernel_size):
        start = history_rows - (kernel_size - 1 - tap) * dil
        output += padded[start : start + tokens] * weights[:, tap][None, :]
    next_history = (
        padded[-history_rows:].copy()
        if history_rows
        else np.zeros((0, channels), dtype=np.float32)
    )
    return output, PLEConvState(next_history, int(pos[-1]) + 1)


def ple_injection(
    residual: ArrayLike,
    embedding: ArrayLike,
    key_weight: ArrayLike,
    value_weight: ArrayLike,
    norm_key_weight: ArrayLike,
    norm_query_weight: ArrayLike,
    norm_conv_weight: ArrayLike,
    conv_kernel: ArrayLike,
    *,
    positions: ArrayLike,
    state: PLEConvState | None,
    dilation: int,
    eps: float = 1e-6,
) -> PLEInjectionResult:
    """Reference PLE projections, branch gate, dilated Conv, and residual update."""

    hidden_state = np.asarray(residual, dtype=np.float32)
    emb = np.asarray(embedding, dtype=np.float32)
    if hidden_state.ndim != 3:
        raise ValueError("residual must have shape [tokens, branches, hidden]")
    tokens, branches, hidden = hidden_state.shape
    if emb.shape != (tokens, hidden):
        raise ValueError("embedding must have shape [tokens, hidden]")
    residual_width = branches * hidden
    key_projection = np.asarray(key_weight, dtype=np.float32)
    value_projection = np.asarray(value_weight, dtype=np.float32)
    if key_projection.shape != (residual_width, hidden):
        raise ValueError("key_weight must have shape [branches * hidden, hidden]")
    if value_projection.shape != (hidden, hidden):
        raise ValueError("value_weight must have shape [hidden, hidden]")

    key = (emb @ key_projection.T).reshape(tokens, branches, hidden).astype(np.float32)
    value = (emb @ value_projection.T).astype(np.float32)
    key = grouped_zero_centered_rmsnorm(key, norm_key_weight, eps=eps)
    query = grouped_zero_centered_rmsnorm(
        hidden_state,
        norm_query_weight,
        eps=eps,
    )
    score = np.sum(key * query, axis=-1, dtype=np.float32) / np.float32(np.sqrt(hidden))
    gate = ple_signed_sqrt_gate(score)
    gated = (value[:, None, :] * gate[:, :, None]).astype(np.float32)
    normalized = grouped_zero_centered_rmsnorm(
        gated,
        norm_conv_weight,
        eps=eps,
    ).reshape(tokens, residual_width)
    conv_raw, next_state = dilated_depthwise_conv(
        normalized,
        conv_kernel,
        dilation=dilation,
        positions=positions,
        state=state,
    )
    conv_output = _silu(conv_raw).reshape(tokens, branches, hidden)
    updated = (hidden_state + gated + conv_output).astype(np.float32)
    return PLEInjectionResult(updated, gate, gated, conv_output, next_state)


def qsa_pool_complete_blocks(
    raw_keys: ArrayLike,
    token_positions: ArrayLike,
    *,
    compression_ratio: int,
) -> QSAPooledKeys:
    """Mean-pool complete logical QSA blocks while retaining physical owners."""

    keys = np.asarray(raw_keys, dtype=np.float32)
    positions = np.asarray(token_positions, dtype=np.int64)
    if keys.ndim != 2:
        raise ValueError("raw_keys must have shape [tokens, index_dim]")
    if positions.shape != (keys.shape[0],):
        raise ValueError("token_positions must have shape [tokens]")
    if positions.size == 0 or np.any(positions < 0):
        raise ValueError("token_positions must be non-empty and non-negative")
    if np.unique(positions).size != positions.size:
        raise ValueError("token_positions must be unique")
    ratio = int(compression_ratio)
    if ratio <= 0:
        raise ValueError("compression_ratio must be positive")

    by_block: dict[int, list[tuple[int, int]]] = {}
    for physical, position in enumerate(positions.tolist()):
        by_block.setdefault(position // ratio, []).append((position, physical))
    highest_block = int(np.max(positions)) // ratio
    block_starts: list[int] = []
    members: list[list[int]] = []
    tail: list[int] = []
    for block_id in sorted(by_block):
        entries = sorted(by_block[block_id])
        expected = list(range(block_id * ratio, (block_id + 1) * ratio))
        logical = [position for position, _ in entries]
        if logical == expected:
            block_starts.append(block_id * ratio)
            members.append([physical for _, physical in entries])
            continue
        if block_id != highest_block:
            raise ValueError(f"incomplete non-tail QSA block at {block_id * ratio}")
        expected_tail = list(range(block_id * ratio, int(np.max(positions)) + 1))
        if logical != expected_tail:
            raise ValueError("incomplete QSA tail contains logical holes")
        tail = [physical for _, physical in entries]

    index_dim = keys.shape[1]
    pooled = np.empty((len(members), index_dim), dtype=np.float32)
    for block_index, physical_members in enumerate(members):
        pooled[block_index] = np.mean(
            keys[np.asarray(physical_members, dtype=np.int64)],
            axis=0,
            dtype=np.float32,
        )
    member_array = (
        np.asarray(members, dtype=np.int64)
        if members
        else np.empty((0, ratio), dtype=np.int64)
    )
    return QSAPooledKeys(
        pooled,
        np.asarray(block_starts, dtype=np.int64),
        member_array,
        np.asarray(tail, dtype=np.int64),
    )


def qsa_interleaved_rope(
    values: ArrayLike,
    *,
    positions: ArrayLike,
    rotary_dim: int,
    theta: float,
) -> np.ndarray:
    """Apply text-path pair-interleaved partial RoPE to QSA Q/K values."""

    x = np.asarray(values, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    if x.ndim < 2:
        raise ValueError("values must have shape [tokens, ..., head_dim]")
    if pos.shape != (x.shape[0],) or np.any(pos < 0):
        raise ValueError("positions must be non-negative with shape [tokens]")
    rotate = int(rotary_dim)
    if rotate <= 0 or rotate > x.shape[-1] or rotate % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    theta_value = float(theta)
    if not np.isfinite(theta_value) or theta_value <= 0.0:
        raise ValueError("theta must be positive and finite")
    dimensions = np.arange(0, rotate, 2, dtype=np.float32)
    inverse_frequency = np.reciprocal(
        np.power(np.float32(theta_value), dimensions / np.float32(rotate))
    ).astype(np.float32)
    angles = pos.astype(np.float32)[:, None] * inverse_frequency[None, :]
    table_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (rotate // 2,)
    cosine = np.cos(angles).astype(np.float32).reshape(table_shape)
    sine = np.sin(angles).astype(np.float32).reshape(table_shape)
    pairs = x[..., :rotate].reshape(*x.shape[:-1], rotate // 2, 2)
    rotated = np.empty_like(pairs, dtype=np.float32)
    rotated[..., 0] = pairs[..., 0] * cosine - pairs[..., 1] * sine
    rotated[..., 1] = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return np.concatenate(
        (rotated.reshape(*x.shape[:-1], rotate), x[..., rotate:]),
        axis=-1,
    ).astype(np.float32)


def qsa_prepare_index_keys(
    raw_keys: ArrayLike,
    token_positions: ArrayLike,
    norm_weight: ArrayLike,
    *,
    compression_ratio: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
) -> QSAPooledKeys:
    """Pool raw keys in FP32, then RMS-normalize and rotate at block starts."""

    pooled = qsa_pool_complete_blocks(
        raw_keys,
        token_positions,
        compression_ratio=compression_ratio,
    )
    gamma = np.asarray(norm_weight, dtype=np.float32)
    if gamma.shape != (pooled.keys.shape[1],):
        raise ValueError("norm_weight must have shape [index_dim]")
    if pooled.keys.shape[0]:
        variance = np.mean(
            pooled.keys * pooled.keys,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        normalized = (
            pooled.keys * np.reciprocal(np.sqrt(variance + np.float32(eps))) * gamma
        ).astype(np.float32)
        prepared = qsa_interleaved_rope(
            normalized[:, None, :],
            positions=pooled.block_starts,
            rotary_dim=rotary_dim,
            theta=theta,
        )[:, 0, :]
    else:
        prepared = pooled.keys.copy()
    return QSAPooledKeys(
        prepared,
        pooled.block_starts,
        pooled.member_indices,
        pooled.tail_indices,
    )


def qsa_index_scores(queries: ArrayLike, pooled_keys: ArrayLike) -> np.ndarray:
    """Rectify each QSA index-head dot product before summing heads."""

    query = np.asarray(queries, dtype=np.float32)
    keys = np.asarray(pooled_keys, dtype=np.float32)
    if query.ndim != 3:
        raise ValueError("queries must have shape [queries, heads, index_dim]")
    if keys.ndim != 2 or keys.shape[1] != query.shape[2]:
        raise ValueError("pooled_keys must have shape [blocks, index_dim]")
    dots = np.einsum("qhd,bd->qhb", query, keys, dtype=np.float32)
    scores = np.sum(np.maximum(dots, np.float32(0.0)), axis=1, dtype=np.float32)
    return (scores / np.float32(np.sqrt(query.shape[2]))).astype(np.float32)


def qsa_select_positions(
    scores: ArrayLike,
    block_starts: ArrayLike,
    *,
    query_positions: ArrayLike,
    available_positions: ArrayLike,
    compression_ratio: int,
    block_budget: int,
) -> QSASelection:
    """Select complete blocks deterministically and append the incomplete tail."""

    score = np.asarray(scores, dtype=np.float32)
    starts = np.asarray(block_starts, dtype=np.int64)
    queries = np.asarray(query_positions, dtype=np.int64)
    available = np.asarray(available_positions, dtype=np.int64)
    if score.ndim != 2 or score.shape[1] != starts.size:
        raise ValueError("scores must have shape [queries, blocks]")
    if queries.shape != (score.shape[0],):
        raise ValueError("query_positions must have shape [queries]")
    if np.unique(available).size != available.size or np.any(available < 0):
        raise ValueError("available_positions must be unique and non-negative")
    ratio = int(compression_ratio)
    budget = int(block_budget)
    if ratio <= 0 or budget <= 0:
        raise ValueError("compression_ratio and block_budget must be positive")
    if starts.size and (np.any(starts < 0) or np.any(starts % ratio != 0)):
        raise ValueError("block_starts must be non-negative and ratio-aligned")
    available_set = set(int(value) for value in available.tolist())

    selected_starts: list[np.ndarray] = []
    selected_positions: list[np.ndarray] = []
    for row, query_position in enumerate(queries.tolist()):
        eligible = np.nonzero(starts + ratio - 1 <= query_position)[0]
        ranking = np.lexsort((starts[eligible], -score[row, eligible]))
        chosen = eligible[ranking[:budget]]
        logical_starts = np.sort(starts[chosen]).astype(np.int64)
        logical_positions: list[int] = []
        for start in logical_starts.tolist():
            logical_positions.extend(range(start, start + ratio))
        if query_position % ratio != ratio - 1:
            tail_start = query_position // ratio * ratio
            logical_positions.extend(range(tail_start, query_position + 1))
        logical_positions = sorted(set(logical_positions))
        missing = [position for position in logical_positions if position not in available_set]
        if missing:
            raise ValueError(f"selected QSA positions are unavailable: {missing[:8]}")
        selected_starts.append(logical_starts)
        selected_positions.append(np.asarray(logical_positions, dtype=np.int64))
    return QSASelection(tuple(selected_starts), tuple(selected_positions))


def qsa_sparse_gqa_attention(
    queries: ArrayLike,
    keys: ArrayLike,
    values: ArrayLike,
    *,
    query_positions: ArrayLike,
    key_positions: ArrayLike,
    selected_positions: Sequence[ArrayLike],
    scale: float | None = None,
) -> np.ndarray:
    """Attend selected logical positions using original uncompressed GQA K/V."""

    query = np.asarray(queries, dtype=np.float32)
    key = np.asarray(keys, dtype=np.float32)
    value = np.asarray(values, dtype=np.float32)
    qpos = np.asarray(query_positions, dtype=np.int64)
    kpos = np.asarray(key_positions, dtype=np.int64)
    if query.ndim != 3:
        raise ValueError("queries must have shape [queries, query_heads, head_dim]")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("keys and values must have shape [tokens, kv_heads, head_dim]")
    if key.shape[:2] != value.shape[:2] or key.shape[2] != query.shape[2]:
        raise ValueError("query/key/value head geometry is incompatible")
    if qpos.shape != (query.shape[0],) or kpos.shape != (key.shape[0],):
        raise ValueError("query_positions/key_positions have incompatible shapes")
    if len(selected_positions) != query.shape[0]:
        raise ValueError("selected_positions must contain one row per query")
    if np.unique(kpos).size != kpos.size:
        raise ValueError("key_positions must be unique")
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    group_size = query_heads // kv_heads
    attention_scale = (
        np.float32(1.0 / np.sqrt(query.shape[2]))
        if scale is None
        else np.float32(scale)
    )
    physical_by_position = {int(position): index for index, position in enumerate(kpos)}
    output = np.empty(
        (query.shape[0], query_heads, value.shape[2]),
        dtype=np.float32,
    )
    for row, selected in enumerate(selected_positions):
        logical = np.asarray(selected, dtype=np.int64)
        if logical.ndim != 1 or logical.size == 0:
            raise ValueError("each selected_positions row must be non-empty")
        if np.any(logical > qpos[row]):
            raise ValueError("selected_positions cannot include future tokens")
        try:
            physical = np.asarray(
                [physical_by_position[int(position)] for position in logical],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(f"selected key position {exc.args[0]} is unavailable") from exc
        for head in range(query_heads):
            kv_head = head // group_size
            logits = (
                key[physical, kv_head, :] @ query[row, head, :]
            ).astype(np.float32) * attention_scale
            probabilities = _softmax(logits)
            output[row, head, :] = probabilities @ value[physical, kv_head, :]
    return output


def qwen4_exp_moe(
    hidden: ArrayLike,
    weights: Qwen4ExpMoEWeights,
) -> Qwen4ExpMoEResult:
    """Reduced dense-weight oracle for normalized top-k plus shared-expert MoE."""

    x = np.asarray(hidden, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    router = np.asarray(weights.router, dtype=np.float32)
    if router.ndim != 2 or router.shape[1] != hidden_size:
        raise ValueError("router must have shape [experts, hidden]")
    experts = router.shape[0]
    used = int(weights.experts_used)
    if used <= 0 or used > experts:
        raise ValueError("experts_used must be in 1..experts")
    gate_weights = np.asarray(weights.expert_gate, dtype=np.float32)
    up_weights = np.asarray(weights.expert_up, dtype=np.float32)
    down_weights = np.asarray(weights.expert_down, dtype=np.float32)
    if gate_weights.ndim != 3 or gate_weights.shape[0] != experts:
        raise ValueError("expert_gate must have shape [experts, ffn, hidden]")
    ffn = gate_weights.shape[1]
    if gate_weights.shape[2] != hidden_size or up_weights.shape != gate_weights.shape:
        raise ValueError("expert gate/up geometry is incompatible")
    if down_weights.shape != (experts, hidden_size, ffn):
        raise ValueError("expert_down must have shape [experts, hidden, ffn]")

    probabilities = _softmax_rows((x @ router.T).astype(np.float32))
    selected = np.argsort(-probabilities, axis=-1, kind="stable")[:, :used]
    routing = np.take_along_axis(probabilities, selected, axis=-1).astype(np.float32)
    routing /= np.sum(routing, axis=-1, keepdims=True, dtype=np.float32)
    routed = np.zeros((tokens, hidden_size), dtype=np.float32)
    for token in range(tokens):
        for slot in range(used):
            expert = int(selected[token, slot])
            gate = gate_weights[expert] @ x[token]
            up = up_weights[expert] @ x[token]
            activated = _silu(gate.astype(np.float32)) * up
            down = down_weights[expert] @ activated
            routed[token] += routing[token, slot] * down

    shared_gate = np.asarray(weights.shared_gate, dtype=np.float32)
    shared_up = np.asarray(weights.shared_up, dtype=np.float32)
    shared_down = np.asarray(weights.shared_down, dtype=np.float32)
    shared_scalar = np.asarray(weights.shared_gate_weight, dtype=np.float32)
    if shared_gate.shape != (ffn, hidden_size) or shared_up.shape != shared_gate.shape:
        raise ValueError("shared gate/up must have shape [ffn, hidden]")
    if shared_down.shape != (hidden_size, ffn):
        raise ValueError("shared_down must have shape [hidden, ffn]")
    if shared_scalar.shape != (hidden_size,):
        raise ValueError("shared_gate_weight must have shape [hidden]")
    shared = _silu((x @ shared_gate.T).astype(np.float32))
    shared *= (x @ shared_up.T).astype(np.float32)
    shared = (shared @ shared_down.T).astype(np.float32)
    scalar = _sigmoid((x @ shared_scalar).astype(np.float32))[:, None]
    output = (routed + scalar * shared).astype(np.float32)
    return Qwen4ExpMoEResult(output, selected.astype(np.int64), routing)


def qwen4_exp_reduced_qsa_layer(
    residual: ArrayLike,
    weights: Qwen4ExpReducedLayerWeights,
    *,
    positions: ArrayLike,
    compression_ratio: int,
    block_budget: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
) -> Qwen4ExpReducedLayerResult:
    """Compose one reduced GR -> QSA -> GR -> MoE Qwen4Exp layer."""

    state = np.asarray(residual, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    if state.ndim != 3:
        raise ValueError("residual must have shape [tokens, branches, hidden]")
    tokens, _, hidden = state.shape
    if pos.shape != (tokens,) or np.any(pos < 0) or np.any(np.diff(pos) != 1):
        raise ValueError("positions must be non-negative, contiguous, and match tokens")
    qsa = weights.qsa
    query_heads = int(qsa.query_heads)
    kv_heads = int(qsa.kv_heads)
    head_dim = int(qsa.head_dim)
    index_heads = int(qsa.index_heads)
    index_dim = int(qsa.index_dim)
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("QSA query_heads must be divisible by positive kv_heads")

    attention_read = gr_read(
        state,
        weights.attention_gr.norm,
        weights.attention_gr.down,
        weights.attention_gr.up,
        weights.attention_gr.inject,
        eps=eps,
    )
    mixed = attention_read.mixed
    q_weight = np.asarray(qsa.q, dtype=np.float32)
    k_weight = np.asarray(qsa.k, dtype=np.float32)
    v_weight = np.asarray(qsa.v, dtype=np.float32)
    output_weight = np.asarray(qsa.output, dtype=np.float32)
    if q_weight.shape != (query_heads * 2 * head_dim, hidden):
        raise ValueError("q weight has incompatible Q+gate geometry")
    if k_weight.shape != (kv_heads * head_dim, hidden):
        raise ValueError("k weight has incompatible geometry")
    if v_weight.shape != (kv_heads * head_dim, hidden):
        raise ValueError("v weight has incompatible geometry")
    if output_weight.shape != (hidden, query_heads * head_dim):
        raise ValueError("output weight has incompatible geometry")

    q_and_gate = (mixed @ q_weight.T).reshape(tokens, query_heads, 2, head_dim)
    query = _rmsnorm_last(q_and_gate[:, :, 0, :], qsa.q_norm, eps=eps)
    query = qsa_interleaved_rope(
        query,
        positions=pos,
        rotary_dim=rotary_dim,
        theta=theta,
    )
    query_gate = q_and_gate[:, :, 1, :]
    key = _rmsnorm_last(
        (mixed @ k_weight.T).reshape(tokens, kv_heads, head_dim),
        qsa.k_norm,
        eps=eps,
    )
    key = qsa_interleaved_rope(
        key,
        positions=pos,
        rotary_dim=rotary_dim,
        theta=theta,
    )
    value = (mixed @ v_weight.T).reshape(tokens, kv_heads, head_dim).astype(np.float32)

    index_q_weight = np.asarray(qsa.index_q, dtype=np.float32)
    index_k_weight = np.asarray(qsa.index_k, dtype=np.float32)
    if index_q_weight.shape != (index_heads * index_dim, hidden):
        raise ValueError("index_q weight has incompatible geometry")
    if index_k_weight.shape != (index_dim, hidden):
        raise ValueError("index_k weight has incompatible geometry")
    index_query = _rmsnorm_last(
        (mixed @ index_q_weight.T).reshape(tokens, index_heads, index_dim),
        qsa.index_q_norm,
        eps=eps,
    )
    index_query = qsa_interleaved_rope(
        index_query,
        positions=pos,
        rotary_dim=min(rotary_dim, index_dim),
        theta=theta,
    )
    raw_index_key = (mixed @ index_k_weight.T).astype(np.float32)
    prepared = qsa_prepare_index_keys(
        raw_index_key,
        pos,
        qsa.index_k_norm,
        compression_ratio=compression_ratio,
        rotary_dim=min(rotary_dim, index_dim),
        theta=theta,
        eps=eps,
    )
    scores = qsa_index_scores(index_query, prepared.keys)
    selection = qsa_select_positions(
        scores,
        prepared.block_starts,
        query_positions=pos,
        available_positions=pos,
        compression_ratio=compression_ratio,
        block_budget=block_budget,
    )
    context = qsa_sparse_gqa_attention(
        query,
        key,
        value,
        query_positions=pos,
        key_positions=pos,
        selected_positions=selection.selected_positions,
    )
    gated_context = (context * _sigmoid(query_gate)).reshape(
        tokens,
        query_heads * head_dim,
    )
    attention_output = (gated_context @ output_weight.T).astype(np.float32)
    after_attention = gr_write(state, attention_output, attention_read.inject_logits)

    ffn_read = gr_read(
        after_attention,
        weights.ffn_gr.norm,
        weights.ffn_gr.down,
        weights.ffn_gr.up,
        weights.ffn_gr.inject,
        eps=eps,
    )
    moe = qwen4_exp_moe(ffn_read.mixed, weights.moe)
    output_state = gr_write(after_attention, moe.output, ffn_read.inject_logits)
    return Qwen4ExpReducedLayerResult(output_state, attention_output, selection, moe)


def _rmsnorm_last(value: ArrayLike, weight: ArrayLike, *, eps: float) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    gamma = np.asarray(weight, dtype=np.float32)
    if gamma.shape != (x.shape[-1],):
        raise ValueError("RMSNorm weight must match the last dimension")
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (
        x * np.reciprocal(np.sqrt(variance + np.float32(eps))) * gamma
    ).astype(np.float32)


def _softmax_rows(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponent = np.exp(shifted).astype(np.float32)
    return (
        exponent / np.sum(exponent, axis=-1, keepdims=True, dtype=np.float32)
    ).astype(np.float32)


def _softmax(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    shifted = x - np.max(x)
    exponent = np.exp(shifted).astype(np.float32)
    return (exponent / np.sum(exponent, dtype=np.float32)).astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return np.reciprocal(np.float32(1.0) + np.exp(-x)).astype(np.float32)


def _silu(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return (x * _sigmoid(x)).astype(np.float32)


__all__ = [
    "GRReadResult",
    "PLEConvState",
    "PLEHashState",
    "PLEInjectionResult",
    "QSAPooledKeys",
    "QSASelection",
    "Qwen4ExpGRWeights",
    "Qwen4ExpMoEResult",
    "Qwen4ExpMoEWeights",
    "Qwen4ExpQSAWeights",
    "Qwen4ExpReducedLayerResult",
    "Qwen4ExpReducedLayerWeights",
    "dilated_depthwise_conv",
    "gr_read",
    "gr_write",
    "grouped_zero_centered_rmsnorm",
    "ple_hash_rows",
    "ple_injection",
    "ple_signed_sqrt_gate",
    "qsa_index_scores",
    "qsa_interleaved_rope",
    "qsa_pool_complete_blocks",
    "qsa_prepare_index_keys",
    "qsa_select_positions",
    "qsa_sparse_gqa_attention",
    "qwen4_exp_moe",
    "qwen4_exp_reduced_qsa_layer",
    "sigmoid_gated_rmsnorm",
]
