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


def _sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return np.reciprocal(np.float32(1.0) + np.exp(-x)).astype(np.float32)


def _silu(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return (x * _sigmoid(x)).astype(np.float32)


__all__ = [
    "GRReadResult",
    "PLEHashState",
    "gr_read",
    "gr_write",
    "grouped_zero_centered_rmsnorm",
    "ple_hash_rows",
    "sigmoid_gated_rmsnorm",
]
