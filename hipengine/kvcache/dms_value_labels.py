"""CPU reference H0-H2 DMS importance labels for bounded fixtures."""
from __future__ import annotations

from math import sqrt
import numpy as np


def _validate_qk(query: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("Q/K must be [tokens,heads,dim]")
    tokens, q_heads, dim = q.shape
    if k.shape[0] != tokens or k.shape[2] != dim or q_heads % k.shape[1]:
        raise ValueError("Q/K geometry is not aligned")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("Q/K must be finite")
    return q, k, tokens, q_heads, int(k.shape[1]), dim


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def continuation_mass_cpu(query: np.ndarray, key: np.ndarray, *, prefix_length: int, window_size: int) -> np.ndarray:
    """H1: mass received by prefix keys from continuation queries only."""
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    if q.ndim != 3 or k.ndim != 3 or k.shape[0] <= 0 or q.shape[1] % k.shape[1] or q.shape[2] != k.shape[2]:
        raise ValueError("continuation Q/K geometry is not aligned")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("Q/K must be finite")
    tokens, q_heads, dim = q.shape
    prefix = int(prefix_length)
    if not 0 < prefix <= k.shape[0] or int(window_size) < 0:
        raise ValueError("invalid prefix/window")
    kv_heads = int(k.shape[1])
    result = np.zeros((prefix, kv_heads), dtype=np.float64)
    groups = q_heads // kv_heads
    for continuation_index in range(tokens):
        absolute_position = prefix + continuation_index
        key_count = min(prefix, absolute_position + 1)
        eligible = np.arange(key_count) < absolute_position - int(window_size)
        if not np.any(eligible):
            continue
        for head in range(kv_heads):
            probs = _softmax((q[continuation_index, head * groups:(head + 1) * groups] @ k[:key_count, head].T) / sqrt(dim))
            result[:key_count, head][eligible] += probs[:, eligible].sum(axis=0)
    return result


def value_perturbation_norm_cpu(query: np.ndarray, key: np.ndarray, value: np.ndarray, *, window_size: int) -> np.ndarray:
    """H2: summed single-key deletion output-change norm by key and KV head."""
    q, k, tokens, q_heads, kv_heads, dim = _validate_qk(query, key)
    v = np.asarray(value, dtype=np.float64)
    if v.shape != k.shape or not np.isfinite(v).all():
        raise ValueError("V must match finite K geometry")
    if int(window_size) < 0:
        raise ValueError("window_size must be non-negative")
    groups = q_heads // kv_heads
    result = np.zeros((tokens, kv_heads), dtype=np.float64)
    for position in range(tokens):
        key_count = position + 1
        for head in range(kv_heads):
            probs = _softmax((q[position, head * groups:(head + 1) * groups] @ k[:key_count, head].T) / sqrt(dim))
            output = probs @ v[:key_count, head]
            eligible = np.arange(key_count) < position - int(window_size)
            for key_index in np.flatnonzero(eligible):
                p = probs[:, key_index]
                denom = np.maximum(1.0 - p, np.finfo(np.float64).tiny)
                delta = (p / denom)[:, None] * (output - v[key_index, head])[None, :]
                result[key_index, head] += float(np.linalg.norm(delta, axis=1).sum())
    return result
