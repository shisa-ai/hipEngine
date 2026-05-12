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


def register_cpu_reference_kernels(*, replace: bool = True) -> None:
    """Register the first CPU-reference primitive set under fp16 keys."""

    kernels = {
        "embed": embed,
        "rmsnorm": rmsnorm,
        "linear": linear,
        "qkv_proj": qkv_proj,
        "rotate": rotate,
        "attention_decode": attention_decode,
        "o_proj": o_proj,
        "lm_head": lm_head,
    }
    for layer, fn in kernels.items():
        register(KernelKey("cpu_reference", layer, "fp16"), fn, replace=replace)


def _half_rotary_table(value: ArrayLike, half: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] == half:
        return arr
    if arr.shape[-1] == half * 2:
        return arr[..., :half]
    raise ValueError(f"{name}.shape[-1] must be {half} or {half * 2}, got {arr.shape[-1]}")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    finite = np.isfinite(x)
    safe_x = np.where(finite, x, -np.inf)
    max_x = np.max(safe_x, axis=axis, keepdims=True)
    shifted = safe_x - max_x
    exp = np.where(finite, np.exp(shifted), 0.0)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom
