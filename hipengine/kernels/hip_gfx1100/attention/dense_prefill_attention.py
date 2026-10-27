"""Causal prefill attention for dense Llama-family decoders.

Geometry: Q and out are ``(tokens, num_heads, head_dim)`` BF16, K/V are
``(tokens, num_kv_heads, head_dim)`` BF16, one contiguous prompt, mask
``key <= query``. Grouped-query attention is the ``num_heads % num_kv_heads``
case, which every dense Llama variant uses.

The executed kernel is the repository's exact masked block attention. The
vendored AOTriton image set on this architecture only carries ``head_dim`` 256
tiles, so the dense 128-wide heads have no flash image to route to; the exact
kernel accepts any head dimension and is the correctness-first owner here. The
tracked follow-up is a native flash kernel for the dense head width, recorded in
``docs/REFACTOR.md``.
"""

from __future__ import annotations

import ctypes

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    Gemma4AttentionScratch,
    gemma4_attention_prefill_bf16,
)
from hipengine.kernels.registry import KernelKey, register


def causal_keep_mask(tokens: int, *, keys: int | None = None) -> np.ndarray:
    """Return the ``key <= query`` mask as a ``(tokens, keys)`` uint8 array."""

    rows = int(tokens)
    key_count = rows if keys is None else int(keys)
    if rows <= 0 or key_count <= 0:
        raise ValueError("tokens and keys must be positive")
    query_positions = np.arange(key_count - rows, key_count, dtype=np.int64)[:, None]
    key_positions = np.arange(key_count, dtype=np.int64)[None, :]
    return np.ascontiguousarray((key_positions <= query_positions).astype(np.uint8))


def dense_causal_prefill_bf16(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    keep_mask_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    keys: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    scratch: Gemma4AttentionScratch | None = None,
) -> None:
    """Run exact causal prefill attention over one contiguous prompt."""

    if int(tokens) <= 0:
        raise ValueError("tokens must be positive")
    if int(num_heads) <= 0 or int(num_kv_heads) <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if int(num_heads) % int(num_kv_heads):
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if int(head_dim) <= 0:
        raise ValueError("head_dim must be positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    gemma4_attention_prefill_bf16(
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        tokens=int(tokens),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        scale=float(scale),
        keys=None if keys is None else int(keys),
        stream=stream,
        library=library,
        runtime=runtime or get_hip_runtime(),
        scratch=scratch,
    )


def register_dense_prefill_attention_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "attention_prefill", "bf16", "exact_causal"),
        dense_causal_prefill_bf16,
        replace=replace,
    )


__all__ = [
    "Gemma4AttentionScratch",
    "causal_keep_mask",
    "dense_causal_prefill_bf16",
    "register_dense_prefill_attention_kernels",
]
