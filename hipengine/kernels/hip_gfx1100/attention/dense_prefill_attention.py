"""Causal prefill attention for dense Llama-family decoders.

Geometry: Q and out are ``(tokens, num_heads, head_dim)`` BF16, K/V are
``(tokens, num_kv_heads, head_dim)`` BF16, one contiguous prompt, mask
``key <= query``. Grouped-query attention is the ``num_heads % num_kv_heads``
case, which every dense Llama variant uses.

The executed kernel is the tiled flash prefill
(:mod:`hipengine.kernels.hip_gfx1100.attention.dense_flash_prefill`) at
``head_dim`` 128. The exact block attention stays registered and selectable as
the strict fallback: it is the correctness owner for this path and for the DMS
tap contract, and it is what a bisection should reach for. It stages one FP32
logit row per key, so it is bounded by the 64 KiB LDS budget; the flash route
tiles query rows and keys, so its shared memory does not mention the context.
"""

from __future__ import annotations

import ctypes

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.attention.dense_flash_prefill import (
    DenseFlashPrefillUnsupported,
    dense_flash_prefill_bf16,
)
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
    strict: bool = False,
) -> None:
    """Run causal prefill attention over one contiguous prompt.

    ``strict`` selects the exact block kernel, which normalizes once against a
    whole-row maximum. The default is the tiled flash kernel: same BF16 operands
    and FP32 accumulation, but an online softmax that rescales per key slice, so
    its output equals the exact kernel's within BF16 representation rather than
    bit-for-bit. Shapes the flash kernel does not carry fall back to the exact
    kernel, which accepts any head dimension.

    ``keep_mask_ptr`` may be 0 on the flash route, meaning "the causal mask".
    The walk is already bounded by causality, so a null mask skips the reads and
    saves the caller a ``tokens x keys`` byte array. The exact fallback needs a
    real mask, so a null pointer is refused when the geometry falls back.
    """

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

    key_count = int(tokens) if keys is None else int(keys)
    if not strict:
        try:
            dense_flash_prefill_bf16(
                query_ptr,
                key_ptr,
                value_ptr,
                keep_mask_ptr,
                out_ptr,
                tokens=int(tokens),
                keys=key_count,
                num_heads=int(num_heads),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                scale=float(scale),
                stream=stream,
                runtime=runtime or get_hip_runtime(),
            )
            return
        except DenseFlashPrefillUnsupported:
            if int(keep_mask_ptr) == 0:
                raise

    if int(keep_mask_ptr) == 0:
        raise ValueError("the exact attention kernel requires a non-null keep mask")

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
        KernelKey("hip_gfx1100", "attention_prefill", "bf16", "flash_tiled"),
        dense_causal_prefill_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "attention_prefill", "bf16", "exact_causal"),
        lambda *args, **kwargs: dense_causal_prefill_bf16(*args, **kwargs, strict=True),
        replace=replace,
    )


__all__ = [
    "Gemma4AttentionScratch",
    "causal_keep_mask",
    "dense_causal_prefill_bf16",
    "register_dense_prefill_attention_kernels",
]
