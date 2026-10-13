"""WMMA masked prefill attention for Qwen4Exp QSA layers.

Three launches: gather the paged BF16 K/V caches into contiguous F16 rows,
fold per-query selected-position lists into four-token-block bitmasks, and
run the attention itself (F16 operands on the matrix units, F32 score and
output accumulation, online softmax) for a dense causal window or a masked
sparse window. The attention kernel is ported from gufo v0.12.0 (commit
88090afb, MIT License); see the header of ``qwen4_exp_qsa_wmma.hip``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_qsa_wmma.hip")
_FAMILY = "qwen4_exp_qsa_wmma"
_OUTPUT_NAME = "qwen4_exp_qsa_wmma.so"
_SYMBOL_GATHER = "hipengine_qwen4_exp_qsa_wmma_gather_kv_f16"
_SYMBOL_MASK = "hipengine_qwen4_exp_qsa_wmma_selection_mask"
_SYMBOL_ATTENTION = "hipengine_qwen4_exp_qsa_wmma_attention_f32"

QSA_WMMA_VARIANT = "wmma_f16_masked_prefill_f32"
QSA_WMMA_QUERY_HEADS = 24
QSA_WMMA_KV_HEADS = 2
QSA_WMMA_HEAD_DIM = 256
QSA_WMMA_BLOCK_RATIO = 4
QSA_WMMA_MAX_MASK_WORDS = 2048


def qsa_wmma_mask_words(context_tokens: int) -> int:
    """32-bit words covering every four-token block of ``context_tokens``."""

    if context_tokens <= 0:
        raise ValueError("QSA WMMA mask needs a positive context")
    blocks = (context_tokens + QSA_WMMA_BLOCK_RATIO - 1) // QSA_WMMA_BLOCK_RATIO
    return (blocks + 31) // 32


def qsa_wmma_geometry_supported(
    *, query_heads: int, kv_heads: int, head_dim: int, compression_ratio: int,
    context_tokens: int,
) -> bool:
    return (
        query_heads == QSA_WMMA_QUERY_HEADS
        and kv_heads == QSA_WMMA_KV_HEADS
        and head_dim == QSA_WMMA_HEAD_DIM
        and compression_ratio == QSA_WMMA_BLOCK_RATIO
        and 0 < context_tokens
        and qsa_wmma_mask_words(context_tokens) <= QSA_WMMA_MAX_MASK_WORDS
    )


def plan_qwen4_exp_qsa_wmma_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_qsa_wmma(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _call(symbol: str, argtypes, args, library, runtime) -> None:
    lib = library or build_qwen4_exp_qsa_wmma(load=True)
    active = runtime or get_hip_runtime()
    fn = getattr(lib, symbol)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    err = fn(*args)
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{symbol} failed: {active.error_string(err)}")


def qwen4_exp_qsa_wmma_gather_kv_f16(
    key_cache_ptr: int,
    value_cache_ptr: int,
    block_table_ptr: int,
    k_out_ptr: int,
    v_out_ptr: int,
    *,
    block_size: int,
    first_position: int,
    context_len: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Gather positions [first_position, context_len) of the paged BF16 K/V
    caches into contiguous F16 rows of 512 elements at their position."""

    if block_size <= 0 or first_position < 0 or context_len <= first_position:
        raise ValueError("QSA WMMA gather needs a non-empty position range")
    _call(
        _SYMBOL_GATHER,
        [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 3,
        (
            ctypes.c_void_p(key_cache_ptr),
            ctypes.c_void_p(value_cache_ptr),
            ctypes.c_void_p(block_table_ptr),
            ctypes.c_int64(block_size),
            ctypes.c_int64(first_position),
            ctypes.c_int64(context_len),
            ctypes.c_void_p(k_out_ptr),
            ctypes.c_void_p(v_out_ptr),
            ctypes.c_void_p(stream),
        ),
        library,
        runtime,
    )


def qwen4_exp_qsa_wmma_selection_mask(
    selected_positions_ptr: int,
    selected_counts_ptr: int,
    mask_ptr: int,
    *,
    rows: int,
    selected_stride: int,
    mask_words: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """mask[row][w] bit b set when ``row`` selected a position in block 32w+b."""

    if rows <= 0 or selected_stride <= 0 or not 0 < mask_words <= QSA_WMMA_MAX_MASK_WORDS:
        raise ValueError("QSA WMMA selection mask needs positive sizes")
    _call(
        _SYMBOL_MASK,
        [ctypes.c_void_p] * 2 + [ctypes.c_int64, ctypes.c_void_p]
        + [ctypes.c_int64] * 2 + [ctypes.c_void_p],
        (
            ctypes.c_void_p(selected_positions_ptr),
            ctypes.c_void_p(selected_counts_ptr),
            ctypes.c_int64(selected_stride),
            ctypes.c_void_p(mask_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(mask_words),
            ctypes.c_void_p(stream),
        ),
        library,
        runtime,
    )


def qwen4_exp_qsa_wmma_attention_f32(
    query_ptr: int,
    k_f16_ptr: int,
    v_f16_ptr: int,
    out_ptr: int,
    *,
    rows: int,
    start_position: int,
    mask_ptr: int = 0,
    mask_words: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Attention for ``rows`` queries at absolute positions start_position..

    ``mask_ptr`` zero runs the dense causal window; otherwise each query sees
    its masked four-token blocks plus its own incomplete tail block.
    """

    if rows <= 0 or start_position < 0:
        raise ValueError("QSA WMMA attention needs positive rows")
    if mask_ptr and not 0 < mask_words <= QSA_WMMA_MAX_MASK_WORDS:
        raise ValueError("QSA WMMA attention mask width is out of range")
    _call(
        _SYMBOL_ATTENTION,
        [ctypes.c_void_p] * 4 + [ctypes.c_int64, ctypes.c_void_p]
        + [ctypes.c_int64] * 2 + [ctypes.c_void_p],
        (
            ctypes.c_void_p(query_ptr),
            ctypes.c_void_p(k_f16_ptr),
            ctypes.c_void_p(v_f16_ptr),
            ctypes.c_void_p(mask_ptr or None),
            ctypes.c_int64(mask_words if mask_ptr else 0),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(start_position),
            ctypes.c_void_p(stream),
        ),
        library,
        runtime,
    )


def register_qwen4_exp_qsa_wmma_kernels(*, replace: bool = True) -> None:
    # The same launch serves the masked sparse window and, with no mask, the
    # dense causal window the Qwen4Exp plans declare under paged_attn_decode.
    for layer, quant in (
        ("qsa_sparse_attention", "bf16_kv"),
        ("paged_attn_decode", "w4_paro"),
    ):
        register(
            KernelKey("hip_gfx1100", layer, quant, QSA_WMMA_VARIANT),
            qwen4_exp_qsa_wmma_attention_f32,
            replace=replace,
        )


register_qwen4_exp_qsa_wmma_kernels()

__all__ = [
    "QSA_WMMA_VARIANT",
    "build_qwen4_exp_qsa_wmma",
    "plan_qwen4_exp_qsa_wmma_build",
    "qsa_wmma_geometry_supported",
    "qsa_wmma_mask_words",
    "qwen4_exp_qsa_wmma_attention_f32",
    "qwen4_exp_qsa_wmma_gather_kv_f16",
    "qwen4_exp_qsa_wmma_selection_mask",
    "register_qwen4_exp_qsa_wmma_kernels",
]
