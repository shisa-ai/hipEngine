"""Raw-pointer wrappers for Qwen3.5 paged full-attention decode kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("paged_attn_decode.hip")
_OUTPUT_NAME = "qwen35_paged_attn_decode.so"
_SYMBOL_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_context_bf16_spans"


def plan_qwen35_paged_attn_decode_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_paged_attn_decode",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_paged_attn_decode(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="qwen35_paged_attn_decode",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_paged_full_attn_decode_context_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode paged BF16 full attention using ``KVLiveSpans`` metadata.

    Fixed-page spans bridge to the preserved parent context-tensor kernel:
    ``spans.base_offsets`` is the int32 page table and ``spans.live_counts`` is
    the int64 context length tensor.
    """

    _check_decode_shape(spans, max_context_len, block_size, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_CONTEXT)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_qwen35_paged_attn_decode_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_context_spans"),
        qwen35_paged_full_attn_decode_context_bf16_spans,
        replace=replace,
    )


def _check_decode_shape(
    spans: KVLiveSpans,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    if spans.spans_mode != "uniform":
        raise ValueError("paged attention decode currently requires uniform spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("paged attention decode currently requires bf16 storage spans")
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("paged attention decode parent bridge requires int64 live_counts")
    _check_positive(max_context_len, "max_context_len")
    _check_positive(block_size, "block_size")
    if block_size != 256:
        raise ValueError("paged attention decode parent kernel requires block_size=256")
    _check_positive(spans.base_offsets.numel, "block_table_len")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if head_dim > 256:
        raise ValueError("head_dim must be <= 256")
    if ((max_context_len + block_size - 1) // block_size) > spans.base_offsets.numel:
        raise ValueError("span base_offsets block table is too short for max_context_len")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_paged_attn_decode_kernels()
