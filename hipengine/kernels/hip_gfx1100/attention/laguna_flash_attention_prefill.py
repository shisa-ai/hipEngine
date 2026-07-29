"""Source-shaped gfx1100 Laguna F16-WMMA FlashAttention primitive."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("laguna_flash_attention_prefill.hip")
_OUTPUT_NAME = "laguna_flash_attention_prefill.so"
_SYMBOL = "hipengine_laguna_flash_attention_prefill_f16_wmma_bf16_spans"
_GLOBAL_BLOCK_SIZE = 256
_NUM_KV_HEADS = 8
_HEAD_DIM = 128


def plan_laguna_flash_attention_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_flash_attention_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_flash_attention_prefill(
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
        family="laguna_flash_attention_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def laguna_flash_attention_prefill_f16_wmma_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run source-shaped causal prefill over complete BF16 ``KVLiveSpans``."""

    parsed_rows = int(rows)
    parsed_q_heads = int(num_q_heads)
    parsed_kv_heads = int(num_kv_heads)
    parsed_head_dim = int(head_dim)
    parsed_scale = float(scale)
    if parsed_rows <= 0:
        raise ValueError("rows must be positive")
    if parsed_q_heads not in {48, 72}:
        raise ValueError("num_q_heads must be a Laguna production width (48 or 72)")
    if parsed_kv_heads != _NUM_KV_HEADS:
        raise ValueError(f"num_kv_heads must be {_NUM_KV_HEADS}")
    if parsed_head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}")
    if not math.isfinite(parsed_scale) or parsed_scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    for name, pointer in (
        ("query_ptr", query_ptr),
        ("key_cache_ptr", key_cache_ptr),
        ("value_cache_ptr", value_cache_ptr),
        ("out_ptr", out_ptr),
    ):
        if int(pointer) <= 0:
            raise ValueError(f"{name} must be non-zero")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("source FlashAttention requires bf16 KV storage")
    if spans.token_positions is None or spans.evict_mask is None:
        raise ValueError("source FlashAttention requires token_positions and evict_mask")
    if spans.row_positions is None or spans.row_positions.numel != 1:
        raise ValueError("source FlashAttention requires one row_positions start scalar")
    if spans.live_counts.numel != 1:
        raise ValueError("source FlashAttention requires one live_counts scalar")

    capacity = int(spans.token_positions.numel)
    if parsed_rows > capacity:
        raise ValueError("rows must not exceed span capacity")
    if spans.evict_mask.numel != capacity:
        raise ValueError("evict_mask must match token_positions capacity")
    if spans.spans_mode == "uniform":
        span_mode = 0
        block_size = _GLOBAL_BLOCK_SIZE
        if spans.base_offsets.numel * block_size < capacity:
            raise ValueError("global base_offsets do not cover span capacity")
        parsed_window = 0
    elif spans.spans_mode == "sliding_ring":
        span_mode = 1
        block_size = 1
        parsed_window = int(sliding_window)
        if spans.base_offsets.numel != capacity:
            raise ValueError("sliding base_offsets must match span capacity")
        if parsed_window <= 0:
            raise ValueError("sliding_window must be positive for sliding spans")
    else:
        raise ValueError("source FlashAttention requires uniform or sliding_ring spans")

    library = library or build_laguna_flash_attention_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
    fn.argtypes = (
        [ctypes.c_void_p] * 9
        + [ctypes.c_int] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int(parsed_rows),
        ctypes.c_int(capacity),
        ctypes.c_int(parsed_q_heads),
        ctypes.c_int(parsed_kv_heads),
        ctypes.c_int(parsed_head_dim),
        ctypes.c_int(span_mode),
        ctypes.c_int(block_size),
        ctypes.c_int(parsed_window),
        ctypes.c_float(parsed_scale),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_laguna_flash_attention_prefill_kernels(
    *, replace: bool = True
) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "laguna_attention_prefill",
            "bf16",
            "source_f16_wmma_q8_gqa8_spans",
        ),
        laguna_flash_attention_prefill_f16_wmma_bf16_spans,
        replace=replace,
    )


register_laguna_flash_attention_prefill_kernels()

__all__ = [
    "build_laguna_flash_attention_prefill",
    "laguna_flash_attention_prefill_f16_wmma_bf16_spans",
    "plan_laguna_flash_attention_prefill_build",
    "register_laguna_flash_attention_prefill_kernels",
]
