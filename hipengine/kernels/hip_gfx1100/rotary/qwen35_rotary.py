"""Raw-pointer wrappers for Qwen3.5 rotary/full-attention prelude kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen35_rotary.hip")
_OUTPUT_NAME = "qwen35_rotary.so"
_SYMBOL_SPLIT_QGATE = "hipengine_qwen35_split_qgate_bf16"
_SYMBOL_SPLIT_QGATE_FP16 = "hipengine_qwen35_split_qgate_fp16"
_SYMBOL_SPLIT_QGATE_FP16_KEY_F32 = "hipengine_qwen35_split_qgate_fp16_key_f32"
_SYMBOL_PARTIAL = "hipengine_qwen35_partial_rotary_f32"
_SYMBOL_HEAD_RMS = "hipengine_qwen35_head_rmsnorm_partial_rotary_f32_bf16"
_SYMBOL_HEAD_RMS_POSITION = "hipengine_qwen35_head_rmsnorm_partial_rotary_position_f32_bf16"
_SYMBOL_HEAD_RMS_POSITIONS = "hipengine_qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16"
_SYMBOL_HEAD_RMS_POSITIONS_Q_BF16 = (
    "hipengine_qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32"
)


def plan_qwen35_rotary_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_rotary",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_rotary(
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
        family="qwen35_rotary",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_split_qgate_bf16(
    q_proj_ptr: int,
    query_out_ptr: int,
    gate_out_ptr: int,
    tokens: int,
    num_q_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split interleaved Qwen full-attention q_proj rows into query/gate buffers."""

    _check_positive(tokens, "tokens")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(head_dim, "head_dim")
    library = library or build_qwen35_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SPLIT_QGATE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(q_proj_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(gate_out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_split_qgate_fp16(
    q_proj_ptr: int,
    query_out_ptr: int,
    gate_out_ptr: int,
    tokens: int,
    num_q_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split interleaved FP16 Qwen full-attention q_proj rows into FP32 query and FP16 gate buffers."""

    _launch_split_qgate(
        _SYMBOL_SPLIT_QGATE_FP16,
        q_proj_ptr,
        query_out_ptr,
        gate_out_ptr,
        tokens,
        num_q_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_split_qgate_fp16_key_f32(
    q_proj_ptr: int,
    key_in_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    gate_out_ptr: int,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split FP16 Q/Gate and cast FP16 key to FP32 in one verifier launch."""

    _check_positive(tokens, "tokens")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    _check_positive(head_dim, "head_dim")
    library = library or build_qwen35_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SPLIT_QGATE_FP16_KEY_F32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(q_proj_ptr),
        ctypes.c_void_p(key_in_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_void_p(gate_out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_partial_rotary_f32(
    query_ptr: int,
    key_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch Qwen3.5 partial rotary on FP32 query/key heads."""

    _check_common_shape(num_q_heads, num_kv_heads, head_dim, rotary_dim)
    library = library or build_qwen35_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PARTIAL)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_head_rmsnorm_partial_rotary_f32_bf16(
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused head RMSNorm + partial rotary using BF16 norm weights."""

    _launch_head_rmsnorm_partial_rotary(
        _SYMBOL_HEAD_RMS,
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        None,
        query_out_ptr,
        key_out_ptr,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        0,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_table_ptr: int,
    sin_table_ptr: int,
    position_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused head RMSNorm + table-indexed partial rotary for one position."""

    _launch_head_rmsnorm_partial_rotary(
        _SYMBOL_HEAD_RMS_POSITION,
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_table_ptr,
        sin_table_ptr,
        position_ptr,
        query_out_ptr,
        key_out_ptr,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        max_positions,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16(
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_table_ptr: int,
    sin_table_ptr: int,
    positions_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused head RMSNorm + per-token table-indexed partial rotary."""

    _launch_head_rmsnorm_partial_rotary_positions(
        _SYMBOL_HEAD_RMS_POSITIONS,
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_table_ptr,
        sin_table_ptr,
        positions_ptr,
        query_out_ptr,
        key_out_ptr,
        eps,
        tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        max_positions,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32(
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_table_ptr: int,
    sin_table_ptr: int,
    positions_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused head RMSNorm + RoPE with BF16 query output and FP32 key output."""

    _launch_head_rmsnorm_partial_rotary_positions(
        _SYMBOL_HEAD_RMS_POSITIONS_Q_BF16,
        query_ptr,
        key_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_table_ptr,
        sin_table_ptr,
        positions_ptr,
        query_out_ptr,
        key_out_ptr,
        eps,
        tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        max_positions,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_qwen35_rotary_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "split_qgate", "w4_paro", "qwen35_bf16"),
        qwen35_split_qgate_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "split_qgate", "w4_paro", "qwen35_fp16"),
        qwen35_split_qgate_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "split_qgate+key_cast", "w4_paro", "qwen35_fp16_key_f32"),
        qwen35_split_qgate_fp16_key_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "partial_rotary", "w4_paro", "qwen35_f32"),
        qwen35_partial_rotary_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "head_rmsnorm+partial_rotary", "w4_paro", "qwen35_f32_bf16"),
        qwen35_head_rmsnorm_partial_rotary_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "head_rmsnorm+partial_rotary",
            "w4_paro",
            "qwen35_position_f32_bf16",
        ),
        qwen35_head_rmsnorm_partial_rotary_position_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "head_rmsnorm+partial_rotary",
            "w4_paro",
            "qwen35_positions_f32_bf16",
        ),
        qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "head_rmsnorm+partial_rotary",
            "w4_paro",
            "qwen35_positions_q_bf16_key_f32",
        ),
        qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32,
        replace=replace,
    )


def _launch_split_qgate(
    symbol: str,
    q_proj_ptr: int,
    query_out_ptr: int,
    gate_out_ptr: int,
    tokens: int,
    num_q_heads: int,
    head_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(head_dim, "head_dim")
    library = library or build_qwen35_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(q_proj_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(gate_out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_head_rmsnorm_partial_rotary(
    symbol: str,
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    position_ptr: int | None,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common_shape(num_q_heads, num_kv_heads, head_dim, rotary_dim)
    if position_ptr is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive")
    library = library or build_qwen35_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    if position_ptr is None:
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        args = (
            ctypes.c_void_p(query_ptr),
            ctypes.c_void_p(key_ptr),
            ctypes.c_void_p(q_weight_ptr),
            ctypes.c_void_p(k_weight_ptr),
            ctypes.c_void_p(cos_ptr),
            ctypes.c_void_p(sin_ptr),
            ctypes.c_void_p(query_out_ptr),
            ctypes.c_void_p(key_out_ptr),
            ctypes.c_float(eps),
            ctypes.c_int64(num_q_heads),
            ctypes.c_int64(num_kv_heads),
            ctypes.c_int64(head_dim),
            ctypes.c_int64(rotary_dim),
            ctypes.c_void_p(stream),
        )
    else:
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        args = (
            ctypes.c_void_p(query_ptr),
            ctypes.c_void_p(key_ptr),
            ctypes.c_void_p(q_weight_ptr),
            ctypes.c_void_p(k_weight_ptr),
            ctypes.c_void_p(cos_ptr),
            ctypes.c_void_p(sin_ptr),
            ctypes.c_void_p(position_ptr),
            ctypes.c_void_p(query_out_ptr),
            ctypes.c_void_p(key_out_ptr),
            ctypes.c_float(eps),
            ctypes.c_int64(num_q_heads),
            ctypes.c_int64(num_kv_heads),
            ctypes.c_int64(head_dim),
            ctypes.c_int64(rotary_dim),
            ctypes.c_int64(max_positions),
            ctypes.c_void_p(stream),
        )
    fn.restype = ctypes.c_int
    err = fn(*args)
    _check_launch(runtime, err)


def _launch_head_rmsnorm_partial_rotary_positions(
    symbol: str,
    query_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    positions_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    eps: float,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_common_shape(num_q_heads, num_kv_heads, head_dim, rotary_dim)
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    library = library or build_qwen35_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(q_weight_ptr),
        ctypes.c_void_p(k_weight_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_void_p(positions_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_common_shape(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> None:
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    _check_positive(head_dim, "head_dim")
    _check_positive(rotary_dim, "rotary_dim")
    if rotary_dim > head_dim:
        raise ValueError("rotary_dim must be <= head_dim")
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_rotary_kernels()
