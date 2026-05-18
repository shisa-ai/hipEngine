"""Raw-pointer DFlash drafter root/query preparation wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dflash_drafter.hip")
_OUTPUT_NAME = "dflash_drafter.so"
_SYMBOL_PREPARE_NOISE_BF16 = "hipengine_dflash_prepare_noise_inputs_bf16_i32"
_SYMBOL_PREPARE_NOISE_F16_TO_BF16 = "hipengine_dflash_prepare_noise_inputs_f16_to_bf16_i32"
_SYMBOL_RMSNORM_BF16 = "hipengine_dflash_rmsnorm_bf16"
_SYMBOL_DENSE_BF16_TO_F32 = "hipengine_dflash_dense_bf16_to_f32"
_SYMBOL_HEAD_RMS_ROTARY = "hipengine_dflash_head_rmsnorm_rotary_f32"
_SYMBOL_GQA_ATTENTION = "hipengine_dflash_gqa_attention_f32_bf16"
_ALLOWED_THREADS = {64, 128, 256}


def plan_dflash_drafter_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dflash_drafter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dflash_drafter(
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
        family="dflash_drafter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dflash_prepare_noise_inputs_bf16_i32(
    root_tokens_i32_ptr: int,
    root_positions_i32_ptr: int,
    embed_tokens_bf16_ptr: int,
    noise_token_ids_i32_ptr: int,
    position_ids_i32_ptr: int,
    noise_embeddings_bf16_ptr: int,
    request_count: int,
    block_size: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Materialize DFlash root+mask ids, positions, and BF16 embedding rows."""

    _launch_prepare_noise_inputs(
        _SYMBOL_PREPARE_NOISE_BF16,
        root_tokens_i32_ptr,
        root_positions_i32_ptr,
        embed_tokens_bf16_ptr,
        noise_token_ids_i32_ptr,
        position_ids_i32_ptr,
        noise_embeddings_bf16_ptr,
        request_count,
        block_size,
        hidden_size,
        vocab_size,
        mask_token_id,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def dflash_prepare_noise_inputs_f16_to_bf16_i32(
    root_tokens_i32_ptr: int,
    root_positions_i32_ptr: int,
    embed_tokens_f16_ptr: int,
    noise_token_ids_i32_ptr: int,
    position_ids_i32_ptr: int,
    noise_embeddings_bf16_ptr: int,
    request_count: int,
    block_size: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Materialize root+mask inputs while converting FP16 embedding rows to BF16."""

    _launch_prepare_noise_inputs(
        _SYMBOL_PREPARE_NOISE_F16_TO_BF16,
        root_tokens_i32_ptr,
        root_positions_i32_ptr,
        embed_tokens_f16_ptr,
        noise_token_ids_i32_ptr,
        position_ids_i32_ptr,
        noise_embeddings_bf16_ptr,
        request_count,
        block_size,
        hidden_size,
        vocab_size,
        mask_token_id,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def dflash_rmsnorm_bf16(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply standard DFlash/Qwen RMSNorm with direct BF16 weight scaling."""

    _check_rmsnorm_shape(rows, hidden_size, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_RMSNORM_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(weight_bf16_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_dense_bf16_to_f32(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_f32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Project BF16 rows with BF16 weights and write FP32 output rows."""

    _check_dense_shape(rows, in_features, out_features, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_BF16_TO_F32)
    fn.argtypes = [
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
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(weight_bf16_ptr),
        ctypes.c_void_p(out_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_head_rmsnorm_rotary_f32(
    query_f32_ptr: int,
    key_f32_ptr: int,
    q_weight_bf16_ptr: int,
    k_weight_bf16_ptr: int,
    cos_table_f32_ptr: int,
    sin_table_f32_ptr: int,
    query_positions_i32_ptr: int,
    key_positions_i32_ptr: int,
    query_out_f32_ptr: int,
    key_out_f32_ptr: int,
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply direct-weight head RMSNorm plus rotary to DFlash Q/K projections."""

    _check_head_rotary_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, max_positions, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_HEAD_RMS_ROTARY)
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
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_f32_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(q_weight_bf16_ptr),
        ctypes.c_void_p(k_weight_bf16_ptr),
        ctypes.c_void_p(cos_table_f32_ptr),
        ctypes.c_void_p(sin_table_f32_ptr),
        ctypes.c_void_p(query_positions_i32_ptr),
        ctypes.c_void_p(key_positions_i32_ptr),
        ctypes.c_void_p(query_out_f32_ptr),
        ctypes.c_void_p(key_out_f32_ptr),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(query_len),
        ctypes.c_int64(kv_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_gqa_attention_f32_bf16(
    query_f32_ptr: int,
    key_f32_ptr: int,
    value_bf16_ptr: int,
    out_bf16_ptr: int,
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    scale: float | None = None,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch slow-but-deterministic non-causal DFlash GQA attention.

    Inputs are row-major ``query[batch, query_len, q_heads, head_dim]``,
    ``key/value[batch, kv_len, kv_heads, head_dim]``. The output is BF16 bits in
    ``out[batch, query_len, q_heads, head_dim]``. This correctness-first kernel
    is intended for the native drafter root/query harness, not final throughput.
    """

    _check_attention_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, threads)
    scale_value = float(head_dim ** -0.5 if scale is None else scale)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GQA_ATTENTION)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_f32_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_bf16_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(query_len),
        ctypes.c_int64(kv_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale_value),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dflash_drafter_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dflash_prepare_noise_inputs", "w4_paro", "bf16_i32"),
        dflash_prepare_noise_inputs_bf16_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_prepare_noise_inputs", "w4_paro", "f16_to_bf16_i32"),
        dflash_prepare_noise_inputs_f16_to_bf16_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_rmsnorm", "w4_paro", "bf16"),
        dflash_rmsnorm_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_f32"),
        dflash_dense_bf16_to_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_head_rmsnorm_rotary", "w4_paro", "f32_bf16"),
        dflash_head_rmsnorm_rotary_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_gqa_attention", "w4_paro", "f32_bf16"),
        dflash_gqa_attention_f32_bf16,
        replace=replace,
    )


def _launch_prepare_noise_inputs(
    symbol: str,
    root_tokens_i32_ptr: int,
    root_positions_i32_ptr: int,
    embed_tokens_ptr: int,
    noise_token_ids_i32_ptr: int,
    position_ids_i32_ptr: int,
    noise_embeddings_bf16_ptr: int,
    request_count: int,
    block_size: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_shape(request_count, block_size, hidden_size, vocab_size, threads)
    if mask_token_id < 0 or mask_token_id >= vocab_size:
        raise ValueError("mask_token_id must be within vocab_size")
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(root_tokens_i32_ptr),
        ctypes.c_void_p(root_positions_i32_ptr),
        ctypes.c_void_p(embed_tokens_ptr),
        ctypes.c_void_p(noise_token_ids_i32_ptr),
        ctypes.c_void_p(position_ids_i32_ptr),
        ctypes.c_void_p(noise_embeddings_bf16_ptr),
        ctypes.c_int64(request_count),
        ctypes.c_int64(block_size),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int32(mask_token_id),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_rmsnorm_shape(rows: int, hidden_size: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_dense_shape(rows: int, in_features: int, out_features: int, threads: int) -> None:
    for name, value in (("rows", rows), ("in_features", in_features), ("out_features", out_features)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_head_rotary_shape(
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    threads: int,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("query_len", query_len),
        ("kv_len", kv_len),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("rotary_dim", rotary_dim),
        ("max_positions", max_positions),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_attention_shape(
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    threads: int,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("query_len", query_len),
        ("kv_len", kv_len),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_shape(request_count: int, block_size: int, hidden_size: int, vocab_size: int, threads: int) -> None:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


register_dflash_drafter_kernels()
