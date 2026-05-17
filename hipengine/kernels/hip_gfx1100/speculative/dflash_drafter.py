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
