"""Raw-pointer wrapper for GGUF Q6_K token embedding lookup."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q6_k_embedding.hip")
_OUTPUT_NAME = "gguf_q6_k_embedding.so"
_ALLOWED_THREADS = {64, 128, 256}


def plan_gguf_q6_k_embedding_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q6_k_embedding",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q6_k_embedding(
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
        family="gguf_q6_k_embedding",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q6_k_embedding_bf16_out(
    token_ids_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch GGUF Q6_K embedding lookup into BF16-bit output."""

    _validate(rows, hidden_size, vocab_size, threads)
    library = library or build_gguf_q6_k_embedding(load=True)
    runtime = runtime or get_hip_runtime()
    fn = library.hipengine_gguf_q6_k_embedding_bf16_out
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
        ctypes.c_void_p(token_ids_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_q6_k_embedding_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "embedding", "gguf_q6_k", "lookup_bf16_out"),
        gguf_q6_k_embedding_bf16_out,
        replace=replace,
    )


def _validate(rows: int, hidden_size: int, vocab_size: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by GGUF Q6_K block size 256")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


register_gguf_q6_k_embedding_kernels()


__all__ = [
    "build_gguf_q6_k_embedding",
    "gguf_q6_k_embedding_bf16_out",
    "plan_gguf_q6_k_embedding_build",
    "register_gguf_q6_k_embedding_kernels",
]
