"""Raw-pointer wrappers for native MTP proposal helper kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("mtp.hip")
_OUTPUT_NAME = "mtp_speculative.so"
_SYMBOL_FUSE_INPUTS = "hipengine_mtp_fuse_inputs_f16_bf16"
_ALLOWED_THREADS = {64, 128, 256}


def plan_mtp_speculative_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="mtp_speculative",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_mtp_speculative(
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
        family="mtp_speculative",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def mtp_fuse_inputs_f16_bf16(
    token_ids_i64_ptr: int,
    embedding_f16_ptr: int,
    target_hidden_bf16_ptr: int,
    embed_norm_weight_bf16_ptr: int,
    hidden_norm_weight_bf16_ptr: int,
    out_concat_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Normalize token embeddings and target hidden rows, then concatenate.

    Output layout is ``[rows, 2 * hidden_size]`` BF16 with the normalized token
    embedding in the first half and normalized target hidden in the second half.
    This is the input expected by ``mtp.fc.weight``.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {sorted(_ALLOWED_THREADS)}")
    library = library or build_mtp_speculative(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_FUSE_INPUTS)
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
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(embedding_f16_ptr),
        ctypes.c_void_p(target_hidden_bf16_ptr),
        ctypes.c_void_p(embed_norm_weight_bf16_ptr),
        ctypes.c_void_p(hidden_norm_weight_bf16_ptr),
        ctypes.c_void_p(out_concat_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{_SYMBOL_FUSE_INPUTS} failed with HIP error {err}")


def register_mtp_speculative_kernels(*, replace: bool = True) -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "mtp_fuse_inputs", "bf16", "f16_embed_bf16_hidden"),
            mtp_fuse_inputs_f16_bf16,
            replace=replace,
        )


register_mtp_speculative_kernels()


__all__ = [
    "build_mtp_speculative",
    "mtp_fuse_inputs_f16_bf16",
    "plan_mtp_speculative_build",
    "register_mtp_speculative_kernels",
]
