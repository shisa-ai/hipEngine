"""Graph-friendly runtime state kernels for Qwen3.5/PARO decode."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("state.hip")
_OUTPUT_NAME = "runtime_state.so"
_SYMBOL_EMBEDDING_LOOKUP = "hipengine_embedding_lookup_bf16_i64"
_SYMBOL_SET_I64 = "hipengine_set_i64_scalar"
_SYMBOL_SET_POSITION = "hipengine_set_decode_position_i64"
_SYMBOL_ADVANCE_POSITION = "hipengine_advance_decode_position_i64"


def plan_runtime_state_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="runtime_state",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_runtime_state(
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
        family="runtime_state",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def embedding_lookup_bf16_i64(
    embedding_bf16_ptr: int,
    token_id_i64_ptr: int,
    out_bf16_ptr: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy ``embedding[token_id[0], :]`` to ``out`` using device token state."""

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_EMBEDDING_LOOKUP)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(embedding_bf16_ptr),
        ctypes.c_void_p(token_id_i64_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def set_i64_scalar(
    out_i64_ptr: int,
    value: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Set one device int64 scalar."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SET_I64)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(out_i64_ptr), ctypes.c_int64(value), ctypes.c_void_p(stream))
    _check_launch(runtime, err)


def set_decode_position_i64(
    position_i64_ptr: int,
    context_i64_ptr: int,
    value: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Set decode append position and attention context count on device."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SET_POSITION)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(position_i64_ptr),
        ctypes.c_void_p(context_i64_ptr),
        ctypes.c_int64(value),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def advance_decode_position_i64(
    position_i64_ptr: int,
    context_i64_ptr: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Increment device decode position and refresh context count."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ADVANCE_POSITION)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(position_i64_ptr), ctypes.c_void_p(context_i64_ptr), ctypes.c_void_p(stream))
    _check_launch(runtime, err)


def register_runtime_state_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "bf16_i64"),
        embedding_lookup_bf16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "w4_paro", "set_i64"),
        set_decode_position_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "w4_paro", "advance_i64"),
        advance_decode_position_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "scalar_state", "w4_paro", "set_i64"),
        set_i64_scalar,
        replace=replace,
    )


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_runtime_state_kernels()
