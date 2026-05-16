"""Small BF16 elementwise helpers for GGUF Qwen3.5 runtime wiring."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_ops.hip")
_OUTPUT_NAME = "gguf_ops.so"
_ALLOWED_THREADS = {64, 128, 256, 512}


def plan_gguf_ops_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_ops(
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
        family="gguf_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_rmsnorm_bf16_f32_weight(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(hidden_size, "hidden_size")
    _check_threads(threads)
    _launch_rmsnorm(
        "hipengine_gguf_rmsnorm_bf16_f32_weight",
        (x_ptr, weight_ptr, out_ptr),
        rows,
        hidden_size,
        eps,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_add_rmsnorm_bf16_f32_weight(
    x_ptr: int,
    add_ptr: int,
    weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(hidden_size, "hidden_size")
    _check_threads(threads)
    _launch_add_rmsnorm(
        "hipengine_gguf_add_rmsnorm_bf16_f32_weight",
        (x_ptr, add_ptr, weight_ptr, norm_out_ptr, residual_out_ptr),
        rows,
        hidden_size,
        eps,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_bf16_add(
    a_ptr: int,
    b_ptr: int,
    out_ptr: int,
    n: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_positive(n, "n")
    _check_threads(threads)
    _launch(
        "hipengine_gguf_bf16_add",
        (a_ptr, b_ptr, out_ptr, n),
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_gate_repeat_value_bf16(
    q_gate_ptr: int,
    value_ptr: int,
    out_ptr: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_positive(head_count, "head_count")
    _check_positive(head_count_kv, "head_count_kv")
    _check_positive(head_dim, "head_dim")
    if head_count % head_count_kv:
        raise ValueError("head_count must be divisible by head_count_kv")
    _check_threads(threads)
    _launch(
        "hipengine_gguf_gate_repeat_value_bf16",
        (q_gate_ptr, value_ptr, out_ptr, head_count, head_count_kv, head_dim),
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_gguf_ops(*, replace: bool = True) -> None:
    register(KernelKey("hip_gfx1100", "elementwise", "bf16", "add"), gguf_bf16_add, replace=replace)
    register(
        KernelKey("hip_gfx1100", "rmsnorm", "gguf_f32_weight", "bf16_out"),
        gguf_rmsnorm_bf16_f32_weight,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "add_rmsnorm", "gguf_f32_weight", "bf16_out"),
        gguf_add_rmsnorm_bf16_f32_weight,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "attention", "gguf_qwen35", "gate_repeat_value_bf16"),
        gguf_gate_repeat_value_bf16,
        replace=replace,
    )


def _launch_rmsnorm(
    symbol: str,
    ptrs: tuple[int, int, int],
    rows: int,
    hidden_size: int,
    eps: float,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_gguf_ops(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(ptrs[0]),
        ctypes.c_void_p(ptrs[1]),
        ctypes.c_void_p(ptrs[2]),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_float(eps),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_add_rmsnorm(
    symbol: str,
    ptrs: tuple[int, int, int, int, int],
    rows: int,
    hidden_size: int,
    eps: float,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_gguf_ops(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(ptrs[0]),
        ctypes.c_void_p(ptrs[1]),
        ctypes.c_void_p(ptrs[2]),
        ctypes.c_void_p(ptrs[3]),
        ctypes.c_void_p(ptrs[4]),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_float(eps),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch(
    symbol: str,
    args: tuple[int, ...],
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_gguf_ops(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * (len(args) - 3) + [
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(args[0]),
        ctypes.c_void_p(args[1]),
        ctypes.c_void_p(args[2]),
        *(ctypes.c_int64(value) for value in args[3:]),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


register_gguf_ops()


__all__ = [
    "build_gguf_ops",
    "gguf_add_rmsnorm_bf16_f32_weight",
    "gguf_bf16_add",
    "gguf_rmsnorm_bf16_f32_weight",
    "gguf_gate_repeat_value_bf16",
    "plan_gguf_ops_build",
    "register_gguf_ops",
]
