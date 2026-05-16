"""Raw-pointer wrappers for Qwen3.5 native router kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("router.hip")
_OUTPUT_NAME = "qwen35_router.so"
_SYMBOL_LOGITS = "hipengine_qwen35_router_logits_bf16"
_SYMBOL_LOGITS_FP16 = "hipengine_qwen35_router_logits_fp16"
_SYMBOL_SELECT = "hipengine_qwen35_router_select"
_SYMBOL_TOPK_SHARED_OUT = "hipengine_qwen35_router_topk_shared_out_bf16"
_SYMBOL_TOPK_SHARED_OUT_FP16 = "hipengine_qwen35_router_topk_shared_out_fp16"
_SYMBOL_TOPK_SHARED_COOP_OUT = "hipengine_qwen35_router_topk_shared_coop_out_bf16"
_SYMBOL_TOPK_SHARED_COOP_OUT_FP16 = "hipengine_qwen35_router_topk_shared_coop_out_fp16"
_ALLOWED_THREADS = {64, 128, 256, 512}


def plan_qwen35_router_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_router",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_router(
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
        family="qwen35_router",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_router_logits_bf16(
    hidden_ptr: int,
    weight_ptr: int,
    logits_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch router logits for BF16-bit hidden/weight buffers and F32 logits."""

    _check_positive(tokens, "tokens")
    _check_positive(hidden_size, "hidden_size")
    _check_positive(num_rows, "num_rows")
    _check_threads(threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LOGITS)
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
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(logits_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(num_rows),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_router_logits_fp16(
    hidden_ptr: int,
    weight_ptr: int,
    logits_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch router logits for FP16 hidden buffers, BF16-bit weights, and F32 logits."""

    _check_positive(tokens, "tokens")
    _check_positive(hidden_size, "hidden_size")
    _check_positive(num_rows, "num_rows")
    _check_threads(threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LOGITS_FP16)
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
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(logits_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(num_rows),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_router_select(
    logits_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    tokens: int,
    logits_stride: int,
    num_experts: int,
    top_k: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch block-parallel top-k + softmax for precomputed router logits."""

    _check_router_select_shape(tokens, logits_stride, num_experts, top_k)
    _check_threads(threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SELECT)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(routing_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(logits_stride),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_router_topk_shared_out_bf16(
    hidden_ptr: int,
    combined_weight_ptr: int,
    logits_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    num_experts: int,
    top_k: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch combined router/shared-gate logits and top-k into caller-owned buffers.

    ``combined_weight`` has ``num_rows`` rows, with the first ``num_experts`` rows used for
    top-k selection and the remaining row(s) available as shared-gate logits.
    """

    _check_positive(tokens, "tokens")
    _check_positive(hidden_size, "hidden_size")
    _check_positive(num_rows, "num_rows")
    _check_router_select_shape(tokens, num_rows, num_experts, top_k)
    if num_experts >= num_rows:
        raise ValueError("num_experts must be smaller than num_rows for shared-gate routing")
    _check_threads(threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPK_SHARED_OUT)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(combined_weight_ptr),
        ctypes.c_void_p(logits_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(routing_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(num_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_router_topk_shared_out_fp16(
    hidden_ptr: int,
    combined_weight_ptr: int,
    logits_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    num_experts: int,
    top_k: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch combined router/shared-gate logits for FP16 hidden and BF16 weights."""

    _check_positive(tokens, "tokens")
    _check_positive(hidden_size, "hidden_size")
    _check_positive(num_rows, "num_rows")
    _check_router_select_shape(tokens, num_rows, num_experts, top_k)
    if num_experts >= num_rows:
        raise ValueError("num_experts must be smaller than num_rows for shared-gate routing")
    _check_threads(threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPK_SHARED_OUT_FP16)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(combined_weight_ptr),
        ctypes.c_void_p(logits_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(routing_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(num_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_router_topk_shared_coop_out_bf16(
    hidden_ptr: int,
    combined_weight_ptr: int,
    logits_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    num_experts: int,
    top_k: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch decode-only cooperative router logits + top-k in one kernel."""

    _check_decode_coop_shape(tokens, hidden_size, num_rows, num_experts, top_k, threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPK_SHARED_COOP_OUT)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(combined_weight_ptr),
        ctypes.c_void_p(logits_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(routing_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(num_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_router_topk_shared_coop_out_fp16(
    hidden_ptr: int,
    combined_weight_ptr: int,
    logits_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    num_experts: int,
    top_k: int,
    *,
    threads: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch decode-only cooperative router logits + top-k for FP16 hidden."""

    _check_decode_coop_shape(tokens, hidden_size, num_rows, num_experts, top_k, threads)
    library = library or build_qwen35_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPK_SHARED_COOP_OUT_FP16)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(combined_weight_ptr),
        ctypes.c_void_p(logits_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(routing_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(num_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_qwen35_router_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "router_logits", "bf16"),
        qwen35_router_logits_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "router_logits", "fp16"),
        qwen35_router_logits_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "router_logits", "w4_paro", "fp16_hidden"),
        qwen35_router_logits_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "router_select", "fp32"),
        qwen35_router_select,
        replace=replace,
    )
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey("hip_gfx1100", "router_topk_shared", quant, "out"),
            qwen35_router_topk_shared_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "router_topk_shared", quant, "out_fp16_hidden"),
            qwen35_router_topk_shared_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "router_topk_shared", quant, "coop_out"),
            qwen35_router_topk_shared_coop_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "router_topk_shared", quant, "coop_out_fp16_hidden"),
            qwen35_router_topk_shared_coop_out_fp16,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "router_topk_shared", "fp16", "out"),
        qwen35_router_topk_shared_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "router_topk_shared", "fp16", "coop_out"),
        qwen35_router_topk_shared_coop_out_fp16,
        replace=replace,
    )


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, 256, or 512")


def _check_router_select_shape(
    tokens: int,
    logits_stride: int,
    num_experts: int,
    top_k: int,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(logits_stride, "logits_stride")
    _check_positive(num_experts, "num_experts")
    _check_positive(top_k, "top_k")
    if top_k > 16:
        raise ValueError("top_k must be <= 16")
    if top_k > num_experts:
        raise ValueError("top_k must be <= num_experts")
    if num_experts > logits_stride:
        raise ValueError("num_experts must be <= logits_stride")


def _check_decode_coop_shape(
    tokens: int,
    hidden_size: int,
    num_rows: int,
    num_experts: int,
    top_k: int,
    threads: int,
) -> None:
    if tokens != 1:
        raise ValueError("cooperative router is decode-only and requires tokens == 1")
    _check_positive(hidden_size, "hidden_size")
    _check_positive(num_rows, "num_rows")
    _check_router_select_shape(tokens, num_rows, num_experts, top_k)
    if num_experts >= num_rows:
        raise ValueError("num_experts must be smaller than num_rows for shared-gate routing")
    _check_threads(threads)


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_router_kernels()
