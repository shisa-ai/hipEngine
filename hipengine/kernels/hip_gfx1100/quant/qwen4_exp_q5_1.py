"""Raw Q5_1 selected-expert wrapper for the Unsloth Qwen4Exp comparator quant."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_q5_1.hip")
_OUTPUT_NAME = "qwen4_exp_q5_1.so"
_ARGS_GATHER = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_GROUPED_WMMA = (
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
    ctypes.c_void_p,
)
_ARGS_GROUPED = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_WEIGHTED = (
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
)
_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_qwen4_exp_q5_1_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_q5_1",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_q5_1(
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
        family="qwen4_exp_q5_1",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_gather_bf16_lanes(
    input_ptr: int,
    sorted_lanes_ptr: int,
    output_ptr: int,
    rows: int,
    features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0 or features <= 0:
        raise ValueError("rows and features must be positive")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gather_bf16_lanes",
        _ARGS_GATHER,
        ctypes.c_int,
    )
    error = fn(input_ptr, sorted_lanes_ptr, output_ptr, rows, features, stream)
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
    input_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run compact grouped Q5_1 down projection through WMMA."""

    if compact_rows <= 0 or num_experts <= 0 or wmma_total_rows <= 0:
        raise ValueError("compact_rows, num_experts, and wmma_total_rows must be positive")
    if wmma_total_rows % 16:
        raise ValueError("wmma_total_rows must be divisible by 16")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 grouped WMMA projection has invalid feature geometry")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
        _ARGS_GROUPED_WMMA,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        wmma_total_rows,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    input_ptr: int,
    expert_start_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run grouped Q5_1 rows with eight-way expert-weight reuse."""

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 grouped projection has invalid feature geometry")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
        _ARGS_GROUPED,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        expert_start_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out(
    input_ptr: int,
    expert_start_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact grouped Q5_1 row8/output4 projection."""

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 grouped projection has invalid feature geometry")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out",
        _ARGS_GROUPED,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        expert_start_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out(
    input_ptr: int,
    expert_start_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact row8/output8 Q5_1 over a fixed 64-CTA expert grid."""

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 grouped projection has invalid feature geometry")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
        _ARGS_GROUPED,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        expert_start_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
    input_ptr: int,
    expert_start_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    _pair: bool = False,
) -> None:
    """Run the PF-3 M1 candidate: fused single-loop logical256 Q5_1 row8/
    output8 over a fixed 64-CTA expert grid (strict fallback:
    ``..._expertgrid64_bf16_bf16_out`` stays production and untouched)."""

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 grouped projection has invalid feature geometry")
    if _pair and in_features > 4096:
        raise ValueError("paired Q5_1 supports at most 4096 input features")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        ("hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out" if _pair else
         "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out"),
        _ARGS_GROUPED,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        expert_start_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out(*args, **kwargs):
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
        *args, **kwargs, _pair=True)


def qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out(
    input_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    routing_weights_ptr: int,
    output_ptr: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact selected Q5_1 down projection plus routed weighted sum."""

    if rows <= 0 or num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("rows, experts, and features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out",
        _ARGS_WEIGHTED,
        ctypes.c_int,
    )
    error = fn(
        input_ptr, selected_ptr, weights_ptr, routing_weights_ptr, output_ptr,
        rows, num_experts, in_features, out_features, stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out(
    input_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact logical-256 selected Q5_1 with 64 physical threads."""

    if x_rows <= 0 or rows <= 0 or rows % x_rows:
        raise ValueError("rows must be positive and divisible by positive x_rows")
    if num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("num_experts, in_features, and out_features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr, selected_ptr, weights_ptr, output_ptr, x_rows, rows,
        num_experts, in_features, out_features, stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out(
    input_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact logical-256 selected Q5_1 with 128 physical threads."""

    if x_rows <= 0 or rows <= 0 or rows % x_rows:
        raise ValueError("rows must be positive and divisible by positive x_rows")
    if num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("num_experts, in_features, and out_features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr, selected_ptr, weights_ptr, output_ptr, x_rows, rows,
        num_experts, in_features, out_features, stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out(
    input_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run selected Q5_1 GEMV with a 64-thread K=640-oriented reduction."""

    if x_rows <= 0 or rows <= 0 or rows % x_rows:
        raise ValueError("rows must be positive and divisible by positive x_rows")
    if num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("num_experts, in_features, and out_features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr, selected_ptr, weights_ptr, output_ptr, x_rows, rows,
        num_experts, in_features, out_features, stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
    input_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run one raw Q5_1 expert projection for each compact BF16 input row."""

    if x_rows <= 0 or rows <= 0 or rows % x_rows:
        raise ValueError("rows must be positive and divisible by positive x_rows")
    if num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("num_experts, in_features, and out_features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    if threads != 256:
        raise ValueError("Q5_1 strict selected GEMV requires threads == 256")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_gemv_bf16_bf16_out",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        selected_ptr,
        weights_ptr,
        output_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def register_qwen4_exp_q5_1_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_grouped_prefill_pair2_bf16_bf16_out"),
        qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
        ),
        qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
        ),
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out",
        ),
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
        ),
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
        ),
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out,
        replace=replace,
    )
    for layer in ("linear", "moe_linear"):
        register(
            KernelKey(
                "hip_gfx1100",
                layer,
                "gguf_q5_1",
                "selected_gemv_wave64_bf16_bf16_out",
            ),
            qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                layer,
                "gguf_q5_1",
                "selected_gemv_logical256_t128_bf16_bf16_out",
            ),
            qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                layer,
                "gguf_q5_1",
                "selected_gemv_logical256_t64_bf16_bf16_out",
            ),
            qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                layer,
                "gguf_q5_1",
                "selected_weighted_sum_logical256_t64_bf16_bf16_out",
            ),
            qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out,
            replace=replace,
        )
    for layer in ("linear", "moe_linear"):
        register(
            KernelKey(
                "hip_gfx1100",
                layer,
                "gguf_q5_1",
                "selected_gemv_bf16_bf16_out",
            ),
            qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
            replace=replace,
        )


register_qwen4_exp_q5_1_kernels()


__all__ = [
    "qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out",
    "build_qwen4_exp_q5_1",
    "plan_qwen4_exp_q5_1_build",
    "qwen4_exp_gather_bf16_lanes",
    "qwen4_exp_q5_1_selected_gemv_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
    "register_qwen4_exp_q5_1_kernels",
]
