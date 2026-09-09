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
    _fold128: bool = False,
    _fold_pair: bool = False,
    _register_cache: bool = False,
    _row_publish: bool = False,
) -> None:
    """Run the PF-3 M1 candidate: fused single-loop logical256 Q5_1 row8/
    output8 over a fixed 64-CTA expert grid (strict fallback:
    ``..._expertgrid64_bf16_bf16_out`` stays production and untouched)."""

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 grouped projection has invalid feature geometry")
    if _register_cache and in_features != 640:
        raise ValueError("register-cache Q5_1 requires in_features=640")
    if (_pair or _fold128 or _fold_pair) and in_features > 4096:
        raise ValueError("paired Q5_1 supports at most 4096 input features")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        ("hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out" if _row_publish else
         "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_register_cache_bf16_bf16_out" if _register_cache else
         "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out" if _fold_pair else
         "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_bf16_bf16_out" if _fold128 else
         "hipengine_qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out" if _pair else
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


def qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_bf16_bf16_out(*args, **kwargs):
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
        *args, **kwargs, _fold128=True)


def qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out(*args, **kwargs):
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
        *args, **kwargs, _fold_pair=True)


def qwen4_exp_q5_1_selected_grouped_prefill_pair2_register_cache_bf16_bf16_out(*args, **kwargs):
    return qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
        *args, **kwargs, _register_cache=True)

def qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out(*args, **kwargs):
    return qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out(
        *args, **kwargs, _register_cache=True, _row_publish=True)


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


def qwen4_exp_q5_1_selected_weighted_sum_warp256_bf16_bf16_out(
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
    """Run the #22 R11 warp-per-expert selected Q5_1 weighted sum.

    One block per output column with 32*rows threads; warp w serves
    expert row w (rows <= 12), lanes own 16-element half-block chunks with vectorized
    loads. Same per-row bf16 rounding + routing-weight fma semantics as
    the logical256_t64 incumbent; the intra-dot reduction order differs
    (T1 vs the incumbent).
    """

    if rows <= 0 or rows > 12 or num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("rows (<=12), experts, and features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_weighted_sum_warp256_bf16_bf16_out",
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


_ARGS_IU8_RISK = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_double,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)

_ARGS_SPARSE_REPAIR = (
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
    ctypes.c_void_p,
)


def qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    output_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    risk_multiplier: float,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the risk-collecting weight-exact iu8-WMMA Q5_1 down prefill.

    Identical published arithmetic to a three-plane residual iu8 chain; the
    outputs whose BF16 rounding boundary distance falls below the Kahan
    bound (times the multiplier) are queued for the sparse exact repair,
    together with rows flagged at risk (nonfinite activations or a raw
    32-element subblock amax below 2^-80).
    """

    if compact_rows <= 0 or num_experts <= 0 or wmma_total_rows <= 0:
        raise ValueError("compact_rows, num_experts, and wmma_total_rows must be positive")
    if wmma_total_rows % 16:
        raise ValueError("wmma_total_rows must be divisible by 16")
    if in_features <= 0 or in_features % 32 or out_features <= 0:
        raise ValueError("Q5_1 iu8 risk projection has invalid feature geometry")
    if max_risks < 0:
        raise ValueError("max_risks must be non-negative")
    if not (risk_multiplier > 0.0) or risk_multiplier != risk_multiplier:
        raise ValueError("risk_multiplier must be a positive float")
    if int(risk_count_ptr) <= 0 or int(risk_indices_ptr) <= 0:
        raise ValueError("iu8 risk prefill requires risk counter and queue")
    if int(x_ptr) <= 0 or int(expert_start_compact_ptr) <= 0 or \
            int(expert_start_wmma_ptr) <= 0 or int(tile_expert_ptr) <= 0 or \
            int(qweight_ptr) <= 0 or int(output_ptr) <= 0:
        raise ValueError("iu8 risk prefill requires non-null operands")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out",
        _ARGS_IU8_RISK,
        ctypes.c_int,
    )
    error = fn(
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_ptr,
        output_ptr,
        risk_count_ptr,
        risk_indices_ptr,
        max_risks,
        risk_multiplier,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        wmma_total_rows,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
    input_ptr: int,
    expert_start_ptr: int,
    qweight_ptr: int,
    output_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    grid_blocks: int = 1024,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Repair queued at-risk iu8 Q5_1 down outputs exactly.

    Recomputes each queued output with the exact arithmetic of the
    production pair2 row-publish parent: 128 logical lanes, the even/odd
    stream split over columns lane + i*128, and the 128-entry halving tree.
    """

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 128 or out_features <= 0:
        raise ValueError("Q5_1 sparse repair requires in_features divisible by 128")
    if max_risks < 0 or grid_blocks <= 0:
        raise ValueError("max_risks must be non-negative and grid_blocks positive")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16",
        _ARGS_SPARSE_REPAIR,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        expert_start_ptr,
        qweight_ptr,
        output_ptr,
        risk_count_ptr,
        risk_indices_ptr,
        max_risks,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        grid_blocks,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def register_qwen4_exp_q5_1_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_wmma_iu8_risk_prefill_bf16_bf16_out"),
        qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_sparse_exact_repair_row_publish_bf16"),
        qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"),
        qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out"),
        qwen4_exp_q5_1_selected_grouped_prefill_pair2_register_cache_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"),
        qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
                  "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"),
        qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_bf16_bf16_out,
        replace=replace,
    )
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
    "qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_pair2_register_cache_bf16_bf16_out",
    "qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out",
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

