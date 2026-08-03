"""Raw-pointer Moonshine per-output-channel symmetric W8A16 kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_w8a16.hip")
_OUTPUT_NAME = "moonshine_w8a16.so"
_ALLOWED_THREADS = {32, 64, 128, 256}
_LM_HEAD_ARGS = (
    *(ctypes.c_void_p for _ in range(4)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_PROJECTION_ARGS = (
    *(ctypes.c_void_p for _ in range(4)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_MLP_FC1_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_MLP_FC2_ARGS = (
    *(ctypes.c_void_p for _ in range(6)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_TRIPLE_ARGS = (
    *(ctypes.c_void_p for _ in range(10)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_PAIR_HEAD_MAJOR_ARGS = (
    *(ctypes.c_void_p for _ in range(7)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_w8a16_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="moonshine_w8a16",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_w8a16(
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
        family="moonshine_w8a16",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _validate(rows: int, in_features: int, outputs: tuple[int, ...]) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if any(output <= 0 for output in outputs):
        raise ValueError("out_features must be positive")


def _validate_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")


def _call(
    library: object | None,
    runtime: HipRuntime | None,
    symbol: str,
    argtypes: tuple[type[ctypes._SimpleCData], ...],
    args: tuple[object, ...],
) -> None:
    loaded = library or build_moonshine_w8a16(load=True)
    hip = runtime or get_hip_runtime()
    function = signed_kernel_fn(loaded, symbol, argtypes, ctypes.c_int)
    error = function(*args)
    if int(error) != HIP_SUCCESS:
        hip.check(int(error))


def moonshine_w8a16_lm_head_wave8(
    input_ptr: int,
    qweight_ptr: int,
    scale_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: object | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,))
    _call(
        library,
        runtime,
        "hipengine_moonshine_w8a16_lm_head_wave8",
        _LM_HEAD_ARGS,
        (input_ptr, qweight_ptr, scale_ptr, output_ptr, rows, in_features, out_features, stream),
    )


def moonshine_w8a16_projection(
    input_ptr: int,
    qweight_ptr: int,
    scale_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: object | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,))
    _validate_threads(threads)
    _call(
        library,
        runtime,
        "hipengine_moonshine_w8a16_projection",
        _PROJECTION_ARGS,
        (
            input_ptr, qweight_ptr, scale_ptr, output_ptr, rows, in_features,
            out_features, threads, stream,
        ),
    )


def moonshine_w8a16_mlp_fc1_gated_silu(
    input_ptr: int,
    qweight_ptr: int,
    scale_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    intermediate_size: int,
    *,
    stream: int = 0,
    library: object | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (intermediate_size,))
    _call(
        library,
        runtime,
        "hipengine_moonshine_w8a16_mlp_fc1_gated_silu",
        _MLP_FC1_ARGS,
        (
            input_ptr, qweight_ptr, scale_ptr, bias_ptr, output_ptr, rows,
            in_features, intermediate_size, stream,
        ),
    )


def moonshine_w8a16_mlp_fc2_residual(
    input_ptr: int,
    qweight_ptr: int,
    scale_ptr: int,
    bias_ptr: int,
    residual_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: object | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,))
    _validate_threads(threads)
    _call(
        library,
        runtime,
        "hipengine_moonshine_w8a16_mlp_fc2_residual",
        _MLP_FC2_ARGS,
        (
            input_ptr, qweight_ptr, scale_ptr, bias_ptr, residual_ptr, output_ptr,
            rows, in_features, out_features, threads, stream,
        ),
    )


def moonshine_w8a16_qkv_triple(
    input_ptr: int,
    qweight_a_ptr: int,
    scale_a_ptr: int,
    qweight_b_ptr: int,
    scale_b_ptr: int,
    qweight_c_ptr: int,
    scale_c_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    output_c_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    out_c_features: int,
    *,
    threads: int = 32,
    stream: int = 0,
    library: object | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_a_features, out_b_features, out_c_features))
    _validate_threads(threads)
    _call(
        library,
        runtime,
        "hipengine_moonshine_w8a16_qkv_triple",
        _TRIPLE_ARGS,
        (
            input_ptr, qweight_a_ptr, scale_a_ptr, qweight_b_ptr, scale_b_ptr,
            qweight_c_ptr, scale_c_ptr, output_a_ptr, output_b_ptr, output_c_ptr,
            rows, in_features, out_a_features, out_b_features, out_c_features,
            threads, stream,
        ),
    )


def moonshine_w8a16_cross_kv_pair_head_major(
    input_ptr: int,
    qweight_a_ptr: int,
    scale_a_ptr: int,
    qweight_b_ptr: int,
    scale_b_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    head_dim: int,
    *,
    threads: int = 32,
    stream: int = 0,
    library: object | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_a_features, out_b_features))
    _validate_threads(threads)
    if head_dim <= 0 or out_a_features % head_dim or out_b_features % head_dim:
        raise ValueError("head_dim must positively divide both output widths")
    _call(
        library,
        runtime,
        "hipengine_moonshine_w8a16_cross_kv_pair_head_major",
        _PAIR_HEAD_MAJOR_ARGS,
        (
            input_ptr, qweight_a_ptr, scale_a_ptr, qweight_b_ptr, scale_b_ptr,
            output_a_ptr, output_b_ptr, rows, in_features, out_a_features,
            out_b_features, head_dim, threads, stream,
        ),
    )


def register_moonshine_w8a16_kernels(*, replace: bool = True) -> None:
    registrations = (
        ("moonshine_lm_head", "tied_wave8_per_output_f32_scale", moonshine_w8a16_lm_head_wave8),
        ("moonshine_projection", "single_per_output_f32_scale", moonshine_w8a16_projection),
        ("moonshine_mlp_fc1", "bias_gated_silu_per_output_f32_scale", moonshine_w8a16_mlp_fc1_gated_silu),
        ("moonshine_mlp_fc2_residual", "bias_rounded_residual_per_output_f32_scale", moonshine_w8a16_mlp_fc2_residual),
        ("moonshine_qkv_proj", "triple_per_output_f32_scale", moonshine_w8a16_qkv_triple),
        ("moonshine_cross_kv_precompute", "pair_head_major_per_output_f32_scale", moonshine_w8a16_cross_kv_pair_head_major),
    )
    for layer, variant, function in registrations:
        register(
            KernelKey("hip_gfx1100", layer, "w8a16", variant),
            function,
            replace=replace,
        )


register_moonshine_w8a16_kernels()


__all__ = [
    "build_moonshine_w8a16",
    "moonshine_w8a16_cross_kv_pair_head_major",
    "moonshine_w8a16_lm_head_wave8",
    "moonshine_w8a16_mlp_fc1_gated_silu",
    "moonshine_w8a16_mlp_fc2_residual",
    "moonshine_w8a16_projection",
    "moonshine_w8a16_qkv_triple",
    "plan_moonshine_w8a16_build",
    "register_moonshine_w8a16_kernels",
]
