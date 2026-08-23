"""Experimental gfx1151 packed-U4 x S4 DOT8/WMMA sidecar wrappers.

This T3 research family is intentionally not wired into model/runtime routing.
The current exact qmicro Q4_K_S gate/up owner remains the declared fallback.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, is_registered, register

_SOURCE = Path(__file__).with_name("iu4_s4_sidecar.hip")
_OUTPUT_NAME = "iu4_s4_sidecar.so"
_QUANT_SYMBOL = "hipengine_iu4_u4_quantize_bf16"
_PROBE_SYMBOL = "hipengine_iu4_s4_matmul_i32_probe"
_DUAL_SYMBOL = "hipengine_iu4_s4_dual_silu_bf16_out"

IU4_U4_QUANT_KEY = KernelKey(
    "hip_gfx1151",
    "activation_quant",
    "iu4_u4_row_v1",
    "bf16_asym_per_row",
)
IU4_S4_DUAL_SILU_KEY = KernelKey(
    "hip_gfx1151",
    "linear_pair_silu",
    "iu4_s4_sidecar_v1",
    "dot8_m2_m16_wmma_bulk_m17_m1024_bf16_out",
)
IU4_S4_STRICT_FALLBACK_KEY = KernelKey(
    "hip_gfx1151",
    "linear_pair_silu",
    "gguf_q4_k_qmicro_t16_v1",
    "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out",
)


def plan_iu4_s4_sidecar_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gfx1151_iu4_s4_sidecar",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch="gfx1151",
        output_name=_OUTPUT_NAME,
    )


def build_iu4_s4_sidecar(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gfx1151_iu4_s4_sidecar",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch="gfx1151",
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def iu4_u4_wmma_nbytes(rows: int, hidden: int) -> int:
    _check_shape(rows, hidden)
    return ((int(rows) + 15) // 16) * 16 * int(hidden) // 2


def _check_shape(rows: int, hidden: int, out_features: int | None = None) -> None:
    if rows <= 0 or rows > 1024:
        raise ValueError("IU4 research rows must be in [1, 1024]")
    if hidden <= 0 or hidden % 32:
        raise ValueError("IU4 hidden size must be a positive multiple of 32")
    if out_features is not None and (out_features <= 0 or out_features % 16):
        raise ValueError("IU4 output size must be a positive multiple of 16")


def _check_status(status: int, runtime: HipRuntime, symbol: str) -> None:
    if int(status) != HIP_SUCCESS:
        raise RuntimeError(
            f"{symbol} failed with HIP status {status}: {runtime.error_string(int(status))}"
        )


def iu4_u4_quantize_bf16(
    input_ptr: int,
    packed_ptr: int,
    scales_ptr: int,
    zero_points_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, hidden)
    library = library or build_iu4_s4_sidecar(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(packed_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(zero_points_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    _check_status(status, runtime, _QUANT_SYMBOL)


def iu4_s4_matmul_i32_probe(
    activations_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, hidden, out_features)
    library = library or build_iu4_s4_sidecar(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PROBE_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(activations_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    _check_status(status, runtime, _PROBE_SYMBOL)


def iu4_s4_dual_silu_bf16_out(
    activations_ptr: int,
    activation_scales_ptr: int,
    activation_zero_points_ptr: int,
    gate_weights_ptr: int,
    gate_scales_ptr: int,
    gate_sums_ptr: int,
    up_weights_ptr: int,
    up_scales_ptr: int,
    up_sums_ptr: int,
    output_ptr: int,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, hidden, out_features)
    if rows < 2:
        raise ValueError("IU4 gate/up candidate deliberately excludes M=1")
    library = library or build_iu4_s4_sidecar(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _DUAL_SYMBOL)
    fn.argtypes = [ctypes.c_void_p] * 10 + [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(activations_ptr),
        ctypes.c_void_p(activation_scales_ptr),
        ctypes.c_void_p(activation_zero_points_ptr),
        ctypes.c_void_p(gate_weights_ptr),
        ctypes.c_void_p(gate_scales_ptr),
        ctypes.c_void_p(gate_sums_ptr),
        ctypes.c_void_p(up_weights_ptr),
        ctypes.c_void_p(up_scales_ptr),
        ctypes.c_void_p(up_sums_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    _check_status(status, runtime, _DUAL_SYMBOL)


def register_iu4_s4_sidecar_kernels(*, replace: bool = True) -> None:
    if replace or not is_registered(IU4_U4_QUANT_KEY):
        register(IU4_U4_QUANT_KEY, iu4_u4_quantize_bf16, replace=replace)
    if replace or not is_registered(IU4_S4_DUAL_SILU_KEY):
        register(IU4_S4_DUAL_SILU_KEY, iu4_s4_dual_silu_bf16_out, replace=replace)


__all__ = [
    "IU4_S4_DUAL_SILU_KEY",
    "IU4_S4_STRICT_FALLBACK_KEY",
    "IU4_U4_QUANT_KEY",
    "build_iu4_s4_sidecar",
    "iu4_s4_dual_silu_bf16_out",
    "iu4_s4_matmul_i32_probe",
    "iu4_u4_quantize_bf16",
    "iu4_u4_wmma_nbytes",
    "plan_iu4_s4_sidecar_build",
    "register_iu4_s4_sidecar_kernels",
]
