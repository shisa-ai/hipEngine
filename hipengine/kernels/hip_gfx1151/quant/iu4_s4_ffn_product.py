"""gfx1151 Hadamard U4 x published-PFS S4 FFN research wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, is_registered, register

_SOURCE = Path(__file__).with_name("iu4_s4_ffn_product.hip")
_OUTPUT_NAME = "iu4_s4_ffn_product.so"
_GATE_PACK_SYMBOL = "hipengine_iu4_pfs_pack_gate_bf16"
_DOWN_PACK_SYMBOL = "hipengine_iu4_pfs_pack_swiglu_down_bf16"
_LINEAR_SYMBOL = "hipengine_iu4_pfs_linear_bf16_out"

IU4_PFS_GATE_PACK_KEY = KernelKey(
    "hip_gfx1151",
    "activation_quant",
    "iu4_s4_kairic_ffn_v1",
    "bf16_block_hadamard1024_gate_u4",
)
IU4_PFS_DOWN_PACK_KEY = KernelKey(
    "hip_gfx1151",
    "activation_quant_swiglu",
    "iu4_s4_kairic_ffn_v1",
    "bf16_block_hadamard1024_down_u4",
)
IU4_PFS_LINEAR_KEY = KernelKey(
    "hip_gfx1151",
    "linear",
    "iu4_s4_kairic_ffn_v1",
    "pfs_wmma_m256_n64_bf16_out",
)


def plan_iu4_s4_ffn_product_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gfx1151_iu4_s4_ffn_product",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch="gfx1151",
        output_name=_OUTPUT_NAME,
    )


def build_iu4_s4_ffn_product(
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
        family="gfx1151_iu4_s4_ffn_product",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch="gfx1151",
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def iu4_pfs_packed_nbytes(rows: int, hidden: int) -> int:
    _check_rows(rows)
    if hidden <= 0 or hidden % 256:
        raise ValueError("PFS IU4 hidden must be a positive multiple of 256")
    return int(rows) * int(hidden) // 2


def _check_rows(rows: int) -> None:
    if rows <= 0 or rows > 2048:
        raise ValueError("PFS IU4 rows must be in [1, 2048]")


def _check_status(status: int, runtime: HipRuntime, symbol: str) -> None:
    if int(status) != HIP_SUCCESS:
        raise RuntimeError(
            f"{symbol} failed with HIP status {status}: {runtime.error_string(int(status))}"
        )


def _pack(
    symbol: str,
    input_ptr: int,
    packed_ptr: int,
    scales_ptr: int,
    zero_points_ptr: int,
    rows: int,
    width: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_rows(rows)
    library = library or build_iu4_s4_ffn_product(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p] * 4 + [
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
        ctypes.c_int64(width),
        ctypes.c_void_p(stream),
    )
    _check_status(status, runtime, symbol)


def iu4_pfs_pack_gate_bf16(
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
    if hidden not in (1024, 5120):
        raise ValueError("PFS gate Hadamard pack supports hidden 1024 test or 5120 product")
    _pack(
        _GATE_PACK_SYMBOL,
        input_ptr,
        packed_ptr,
        scales_ptr,
        zero_points_ptr,
        rows,
        hidden,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def iu4_pfs_pack_swiglu_down_bf16(
    gate_up_ptr: int,
    packed_ptr: int,
    scales_ptr: int,
    zero_points_ptr: int,
    rows: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if width != 17408:
        raise ValueError("PFS down SwiGLU Hadamard pack requires width 17408")
    _pack(
        _DOWN_PACK_SYMBOL,
        gate_up_ptr,
        packed_ptr,
        scales_ptr,
        zero_points_ptr,
        rows,
        width,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def iu4_pfs_linear_bf16_out(
    activations_ptr: int,
    activation_scales_ptr: int,
    activation_zero_points_ptr: int,
    weights_ptr: int,
    weight_scales_ptr: int,
    weight_sums_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_rows(rows)
    if in_features <= 0 or in_features % 256:
        raise ValueError("PFS linear K must be a positive multiple of 256")
    if out_features <= 0 or out_features % 64:
        raise ValueError("PFS linear N must be a positive multiple of 64")
    library = library or build_iu4_s4_ffn_product(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _LINEAR_SYMBOL)
    fn.argtypes = [ctypes.c_void_p] * 7 + [
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
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(weight_scales_ptr),
        ctypes.c_void_p(weight_sums_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    _check_status(status, runtime, _LINEAR_SYMBOL)


def register_iu4_s4_ffn_product_kernels(*, replace: bool = True) -> None:
    for key, fn in (
        (IU4_PFS_GATE_PACK_KEY, iu4_pfs_pack_gate_bf16),
        (IU4_PFS_DOWN_PACK_KEY, iu4_pfs_pack_swiglu_down_bf16),
        (IU4_PFS_LINEAR_KEY, iu4_pfs_linear_bf16_out),
    ):
        if replace or not is_registered(key):
            register(key, fn, replace=replace)


__all__ = [
    "IU4_PFS_DOWN_PACK_KEY",
    "IU4_PFS_GATE_PACK_KEY",
    "IU4_PFS_LINEAR_KEY",
    "build_iu4_s4_ffn_product",
    "iu4_pfs_linear_bf16_out",
    "iu4_pfs_pack_gate_bf16",
    "iu4_pfs_pack_swiglu_down_bf16",
    "iu4_pfs_packed_nbytes",
    "plan_iu4_s4_ffn_product_build",
    "register_iu4_s4_ffn_product_kernels",
]
