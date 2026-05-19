"""Raw-pointer wrappers for GGUF Q5_K/Q6_K selected compact WMMA prefill.

P8.5 selected MoE down-projection kernels. These wrappers use the same compact
scheduler ABI as P8.4 selected Q4_K:

``x[compact_rows, in_features]``, ``expert_start_compact[E+1]``,
``expert_start_wmma[E+1]``, ``tile_expert[wmma_total_rows/16]``, raw rank-3
Q5_K/Q6_K expert weights ``[E, out_features, raw_bytes_per_row]``, and one
row-major output ``[compact_rows, out_features]``.

No resident repack/sidecar is introduced; raw GGUF expert bytes stay on device
and are dequantized in-register inside the WMMA K-loop.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_selected_prefill.hip")
_OUTPUT_NAME = "gguf_k_selected_prefill.so"
_QUANTS = ("gguf_q5_k", "gguf_q6_k")
_QTYPE_BLOCK_SIZE = {"gguf_q5_k": 256, "gguf_q6_k": 256}
_ALLOWED_TILES = {(16, 16), (32, 16), (16, 32), (32, 32), (64, 16), (64, 32)}
_ENV_TILE_M = {
    "gguf_q5_k": "HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_M",
    "gguf_q6_k": "HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_M",
}
_ENV_TILE_N = {
    "gguf_q5_k": "HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_N",
    "gguf_q6_k": "HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_N",
}
_ENV_LAUNCH_BOUNDS = "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS"
_SYMBOLS = {
    ("gguf_q5_k", "bf16"): "hipengine_gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q5_k", "fp16"): "hipengine_gguf_q5_k_selected_wmma_prefill_compact_fp16_fp16_out",
    ("gguf_q6_k", "bf16"): "hipengine_gguf_q6_k_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q6_k", "fp16"): "hipengine_gguf_q6_k_selected_wmma_prefill_compact_fp16_fp16_out",
}
_SYMBOLS_Q5_OPT = {
    "bf16": "hipengine_gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out",
    "fp16": "hipengine_gguf_q5_k_selected_wmma_prefill_compact_opt_fp16_fp16_out",
}


def plan_gguf_k_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_selected_prefill(
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
        family="gguf_k_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _extra_flags() -> tuple[str, ...]:
    value = os.environ.get(_ENV_LAUNCH_BOUNDS)
    if not value:
        return ("-mcumode",)
    min_blocks = int(value)
    if min_blocks not in {1, 2, 4, 8}:
        raise ValueError(f"{_ENV_LAUNCH_BOUNDS} must be one of 1, 2, 4, 8")
    return ("-mcumode", f"-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS={min_blocks}")


def _make_wrapper(quant: str, dtype: str):
    symbol = _SYMBOLS[(quant, dtype)]

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        expert_start_wmma_ptr: int,
        tile_expert_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        compact_rows: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        wmma_total_rows: int,
        *,
        tile_m: int | None = None,
        tile_n: int | None = None,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _launch_selected(
            quant,
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            expert_start_wmma_ptr,
            tile_expert_ptr,
            qweight_ptr,
            out_ptr,
            compact_rows,
            in_features,
            out_features,
            num_experts,
            wmma_total_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"{quant}_selected_wmma_prefill_compact_{dtype}_{dtype}_out"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = f"Launch {quant} selected compact WMMA prefill ({dtype}->{dtype})."
    return wrapper


# Public wrapper functions.
gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q5_k", "bf16")
gguf_q5_k_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q5_k", "fp16")
gguf_q6_k_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q6_k", "bf16")
gguf_q6_k_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q6_k", "fp16")


def _make_q5_opt_wrapper(dtype: str):
    symbol = _SYMBOLS_Q5_OPT[dtype]

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        expert_start_wmma_ptr: int,
        tile_expert_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
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
        _launch_q5_opt(
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            expert_start_wmma_ptr,
            tile_expert_ptr,
            qweight_ptr,
            out_ptr,
            compact_rows,
            in_features,
            out_features,
            num_experts,
            wmma_total_rows,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"gguf_q5_k_selected_wmma_prefill_compact_opt_{dtype}_{dtype}_out"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = f"Launch optimized {dtype} Q5_K selected compact WMMA prefill."
    return wrapper


gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out = _make_q5_opt_wrapper("bf16")
gguf_q5_k_selected_wmma_prefill_compact_opt_fp16_fp16_out = _make_q5_opt_wrapper("fp16")


def _launch_q5_opt(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common("gguf_q5_k", compact_rows, in_features, out_features, num_experts, wmma_total_rows)
    library = library or build_gguf_k_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_selected(
    quant: str,
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    tile_m: int | None,
    tile_n: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    tile_m, tile_n = _resolve_tiles(quant, tile_m, tile_n)
    _check_common(quant, compact_rows, in_features, out_features, num_experts, wmma_total_rows)
    library = library or build_gguf_k_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_int64(tile_m),
        ctypes.c_int64(tile_n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _resolve_tiles(quant: str, tile_m: int | None, tile_n: int | None) -> tuple[int, int]:
    if quant not in _QTYPE_BLOCK_SIZE:
        raise ValueError(f"unsupported quant: {quant}")
    if tile_m is None:
        value = os.environ.get(_ENV_TILE_M[quant])
        tile_m = int(value) if value else 16
    if tile_n is None:
        value = os.environ.get(_ENV_TILE_N[quant])
        tile_n = int(value) if value else 16
    if (tile_m, tile_n) not in _ALLOWED_TILES:
        allowed = ", ".join(f"({m}, {n})" for m, n in sorted(_ALLOWED_TILES))
        raise ValueError(
            f"tile (tile_m={tile_m}, tile_n={tile_n}) is not supported; "
            f"supported tiles: {allowed}"
        )
    return tile_m, tile_n


def selected_wmma_prefill_compact_default_tiles(quant: str) -> tuple[int, int]:
    """Return the P9.C1 default tile for Q5_K/Q6_K selected WMMA prefill.

    The generic multi-tile variants remain available for sweeps via env
    overrides, but measured qwen35moe 512/0 evidence keeps the legacy 16x16
    kernel as the default.
    """

    return _resolve_tiles(quant, None, None)


def _check_common(
    quant: str,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    if quant not in _QTYPE_BLOCK_SIZE:
        raise ValueError(f"unsupported quant: {quant}")
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    block = _QTYPE_BLOCK_SIZE[quant]
    if in_features % block != 0:
        raise ValueError(f"in_features must be divisible by GGUF {quant} block size {block}")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


_WRAPPERS = {
    "gguf_q5_k": {
        "selected_wmma_prefill_compact_bf16_bf16_out": gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out,
        "selected_wmma_prefill_compact_fp16_fp16_out": gguf_q5_k_selected_wmma_prefill_compact_fp16_fp16_out,
        "selected_wmma_prefill_compact_opt_bf16_bf16_out": gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out,
        "selected_wmma_prefill_compact_opt_fp16_fp16_out": gguf_q5_k_selected_wmma_prefill_compact_opt_fp16_fp16_out,
        "selected_wmma_prefill_bf16_bf16_out": gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out,
        "selected_wmma_prefill_fp16_fp16_out": gguf_q5_k_selected_wmma_prefill_compact_fp16_fp16_out,
    },
    "gguf_q6_k": {
        "selected_wmma_prefill_compact_bf16_bf16_out": gguf_q6_k_selected_wmma_prefill_compact_bf16_bf16_out,
        "selected_wmma_prefill_compact_fp16_fp16_out": gguf_q6_k_selected_wmma_prefill_compact_fp16_fp16_out,
        "selected_wmma_prefill_bf16_bf16_out": gguf_q6_k_selected_wmma_prefill_compact_bf16_bf16_out,
        "selected_wmma_prefill_fp16_fp16_out": gguf_q6_k_selected_wmma_prefill_compact_fp16_fp16_out,
    },
}


def register_gguf_k_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register P8.5 selected raw-Q5_K/Q6_K WMMA down kernels."""

    for quant, variants in _WRAPPERS.items():
        for variant, fn in variants.items():
            register(KernelKey("hip_gfx1100", "moe_linear", quant, variant), fn, replace=replace)


register_gguf_k_selected_prefill_kernels()


__all__ = [
    "build_gguf_k_selected_prefill",
    "gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q5_k_selected_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out",
    "gguf_q5_k_selected_wmma_prefill_compact_opt_fp16_fp16_out",
    "gguf_q6_k_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q6_k_selected_wmma_prefill_compact_fp16_fp16_out",
    "plan_gguf_k_selected_prefill_build",
    "register_gguf_k_selected_prefill_kernels",
    "selected_wmma_prefill_compact_default_tiles",
]
