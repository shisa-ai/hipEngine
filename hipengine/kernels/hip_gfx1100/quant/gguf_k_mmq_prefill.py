"""Raw Q5_K/Q6_K producer-row Q8_1 MMQ32 prefill wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_mmq_prefill.hip")
_OUTPUT_NAME = "gguf_k_mmq_prefill.so"
_SOURCE_OUTPUT_NAME = "gguf_q5_k_source_mmq_prefill.so"
_QUANT_DS4_KMAJOR_SYMBOL = "hipengine_gguf_q8_1_ds4_quantize_bf16_kmajor"
_QUANT_D4S4_KMAJOR_SYMBOL = (
    "hipengine_gguf_q8_1_d4s4_f32_quantize_bf16_kmajor"
)
_QUANT_SYMBOL = "hipengine_gguf_q8_1_d4s4_f32_quantize_bf16"
_QUANT_D8_SYMBOL = "hipengine_gguf_q8_1_d8s8_f32_quantize_bf16"
_QUANT_D8R8_SYMBOL = "hipengine_gguf_q8_1_d8r8s8_f32_quantize_bf16"
_VARIANT_BF16 = "mmq32_q8_1_d4s4_f32_bf16_bf16_out"
_VARIANT_F32 = "mmq32_q8_1_d4s4_f32_bf16_f32_out"
_VARIANT_D8_BF16 = "mmq32_q8_1_d8s8_f32_bf16_bf16_out"
_VARIANT_D8_F32 = "mmq32_q8_1_d8s8_f32_bf16_f32_out"
_VARIANT_D8R8_BF16 = "mmq32_q8_1_d8r8s8_f32_bf16_bf16_out"
_VARIANT_D8R8_F32 = "mmq32_q8_1_d8r8s8_f32_bf16_f32_out"
_SOURCE_VARIANT_BF16 = "mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out"
_SOURCE_VARIANT_F32 = "mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out"
_SOURCE_C8_VARIANT_BF16 = (
    "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out"
)
_SOURCE_C8_VARIANT_F32 = (
    "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out"
)
_Q8_BLOCK = 128
_Q8_DS4_BLOCK_BYTES = 144
_Q8_BLOCK_BYTES = 160
_Q8_D8_BLOCK_BYTES = 192
_Q8_D8R8_BLOCK_BYTES = 352
_SOURCE_MMQ_FLAGS = (
    "-funsafe-math-optimizations",
    "-ffast-math",
    "-fno-finite-math-only",
)


def plan_gguf_k_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_mmq_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_k_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def plan_gguf_q5_k_source_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q5_k_source_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_SOURCE_MMQ_FLAGS,
        output_name=_SOURCE_OUTPUT_NAME,
    )


def build_gguf_q5_k_source_mmq_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    """Build the source Q5 consumer with llama.cpp's MMQ math flags."""

    return build_hip(
        sources=[_SOURCE],
        family="gguf_q5_k_source_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_SOURCE_MMQ_FLAGS,
        output_name=_SOURCE_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def q8_1_ds4_kmajor_nbytes(rows: int, hidden: int) -> int:
    """Return bytes for llama.cpp-compatible K-major DS4 Q8_1 blocks."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % _Q8_BLOCK != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    return rows * (hidden // _Q8_BLOCK) * _Q8_DS4_BLOCK_BYTES


def q8_1_d4s4_f32_nbytes(rows: int, hidden: int) -> int:
    """Return bytes for row-major K128 Q8_1 blocks with FP32 scale/sum."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % _Q8_BLOCK != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    return rows * (hidden // _Q8_BLOCK) * _Q8_BLOCK_BYTES


def q8_1_d4s4_f32_kmajor_nbytes(rows: int, hidden: int) -> int:
    """Return bytes for K-major K128 Q8_1 blocks with FP32 scale/sum."""

    return q8_1_d4s4_f32_nbytes(rows, hidden)


def q8_1_d8s8_f32_nbytes(rows: int, hidden: int) -> int:
    """Return bytes for row-major K128 Q8_1 blocks with eight FP32 groups."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % _Q8_BLOCK != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    return rows * (hidden // _Q8_BLOCK) * _Q8_D8_BLOCK_BYTES


def q8_1_d8r8s8_f32_nbytes(rows: int, hidden: int) -> int:
    """Return bytes for two-stage per-K16 Q8_1 residual blocks."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % _Q8_BLOCK != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    return rows * (hidden // _Q8_BLOCK) * _Q8_D8R8_BLOCK_BYTES


def gguf_q8_1_ds4_quantize_bf16_kmajor(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 rows into source-compatible K-major FP16 DS4 records."""

    q8_1_ds4_kmajor_nbytes(rows, hidden)
    library = library or build_gguf_k_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_DS4_KMAJOR_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    error = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q8_1_d4s4_f32_quantize_bf16_kmajor(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Quantize BF16 rows to precision-preserving K-major Q8_1 blocks."""

    q8_1_d4s4_f32_kmajor_nbytes(rows, hidden)
    library = library or build_gguf_k_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_D4S4_KMAJOR_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    error = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q8_1_d4s4_f32_quantize_bf16(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Quantize each BF16 producer row once to range-safe Q8_1 blocks."""

    q8_1_d4s4_f32_nbytes(rows, hidden)
    library = library or build_gguf_k_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    error = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q8_1_d8s8_f32_quantize_bf16(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Quantize each BF16 producer row to per-K16 Q8_1 groups."""

    q8_1_d8s8_f32_nbytes(rows, hidden)
    library = library or build_gguf_k_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_D8_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    error = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q8_1_d8r8s8_f32_quantize_bf16(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Quantize BF16 rows to primary plus residual per-K16 Q8_1 groups."""

    q8_1_d8r8s8_f32_nbytes(rows, hidden)
    library = library or build_gguf_k_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_D8R8_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    error = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _launch_mmq(
    quant: str,
    output_dtype: str,
    xq_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    producer_layout: str = "d4s4",
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    library = library or build_gguf_k_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    symbol = (
        f"hipengine_{quant}_mmq32_q8_1_{producer_layout}_f32_"
        f"bf16_{output_dtype}_out"
    )
    fn = getattr(library, symbol)
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
    error = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _launch_q5_source_mmq(
    output_dtype: str,
    xq_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    tile: str = "i128_j128",
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if tile not in {"i128_j128", "i64_j32", "i64_j16"}:
        raise ValueError(f"unsupported source MMQ tile: {tile}")
    library = library or build_gguf_q5_k_source_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    producer_layout = (
        "ds4" if tile == "i128_j128" else "d4s4_f32_kmajor"
    )
    symbol = (
        f"hipengine_gguf_q5_k_mmq_{tile}_k256_q8_1_{producer_layout}_"
        f"bf16_{output_dtype}_out"
    )
    fn = getattr(library, symbol)
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
    error = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_q5_source_mmq("bf16", *args, **kwargs)


def gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out(
    *args, **kwargs
) -> None:
    _launch_q5_source_mmq("f32", *args, **kwargs)


def gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_q5_source_mmq("bf16", *args, tile="i64_j32", **kwargs)


def gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out(
    *args, **kwargs
) -> None:
    _launch_q5_source_mmq("f32", *args, tile="i64_j32", **kwargs)


def gguf_q5_k_mmq_i64_j16_forced_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
    *args, **kwargs
) -> None:
    """Forced J16 minitile launch (C8-P2 Q5 candidate leaf)."""
    _launch_q5_source_mmq("bf16", *args, tile="i64_j16", **kwargs)


def gguf_q5_k_mmq_i64_j16_forced_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out(
    *args, **kwargs
) -> None:
    _launch_q5_source_mmq("f32", *args, tile="i64_j16", **kwargs)


def gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "bf16", *args, **kwargs)


def gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "f32", *args, **kwargs)


def gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "bf16", *args, **kwargs)


def gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "f32", *args, **kwargs)


def gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "bf16", *args, producer_layout="d8s8", **kwargs)


def gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "f32", *args, producer_layout="d8s8", **kwargs)


def gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "bf16", *args, producer_layout="d8s8", **kwargs)


def gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "f32", *args, producer_layout="d8s8", **kwargs)


def gguf_q5_k_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "bf16", *args, producer_layout="d8r8s8", **kwargs)


def gguf_q5_k_mmq32_q8_1_d8r8s8_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "f32", *args, producer_layout="d8r8s8", **kwargs)


def gguf_q6_k_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "bf16", *args, producer_layout="d8r8s8", **kwargs)


def gguf_q6_k_mmq32_q8_1_d8r8s8_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "f32", *args, producer_layout="d8r8s8", **kwargs)


def register_gguf_k_mmq_prefill_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "activation_quant", "q8_1_ds4", "bf16_kmajor"),
        gguf_q8_1_ds4_quantize_bf16_kmajor,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "activation_quant", "q8_1_d4s4_f32", "bf16_kmajor"
        ),
        gguf_q8_1_d4s4_f32_quantize_bf16_kmajor,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "activation_quant", "q8_1_d4s4_f32", "bf16"),
        gguf_q8_1_d4s4_f32_quantize_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "activation_quant", "q8_1_d8s8_f32", "bf16"),
        gguf_q8_1_d8s8_f32_quantize_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "activation_quant", "q8_1_d8r8s8_f32", "bf16"),
        gguf_q8_1_d8r8s8_f32_quantize_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear", "gguf_q5_k", _SOURCE_VARIANT_BF16
        ),
        gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear", "gguf_q5_k", _SOURCE_VARIANT_F32
        ),
        gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear", "gguf_q5_k", _SOURCE_C8_VARIANT_BF16
        ),
        gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear", "gguf_q5_k", _SOURCE_C8_VARIANT_F32
        ),
        gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out,
        replace=replace,
    )
    for quant, bf16_fn, f32_fn in (
        (
            "gguf_q5_k",
            gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
            gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
        ),
        (
            "gguf_q6_k",
            gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
            gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "linear", quant, _VARIANT_BF16),
            bf16_fn,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "linear", quant, _VARIANT_F32),
            f32_fn,
            replace=replace,
        )
    for quant, bf16_fn, f32_fn in (
        (
            "gguf_q5_k",
            gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out,
            gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_f32_out,
        ),
        (
            "gguf_q6_k",
            gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out,
            gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_f32_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "linear", quant, _VARIANT_D8_BF16),
            bf16_fn,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "linear", quant, _VARIANT_D8_F32),
            f32_fn,
            replace=replace,
        )
    for quant, bf16_fn, f32_fn in (
        (
            "gguf_q5_k",
            gguf_q5_k_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out,
            gguf_q5_k_mmq32_q8_1_d8r8s8_f32_bf16_f32_out,
        ),
        (
            "gguf_q6_k",
            gguf_q6_k_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out,
            gguf_q6_k_mmq32_q8_1_d8r8s8_f32_bf16_f32_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "linear", quant, _VARIANT_D8R8_BF16),
            bf16_fn,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "linear", quant, _VARIANT_D8R8_F32),
            f32_fn,
            replace=replace,
        )


register_gguf_k_mmq_prefill_kernels()


__all__ = [
    "build_gguf_k_mmq_prefill",
    "build_gguf_q5_k_source_mmq_prefill",
    "gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out",
    "gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out",
    "gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out",
    "gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out",
    "gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out",
    "gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out",
    "gguf_q5_k_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out",
    "gguf_q5_k_mmq32_q8_1_d8r8s8_f32_bf16_f32_out",
    "gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out",
    "gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_f32_out",
    "gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out",
    "gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out",
    "gguf_q6_k_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out",
    "gguf_q6_k_mmq32_q8_1_d8r8s8_f32_bf16_f32_out",
    "gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out",
    "gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_f32_out",
    "gguf_q8_1_d4s4_f32_quantize_bf16",
    "gguf_q8_1_d4s4_f32_quantize_bf16_kmajor",
    "gguf_q8_1_ds4_quantize_bf16_kmajor",
    "gguf_q8_1_d8r8s8_f32_quantize_bf16",
    "gguf_q8_1_d8s8_f32_quantize_bf16",
    "plan_gguf_k_mmq_prefill_build",
    "plan_gguf_q5_k_source_mmq_prefill_build",
    "q8_1_d4s4_f32_kmajor_nbytes",
    "q8_1_d4s4_f32_nbytes",
    "q8_1_ds4_kmajor_nbytes",
    "q8_1_d8r8s8_f32_nbytes",
    "q8_1_d8s8_f32_nbytes",
    "register_gguf_k_mmq_prefill_kernels",
]
