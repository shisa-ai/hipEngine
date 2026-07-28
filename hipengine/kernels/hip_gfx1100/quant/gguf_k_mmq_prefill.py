"""Raw Q5_K/Q6_K producer-row Q8_1 MMQ32 prefill wrappers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_mmq_prefill.hip")
_OUTPUT_NAME = "gguf_k_mmq_prefill.so"
_QUANT_SYMBOL = "hipengine_gguf_q8_1_d4s4_f32_quantize_bf16"
_VARIANT_BF16 = "mmq32_q8_1_d4s4_f32_bf16_bf16_out"
_VARIANT_F32 = "mmq32_q8_1_d4s4_f32_bf16_f32_out"
_Q8_BLOCK = 128
_Q8_BLOCK_BYTES = 160


@dataclass(frozen=True)
class RawKMMQPrefillPolicy:
    """Quant-plugin crossover and variant selection for raw-K MMQ32."""

    min_rows: int = 9
    min_out_features: int = 1_024

    def variant(
        self,
        source_variant: str,
        rows: int,
        hidden: int,
        out_features: int,
    ) -> str | None:
        if (
            int(rows) < self.min_rows
            or int(hidden) <= 0
            or int(hidden) % 256 != 0
            or int(out_features) < self.min_out_features
        ):
            return None
        return {
            "prefill_bf16_bf16_out": _VARIANT_BF16,
            "prefill_bf16_f32_out": _VARIANT_F32,
        }.get(str(source_variant))


_RAW_K_MMQ_PREFILL_POLICY = RawKMMQPrefillPolicy()


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


def q8_1_d4s4_f32_nbytes(rows: int, hidden: int) -> int:
    """Return bytes for row-major K128 Q8_1 blocks with FP32 scale/sum."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % _Q8_BLOCK != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    return rows * (hidden // _Q8_BLOCK) * _Q8_BLOCK_BYTES


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
    symbol = f"hipengine_{quant}_mmq32_q8_1_d4s4_f32_bf16_{output_dtype}_out"
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


def gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "bf16", *args, **kwargs)


def gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q5_k", "f32", *args, **kwargs)


def gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "bf16", *args, **kwargs)


def gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(*args, **kwargs) -> None:
    _launch_mmq("gguf_q6_k", "f32", *args, **kwargs)


def register_gguf_k_mmq_prefill_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "activation_quant", "q8_1_d4s4_f32", "bf16"),
        gguf_q8_1_d4s4_f32_quantize_bf16,
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
            KernelKey(
                "hip_gfx1100",
                "linear_prefill_policy",
                quant,
                "raw_k_q8_1_mmq32",
            ),
            _RAW_K_MMQ_PREFILL_POLICY,
            replace=replace,
        )
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


register_gguf_k_mmq_prefill_kernels()


__all__ = [
    "RawKMMQPrefillPolicy",
    "build_gguf_k_mmq_prefill",
    "gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out",
    "gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out",
    "gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out",
    "gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out",
    "gguf_q8_1_d4s4_f32_quantize_bf16",
    "plan_gguf_k_mmq_prefill_build",
    "q8_1_d4s4_f32_nbytes",
    "register_gguf_k_mmq_prefill_kernels",
]
