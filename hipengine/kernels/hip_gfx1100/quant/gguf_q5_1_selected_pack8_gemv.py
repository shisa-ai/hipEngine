"""Raw-pointer wrappers for GGUF Q5_1 selected compact pack8 GEMV decode.

Compact selected down-projection GEMV decode for raw GGUF Q5_1 experts —
the quant of the Gemma 4 26B-A4B MoE ``ffn_down_exps`` tensors. Consumes the
compact-MoE scheduler ABI (``x`` compact slab, ``expert_start_compact[E+1]``,
raw rank-3 ``qweight[E, out_features, row_bytes]``, row-major ``out``) and
mirrors ``gguf_k_selected_pack8_gemv`` structurally: one 128-thread block
per (output pack, compact row), 4-wave32 reduction, expert id recovered by
a linear scan over ``expert_start_compact``.

The inner product follows Q5_1's 32-element/24-byte blocks (``w = d*q5 + m``,
bit-exact with the legacy ``dequant_q5_1``), so the only wrapper constraint
on width is ``in_features % 32 == 0`` — including widths like 704 that are
not multiples of 256.

No new compact-MoE ABI and no resident weight sidecar/repack are introduced.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q5_1_selected_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_q5_1_selected_pack8_gemv.so"
_Q5_1_BF16 = "hipengine_gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out"
_QK_Q5_1 = 32


def plan_gguf_q5_1_selected_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q5_1_selected_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q5_1_selected_pack8_gemv(
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
        family="gguf_q5_1_selected_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact raw-Q5_1 pack8 GEMV decode."""

    _check_common(compact_rows, in_features, out_features, num_experts)
    library = library or build_gguf_q5_1_selected_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _Q5_1_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
) -> None:
    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if in_features % _QK_Q5_1 != 0:
        raise ValueError("in_features must be divisible by GGUF Q5_1 block size 32")
    if out_features % 8 != 0:
        raise ValueError("out_features must be a multiple of 8 (pack8 lane)")


def register_gguf_q5_1_selected_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register the compact selected raw-Q5_1 pack8 GEMV decode kernel."""

    fn_bf16 = gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_pack8_gemv_decode_compact_bf16_bf16_out",
        ),
        fn_bf16,
        replace=replace,
    )
    # Shorthand alias matching the docs/reference/GGUF.md pipeline language.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_pack8_gemv_decode_bf16_bf16_out",
        ),
        fn_bf16,
        replace=replace,
    )


register_gguf_q5_1_selected_pack8_gemv_kernels()


__all__ = [
    "build_gguf_q5_1_selected_pack8_gemv",
    "gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out",
    "plan_gguf_q5_1_selected_pack8_gemv_build",
    "register_gguf_q5_1_selected_pack8_gemv_kernels",
]