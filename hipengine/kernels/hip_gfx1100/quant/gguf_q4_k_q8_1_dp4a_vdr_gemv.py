"""Wrappers for the dense single-row q8_1-DP4A Q4_K load-reuse A/B screen.

Leaf screen for the nasone32/llama.cpp k-quant load-reuse idea
(efa4e86410c07723deaa458bdadd8c08f1029928): the VDR kernel amortizes the
Q4_K block-header, sub-scales/mins and q8_1 block-scale loads across a
whole 32-element subblock and all 8 output columns, while the control
kernel re-decodes them per 4-k pack. Both share the same thread mapping
and f32 evaluation order, so their outputs are bit-identical; only the
load schedule differs. This module registers leaf-screen variants — the
production dense Q4_K decode owner (float pack8 GEMV) is untouched and
remains the registered exact path until any promotion passes the full
execution-profile gate.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_q8_1_dp4a_vdr_gemv.hip")
_OUTPUT_NAME = "gguf_q4_k_q8_1_dp4a_vdr_gemv.so"
_CTL_F32 = "hipengine_gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out"
_VDR_F32 = "hipengine_gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out"
_CTL_BF16 = "hipengine_gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out"
_VDR_BF16 = "hipengine_gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out"


def plan_gguf_q4_k_q8_1_dp4a_vdr_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_q8_1_dp4a_vdr_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_q8_1_dp4a_vdr_gemv(
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
        family="gguf_q4_k_q8_1_dp4a_vdr_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


_LIBRARY: ctypes.CDLL | None = None


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = build_gguf_q4_k_q8_1_dp4a_vdr_gemv(load=True)
    return _LIBRARY


def _launch(
    symbol: str,
    xq_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    del runtime  # default stream/device context is used by the launcher
    resolved = library or _library()
    fn = getattr(resolved, symbol)
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
    err = fn(
        ctypes.c_void_p(int(xq_ptr)),
        ctypes.c_void_p(int(qweight_ptr)),
        ctypes.c_void_p(int(out_ptr)),
        ctypes.c_int64(int(rows)),
        ctypes.c_int64(int(in_features)),
        ctypes.c_int64(int(out_features)),
        ctypes.c_void_p(int(stream)),
    )
    if err != 0:
        raise RuntimeError(f"q4 k q8_1 dp4a vdr screen launch failed: {err}")


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0 or rows > 64:
        raise ValueError("dense q8_1 dp4a screen rows must be in [1, 64]")
    if in_features <= 0 or in_features % 256 != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 8 != 0:
        raise ValueError("out_features must be a positive multiple of 8")


def _make_launch(symbol: str):
    def launch(
        xq_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _validate_shape(rows, in_features, out_features)
        _launch(
            symbol,
            xq_ptr,
            qweight_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    return launch


gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out = _make_launch(_CTL_F32)
gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out = _make_launch(_VDR_F32)
gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out = _make_launch(_CTL_BF16)
gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out = _make_launch(_VDR_BF16)


_CTL_VARIANT_F32 = "q8_1_dp4a_ctl_bf16_f32_out"
_VDR_VARIANT_F32 = "q8_1_dp4a_vdr_bf16_f32_out"
_CTL_VARIANT_BF16 = "q8_1_dp4a_ctl_bf16_bf16_out"
_VDR_VARIANT_BF16 = "q8_1_dp4a_vdr_bf16_bf16_out"


def register_gguf_q4_k_q8_1_dp4a_vdr_gemv_kernels(
    *, replace: bool = True
) -> None:
    """Register the leaf-screen load-reuse variants on the linear axis."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", _CTL_VARIANT_F32),
        gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", _VDR_VARIANT_F32),
        gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", _CTL_VARIANT_BF16),
        gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", _VDR_VARIANT_BF16),
        gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out,
        replace=replace,
    )


# Module-import registration, mirroring the qmicro grouped leaf screen.
register_gguf_q4_k_q8_1_dp4a_vdr_gemv_kernels()


__all__ = [
    "build_gguf_q4_k_q8_1_dp4a_vdr_gemv",
    "plan_gguf_q4_k_q8_1_dp4a_vdr_gemv_build",
    "gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out",
    "gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out",
    "gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out",
    "gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out",
    "register_gguf_q4_k_q8_1_dp4a_vdr_gemv_kernels",
]
