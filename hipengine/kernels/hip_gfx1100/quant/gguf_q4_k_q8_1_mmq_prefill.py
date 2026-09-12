"""Wrappers for the dense bulk-prefill raw-Q4_K x DS4-Q8_1 integer MMQ leaf screen.

Leaf screen for the 2026-09-08 engine-comparison PP8192 attribution follow-up
(worklog pp8192-gap-attribution-8bb338): the candidate arithmetic class is
nasone32-style raw-Q4_K x Q8_1-activation integer MMQ with efa4e8641-family
load reuse, versus the retained float Q4T16 WMMA projection prefill owners.
Two consumer classes ship here, each with ctl/vdr siblings that share thread
mapping, integer dot terms and per-subblock f32 evaluation order (bit-exact
RED contract in tests/test_gpu_gguf_q4_k_q8_1_mmq_prefill.py):

  * mmq32 — local128 staged-dp4a consumer, 32x32 output tiles (dense port
    of the retained raw Q5 MMQ32 C8-owner idiom).
  * wmma32 — direct-global iu8 WMMA consumer, 32x16 output tiles (dense
    port of the 2026-06-16 selected ds4-wmma32 family winner).

Activations arrive as llama.cpp-style DS4 block_q8_1_mmq from the existing
gguf_q8_1_mmq_ds4_pack_bf16. This module registers leaf-screen variants —
the production dense Q4 projection prefill owners are untouched until any
promotion passes the full execution-profile gate.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_q8_1_mmq_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_q8_1_mmq_prefill.so"
_MMQ32_CTL_F32 = "hipengine_gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out"
_MMQ32_VDR_F32 = "hipengine_gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out"
_WMMA32_CTL_F32 = "hipengine_gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out"
_WMMA32_VDR_F32 = "hipengine_gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out"
_MMQ32_CTL_BF16 = "hipengine_gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out"
_MMQ32_VDR_BF16 = "hipengine_gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out"
_WMMA32_CTL_BF16 = "hipengine_gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out"
_WMMA32_VDR_BF16 = "hipengine_gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out"

_MAX_ROWS = 1 << 20


def plan_gguf_q4_k_q8_1_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_q8_1_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_q8_1_mmq_prefill(
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
        family="gguf_q4_k_q8_1_mmq_prefill",
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
        _LIBRARY = build_gguf_q4_k_q8_1_mmq_prefill(load=True)
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
        raise RuntimeError(f"q4 k q8_1 mmq prefill screen launch failed: {err}")


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0 or rows > _MAX_ROWS:
        raise ValueError(
            f"dense q8_1 mmq screen rows must be in [1, {_MAX_ROWS}]"
        )
    if in_features <= 0 or in_features % 256 != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 32 != 0:
        raise ValueError("out_features must be a positive multiple of 32")


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


gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out = _make_launch(_MMQ32_CTL_F32)
gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out = _make_launch(_MMQ32_VDR_F32)
gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out = _make_launch(_WMMA32_CTL_F32)
gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out = _make_launch(_WMMA32_VDR_F32)
gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out = _make_launch(_MMQ32_CTL_BF16)
gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out = _make_launch(_MMQ32_VDR_BF16)
gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out = _make_launch(_WMMA32_CTL_BF16)
gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out = _make_launch(_WMMA32_VDR_BF16)


_VARIANTS = (
    "mmq32_ctl_dense_bf16_f32_out",
    "mmq32_vdr_dense_bf16_f32_out",
    "wmma32_ctl_dense_bf16_f32_out",
    "wmma32_vdr_dense_bf16_f32_out",
    "mmq32_ctl_dense_bf16_bf16_out",
    "mmq32_vdr_dense_bf16_bf16_out",
    "wmma32_ctl_dense_bf16_bf16_out",
    "wmma32_vdr_dense_bf16_bf16_out",
)
_LAUNCHERS = {
    "mmq32_ctl_dense_bf16_f32_out": gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out,
    "mmq32_vdr_dense_bf16_f32_out": gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out,
    "wmma32_ctl_dense_bf16_f32_out": gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out,
    "wmma32_vdr_dense_bf16_f32_out": gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out,
    "mmq32_ctl_dense_bf16_bf16_out": gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out,
    "mmq32_vdr_dense_bf16_bf16_out": gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out,
    "wmma32_ctl_dense_bf16_bf16_out": gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out,
    "wmma32_vdr_dense_bf16_bf16_out": gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out,
}


def register_gguf_q4_k_q8_1_mmq_prefill_kernels(
    *, replace: bool = True
) -> None:
    """Register the leaf-screen MMQ variants on the linear axis."""

    for variant in _VARIANTS:
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", variant),
            _LAUNCHERS[variant],
            replace=replace,
        )


# Module-import registration, mirroring the VDR leaf screen.
register_gguf_q4_k_q8_1_mmq_prefill_kernels()


__all__ = [
    "build_gguf_q4_k_q8_1_mmq_prefill",
    "plan_gguf_q4_k_q8_1_mmq_prefill_build",
    "gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out",
    "gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out",
    "gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out",
    "gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out",
    "gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out",
    "gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out",
    "gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out",
    "gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out",
    "register_gguf_q4_k_q8_1_mmq_prefill_kernels",
]
