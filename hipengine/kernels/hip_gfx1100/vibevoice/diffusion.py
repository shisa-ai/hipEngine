"""VibeVoice-TTS diffusion head kernel build + launch wrappers (diffusion.hip).

Elementwise companions to ``dense_gemv`` for the MLP-DiT head. Each wrapper
reproduces one eager rounding chain; see the .hip header for the op list.
All launch helpers take raw device pointers and register under the shared
``vibevoice`` family (``vv_diff_*`` keys, bf16 quant, strict variant).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("diffusion.hip")
_OUTPUT_NAME = "vibevoice_tts_diffusion"

_P = ctypes.c_void_p
_I = ctypes.c_int64
_S = ctypes.c_void_p
_F = ctypes.c_float

_ARGTYPES_POINTWISE = (_P, _P, _I, _S)
_ARGTYPES_BINARY = (_P, _P, _P, _I, _S)
_ARGTYPES_RMSNORM = (_P, _P, _P, _I, _I, _F, _S)
_ARGTYPES_RMSNORM_MODULATE = (_P, _P, _P, _P, _I, _I, _I, _F, _S)
_ARGTYPES_MODULATE = (_P, _P, _P, _I, _I, _I, _S)
_ARGTYPES_GATED = (_P, _P, _P, _P, _I, _I, _I, _S)
_ARGTYPES_CFG = (_P, _P, _F, _P, _I, _S)


def plan_vibevoice_diffusion_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "baseline",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="vibevoice_tts_diffusion",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_vibevoice_diffusion(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "baseline",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="vibevoice_tts_diffusion",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _library() -> ctypes.CDLL:
    library = build_vibevoice_diffusion()
    if library is None:
        raise RuntimeError("vibevoice_tts_diffusion build returned no library")
    return library


def _launch(
    symbol: str,
    argtypes: tuple,
    args: tuple,
    *,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*args)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_diff_rmsnorm_bf16(
    x_ptr: int, w_ptr: int | None, out_ptr: int, rows: int, hidden: int, eps: float,
    *, library=None, runtime=None,
) -> None:
    """Two-rounding RMSNorm (w_ptr=None: no-affine, single rounding)."""
    _launch(
        "hipengine_vv_diff_rmsnorm_bf16",
        _ARGTYPES_RMSNORM,
        (_P(x_ptr), _P(w_ptr if w_ptr is not None else 0), _P(out_ptr),
         _I(rows), _I(hidden), _F(eps), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_rmsnorm_modulate_bf16(
    x_ptr: int, w_ptr: int | None, chunk_ptr: int, out_ptr: int, rows: int,
    hidden: int, src_stride: int, eps: float, *, library=None, runtime=None,
) -> None:
    """rmsnorm + adaLN modulate in one launch, bit-identical to the pair."""
    _launch(
        "hipengine_vv_diff_rmsnorm_modulate_bf16",
        _ARGTYPES_RMSNORM_MODULATE,
        (_P(x_ptr), _P(w_ptr if w_ptr is not None else 0), _P(chunk_ptr),
         _P(out_ptr), _I(rows), _I(hidden), _I(src_stride), _F(eps), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_silu_bf16(
    g_ptr: int, out_ptr: int, n: int, *, library=None, runtime=None
) -> None:
    _launch(
        "hipengine_vv_diff_silu_bf16",
        _ARGTYPES_POINTWISE,
        (_P(g_ptr), _P(out_ptr), _I(n), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_add_bf16(
    a_ptr: int, b_ptr: int, out_ptr: int, n: int, *, library=None, runtime=None
) -> None:
    _launch(
        "hipengine_vv_diff_add_bf16",
        _ARGTYPES_BINARY,
        (_P(a_ptr), _P(b_ptr), _P(out_ptr), _I(n), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_mul_bf16(
    a_ptr: int, b_ptr: int, out_ptr: int, n: int, *, library=None, runtime=None
) -> None:
    _launch(
        "hipengine_vv_diff_mul_bf16",
        _ARGTYPES_BINARY,
        (_P(a_ptr), _P(b_ptr), _P(out_ptr), _I(n), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_modulate_bf16(
    x_ptr: int, chunk_ptr: int, out_ptr: int, rows: int, width: int,
    src_stride: int, *, library=None, runtime=None,
) -> None:
    """r(r(x * r(1 + scale)) + shift) with row-major adaLN chunk slices."""
    _launch(
        "hipengine_vv_diff_modulate_bf16",
        _ARGTYPES_MODULATE,
        (_P(x_ptr), _P(chunk_ptr), _P(out_ptr), _I(rows), _I(width), _I(src_stride), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_gated_residual_bf16(
    h_ptr: int, gate_chunk_ptr: int, y_ptr: int, out_ptr: int, rows: int,
    width: int, gate_stride: int, *, library=None, runtime=None,
) -> None:
    _launch(
        "hipengine_vv_diff_gated_residual_bf16",
        _ARGTYPES_GATED,
        (_P(h_ptr), _P(gate_chunk_ptr), _P(y_ptr), _P(out_ptr),
         _I(rows), _I(width), _I(gate_stride), _S(0)),
        library=library,
        runtime=runtime,
    )


def vv_diff_cfg_combine_bf16(
    c_ptr: int, u_ptr: int, cfg: float, out_ptr: int, n: int,
    *, library=None, runtime=None,
) -> None:
    _launch(
        "hipengine_vv_diff_cfg_combine_bf16",
        _ARGTYPES_CFG,
        (_P(c_ptr), _P(u_ptr), _F(cfg), _P(out_ptr), _I(n), _S(0)),
        library=library,
        runtime=runtime,
    )
