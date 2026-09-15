"""VibeVoice-TTS decoder kernel build + launch wrappers (decoder.hip).

All launch helpers take raw device pointers and the four-axis registry
family ``vibevoice``. Storage is bf16; accumulation is fp32. The streaming
transposed-conv kernel takes an explicit prefix buffer holding the previous
frames' tail rows (fixed length K-1, left-zero-padded) so per-frame carries
equal a single-pass forward exactly.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("decoder.hip")
_OUTPUT_NAME = "vibevoice_tts_decoder"

_P = ctypes.c_void_p
_I = ctypes.c_int64
_S = ctypes.c_void_p

_ARGTYPES_CONVTR = (_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _S)


def plan_vibevoice_decoder_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "baseline",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="vibevoice_tts_decoder",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_vibevoice_decoder(
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
        family="vibevoice_tts_decoder",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _library() -> ctypes.CDLL:
    library = build_vibevoice_decoder()
    if library is None:
        raise RuntimeError("vibevoice_tts_decoder build returned no library")
    return library


def vv_convtr_gemm_bf16(
    prefix_ptr: int,
    x_ptr: int,
    w_t_ptr: int,
    b_ptr: int,
    out_ptr: int,
    prefix_rows: int,
    rows: int,
    rows_out: int,
    c_in: int,
    c_out: int,
    k_len: int,
    stride: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Causal streaming transposed conv: rows_out = rows * stride."""
    if prefix_rows != k_len - 1:
        raise ValueError("prefix_rows must equal k_len - 1 for the streaming convtr")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_convtr_gemm_bf16", _ARGTYPES_CONVTR, ctypes.c_int)
    err = fn(
        prefix_ptr, x_ptr, w_t_ptr, b_ptr, out_ptr,
        prefix_rows, rows, rows_out, c_in, c_out, k_len, stride, stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
