"""GPU byte-lossless raw GGUF Q8_0 to Q8T16 pair repacking (SH6-P1)."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_raw_to_t16.hip")
_OUTPUT_NAME = "gguf_q8_0_raw_to_t16.so"
_SYMBOL = "hipengine_gguf_q8_0_raw_pair_to_t16"
_Q8_0_QK = 32
_Q8_0_BLOCK_BYTES = 34
_T16_COLS = 16
_ALLOWED_THREADS = frozenset({64, 128, 256})


def plan_gguf_q8_0_raw_to_t16_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_raw_to_t16",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_raw_to_t16(
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
        family="gguf_q8_0_raw_to_t16",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _validate_shape(in_features: int, out_features_a: int, out_features_b: int) -> None:
    if in_features <= 0 or in_features % _Q8_0_QK:
        raise ValueError("in_features must be divisible by 32")
    if (
        out_features_a <= 0
        or out_features_b <= 0
        or out_features_a % _T16_COLS
        or out_features_b % _T16_COLS
    ):
        raise ValueError("out_features_a and out_features_b must be positive and divisible by 16")


def gguf_q8_0_raw_pair_to_t16_nbytes(
    in_features: int,
    out_features_a: int,
    out_features_b: int,
) -> int:
    """Return the byte-neutral pair scratch requirement for the transform."""

    _validate_shape(in_features, out_features_a, out_features_b)
    blocks_per_row = in_features // _Q8_0_QK
    return (out_features_a + out_features_b) * blocks_per_row * _Q8_0_BLOCK_BYTES


def gguf_q8_0_raw_pair_to_t16(
    raw_a_ptr: int,
    raw_b_ptr: int,
    t16_a_ptr: int,
    t16_b_ptr: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Repack one raw-Q8_0 pair into host-packer-identical Q8T16 bytes."""

    _validate_shape(in_features, out_features_a, out_features_b)
    if threads not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {sorted(_ALLOWED_THREADS)}")
    library = library or build_gguf_q8_0_raw_to_t16(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
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
        ctypes.c_void_p(raw_a_ptr),
        ctypes.c_void_p(raw_b_ptr),
        ctypes.c_void_p(t16_a_ptr),
        ctypes.c_void_p(t16_b_ptr),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_q8_0_raw_to_t16_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "layout_transform",
            "gguf_q8_0",
            "raw_pair_to_t16",
        ),
        gguf_q8_0_raw_pair_to_t16,
        replace=replace,
    )


register_gguf_q8_0_raw_to_t16_kernels()


__all__ = [
    "build_gguf_q8_0_raw_to_t16",
    "gguf_q8_0_raw_pair_to_t16",
    "gguf_q8_0_raw_pair_to_t16_nbytes",
    "plan_gguf_q8_0_raw_to_t16_build",
    "register_gguf_q8_0_raw_to_t16_kernels",
]
