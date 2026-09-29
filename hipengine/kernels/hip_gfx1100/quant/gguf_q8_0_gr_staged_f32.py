"""Raw-pointer wrapper for the LDS-staged f32 GR-up GEMM prototype.

Screen-only. See the ``.hip`` header for the design and for why the mechanism
is worth testing: the GR up projection re-reads a 3.48 MB Q8_0 weight once per
4-row batch (3.565 GB per launch) and is weight-traffic-bound, while the
promoted ``dense_wide256`` tile stages the weight once per K-tile in LDS and is
4.6x faster. That tile is 4.6x faster *and* fails the production gate because
its matrix-core multiply rounds both operands to f16. This prototype keeps the
staging and tiling and does the multiply in f32 FMA on f32 operands, so the only
arithmetic difference from the exact chain is the summation order.

Registered as ``coltile2_branch4_rowbatch4_staged_f32`` under
``linear+gr_gated_mean`` so the four-axis registry knows the variant and
``is_registered`` can gate it, but **default-off and not selected unless
``HIPENGINE_QWEN4_EXP_GR_STAGED`` is set** - the route changes arithmetic
(summation order) relative to the fused exact parent and has not crossed the GR
prefill-boundary KL screen.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("gguf_q8_0_gr_staged_f32.hip")
_OUTPUT_NAME = "gguf_q8_0_gr_staged_f32.so"
_FAMILY = "gguf_q8_0_gr_staged_f32"

SYMBOL = "hipengine_gr_staged_f32_up"


def plan_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build(
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
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gr_staged_f32_up(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the staged-f32 GR-up prototype over Q8_0 weight bytes.

    ``threads`` is accepted and ignored so this matches the ``raw`` launch ABI
    the runtime's GGUF linear dispatch uses (``_launch_raw`` forwards its kwargs
    verbatim). The prototype's block size is fixed at 256 by its BM=128/BN=64
    tiling; honouring a caller-supplied thread count would silently change the
    tiling. Passing a value that is not None and not 256 raises rather than
    ignoring it, because a caller that believes it is selecting a geometry
    should find out that it is not.
    """

    if threads is not None and threads != 256:
        raise ValueError(
            "the staged-f32 tile has a fixed 256-thread block; "
            f"got threads={threads}"
        )
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0 or in_features % 32 != 0:
        raise ValueError("in_features must be a positive multiple of 32")
    if out_features <= 0:
        raise ValueError("out_features must be positive")

    library = library or build(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, SYMBOL)
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


GR_STAGED_F32_VARIANT = "coltile2_branch4_rowbatch4_staged_f32"


def register_gr_staged_f32_kernel(*, replace: bool = True) -> None:
    """Register the staged-f32 GR-up variant under ``linear+gr_gated_mean``.

    Registration is separate from selection: the runtime chooses this variant
    only behind its own default-off env gate. Registering it here (rather than
    in ``gguf_k_gemv``) keeps the dependency pointing from the prototype module
    to the registry instead of the reverse.
    """

    from hipengine.kernels.registry import KernelKey, register

    register(
        KernelKey(
            "hip_gfx1100",
            "linear+gr_gated_mean",
            "gguf_q8_0",
            GR_STAGED_F32_VARIANT,
        ),
        gr_staged_f32_up,
        replace=replace,
    )


register_gr_staged_f32_kernel()


__all__ = [
    "GR_STAGED_F32_VARIANT",
    "SYMBOL",
    "build",
    "gr_staged_f32_up",
    "plan_build",
    "register_gr_staged_f32_kernel",
]
