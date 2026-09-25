"""E6b-1: exact rows==1 (IQ4_XS gate, Q4_K up) pair + SiLU owner.

The ordered mixed-quant family (IQ4_XS gate, Q4_K up) - 7 gate/up layers
in the UD artifact - with the fused owner bit-identical to the production
unfused chain: IQ4_XS local32 single + dense Q4T16 single +
silu_mul_separate_out. Side A follows the local32 dual's split-K wave
contract; side B runs the dense Q4T16 single's full single-wave chain in
every wave and publishes wave 0, so neither side re-associates K.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq4_q4_pair.hip")
_OUTPUT_NAME = "gguf_iq4_q4_pair.so"
_SYMBOL = "hipengine_gguf_iq4_q4_pair_silu"
_SYMBOL_Q4_GATE = "hipengine_gguf_q4_iq4_pair_silu"
_SYMBOL_IQ4_Q5 = "hipengine_gguf_iq4_q5_pair_silu"
_SYMBOL_Q4_Q5 = "hipengine_gguf_q4_q5_pair_silu"
_SYMBOL_Q3_IQ4 = "hipengine_gguf_q3_iq4_pair_silu"
_SYMBOL_Q5_Q4 = "hipengine_gguf_q5_q4_pair_silu"
_SYMBOL_IQ4_Q3 = "hipengine_gguf_iq4_q3_pair_silu"
_SYMBOL_IQ3S_IQ4 = "hipengine_gguf_iq3s_iq4_pair_silu"
_SYMBOL_IQ4NL_Q5 = "hipengine_gguf_iq4nl_q5_pair_silu"
_SYMBOL_Q5_Q6 = "hipengine_gguf_q5_q6_pair_silu"
_QUANT = "gguf_iq4_xs+gguf_q4_k_t16_v1"
_VARIANT = "iq4_q4_pair_silu_bf16_bf16_out"
_QUANT_Q4_GATE = "gguf_q4_k_t16_v1+gguf_iq4_xs"
_VARIANT_Q4_GATE = "q4_iq4_pair_silu_bf16_bf16_out"
_QUANT_IQ4_Q5 = "gguf_iq4_xs+gguf_q5_k_t16_v1"
_VARIANT_IQ4_Q5 = "iq4_q5_pair_silu_bf16_bf16_out"
_QUANT_Q4_Q5 = "gguf_q4_k_t16_v1+gguf_q5_k_t16_v1"
_VARIANT_Q4_Q5 = "q4_q5_pair_silu_bf16_bf16_out"
_QUANT_Q3_IQ4 = "gguf_q3_k+gguf_iq4_xs"
_VARIANT_Q3_IQ4 = "q3_iq4_pair_silu_bf16_bf16_out"
_QUANT_Q5_Q4 = "gguf_q5_k_t16_v1+gguf_q4_k_t16_v1"
_VARIANT_Q5_Q4 = "q5_q4_pair_silu_bf16_bf16_out"
_QUANT_IQ4_Q3 = "gguf_iq4_xs+gguf_q3_k"
_VARIANT_IQ4_Q3 = "iq4_q3_pair_silu_bf16_bf16_out"

_ARGTYPES = (
    [ctypes.c_void_p] * 4
    + [ctypes.c_int64] * 4
    + [ctypes.c_void_p]
)


def plan_gguf_iq4_q4_pair_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq4_q4_pair",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq4_q4_pair(
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
        family="gguf_iq4_q4_pair",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _default_library() -> ctypes.CDLL:
    lib = build_gguf_iq4_q4_pair()
    assert lib is not None
    return lib


def gguf_iq4_q4_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wa_ptr: int,
    wb_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (IQ4_XS, Q4_K) pair + SiLU decode owner.

    Bit-exact with single/single/silu_mul: side A carries the local32
    dual's split-K wave contract (the shared ``_local32_waves`` rule), side
    B the dense Q4T16 single's single-wave chain, and both accumulators
    bf16-round exactly where the elementwise kernel would read them.
    """
    if rows != 1:
        raise ValueError("IQ4_XS/Q4_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 8:
        raise ValueError("out_features must be a positive multiple of 8")
    if not all((x_ptr, wa_ptr, wb_ptr, out_ptr)):
        raise ValueError("IQ4_XS/Q4_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL, _ARGTYPES, ctypes.c_int)
    # Split-K wave count: the shared local32 rule (see _local32_waves in
    # gguf_iq_dense), imported rather than copied so side A's cross-wave
    # order cannot drift from the single it must match bit-for-bit.
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wa_ptr),
        ctypes.c_void_p(wb_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"IQ4_XS/Q4_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_q4_iq4_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wq4_ptr: int,
    wiq4_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (Q4_K gate, IQ4_XS up) mirror pair + SiLU owner.

    E6b-2's mirror of :func:`gguf_iq4_q4_pair_silu_bf16_bf16_out`: the
    route passes gate-first (Q4_K tiles, IQ4_XS raw), and this wrapper
    reorders to the C ABI's geometry order (IQ4 first) - only the
    epilogue differs on device, taking the gate from the Q4 chain.
    Bit-exact with q4 single + iq4 single + silu_mul for the same reason.
    """
    if rows != 1:
        raise ValueError("Q4_K/IQ4_XS pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 8:
        raise ValueError("out_features must be a positive multiple of 8")
    if not all((x_ptr, wq4_ptr, wiq4_ptr, out_ptr)):
        raise ValueError("Q4_K/IQ4_XS pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_Q4_GATE, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wiq4_ptr),  # C geometry order: IQ4 side first
        ctypes.c_void_p(wq4_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"Q4_K/IQ4_XS pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_iq4_q5_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wa_ptr: int,
    wb_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (IQ4_XS gate, Q5_K up) pair + SiLU owner.

    E6b-3: side A is the IQ4_XS local32 raw weight (unchanged split-K
    chain), side B the Q5_K T16 tiles whose chain is the tile8 single's
    exact 4-group emulation. Gate-first argument order already matches
    the C geometry (IQ4 first), so no reorder is needed here.
    Bit-exact with iq4 single + q5 tile8 single + silu_mul.
    """
    if rows != 1:
        raise ValueError("IQ4_XS/Q5_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16:
        raise ValueError("out_features must be a positive multiple of 16")
    if not all((x_ptr, wa_ptr, wb_ptr, out_ptr)):
        raise ValueError("IQ4_XS/Q5_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_IQ4_Q5, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wa_ptr),
        ctypes.c_void_p(wb_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"IQ4_XS/Q5_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_q4_q5_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wa_ptr: int,
    wb_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (Q4_K gate, Q5_K up) pair + SiLU owner.

    E6b-4: neither side is IQ4. Side A is the Q4_K T16 tiles (the
    dense single's chain under A_IS_CHAIN), side B the Q5_K T16 tiles
    whose chain is the tile8 single's exact 4-group emulation.
    Gate-first argument order already matches (Q4 gate first), so no
    reorder is needed here. Bit-exact with q4 single + q5 tile8 single
    + silu_mul.
    """
    if rows != 1:
        raise ValueError("Q4_K/Q5_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16:
        raise ValueError("out_features must be a positive multiple of 16")
    if not all((x_ptr, wa_ptr, wb_ptr, out_ptr)):
        raise ValueError("Q4_K/Q5_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_Q4_Q5, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wa_ptr),
        ctypes.c_void_p(wb_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"Q4_K/Q5_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_q3_iq4_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wq3_ptr: int,
    wiq4_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (Q3_K gate, IQ4_XS up) pair + SiLU owner.

    E6b-5: the route passes gate-first (Q3_K raw, IQ4_XS raw), and
    this wrapper reorders to the C ABI's geometry order (IQ4 first in
    side A's slot) - the E6b-2 mirror pattern. Side B runs the strict
    per-row Q3_K GEMV's exact 128-thread tile emulated across the
    pair's waves (B_KIND=2), and the epilogue takes the gate from side
    B. Bit-exact with q3 strict single + iq4 local32 single + silu_mul.
    """
    if rows != 1:
        raise ValueError("Q3_K/IQ4_XS pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 8:
        raise ValueError("out_features must be a positive multiple of 8")
    if not all((x_ptr, wq3_ptr, wiq4_ptr, out_ptr)):
        raise ValueError("Q3_K/IQ4_XS pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_Q3_IQ4, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wiq4_ptr),  # C geometry order: IQ4 side first
        ctypes.c_void_p(wq3_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"Q3_K/IQ4_XS pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_q5_q4_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wq5_ptr: int,
    wq4_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (Q5_K gate, Q4_K up) pair + SiLU owner.

    E6b-6: exact role-swap of :func:`gguf_q4_q5_pair_silu_bf16_bf16_out`
    with no new arithmetic. The route passes gate-first (Q5_K tiles,
    Q4_K tiles), and this wrapper reorders to the C ABI's geometry
    order (Q4 chain first) - side A runs the dense Q4T16 chain as the
    UP, side B the tile8 chain as the GATE, and GATE_IS_Q4=true takes
    the gate from side B. Bit-exact with q5 tile8 single + q4 single +
    silu_mul for the same reason.
    """
    if rows != 1:
        raise ValueError("Q5_K/Q4_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16:
        raise ValueError("out_features must be a positive multiple of 16")
    if not all((x_ptr, wq5_ptr, wq4_ptr, out_ptr)):
        raise ValueError("Q5_K/Q4_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_Q5_Q4, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wq4_ptr),  # C geometry order: Q4 chain side first
        ctypes.c_void_p(wq5_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"Q5_K/Q4_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_iq4_q3_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wiq4_ptr: int,
    wq3_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (IQ4_XS gate, Q3_K up) pair + SiLU owner.

    E6b-7: the original E6b-5 geometry (IQ4 in side A's slot, Q3's
    strict-GEMV emulation in side B's) under the opposite epilogue
    role. The route passes gate-first (IQ4 raw, Q3 raw), which already
    matches the C geometry order - no reorder here - and
    GATE_IS_Q4=false takes the gate from side A (the IQ4 local32
    chain). Bit-exact with iq4 local32 single + q3 strict single +
    silu_mul for the same reason as the dormant E6b-5 instance.
    """
    if rows != 1:
        raise ValueError("IQ4_XS/Q3_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 8:
        raise ValueError("out_features must be a positive multiple of 8")
    if not all((x_ptr, wiq4_ptr, wq3_ptr, out_ptr)):
        raise ValueError("IQ4_XS/Q3_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_IQ4_Q3, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wiq4_ptr),  # geometry order: IQ4 side first
        ctypes.c_void_p(wq3_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"IQ4_XS/Q3_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_iq3s_iq4_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wiq3s_ptr: int,
    wiq4_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (IQ3_S gate, IQ4_XS up) pair + SiLU owner.

    E6 closeout family. The route passes gate-first (IQ3_S raw,
    IQ4_XS raw), but the C geometry keeps IQ4 in side A's slot: the
    wrapper reorders so wa is IQ4_XS (A_KIND=0 local32 split-K) and
    wb IQ3_S (B_KIND=3, the local32 decode owner's Q==2 split-K path
    verbatim); GATE_IS_Q4=true takes the gate from side B (the IQ3_S
    side). Bit-exact with iq4 local32 single + iq3_s local32 single +
    silu_mul by the same two side contracts as E6b-1..6.
    """
    if rows != 1:
        raise ValueError("IQ3_S/IQ4_XS pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 8:
        raise ValueError("out_features must be a positive multiple of 8")
    if not all((x_ptr, wiq3s_ptr, wiq4_ptr, out_ptr)):
        raise ValueError("IQ3_S/IQ4_XS pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_IQ3S_IQ4, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wiq4_ptr),  # geometry order: IQ4 side A first
        ctypes.c_void_p(wiq3s_ptr),  # IQ3_S strict emulation side B
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"IQ3_S/IQ4_XS pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_iq4nl_q5_pair_silu_bf16_bf16_out(
    x_ptr: int,
    winl_ptr: int,
    wq5_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (IQ4_NL gate, Q5_K up) pair + SiLU owner.

    E6 closeout family. The route passes gate-first (IQ4_NL raw,
    Q5_K tiles), which already matches the C geometry order - no
    reorder here: wa is IQ4_NL (A_KIND=2, the NL single's split-K
    verbatim), wb the Q5_K T16 tiles (B_KIND=1, the tile8 single's
    exact emulation); GATE_IS_Q4=false takes the gate from side A (the
    NL chain). Bit-exact with iq4_nl local32 single + q5 tile8 single
    + silu_mul.
    """
    if rows != 1:
        raise ValueError("IQ4_NL/Q5_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16:
        raise ValueError("out_features must be a positive multiple of 16")
    if not all((x_ptr, winl_ptr, wq5_ptr, out_ptr)):
        raise ValueError("IQ4_NL/Q5_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_IQ4NL_Q5, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(winl_ptr),  # geometry order already matches
        ctypes.c_void_p(wq5_ptr),
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"IQ4_NL/Q5_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def gguf_q5_q6_pair_silu_bf16_bf16_out(
    x_ptr: int,
    wq5_ptr: int,
    wq6_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows==1 (Q5_K gate, Q6_K planar up) pair + SiLU owner.

    E6 closeout family. The route passes gate-first (Q5_K tiles,
    Q6_K planar tiles), but the C geometry keeps the new side in side
    A's slot: the wrapper reorders so wa is Q6 planar (A_KIND=3, the
    planar single's exact 4-wave chain) and wb Q5_K tiles (B_KIND=1,
    the tile8 single's emulation = the GATE); GATE_IS_Q4=true takes
    the gate from side B (the tile8 chain). Bit-exact with q5 tile8
    single + q6 planar single + silu_mul.
    """
    if rows != 1:
        raise ValueError("Q5_K/Q6_K pair + SiLU decode requires rows == 1")
    if in_features <= 0 or in_features % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16:
        raise ValueError("out_features must be a positive multiple of 16")
    if not all((x_ptr, wq5_ptr, wq6_ptr, out_ptr)):
        raise ValueError("Q5_K/Q6_K pair pointers must be nonzero")
    lib = library or _default_library()
    fn = signed_kernel_fn(lib, _SYMBOL_Q5_Q6, _ARGTYPES, ctypes.c_int)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _local32_waves

    waves = _local32_waves(in_features, out_features)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(wq6_ptr),  # geometry order: Q6 planar side A
        ctypes.c_void_p(wq5_ptr),  # Q5 tile8 side B = the GATE
        ctypes.c_void_p(out_ptr),
        rows,
        in_features,
        out_features,
        waves,
        ctypes.c_void_p(stream),
    )
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f"Q5_K/Q6_K pair + SiLU decode failed: {rt.error_string(err)}"
        )


def register_gguf_iq4_q4_pair_kernels(*, replace: bool = False) -> None:
    register(
        KernelKey("hip_gfx1100", "linear_pair_silu", _QUANT, _VARIANT),
        gguf_iq4_q4_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear_pair_silu", _QUANT_Q4_GATE, _VARIANT_Q4_GATE
        ),
        gguf_q4_iq4_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear_pair_silu", _QUANT_IQ4_Q5, _VARIANT_IQ4_Q5
        ),
        gguf_iq4_q5_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear_pair_silu", _QUANT_Q4_Q5, _VARIANT_Q4_Q5
        ),
        gguf_q4_q5_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear_pair_silu", _QUANT_Q3_IQ4, _VARIANT_Q3_IQ4
        ),
        gguf_q3_iq4_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear_pair_silu", _QUANT_Q5_Q4, _VARIANT_Q5_Q4
        ),
        gguf_q5_q4_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "linear_pair_silu", _QUANT_IQ4_Q3, _VARIANT_IQ4_Q3
        ),
        gguf_iq4_q3_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq3_s+gguf_iq4_xs",
            "iq3s_iq4_pair_silu_bf16_bf16_out",
        ),
        gguf_iq3s_iq4_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq4_nl+gguf_q5_k_t16_v1",
            "iq4nl_q5_pair_silu_bf16_bf16_out",
        ),
        gguf_iq4nl_q5_pair_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q5_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1",
            "q5_q6_pair_silu_bf16_bf16_out",
        ),
        gguf_q5_q6_pair_silu_bf16_bf16_out,
        replace=replace,
    )


register_gguf_iq4_q4_pair_kernels()