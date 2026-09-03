"""Raw Q8_0 x D4-Q8_1 MMQ128 prefill wrappers and Q3 policy."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_mmq_prefill.hip")
_OUTPUT_NAME = "gguf_q8_0_mmq_prefill.so"
_QUANT_SYMBOL = "hipengine_gguf_q8_0_mmq128_quantize_bf16_d4"
_QUANT_X2_SYMBOL = "hipengine_gguf_q8_0_mmq128_quantize_bf16_d4x2"
_QUANT_X3_SYMBOL = "hipengine_gguf_q8_0_mmq128_quantize_bf16_d4x3"
_PREFILL_SYMBOL = "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out"
_PREFILL_X2_SYMBOL = "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out"
_PREFILL_X3_SYMBOL = "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out"
_PREFILL_X3_GUARDED_SYMBOL = (
    "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out"
)
_PREFILL_X3_F32_SYMBOL = "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_f32_out"
_SPARSE_EXACT_SYMBOL = "hipengine_gguf_q8_0_mmq128_sparse_exact_correct_bf16"
_PREFILL_VARIANT = "mmq128_prefill_q8_1_d4_bf16_bf16_out"
_PREFILL_X2_VARIANT = "mmq128_prefill_q8_1_d4x2_bf16_bf16_out"
_PREFILL_X3_VARIANT = "mmq128_prefill_q8_1_d4x3_bf16_bf16_out"
_PREFILL_X3_GUARDED_VARIANT = "mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out"
_PREFILL_X3_F32_VARIANT = "mmq128_prefill_q8_1_d4x3_bf16_f32_out"
_POLICY_VARIANT = "raw_q8_mmq128"
_QUANT_F32_X3_SYMBOL = "hipengine_gguf_q8_0_mmq128_quantize_f32_d4x3"
_PREFILL_X3_GUARDED_F32_SYMBOL = (
    "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
)
_QUANTIZE_F32_D4X2_SYMBOL = (
    "hipengine_gguf_q8_0_mmq128_quantize_f32_d4x2"
)
_PREFILL_D4X2_GUARDED_F32_SYMBOL = (
    "hipengine_gguf_q8_0_mmq128_prefill_q8_1_d4x2_guarded_f32_f32_out"
)
_PREFILL_X3_GUARDED_F32_VARIANT = (
    "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
)
_SPARSE_EXACT_F32_SYMBOL = "hipengine_gguf_q8_0_mmq128_sparse_exact_correct_f32"

Q8_MMQ_PREFILL_POLICY_KEY = KernelKey(
    "hip_gfx1100",
    "linear_prefill_policy",
    "gguf_ud_q3_k_m",
    _POLICY_VARIANT,
)

@dataclass(frozen=True)
class Q8MMQPrefillPolicy:
    """Model-plugin admission, correction, and scratch contract for Q8 MMQ."""

    min_rows: Mapping[tuple[int, int], int]
    max_rows: int
    risk_threshold: float
    max_out_features: int

    def __call__(self, rows: int, hidden: int, out_features: int) -> bool:
        threshold = self.min_rows.get((int(hidden), int(out_features)))
        return (
            threshold is not None
            and int(rows) >= threshold
            and int(rows) <= self.max_rows
        )

    def risk_capacity(self, rows: int) -> int:
        if rows <= 0 or rows > self.max_rows:
            raise ValueError(f"rows must be in [1, {self.max_rows}]")
        return int(rows) * self.max_out_features

    def risk_indices_nbytes(self, rows: int) -> int:
        return self.risk_capacity(rows) * ctypes.sizeof(ctypes.c_int32)

# GPU1 RX 7900 XTX crossover gates against the retained exact raw-Q8 family.
# The narrow/short shapes lose once exact BF16 boundary repair is included, so
# they intentionally retain the exact tiled fallback.
_UD_Q3_K_M_MIN_ROWS: dict[tuple[int, int], int] = {
    (2048, 8192): 32,
    (2048, 4096): 48,
}

UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY = Q8MMQPrefillPolicy(
    min_rows=_UD_Q3_K_M_MIN_ROWS,
    max_rows=4096,
    risk_threshold=1.0e-5,
    max_out_features=8192,
)

# Qwen4Exp (UD-Q4_K_XL) dense-Q8_0 projections on gfx1151. Gates measured in
# 2026-08-29 targeting: the float-coltile owner loses 6.4-7.3x to MMQ128 at
# these shapes; constraint-failing shapes (K%256!=0: hc down 320, shexp gate)
# never enter the dispatch check and stay exact.
_QWEN4EXP_MIN_ROWS: dict[tuple[int, int], int] = {
    (2560, 10240): 64,  # GDN attn_qkv + PLE key
    (2560, 12288): 64,  # QSA attn_q
    (6144, 2560): 64,   # GDN ssm_out
    (10240, 320): 64,   # GR hc_*_up
    (2560, 2560): 64,   # PLE value
    (2560, 640): 64,    # shared-expert gate/up
    (2560, 512): 64,    # QSA attn_v
}

QWEN4EXP_Q8_MMQ_PREFILL_POLICY = Q8MMQPrefillPolicy(
    min_rows=_QWEN4EXP_MIN_ROWS,
    max_rows=2048,
    # The guard criterion is "near a BF16 rounding boundary", which only
    # protects BF16 outputs. This path emits F32: a 1e-5 threshold queues a
    # large fraction of all floats and degenerates the repair pass into a
    # near-full exact recompute (measured 0.634 s at pp508). Threshold zero
    # queues only exact-boundary values, making the repair effectively free;
    # the arithmetic change is covered by the production profile gate.
    risk_threshold=0.0,
    max_out_features=12288,
)

def plan_gguf_q8_0_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )

def build_gguf_q8_0_mmq_prefill(
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
        family="gguf_q8_0_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )

def q8_mmq_d4_nbytes(rows: int, hidden: int) -> int:
    """Return the source-compatible D4 activation workspace size."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 128 != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    return (hidden // 128) * rows * 144

def q8_mmq_d4x2_nbytes(rows: int, hidden: int) -> int:
    """Return the primary-plus-residual D4 activation workspace size."""

    return 2 * q8_mmq_d4_nbytes(rows, hidden)

def q8_mmq_d4x3_nbytes(rows: int, hidden: int) -> int:
    """Return the three-plane residual D4 activation workspace size."""

    return 3 * q8_mmq_d4_nbytes(rows, hidden)

def ud_q3_k_m_q8_mmq_prefill_policy(rows: int, hidden: int, out_features: int) -> bool:
    """Compatibility wrapper for the registered Q3 policy object."""

    return UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY(rows, hidden, out_features)

def gguf_q8_0_mmq128_quantize_bf16_d4(
    x_ptr: int,
    out_d4_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 rows into source-compatible K-major D4 MMQ blocks."""

    q8_mmq_d4_nbytes(rows, hidden)
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
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
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_d4_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_quantize_bf16_d4x2(
    x_ptr: int,
    out_d4_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 rows as primary D4 plus a second D4 residual plane."""

    q8_mmq_d4x2_nbytes(rows, hidden)
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_X2_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_d4_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_quantize_bf16_d4x3(
    x_ptr: int,
    out_d4_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 rows as primary D4 plus two residual D4 planes."""

    q8_mmq_d4x3_nbytes(rows, hidden)
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_X3_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_d4_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out(
    x_d4_ptr: int,
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
    """Launch the output-major 128x128 K256 MMQ body."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_SYMBOL)
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
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out(
    x_d4_ptr: int,
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
    """Launch MMQ with primary and residual D4 passes sharing staged weights."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_X2_SYMBOL)
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
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out(
    x_d4_ptr: int,
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
    """Launch MMQ with three residual D4 passes sharing staged weights."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_X3_SYMBOL)
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
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out(
    x_d4_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    risk_threshold: float,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch D4x3 MMQ and enqueue outputs near BF16 rounding boundaries."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    if max_risks <= 0:
        raise ValueError("max_risks must be positive")
    if risk_threshold < 0:
        raise ValueError("risk_threshold must be non-negative")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_X3_GUARDED_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_float(risk_threshold),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_sparse_exact_correct_bf16(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Recompute queued output elements with the exact 128-thread reduction."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 32 != 0:
        raise ValueError("hidden must be a positive multiple of 32")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if max_risks <= 0:
        raise ValueError("max_risks must be positive")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SPARSE_EXACT_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_quantize_f32_d4x3(
    x_ptr: int,
    out_d4_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack F32 rows into source-compatible K-major D4 MMQ blocks."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 128 != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANT_F32_X3_SYMBOL)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * 2 + [
        ctypes.c_void_p
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_d4_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_quantize_f32_d4x2(
    x_ptr: int,
    out_d4_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack F32 rows into two-plane residual D4 MMQ blocks (PF-1d candidate)."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 128 != 0:
        raise ValueError("hidden must be a positive multiple of 128")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _QUANTIZE_F32_D4X2_SYMBOL)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * 2 + [
        ctypes.c_void_p
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_d4_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out(
    x_d4_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    risk_threshold: float,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch D4x3 MMQ (F32 output) and enqueue near-boundary outputs."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    if max_risks <= 0:
        raise ValueError("max_risks must be positive")
    if risk_threshold < 0:
        raise ValueError("risk_threshold must be non-negative")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_X3_GUARDED_F32_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_float(risk_threshold),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4x2_guarded_f32_f32_out(
    x_d4_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    risk_threshold: float,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch two-plane D4x2 MMQ (F32 output, guarded) - PF-1d candidate."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    if max_risks <= 0:
        raise ValueError("max_risks must be positive")
    if risk_threshold < 0:
        raise ValueError("risk_threshold must be non-negative")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_D4X2_GUARDED_F32_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_float(risk_threshold),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_sparse_exact_correct_f32(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Recompute queued F32 output elements with the exact reduction."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 32 != 0:
        raise ValueError("hidden must be a positive multiple of 32")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if max_risks <= 0:
        raise ValueError("max_risks must be positive")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SPARSE_EXACT_F32_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_f32_out(
    x_d4_ptr: int,
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
    """Launch the three-pass MMQ diagnostic with FP32 output."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or build_gguf_q8_0_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _PREFILL_X3_F32_SYMBOL)
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
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))

def register_gguf_q8_0_mmq_prefill_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "activation_quant", "q8_1_d4x3", "bf16"),
        gguf_q8_0_mmq128_quantize_bf16_d4x3,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", _PREFILL_X3_GUARDED_VARIANT),
        gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", _PREFILL_X3_GUARDED_F32_VARIANT),
        gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "activation_quant",
            "q8_1_d4x2",
            "f32",
        ),
        gguf_q8_0_mmq128_quantize_f32_d4x2,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q8_0",
            "mmq128_prefill_q8_1_d4x2_guarded_f32_f32_out",
        ),
        gguf_q8_0_mmq128_prefill_q8_1_d4x2_guarded_f32_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_prefill_policy",
            "gguf_ud_q4_k_xl",
            _POLICY_VARIANT,
        ),
        QWEN4EXP_Q8_MMQ_PREFILL_POLICY,
        replace=replace,
    )
    register(
        Q8_MMQ_PREFILL_POLICY_KEY,
        UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY,
        replace=replace,
    )

register_gguf_q8_0_mmq_prefill_kernels()

__all__ = [
    "Q8MMQPrefillPolicy",
    "Q8_MMQ_PREFILL_POLICY_KEY",
    "QWEN4EXP_Q8_MMQ_PREFILL_POLICY",
    "UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY",
    "build_gguf_q8_0_mmq_prefill",
    "gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out",
    "gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out",
    "gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out",
    "gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out",
    "gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
    "gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_f32_out",
    "gguf_q8_0_mmq128_sparse_exact_correct_bf16",
    "gguf_q8_0_mmq128_sparse_exact_correct_f32",
    "gguf_q8_0_mmq128_quantize_bf16_d4",
    "gguf_q8_0_mmq128_quantize_bf16_d4x2",
    "gguf_q8_0_mmq128_quantize_bf16_d4x3",
    "gguf_q8_0_mmq128_quantize_f32_d4x3",
    "plan_gguf_q8_0_mmq_prefill_build",
    "q8_mmq_d4_nbytes",
    "q8_mmq_d4x2_nbytes",
    "q8_mmq_d4x3_nbytes",
    "register_gguf_q8_0_mmq_prefill_kernels",
    "ud_q3_k_m_q8_mmq_prefill_policy",
]
