"""Q5_K iu8-WMMA selected dual gate/up prefill wrappers.

#15 Q5_K bundle KL-shaving variant: the production-enveloped iu8-WMMA
risk+repair machinery (see ``gguf_q4_k_selected_prefill.py``) adapted to
Q5_K weights. Activations use the same 3-plane residual int8 quantization
with the Kahan-bounded risk criterion; Q5_K weight bytes are the raw q
values [0, 31] so the weight path has no requantization error. The sparse
exact repair reproduces the strict row4 parent
(``gguf_q5_k_selected_grouped_row4_bf16_kernel``: 128 threads, strided k
ownership, shuffle tree + serial wave sum) bit-exactly.

Consumes the compact scheduler ABI emitted by ``qwen35_moe_group_*`` and
``qwen35_moe_wmma_tile_map``: ``x[compact_rows, in_features]`` (compacted
activations), ``expert_start_compact[E+1]``,
``expert_start_wmma[E+1]``, ``tile_expert[wmma_total_rows/16]``, raw
rank-3 Q5_K expert weights ``[E, out_features, raw_bytes_per_row]`` and one
row-major concatenated output ``[compact_rows, a+b]``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q5_k_q8_1_selected_prefill.hip")
_OUTPUT_NAME = "gguf_q5_k_q8_1_selected_prefill.so"
_SYMBOL_IU8_RISK_BF16 = (
    "hipengine_gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"
)
_SYMBOL_SPARSE_EXACT_REPAIR_BF16 = (
    "hipengine_gguf_q5_k_selected_dual_sparse_exact_repair_bf16"
)
_Q5_K_BLOCK = 256


def _extra_flags() -> tuple[str, ...]:
    value = os.environ.get("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS")
    if not value:
        return ("-mcumode",)
    min_blocks = int(value)
    if min_blocks not in {1, 2, 4, 8}:
        raise ValueError(
            "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS must be one of 1, 2, 4, 8"
        )
    return ("-mcumode", f"-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS={min_blocks}")


def plan_gguf_q5_k_q8_1_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q5_k_q8_1_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q5_k_q8_1_selected_prefill(
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
        family="gguf_q5_k_q8_1_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_positive(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if in_features % _Q5_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q5_K block size 256")
    if out_features_a % 16 != 0:
        raise ValueError("out_features_a must be a multiple of 16")
    if out_features_b % 16 != 0:
        raise ValueError("out_features_b must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _launch_wmma_iu8_risk(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    risk_multiplier: float,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    if in_features % 256 != 0:
        raise ValueError("iu8 selected dual prefill requires in_features % 256 == 0")
    if int(risk_count_ptr) <= 0 or int(risk_indices_ptr) <= 0:
        raise ValueError("iu8 risk prefill requires risk counter and index buffers")
    if int(max_risks) < 0 or int(max_risks) > 2**31 - 1:
        raise ValueError("max_risks must be a non-negative int32 count")
    if not (risk_multiplier >= 0.0):
        raise ValueError("risk_multiplier must be non-negative")
    library = library or build_gguf_q5_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_IU8_RISK_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_float(risk_multiplier),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_sparse_exact_repair(
    input_ptr: int,
    expert_start_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    grid_blocks: int = 1024,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    for value, name in (
        (compact_rows, "compact_rows"),
        (in_features, "in_features"),
        (out_features_a, "out_features_a"),
        (out_features_b, "out_features_b"),
        (num_experts, "num_experts"),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if in_features % 256 != 0:
        raise ValueError("sparse exact repair requires in_features % 256 == 0")
    if int(risk_count_ptr) <= 0 or int(risk_indices_ptr) <= 0:
        raise ValueError("sparse exact repair requires risk buffers")
    if int(max_risks) < 0 or int(grid_blocks) <= 0:
        raise ValueError("max_risks must be non-negative and grid_blocks positive")
    library = library or build_gguf_q5_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SPARSE_EXACT_REPAIR_BF16)
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 7 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(expert_start_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(risk_count_ptr),
        ctypes.c_void_p(risk_indices_ptr),
        ctypes.c_int64(max_risks),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(grid_blocks),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    risk_multiplier: float,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the risk-collecting Q5_K iu8-WMMA selected dual prefill."""

    _launch_wmma_iu8_risk(
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        risk_count_ptr,
        risk_indices_ptr,
        max_risks,
        risk_multiplier,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_selected_dual_sparse_exact_repair_bf16(
    input_ptr: int,
    expert_start_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    risk_count_ptr: int,
    risk_indices_ptr: int,
    max_risks: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    grid_blocks: int = 1024,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Repair queued at-risk Q5_K iu8-WMMA selected dual outputs exactly."""

    _launch_sparse_exact_repair(
        input_ptr,
        expert_start_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        risk_count_ptr,
        risk_indices_ptr,
        max_risks,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        grid_blocks=grid_blocks,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_gguf_q5_k_q8_1_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register the Q5_K iu8-WMMA risk+repair selected dual kernels.

    Registers under both the hip_gfx1100 source backend and the hip_gfx1151
    alias so resolution works regardless of when the gfx1151 alias pass
    runs relative to this registration (the production runner resolves
    gfx1151 keys lazily inside run_qwen4_exp_moe).
    """

    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(
                backend,
                "moe_linear",
                "gguf_q5_k",
                "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            ),
            gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out,
            replace=replace,
        )
        register(
            KernelKey(
                backend,
                "moe_linear",
                "gguf_q5_k",
                "selected_dual_sparse_exact_repair_bf16",
            ),
            gguf_q5_k_selected_dual_sparse_exact_repair_bf16,
            replace=replace,
        )
