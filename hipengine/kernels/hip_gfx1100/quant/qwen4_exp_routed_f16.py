"""Wide-tile routed-expert F16 WMMA GEMMs over raw GGUF expert weights.

The kernels read the raw Q4_K / Q5_K / Q5_1 / Q8_0 expert tensors directly,
decode each weight tile to F16 once per ``tile_rows`` grouped rows, and
accumulate in F32 with no repair pass. Inputs and outputs are BF16 rows in
the grouped (expert-sorted) order produced by the grouped MoE scheduler.

The kernel body is ported from gufo v0.12.0 (commit 88090afb, MIT License);
see the header of ``qwen4_exp_routed_f16.hip``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_routed_f16.hip")
_FAMILY = "qwen4_exp_routed_f16"
_OUTPUT_NAME = "qwen4_exp_routed_f16.so"
_SYMBOL_TILE_MAP = "hipengine_qwen4_exp_routed_f16_tile_map"
_SYMBOL_GATE_UP = "hipengine_qwen4_exp_routed_f16_gate_up_silu_bf16"
_SYMBOL_GATE_UP_INDEXED = _SYMBOL_GATE_UP + "_indexed"
_SYMBOL_LINEAR = "hipengine_qwen4_exp_routed_f16_linear_bf16"

# GGUF type ids the kernels accept, by registry quant.
ROUTED_F16_GGML_TYPES = {
    "gguf_q5_1": 7,
    "gguf_q8_0": 8,
    "gguf_q4_k": 12,
    "gguf_q5_k": 13,
}
ROUTED_F16_GATE_UP_VARIANT = "selected_routed_f16_dual_silu_wmma_prefill_bf16_bf16_out"
ROUTED_F16_LINEAR_VARIANT = "selected_routed_f16_wmma_prefill_bf16_bf16_out"
_GATE_UP_TILE_ROWS = (64, 128)
_LINEAR_TILE_ROWS = {
    "gguf_q5_1": (16, 48, 64),
    "gguf_q8_0": (16, 48, 64),
    "gguf_q4_k": (16, 48),
    "gguf_q5_k": (16, 48),
}
_K_BLOCK = {"gguf_q5_1": 64, "gguf_q8_0": 64, "gguf_q4_k": 256, "gguf_q5_k": 256}


def plan_qwen4_exp_routed_f16_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_routed_f16(
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
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def routed_f16_tile_capacity(compact_rows: int, num_experts: int, tile_rows: int) -> int:
    """Static bound on the (expert, macro tile) entries of a grouped prefix."""

    if compact_rows <= 0 or num_experts <= 0 or tile_rows <= 0:
        raise ValueError("routed F16 tile capacity needs positive sizes")
    active = min(compact_rows, num_experts)
    return active + (compact_rows - active) // tile_rows


def _check(err: int, runtime: HipRuntime, name: str) -> None:
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{name} failed: {runtime.error_string(err)}")


def qwen4_exp_routed_f16_tile_map(
    expert_start_ptr: int,
    tiles_ptr: int,
    tile_count_ptr: int,
    num_experts: int,
    *,
    tile_rows: int,
    tile_capacity: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expand the int64 grouped prefix into int32 (expert, macro tile) entries.

    ``tiles_ptr`` holds ``tile_capacity`` int32 entries and ``tile_count_ptr``
    one int32. A prefix that needs more than ``tile_capacity`` entries
    publishes a zero count, so the GEMMs launch no tile.
    """

    if not 0 < num_experts <= 1024:
        raise ValueError("routed F16 tile map supports 1..1024 experts")
    if tile_rows < 16 or tile_rows % 16 or tile_capacity <= 0:
        raise ValueError("routed F16 tile map needs 16-multiple tile rows")
    library = library or build_qwen4_exp_routed_f16(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TILE_MAP)
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _check(
        fn(
            ctypes.c_void_p(expert_start_ptr),
            ctypes.c_void_p(tiles_ptr),
            ctypes.c_void_p(tile_count_ptr),
            ctypes.c_int64(num_experts),
            ctypes.c_int64(tile_rows),
            ctypes.c_int64(tile_capacity),
            ctypes.c_void_p(stream),
        ),
        runtime,
        _SYMBOL_TILE_MAP,
    )


def _make_gate_up(quant: str):
    ggml_type = ROUTED_F16_GGML_TYPES[quant]

    def gate_up(
        x_ptr: int,
        expert_start_ptr: int,
        tiles_ptr: int,
        tile_count_ptr: int,
        gate_ptr: int,
        up_ptr: int,
        out_ptr: int,
        in_features: int,
        out_features: int,
        *,
        tile_rows: int,
        tile_capacity: int,
        row_lanes_ptr: int = 0,
        top_k: int = 1,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        """out[row] = silu(gate[e] x[row]) * (up[e] x[row]) over grouped rows.

        With ``row_lanes_ptr`` (int64 routed lane of each grouped row), ``x``
        is token-major and grouped row ``r`` reads token
        ``row_lanes[r] // top_k``; without it ``x`` holds the grouped rows.
        """

        if tile_rows not in _GATE_UP_TILE_ROWS:
            raise ValueError(f"routed F16 gate/up tile rows must be in {_GATE_UP_TILE_ROWS}")
        if in_features <= 0 or in_features % 256 or out_features <= 0 or tile_capacity <= 0:
            raise ValueError("routed F16 gate/up needs K % 256 == 0 and positive sizes")
        lib = library or build_qwen4_exp_routed_f16(load=True)
        active = runtime or get_hip_runtime()
        fn = getattr(lib, _SYMBOL_GATE_UP_INDEXED)
        fn.argtypes = (
            [ctypes.c_void_p] * 2 + [ctypes.c_int64] + [ctypes.c_void_p] * 5
            + [ctypes.c_int64] * 4 + [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
        )
        fn.restype = ctypes.c_int
        _check(
            fn(
                ctypes.c_void_p(gate_ptr),
                ctypes.c_void_p(up_ptr),
                ctypes.c_int64(ggml_type),
                ctypes.c_void_p(x_ptr),
                ctypes.c_void_p(tiles_ptr),
                ctypes.c_void_p(tile_count_ptr),
                ctypes.c_void_p(expert_start_ptr),
                ctypes.c_void_p(out_ptr),
                ctypes.c_int64(out_features),
                ctypes.c_int64(in_features),
                ctypes.c_int64(tile_rows),
                ctypes.c_int64(tile_capacity),
                ctypes.c_void_p(row_lanes_ptr or None),
                ctypes.c_int64(top_k),
                ctypes.c_void_p(stream),
            ),
            active,
            _SYMBOL_GATE_UP_INDEXED,
        )

    gate_up.__name__ = f"qwen4_exp_{quant}_routed_f16_gate_up_silu_bf16_bf16_out"
    return gate_up


def _make_linear(quant: str):
    ggml_type = ROUTED_F16_GGML_TYPES[quant]

    def linear(
        x_ptr: int,
        expert_start_ptr: int,
        tiles_ptr: int,
        tile_count_ptr: int,
        weights_ptr: int,
        out_ptr: int,
        in_features: int,
        out_features: int,
        *,
        tile_rows: int,
        tile_capacity: int,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        """out[row] = w[e] x[row] over grouped rows."""

        if tile_rows not in _LINEAR_TILE_ROWS[quant]:
            raise ValueError(
                f"routed F16 {quant} tile rows must be in {_LINEAR_TILE_ROWS[quant]}"
            )
        if (
            in_features <= 0
            or in_features % _K_BLOCK[quant]
            or out_features <= 0
            or tile_capacity <= 0
        ):
            raise ValueError(
                f"routed F16 {quant} needs K % {_K_BLOCK[quant]} == 0 and positive sizes"
            )
        lib = library or build_qwen4_exp_routed_f16(load=True)
        active = runtime or get_hip_runtime()
        fn = getattr(lib, _SYMBOL_LINEAR)
        fn.argtypes = (
            [ctypes.c_void_p, ctypes.c_int64] + [ctypes.c_void_p] * 5
            + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
        )
        fn.restype = ctypes.c_int
        _check(
            fn(
                ctypes.c_void_p(weights_ptr),
                ctypes.c_int64(ggml_type),
                ctypes.c_void_p(x_ptr),
                ctypes.c_void_p(tiles_ptr),
                ctypes.c_void_p(tile_count_ptr),
                ctypes.c_void_p(expert_start_ptr),
                ctypes.c_void_p(out_ptr),
                ctypes.c_int64(out_features),
                ctypes.c_int64(in_features),
                ctypes.c_int64(tile_rows),
                ctypes.c_int64(tile_capacity),
                ctypes.c_void_p(stream),
            ),
            active,
            _SYMBOL_LINEAR,
        )

    linear.__name__ = f"qwen4_exp_{quant}_routed_f16_linear_bf16_bf16_out"
    return linear


qwen4_exp_q4_k_routed_f16_gate_up_silu_bf16_bf16_out = _make_gate_up("gguf_q4_k")
qwen4_exp_q5_k_routed_f16_gate_up_silu_bf16_bf16_out = _make_gate_up("gguf_q5_k")
qwen4_exp_q5_1_routed_f16_linear_bf16_bf16_out = _make_linear("gguf_q5_1")
qwen4_exp_q8_0_routed_f16_linear_bf16_bf16_out = _make_linear("gguf_q8_0")
qwen4_exp_q4_k_routed_f16_linear_bf16_bf16_out = _make_linear("gguf_q4_k")
qwen4_exp_q5_k_routed_f16_linear_bf16_bf16_out = _make_linear("gguf_q5_k")


def register_qwen4_exp_routed_f16_kernels(*, replace: bool = False) -> None:
    for quant, fn in (
        ("gguf_q4_k", qwen4_exp_q4_k_routed_f16_gate_up_silu_bf16_bf16_out),
        ("gguf_q5_k", qwen4_exp_q5_k_routed_f16_gate_up_silu_bf16_bf16_out),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant, ROUTED_F16_GATE_UP_VARIANT),
            fn,
            replace=replace,
        )
    for quant, fn in (
        ("gguf_q5_1", qwen4_exp_q5_1_routed_f16_linear_bf16_bf16_out),
        ("gguf_q8_0", qwen4_exp_q8_0_routed_f16_linear_bf16_bf16_out),
        ("gguf_q4_k", qwen4_exp_q4_k_routed_f16_linear_bf16_bf16_out),
        ("gguf_q5_k", qwen4_exp_q5_k_routed_f16_linear_bf16_bf16_out),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant, ROUTED_F16_LINEAR_VARIANT),
            fn,
            replace=replace,
        )
    # The Q5_K gate/up and Q8_0 expert-down scopes are declared under the
    # ``linear`` layer in the Qwen4Exp plans, so the same routes are keyed
    # there for manifest resolution.
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q5_k", ROUTED_F16_GATE_UP_VARIANT),
        qwen4_exp_q5_k_routed_f16_gate_up_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", ROUTED_F16_LINEAR_VARIANT),
        qwen4_exp_q8_0_routed_f16_linear_bf16_bf16_out,
        replace=replace,
    )


register_qwen4_exp_routed_f16_kernels()

__all__ = [
    "ROUTED_F16_GATE_UP_VARIANT",
    "ROUTED_F16_GGML_TYPES",
    "ROUTED_F16_LINEAR_VARIANT",
    "build_qwen4_exp_routed_f16",
    "plan_qwen4_exp_routed_f16_build",
    "qwen4_exp_q4_k_routed_f16_gate_up_silu_bf16_bf16_out",
    "qwen4_exp_q4_k_routed_f16_linear_bf16_bf16_out",
    "qwen4_exp_q5_1_routed_f16_linear_bf16_bf16_out",
    "qwen4_exp_q5_k_routed_f16_gate_up_silu_bf16_bf16_out",
    "qwen4_exp_q5_k_routed_f16_linear_bf16_bf16_out",
    "qwen4_exp_q8_0_routed_f16_linear_bf16_bf16_out",
    "qwen4_exp_routed_f16_tile_map",
    "register_qwen4_exp_routed_f16_kernels",
    "routed_f16_tile_capacity",
]
