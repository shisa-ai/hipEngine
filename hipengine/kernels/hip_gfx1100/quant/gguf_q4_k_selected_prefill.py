"""Raw-pointer wrappers for GGUF Q4_K selected compact WMMA prefill.

P8.4 grouped/selected MoE gate+up kernel. The wrapper consumes the compact
scheduler ABI already emitted by ``qwen35_moe_group_count/prefix``,
``qwen35_moe_group_scatter_gather``, and ``qwen35_moe_wmma_tile_map``:

``x[compact_rows, in_features]``, ``expert_start_compact[E+1]``,
``expert_start_wmma[E+1]``, ``tile_expert[wmma_total_rows/16]``, raw rank-3
Q4_K expert weights ``[E, out_features, raw_bytes_per_row]``, and one
row-major concatenated output ``[compact_rows, out_features_a+out_features_b]``.

No new compact-MoE ABI and no resident weight sidecar/repack are introduced;
the HIP kernel only swaps the AWQ compact WMMA inner dequant loop for raw GGUF
``block_q4_K`` decoding.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_selected_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_selected_prefill.so"
_SYMBOL_DUAL_BF16 = "hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out"
_SYMBOL_DUAL_FP16 = "hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out"
_SYMBOL_HOT_BF16 = "hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out"
_SYMBOL_HOT_FP16 = "hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out"
_SYMBOL_SIDEMETA_BF16 = "hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out"
_SYMBOL_SIDEMETA_FP16 = "hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out"
_Q4_K_BLOCK = 256
_ALLOWED_TILES = {(16, 16), (32, 16), (16, 32), (32, 32), (64, 16), (64, 32)}
_ENV_TILE_M = "HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M"
_ENV_TILE_N = "HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_N"
_ENV_LAUNCH_BOUNDS = "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS"


def plan_gguf_q4_k_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_selected_prefill(
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
        family="gguf_q4_k_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def q4_k_predecode_scale_min_sidemeta(raw: object) -> object:
    """Predecode GGUF Q4_K per-block scale/min metadata to float32.

    ``raw`` must have shape ``[..., row_bytes]`` where ``row_bytes`` is a
    multiple of the Q4_K block size (144 bytes). The returned array has shape
    ``[..., blocks_per_row, 8, 2]`` with the last dimension ``(scale, min)``.
    The quant nibbles remain in the raw weight tensor; this sidecar only removes
    the d/dmin + packed-scale bitfield decode from the WMMA inner loop.
    """

    import numpy as np

    arr = np.ascontiguousarray(raw, dtype=np.uint8)
    if arr.ndim < 1 or arr.shape[-1] % 144 != 0:
        raise ValueError("raw Q4_K side metadata input must end in row_bytes divisible by 144")
    blocks_per_row = arr.shape[-1] // 144
    blocks = arr.reshape(*arr.shape[:-1], blocks_per_row, 144)
    d_bits = blocks[..., 0].astype(np.uint16) | (blocks[..., 1].astype(np.uint16) << 8)
    dmin_bits = blocks[..., 2].astype(np.uint16) | (blocks[..., 3].astype(np.uint16) << 8)
    d = d_bits.view(np.float16).astype(np.float32)
    dmin = dmin_bits.view(np.float16).astype(np.float32)
    packed = blocks[..., 4:16]
    scale = np.empty((*arr.shape[:-1], blocks_per_row, 8), dtype=np.uint8)
    mins = np.empty_like(scale)
    scale[..., 0:4] = packed[..., 0:4] & np.uint8(0x3F)
    mins[..., 0:4] = packed[..., 4:8] & np.uint8(0x3F)
    for idx in range(4):
        scale[..., 4 + idx] = (packed[..., 8 + idx] & np.uint8(0x0F)) | (
            (packed[..., idx] >> np.uint8(2)) & np.uint8(0x30)
        )
        mins[..., 4 + idx] = (packed[..., 8 + idx] >> np.uint8(4)) | (
            (packed[..., 4 + idx] >> np.uint8(2)) & np.uint8(0x30)
        )
    out = np.empty((*arr.shape[:-1], blocks_per_row, 8, 2), dtype=np.float16)
    out[..., 0] = (d[..., None] * scale.astype(np.float32)).astype(np.float16)
    out[..., 1] = (dmin[..., None] * mins.astype(np.float32)).astype(np.float16)
    return np.ascontiguousarray(out)


def _extra_flags() -> tuple[str, ...]:
    value = os.environ.get(_ENV_LAUNCH_BOUNDS)
    if not value:
        return ("-mcumode",)
    min_blocks = int(value)
    if min_blocks not in {1, 2, 4, 8}:
        raise ValueError(f"{_ENV_LAUNCH_BOUNDS} must be one of 1, 2, 4, 8")
    return ("-mcumode", f"-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS={min_blocks}")


def gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact raw-Q4_K dual gate+up WMMA prefill."""

    _launch_dual(
        _SYMBOL_DUAL_BF16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        tile_m=tile_m,
        tile_n=tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    sidemeta_a_ptr: int,
    sidemeta_b_ptr: int,
    out_ptr: int,
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
    """Launch P9.C5 Q4_K side-metadata dual WMMA prototype (BF16)."""

    _launch_sidemeta(
        _SYMBOL_SIDEMETA_BF16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        sidemeta_a_ptr,
        sidemeta_b_ptr,
        out_ptr,
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


def gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    sidemeta_a_ptr: int,
    sidemeta_b_ptr: int,
    out_ptr: int,
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
    """Launch P9.C5 Q4_K side-metadata dual WMMA prototype (FP16)."""

    _launch_sidemeta(
        _SYMBOL_SIDEMETA_FP16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        sidemeta_a_ptr,
        sidemeta_b_ptr,
        out_ptr,
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


def gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    hot_threshold: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch P9.C4 hot/full-tile Q4_K dual prototype (BF16)."""

    _launch_hot_fulltile(
        _SYMBOL_HOT_BF16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        hot_threshold=hot_threshold,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact raw-Q4_K dual gate+up WMMA prefill."""

    _launch_dual(
        _SYMBOL_DUAL_FP16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        tile_m=tile_m,
        tile_n=tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    hot_threshold: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch P9.C4 hot/full-tile Q4_K dual prototype (FP16)."""

    _launch_hot_fulltile(
        _SYMBOL_HOT_FP16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        hot_threshold=hot_threshold,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    tile_m: int | None,
    tile_n: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    tile_m, tile_n = _resolve_tiles(tile_m, tile_n)
    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
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
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_int64(tile_m),
        ctypes.c_int64(tile_n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_sidemeta(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    sidemeta_a_ptr: int,
    sidemeta_b_ptr: int,
    out_ptr: int,
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
    library = library or build_gguf_q4_k_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(sidemeta_a_ptr),
        ctypes.c_void_p(sidemeta_b_ptr),
        ctypes.c_void_p(out_ptr),
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


def _launch_hot_fulltile(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    hot_threshold: int,
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
    if hot_threshold <= 0:
        raise ValueError("hot_threshold must be positive")
    library = library or build_gguf_q4_k_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
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
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_int64(hot_threshold),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _resolve_tiles(tile_m: int | None, tile_n: int | None) -> tuple[int, int]:
    if tile_m is None:
        value = os.environ.get(_ENV_TILE_M)
        tile_m = int(value) if value else 32
    if tile_n is None:
        value = os.environ.get(_ENV_TILE_N)
        tile_n = int(value) if value else 16
    if (tile_m, tile_n) not in _ALLOWED_TILES:
        allowed = ", ".join(f"({m}, {n})" for m, n in sorted(_ALLOWED_TILES))
        raise ValueError(
            f"tile (tile_m={tile_m}, tile_n={tile_n}) is not supported; "
            f"supported tiles: {allowed}"
        )
    return tile_m, tile_n


def selected_dual_wmma_prefill_compact_default_tiles() -> tuple[int, int]:
    """Return the P9.C1 default tile for Q4_K selected dual WMMA prefill.

    The generic multi-tile variants remain available for sweeps via env
    overrides; measured qwen35moe 512/0 rocprof evidence picks 32x16 as the
    default for the Q4_K dual gate+up shape.
    """

    return _resolve_tiles(None, None)


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
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if out_features_a % 16 != 0:
        raise ValueError("out_features_a must be a multiple of 16")
    if out_features_b % 16 != 0:
        raise ValueError("out_features_b must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_q4_k_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register P8.4 compact selected raw-Q4_K WMMA kernels."""

    # Primary P8.4 key: compact ABI, row-major concatenated gate+up output.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_compact_fp16_fp16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out,
        replace=replace,
    )
    # Alias matching the docs/GGUF.md pipeline shorthand
    # ``gguf_q4_k_selected_dual_wmma_prefill``. Future runtime code can choose
    # either spelling without changing the wrapper ABI.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_fp16_fp16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out",
        ),
        gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out,
        replace=replace,
    )


register_gguf_q4_k_selected_prefill_kernels()


__all__ = [
    "build_gguf_q4_k_selected_prefill",
    "gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out",
    "gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out",
    "gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out",
    "gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out",
    "q4_k_predecode_scale_min_sidemeta",
    "plan_gguf_q4_k_selected_prefill_build",
    "register_gguf_q4_k_selected_prefill_kernels",
    "selected_dual_wmma_prefill_compact_default_tiles",
]
