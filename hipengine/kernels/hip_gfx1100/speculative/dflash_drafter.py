"""Raw-pointer DFlash drafter root/query preparation wrappers."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dflash_drafter.hip")
_OUTPUT_NAME = "dflash_drafter.so"
_SYMBOL_PREPARE_NOISE_BF16 = "hipengine_dflash_prepare_noise_inputs_bf16_i32"
_SYMBOL_PREPARE_NOISE_F16_TO_BF16 = "hipengine_dflash_prepare_noise_inputs_f16_to_bf16_i32"
_SYMBOL_ADD_BF16 = "hipengine_dflash_add_bf16"
_SYMBOL_CONCAT_F32 = "hipengine_dflash_concat_rows_f32"
_SYMBOL_CONCAT_BF16 = "hipengine_dflash_concat_rows_bf16"
_SYMBOL_RMSNORM_BF16 = "hipengine_dflash_rmsnorm_bf16"
_SYMBOL_ADD_RMSNORM_BF16 = "hipengine_dflash_add_rmsnorm_bf16"
_SYMBOL_SILU_MUL_BF16 = "hipengine_dflash_silu_mul_bf16"
_SYMBOL_SILU_MUL_GATE_UP_ROUTES_BF16 = "hipengine_dflash_silu_mul_gate_up_routes_bf16"
_SYMBOL_DENSE_BF16_TO_BF16 = "hipengine_dflash_dense_bf16_to_bf16"
_SYMBOL_DENSE_BF16_TO_BF16_EXPERT = "hipengine_dflash_dense_bf16_to_bf16_expert"
_SYMBOL_DENSE_BF16_TO_BF16_EXPERT_ROUTES = "hipengine_dflash_dense_bf16_to_bf16_expert_routes"
_SYMBOL_DENSE_BF16_TO_F32 = "hipengine_dflash_dense_bf16_to_f32"
_SYMBOL_DENSE_BF16_TO_BF16_WMMA = "hipengine_dflash_dense_bf16_to_bf16_wmma"
_SYMBOL_DENSE_BF16_TO_F32_WMMA = "hipengine_dflash_dense_bf16_to_f32_wmma"
_SYMBOL_QKV_PROJ_BF16_MIXED = "hipengine_dflash_qkv_proj_bf16_mixed"
_SYMBOL_QKV_PROJ_BF16_MIXED_INDEXED_V = "hipengine_dflash_qkv_proj_bf16_mixed_indexed_v"
_SYMBOL_HEAD_RMS_ROTARY = "hipengine_dflash_head_rmsnorm_rotary_f32"
_SYMBOL_HEAD_RMS_ROTARY_INDEXED_KEY = "hipengine_dflash_head_rmsnorm_rotary_indexed_key_f32"
_SYMBOL_KEY_RMS_ROTARY = "hipengine_dflash_key_rmsnorm_rotary_f32"
_SYMBOL_UPDATE_KV_METADATA = "hipengine_dflash_update_kv_metadata_i32"
_SYMBOL_GQA_ATTENTION = "hipengine_dflash_gqa_attention_f32_bf16"
_SYMBOL_GQA_ATTENTION_BUCKETED = "hipengine_dflash_gqa_attention_f32_bf16_bucketed"
_ALLOWED_THREADS = {64, 128, 256}

_DFLASH_WMMA_TILE_K = 16


def _drafter_dense_use_wmma() -> bool:
    """Return True when the drafter dense kernels should use the WMMA variant.

    Controlled by ``HIPENGINE_DFLASH_DRAFTER_DENSE`` (``naive`` / ``wmma``).
    Defaults to ``wmma`` after R3.4 validated exact-AR + per-prompt perf gain on
    W7900 same-session suite (commit cebf6f7+).  Set ``naive`` to revert.
    """

    value = os.environ.get("HIPENGINE_DFLASH_DRAFTER_DENSE", "wmma").strip().lower()
    if value in {"naive", "0", "false", "off"}:
        return False
    if value in {"", "wmma", "1", "true", "on"}:
        return True
    raise ValueError(
        "HIPENGINE_DFLASH_DRAFTER_DENSE must be one of: naive, wmma (got " + repr(value) + ")"
    )


def plan_dflash_drafter_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dflash_drafter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dflash_drafter(
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
        family="dflash_drafter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dflash_prepare_noise_inputs_bf16_i32(
    root_tokens_i32_ptr: int,
    root_positions_i32_ptr: int,
    embed_tokens_bf16_ptr: int,
    noise_token_ids_i32_ptr: int,
    position_ids_i32_ptr: int,
    noise_embeddings_bf16_ptr: int,
    request_count: int,
    block_size: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Materialize DFlash root+mask ids, positions, and BF16 embedding rows."""

    _launch_prepare_noise_inputs(
        _SYMBOL_PREPARE_NOISE_BF16,
        root_tokens_i32_ptr,
        root_positions_i32_ptr,
        embed_tokens_bf16_ptr,
        noise_token_ids_i32_ptr,
        position_ids_i32_ptr,
        noise_embeddings_bf16_ptr,
        request_count,
        block_size,
        hidden_size,
        vocab_size,
        mask_token_id,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def dflash_prepare_noise_inputs_f16_to_bf16_i32(
    root_tokens_i32_ptr: int,
    root_positions_i32_ptr: int,
    embed_tokens_f16_ptr: int,
    noise_token_ids_i32_ptr: int,
    position_ids_i32_ptr: int,
    noise_embeddings_bf16_ptr: int,
    request_count: int,
    block_size: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Materialize root+mask inputs while converting FP16 embedding rows to BF16."""

    _launch_prepare_noise_inputs(
        _SYMBOL_PREPARE_NOISE_F16_TO_BF16,
        root_tokens_i32_ptr,
        root_positions_i32_ptr,
        embed_tokens_f16_ptr,
        noise_token_ids_i32_ptr,
        position_ids_i32_ptr,
        noise_embeddings_bf16_ptr,
        request_count,
        block_size,
        hidden_size,
        vocab_size,
        mask_token_id,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def dflash_add_bf16(
    a_bf16_ptr: int,
    b_bf16_ptr: int,
    out_bf16_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Elementwise BF16 residual add for DFlash block wiring."""

    _check_elements(elements, threads)
    _launch_simple(_SYMBOL_ADD_BF16, (a_bf16_ptr, b_bf16_ptr, out_bf16_ptr, elements), threads, stream, library, runtime)


def dflash_concat_rows_f32(
    context_rows_f32_ptr: int,
    query_rows_f32_ptr: int,
    out_rows_f32_ptr: int,
    batch_size: int,
    context_len: int,
    query_len: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Concatenate context+query rows for FP32 K tensors on device."""

    _launch_concat(_SYMBOL_CONCAT_F32, context_rows_f32_ptr, query_rows_f32_ptr, out_rows_f32_ptr, batch_size, context_len, query_len, features, threads, stream, library, runtime)


def dflash_concat_rows_bf16(
    context_rows_bf16_ptr: int,
    query_rows_bf16_ptr: int,
    out_rows_bf16_ptr: int,
    batch_size: int,
    context_len: int,
    query_len: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Concatenate context+query rows for BF16 V tensors on device."""

    _launch_concat(_SYMBOL_CONCAT_BF16, context_rows_bf16_ptr, query_rows_bf16_ptr, out_rows_bf16_ptr, batch_size, context_len, query_len, features, threads, stream, library, runtime)


def dflash_rmsnorm_bf16(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply standard DFlash/Qwen RMSNorm with direct BF16 weight scaling."""

    _check_rmsnorm_shape(rows, hidden_size, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_RMSNORM_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(weight_bf16_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_add_rmsnorm_bf16(
    input_bf16_ptr: int,
    residual_bf16_ptr: int,
    weight_bf16_ptr: int,
    hidden_out_bf16_ptr: int,
    norm_out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """R3.6 C1: fused DFlash post-attention add + RMSNorm.

    Replaces the unfused chain ``dflash_add_bf16(input, residual -> hidden_out)``
    followed by ``dflash_rmsnorm_bf16(hidden_out, weight -> norm_out)``.  The
    fused kernel writes BOTH ``hidden_out`` (the residual sum needed by the MLP
    residual path) AND ``norm_out`` (the normalized output consumed by the next
    projection).  The unfused path remains registered as the fallback.

    Numerically equivalent to the unfused chain because the residual sum is
    rounded to BF16 before the RMS reduction reads it (matching the unfused
    HBM round-trip).

    Constraint: ``hidden_size <= 4096`` so the per-block scratch fits in LDS.
    """

    _check_add_rmsnorm_shape(rows, hidden_size, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ADD_RMSNORM_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(input_bf16_ptr),
        ctypes.c_void_p(residual_bf16_ptr),
        ctypes.c_void_p(weight_bf16_ptr),
        ctypes.c_void_p(hidden_out_bf16_ptr),
        ctypes.c_void_p(norm_out_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_add_rmsnorm_shape(rows: int, hidden_size: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if hidden_size > 4096:
        raise ValueError("add_rmsnorm fused kernel requires hidden_size <= 4096")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _drafter_dense_use_add_rmsnorm() -> bool:
    """Return True when the drafter post-attention path should fuse add+rmsnorm.

    Controlled by ``HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM`` (``off`` / ``on``).
    Defaults to ``off``; flip to ``on`` to opt into the R3.6 C1 fused kernel.
    Existing unfused path (``dflash_add_bf16`` + ``dflash_rmsnorm_bf16``) remains
    the registered fallback.
    """

    value = os.environ.get("HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM", "off").strip().lower()
    if value in {"", "off", "0", "false", "naive"}:
        return False
    if value in {"on", "1", "true", "fused"}:
        return True
    raise ValueError(
        "HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM must be one of: off, on (got " + repr(value) + ")"
    )


def dflash_silu_mul_bf16(
    gate_bf16_ptr: int,
    up_bf16_ptr: int,
    out_bf16_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Elementwise BF16 SiLU(gate) * up for DFlash MLP wiring."""

    _check_elements(elements, threads)
    _launch_simple(_SYMBOL_SILU_MUL_BF16, (gate_bf16_ptr, up_bf16_ptr, out_bf16_ptr, elements), threads, stream, library, runtime)


def dflash_silu_mul_gate_up_routes_bf16(
    gate_up_routes_bf16_ptr: int,
    out_routes_bf16_ptr: int,
    routes: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Route-major BF16 SiLU for packed ``[route][gate, up]`` slabs."""

    if routes <= 0:
        raise ValueError("routes must be positive")
    _check_elements(features, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SILU_MUL_GATE_UP_ROUTES_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(gate_up_routes_bf16_ptr),
        ctypes.c_void_p(out_routes_bf16_ptr),
        ctypes.c_int64(routes),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_dense_bf16_to_bf16(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Project BF16 rows with BF16 weights and write BF16 output rows.

    Honors ``HIPENGINE_DFLASH_DRAFTER_DENSE={naive,wmma}`` (default ``wmma``)
    so the WMMA variant can be A/B tested without touching call sites.  The
    WMMA variant requires ``in_features % 16 == 0``; otherwise it transparently
    falls back to the naive kernel.
    """

    if _drafter_dense_use_wmma() and (in_features % _DFLASH_WMMA_TILE_K) == 0:
        dflash_dense_bf16_to_bf16_wmma(
            x_bf16_ptr,
            weight_bf16_ptr,
            out_bf16_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return
    _launch_dense(_SYMBOL_DENSE_BF16_TO_BF16, x_bf16_ptr, weight_bf16_ptr, out_bf16_ptr, rows, in_features, out_features, threads, stream, library, runtime)


def dflash_dense_bf16_to_bf16_expert(
    x_bf16_ptr: int,
    expert_weights_base_ptr: int,
    expert_ids_i32_ptr: int,
    out_bf16_ptr: int,
    route: int,
    expert_stride_elems: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expert-indexed dense GEMV: weight base resolved on-device from
    ``expert_ids[route]``; no router D2H readback (graph-capture-safe)."""

    _check_dense_shape(rows, in_features, out_features, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_BF16_TO_BF16_EXPERT)
    fn.argtypes = [
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
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(expert_weights_base_ptr),
        ctypes.c_void_p(expert_ids_i32_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(route),
        ctypes.c_int64(expert_stride_elems),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_dense_bf16_to_bf16_expert_routes(
    x_bf16_ptr: int,
    expert_weights_base_ptr: int,
    expert_ids_i32_ptr: int,
    out_bf16_ptr: int,
    routes: int,
    x_route_stride_elems: int,
    expert_stride_elems: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expert-indexed dense GEMV for all routes in one launch.

    ``x_route_stride_elems == 0`` means every route reads the same input row.
    Positive strides mean route ``r`` reads ``x + r * x_route_stride_elems``.
    Output is route-major ``[routes, rows, out_features]``.
    """

    if routes <= 0:
        raise ValueError("routes must be positive")
    if x_route_stride_elems < 0:
        raise ValueError("x_route_stride_elems must be non-negative")
    if expert_stride_elems <= 0:
        raise ValueError("expert_stride_elems must be positive")
    _check_dense_shape(rows, in_features, out_features, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_BF16_TO_BF16_EXPERT_ROUTES)
    fn.argtypes = [
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
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(expert_weights_base_ptr),
        ctypes.c_void_p(expert_ids_i32_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(routes),
        ctypes.c_int64(x_route_stride_elems),
        ctypes.c_int64(expert_stride_elems),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_dense_bf16_to_f32(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_f32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Project BF16 rows with BF16 weights and write FP32 output rows.

    Honors ``HIPENGINE_DFLASH_DRAFTER_DENSE={naive,wmma}`` (default ``naive``).
    WMMA fallback requirements match :func:`dflash_dense_bf16_to_bf16`.
    """

    if _drafter_dense_use_wmma() and (in_features % _DFLASH_WMMA_TILE_K) == 0:
        dflash_dense_bf16_to_f32_wmma(
            x_bf16_ptr,
            weight_bf16_ptr,
            out_f32_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return
    _launch_dense(_SYMBOL_DENSE_BF16_TO_F32, x_bf16_ptr, weight_bf16_ptr, out_f32_ptr, rows, in_features, out_features, threads, stream, library, runtime)


def dflash_dense_bf16_to_bf16_wmma(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """WMMA-tiled BF16-to-BF16 dense kernel for the DFlash drafter.

    Uses RDNA3's ``v_wmma_f32_16x16x16_bf16`` over wave32 workgroups.  Each
    workgroup computes one 16x16 output tile.  ``in_features`` must be a
    multiple of 16; ``rows`` and ``out_features`` may be arbitrary positive
    values (tiles are padded and stores are bounds-checked).
    """

    _launch_dense_wmma(
        _SYMBOL_DENSE_BF16_TO_BF16_WMMA,
        x_bf16_ptr,
        weight_bf16_ptr,
        out_bf16_ptr,
        rows,
        in_features,
        out_features,
        stream,
        library,
        runtime,
    )


def dflash_dense_bf16_to_f32_wmma(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_f32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """WMMA-tiled BF16-to-FP32 dense kernel for the DFlash drafter.

    Same shape constraints as :func:`dflash_dense_bf16_to_bf16_wmma`.
    """

    _launch_dense_wmma(
        _SYMBOL_DENSE_BF16_TO_F32_WMMA,
        x_bf16_ptr,
        weight_bf16_ptr,
        out_f32_ptr,
        rows,
        in_features,
        out_features,
        stream,
        library,
        runtime,
    )


def dflash_qkv_proj_bf16_mixed(
    x_bf16_ptr: int,
    q_weight_bf16_ptr: int,
    k_weight_bf16_ptr: int,
    v_weight_bf16_ptr: int,
    q_out_f32_ptr: int,
    k_out_f32_ptr: int,
    v_out_bf16_ptr: int,
    rows: int,
    in_features: int,
    q_features: int,
    kv_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused DFlash query-side Q/K/V projections.

    This is numerically equivalent to the unfused sequence
    ``dense_bf16_to_f32(Q)``, ``dense_bf16_to_f32(K)``, and
    ``dense_bf16_to_bf16(V)``.  It reduces three tiny drafter launches to one
    while preserving the same per-output-column reduction order.
    """

    _launch_qkv_proj(
        _SYMBOL_QKV_PROJ_BF16_MIXED,
        x_bf16_ptr,
        q_weight_bf16_ptr,
        k_weight_bf16_ptr,
        v_weight_bf16_ptr,
        q_out_f32_ptr,
        k_out_f32_ptr,
        v_out_bf16_ptr,
        rows,
        in_features,
        q_features,
        kv_features,
        threads,
        stream,
        library,
        runtime,
    )


def dflash_qkv_proj_bf16_mixed_indexed_v(
    x_bf16_ptr: int,
    q_weight_bf16_ptr: int,
    k_weight_bf16_ptr: int,
    v_weight_bf16_ptr: int,
    q_out_f32_ptr: int,
    k_out_f32_ptr: int,
    value_cache_bf16_base_ptr: int,
    cache_slot_i32_ptr: int,
    cache_rows: int,
    rows: int,
    in_features: int,
    q_features: int,
    kv_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused Q/K/V projection with V written to a device-selected cache slot."""

    _launch_qkv_proj_indexed_v(
        _SYMBOL_QKV_PROJ_BF16_MIXED_INDEXED_V,
        x_bf16_ptr,
        q_weight_bf16_ptr,
        k_weight_bf16_ptr,
        v_weight_bf16_ptr,
        q_out_f32_ptr,
        k_out_f32_ptr,
        value_cache_bf16_base_ptr,
        cache_slot_i32_ptr,
        cache_rows,
        rows,
        in_features,
        q_features,
        kv_features,
        threads,
        stream,
        library,
        runtime,
    )


def dflash_head_rmsnorm_rotary_f32(
    query_f32_ptr: int,
    key_f32_ptr: int,
    q_weight_bf16_ptr: int,
    k_weight_bf16_ptr: int,
    cos_table_f32_ptr: int,
    sin_table_f32_ptr: int,
    query_positions_i32_ptr: int,
    key_positions_i32_ptr: int,
    query_out_f32_ptr: int,
    key_out_f32_ptr: int,
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply direct-weight head RMSNorm plus rotary to DFlash Q/K projections."""

    _check_head_rotary_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, max_positions, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_HEAD_RMS_ROTARY)
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
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_f32_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(q_weight_bf16_ptr),
        ctypes.c_void_p(k_weight_bf16_ptr),
        ctypes.c_void_p(cos_table_f32_ptr),
        ctypes.c_void_p(sin_table_f32_ptr),
        ctypes.c_void_p(query_positions_i32_ptr),
        ctypes.c_void_p(key_positions_i32_ptr),
        ctypes.c_void_p(query_out_f32_ptr),
        ctypes.c_void_p(key_out_f32_ptr),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(query_len),
        ctypes.c_int64(kv_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_head_rmsnorm_rotary_indexed_key_f32(
    query_f32_ptr: int,
    key_f32_ptr: int,
    q_weight_bf16_ptr: int,
    k_weight_bf16_ptr: int,
    cos_table_f32_ptr: int,
    sin_table_f32_ptr: int,
    query_positions_i32_ptr: int,
    key_positions_i32_ptr: int,
    query_out_f32_ptr: int,
    key_cache_f32_base_ptr: int,
    cache_slot_i32_ptr: int,
    cache_rows: int,
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply head RMSNorm/RoPE with K written to a device-selected cache slot."""

    _check_head_rotary_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, max_positions, threads)
    if batch_size != 1:
        raise ValueError("indexed-key head rotary currently requires batch_size=1")
    if cache_rows <= 0:
        raise ValueError("cache_rows must be positive")
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_HEAD_RMS_ROTARY_INDEXED_KEY)
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
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_f32_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(q_weight_bf16_ptr),
        ctypes.c_void_p(k_weight_bf16_ptr),
        ctypes.c_void_p(cos_table_f32_ptr),
        ctypes.c_void_p(sin_table_f32_ptr),
        ctypes.c_void_p(query_positions_i32_ptr),
        ctypes.c_void_p(key_positions_i32_ptr),
        ctypes.c_void_p(query_out_f32_ptr),
        ctypes.c_void_p(key_cache_f32_base_ptr),
        ctypes.c_void_p(cache_slot_i32_ptr),
        ctypes.c_int64(cache_rows),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(query_len),
        ctypes.c_int64(kv_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_key_rmsnorm_rotary_f32(
    key_f32_ptr: int,
    k_weight_bf16_ptr: int,
    cos_table_f32_ptr: int,
    sin_table_f32_ptr: int,
    key_positions_i32_ptr: int,
    key_out_f32_ptr: int,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply direct-weight K head RMSNorm plus rotary for draft context materialization."""

    _check_key_rotary_shape(rows, num_kv_heads, head_dim, rotary_dim, max_positions, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_KEY_RMS_ROTARY)
    fn.argtypes = [
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
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(k_weight_bf16_ptr),
        ctypes.c_void_p(cos_table_f32_ptr),
        ctypes.c_void_p(sin_table_f32_ptr),
        ctypes.c_void_p(key_positions_i32_ptr),
        ctypes.c_void_p(key_out_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_update_kv_metadata_i32(
    append_positions_i32_ptr: int,
    cache_positions_i32_ptr: int,
    live_count_i32_ptr: int,
    *,
    start: int,
    count: int,
    end: int,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy appended positions and update draft KV live count on device."""

    if start < 0:
        raise ValueError("start must be non-negative")
    if count < 0:
        raise ValueError("count must be non-negative")
    if end < start:
        raise ValueError("end must be no smaller than start")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_UPDATE_KV_METADATA)
    fn.argtypes = [
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
        ctypes.c_void_p(append_positions_i32_ptr),
        ctypes.c_void_p(cache_positions_i32_ptr),
        ctypes.c_void_p(live_count_i32_ptr),
        ctypes.c_int64(start),
        ctypes.c_int64(count),
        ctypes.c_int64(end),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_gqa_attention_f32_bf16(
    query_f32_ptr: int,
    key_f32_ptr: int,
    value_bf16_ptr: int,
    out_bf16_ptr: int,
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    scale: float | None = None,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch slow-but-deterministic non-causal DFlash GQA attention.

    Inputs are row-major ``query[batch, query_len, q_heads, head_dim]``,
    ``key/value[batch, kv_len, kv_heads, head_dim]``. The output is BF16 bits in
    ``out[batch, query_len, q_heads, head_dim]``. This correctness-first kernel
    is intended for the native drafter root/query harness, not final throughput.
    """

    _check_attention_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, threads)
    scale_value = float(head_dim ** -0.5 if scale is None else scale)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GQA_ATTENTION)
    fn.argtypes = [
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
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_f32_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_bf16_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(query_len),
        ctypes.c_int64(kv_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale_value),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash_gqa_attention_f32_bf16_bucketed(
    query_f32_ptr: int,
    key_f32_ptr: int,
    value_bf16_ptr: int,
    out_bf16_ptr: int,
    live_context_len_i32_ptr: int,
    batch_size: int,
    query_len: int,
    kv_len: int,
    bucket_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    scale: float | None = None,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the bucketed DFlash GQA attention kernel.

    The KV layout reserves ``bucket_context_len`` rows for context (only the first
    ``*live_context_len_i32_ptr`` are valid; trailing rows are masked) followed
    by ``kv_len - bucket_context_len`` query rows.  ``live_context_len_i32_ptr``
    is a device-resident int32 scalar so the same captured HIP graph can replay
    across cycles with different live context lengths.

    Mathematically bit-equivalent to :func:`dflash_gqa_attention_f32_bf16` when
    ``*live_context_len_i32_ptr == bucket_context_len`` (no rows masked).
    """

    _check_attention_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, threads)
    if bucket_context_len < 0 or bucket_context_len > kv_len:
        raise ValueError("bucket_context_len must be in [0, kv_len]")
    scale_value = float(head_dim ** -0.5 if scale is None else scale)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GQA_ATTENTION_BUCKETED)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_f32_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_bf16_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_void_p(live_context_len_i32_ptr),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(query_len),
        ctypes.c_int64(kv_len),
        ctypes.c_int64(bucket_context_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale_value),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dflash_drafter_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dflash_prepare_noise_inputs", "w4_paro", "bf16_i32"),
        dflash_prepare_noise_inputs_bf16_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_prepare_noise_inputs", "w4_paro", "f16_to_bf16_i32"),
        dflash_prepare_noise_inputs_f16_to_bf16_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_add", "w4_paro", "bf16"),
        dflash_add_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_concat_rows", "w4_paro", "f32"),
        dflash_concat_rows_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_concat_rows", "w4_paro", "bf16"),
        dflash_concat_rows_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_rmsnorm", "w4_paro", "bf16"),
        dflash_rmsnorm_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_add_rmsnorm", "w4_paro", "bf16"),
        dflash_add_rmsnorm_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_silu_mul", "w4_paro", "bf16"),
        dflash_silu_mul_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_silu_mul_routes", "w4_paro", "bf16"),
        dflash_silu_mul_gate_up_routes_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_bf16"),
        dflash_dense_bf16_to_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_bf16_expert_routes"),
        dflash_dense_bf16_to_bf16_expert_routes,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_f32"),
        dflash_dense_bf16_to_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_bf16_wmma"),
        dflash_dense_bf16_to_bf16_wmma,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_dense", "w4_paro", "bf16_to_f32_wmma"),
        dflash_dense_bf16_to_f32_wmma,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_qkv_proj", "w4_paro", "bf16_mixed"),
        dflash_qkv_proj_bf16_mixed,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_qkv_proj", "w4_paro", "bf16_mixed_indexed_v"),
        dflash_qkv_proj_bf16_mixed_indexed_v,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_head_rmsnorm_rotary", "w4_paro", "f32_bf16"),
        dflash_head_rmsnorm_rotary_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_head_rmsnorm_rotary", "w4_paro", "f32_bf16_indexed_key"),
        dflash_head_rmsnorm_rotary_indexed_key_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_key_rmsnorm_rotary", "w4_paro", "f32_bf16"),
        dflash_key_rmsnorm_rotary_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_update_kv_metadata", "w4_paro", "i32"),
        dflash_update_kv_metadata_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_gqa_attention", "w4_paro", "f32_bf16"),
        dflash_gqa_attention_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash_gqa_attention", "w4_paro", "f32_bf16_bucketed"),
        dflash_gqa_attention_f32_bf16_bucketed,
        replace=replace,
    )


def _launch_prepare_noise_inputs(
    symbol: str,
    root_tokens_i32_ptr: int,
    root_positions_i32_ptr: int,
    embed_tokens_ptr: int,
    noise_token_ids_i32_ptr: int,
    position_ids_i32_ptr: int,
    noise_embeddings_bf16_ptr: int,
    request_count: int,
    block_size: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_shape(request_count, block_size, hidden_size, vocab_size, threads)
    if mask_token_id < 0 or mask_token_id >= vocab_size:
        raise ValueError("mask_token_id must be within vocab_size")
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(root_tokens_i32_ptr),
        ctypes.c_void_p(root_positions_i32_ptr),
        ctypes.c_void_p(embed_tokens_ptr),
        ctypes.c_void_p(noise_token_ids_i32_ptr),
        ctypes.c_void_p(position_ids_i32_ptr),
        ctypes.c_void_p(noise_embeddings_bf16_ptr),
        ctypes.c_int64(request_count),
        ctypes.c_int64(block_size),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int32(mask_token_id),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dense(
    symbol: str,
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_dense_shape(rows, in_features, out_features, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_dense_wmma_shape(rows: int, in_features: int, out_features: int) -> None:
    for name, value in (("rows", rows), ("in_features", in_features), ("out_features", out_features)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if (in_features % _DFLASH_WMMA_TILE_K) != 0:
        raise ValueError(
            "in_features must be a multiple of 16 for the WMMA dense kernel; "
            f"got in_features={in_features}"
        )


def _launch_dense_wmma(
    symbol: str,
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_dense_wmma_shape(rows, in_features, out_features)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_qkv_proj(
    symbol: str,
    x_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    v_weight_ptr: int,
    q_out_ptr: int,
    k_out_ptr: int,
    v_out_ptr: int,
    rows: int,
    in_features: int,
    q_features: int,
    kv_features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_qkv_projection_shape(rows, in_features, q_features, kv_features, threads)
    library = library or build_dflash_drafter(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(q_weight_ptr),
        ctypes.c_void_p(k_weight_ptr),
        ctypes.c_void_p(v_weight_ptr),
        ctypes.c_void_p(q_out_ptr),
        ctypes.c_void_p(k_out_ptr),
        ctypes.c_void_p(v_out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(q_features),
        ctypes.c_int64(kv_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_qkv_proj_indexed_v(
    symbol: str,
    x_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    v_weight_ptr: int,
    q_out_ptr: int,
    k_out_ptr: int,
    value_cache_base_ptr: int,
    cache_slot_i32_ptr: int,
    cache_rows: int,
    rows: int,
    in_features: int,
    q_features: int,
    kv_features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_qkv_projection_shape(rows, in_features, q_features, kv_features, threads)
    if cache_rows <= 0:
        raise ValueError("cache_rows must be positive")
    library = library or build_dflash_drafter(load=True)
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
        ctypes.c_void_p(q_weight_ptr),
        ctypes.c_void_p(k_weight_ptr),
        ctypes.c_void_p(v_weight_ptr),
        ctypes.c_void_p(q_out_ptr),
        ctypes.c_void_p(k_out_ptr),
        ctypes.c_void_p(value_cache_base_ptr),
        ctypes.c_void_p(cache_slot_i32_ptr),
        ctypes.c_int64(cache_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(q_features),
        ctypes.c_int64(kv_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_simple(
    symbol: str,
    args: tuple[int, ...],
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(args[0]),
        ctypes.c_void_p(args[1]),
        ctypes.c_void_p(args[2]),
        ctypes.c_int64(args[3]),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_concat(
    symbol: str,
    context_ptr: int,
    query_ptr: int,
    out_ptr: int,
    batch_size: int,
    context_len: int,
    query_len: int,
    features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_concat_shape(batch_size, context_len, query_len, features, threads)
    library = library or build_dflash_drafter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(context_ptr),
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(batch_size),
        ctypes.c_int64(context_len),
        ctypes.c_int64(query_len),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_elements(elements: int, threads: int) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_concat_shape(batch_size: int, context_len: int, query_len: int, features: int, threads: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_len < 0:
        raise ValueError("context_len must be non-negative")
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    if features <= 0:
        raise ValueError("features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_rmsnorm_shape(rows: int, hidden_size: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_dense_shape(rows: int, in_features: int, out_features: int, threads: int) -> None:
    for name, value in (("rows", rows), ("in_features", in_features), ("out_features", out_features)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_qkv_projection_shape(rows: int, in_features: int, q_features: int, kv_features: int, threads: int) -> None:
    for name, value in (("rows", rows), ("in_features", in_features), ("q_features", q_features), ("kv_features", kv_features)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_head_rotary_shape(
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    threads: int,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("query_len", query_len),
        ("kv_len", kv_len),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("rotary_dim", rotary_dim),
        ("max_positions", max_positions),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_key_rotary_shape(rows: int, num_kv_heads: int, head_dim: int, rotary_dim: int, max_positions: int, threads: int) -> None:
    for name, value in (
        ("rows", rows),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("rotary_dim", rotary_dim),
        ("max_positions", max_positions),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and no larger than head_dim")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_attention_shape(
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    threads: int,
) -> None:
    for name, value in (
        ("batch_size", batch_size),
        ("query_len", query_len),
        ("kv_len", kv_len),
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_shape(request_count: int, block_size: int, hidden_size: int, vocab_size: int, threads: int) -> None:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


register_dflash_drafter_kernels()
