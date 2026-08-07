"""Raw-pointer wrappers for Maple's official MLX ternary/affine4 storage."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("maple_ternary.hip")
_OUTPUT_NAME = "maple_ternary.so"
_QUANT = "maple_ternary2"

_PTR = ctypes.c_void_p
_I64 = ctypes.c_int64


def plan_maple_ternary_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="maple_ternary",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_maple_ternary(
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
        family="maple_ternary",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def maple_ternary_gemv_bf16(
    x_ptr: int,
    weight_ptr: int,
    row_alpha_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_ternary_gemv_bf16",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (x_ptr, weight_ptr, row_alpha_ptr, out_ptr, in_features, out_features),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_ternary_gemm_bf16(
    x_ptr: int,
    weight_ptr: int,
    row_alpha_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Batched [rows, out] = [rows, in] x ternary W (P1 prefill GEMM)."""

    _launch(
        "hipengine_maple_ternary_gemm_bf16",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _I64, _PTR),
        (x_ptr, weight_ptr, row_alpha_ptr, out_ptr, rows, in_features, out_features),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_ternary_qkv_gemm_bf16(
    x_ptr: int,
    q_weight_ptr: int,
    q_alpha_ptr: int,
    k_weight_ptr: int,
    k_alpha_ptr: int,
    v_weight_ptr: int,
    v_alpha_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    q_features: int,
    kv_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Batched QKV ternary GEMM: [rows, q+2kv] qkv buffer (P1 prefill)."""

    _launch(
        "hipengine_maple_ternary_qkv_gemm_bf16",
        (_PTR,) * 8 + (_I64, _I64, _I64, _I64, _PTR),
        (
            x_ptr,
            q_weight_ptr,
            q_alpha_ptr,
            k_weight_ptr,
            k_alpha_ptr,
            v_weight_ptr,
            v_alpha_ptr,
            out_ptr,
            rows,
            in_features,
            q_features,
            kv_features,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_ternary_qkv_gemv_bf16(
    x_ptr: int,
    q_weight_ptr: int,
    q_alpha_ptr: int,
    k_weight_ptr: int,
    k_alpha_ptr: int,
    v_weight_ptr: int,
    v_alpha_ptr: int,
    out_ptr: int,
    in_features: int,
    q_features: int,
    kv_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_ternary_qkv_gemv_bf16",
        (_PTR,) * 8 + (_I64, _I64, _I64, _PTR),
        (
            x_ptr,
            q_weight_ptr,
            q_alpha_ptr,
            k_weight_ptr,
            k_alpha_ptr,
            v_weight_ptr,
            v_alpha_ptr,
            out_ptr,
            in_features,
            q_features,
            kv_features,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_selected_ternary_dual_gemv_bf16(
    x_ptr: int,
    a_weight_ptr: int,
    a_alpha_ptr: int,
    b_weight_ptr: int,
    b_alpha_ptr: int,
    selected_experts_ptr: int,
    a_out_ptr: int,
    b_out_ptr: int,
    num_experts: int,
    top_k: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_selected_ternary_dual_gemv_bf16",
        (_PTR,) * 8 + (_I64, _I64, _I64, _I64, _PTR),
        (
            x_ptr,
            a_weight_ptr,
            a_alpha_ptr,
            b_weight_ptr,
            b_alpha_ptr,
            selected_experts_ptr,
            a_out_ptr,
            b_out_ptr,
            num_experts,
            top_k,
            in_features,
            out_features,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_selected_ternary_gemv_bf16(
    x_ptr: int,
    weight_ptr: int,
    row_alpha_ptr: int,
    selected_experts_ptr: int,
    out_ptr: int,
    num_experts: int,
    top_k: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_selected_ternary_gemv_bf16",
        (_PTR,) * 5 + (_I64, _I64, _I64, _I64, _PTR),
        (
            x_ptr,
            weight_ptr,
            row_alpha_ptr,
            selected_experts_ptr,
            out_ptr,
            num_experts,
            top_k,
            in_features,
            out_features,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_affine4_embed_bf16(
    weight_ptr: int,
    scales_ptr: int,
    biases_ptr: int,
    out_ptr: int,
    token_id: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_affine4_embed_bf16",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (weight_ptr, scales_ptr, biases_ptr, out_ptr, token_id, hidden_size),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_affine4_embed_batched_bf16(
    weight_ptr: int,
    scales_ptr: int,
    biases_ptr: int,
    token_ids_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Batched affine4 embed of T token IDs into [T, hidden] (P4 prefill)."""

    _launch(
        "hipengine_maple_affine4_embed_batched_bf16",
        (_PTR, _PTR, _PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (weight_ptr, scales_ptr, biases_ptr, token_ids_ptr, out_ptr, rows, hidden_size),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_affine4_gemv_f32(
    x_ptr: int,
    weight_ptr: int,
    scales_ptr: int,
    biases_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_affine4_gemv_f32",
        (_PTR, _PTR, _PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (x_ptr, weight_ptr, scales_ptr, biases_ptr, out_ptr, in_features, out_features),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_maple_ternary_kernels(
    *,
    backend: str = "hip_gfx1100",
    replace: bool = True,
) -> None:
    kernels = {
        ("maple_ternary_gemv", "row_alpha"): maple_ternary_gemv_bf16,
        ("maple_ternary_gemm", "row_alpha"): maple_ternary_gemm_bf16,
        ("maple_ternary_qkv", "fused_split_weights"): maple_ternary_qkv_gemv_bf16,
        ("maple_ternary_qkv", "fused_split_weights_gemm"): maple_ternary_qkv_gemm_bf16,
        (
            "maple_selected_ternary_dual",
            "row_alpha",
        ): maple_selected_ternary_dual_gemv_bf16,
        ("maple_selected_ternary", "row_alpha"): maple_selected_ternary_gemv_bf16,
        ("maple_affine4_embed", "group64"): maple_affine4_embed_bf16,
        ("maple_affine4_embed", "group64_batched"): maple_affine4_embed_batched_bf16,
        ("maple_affine4_gemv", "group64"): maple_affine4_gemv_f32,
    }
    for (layer, variant), kernel in kernels.items():
        register(
            KernelKey(backend, layer, _QUANT, variant),
            kernel,
            replace=replace,
        )


def _launch(
    symbol: str,
    argtypes: tuple,
    args: tuple[int, ...],
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_maple_ternary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*args, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_maple_ternary_kernels()


__all__ = [
    "build_maple_ternary",
    "maple_affine4_embed_batched_bf16",
    "maple_affine4_embed_bf16",
    "maple_affine4_gemv_f32",
    "maple_selected_ternary_dual_gemv_bf16",
    "maple_selected_ternary_gemv_bf16",
    "maple_ternary_gemm_bf16",
    "maple_ternary_gemv_bf16",
    "maple_ternary_qkv_gemm_bf16",
    "maple_ternary_qkv_gemv_bf16",
    "plan_maple_ternary_build",
    "register_maple_ternary_kernels",
]
