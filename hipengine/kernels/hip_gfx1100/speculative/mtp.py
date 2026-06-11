"""Raw-pointer wrappers for native MTP proposal helper kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("mtp.hip")
_OUTPUT_NAME = "mtp_speculative.so"
_SYMBOL_FUSE_INPUTS = "hipengine_mtp_fuse_inputs_f16_bf16"
_SYMBOL_RMSNORM_ONEPLUS = "hipengine_mtp_rmsnorm_bf16_oneplus"
_SYMBOL_ADD_RMSNORM_ONEPLUS = "hipengine_mtp_add_rmsnorm_bf16_oneplus"
_SYMBOL_SPLIT_Q_GATE = "hipengine_mtp_split_q_gate_f32_bf16"
_SYMBOL_GATE_MUL_BF16 = "hipengine_mtp_gate_mul_bf16"
_SYMBOL_SOFTMAX_TOPK = "hipengine_mtp_softmax_topk_f32"
_SYMBOL_ROUTER_TOPK_SOFTMAX = "hipengine_mtp_router_topk_softmax_256x8_f32"
_SYMBOL_ACCUM_ROUTE = "hipengine_mtp_accumulate_route_bf16_to_f32"
_SYMBOL_ACCUM_ROUTES = "hipengine_mtp_accumulate_routes_bf16_to_f32"
_SYMBOL_ACCUM_SIGMOID_GATE = "hipengine_mtp_accumulate_sigmoid_gate_bf16_to_f32"
_SYMBOL_FINALIZE_F32_TO_BF16 = "hipengine_mtp_finalize_f32_to_bf16"
_ALLOWED_THREADS = {64, 128, 256}
_MAX_TOPK = 8
_PTR_ARG_COUNTS = {
    _SYMBOL_RMSNORM_ONEPLUS: 3,
    _SYMBOL_ADD_RMSNORM_ONEPLUS: 5,
    _SYMBOL_SPLIT_Q_GATE: 3,
    _SYMBOL_GATE_MUL_BF16: 3,
    _SYMBOL_SOFTMAX_TOPK: 2,
    _SYMBOL_ROUTER_TOPK_SOFTMAX: 4,
    _SYMBOL_ACCUM_ROUTE: 3,
    _SYMBOL_ACCUM_ROUTES: 3,
    _SYMBOL_ACCUM_SIGMOID_GATE: 3,
    _SYMBOL_FINALIZE_F32_TO_BF16: 2,
}


def plan_mtp_speculative_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="mtp_speculative",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_mtp_speculative(
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
        family="mtp_speculative",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def mtp_fuse_inputs_f16_bf16(
    token_ids_i64_ptr: int,
    embedding_f16_ptr: int,
    target_hidden_bf16_ptr: int,
    embed_norm_weight_bf16_ptr: int,
    hidden_norm_weight_bf16_ptr: int,
    out_concat_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Normalize token embeddings and target hidden rows, then concatenate.

    Output layout is ``[rows, 2 * hidden_size]`` BF16 with the normalized token
    embedding in the first half and normalized target hidden in the second half.
    This is the input expected by ``mtp.fc.weight``.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {sorted(_ALLOWED_THREADS)}")
    library = library or build_mtp_speculative(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_FUSE_INPUTS)
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
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(embedding_f16_ptr),
        ctypes.c_void_p(target_hidden_bf16_ptr),
        ctypes.c_void_p(embed_norm_weight_bf16_ptr),
        ctypes.c_void_p(hidden_norm_weight_bf16_ptr),
        ctypes.c_void_p(out_concat_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_float(float(eps)),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{_SYMBOL_FUSE_INPUTS} failed with HIP error {err}")


def mtp_rmsnorm_bf16_oneplus(
    x_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply Qwen3.5/Qwen3.6 zero-centered RMSNorm as ``x * (1 + weight)``."""

    _check_rows_hidden(rows, hidden_size, threads)
    _launch(_SYMBOL_RMSNORM_ONEPLUS, [x_bf16_ptr, weight_bf16_ptr, out_bf16_ptr, rows, hidden_size, float(eps), threads, stream], library, runtime)


def mtp_add_rmsnorm_bf16_oneplus(
    x_bf16_ptr: int,
    residual_bf16_ptr: int,
    weight_bf16_ptr: int,
    out_bf16_ptr: int,
    residual_out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-6,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """BF16 residual add followed by Qwen zero-centered RMSNorm."""

    _check_rows_hidden(rows, hidden_size, threads)
    _launch(
        _SYMBOL_ADD_RMSNORM_ONEPLUS,
        [x_bf16_ptr, residual_bf16_ptr, weight_bf16_ptr, out_bf16_ptr, residual_out_bf16_ptr, rows, hidden_size, float(eps), threads, stream],
        library,
        runtime,
    )


def mtp_split_q_gate_f32_bf16(
    q_proj_f32_ptr: int,
    query_out_f32_ptr: int,
    gate_out_bf16_ptr: int,
    rows: int,
    num_q_heads: int,
    head_dim: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split MTP q_proj layout ``[heads, 2 * head_dim]`` into query and gate."""

    if rows <= 0 or num_q_heads <= 0 or head_dim <= 0:
        raise ValueError("rows, num_q_heads, and head_dim must be positive")
    _check_threads(threads)
    _launch(_SYMBOL_SPLIT_Q_GATE, [q_proj_f32_ptr, query_out_f32_ptr, gate_out_bf16_ptr, rows, num_q_heads, head_dim, threads, stream], library, runtime)


def mtp_gate_mul_bf16(
    attn_bf16_ptr: int,
    gate_bf16_ptr: int,
    out_bf16_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply sigmoid gate to BF16 attention output and write BF16."""

    _check_elements(elements, threads)
    _launch(_SYMBOL_GATE_MUL_BF16, [attn_bf16_ptr, gate_bf16_ptr, out_bf16_ptr, elements, threads, stream], library, runtime)


def mtp_softmax_topk_f32(
    values_f32_ptr: int,
    routing_f32_ptr: int,
    rows: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Softmax row-wise top-k logits produced by the router top-k kernel."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if top_k <= 0 or top_k > _MAX_TOPK:
        raise ValueError(f"top_k must be in [1, {_MAX_TOPK}]")
    _launch(_SYMBOL_SOFTMAX_TOPK, [values_f32_ptr, routing_f32_ptr, rows, top_k, stream], library, runtime)


def mtp_router_topk_softmax_f32(
    logits_f32_ptr: int,
    out_values_f32_ptr: int,
    out_indices_i32_ptr: int,
    routing_f32_ptr: int,
    num_experts: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused proposer router top-k + softmax for the Qwen MTP 256-expert/top-8 shape."""

    if num_experts != 256:
        raise ValueError("num_experts must be 256 for fused MTP proposer router top-k")
    if top_k != 8:
        raise ValueError("top_k must be 8 for fused MTP proposer router top-k")
    _launch(
        _SYMBOL_ROUTER_TOPK_SOFTMAX,
        [logits_f32_ptr, out_values_f32_ptr, out_indices_i32_ptr, routing_f32_ptr, num_experts, top_k, stream],
        library,
        runtime,
    )


def mtp_accumulate_route_bf16_to_f32(
    src_bf16_ptr: int,
    routing_f32_ptr: int,
    accum_f32_ptr: int,
    elements: int,
    route_index: int,
    *,
    reset_output: bool = False,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Accumulate ``routing[route_index] * src`` into an FP32 MoE accumulator.

    When ``reset_output`` is true, the kernel writes the scaled route directly
    instead of reading the previous accumulator value.  The MTP proposer uses
    this for route 0 to remove a standalone memset launch.
    """

    _check_elements(elements, threads)
    if route_index < 0:
        raise ValueError("route_index must be non-negative")
    _launch(
        _SYMBOL_ACCUM_ROUTE,
        [src_bf16_ptr, routing_f32_ptr, accum_f32_ptr, elements, route_index, int(bool(reset_output)), threads, stream],
        library,
        runtime,
    )


def mtp_accumulate_routes_bf16_to_f32(
    src_routes_bf16_ptr: int,
    routing_f32_ptr: int,
    accum_f32_ptr: int,
    routes: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Accumulate route-major BF16 route outputs into one FP32 row.

    The device loop visits routes in ascending order, matching the scalar
    ``mtp_accumulate_route_bf16_to_f32`` loop used by the persistent proposer.
    """

    if routes <= 0:
        raise ValueError("routes must be positive")
    _check_elements(elements, threads)
    _launch(
        _SYMBOL_ACCUM_ROUTES,
        [src_routes_bf16_ptr, routing_f32_ptr, accum_f32_ptr, routes, elements, threads, stream],
        library,
        runtime,
    )


def mtp_accumulate_sigmoid_gate_bf16_to_f32(
    src_bf16_ptr: int,
    gate_f32_ptr: int,
    accum_f32_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Accumulate ``sigmoid(gate[0]) * src`` into an FP32 MoE accumulator."""

    _check_elements(elements, threads)
    _launch(_SYMBOL_ACCUM_SIGMOID_GATE, [src_bf16_ptr, gate_f32_ptr, accum_f32_ptr, elements, threads, stream], library, runtime)


def mtp_finalize_f32_to_bf16(
    src_f32_ptr: int,
    out_bf16_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Round an FP32 row accumulator to BF16."""

    _check_elements(elements, threads)
    _launch(_SYMBOL_FINALIZE_F32_TO_BF16, [src_f32_ptr, out_bf16_ptr, elements, threads, stream], library, runtime)


def register_mtp_speculative_kernels(*, replace: bool = True) -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "mtp_fuse_inputs", "bf16", "f16_embed_bf16_hidden"),
            mtp_fuse_inputs_f16_bf16,
            replace=replace,
        )
        register(KernelKey(backend, "mtp_rmsnorm", "bf16", "oneplus"), mtp_rmsnorm_bf16_oneplus, replace=replace)
        register(KernelKey(backend, "mtp_add_rmsnorm", "bf16", "oneplus"), mtp_add_rmsnorm_bf16_oneplus, replace=replace)
        register(KernelKey(backend, "mtp_split_q_gate", "f32", "bf16_gate"), mtp_split_q_gate_f32_bf16, replace=replace)
        register(KernelKey(backend, "mtp_gate_mul", "bf16", "bf16"), mtp_gate_mul_bf16, replace=replace)
        register(KernelKey(backend, "mtp_softmax_topk", "f32", "rows"), mtp_softmax_topk_f32, replace=replace)
        register(KernelKey(backend, "mtp_router_topk_softmax", "f32", "256x8"), mtp_router_topk_softmax_f32, replace=replace)
        register(KernelKey(backend, "mtp_accumulate_route", "bf16", "f32"), mtp_accumulate_route_bf16_to_f32, replace=replace)
        register(KernelKey(backend, "mtp_accumulate_routes", "bf16", "f32"), mtp_accumulate_routes_bf16_to_f32, replace=replace)
        register(KernelKey(backend, "mtp_accumulate_sigmoid_gate", "bf16", "f32"), mtp_accumulate_sigmoid_gate_bf16_to_f32, replace=replace)
        register(KernelKey(backend, "mtp_finalize", "f32", "bf16"), mtp_finalize_f32_to_bf16, replace=replace)


register_mtp_speculative_kernels()


def _launch(symbol: str, args: list[int | float], library: ctypes.CDLL | None, runtime: HipRuntime | None) -> None:
    library = library or build_mtp_speculative(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    pointer_args = _PTR_ARG_COUNTS[symbol]
    argtypes = []
    converted = []
    for idx, value in enumerate(args):
        if idx == len(args) - 1:
            # Last argument is always the hipStream_t.
            argtypes.append(ctypes.c_void_p)
            converted.append(ctypes.c_void_p(int(value)))
        elif idx < pointer_args:
            argtypes.append(ctypes.c_void_p)
            converted.append(ctypes.c_void_p(int(value)))
        elif isinstance(value, float):
            argtypes.append(ctypes.c_float)
            converted.append(ctypes.c_float(value))
        else:
            argtypes.append(ctypes.c_int64)
            converted.append(ctypes.c_int64(int(value)))
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    err = fn(*converted)
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{symbol} failed with HIP error {err}")


def _check_rows_hidden(rows: int, hidden_size: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    _check_threads(threads)


def _check_elements(elements: int, threads: int) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    _check_threads(threads)


def _check_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {sorted(_ALLOWED_THREADS)}")


__all__ = [
    "build_mtp_speculative",
    "mtp_accumulate_route_bf16_to_f32",
    "mtp_accumulate_routes_bf16_to_f32",
    "mtp_accumulate_sigmoid_gate_bf16_to_f32",
    "mtp_add_rmsnorm_bf16_oneplus",
    "mtp_finalize_f32_to_bf16",
    "mtp_fuse_inputs_f16_bf16",
    "mtp_gate_mul_bf16",
    "mtp_rmsnorm_bf16_oneplus",
    "mtp_router_topk_softmax_f32",
    "mtp_softmax_topk_f32",
    "mtp_split_q_gate_f32_bf16",
    "plan_mtp_speculative_build",
    "register_mtp_speculative_kernels",
]
