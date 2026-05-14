"""Raw-pointer wrappers for Qwen3.5 linear-attention GDN kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gdn.hip")
_OUTPUT_NAME = "qwen35_linear_attn_gdn.so"
_SYMBOL_LOWP = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16"
_SYMBOL_PREFILL = "hipengine_qwen35_gdn_prefill_recurrent_f32"
_SYMBOL_PREFILL_K2 = "hipengine_qwen35_gdn_prefill_recurrent_k2_f32"


def plan_qwen35_linear_attn_gdn_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_linear_attn_gdn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_linear_attn_gdn(
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
        family="qwen35_linear_attn_gdn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    eps: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-gated recurrent GDN RMSNorm+gate kernel."""

    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LOWP)
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
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_f32(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch native FP32 GDN recurrent prefill kernel."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL,
        query_ptr,
        key_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        recurrent_state_ptr,
        out_ptr,
        tokens,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_recurrent_k2_f32(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch native FP32 GDN recurrent prefill K2 kernel."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL_K2,
        query_ptr,
        key_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        recurrent_state_ptr,
        out_ptr,
        tokens,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_qwen35_linear_attn_gdn_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "gdn_recurrent_rmsnorm_gate", "w4_paro", "bf16_lowp"),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32"),
        qwen35_gdn_prefill_recurrent_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32_k2"),
        qwen35_gdn_prefill_recurrent_k2_f32,
        replace=replace,
    )


def _launch_prefill_recurrent(
    symbol: str,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(num_v_heads, "num_v_heads")
    _check_positive(head_k_dim, "head_k_dim")
    _check_positive(head_v_dim, "head_v_dim")
    if head_k_dim != 128:
        raise ValueError("head_k_dim must be 128 for native prefill GDN")
    library = library or build_qwen35_linear_attn_gdn(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_gdn_shape(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    _check_positive(num_k_heads, "num_k_heads")
    _check_positive(num_v_heads, "num_v_heads")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    _check_positive(head_k_dim, "head_k_dim")
    _check_positive(head_v_dim, "head_v_dim")
    if head_v_dim > 128:
        raise ValueError("head_v_dim must be <= 128")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_linear_attn_gdn_kernels()
