"""Raw-pointer strict Qwen4Exp GDN decode wrapper."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_gdn.hip")
_OUTPUT_NAME = "qwen4_exp_gdn.so"
_ARGS = (ctypes.c_void_p,) * 9 + (ctypes.c_int64,) * 4 + (ctypes.c_void_p,)
_PREFILL_ARGS = (ctypes.c_void_p,) * 9 + (ctypes.c_int64,) * 5 + (ctypes.c_void_p,)
_PREPARE_ARGS = (ctypes.c_void_p,) * 10 + (ctypes.c_int64,) * 5 + (ctypes.c_void_p,)
_GATE_ARGS = (ctypes.c_void_p,) * 3 + (ctypes.c_int64,) * 3 + (ctypes.c_void_p,)
_STATE_LAYOUT_ARGS = (ctypes.c_void_p,) * 2 + (ctypes.c_int64,) * 3 + (ctypes.c_int32, ctypes.c_void_p)


def plan_qwen4_exp_gdn_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_gdn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_gdn(
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
        family="qwen4_exp_gdn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_gdn_decode_f32(
    conv_ptr: int,
    output_gate_ptr: int,
    alpha_ptr: int,
    beta_logits_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    output_ptr: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run one strict FP32-state Qwen4Exp decode recurrence with sigmoid gate."""

    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by positive num_k_heads")
    if head_k_dim <= 0 or head_v_dim <= 0:
        raise ValueError("head dimensions must be positive")
    library = library or build_qwen4_exp_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gdn_decode_f32",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        conv_ptr,
        output_gate_ptr,
        alpha_ptr,
        beta_logits_ptr,
        dt_bias_ptr,
        a_ptr,
        norm_weight_ptr,
        recurrent_state_ptr,
        output_ptr,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _qwen4_exp_gdn_state_transpose_f32(
    source_ptr: int, destination_ptr: int, heads: int, key_dim: int,
    value_dim: int, *, strict_to_transposed: bool, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    library = library or build_qwen4_exp_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_qwen4_exp_gdn_state_transpose_f32", _STATE_LAYOUT_ARGS, ctypes.c_int)
    error = fn(source_ptr, destination_ptr, heads, key_dim, value_dim, int(strict_to_transposed), stream)
    if int(error) != HIP_SUCCESS: runtime.check(int(error))


def qwen4_exp_gdn_state_strict_to_transposed_f32(*args, **kwargs) -> None:
    kwargs["strict_to_transposed"] = True
    _qwen4_exp_gdn_state_transpose_f32(*args, **kwargs)


def qwen4_exp_gdn_state_transposed_to_strict_f32(*args, **kwargs) -> None:
    kwargs["strict_to_transposed"] = False
    _qwen4_exp_gdn_state_transpose_f32(*args, **kwargs)


def qwen4_exp_gdn_prefill_prepare_f32(
    conv_ptr: int,
    alpha_ptr: int,
    beta_logits_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Prepare exact Qwen4Exp normalized Q/K and recurrent scalars."""

    if tokens <= 0 or num_k_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("invalid Qwen4Exp GDN prepare geometry")
    library = library or build_qwen4_exp_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gdn_prefill_prepare_f32",
        _PREPARE_ARGS,
        ctypes.c_int,
    )
    error = fn(
        conv_ptr, alpha_ptr, beta_logits_ptr, dt_bias_ptr, a_ptr,
        query_ptr, key_ptr, value_ptr, beta_ptr, decay_ptr,
        tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_gdn_prefill_sigmoid_gate_f32(
    core_ptr: int,
    output_gate_ptr: int,
    norm_weight_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply Qwen4Exp sigmoid output gate and RMSNorm in place."""

    if tokens <= 0 or num_v_heads <= 0 or head_v_dim <= 0:
        raise ValueError("invalid Qwen4Exp GDN gate geometry")
    library = library or build_qwen4_exp_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gdn_prefill_sigmoid_gate_f32",
        _GATE_ARGS,
        ctypes.c_int,
    )
    error = fn(
        core_ptr, output_gate_ptr, norm_weight_ptr,
        tokens, num_v_heads, head_v_dim, stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_gdn_peer_prefill_f32(
    conv_ptr: int,
    output_gate_ptr: int,
    alpha_ptr: int,
    beta_logits_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    output_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the compact peer-wave32 Qwen4Exp GDN prefill chain."""

    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
        qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32,
    )

    runtime = runtime or get_hip_runtime()
    qwen4_exp_gdn_prefill_prepare_f32(
        conv_ptr,
        alpha_ptr,
        beta_logits_ptr,
        dt_bias_ptr,
        a_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        beta_logits_ptr,
        alpha_ptr,
        tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        runtime=runtime,
    )
    qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32(
        query_ptr,
        key_ptr,
        value_ptr,
        beta_logits_ptr,
        alpha_ptr,
        recurrent_state_ptr,
        output_ptr,
        tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        runtime=runtime,
    )
    qwen4_exp_gdn_prefill_sigmoid_gate_f32(
        output_ptr,
        output_gate_ptr,
        norm_weight_ptr,
        tokens,
        num_v_heads,
        head_v_dim,
        stream=stream,
        runtime=runtime,
    )


def qwen4_exp_gdn_prefill_columnwarps_f32(
    conv_ptr: int,
    output_gate_ptr: int,
    alpha_ptr: int,
    beta_logits_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    core_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the column-warp (llama-layout) FP32-state GDN recurrence."""

    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by positive num_k_heads")
    if head_k_dim != 128 or head_v_dim != 128:
        raise ValueError("column-warp GDN prefill requires 128x128 heads")
    library = library or build_qwen4_exp_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    args = (ctypes.c_void_p,) * 9 + (ctypes.c_int64,) * 5 + (ctypes.c_void_p,)
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gdn_prefill_columnwarps_f32",
        args,
        ctypes.c_int,
    )
    error = fn(
        conv_ptr,
        output_gate_ptr,
        alpha_ptr,
        beta_logits_ptr,
        dt_bias_ptr,
        a_ptr,
        norm_weight_ptr,
        recurrent_state_ptr,
        core_ptr,
        tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def qwen4_exp_gdn_prefill_f32(
    conv_ptr: int,
    output_gate_ptr: int,
    alpha_ptr: int,
    beta_logits_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    output_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact serial-order FP32-state Qwen4Exp recurrence for token rows."""

    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by positive num_k_heads")
    if head_k_dim <= 0 or head_v_dim <= 0:
        raise ValueError("head dimensions must be positive")
    library = library or build_qwen4_exp_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gdn_prefill_f32",
        _PREFILL_ARGS,
        ctypes.c_int,
    )
    error = fn(
        conv_ptr,
        output_gate_ptr,
        alpha_ptr,
        beta_logits_ptr,
        dt_bias_ptr,
        a_ptr,
        norm_weight_ptr,
        recurrent_state_ptr,
        output_ptr,
        tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def register_qwen4_exp_gdn_kernels(*, replace: bool = True) -> None:
    registrations = {
        "qwen4exp_sigmoid_strict": qwen4_exp_gdn_decode_f32,
        "qwen4exp_sigmoid_strict_prefill": qwen4_exp_gdn_prefill_f32,
        "qwen4exp_sigmoid_peer_prefill": qwen4_exp_gdn_peer_prefill_f32,
        "qwen4exp_gdn_columnwarps_prefill": qwen4_exp_gdn_prefill_columnwarps_f32,
    }
    for variant, function in registrations.items():
        register(
            KernelKey(
                "hip_gfx1100",
                "gdn_recurrence_norm_gate",
                "f32_state",
                variant,
            ),
            function,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "gdn_state_layout", "f32_state", "strict_to_transposed"),
        qwen4_exp_gdn_state_strict_to_transposed_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_state_layout", "f32_state", "transposed_to_strict"),
        qwen4_exp_gdn_state_transposed_to_strict_f32,
        replace=replace,
    )


register_qwen4_exp_gdn_kernels()


__all__ = [
    "qwen4_exp_gdn_prefill_columnwarps_f32",
    "build_qwen4_exp_gdn",
    "plan_qwen4_exp_gdn_build",
    "qwen4_exp_gdn_decode_f32",
    "qwen4_exp_gdn_state_strict_to_transposed_f32",
    "qwen4_exp_gdn_state_transposed_to_strict_f32",
    "qwen4_exp_gdn_peer_prefill_f32",
    "qwen4_exp_gdn_prefill_f32",
    "qwen4_exp_gdn_prefill_prepare_f32",
    "qwen4_exp_gdn_prefill_sigmoid_gate_f32",
    "register_qwen4_exp_gdn_kernels",
]
