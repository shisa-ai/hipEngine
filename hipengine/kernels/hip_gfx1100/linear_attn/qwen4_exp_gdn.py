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


def register_qwen4_exp_gdn_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrence_norm_gate",
            "f32_state",
            "qwen4exp_sigmoid_strict",
        ),
        qwen4_exp_gdn_decode_f32,
        replace=replace,
    )


register_qwen4_exp_gdn_kernels()


__all__ = [
    "build_qwen4_exp_gdn",
    "plan_qwen4_exp_gdn_build",
    "qwen4_exp_gdn_decode_f32",
    "register_qwen4_exp_gdn_kernels",
]
