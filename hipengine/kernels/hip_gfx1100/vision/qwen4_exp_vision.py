"""Raw-pointer Qwen4Exp vision helper wrappers."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name('qwen4_exp_vision.hip')
_OUTPUT_NAME = 'qwen4_exp_vision.so'


def plan_qwen4_exp_vision_build(*, cache_root=None, compiler_version=None, profile: ProfileName='prefill') -> BuildArtifact:
    return plan_hip_build(sources=[_SOURCE], family='qwen4_exp_vision', profile=profile, cache_root=cache_root, compiler_version=compiler_version, output_name=_OUTPUT_NAME)


def build_qwen4_exp_vision(*, cache_root=None, compiler_version=None, profile: ProfileName='prefill', dry_run=False, load=True, require_cached=False):
    return build_hip(sources=[_SOURCE], family='qwen4_exp_vision', profile=profile, cache_root=cache_root, compiler_version=compiler_version, output_name=_OUTPUT_NAME, dry_run=dry_run, load=load, require_cached=require_cached)


def _launch(symbol, argtypes, args, *, library, runtime):
    lib = library or build_qwen4_exp_vision(load=True)
    rt = runtime or get_hip_runtime()
    fn = signed_kernel_fn(lib, symbol, argtypes, ctypes.c_int)
    error = fn(*args)
    if int(error) != HIP_SUCCESS:
        rt.check(int(error))


def qwen4_exp_vision_layernorm_f32(input_ptr, weight_ptr, bias_ptr, output_ptr, rows, features, eps=1e-6, *, stream=0, library=None, runtime=None):
    if rows <= 0 or features <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid vision layernorm shape/epsilon')
    _launch('hipengine_qwen4_exp_vision_layernorm_f32', (ctypes.c_void_p,)*4 + (ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p), (input_ptr, weight_ptr, bias_ptr, output_ptr, rows, features, eps, stream), library=library, runtime=runtime)


def qwen4_exp_vision_add_bias_residual_f32(input_ptr, bias_ptr, residual_ptr, output_ptr, rows, features, *, stream=0, library=None, runtime=None):
    if rows <= 0 or features <= 0:
        raise ValueError('invalid vision bias/residual shape')
    _launch('hipengine_qwen4_exp_vision_add_bias_residual_f32', (ctypes.c_void_p,)*4 + (ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p), (input_ptr, bias_ptr, residual_ptr, output_ptr, rows, features, stream), library=library, runtime=runtime)


def qwen4_exp_vision_bias_gelu_tanh_f32(input_ptr, bias_ptr, output_ptr, rows, features, *, stream=0, library=None, runtime=None):
    if rows <= 0 or features <= 0:
        raise ValueError('invalid vision GELU shape')
    _launch('hipengine_qwen4_exp_vision_bias_gelu_tanh_f32', (ctypes.c_void_p,)*3 + (ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p), (input_ptr, bias_ptr, output_ptr, rows, features, stream), library=library, runtime=runtime)


def qwen4_exp_vision_attention_f32(qkv_ptr, qkv_bias_ptr, pos_h_ptr, pos_w_ptr, output_ptr, tokens, heads=16, head_dim=72, *, scale=None, stream=0, library=None, runtime=None):
    value = head_dim ** -0.5 if scale is None else float(scale)
    if tokens <= 0 or heads <= 0 or head_dim != 72 or value <= 0:
        raise ValueError('invalid Qwen4Exp vision attention shape')
    _launch('hipengine_qwen4_exp_vision_attention_f32', (ctypes.c_void_p,)*5 + (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p), (qkv_ptr, qkv_bias_ptr, pos_h_ptr, pos_w_ptr, output_ptr, tokens, heads, head_dim, value, stream), library=library, runtime=runtime)


def register_qwen4_exp_vision_kernels(*, replace=True):
    rows = {
        KernelKey('hip_gfx1100','vision_layernorm','f32','qwen3vl'): qwen4_exp_vision_layernorm_f32,
        KernelKey('hip_gfx1100','vision_add_bias_residual','f32','qwen3vl'): qwen4_exp_vision_add_bias_residual_f32,
        KernelKey('hip_gfx1100','vision_gelu','f32','qwen3vl_tanh'): qwen4_exp_vision_bias_gelu_tanh_f32,
        KernelKey('hip_gfx1100','vision_attention','f32','qwen3vl_rope'): qwen4_exp_vision_attention_f32,
    }
    for key, fn in rows.items(): register(key, fn, replace=replace)

register_qwen4_exp_vision_kernels()

__all__=['build_qwen4_exp_vision','plan_qwen4_exp_vision_build','qwen4_exp_vision_layernorm_f32','qwen4_exp_vision_add_bias_residual_f32','qwen4_exp_vision_bias_gelu_tanh_f32','qwen4_exp_vision_attention_f32','register_qwen4_exp_vision_kernels']
