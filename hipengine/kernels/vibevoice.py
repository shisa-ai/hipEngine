"""Resolve VibeVoice kernels once through the four-axis registry."""
from types import SimpleNamespace
from hipengine.kernels.backends import load_backend_kernel_package,resolve_backend
from hipengine.kernels.registry import resolve,KernelKey,is_registered,MissingKernelError

PRIMITIVES = (
    'build_vibevoice_encoder','vv_add_bias_bf16','vv_add_bias_f32','vv_rmsnorm_bf16',
    'vv_depthwise_accumulate_f32','vv_depthwise_residual_bf16',
    'vv_gelu_bf16','vv_scale_residual_bf16','vv_depthwise_conv_bf16','vv_conv_gemm_bf16',
    'vv_im2col_bf16','vv_add_scaled_noise_bf16','vv_rope_positions_f32',
    'vv_kv_write_spans','vv_attention_spans','bf16_to_f32','bf16_to_fp16','f32_to_bf16','f32_to_fp16',
    'silu_mul_dual_out_bf16','silu_mul_separate_out_bf16','dense_dual_gemv_out_bf16',
    'dense_gemv_bf16_f32_out','dense_gemv_f32_bf16w_f32_out','dense_gemv_out_bf16',
    'qwen35_partial_rotary_f32',
)


def resolve_vibevoice_kernel(backend,layer,variant='strict'):
    key=KernelKey(backend,layer,'bf16',variant)
    if not is_registered(key):
        raise MissingKernelError(key,(key,))
    return resolve(backend=backend,layer=layer,quant='bf16',variant=variant)


def resolve_vibevoice_kernels(backend='auto',*,frontend_variant='wmma'):
    backend=resolve_backend(backend)
    package=load_backend_kernel_package(backend)
    package.register_vibevoice_kernels()
    ops={name:resolve_vibevoice_kernel(backend,name) for name in PRIMITIVES}
    if frontend_variant != 'strict':
        ops['vv_depthwise_conv_bf16']=resolve_vibevoice_kernel(backend,'vv_depthwise_conv_bf16','fused')
    ops['frontend_gemm']=resolve_vibevoice_kernel(backend,'vibevoice_frontend_gemm',frontend_variant)
    return SimpleNamespace(backend=backend,**ops)

DECODER_PRIMITIVES = ('bf16_to_fp16', 'build_vibevoice_encoder', 'dense_dual_gemv_out_bf16', 'dense_gemv_bf16_f32_out', 'dense_gemv_f32_bf16w_f32_out', 'dense_gemv_out_bf16', 'f32_to_bf16', 'f32_to_fp16', 'qwen35_partial_rotary_f32', 'silu_mul_dual_out_bf16', 'silu_mul_separate_out_bf16', 'vv_add_bias_f32', 'vv_attention_spans', 'vv_kv_write_spans', 'vv_rmsnorm_bf16', 'vv_rope_positions_f32', 'vv_scale_residual_bf16')

FRONTEND_PRIMITIVES = ('build_vibevoice_encoder', 'vv_add_bias_bf16', 'vv_add_scaled_noise_bf16', 'vv_conv_gemm_bf16', 'vv_depthwise_conv_bf16', 'vv_gelu_bf16', 'vv_im2col_bf16', 'vv_rmsnorm_bf16', 'vv_scale_residual_bf16')
