"""Bind exact TimesFM raw-pointer kernel keys before device allocation."""

from types import SimpleNamespace

from hipengine.kernels.backends import load_backend_kernel_package, resolve_backend
from hipengine.kernels.registry import KernelKey, MissingKernelError, is_registered, resolve

TIMESFM_DUAL = (
    "timesfm_rmsnorm_f32", "timesfm_norm_add_f32", "timesfm_bias_f32",
    "timesfm_bias_swish_f32", "timesfm_swish_f32", "timesfm_add_f32",
    "timesfm_rope_f32", "timesfm_head_rmsnorm_f32", "timesfm_head_perdim_scale_f32",
    "timesfm_scatter_kv_f32", "timesfm_attention_f32",
)
TIMESFM_F16 = (
    "timesfm_flash_attention_f16", "timesfm_q_norm_transpose_f16",
    "timesfm_k_norm_scatter_f16", "timesfm_v_scatter_f16",
    "timesfm_mask_softmax_f16", "timesfm_transpose_heads_f16",
    "timesfm_rope_norm_scatter_f16", "timesfm_qkv_norm_scatter_f16",
)
TIMESFM3_DUAL = ("timesfm3_relu", "timesfm3_var_attention")
TIMESFM3_F16 = ("timesfm3_qkv_norm_scatter_f16",)
TIMESFM3_F32 = ("timesfm3_sgemm_f64acc",)


def timesfm_key(backend: str, name: str, precision: str) -> KernelKey:
    return KernelKey(backend, name.removesuffix("_f32").removesuffix("_f16"), precision)


def bind_timesfm_kernels(backend: str, precision: str, *, multivariate: bool = False):
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be 'fp16' or 'fp32'")
    backend = resolve_backend(backend, execution_device=True)
    package = load_backend_kernel_package(backend)
    # Family hook restores isolated registries without replacing caller overrides.
    registrar = getattr(package, "register_timesfm_kernels", None)
    if registrar is not None:
        registrar()
    names = TIMESFM_DUAL + (TIMESFM_F16 if precision == "fp16" else ())
    if multivariate:
        names += TIMESFM3_DUAL + (TIMESFM3_F16 if precision == "fp16" else TIMESFM3_F32)
    kernels = {}
    for name in names:
        key = timesfm_key(backend, name, precision)
        # Generic registry fallback may change dtype or pointer ABI. Neither is
        # legal for these resident device buffers; require the exact key.
        if not is_registered(key):
            raise MissingKernelError(key, (key,))
        kernels[name] = resolve(backend=key.backend, layer=key.layer, quant=key.quant)
    return backend, SimpleNamespace(**kernels)
