"""CPU-reference backend.

Importing this package self-registers the first NumPy reference kernels. Tests that clear the
kernel registry can call ``register_cpu_reference_kernels()`` to restore them.
"""

from hipengine.kernels.cpu_reference.fixtures import (
    LayerCheckResult,
    LayerFixture,
    Tolerances,
    load_fixture,
    run_fixture,
    save_fixture,
)
from hipengine.kernels.cpu_reference.ops import (
    attention_decode,
    dequantize_kv_int8_per_token_head,
    embed,
    full_attn_prefill,
    full_attn_prefill_varlen,
    gdn_prefill_recurrent_segments,
    linear,
    kv_dequant_int8_per_token_head,
    linear_attn_conv_prefill_segments,
    lm_head,
    o_proj,
    paged_attn_decode_int8_per_token_head,
    qkv_proj,
    quantize_kv_int8_per_token_head,
    register_cpu_reference_kernels,
    rmsnorm,
    rotate,
    write_paged_kv_int8_per_token_head,
)

register_cpu_reference_kernels()

__all__ = [
    "LayerCheckResult",
    "LayerFixture",
    "Tolerances",
    "attention_decode",
    "dequantize_kv_int8_per_token_head",
    "embed",
    "full_attn_prefill",
    "full_attn_prefill_varlen",
    "gdn_prefill_recurrent_segments",
    "kv_dequant_int8_per_token_head",
    "linear",
    "linear_attn_conv_prefill_segments",
    "lm_head",
    "load_fixture",
    "o_proj",
    "paged_attn_decode_int8_per_token_head",
    "qkv_proj",
    "quantize_kv_int8_per_token_head",
    "register_cpu_reference_kernels",
    "rmsnorm",
    "rotate",
    "run_fixture",
    "save_fixture",
    "write_paged_kv_int8_per_token_head",
]
