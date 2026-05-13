"""gfx1100 attention and KV-cache kernel wrappers."""

from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    build_qwen35_paged_attn_decode,
    plan_qwen35_paged_attn_decode_build,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_f32_spans,
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    build_qwen35_paged_kv_write,
    plan_qwen35_paged_kv_write_build,
    qwen35_write_paged_kv_f32_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
    register_qwen35_paged_kv_write_kernels,
)

__all__ = [
    "build_qwen35_paged_attn_decode",
    "build_qwen35_paged_kv_write",
    "plan_qwen35_paged_attn_decode_build",
    "plan_qwen35_paged_kv_write_build",
    "qwen35_paged_full_attn_decode_context_bf16_spans",
    "qwen35_paged_full_attn_decode_split_k_bf16_spans",
    "qwen35_paged_full_attn_decode_split_k_gate_bf16_spans",
    "qwen35_paged_full_attn_decode_split_k_gate_f32_spans",
    "qwen35_write_paged_kv_f32_spans",
    "qwen35_write_paged_kv_mixed_value_bf16_spans",
    "register_qwen35_paged_attn_decode_kernels",
    "register_qwen35_paged_kv_write_kernels",
]
