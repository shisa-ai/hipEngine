"""gfx1100 indexed-singleton GDN decode route tests.

The 2026-09-05 gfx1100 RDNA3 audit (docs/20260905-gfx1100-audit.md, packet A)
transfers the gfx1151 indexed-singleton GDN decode policy after independent
W7900 qualification. These tests pin the route contract: the backend capability
selects the registered one-token-per-active-row indexed singleton kernel for
packed decode, the arbitrary-length segmented recurrence stays registered as
the strict fallback, and the packed decode manifest accepts the
``indexed_singleton`` GDN path.
"""

from __future__ import annotations

import ctypes

import pytest

from hipengine.kernels.hip_gfx1100.linear_attn import (
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state,
)
from hipengine.kernels.registry import resolve
from hipengine.runtime.gguf_packed_manifest import build_packed_decode_execution_manifest
from hipengine.runtime.qwen35_gguf_runner import (
    _resolve_gguf_linear_attention_decode_batch_plan,
)
from hipengine.kernels.backends import backend_package_capability


def test_gfx1100_gdn_indexed_singleton_capability_is_enabled() -> None:
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_GDN_INDEXED_SINGLETON_DECODE",
            False,
        )
        is True
    )


def test_gfx1100_decode_batch_plan_selects_indexed_singleton() -> None:
    plan = _resolve_gguf_linear_attention_decode_batch_plan("hip_gfx1100")
    assert plan.gdn_decode_path == "indexed_singleton"
    assert (
        plan.gdn_indexed_singleton
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16
    )
    # Strict arbitrary-length fallback stays registered on the same plan.
    assert plan.gdn_segments is qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16


def test_gfx1100_fp16_state_plan_keeps_indexed_singleton_sibling() -> None:
    fp16_plan = _resolve_gguf_linear_attention_decode_batch_plan(
        "hip_gfx1100",
        use_fp16_state=True,
    )
    assert fp16_plan.gdn_decode_path == "indexed_singleton"
    assert (
        fp16_plan.gdn_indexed_singleton
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state
    )
    assert (
        fp16_plan.gdn_segments
        is qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state
    )


def test_gfx1100_registry_resolves_indexed_singleton_variants() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrent_rmsnorm_gate",
            quant="gguf_qwen35",
            variant="bf16_indexed_singleton",
        )
        is qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16
    )


def test_packed_manifest_accepts_indexed_singleton_gdn_path() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=8,
        layer_types=("linear_attention",) * 30 + ("full_attention",) * 10,
        imported_slot_indices=(),
        import_positions=(513,) * 8,
        scatter_state=False,
        blocks_per_slot=4,
        full_attention_decode_path="kv_live_spans_batch",
        moe_decode_path="selected_rows_batch",
        moe_top_k=8,
        lm_head_decode_path="q6_rowtile_f32_logits",
        sampler_decode_path="argmax_i32_rows",
        metadata_prepare_path="host_upload",
        linear_attention_decode_path="indexed_batch",
        gdn_recurrent_decode_path="indexed_singleton",
    )
    conv_gdn = manifest["layer_families"]["conv_gdn"]
    assert conv_gdn["execution"] == "packed_native"
    assert conv_gdn["packed_native_work"] == [
        "conv_decode_indexed",
        "gdn_recurrent_decode_indexed_fp32_out",
    ]
    assert conv_gdn["exact_row_local_kernel_launches"] == 0


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_gfx1100_indexed_singleton_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 2, 8, 4
        )
