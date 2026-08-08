from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

import hipengine.kernels.hip_gfx1100.attention.paged_attn_decode as paged_attn_decode
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference import attention_decode
from hipengine.kernels.hip_gfx1100.attention import (
    plan_qwen35_paged_attn_decode_build,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans,
    qwen35_paged_attn_decode_int8_gqa_splitk_spans,
    qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans,
    qwen35_full_attn_decode_context_bf16,
    qwen35_full_attn_gate_mul_bf16,
    qwen35_full_attn_gate_mul_fp16,
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_q3_c1_exact_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_f32_spans,
    qwen35_paged_full_attn_decode_split_k_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_bf16_decode_order_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans,
    qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans,
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.loading.materialize import float_array_to_bf16_bits


def setup_function() -> None:
    clear_registry_for_tests()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans(*, storage_dtype: str = "bf16", live_dtype: str = "int64") -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (1,), "int32"),
        live_counts=_tensor(0x2000, (1,), live_dtype),
        max_live_count=2,
        storage_dtype=storage_dtype,
    )


def _int8_spans(
    *,
    block_table_shape: tuple[int, ...] = (1,),
    scale_shape: tuple[int, ...] = (1, 256, 2),
    live_dtype: str = "int64",
    scale_dtype: str = "fp16",
) -> KVLiveSpans:
    metadata = KVScaleMetadata(
        k_scale=_tensor(0x3000, scale_shape, scale_dtype),
        v_scale=_tensor(0x4000, scale_shape, scale_dtype),
        scale_dtype=scale_dtype,
    )
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, block_table_shape, "int32"),
        live_counts=_tensor(0x2000, (1,), live_dtype),
        max_live_count=2,
        storage_dtype="int8_per_token_head",
        scale_metadata=metadata,
    )


def test_qwen35_paged_attn_decode_registers_span_variant() -> None:
    register_qwen35_paged_attn_decode_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_gate_mul",
            quant="w4_paro",
            variant="bf16",
        )
        is qwen35_full_attn_gate_mul_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_gate_mul",
            quant="w4_paro",
            variant="fp16",
        )
        is qwen35_full_attn_gate_mul_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_decode",
            quant="w4_paro",
            variant="bf16_context",
        )
        is qwen35_full_attn_decode_context_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_batch_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_batch_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_batch_c1_exact_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_batch_paged_c1_exact_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="gguf_ud_q3_k_m",
            variant="bf16_context_batch_native_exact_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_batch_q3_c1_exact_spans
    )
    shared_native = getattr(
        paged_attn_decode,
        "qwen35_paged_full_attn_decode_context_bf16_batch_shared_native_exact_spans",
        None,
    )
    assert shared_native is not None
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="gguf_q4_k_m",
            variant="bf16_context_batch_shared_native_exact_spans",
        )
        is shared_native
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gate_f32_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gate_f32_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_warp_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_warp_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_warp_gate_bf16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_warp_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_bf16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_bf16_batch_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_bf16_parallel_reduce_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_fp16_batch_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_fp16_batch_direct_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gate_bf16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="gqa_splitk_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="gqa_splitk_gate_bf16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="gqa_splitk_gate_fp16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_splitk_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_splitk_gate_bf16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_splitk_gate_fp16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="qwen35_causal_gqa_gate_fp16",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="gguf_ud_q3_k_m",
            variant="causal_gqa_gate_bf16",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_bf16_decode_order_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_prefill",
            quant="w4_paro",
            variant="bf16_gqa_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_prefill",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_gate_fp16_spans",
        )
        is qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="qwen35_tree_gqa_gate_fp16",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="qwen35_varlen_causal_gqa_gate_fp16",
        )
        is qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans
    )


def test_qwen35_decode_order_prefill_uses_query_batch_gqa_crossover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def record(name: str):
        def launch(*args, **kwargs):
            calls.append((name, args, kwargs))

        return launch

    monkeypatch.setattr(paged_attn_decode, "_launch_prefill_gqa_gate", record("dense"))
    monkeypatch.setattr(
        paged_attn_decode,
        "qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_batch_spans",
        record("warp"),
    )
    monkeypatch.setattr(
        paged_attn_decode,
        "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans",
        record("gqa"),
    )

    def spans(rows: int, blocks: int) -> KVLiveSpans:
        return KVLiveSpans.paged_uniform(
            block_table=_tensor(0x1000, (rows, blocks), "int32"),
            live_counts=_tensor(0x2000, (rows,), "int64"),
            max_live_count=rows,
            storage_dtype="bf16",
            row_positions=_tensor(0x3000, (rows,), "int64"),
            span_role="prefill",
        )

    common = {
        "query_ptr": 0x100000,
        "key_cache_ptr": 0x200000,
        "value_cache_ptr": 0x300000,
        "gate_ptr": 0x400000,
        "out_ptr": 0x500000,
        "split_partial_out_ptr": 0x600000,
        "split_partial_m_ptr": 0x700000,
        "split_partial_l_ptr": 0x800000,
        "split_batch_rows": 16,
        "block_size": 256,
        "num_q_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 256,
        "gate_stride1": 256,
        "gate_stride2": 1,
        "scale": 0.0625,
        "library": object(),
        "runtime": object(),
    }

    qwen35_paged_full_attn_prefill_gqa_gate_bf16_decode_order_spans(
        spans=spans(1, 4),
        rows=1,
        max_context_len=1024,
        split_count=4,
        **common,
    )
    assert [name for name, _, _ in calls] == ["warp"]
    assert calls[0][1][9:12] == (1, 256, 4)

    calls.clear()
    qwen35_paged_full_attn_prefill_gqa_gate_bf16_decode_order_spans(
        spans=spans(5, 5),
        rows=5,
        max_context_len=1026,
        split_count=5,
        **common,
    )
    assert [name for name, _, _ in calls] == ["dense", "gqa"]
    dense_args = calls[0][1]
    assert dense_args[7:9] == (2, 1023)
    assert dense_args[6].base_offsets.shape == (2, 5)
    gqa_args = calls[1][1]
    assert gqa_args[0] == common["query_ptr"] + 2 * 16 * 256 * 4
    assert gqa_args[3] == common["gate_ptr"] + 2 * 16 * 256 * 2
    assert gqa_args[8].base_offsets.shape == (3, 5)
    assert gqa_args[8].base_offsets.ptr == 0x1000 + 2 * 5 * 4
    assert gqa_args[9:12] == (3, 256, 5)

    calls.clear()
    qwen35_paged_full_attn_prefill_gqa_gate_bf16_decode_order_spans(
        spans=spans(4, 17),
        rows=4,
        max_context_len=4097,
        split_count=17,
        **common,
    )
    assert [name for name, _, _ in calls] == ["gqa"]
    gqa_args = calls[0][1]
    assert gqa_args[8].base_offsets.shape == (4, 17)
    assert gqa_args[9:12] == (4, 256, 17)


def test_int8_gqa_splitk_producer_grid_is_owned_by_kv_head() -> None:
    """Freeze the grouped-GQA load-sharing geometry, not only its output math."""

    source = Path(paged_attn_decode.__file__).with_suffix(".hip").read_text()
    producer = source.split(
        "__global__ void "
        "qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel(",
        1,
    )[1].split("\ntemplate <", 1)[0]
    launcher = source.split(
        "int launch_qwen35_paged_full_attn_decode_split_k_gqa_context_int8(",
        1,
    )[1].split("\ntemplate <typename scale_t>", 1)[0]

    assert "const int64_t kv_head = blockIdx.x;" in producer
    assert "const int64_t q_base = kv_head * q_per_kv;" in producer
    assert "token_offset * num_kv_heads + kv_head" in producer
    assert producer.count("h < q_per_kv") >= 10
    assert "const int64_t q_head = blockIdx.x;" not in producer

    assert "const int64_t q_per_kv = 8;" in launcher
    assert "<scale_t, 8, 16, 2>" in launcher
    assert (
        "dim3(static_cast<unsigned int>(num_kv_heads), "
        "static_cast<unsigned int>(num_splits))"
    ) in launcher
    assert "dim3(static_cast<unsigned int>(num_q_heads)" not in launcher


def test_qwen35_paged_attn_decode_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_paged_attn_decode_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 paged attn decode test version",
    )

    assert artifact.family == "qwen35_paged_attn_decode"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_paged_attn_decode.so"
    assert artifact.compiler_version == "hipcc qwen35 paged attn decode test version"
    assert any(str(path).endswith("paged_attn_decode.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_paged_attn_decode_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="total must be positive"):
        qwen35_full_attn_gate_mul_bf16(0, 0, 0, 0)
    with pytest.raises(ValueError, match="total must be positive"):
        qwen35_full_attn_gate_mul_fp16(0, 0, 0, 0)
    with pytest.raises(ValueError, match="max_context_len must be positive"):
        qwen35_full_attn_decode_context_bf16(0, 0, 0, 0, 0, 0, 2, 1, 4, 1.0)
    with pytest.raises(ValueError, match="num_q_heads must be divisible"):
        qwen35_full_attn_decode_context_bf16(0, 0, 0, 0, 0, 2, 3, 2, 4, 1.0)
    with pytest.raises(ValueError, match="block_size=256"):
        qwen35_paged_full_attn_decode_context_bf16_spans(0, 0, 0, 0, _spans(), 2, 4, 2, 1, 4, 1.0)
    with pytest.raises(ValueError, match="int64 live_counts"):
        qwen35_paged_full_attn_decode_context_bf16_spans(
            0, 0, 0, 0, _spans(live_dtype="int32"), 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="bf16 storage"):
        qwen35_paged_full_attn_decode_context_bf16_spans(
            0, 0, 0, 0, _spans(storage_dtype="fp16"), 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="num_q_heads must be divisible"):
        qwen35_paged_full_attn_decode_context_bf16_spans(0, 0, 0, 0, _spans(), 2, 256, 3, 2, 4, 1.0)
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            0, 0, 0, 0, _spans(), 0, 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="live_counts"):
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            0,
            0,
            0,
            0,
            KVLiveSpans.paged_uniform(
                block_table=_tensor(0x1000, (2,), "int32"),
                live_counts=_tensor(0x2000, (1,), "int64"),
                max_live_count=2,
                storage_dtype="bf16",
            ),
            2,
            2,
            256,
            2,
            1,
            4,
            1.0,
        )
    with pytest.raises(ValueError, match="head_dim divisible by 8"):
        qwen35_paged_full_attn_decode_split_k_bf16_spans(
            0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 1, 2, 256, 2, 1, 8, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0x4000, 0, 0, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="tree_committed_count"):
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0x4000, -1, 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="ancestor_mask_ptr"):
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0, 0, 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="segments"):
        qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0, 0, 1, 0, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_decode_split_k_gate_f32_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 2, 1, 8, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_decode_split_k_gate_fp16_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 2, 1, 8, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="Qwen3.5 GQA"):
        qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(
            0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 8, 1, 256, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 16, 2, 256, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 0, 2, 2, 256, 16, 2, 256, 256, 1, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 0, 2, 2, 256, 16, 2, 256, 256, 1, 1.0
        )
    with pytest.raises(ValueError, match="Qwen3.5 GQA"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 1, 2, 2, 256, 8, 1, 256, 256, 1, 1.0
        )
    with pytest.raises(ValueError, match="num_splits=1"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans(
            0, 0, 0, 0, 0, _spans(), 1, 2, 2, 256, 16, 2, 256, 256, 1, 1.0
        )
    int8_spans = _int8_spans()
    with pytest.raises(ValueError, match="int8_per_token_head storage"):
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, _spans(), 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="int64 live_counts"):
        bad_live = _int8_spans(live_dtype="int32")
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, bad_live.scale_metadata.k_scale.ptr, bad_live.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, bad_live, 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="Qwen3.5 INT8 GQA"):
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, int8_spans, 256, 1, 256, 8, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="k_scale_ptr"):
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr + 1, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, int8_spans, 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="scale tensor shape"):
        bad_scale_shape = _int8_spans(scale_shape=(1, 128, 2))
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, bad_scale_shape.scale_metadata.k_scale.ptr, bad_scale_shape.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, bad_scale_shape, 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, 0, int8_spans, 256, 1, 256, 16, 2, 256, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="int8_per_token_head storage"):
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, _spans(), 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="head_dim <= 256"):
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, int8_spans, 1, 2, 256, 2, 1, 512, 512, 1, 1.0
        )
    with pytest.raises(ValueError, match="k_scale_ptr"):
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr + 1, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, int8_spans, 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.parametrize(
    "batch_kernel",
    (
        qwen35_paged_full_attn_decode_context_bf16_batch_spans,
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
    ),
    ids=("adaptive", "fixed256"),
)
def test_qwen35_paged_attn_decode_batch_honors_shared_physical_blocks(batch_kernel) -> None:
    """Batch row block tables contain physical block IDs, not row-local IDs."""

    runtime = get_hip_runtime()
    rows = 2
    block_size = 256
    blocks = 2
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 8
    max_context_len = 5
    scale = head_dim ** -0.5
    rng = np.random.default_rng(20260629)

    query = rng.normal(0.0, 0.25, size=(rows, num_q_heads, head_dim)).astype(np.float32)
    key_f32 = rng.normal(0.0, 0.25, size=(blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
    value_f32 = rng.normal(0.0, 0.25, size=(blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
    key_f32[1] += 2.0
    value_f32[1] -= 2.0
    key_cache = float_array_to_bf16_bits(key_f32)
    value_cache = float_array_to_bf16_bits(value_f32)
    block_table = np.zeros((rows, 1), dtype=np.int32)
    live_counts = np.asarray([3, 5], dtype=np.int64)
    batch_out = np.empty((rows, num_q_heads, head_dim), dtype=np.float32)
    c1_out = np.empty_like(batch_out)

    buffers = []
    try:
        query_b = malloc(query.nbytes, runtime=runtime)
        key_b = malloc(key_cache.nbytes, runtime=runtime)
        value_b = malloc(value_cache.nbytes, runtime=runtime)
        table_b = malloc(block_table.nbytes, runtime=runtime)
        live_b = malloc(live_counts.nbytes, runtime=runtime)
        batch_out_b = malloc(batch_out.nbytes, runtime=runtime)
        c1_out_b = malloc(c1_out.nbytes, runtime=runtime)
        buffers.extend((query_b, key_b, value_b, table_b, live_b, batch_out_b, c1_out_b))
        copy_host_to_device(query_b, host_array_ptr(query), query.nbytes, runtime=runtime)
        copy_host_to_device(key_b, host_array_ptr(key_cache), key_cache.nbytes, runtime=runtime)
        copy_host_to_device(value_b, host_array_ptr(value_cache), value_cache.nbytes, runtime=runtime)
        copy_host_to_device(table_b, host_array_ptr(block_table), block_table.nbytes, runtime=runtime)
        copy_host_to_device(live_b, host_array_ptr(live_counts), live_counts.nbytes, runtime=runtime)

        batch_spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table_b.ptr, block_table.shape, "int32"),
            live_counts=_tensor(live_b.ptr, live_counts.shape, "int64"),
            max_live_count=max_context_len,
            storage_dtype=DType.BF16,
        )
        batch_kernel(
            query_b.ptr,
            key_b.ptr,
            value_b.ptr,
            batch_out_b.ptr,
            batch_spans,
            rows,
            max_context_len,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            runtime=runtime,
        )

        row_query_nbytes = num_q_heads * head_dim * DType.FP32.itemsize
        row_out_nbytes = row_query_nbytes
        row_table_nbytes = DType.INT32.itemsize
        row_live_nbytes = DType.INT64.itemsize
        for row in range(rows):
            row_spans = KVLiveSpans.paged_uniform(
                block_table=_tensor(table_b.ptr + row * row_table_nbytes, (1,), "int32"),
                live_counts=_tensor(live_b.ptr + row * row_live_nbytes, (1,), "int64"),
                max_live_count=int(live_counts[row]),
                storage_dtype=DType.BF16,
            )
            qwen35_paged_full_attn_decode_context_bf16_spans(
                query_b.ptr + row * row_query_nbytes,
                key_b.ptr,
                value_b.ptr,
                c1_out_b.ptr + row * row_out_nbytes,
                row_spans,
                int(live_counts[row]),
                block_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                runtime=runtime,
            )

        copy_device_to_host(host_array_ptr(batch_out), batch_out_b, batch_out.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(c1_out), c1_out_b, c1_out.nbytes, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(batch_out, c1_out)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_qwen35_paged_attn_decode_batch_shared_table_matches_scalar_and_cpu() -> None:
    batch_kernel = getattr(
        paged_attn_decode,
        "qwen35_paged_full_attn_decode_context_bf16_batch_shared_native_exact_spans",
        None,
    )
    assert batch_kernel is not None

    runtime = get_hip_runtime()
    rows = 4
    block_size = 256
    blocks = 2
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    live_counts = np.asarray([254, 255, 256, 257], dtype=np.int64)
    max_context_len = int(live_counts[-1])
    scale = head_dim ** -0.5
    rng = np.random.default_rng(20260804)

    query = rng.normal(0.0, 0.125, size=(rows, num_q_heads, head_dim)).astype(np.float32)
    key_f32 = rng.normal(
        0.0,
        0.125,
        size=(blocks, block_size, num_kv_heads, head_dim),
    ).astype(np.float32)
    value_f32 = rng.normal(
        0.0,
        0.125,
        size=(blocks, block_size, num_kv_heads, head_dim),
    ).astype(np.float32)
    key_f32[1] += np.float32(1.0)
    value_f32[1] -= np.float32(1.0)
    key_cache = float_array_to_bf16_bits(key_f32)
    value_cache = float_array_to_bf16_bits(value_f32)
    block_table = np.asarray([1, 0], dtype=np.int32)
    batch_out = np.empty_like(query)
    scalar_out = np.empty_like(query)

    buffers = []
    try:
        query_b = malloc(query.nbytes, runtime=runtime)
        key_b = malloc(key_cache.nbytes, runtime=runtime)
        value_b = malloc(value_cache.nbytes, runtime=runtime)
        table_b = malloc(block_table.nbytes, runtime=runtime)
        live_b = malloc(live_counts.nbytes, runtime=runtime)
        batch_out_b = malloc(batch_out.nbytes, runtime=runtime)
        scalar_out_b = malloc(scalar_out.nbytes, runtime=runtime)
        buffers.extend(
            (query_b, key_b, value_b, table_b, live_b, batch_out_b, scalar_out_b)
        )
        for buffer, host in (
            (query_b, query),
            (key_b, key_cache),
            (value_b, value_cache),
            (table_b, block_table),
            (live_b, live_counts),
        ):
            copy_host_to_device(buffer, host_array_ptr(host), host.nbytes, runtime=runtime)

        shared_spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table_b.ptr, block_table.shape, "int32"),
            live_counts=_tensor(live_b.ptr, live_counts.shape, "int64"),
            max_live_count=max_context_len,
            storage_dtype=DType.BF16,
            span_role="verify_chain",
        )
        batch_kernel(
            query_b.ptr,
            key_b.ptr,
            value_b.ptr,
            batch_out_b.ptr,
            shared_spans,
            rows,
            max_context_len,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            runtime=runtime,
        )

        row_nbytes = num_q_heads * head_dim * DType.FP32.itemsize
        for row in range(rows):
            row_spans = KVLiveSpans.paged_uniform(
                block_table=_tensor(table_b.ptr, block_table.shape, "int32"),
                live_counts=_tensor(
                    live_b.ptr + row * DType.INT64.itemsize,
                    (1,),
                    "int64",
                ),
                max_live_count=max_context_len,
                storage_dtype=DType.BF16,
                span_role="verify_chain",
            )
            qwen35_paged_full_attn_decode_context_bf16_spans(
                query_b.ptr + row * row_nbytes,
                key_b.ptr,
                value_b.ptr,
                scalar_out_b.ptr + row * row_nbytes,
                row_spans,
                max_context_len,
                block_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                runtime=runtime,
            )

        copy_device_to_host(
            host_array_ptr(batch_out), batch_out_b, batch_out.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(scalar_out), scalar_out_b, scalar_out.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(batch_out, scalar_out)

    key_rounded = (key_cache.astype(np.uint32) << 16).view(np.float32)
    value_rounded = (value_cache.astype(np.uint32) << 16).view(np.float32)
    logical_key = np.concatenate([key_rounded[index] for index in block_table], axis=0)
    logical_value = np.concatenate([value_rounded[index] for index in block_table], axis=0)
    q_per_kv = num_q_heads // num_kv_heads
    cpu_out = np.empty_like(batch_out)
    for row, live_count in enumerate(live_counts):
        key_heads = np.repeat(
            np.transpose(logical_key[:live_count], (1, 0, 2)),
            q_per_kv,
            axis=0,
        )
        value_heads = np.repeat(
            np.transpose(logical_value[:live_count], (1, 0, 2)),
            q_per_kv,
            axis=0,
        )
        cpu_out[row] = attention_decode(
            query[row, :, None, :],
            key_heads,
            value_heads,
            scale=scale,
        )[:, 0, :]
    np.testing.assert_allclose(batch_out, cpu_out, rtol=5e-4, atol=2e-5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.parametrize(
    ("batch_kernel", "bit_exact"),
    (
        (qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans, True),
        (qwen35_paged_full_attn_decode_context_bf16_batch_q3_c1_exact_spans, False),
    ),
    ids=("main", "q3-retained"),
)
def test_qwen35_paged_attn_decode_batch_c1_exact_matches_c1_model_shape(
    batch_kernel,
    bit_exact: bool,
) -> None:
    runtime = get_hip_runtime()
    rows = 2
    block_size = 256
    blocks_per_row = 3
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    max_context_len = 513
    scale = head_dim ** -0.5
    rng = np.random.default_rng(20260630)

    query = rng.normal(0.0, 0.25, size=(rows, num_q_heads, head_dim)).astype(np.float32)
    cache_shape = (rows * blocks_per_row, block_size, num_kv_heads, head_dim)
    key_f32 = rng.normal(0.0, 0.25, size=cache_shape).astype(np.float32)
    value_f32 = rng.normal(0.0, 0.25, size=cache_shape).astype(np.float32)
    key_f32[blocks_per_row:] += 2.0
    value_f32[blocks_per_row:] -= 2.0
    key_cache = float_array_to_bf16_bits(key_f32)
    value_cache = float_array_to_bf16_bits(value_f32)
    block_table = np.arange(rows * blocks_per_row, dtype=np.int32).reshape(rows, blocks_per_row)
    live_counts = np.full((rows,), max_context_len, dtype=np.int64)
    batch_out = np.empty((rows, num_q_heads, head_dim), dtype=np.float32)
    c1_out = np.empty_like(batch_out)

    buffers = []
    try:
        query_b = malloc(query.nbytes, runtime=runtime)
        key_b = malloc(key_cache.nbytes, runtime=runtime)
        value_b = malloc(value_cache.nbytes, runtime=runtime)
        table_b = malloc(block_table.nbytes, runtime=runtime)
        live_b = malloc(live_counts.nbytes, runtime=runtime)
        batch_out_b = malloc(batch_out.nbytes, runtime=runtime)
        c1_out_b = malloc(c1_out.nbytes, runtime=runtime)
        buffers.extend((query_b, key_b, value_b, table_b, live_b, batch_out_b, c1_out_b))
        copy_host_to_device(query_b, host_array_ptr(query), query.nbytes, runtime=runtime)
        copy_host_to_device(key_b, host_array_ptr(key_cache), key_cache.nbytes, runtime=runtime)
        copy_host_to_device(value_b, host_array_ptr(value_cache), value_cache.nbytes, runtime=runtime)
        copy_host_to_device(table_b, host_array_ptr(block_table), block_table.nbytes, runtime=runtime)
        copy_host_to_device(live_b, host_array_ptr(live_counts), live_counts.nbytes, runtime=runtime)

        batch_spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table_b.ptr, block_table.shape, "int32"),
            live_counts=_tensor(live_b.ptr, live_counts.shape, "int64"),
            max_live_count=max_context_len,
            storage_dtype=DType.BF16,
        )
        batch_kernel(
            query_b.ptr,
            key_b.ptr,
            value_b.ptr,
            batch_out_b.ptr,
            batch_spans,
            rows,
            max_context_len,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            runtime=runtime,
        )

        row_query_nbytes = num_q_heads * head_dim * DType.FP32.itemsize
        row_out_nbytes = row_query_nbytes
        row_cache_nbytes = blocks_per_row * block_size * num_kv_heads * head_dim * DType.BF16.itemsize
        row_live_nbytes = DType.INT64.itemsize
        for row in range(rows):
            qwen35_full_attn_decode_context_bf16(
                query_b.ptr + row * row_query_nbytes,
                key_b.ptr + row * row_cache_nbytes,
                value_b.ptr + row * row_cache_nbytes,
                c1_out_b.ptr + row * row_out_nbytes,
                live_b.ptr + row * row_live_nbytes,
                int(live_counts[row]),
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                runtime=runtime,
            )

        copy_device_to_host(host_array_ptr(batch_out), batch_out_b, batch_out.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(c1_out), c1_out_b, c1_out.nbytes, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    if bit_exact:
        np.testing.assert_array_equal(batch_out, c1_out)
    else:
        # The retained Q3 reduction is full-model exact at its admitted native
        # short-context shapes; this wider synthetic context differs only at
        # sub-2e-6 FP32 accumulation noise.
        np.testing.assert_allclose(batch_out, c1_out, rtol=3e-4, atol=2e-6)
