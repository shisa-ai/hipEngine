from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    paged_kv_write as paged_kv_write_module,
    plan_qwen35_paged_kv_write_build,
    qwen35_write_paged_kv_f32_spans,
    qwen35_write_paged_kv_int8_per_token_head_batch_spans,
    qwen35_write_paged_kv_int8_per_token_head_prompt_spans,
    qwen35_write_paged_kv_int8_per_token_head_spans,
    qwen35_write_paged_kv_mixed_value_bf16_batch_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
    qwen35_write_paged_kv_mixed_value_fp16_batch_spans,
    qwen35_write_paged_kv_mixed_value_fp16_prompt_spans,
    qwen35_write_paged_kv_mixed_value_fp16_spans,
    register_qwen35_paged_kv_write_kernels,
)
from hipengine.kernels.registry import (
    KernelKey,
    clear_registry_for_tests,
    is_registered,
    resolve,
)
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata


def setup_function() -> None:
    clear_registry_for_tests()


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans(*, storage_dtype: str = "bf16", live_dtype: str = "int64", max_live: int = 1) -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2,), "int32"),
        live_counts=_tensor(0x2000, (1,), live_dtype),
        max_live_count=max_live,
        storage_dtype=storage_dtype,
    )


def _shared_batch_spans(*, rows: int = 2, table_len: int = 2) -> KVLiveSpans:
    positions = _tensor(0x2000, (rows,), "int64")
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (table_len,), "int32"),
        live_counts=positions,
        max_live_count=table_len * 4 - 1,
        storage_dtype="bf16",
        row_positions=positions,
        span_role="verify_chain",
    )


def _int8_spans(
    *,
    block_table_shape: tuple[int, ...] = (2,),
    live_shape: tuple[int, ...] = (1,),
    scale_shape: tuple[int, ...] = (2, 4, 2),
    scale_dtype: str = "fp32",
    live_dtype: str = "int64",
    max_live: int = 1,
) -> KVLiveSpans:
    metadata = KVScaleMetadata(
        k_scale=_tensor(0x3000, scale_shape, scale_dtype),
        v_scale=_tensor(0x4000, scale_shape, scale_dtype),
        scale_dtype=scale_dtype,
    )
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, block_table_shape, "int32"),
        live_counts=_tensor(0x2000, live_shape, live_dtype),
        max_live_count=max_live,
        storage_dtype="int8_per_token_head",
        scale_metadata=metadata,
    )


def test_qwen35_paged_kv_write_registers_span_variants() -> None:
    register_qwen35_paged_kv_write_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="w4_paro",
            variant="mixed_bf16_spans",
        )
        is qwen35_write_paged_kv_mixed_value_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="w4_paro",
            variant="mixed_bf16_batch_spans",
        )
        is qwen35_write_paged_kv_mixed_value_bf16_batch_spans
    )
    shared_batch = getattr(
        paged_kv_write_module,
        "qwen35_write_paged_kv_mixed_value_bf16_shared_batch_spans",
        None,
    )
    assert callable(shared_batch)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="gguf_q4_k_m",
            variant="mixed_bf16_shared_batch_spans",
        )
        is shared_batch
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="w4_paro",
            variant="mixed_fp16_spans",
        )
        is qwen35_write_paged_kv_mixed_value_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="w4_paro",
            variant="mixed_fp16_batch_spans",
        )
        is qwen35_write_paged_kv_mixed_value_fp16_batch_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="w4_paro",
            variant="mixed_fp16_prompt_spans",
        )
        is qwen35_write_paged_kv_mixed_value_fp16_prompt_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="w4_paro",
            variant="f32_spans",
        )
        is qwen35_write_paged_kv_f32_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="int8_per_token_head",
            variant="per_token_head_spans",
        )
        is qwen35_write_paged_kv_int8_per_token_head_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="int8_per_token_head",
            variant="per_token_head_prompt_spans",
        )
        is qwen35_write_paged_kv_int8_per_token_head_prompt_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_kv_write",
            quant="int8_per_token_head",
            variant="per_token_head_batch_spans",
        )
        is qwen35_write_paged_kv_int8_per_token_head_batch_spans
    )


def test_qwen35_shared_batch_kv_write_is_not_aliased_to_gfx1151() -> None:
    register_qwen35_paged_kv_write_kernels()
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "paged_kv_write",
            "gguf_q4_k_m",
            "mixed_bf16_shared_batch_spans",
        )
    )


def test_qwen35_paged_kv_write_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_paged_kv_write_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 paged kv write test version",
    )

    assert artifact.family == "qwen35_paged_kv_write"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_paged_kv_write.so"
    assert artifact.compiler_version == "hipcc qwen35 paged kv write test version"
    assert any(str(path).endswith("paged_kv_write.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_paged_kv_write_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        qwen35_write_paged_kv_mixed_value_bf16_spans(0, 0, 0, 0, _spans(), 0, 1, 8)
    with pytest.raises(ValueError, match="int64 live_counts"):
        qwen35_write_paged_kv_mixed_value_bf16_spans(
            0, 0, 0, 0, _spans(live_dtype="int32"), 4, 1, 8
        )
    with pytest.raises(ValueError, match="bf16 storage"):
        qwen35_write_paged_kv_f32_spans(0, 0, 0, 0, _spans(storage_dtype="fp16"), 4, 1, 8)
    with pytest.raises(ValueError, match="max_live_count"):
        qwen35_write_paged_kv_f32_spans(0, 0, 0, 0, _spans(max_live=8), 4, 1, 8)
    with pytest.raises(ValueError, match="rows"):
        qwen35_write_paged_kv_mixed_value_bf16_batch_spans(0, 0, 0, 0, _spans(), 0, 4, 1, 8)
    shared_batch = getattr(
        paged_kv_write_module,
        "qwen35_write_paged_kv_mixed_value_bf16_shared_batch_spans",
        None,
    )
    assert callable(shared_batch)
    with pytest.raises(ValueError, match="rows must be 2 or 4"):
        shared_batch(0, 0, 0, 0, _shared_batch_spans(rows=3), 3, 4, 1, 8)
    with pytest.raises(ValueError, match="row_positions"):
        spans = _shared_batch_spans()
        spans = KVLiveSpans.paged_uniform(
            block_table=spans.base_offsets,
            live_counts=spans.live_counts,
            max_live_count=spans.max_live_count,
            storage_dtype=spans.storage_dtype,
            span_role=spans.span_role,
        )
        shared_batch(0, 0, 0, 0, spans, 2, 4, 1, 8)
    with pytest.raises(ValueError, match="block_size must be positive"):
        qwen35_write_paged_kv_mixed_value_fp16_spans(0, 0, 0, 0, _spans(), 0, 1, 8)
    with pytest.raises(ValueError, match="rows"):
        qwen35_write_paged_kv_mixed_value_fp16_batch_spans(0, 0, 0, 0, _spans(), 0, 4, 1, 8)
    with pytest.raises(ValueError, match="rows"):
        qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(0, 0, 0, 0, _spans(), 0, 4, 1, 8)
    with pytest.raises(ValueError, match="live_counts"):
        qwen35_write_paged_kv_mixed_value_bf16_batch_spans(
            0,
            0,
            0,
            0,
            KVLiveSpans.paged_uniform(
                block_table=_tensor(0x1000, (4,), "int32"),
                live_counts=_tensor(0x2000, (1,), "int64"),
                max_live_count=1,
                storage_dtype="bf16",
            ),
            2,
            4,
            1,
            8,
        )
    int8_spans = _int8_spans()
    with pytest.raises(ValueError, match="int8_per_token_head storage"):
        qwen35_write_paged_kv_int8_per_token_head_spans(0, 0, 0, 0, 0, 0, _spans(), 4, 2, 8)
    with pytest.raises(ValueError, match="int64 live_counts"):
        bad_live = _int8_spans(live_dtype="int32")
        qwen35_write_paged_kv_int8_per_token_head_spans(
            0,
            0,
            0,
            0,
            bad_live.scale_metadata.k_scale.ptr,
            bad_live.scale_metadata.v_scale.ptr,
            bad_live,
            4,
            2,
            8,
        )
    with pytest.raises(ValueError, match="k_scale_ptr"):
        qwen35_write_paged_kv_int8_per_token_head_spans(
            0,
            0,
            0,
            0,
            int8_spans.scale_metadata.k_scale.ptr + 1,
            int8_spans.scale_metadata.v_scale.ptr,
            int8_spans,
            4,
            2,
            8,
        )
    with pytest.raises(ValueError, match="scale tensor shape"):
        bad_shape = _int8_spans(scale_shape=(2, 3, 2))
        qwen35_write_paged_kv_int8_per_token_head_spans(
            0,
            0,
            0,
            0,
            bad_shape.scale_metadata.k_scale.ptr,
            bad_shape.scale_metadata.v_scale.ptr,
            bad_shape,
            4,
            2,
            8,
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_write_paged_kv_int8_per_token_head_batch_spans(
            0,
            0,
            0,
            0,
            int8_spans.scale_metadata.k_scale.ptr,
            int8_spans.scale_metadata.v_scale.ptr,
            int8_spans,
            0,
            4,
            2,
            8,
        )
    with pytest.raises(ValueError, match="live_counts"):
        short_live = _int8_spans(block_table_shape=(4,), live_shape=(1,), scale_shape=(4, 4, 2))
        qwen35_write_paged_kv_int8_per_token_head_prompt_spans(
            0,
            0,
            0,
            0,
            short_live.scale_metadata.k_scale.ptr,
            short_live.scale_metadata.v_scale.ptr,
            short_live,
            2,
            4,
            2,
            8,
        )
