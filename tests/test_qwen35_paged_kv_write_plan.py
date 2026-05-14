from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    plan_qwen35_paged_kv_write_build,
    qwen35_write_paged_kv_f32_spans,
    qwen35_write_paged_kv_mixed_value_bf16_batch_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
    qwen35_write_paged_kv_mixed_value_fp16_batch_spans,
    qwen35_write_paged_kv_mixed_value_fp16_spans,
    register_qwen35_paged_kv_write_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.kvcache import KVLiveSpans


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
            variant="f32_spans",
        )
        is qwen35_write_paged_kv_f32_spans
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
    with pytest.raises(ValueError, match="block_size must be positive"):
        qwen35_write_paged_kv_mixed_value_fp16_spans(0, 0, 0, 0, _spans(), 0, 1, 8)
    with pytest.raises(ValueError, match="rows"):
        qwen35_write_paged_kv_mixed_value_fp16_batch_spans(0, 0, 0, 0, _spans(), 0, 4, 1, 8)
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
