from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    plan_qwen35_paged_attn_decode_build,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.kvcache import KVLiveSpans


def setup_function() -> None:
    clear_registry_for_tests()


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans(*, storage_dtype: str = "bf16", live_dtype: str = "int64") -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (1,), "int32"),
        live_counts=_tensor(0x2000, (1,), live_dtype),
        max_live_count=2,
        storage_dtype=storage_dtype,
    )


def test_qwen35_paged_attn_decode_registers_span_variant() -> None:
    register_qwen35_paged_attn_decode_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_spans
    )


def test_qwen35_paged_attn_decode_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_paged_attn_decode_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 paged attn decode test version",
    )

    assert artifact.family == "qwen35_paged_attn_decode"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 64
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_paged_attn_decode.so"
    assert artifact.compiler_version == "hipcc qwen35 paged attn decode test version"
    assert any(str(path).endswith("paged_attn_decode.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_paged_attn_decode_wrapper_validates_before_gpu_load() -> None:
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
