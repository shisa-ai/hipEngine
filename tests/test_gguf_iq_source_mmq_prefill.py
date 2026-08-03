"""Source-faithful llama.cpp-shaped IQ3/IQ4 selected-down MMQ gates.

The candidate reuses compact expert metadata, packs each BF16 routed row into
llama.cpp's K-major 144-byte DS4 Q8_1 records, and consumes raw expert-major
IQ3_XXS/IQ4_XS bytes with I128/J128/K256 integer-WMMA ownership.  The exact
grouped raw-IQ kernels remain the correctness fallback.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_k_mmq_prefill as q8_mmq
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests.test_gguf_iq_gemv import (
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_iq3_weight,
    _make_iq4_weight,
    _make_x,
)
from tests.test_gguf_iq_selected_prefill import CompactMeta, _compact_meta, _run_single_grouped

_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_iq_source_mmq_prefill.hip"
)
_VARIANT = "selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out"
_COUNTS = [0, 1, 17, 128, 129]


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _device_buffer(array: np.ndarray, buffers: list[object]):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    buffers.append(buffer)
    return buffer


def test_iq_source_mmq_registry_build_scope_and_metadata_contract() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    iq_mmq.register_gguf_iq_source_mmq_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    metadata = iq_mmq.build_iq_source_mmq128_metadata(_COUNTS)
    np.testing.assert_array_equal(
        metadata.expert_start_mmq,
        np.asarray([0, 0, 128, 256, 384, 640], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        metadata.tile_expert,
        np.asarray([1, 2, 3, 4, 4], dtype=np.int64),
    )
    assert metadata.mmq_total_rows == 640
    with pytest.raises(ValueError, match="non-empty"):
        iq_mmq.build_iq_source_mmq128_metadata([])
    with pytest.raises(ValueError, match="non-negative"):
        iq_mmq.build_iq_source_mmq128_metadata([1, -1])

    for quant, fn in (
        (
            "gguf_iq3_xxs",
            iq_mmq.gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            iq_mmq.gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
    ):
        key = KernelKey("hip_gfx1100", "moe_linear", quant, _VARIANT)
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is fn
        assert not is_registered(
            KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
        )

    artifact = iq_mmq.plan_gguf_iq_source_mmq_prefill_build(
        compiler_version="test"
    )
    assert artifact.output_path.name == "gguf_iq_source_mmq_prefill.so"
    assert "-ffast-math" in artifact.flags
    assert any(path.name == _SOURCE.name for path in artifact.sources)
    source = _SOURCE.read_text()
    assert "llama.cpp HIP c0bc8591e8815c63cb01dd3f051a8b0df02501c9" in source
    assert "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32" in source
    assert "torch::Tensor" not in source


def test_iq_source_mmq_wrappers_reject_unsupported_shapes_before_loading() -> None:
    kwargs = dict(
        xq_ptr=1,
        expert_start_compact_ptr=2,
        expert_start_mmq_ptr=3,
        tile_expert_ptr=4,
        qweight_ptr=5,
        out_ptr=6,
        compact_rows=17,
        in_features=1024,
        out_features=128,
        num_experts=4,
        mmq_total_rows=128,
    )
    for fn in (
        iq_mmq.gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        iq_mmq.gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
    ):
        with pytest.raises(ValueError, match="divisible by 256"):
            fn(**{**kwargs, "in_features": 896})
        with pytest.raises(ValueError, match="multiple of 128"):
            fn(**{**kwargs, "out_features": 127})
        with pytest.raises(ValueError, match="multiple of 128"):
            fn(**{**kwargs, "mmq_total_rows": 127})


def _run_candidate(
    *,
    quant: str,
    compact: CompactMeta,
    x_bf16: np.ndarray,
    qweight: np.ndarray,
) -> np.ndarray:
    metadata = iq_mmq.build_iq_source_mmq128_metadata(compact.counts)
    out = np.zeros((compact.compact_rows, qweight.shape[1]), dtype=np.uint16)
    packed_nbytes = q8_mmq.q8_1_ds4_kmajor_nbytes(
        compact.compact_rows, x_bf16.shape[1]
    )
    producer_library = q8_mmq.build_gguf_k_mmq_prefill(load=True)
    consumer_library = iq_mmq.build_gguf_iq_source_mmq_prefill(load=True)
    before = memory_stats()
    buffers: list[object] = []
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        packed_buf = malloc(packed_nbytes)
        buffers.append(packed_buf)
        compact_start_buf = _device_buffer(compact.expert_start_compact, buffers)
        mmq_start_buf = _device_buffer(metadata.expert_start_mmq, buffers)
        tile_buf = _device_buffer(metadata.tile_expert, buffers)
        weight_buf = _device_buffer(qweight, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        q8_mmq.gguf_q8_1_ds4_quantize_bf16_kmajor(
            x_buf.ptr,
            packed_buf.ptr,
            compact.compact_rows,
            x_bf16.shape[1],
            library=producer_library,
        )
        fn = (
            iq_mmq.gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out
            if quant == "gguf_iq3_xxs"
            else iq_mmq.gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out
        )
        fn(
            packed_buf.ptr,
            compact_start_buf.ptr,
            mmq_start_buf.ptr,
            tile_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            compact_rows=compact.compact_rows,
            in_features=x_bf16.shape[1],
            out_features=qweight.shape[1],
            num_experts=compact.num_experts,
            mmq_total_rows=metadata.mmq_total_rows,
            library=consumer_library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
    return out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("quant", ["gguf_iq3_xxs", "gguf_iq4_xs"])
def test_iq_source_mmq_expert_tails_are_finite_and_pass_exact_quality(
    quant: str,
) -> None:
    compact = _compact_meta(_COUNTS)
    in_features = 1024
    out_features = 128
    x_bf16 = _f32_to_bf16_u16(_make_x(compact.compact_rows, in_features))
    if quant == "gguf_iq3_xxs":
        qweight = _make_iq3_weight(compact.num_experts, out_features, in_features)
        exact_wrapper = (
            gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out
        )
    else:
        qweight = _make_iq4_weight(compact.num_experts, out_features, in_features)
        exact_wrapper = gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out
    exact = _run_single_grouped(
        exact_wrapper,
        build_gguf_iq_selected_prefill(load=True),
        x_bf16=x_bf16,
        meta=compact,
        qweight=qweight,
        wmma=False,
    )
    candidate = _run_candidate(
        quant=quant,
        compact=compact,
        x_bf16=x_bf16,
        qweight=qweight,
    )
    exact_f32 = _bf16_u16_to_f32(exact)
    candidate_f32 = _bf16_u16_to_f32(candidate)
    assert np.all(np.isfinite(candidate_f32))
    result = evaluate_logits(exact_f32, candidate_f32)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result
