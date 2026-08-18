"""Strict RED gate for the HIP DMS decision-extraction kernel (C2-7 U1).

The borrowed-neuron DMS convention stores one scalar eviction decision in the
last channel of the first query head of each GQA group. The device kernel must
be bit-exact against the registered ``cpu_reference`` ``dms_extract_decision``
(NumPy float64 scalar comparisons, in-place channel zeroing). GPU cases skip
cleanly on no-ROCm runners.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.kvcache.dms import DMSRetrofitConfig, extract_dms_eviction_decisions


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_round(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (float32 bf16-rounded values, bf16 bit pattern as uint16)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000))
    return rounded.view(np.float32).copy(), (rounded >> np.uint32(16)).astype(np.uint16)


def _config(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    alpha_scale: float,
    alpha_offset: float,
) -> DMSRetrofitConfig:
    return DMSRetrofitConfig(
        artifact_fingerprint="fixture:dms-extract",
        model_family="qwen35",
        num_layers=2,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        window_size=16,
        target_compression_ratio=2,
        alpha_scale=alpha_scale,
        alpha_offset=alpha_offset,
        borrowed_query_channel=head_dim - 1,
        corrected_mask=True,
        trained_checkpoint=True,
        evidence_source="unit fixture",
        source_path="tests/fixtures/dms_extract",
    )


def _fixture(q_heads: int, kv_heads: int, head_dim: int, *, seed: int) -> np.ndarray:
    """Q with decision channels straddling the eviction threshold."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((5, q_heads, head_dim)).astype(np.float32)
    # Overwrite the borrowed channel (first q-head of each group, last dim)
    # with values straddling the decision boundary so both eviction outcomes
    # are exercised. The threshold lives at alpha_offset / alpha_scale = 0.05
    # for the positive-scale fixture; use a boundary-independent spread.
    for h in range(kv_heads):
        channel = q[:, h * (q_heads // kv_heads), head_dim - 1]
        channel[:] = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    return q


def _run_device(
    q_bits: np.ndarray,
    *,
    alpha_scale: float,
    alpha_offset: float,
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    library,
) -> tuple[np.ndarray, np.ndarray]:
    from hipengine.kernels.hip_gfx1100.attention import dms_extract_decision_bf16

    evict = np.zeros((tokens, kv_heads), dtype=np.uint8)
    q_buf = malloc(q_bits.nbytes)
    evict_buf = malloc(evict.nbytes)
    try:
        copy_host_to_device(q_buf, host_array_ptr(np.ascontiguousarray(q_bits)), q_bits.nbytes)
        copy_host_to_device(evict_buf, host_array_ptr(evict), evict.nbytes)
        dms_extract_decision_bf16(
            q_buf.ptr,
            evict_buf.ptr,
            alpha_scale,
            alpha_offset,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            library=library,
        )
        out_evict = np.empty_like(evict)
        out_q_bits = np.empty_like(q_bits)
        copy_device_to_host(host_array_ptr(out_evict), evict_buf, evict.nbytes)
        copy_device_to_host(
            host_array_ptr(out_q_bits), q_buf, q_bits.nbytes
        )
    finally:
        free(evict_buf)
        free(q_buf)
    return out_evict, out_q_bits


def _strict_case(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    alpha_scale: float,
    alpha_offset: float,
    seed: int,
) -> None:
    from hipengine.kernels.hip_gfx1100.attention import build_dms_compact

    q_f32, q_bits = _bf16_round(_fixture(q_heads, kv_heads, head_dim, seed=seed))
    tokens = q_f32.shape[0]
    config = _config(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        alpha_scale=alpha_scale,
        alpha_offset=alpha_offset,
    )
    cleaned_ref, evict_ref = extract_dms_eviction_decisions(q_f32, config, inplace=False)
    evict_ref_bits = evict_ref.astype(np.uint8)
    cleaned_ref_bits = _bf16_round(cleaned_ref.reshape(tokens, -1))[1].reshape(
        tokens, q_heads, head_dim
    )
    assert evict_ref_bits.any(), "fixture must exercise eviction"
    assert not evict_ref_bits.all(), "fixture must exercise retention"

    library = build_dms_compact(load=True)
    out_evict, out_q_bits = _run_device(
        q_bits,
        alpha_scale=alpha_scale,
        alpha_offset=alpha_offset,
        tokens=tokens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        library=library,
    )
    np.testing.assert_array_equal(
        out_evict, evict_ref_bits, err_msg="eviction bits differ from CPU reference"
    )
    np.testing.assert_array_equal(
        out_q_bits,
        cleaned_ref_bits,
        err_msg="cleaned Q (borrowed-channel zeroing) differs from CPU reference",
    )


def test_dms_extract_decision_registers_and_build_plan() -> None:
    clear_registry_for_tests()
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_extract_decision_bf16,
        plan_dms_compact_build,
        register_dms_compact_kernels,
    )

    register_dms_compact_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_extract_decision",
            quant="bf16",
            variant="corrected_mask",
        )
        is dms_extract_decision_bf16
    )
    artifact = plan_dms_compact_build(compiler_version="dms-test-version")
    assert artifact.family == "dms_compact"
    assert artifact.output_path.name == "dms_compact.so"
    assert any(path.name == "dms_compact.hip" for path in artifact.sources)


def test_dms_extract_decision_wrapper_validates_before_gpu_load() -> None:
    from hipengine.kernels.hip_gfx1100.attention import dms_extract_decision_bf16

    with pytest.raises(ValueError, match="q_heads"):
        dms_extract_decision_bf16(0, 0, 100.0, 5.0, 4, 8, 3, 16)
    with pytest.raises(ValueError, match="tokens"):
        dms_extract_decision_bf16(0, 0, 100.0, 5.0, 0, 8, 4, 16)
    with pytest.raises(ValueError, match="alpha_scale"):
        dms_extract_decision_bf16(0, 0, 0.0, 5.0, 4, 8, 4, 16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_extract_decision_positive_scale_bit_exact() -> None:
    _strict_case(
        q_heads=8,
        kv_heads=4,
        head_dim=16,
        alpha_scale=100.0,
        alpha_offset=5.0,
        seed=11,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_extract_decision_production_head_geometry_bit_exact() -> None:
    _strict_case(
        q_heads=24,
        kv_heads=4,
        head_dim=128,
        alpha_scale=100.0,
        alpha_offset=5.0,
        seed=22,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_extract_decision_negative_scale_branch_bit_exact() -> None:
    _strict_case(
        q_heads=8,
        kv_heads=2,
        head_dim=16,
        alpha_scale=-100.0,
        alpha_offset=5.0,
        seed=33,
    )
