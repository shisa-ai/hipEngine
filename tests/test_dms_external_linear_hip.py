"""Production-profile RED gate for schema-v2 GPU-resident DMS decisions."""

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
from hipengine.kvcache import (
    DMSLinearSidecarSpec,
    DMSRetrofitConfig,
    DMSTrainingProvenance,
)
from hipengine.kvcache.dms_device import DMSExternalLinearDeviceProjector
from hipengine.kvcache.dms_sidecar import ExternalDMSLinearSidecar
from hipengine.runtime.qwen35_gguf_runner import _ExternalDMSDevicePrefillCollector


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (
        bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    return np.ascontiguousarray(rounded >> np.uint32(16), dtype=np.uint16)


def _bf16_float(values: np.ndarray) -> np.ndarray:
    return (
        _bf16_bits(values).astype(np.uint32) << np.uint32(16)
    ).view(np.float32)


def _source(*, hidden_size: int = 5120, kv_heads: int = 4) -> ExternalDMSLinearSidecar:
    sidecar = DMSLinearSidecarSpec(
        path="sidecar.safetensors",
        format="safetensors",
        dtype="bfloat16",
        weight_tensor="weight",
        bias_tensor="bias",
        weight_shape=(1, kv_heads, hidden_size),
        bias_shape=(1, kv_heads),
        sha256="a" * 64,
    )
    training = DMSTrainingProvenance(
        method="future_attention_distillation_v1",
        data_manifest_sha256="b" * 64,
        trainer_commit="c" * 40,
        fastdms_reference_commit="c602b0ec3266da7f74d6a658b3dafcddb443fddd",
        seed=0,
    )
    config = DMSRetrofitConfig(
        schema_version=2,
        artifact_fingerprint="d" * 64,
        model_family="qwen35_dense_hybrid",
        decision_source="external_linear_sidecar_v1",
        physical_layer_ids=(3,),
        num_layers=1,
        num_q_heads=24,
        num_kv_heads=kv_heads,
        head_dim=256,
        hidden_size=hidden_size,
        input_stage="post_attn_rmsnorm_pre_q_projection",
        window_size=256,
        target_compression_ratio=2,
        alpha_scale=1.0,
        alpha_offset=0.15,
        borrowed_query_channel=None,
        zero_borrowed_query_channel=False,
        corrected_mask=False,
        trained_checkpoint=True,
        evidence_source="fixture",
        source_path="fixture.json",
        sidecar=sidecar,
        training=training,
    )
    rng = np.random.default_rng(20260823)
    weight = rng.standard_normal((1, kv_heads, hidden_size)).astype(np.float32)
    weight *= np.float32(hidden_size**-0.5)
    weight = _bf16_float(weight)
    bias = _bf16_float(
        np.asarray([[-0.75, -0.2, 0.35, 0.9]], dtype=np.float32)[:, :kv_heads]
    )
    return ExternalDMSLinearSidecar(config=config, weight=weight, bias=bias)


def test_external_linear_device_decision_registers_for_both_rdna3_backends() -> None:
    clear_registry_for_tests()
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_external_linear_decision_bf16,
        plan_dms_compact_build,
        register_dms_compact_kernels,
    )

    register_dms_compact_kernels()
    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package("hip_gfx1151")
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        assert (
            resolve(
                backend=backend,
                layer="dms_decision_source",
                quant="bf16",
                variant="external_linear_sidecar_v1",
            )
            is dms_external_linear_decision_bf16
        )
    artifact = plan_dms_compact_build(compiler_version="dms-external-test")
    assert artifact.family == "dms_compact"


def test_external_linear_device_wrapper_validates_before_gpu_load() -> None:
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_external_linear_decision_bf16,
    )

    with pytest.raises(ValueError, match="tokens"):
        dms_external_linear_decision_bf16(
            0, 0, 0, 0, 0, 1.0, 0.0, 0, 5120, 4
        )
    with pytest.raises(ValueError, match="hidden_size"):
        dms_external_linear_decision_bf16(
            0, 0, 0, 0, 0, 1.0, 0.0, 1, 0, 4
        )
    with pytest.raises(ValueError, match="alpha_scale"):
        dms_external_linear_decision_bf16(
            0, 0, 0, 0, 0, 0.0, 0.0, 1, 5120, 4
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_integrated_no_evict_collector_publishes_all_false_decisions() -> None:
    from hipengine.core.hip import get_hip_runtime

    source = _source()
    collector = _ExternalDMSDevicePrefillCollector(
        source,
        token_count=7,
        backend="hip_gfx1151",
        runtime=get_hip_runtime(),
        decision_mode="no_evict",
    )
    try:
        collector.capture_device_chunk(
            physical_layer_id=3,
            compact_layer_index=0,
            start=0,
            rows=7,
            hidden_ptr=1,
            stream=0,
        )
        decisions = collector.finalize()
    finally:
        collector.close()

    assert decisions.shape == (7, 1, 4)
    assert not bool(np.any(decisions))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_external_linear_device_projector_matches_bf16_cpu_decisions_production_geometry() -> None:
    source = _source()
    tokens = 7
    rng = np.random.default_rng(77)
    hidden_bits = _bf16_bits(
        rng.standard_normal((tokens, int(source.config.hidden_size))).astype(np.float32)
    )
    expected_logits, expected_decisions = source.project(
        hidden_bits,
        compact_layer_index=0,
    )
    # Keep the fixture away from the threshold so changed control decisions are
    # a real kernel error rather than an accumulation-order boundary ambiguity.
    margins = np.abs(
        expected_logits * source.config.alpha_scale - source.config.alpha_offset
    )
    assert float(np.min(margins)) > 0.05

    hidden_buf = malloc(hidden_bits.nbytes)
    logits = np.empty((tokens, source.config.num_kv_heads), dtype=np.float32)
    decisions = np.empty((tokens, source.config.num_kv_heads), dtype=np.uint8)
    logits_buf = malloc(logits.nbytes)
    decisions_buf = malloc(decisions.nbytes)
    projector = DMSExternalLinearDeviceProjector(source, backend="hip_gfx1151")
    try:
        copy_host_to_device(hidden_buf, host_array_ptr(hidden_bits), hidden_bits.nbytes)
        projector.project(
            hidden_ptr=hidden_buf.ptr,
            compact_layer_index=0,
            tokens=tokens,
            logits_ptr=logits_buf.ptr,
            evict_ptr=decisions_buf.ptr,
        )
        copy_device_to_host(host_array_ptr(logits), logits_buf, logits.nbytes)
        copy_device_to_host(
            host_array_ptr(decisions), decisions_buf, decisions.nbytes
        )
        assert projector.resident_bytes == (4 * 5120 + 4) * 2
    finally:
        projector.close()
        free(decisions_buf)
        free(logits_buf)
        free(hidden_buf)

    np.testing.assert_allclose(logits, expected_logits, rtol=3e-4, atol=3e-4)
    np.testing.assert_array_equal(decisions.astype(np.bool_), expected_decisions)
