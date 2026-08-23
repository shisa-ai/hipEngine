from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.dms import register_dms_cpu_reference_kernels
from hipengine.kernels.registry import resolve
from hipengine.kvcache import (
    DMSLinearSidecarSpec,
    DMSRetrofitConfig,
    DMSTrainingProvenance,
    create_dms_bf16_backend,
)
from hipengine.kvcache.dms_sidecar import (
    DMSExternalDecisionRuntime,
    ExternalDMSDecisionCollector,
    ExternalDMSLinearSidecar,
)
from hipengine.models.qwen35_dms import resolve_qwen35_dms_decision_capability
from hipengine.runtime.qwen35_gguf_runner import _normalize_external_dms_decision_mode


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("sidecar", "sidecar"), ("NO-EVICT", "no_evict")),
)
def test_integrated_external_dms_decision_mode_normalizes_controls(
    raw: str,
    expected: str,
) -> None:
    assert _normalize_external_dms_decision_mode(raw) == expected


def test_integrated_external_dms_decision_mode_rejects_unknown_control() -> None:
    with pytest.raises(ValueError, match="dms_decision_mode"):
        _normalize_external_dms_decision_mode("dense")


def _config() -> DMSRetrofitConfig:
    sidecar = DMSLinearSidecarSpec(
        path="sidecar.safetensors",
        format="safetensors",
        dtype="bfloat16",
        weight_tensor="weight",
        bias_tensor="bias",
        weight_shape=(2, 2, 3),
        bias_shape=(2, 2),
        sha256="a" * 64,
    )
    training = DMSTrainingProvenance(
        method="future_attention_distillation_v1",
        data_manifest_sha256="b" * 64,
        trainer_commit="c" * 40,
        fastdms_reference_commit="c602b0ec3266da7f74d6a658b3dafcddb443fddd",
        seed=0,
    )
    return DMSRetrofitConfig(
        schema_version=2,
        artifact_fingerprint="d" * 64,
        model_family="qwen35_dense_hybrid",
        decision_source="external_linear_sidecar_v1",
        physical_layer_ids=(1, 3),
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=2,
        hidden_size=3,
        input_stage="post_attn_rmsnorm_pre_q_projection",
        window_size=1,
        target_compression_ratio=2,
        alpha_scale=1.0,
        alpha_offset=0.0,
        borrowed_query_channel=None,
        zero_borrowed_query_channel=False,
        corrected_mask=False,
        trained_checkpoint=True,
        evidence_source="fixture",
        source_path="fixture.json",
        sidecar=sidecar,
        training=training,
    )


def _source() -> ExternalDMSLinearSidecar:
    config = _config()
    weight = np.asarray(
        [
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    bias = np.zeros((2, 2), dtype=np.float32)
    return ExternalDMSLinearSidecar(config=config, weight=weight, bias=bias)


def test_external_decision_source_is_registered_with_cpu_strict_fallback() -> None:
    register_dms_cpu_reference_kernels(replace=True)
    kernel = resolve(
        backend="cpu_reference",
        layer="dms_decision_source",
        quant="bf16",
        variant="external_linear_sidecar_v1",
    )
    hidden = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    weight = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    bias = np.asarray([0.5, 0.25], dtype=np.float32)

    logits, decisions = kernel(
        hidden,
        weight,
        bias,
        alpha_scale=1.0,
        alpha_offset=0.0,
    )

    np.testing.assert_array_equal(logits, [[1.5, -1.75]])
    np.testing.assert_array_equal(decisions, [[True, False]])


def test_qwen_model_plugin_validates_hybrid_layer_map_and_fail_closed_modes() -> None:
    config = _config()
    capability = resolve_qwen35_dms_decision_capability(
        config,
        layer_types=("linear_attention", "full_attention", "linear_attention", "full_attention"),
        hidden_size=3,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )

    assert capability.physical_layer_ids == (1, 3)
    assert capability.strict_fallback == "paged_dense_bf16"
    assert capability.prefix_mode == "unsupported"
    assert capability.speculative_modes == ()
    with pytest.raises(ValueError, match="physical layer map"):
        resolve_qwen35_dms_decision_capability(
            config,
            layer_types=("full_attention",) * 4,
            hidden_size=3,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=2,
        )
    with pytest.raises(ValueError, match="sidecar geometry"):
        resolve_qwen35_dms_decision_capability(
            config,
            layer_types=("linear_attention", "full_attention", "linear_attention", "full_attention"),
            hidden_size=4,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=2,
        )


def test_external_runtime_projects_prefill_and_feeds_compact_backend_without_q_mutation() -> None:
    source = _source()
    runtime = DMSExternalDecisionRuntime(source)
    backend = create_dms_bf16_backend(
        retrofit=source.config,
        slots_per_layer=64,
        max_request_rows=1,
        max_pack_rows=16,
    )
    request = SimpleNamespace(
        request_id=7,
        prompt_tokens=tuple(range(6)),
        max_new_tokens=2,
    )
    lease = backend.reserve(backend.estimate(request, None, {}))
    hidden = np.zeros((6, 2, 3), dtype=np.float32)
    hidden[:, 0, 0] = [-3, -2, -1, 1, 2, 3]
    hidden[:, 1, 1] = [3, 2, 1, -1, -2, -3]
    original = hidden.copy()
    k = np.arange(6 * 2 * 2 * 2, dtype=np.float32).reshape(6, 2, 2, 2)
    v = k + 100.0

    decisions = runtime.streaming_pack(
        backend,
        request_id=7,
        hidden=hidden,
        k=k,
        v=v,
        span_role="prefill",
    )

    assert decisions.shape == (6, 2, 2)
    np.testing.assert_array_equal(hidden, original)
    state = backend.state_for_request(7)
    assert np.all(state.live_counts > 0)
    assert "external_linear_sidecar_v1" in backend.spec.kernel_bundle_key
    snapshot = backend.observability_snapshot()
    assert snapshot["backend"]["decision_source"] == "external_linear_sidecar_v1"
    assert snapshot["backend"]["physical_layer_ids"] == [1, 3]
    with pytest.raises(ValueError, match="unsupported span role"):
        runtime.streaming_pack(
            backend,
            request_id=7,
            hidden=hidden,
            k=k,
            v=v,
            span_role="verify_chain",
        )
    backend.reclaim(lease)


def test_external_runtime_decode_append_rolls_back_after_post_mutation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    runtime = DMSExternalDecisionRuntime(source)
    backend = create_dms_bf16_backend(
        retrofit=source.config,
        slots_per_layer=64,
        max_request_rows=1,
        max_pack_rows=16,
    )
    request = SimpleNamespace(request_id=9, prompt_tokens=tuple(range(4)), max_new_tokens=2)
    lease = backend.reserve(backend.estimate(request, None, {}))
    hidden = np.ones((4, 2, 3), dtype=np.float32)
    k = np.ones((4, 2, 2, 2), dtype=np.float32)
    v = np.full_like(k, 2.0)
    runtime.streaming_pack(backend, request_id=9, hidden=hidden, k=k, v=v)
    state = backend.state_for_request(9)
    before_counts = state.live_counts.copy()
    before_positions = state.token_positions.copy()
    original_append = backend.append_decode

    def mutate_then_fail(*args, **kwargs):
        original_append(*args, **kwargs)
        raise RuntimeError("injected post-mutation failure")

    monkeypatch.setattr(backend, "append_decode", mutate_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.append_decode(
            backend,
            request_id=9,
            hidden=np.ones((2, 3), dtype=np.float32),
            k=np.ones((2, 2, 2), dtype=np.float32),
            v=np.ones((2, 2, 2), dtype=np.float32),
            position=4,
        )

    np.testing.assert_array_equal(state.live_counts, before_counts)
    np.testing.assert_array_equal(state.token_positions, before_positions)
    backend.reclaim(lease)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_external_runtime_device_rollback_restores_request_extent_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    runtime = DMSExternalDecisionRuntime(source)
    backend = create_dms_bf16_backend(
        retrofit=source.config,
        slots_per_layer=64,
        max_request_rows=1,
        max_pack_rows=16,
        device_payloads=True,
    )
    try:
        request = SimpleNamespace(
            request_id=11,
            prompt_tokens=tuple(range(4)),
            max_new_tokens=2,
        )
        backend.reserve(backend.estimate(request, None, {}))
        hidden = np.ones((4, 2, 3), dtype=np.float32)
        k = np.ones((4, 2, 2, 2), dtype=np.float32)
        v = np.full_like(k, 2.0)
        runtime.streaming_pack(backend, request_id=11, hidden=hidden, k=k, v=v)
        before = [backend.device_layer_view(11, layer) for layer in range(2)]
        original_append = backend.append_decode

        def mutate_then_fail(*args, **kwargs):
            original_append(*args, **kwargs)
            raise RuntimeError("injected external device failure")

        monkeypatch.setattr(backend, "append_decode", mutate_then_fail)
        with pytest.raises(RuntimeError, match="injected external"):
            runtime.append_decode(
                backend,
                request_id=11,
                hidden=np.ones((2, 3), dtype=np.float32),
                k=np.ones((2, 2, 2), dtype=np.float32),
                v=np.ones((2, 2, 2), dtype=np.float32),
                position=4,
            )
        after = [backend.device_layer_view(11, layer) for layer in range(2)]
        for layer in range(2):
            np.testing.assert_array_equal(before[layer].k_bits, after[layer].k_bits)
            np.testing.assert_array_equal(before[layer].v_bits, after[layer].v_bits)
            np.testing.assert_array_equal(before[layer].positions, after[layer].positions)
            np.testing.assert_array_equal(before[layer].evict, after[layer].evict)
    finally:
        backend.close()


def test_external_decision_collector_maps_runtime_chunks_and_preserves_hidden() -> None:
    source = _source()
    collector = ExternalDMSDecisionCollector(source, token_count=4)
    hidden = np.asarray(
        [[-1.0, 2.0, 0.0], [1.0, -2.0, 0.0]],
        dtype=np.float32,
    )
    original = hidden.copy()
    unused_q = np.zeros((2, 4, 2), dtype=np.float32)
    unused_k = np.zeros((2, 2, 2), dtype=np.float32)
    for compact, physical in enumerate((1, 3)):
        collector.capture_chunk(
            physical_layer_id=physical,
            compact_layer_index=compact,
            positions=np.asarray([0, 1], dtype=np.int32),
            hidden_bf16=_bf16_bits(hidden),
            query_f32=unused_q,
            key_f32=unused_k,
        )
        collector.capture_chunk(
            physical_layer_id=physical,
            compact_layer_index=compact,
            positions=np.asarray([2, 3], dtype=np.int32),
            hidden_bf16=_bf16_bits(hidden),
            query_f32=unused_q,
            key_f32=unused_k,
        )
    collector.capture_teacher_logits(np.zeros((8,), dtype=np.float32))

    decisions = collector.finalize()

    assert decisions.shape == (4, 2, 2)
    np.testing.assert_array_equal(hidden, original)
    assert collector.teacher_vocab_size == 8


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)
