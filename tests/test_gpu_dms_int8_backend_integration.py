"""Qualified compact backends must carry INT8 payloads and FP32 scales."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core import DType
from hipengine.kvcache.dms import DMSCompactBackend, DMSCodecQualification, encode_dms_payload
from tests.test_gpu_kvcache_dms_device_hip import _make_backend
from tests.test_gpu_dms_streaming_pack_hip import _admit, _bf16_bits, _bf16_from_bits, _hip_available


def _backend(device):
    parent = _make_backend(num_layers=1, heads=2, dim=16, window=2,
                           slots=128, device=False)
    return DMSCompactBackend(
        retrofit=parent.retrofit, codec="int8_per_token_head", slots_per_layer=128,
        max_request_rows=2, max_pack_rows=32, device_payloads=device,
        codec_qualification=DMSCodecQualification(
            codec="int8_per_token_head", artifact_fingerprint=parent.retrofit.artifact_fingerprint,
            kl_divergence=0, top1_agreement=1, no_dense_shadow=True,
            evidence_source="synthetic test fixture; not model qualification"))


def test_int8_batch_scale_contract_is_fp32():
    backend = _backend(False)
    _admit(backend, request_id=0, tokens=9)
    view = backend.prepare(SimpleNamespace(request_ids=(0,), span_role="decode"))
    assert view.live_spans.scale_metadata.k_scale.dtype == DType.FP32
    assert view.live_spans.scale_metadata.v_scale.dtype == DType.FP32


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")
def test_int8_backend_pack_uses_compact_device_codec():
    backend = _backend(True)
    try:
        assert backend.device_payloads_enabled
        rng = np.random.default_rng(105)
        for rid, tokens in enumerate((9, 6)):
            _admit(backend, request_id=rid, tokens=tokens)
            k = _bf16_from_bits(_bf16_bits(rng.normal(size=(tokens, 1, 2, 16)).astype(np.float32)))
            v = _bf16_from_bits(_bf16_bits(rng.normal(size=k.shape).astype(np.float32)))
            backend.streaming_pack(rid, k, v, np.ones((tokens, 1, 2), dtype=bool))
            state = backend.state_for_request(rid)
            assert not state.k_payload and not state.v_payload
            view = backend.device_layer_view(rid, 0)
            assert view.k_bits.dtype == np.int8
            for head in range(2):
                n = int(state.live_counts[0, head]); base = int(state.base_offsets[0, head])
                for values, payload, scales in ((k, view.k_bits, view.k_scales), (v, view.v_bits, view.v_scales)):
                    q, s = encode_dms_payload(values[-n:, 0, head], codec="int8_per_token_head")
                    np.testing.assert_array_equal(payload[base:base+n], q)
                    np.testing.assert_array_equal(scales[base:base+n], s)
            kernel = backend.device_layer_kernel_view(rid, 0)
            assert kernel['codec'] == 'int8_per_token_head'
            assert kernel['k_scale_ptr'] and kernel['v_scale_ptr']
    finally:
        backend.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")
def test_int8_backend_rollback_cancel_refill_preserves_other_request():
    backend = _backend(True)
    try:
        values = np.ones((9, 1, 2, 16), dtype=np.float32)
        for rid in (0, 1):
            _admit(backend, request_id=rid, tokens=9)
            backend.streaming_pack(rid, values * (rid + 1), values, np.ones((9, 1, 2), bool))
        state0 = backend.state_for_request(0)
        state1 = backend.state_for_request(1)
        transaction = backend.begin_transaction([state0.lease], None)
        before = backend.device_layer_view(0, 0)
        live_before = state0.live_counts.copy()
        backend.append_decode(0, values[0] * 3, values[0], np.zeros((1, 2), bool), position=9)
        assert backend.evicted_tokens > 0
        backend.rollback(transaction)
        after = backend.device_layer_view(0, 0)
        np.testing.assert_array_equal(state0.live_counts, live_before)
        for head in range(2):
            base = int(state0.base_offsets[0, head]); cap = int(state0.range_capacity[0, head])
            for name in ('k_bits', 'v_bits', 'k_scales', 'v_scales', 'positions', 'evict'):
                np.testing.assert_array_equal(getattr(before, name)[base:base+cap],
                                              getattr(after, name)[base:base+cap])
        backend.reclaim(state0.lease)
        _admit(backend, request_id=2, tokens=9)
        backend.streaming_pack(2, values * 4, values, np.zeros((9, 1, 2), bool))
        final = backend.device_layer_view(1, 0)
        for head in range(2):
            base = int(state1.base_offsets[0, head]); n = int(state1.live_counts[0, head])
            for name in ('k_bits', 'v_bits', 'k_scales', 'v_scales', 'positions', 'evict'):
                np.testing.assert_array_equal(getattr(before, name)[base:base+n],
                                              getattr(final, name)[base:base+n])
        backend.assert_conserved()
        for rid in (1, 2):
            backend.reclaim(backend.state_for_request(rid).lease)
        backend.assert_conserved()
        assert not backend._states
    finally:
        backend.close()
