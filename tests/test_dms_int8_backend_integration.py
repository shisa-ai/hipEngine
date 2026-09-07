"""Qualified compact backends must carry INT8 payloads and FP32 scales."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core import DType
from hipengine.kvcache.dms import DMSCompactBackend, DMSCodecQualification, encode_dms_payload
from tests.test_kvcache_dms_device_hip import _make_backend
from tests.test_dms_streaming_pack_hip import _admit, _bf16_bits, _bf16_from_bits, _hip_available


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
