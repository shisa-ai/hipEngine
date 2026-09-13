"""Device INT8 storage must preserve DMS ownership and move scales with rows."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kvcache.dms_device import DMSDevicePayloadStore
from hipengine.kernels.cpu_reference.dms import encode_dms_payload
from tests.test_gpu_dms_streaming_pack_hip import _bf16_bits, _bf16_from_bits, _hip_available

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def _quantize(bits):
    return encode_dms_payload(_bf16_from_bits(bits), codec="int8_per_token_head")


@pytest.mark.parametrize("tokens,dim,heads,q_heads", [(19, 16, 2, 8), (1025, 256, 2, 8), (1025, 256, 4, 24), (8193, 256, 4, 24), (16385, 256, 4, 24)])
def test_int8_pack_append_attention_and_restore(tokens, dim, heads, q_heads, monkeypatch):
    window = 8192 if tokens > 8192 else 7
    slots = (tokens + 8) * heads + 11
    retrofit = SimpleNamespace(num_layers=1, num_kv_heads=heads,
                              num_q_heads=q_heads, head_dim=dim, window_size=window)
    store = DMSDevicePayloadStore(retrofit=retrofit, slots_per_layer=slots,
                                  max_pack_rows=tokens, codec="int8_per_token_head")
    rng = np.random.default_rng(822)
    k = _bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32))
    v = _bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32))
    k[0] = 0  # positive finite scales for all-zero vectors
    evict = (rng.random((tokens, heads)) < .6).astype(np.uint8)
    base = np.array([3 + h * (tokens + 7) for h in range(heads)], dtype=np.int32)
    cap = np.full(heads, tokens + 3, dtype=np.int32)
    try:
        bf16 = DMSDevicePayloadStore(retrofit=retrofit, slots_per_layer=slots,
                                     max_pack_rows=tokens)
        try:
            # Per-layer payloads are allocated lazily on first touch; touch the
            # single layer on both codecs before comparing residency.
            bf16._ensure_layer(0)
            store._ensure_layer(0)
            assert bf16.resident_bytes - store.resident_bytes == slots * (2 * dim - 8)
        finally:
            bf16.close()
        store.pack_layer(0, k, v, evict, base, cap)
        view = store.layer_view(0)
        assert view.k_bits.dtype == view.v_bits.dtype == np.int8
        assert view.k_scales.dtype == view.v_scales.dtype == np.float32
        expected = []
        for h in range(heads):
            keep = np.flatnonzero((evict[:, h] == 0) | (tokens - 1 - np.arange(tokens) <= window))
            expected.append(keep)
            dst = slice(base[h], base[h] + len(keep))
            for bits, payload, scales in [(k, view.k_bits, view.k_scales),
                                           (v, view.v_bits, view.v_scales)]:
                qp, sp = _quantize(bits[keep, h])
                np.testing.assert_array_equal(payload[dst], qp)
                np.testing.assert_array_equal(scales[dst], sp)
            np.testing.assert_array_equal(view.positions[dst], keep)
            np.testing.assert_array_equal(view.evict[dst], evict[keep, h])
        live = np.array([len(x) for x in expected], dtype=np.int32)
        np.testing.assert_array_equal(store.live_counts(0), live)
        snapshot = store.snapshot(base[None, :], cap[None, :])
        # Capacity failure must leave payload, scales, metadata and live counts intact.
        from hipengine.kvcache.dms_device import ENV_TRIPWIRE
        monkeypatch.setenv(ENV_TRIPWIRE, "1")
        with pytest.raises(RuntimeError, match="overflow"):
            store.append_layer(0, k[-1], v[-1], np.zeros(heads, np.uint8), tokens,
                               base, np.ones_like(cap), live)
        unchanged = store.layer_view(0)
        np.testing.assert_array_equal(store.live_counts(0), live)
        for name in ("k_bits", "v_bits", "k_scales", "v_scales", "positions", "evict"):
            np.testing.assert_array_equal(getattr(unchanged, name), getattr(view, name))
        kn = _bf16_bits(rng.normal(size=(heads, dim)).astype(np.float32))
        vn = _bf16_bits(rng.normal(size=(heads, dim)).astype(np.float32))
        store.append_layer(0, kn, vn, np.zeros(heads, np.uint8), tokens, base, cap, live)
        after = store.layer_view(0)
        new_live = store.live_counts(0)
        for h, positions in enumerate(expected):
            retained = positions[(evict[positions, h] == 0) | (tokens - positions <= window)]
            dst = slice(base[h], base[h] + len(retained) + 1)
            np.testing.assert_array_equal(after.positions[dst], np.append(retained, tokens))
            for bits, newbits, payload, scales in [(k, kn, after.k_bits, after.k_scales),
                                                   (v, vn, after.v_bits, after.v_scales)]:
                qp, sp = _quantize(np.concatenate([bits[retained, h], newbits[h:h+1]]))
                np.testing.assert_array_equal(payload[dst], qp)
                np.testing.assert_array_equal(scales[dst], sp)
        q = rng.normal(size=(q_heads, dim)).astype(np.float32)
        out = np.empty_like(q)
        store.attention_layer(0, q=q, out=out, base=base, live=new_live)
        reference = np.empty_like(q)
        for g in range(q_heads):
            h = g // (q_heads // heads)
            sl = slice(base[h], base[h] + new_live[h])
            keys = after.k_bits[sl].astype(np.float32) * after.k_scales[sl, None]
            vals = after.v_bits[sl].astype(np.float32) * after.v_scales[sl, None]
            scores = keys @ q[g] / np.sqrt(dim)
            probs = np.exp(scores - np.max(scores)); probs /= probs.sum()
            reference[g] = probs @ vals
        np.testing.assert_allclose(out, reference, rtol=3e-5, atol=3e-5)
        assert np.isfinite(out).all()
        def softmax(x):
            e = np.exp(x.astype(np.float64) - np.max(x, axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        p, qprob = softmax(reference), softmax(out)
        assert np.max(np.sum(p * np.log(p / qprob), axis=-1)) <= .05
        assert np.mean(np.argmax(out, axis=-1) == np.argmax(reference, axis=-1)) >= .9
        repeat = np.empty_like(out)
        for _ in range(16):
            store.attention_layer(0, q=q, out=repeat, base=base, live=new_live)
            np.testing.assert_array_equal(out, repeat)
        store.restore(snapshot)
        restored = store.layer_view(0)
        for h in range(heads):
            sl = slice(base[h], base[h] + cap[h])
            for name in ("k_bits", "v_bits", "k_scales", "v_scales", "positions", "evict"):
                np.testing.assert_array_equal(getattr(restored, name)[sl], getattr(view, name)[sl])
    finally:
        store.close()
