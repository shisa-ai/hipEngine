"""Device INT8 storage must preserve DMS ownership and move scales with rows."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kvcache.dms_device import DMSDevicePayloadStore
from hipengine.kernels.cpu_reference.dms import encode_dms_payload
from hipengine.kernels.registry import resolve
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


def test_int8_verify_rows_bind_per_head_variable_spans():
    """A verify chain binds [rows, kv_heads] spans through the store, not just the kernel.

    The DMS AR route attends one row: the store publishes one ``[kv_heads]``
    extent plane and sizes its split workspace for a single row. The verifier
    has to attend N verifier rows of the same request, each row reading its own
    per-head live count over the same per-head extents, so the span set is
    ``[rows, kv_heads]`` and the split workspace has to cover ``rows`` as well
    as the split count.

    The oracle is that AR route itself: the same kernel called once per row with
    that row's own ``[kv_heads]`` counts and its own one-row workspace.
    """

    heads, q_heads, dim = 2, 8, 64
    tokens, rows, window = 520, 3, 4
    slots = (tokens + rows + 4) * heads + 5
    retrofit = SimpleNamespace(
        num_layers=1,
        num_kv_heads=heads,
        num_q_heads=q_heads,
        head_dim=dim,
        window_size=window,
    )
    store = DMSDevicePayloadStore(
        retrofit=retrofit,
        slots_per_layer=slots,
        max_pack_rows=tokens + rows,
        codec="int8_per_token_head",
    )
    rng = np.random.default_rng(9301)
    try:
        k = _bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32))
        v = _bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32))
        k[0] = 0  # positive finite scales for an all-zero vector
        # Head 0 never evicts, so it retains the whole prompt. Head 1 evicts
        # every other token outside the recency window, so the two heads hold
        # different numbers of tokens: the DMS law the verifier must read.
        # Evictions stop before the window, which keeps every surviving token
        # non-evicted, so the appends below extend both heads by exactly one.
        evict = np.zeros((tokens, heads), dtype=np.uint8)
        evict[1:tokens - window - 1:2, 1] = 1
        base = np.array(
            [5 + h * (tokens + rows + 2) for h in range(heads)], dtype=np.int32
        )
        cap = np.full(heads, tokens + rows, dtype=np.int32)
        store.pack_layer(0, k, v, evict, base, cap)
        live0 = store.live_counts(0).copy()
        assert live0[0] == tokens, live0
        assert 0 < live0[1] < tokens, live0

        # The verifier appends its own draft tokens to the same extents. This
        # unit binds the span set; the retention policy for those appends is a
        # separate unit, so nothing is evicted here.
        live = live0.copy()
        for r in range(rows):
            kn = _bf16_bits(rng.normal(size=(heads, dim)).astype(np.float32))
            vn = _bf16_bits(rng.normal(size=(heads, dim)).astype(np.float32))
            store.append_layer(
                0, kn, vn, np.zeros(heads, np.uint8), tokens + r, base, cap, live
            )
            live = store.live_counts(0).copy()
        assert np.array_equal(live, live0 + rows), (live, live0)

        # Row r reads the request's extent plus the draft tokens rows 0..r wrote.
        row_live = np.stack([live0 + r + 1 for r in range(rows)]).astype(np.int32)
        row_base = np.repeat(base[None, :], rows, axis=0).astype(np.int32)
        assert np.all(row_live <= live), (row_live, live)
        q = rng.normal(size=(rows, q_heads, dim)).astype(np.float32)

        # The workspace has to cover every row, not just the widest split.
        capacity = int(row_live.max())
        splits = max(1, (capacity + 255) // 256)
        assert splits > 1, "case must exercise the split-K path"

        out = np.empty_like(q)
        store.attention_rows(0, q=q, out=out, base=row_base, live=row_live)
        assert np.isfinite(out).all(), "canary read produced non-finite output"

        # Oracle: the AR route, one row at a time, same kernel and same payload.
        reference = np.empty_like(out)
        for r in range(rows):
            store.attention_layer(
                0, q=q[r], out=reference[r], base=base, live=row_live[r]
            )
        np.testing.assert_allclose(out, reference, rtol=3e-5, atol=3e-5), {
            "per_row_ar_mismatch": {
                "max_abs_diff": float(np.max(np.abs(out - reference))),
            }
        }

        # A binding that ignored the row axis would repeat one row's answer.
        assert not np.allclose(out[0], out[-1], rtol=1e-3, atol=1e-3)

        # The device entry point is the one a graph-resident verifier uses: no
        # host staging for Q or the output, only the bound extent planes.
        base_ptr, live_ptr = store.bind_row_spans(0, base=row_base, live=row_live)
        assert store.split_workspace_bytes >= rows * q_heads * splits * dim * 4
        q_dev = malloc(q.nbytes)
        out_dev = malloc(q.nbytes)
        try:
            copy_host_to_device(q_dev, host_array_ptr(np.ascontiguousarray(q)), q.nbytes)
            store.attention_rows_device(
                0,
                q_ptr=q_dev.ptr,
                out_ptr=out_dev.ptr,
                rows=rows,
                score_capacity=capacity,
                base_ptr=base_ptr,
                live_ptr=live_ptr,
            )
            direct = np.empty_like(q)
            copy_device_to_host(host_array_ptr(direct), out_dev, direct.nbytes)
        finally:
            free(q_dev)
            free(out_dev)
        np.testing.assert_allclose(direct, reference, rtol=3e-5, atol=3e-5)

        # Repeating the binding must be deterministic.
        repeat = np.empty_like(q)
        for _ in range(8):
            store.attention_rows(0, q=q, out=repeat, base=row_base, live=row_live)
            np.testing.assert_array_equal(out, repeat)
    finally:
        store.close()


def test_int8_verify_chain_leaf_binds_dms_spans_and_gates():
    """The compact verify-chain leaf, against the AR route finished by the same gate.

    The paged verify-chain leaf reads one page table and one live count per row.
    A DMS span set is a dense extent per ``(row, kv head)`` with per-slot int8
    scale planes, so the compact layer carries its own leaf. The oracle is the
    AR route's own attention for each row, through the same gate multiply.
    """

    from hipengine.core.device import Device
    from hipengine.core.tensor import Tensor
    from hipengine.kvcache.spans import KVLiveSpans, KVScaleMetadata

    heads, q_heads, dim = 2, 8, 64
    tokens, rows, window = 300, 3, 4
    slots = (tokens + rows + 4) * heads + 5
    retrofit = SimpleNamespace(
        num_layers=1,
        num_kv_heads=heads,
        num_q_heads=q_heads,
        head_dim=dim,
        window_size=window,
    )
    store = DMSDevicePayloadStore(
        retrofit=retrofit,
        slots_per_layer=slots,
        max_pack_rows=tokens + rows,
        codec="int8_per_token_head",
    )
    rng = np.random.default_rng(9317)
    buffers: dict[str, object] = {}

    def upload(name: str, array: np.ndarray) -> object:
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes)
        buffers[name] = buf
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
        return buf

    try:
        k = _bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32))
        v = _bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32))
        k[0] = 0
        evict = np.zeros((tokens, heads), dtype=np.uint8)
        evict[1:tokens - window - 1:2, 1] = 1
        base = np.array(
            [5 + h * (tokens + rows + 2) for h in range(heads)], dtype=np.int32
        )
        cap = np.full(heads, tokens + rows, dtype=np.int32)
        store.pack_layer(0, k, v, evict, base, cap)
        live0 = store.live_counts(0).copy()
        live = live0.copy()
        for r in range(rows):
            kn = _bf16_bits(rng.normal(size=(heads, dim)).astype(np.float32))
            vn = _bf16_bits(rng.normal(size=(heads, dim)).astype(np.float32))
            store.append_layer(
                0, kn, vn, np.zeros(heads, np.uint8), tokens + r, base, cap, live
            )
            live = store.live_counts(0).copy()

        row_live = np.stack([live0 + r + 1 for r in range(rows)]).astype(np.int32)
        row_base = np.repeat(base[None, :], rows, axis=0).astype(np.int32)
        capacity = int(row_live.max())
        chunk = 256
        splits = max(1, (capacity + chunk - 1) // chunk)

        # The span set the runner hands the leaf: one extent per (row, kv head),
        # declared as [rows, layers, kv_heads]. The store's own per-layer planes
        # are what the kernel indexes, exactly as the AR route passes its own.
        base_ptr, live_ptr = store.bind_row_spans(0, base=row_base, live=row_live)
        device = Device("hip", 0)
        key_ptrs = store.layer_device_ptrs(0)
        scale_ptrs = store.layer_scale_ptrs(0)
        spans = KVLiveSpans(
            base_offsets=Tensor.from_handle(
                base_ptr, (rows, 1, heads), DType.INT32, device
            ),
            live_counts=Tensor.from_handle(
                live_ptr, (rows, 1, heads), DType.INT32, device
            ),
            max_live_count=capacity,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="per_head_variable",
            span_role="verify_chain",
            scale_metadata=KVScaleMetadata(
                scale_dtype=DType.FP32,
                k_scale=Tensor.from_handle(
                    scale_ptrs["k_scale_ptr"], (rows, 1, heads, tokens + rows), DType.FP32, device
                ),
                v_scale=Tensor.from_handle(
                    scale_ptrs["v_scale_ptr"], (rows, 1, heads, tokens + rows), DType.FP32, device
                ),
            ),
        )

        q = rng.normal(size=(rows, q_heads, dim)).astype(np.float32)
        gate_f32 = rng.normal(size=(rows, q_heads, dim)).astype(np.float32)
        gate = _bf16_bits(gate_f32)
        scale = float(dim) ** -0.5
        q_dev = upload("q", q)
        gate_dev = upload("gate", gate)
        out_dev = upload("out", np.zeros((rows, q_heads, dim), dtype=np.uint16))
        result_dev = upload("result", np.zeros_like(q))
        partial_out = upload(
            "po", np.zeros((rows * q_heads * splits, dim), dtype=np.float32)
        )
        partial_m = upload("pm", np.zeros((rows * q_heads * splits,), dtype=np.float32))
        partial_l = upload("pl", np.zeros((rows * q_heads * splits,), dtype=np.float32))

        from hipengine.kernels.hip_gfx1100.attention.dms_compact_int8 import (
            dms_compact_int8_verify_chain_gate_bf16_spans,
            register_dms_compact_int8_kernels,
        )

        register_dms_compact_int8_kernels()
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="dms_compact_attn_decode",
                quant="int8_per_token_head",
                variant="verify_chain_gate_bf16_spans",
            )
            is dms_compact_int8_verify_chain_gate_bf16_spans
        )

        dms_compact_int8_verify_chain_gate_bf16_spans(
            q_dev.ptr,
            key_ptrs[0],
            key_ptrs[1],
            scale_ptrs["k_scale_ptr"],
            scale_ptrs["v_scale_ptr"],
            gate_dev.ptr,
            out_dev.ptr,
            result_dev.ptr,
            partial_out.ptr,
            partial_m.ptr,
            partial_l.ptr,
            base_ptr,
            live_ptr,
            spans,
            rows,
            chunk,
            splits,
            q_heads,
            heads,
            dim,
            q_heads * dim,
            q_heads * dim,
            dim,
            1,
            q_heads * dim,
            dim,
            1,
            scale,
        )
        got = np.zeros((rows, q_heads, dim), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(got), out_dev, got.nbytes)

        # Oracle: the AR route, one row at a time, finished by the same gate.
        from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
            qwen35_full_attn_gate_mul_bf16,
        )

        reference = np.zeros_like(got)
        for r in range(rows):
            row_out = np.empty((q_heads, dim), dtype=np.float32)
            store.attention_layer(0, q=q[r], out=row_out, base=base, live=row_live[r])
            row_out_dev = upload(f"row{r}", row_out)
            row_gate_dev = upload(f"rgate{r}", gate[r])
            row_ref_dev = upload(f"rref{r}", np.zeros((q_heads, dim), dtype=np.uint16))
            qwen35_full_attn_gate_mul_bf16(
                row_out_dev.ptr, row_gate_dev.ptr, row_ref_dev.ptr, q_heads * dim
            )
            copy_device_to_host(
                host_array_ptr(reference[r : r + 1]), row_ref_dev, reference[r : r + 1].nbytes
            )

        got_f32 = _bf16_from_bits(got).astype(np.float32)
        ref_f32 = _bf16_from_bits(reference).astype(np.float32)
        assert np.isfinite(got_f32).all(), "canary read produced non-finite output"
        np.testing.assert_allclose(got_f32, ref_f32, rtol=3e-5, atol=3e-5), {
            "verify_chain_leaf_mismatch": {
                "max_abs_diff": float(np.max(np.abs(got_f32 - ref_f32))),
            }
        }
        # Rows must differ: a leaf that ignored the row axis would repeat one.
        assert not np.allclose(got_f32[0], got_f32[-1], rtol=1e-3, atol=1e-3)
        # The gate is applied, not skipped, and with its exact law: a zero gate
        # is sigmoid(0) = 0.5, so the output must be half the attention plane.
        zero_gate = upload("zerogate", np.zeros_like(gate))
        dms_compact_int8_verify_chain_gate_bf16_spans(
            q_dev.ptr,
            key_ptrs[0],
            key_ptrs[1],
            scale_ptrs["k_scale_ptr"],
            scale_ptrs["v_scale_ptr"],
            zero_gate.ptr,
            out_dev.ptr,
            result_dev.ptr,
            partial_out.ptr,
            partial_m.ptr,
            partial_l.ptr,
            base_ptr,
            live_ptr,
            spans,
            rows,
            chunk,
            splits,
            q_heads,
            heads,
            dim,
            q_heads * dim,
            q_heads * dim,
            dim,
            1,
            q_heads * dim,
            dim,
            1,
            scale,
        )
        gated = np.zeros_like(got)
        copy_device_to_host(host_array_ptr(gated), out_dev, gated.nbytes)
        raw = np.zeros_like(q)
        copy_device_to_host(host_array_ptr(raw), result_dev, raw.nbytes)
        assert np.isfinite(raw).all()
        np.testing.assert_allclose(
            _bf16_from_bits(gated).astype(np.float32), 0.5 * raw, rtol=1e-2, atol=1e-2
        ), {"zero_gate_is_not_half_attention": float(np.max(np.abs(_bf16_from_bits(gated).astype(np.float32) - 0.5 * raw)))}
    finally:
        for buf in buffers.values():
            free(buf)
        store.close()
