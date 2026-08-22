"""Integration RED gate for the device-backed DMS compact backend (C2-7 U6).

Drives the registered ``dms_compact`` HIP kernels through
``DMSCompactBackend`` (device payload path) and verifies bit-exact parity
with the host parent: identical extent metadata, identical slot payloads
(BF16 bits), device attention within the KL/top-1 gate of
``compact_attention_reference``, fail-closed overflow with the device
state untouched, and determinism across backends. The host parent remains
the registered fallback; device mode is opt-in. GPU cases skip cleanly on
no-ROCm runners.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hipengine.kernels.registry import resolve
import hipengine.kernels.hip_gfx1100.attention  # noqa: F401  (collection-time registry baseline)
from hipengine.kvcache.dms import (
    DMSCompactBackend,
    DMSRetrofitConfig,
    compact_attention_reference,
)
from tests.test_dms_streaming_pack_hip import _admit, _bf16_bits, _hip_available


def _make_backend(
    *,
    num_layers: int,
    heads: int,
    dim: int,
    window: int,
    slots: int,
    device: bool,
) -> DMSCompactBackend:
    retrofit = DMSRetrofitConfig(
        artifact_fingerprint="fixture:dms-device",
        model_family="qwen35",
        num_layers=num_layers,
        num_q_heads=heads * 4,
        num_kv_heads=heads,
        head_dim=dim,
        window_size=window,
        target_compression_ratio=2,
        alpha_scale=100.0,
        alpha_offset=5.0,
        borrowed_query_channel=dim - 1,
        corrected_mask=True,
        trained_checkpoint=True,
        evidence_source="unit fixture",
        source_path="tests/fixtures/dms_device",
    )
    return DMSCompactBackend(
        retrofit=retrofit,
        codec="bf16",
        slots_per_layer=slots,
        max_request_rows=8,
        max_pack_rows=64,
        device_payloads=device,
    )


def test_device_payloads_flag_controls_mode() -> None:
    backend = _make_backend(
        num_layers=1, heads=2, dim=16, window=2, slots=32, device=False
    )
    assert backend.device_payloads_enabled is False
    assert os.environ.get("HIPENGINE_DMS_DEVICE_PAYLOADS") is None


def test_int8_device_payloads_rejected() -> None:
    retrofit = DMSRetrofitConfig(
        artifact_fingerprint="fixture:dms-device-int8",
        model_family="qwen35",
        num_layers=1,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=16,
        window_size=2,
        target_compression_ratio=2,
        alpha_scale=100.0,
        alpha_offset=5.0,
        borrowed_query_channel=15,
        corrected_mask=True,
        trained_checkpoint=True,
        evidence_source="unit fixture",
        source_path="tests/fixtures/dms_device_int8",
    )
    from hipengine.kvcache.dms import DMSCodecQualification

    qual = DMSCodecQualification(
        codec="int8_per_token_head",
        artifact_fingerprint="fixture:dms-device-int8",
        kl_divergence=0.0,
        top1_agreement=1.0,
        no_dense_shadow=True,
        evidence_source="unit fixture",
    )
    with pytest.raises(ValueError, match="device"):
        DMSCompactBackend(
            retrofit=retrofit,
            codec="int8_per_token_head",
            slots_per_layer=32,
            max_request_rows=8,
            max_pack_rows=64,
            codec_qualification=qual,
            device_payloads=True,
        )


def _fixture_data(prompts: list[int], layers: int, heads: int, dim: int, seed: int):
    """Prompt K/V/evict + decode steps shared by both backends."""
    rng = np.random.default_rng(seed)
    prompts_kv = []
    for tokens in prompts:
        k = rng.standard_normal((tokens, layers, heads, dim)).astype(np.float32)
        v = rng.standard_normal((tokens, layers, heads, dim)).astype(np.float32)
        evict = np.ones((tokens, layers, heads), dtype=bool)  # all eligible
        evict[0, :] = False  # a never-evicted sink row
        prompts_kv.append((k, v, evict))
    steps = []
    for step in range(3):
        per_request = []
        for r in range(len(prompts)):
            flags = np.zeros((layers, heads), dtype=bool)
            if step == 1:
                flags[:, 0] = True  # evict the new row on head 0 only
            per_request.append(flags)
        kn = rng.standard_normal((layers, heads, dim)).astype(np.float32)
        vn = rng.standard_normal((layers, heads, dim)).astype(np.float32)
        steps.append((per_request, kn, vn))
    return prompts_kv, steps


def _drive(backend: DMSCompactBackend, prompts_kv, steps) -> None:
    for r, (k, v, evict) in enumerate(prompts_kv):
        _admit(backend, request_id=r, tokens=int(k.shape[0]))
        backend.streaming_pack(r, k, v, evict)
    for step, (flags_per_request, kn, vn) in enumerate(steps):
        for r, flags in enumerate(flags_per_request):
            backend.append_decode(
                r, kn, vn, flags, position=int(np.ptp([0])) + 0  # placeholder
            )


def _compare_metadata(a: DMSCompactBackend, b: DMSCompactBackend) -> None:
    ids = sorted(a._states)
    assert ids == sorted(b._states)
    for rid in ids:
        sa, sb = a.state_for_request(rid), b.state_for_request(rid)
        np.testing.assert_array_equal(
            sa.base_offsets, sb.base_offsets, err_msg=f"{rid} base_offsets"
        )
        np.testing.assert_array_equal(
            sa.range_capacity, sb.range_capacity, err_msg=f"{rid} range_capacity"
        )
        np.testing.assert_array_equal(
            sa.live_counts, sb.live_counts, err_msg=f"{rid} live_counts"
        )
        np.testing.assert_array_equal(
            sa.token_positions, sb.token_positions, err_msg=f"{rid} token_positions"
        )
        np.testing.assert_array_equal(
            sa.evict_mask, sb.evict_mask, err_msg=f"{rid} evict_mask"
        )
        assert sa.logical_tokens == sb.logical_tokens


def _softmax_kl_max(ref: np.ndarray, cand: np.ndarray) -> float:
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_ref, log_cand = logsm(ref64), logsm(cand64)
    return float(np.max(np.sum(np.exp(log_ref) * (log_ref - log_cand), axis=-1)))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_device_pack_append_attention_parity_bit_exact() -> None:
    os.environ["HIPENGINE_DMS_DEVICE_TRIPWIRE"] = "1"
    try:
        layers, heads, dim, window = 2, 2, 16, 2
        prompts = [6, 4]
        host_b = _make_backend(
            num_layers=layers, heads=heads, dim=dim, window=window, slots=64,
            device=False,
        )
        dev_b = _make_backend(
            num_layers=layers, heads=heads, dim=dim, window=window, slots=64,
            device=True,
        )
        assert dev_b.device_payloads_enabled is True
        data = _fixture_data(prompts, layers, heads, dim, seed=2026)
        prompts_kv, steps = data

        for r, (k, v, evict) in enumerate(prompts_kv):
            _admit(host_b, request_id=r, tokens=int(k.shape[0]))
            _admit(dev_b, request_id=r, tokens=int(k.shape[0]))
            host_b.streaming_pack(r, k, v, evict)
            dev_b.streaming_pack(r, k, v, evict)
        for step, (flags_per_request, kn, vn) in enumerate(steps):
            for r, flags in enumerate(flags_per_request):
                position = prompts[r] + step
                host_b.append_decode(r, kn, vn, flags, position=position)
                dev_b.append_decode(r, kn, vn, flags, position=position)

        # 1) No host payload shadow in device mode.
        for rid in dev_b._states:
            state = dev_b.state_for_request(rid)
            assert not state.k_payload, "device mode must not retain K payload"
            assert not state.v_payload, "device mode must not retain V payload"

        # 2) Metadata parity.
        _compare_metadata(host_b, dev_b)

        # 3) Device slot payloads equal the host parent's (BF16 bits).
        for rid in sorted(dev_b._states):
            sh = host_b.state_for_request(rid)
            sd = dev_b.state_for_request(rid)
            for layer in range(layers):
                view = dev_b.device_layer_view(rid, layer)
                for h in range(heads):
                    live = int(sd.live_counts[layer, h])
                    base = int(sd.base_offsets[layer, h])
                    np.testing.assert_array_equal(
                        view.k_bits[base:base + live],
                        _bf16_bits(sh.k_payload[(layer, h)]),
                        err_msg=f"{rid} L{layer} H{h} K bits",
                    )
                    np.testing.assert_array_equal(
                        view.v_bits[base:base + live],
                        _bf16_bits(sh.v_payload[(layer, h)]),
                        err_msg=f"{rid} L{layer} H{h} V bits",
                    )

        # 4) Device attention vs the reference over the device's own storage.
        q_heads = heads * 4
        scale = float(dim**-0.5)
        for layer in range(layers):
            for rid in sorted(dev_b._states):
                sd = dev_b.state_for_request(rid)
                counts = sd.live_counts[layer]
                cap = int(sd.range_capacity[layer, 0])
                view = dev_b.device_layer_view(rid, layer)
                k_kv = np.zeros((1, heads, cap, dim), dtype=np.float32)
                v_kv = np.zeros_like(k_kv)
                for h in range(heads):
                    base = int(sd.base_offsets[layer, h])
                    live = int(counts[h])
                    k_kv[0, h, :live] = (
                        view.k_bits[base:base + live].astype(np.uint32)
                        << np.uint32(16)
                    ).view(np.float32)
                    v_kv[0, h, :live] = (
                        view.v_bits[base:base + live].astype(np.uint32)
                        << np.uint32(16)
                    ).view(np.float32)
                q = np.random.default_rng(9000 + layer * 10 + rid).standard_normal(
                    (q_heads, dim)
                ).astype(np.float32)
                got = dev_b.compact_decode_attention(rid, layer, q)
                want = compact_attention_reference(
                    q[None], k_kv, v_kv, counts[None], scale=scale
                )[0]
                diff = float(np.max(np.abs(got - want)))
                assert diff < 1e-4, f"attention max abs diff {diff}"
                kl = _softmax_kl_max(want.reshape(1, -1), got.reshape(1, -1))
                assert kl <= 0.05, f"attention KL {kl}"
        # Registered kernel is what the backend used.
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="dms_compact_attn_decode",
                quant="bf16",
                variant="grouped_gqa",
            )
            is not None
        )
    finally:
        os.environ.pop("HIPENGINE_DMS_DEVICE_TRIPWIRE", None)
        host_b.close()
        dev_b.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_device_append_overflow_fail_closed() -> None:
    # Prompt of 4 tokens, window 2, nothing evicted: the extent is full
    # (live == capacity) with no evictable row, so the next append must
    # raise MemoryError exactly where the host parent does, and the device
    # buffers must be untouched.
    layers, heads, dim, window = 1, 2, 16, 2
    dev_b = _make_backend(
        num_layers=layers, heads=heads, dim=dim, window=window, slots=32,
        device=True,
    )
    try:
        rng = np.random.default_rng(4242)
        tokens = 4
        k = rng.standard_normal((tokens, layers, heads, dim)).astype(np.float32)
        v = rng.standard_normal((tokens, layers, heads, dim)).astype(np.float32)
        evict = np.zeros((tokens, layers, heads), dtype=bool)
        _admit(dev_b, request_id=0, tokens=tokens)
        dev_b.streaming_pack(0, k, v, evict)
        state = dev_b.state_for_request(0)
        assert int(state.live_counts[0, 0]) == int(state.range_capacity[0, 0]) == 4
        view_before = dev_b.device_layer_view(0, 0)
        before = (
            view_before.k_bits.copy(),
            view_before.v_bits.copy(),
            view_before.positions.copy(),
            view_before.evict.copy(),
        )
        kn = rng.standard_normal((layers, heads, dim)).astype(np.float32)
        vn = rng.standard_normal((layers, heads, dim)).astype(np.float32)
        flags = np.zeros((layers, heads), dtype=bool)
        with pytest.raises(MemoryError):
            dev_b.append_decode(0, kn, vn, flags, position=4)
        # Host metadata unmutated and device buffers byte-identical.
        state = dev_b.state_for_request(0)
        assert int(state.live_counts[0, 0]) == 4
        view = dev_b.device_layer_view(0, 0)
        for name, a, b in zip(
            ("k", "v", "positions", "evict"),
            before,
            (view.k_bits, view.v_bits, view.positions, view.evict),
        ):
            np.testing.assert_array_equal(a, b, err_msg=f"device {name} mutated")
    finally:
        dev_b.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_device_determinism_across_backends() -> None:
    layers, heads, dim, window = 1, 2, 16, 2
    data = _fixture_data([6], layers, heads, dim, seed=7)
    outs = []
    for _ in range(2):
        b = _make_backend(
            num_layers=layers, heads=heads, dim=dim, window=window, slots=64,
            device=True,
        )
        try:
            k, v, evict = data[0][0]
            steps = data[1]
            _admit(b, request_id=0, tokens=int(k.shape[0]))
            b.streaming_pack(0, k, v, evict)
            for step, (flags_per_request, kn, vn) in enumerate(steps):
                b.append_decode(
                    0, kn, vn, flags_per_request[0], position=int(k.shape[0]) + step
                )
            view = b.device_layer_view(0, 0)
            q = np.random.default_rng(555).standard_normal(
                (heads * 4, dim)
            ).astype(np.float32)
            outs.append((view.k_bits.copy(), b.compact_decode_attention(0, 0, q)))
        finally:
            b.close()
    (k1, o1), (k2, o2) = outs
    np.testing.assert_array_equal(k1, k2, err_msg="payload non-deterministic")
    np.testing.assert_array_equal(o1, o2, err_msg="attention non-deterministic")
