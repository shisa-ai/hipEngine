"""GPU tests for the VibeVoice Qwen2 backbone device path (tiny geometry)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import vibevoice_qwen2 as q2

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_OK = True
except OSError:  # pragma: no cover - no ROCm in the environment
    _HIP_OK = False

TINY = q2.Qwen2Geometry(
    hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    intermediate_size=5,
    vocab_size=11,
    rope_theta=100.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)


def _fake_checkpoint_payloads(geom: q2.Qwen2Geometry, seed: int = 13):
    rng = np.random.default_rng(seed)
    payloads = {}
    for name, shape in q2.expected_weight_shapes(geom).items():
        value = (rng.standard_normal(shape) * 0.3).astype(np.float32)
        bits = value.view(np.uint32) >> 16
        lo = (bits & 0xFF).astype(np.uint8)
        hi = (bits >> 8).astype(np.uint8)
        payloads[q2.PREFIX + name] = np.stack([lo, hi], axis=-1).tobytes()
    return payloads


@pytest.mark.skipif(not _HIP_OK, reason="ROCm/HIP runtime not available")
def test_gpu_prefill_and_decode_match_cpu_reference():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.models.vibevoice_lm import VibeVoiceQwen2Device

    geom = TINY
    payloads = _fake_checkpoint_payloads(geom)
    weights = q2.load_backbone_weights(payloads.get, geom)
    tokens = np.array([2, 9, 4, 7, 1])

    joint_hidden, _ = q2.forward_hidden_states(tokens, weights, geom)
    joint_logits, _ = q2.forward_logits(tokens, weights, geom)

    runtime = get_hip_runtime()
    device = VibeVoiceQwen2Device.from_checkpoint(
        None, runtime=runtime, geometry=geom, max_context=64,
        read_tensor=payloads.get,
    )
    try:
        got_hidden = device.forward_hidden(tokens)
        assert np.allclose(got_hidden, joint_hidden, atol=2e-2, rtol=2e-2), (
            np.abs(got_hidden - joint_hidden).max()
        )
        got_logits = device.logits(got_hidden)
        assert np.allclose(got_logits, joint_logits, atol=6e-2, rtol=2e-2)

        # Incremental decode from a 3-token prefill must track the reference.
        device.reset()
        first = device.forward_hidden(tokens[:3])
        assert np.allclose(first, joint_hidden[:3], atol=2e-2, rtol=2e-2)
        step4 = device.decode_step(int(tokens[3]))
        ref4, _ = q2.forward_hidden_states(tokens[:4], weights, geom)
        assert np.allclose(step4, ref4[3], atol=3e-2, rtol=2e-2)
        step5 = device.decode_step(int(tokens[4]))
        ref5, _ = q2.forward_hidden_states(tokens, weights, geom)
        assert np.allclose(step5, ref5[4], atol=3e-2, rtol=2e-2)
        assert device.length == 5

        # A second prefill after reset reproduces the joint forward.
        device.reset()
        again = device.forward_hidden(tokens)
        assert np.allclose(again, joint_hidden, atol=2e-2, rtol=2e-2)
    finally:
        device.close()


@pytest.mark.skipif(not _HIP_OK, reason="ROCm/HIP runtime not available")
def test_gpu_prefill_chunked_equivalence():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.models.vibevoice_lm import VibeVoiceQwen2Device

    geom = TINY
    payloads = _fake_checkpoint_payloads(geom, seed=17)
    weights = q2.load_backbone_weights(payloads.get, geom)
    tokens = np.array([5, 3, 8, 6, 2, 10])
    joint, _ = q2.forward_hidden_states(tokens, weights, geom)

    runtime = get_hip_runtime()
    device = VibeVoiceQwen2Device.from_checkpoint(
        None, runtime=runtime, geometry=geom, max_context=64,
        read_tensor=payloads.get,
    )
    try:
        # 2-token then 1-token prefills must compose to the joint result.
        device.forward_hidden(tokens[:2])
        # decode_step for token 2 (position 2, cache length 2)
        h3 = device.decode_step(int(tokens[2]))
        ref3, _ = q2.forward_hidden_states(tokens[:3], weights, geom)
        assert np.allclose(h3, ref3[2], atol=3e-2, rtol=2e-2)
    finally:
        device.close()
