"""Checked-index GDN parity, chunk replay, CPU floor and buffer ownership."""

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer, copy_host_to_device, host_array_ptr
from tests import test_gpu_qwen4exp_gdn_register as base
from tests.test_gpu_qwen4_exp_gdn_hip import _download, _upload

pytestmark = pytest.mark.skipif(not base.hip_available(), reason="HIP unavailable")


@pytest.mark.parametrize("tokens", [1, 17, 64, 512, 1024, 4096])
def test_narrow_exact_replay_and_ownership(tokens):
    f = base.Fixture(tokens)
    guard = 32
    sentinel = np.uint32(0xC1234567)
    try:
        # Guard each allocation, including every read-only input. The nonzero
        # initial state and zero Q/K head are inherited from the base fixture.
        original_inputs = [
            _download(x, (x.nbytes // 4,), np.float32, f.runtime) for x in f.inputs
        ]
        guarded = []
        for i, buf in enumerate([*f.inputs, *f.states, *f.outputs]):
            bits = np.full(buf.nbytes // 4 + 2 * guard, sentinel, dtype=np.uint32)
            if i < 7:
                bits[guard:-guard] = original_inputs[i].view(np.uint32)
            elif i < 9:
                bits[guard:-guard] = f.state.reshape(-1).view(np.uint32)
            owner = _upload(bits, f.runtime, f.allocations)
            view = DeviceBuffer(owner.ptr + guard * 4, buf.nbytes)
            guarded.append((owner, view, bits))
        f.inputs = [x[1] for x in guarded[:7]]
        f.states = [x[1] for x in guarded[7:9]]
        f.outputs = [x[1] for x in guarded[9:]]
        f.run(False, fn=base.gdn.qwen4_exp_gdn_wave_norm_prefill_f32)
        expected = f.result(False)
        for split in ([None, None, 7] if tokens > 7 else [None, None]):
            owner, _, bits = guarded[10]
            copy_host_to_device(owner, host_array_ptr(bits), runtime=f.runtime)
            f.run(True, split=split, fn=base.gdn.qwen4_exp_gdn_wave_norm_narrow_prefill_f32)
            for got, ref in zip(f.result(True), expected):
                np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
            for i, (owner, view, bits) in enumerate(guarded):
                actual = _download(owner, bits.shape, np.uint32, f.runtime)
                np.testing.assert_array_equal(actual[:guard], bits[:guard])
                np.testing.assert_array_equal(actual[-guard:], bits[-guard:])
                if i < 7:
                    np.testing.assert_array_equal(actual, bits)
                elif i == 7:
                    np.testing.assert_array_equal(actual[guard:-guard], expected[1].reshape(-1).view(np.uint32))
                elif i == 9:
                    np.testing.assert_array_equal(actual[guard:-guard], expected[0].reshape(-1).view(np.uint32))
        if tokens <= 64:
            mapping = np.arange(48) % 16
            q, k = f.q[:, mapping].copy(), f.k[:, mapping].copy()
            q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
            q /= np.sqrt(np.float32(128))
            k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
            beta = 1 / (1 + np.exp(-f.beta))
            decay = np.exp(f.a * np.log1p(np.exp(f.alpha + f.dt)))
            core, state = base.gdn_prefill_recurrent_segments(
                q, k, f.v, beta, decay, f.state[None], [0, tokens], [0])
            reference = base.sigmoid_gated_rmsnorm(core, f.norm, f.gate)
            actual = f.result(True)
            np.testing.assert_allclose(actual[0], reference, rtol=5e-5, atol=5e-5)
            np.testing.assert_allclose(actual[1], state[0], rtol=5e-5, atol=5e-5)
            def log_softmax(x):
                x = x.astype(np.float64).reshape(-1, 128)
                x -= x.max(axis=-1, keepdims=True)
                return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
            lp, lq = log_softmax(reference), log_softmax(actual[0])
            assert np.max(np.sum(np.exp(lp) * (lp - lq), axis=-1)) <= 0.05
            assert np.mean(lp.argmax(-1) == lq.argmax(-1)) >= 0.90
    finally:
        f.close()


def test_state_head_isolation():
    f = base.Fixture(17)
    try:
        candidate = base.gdn.qwen4_exp_gdn_wave_norm_narrow_prefill_f32
        f.run(False, fn=candidate)
        expected = f.result(False)
        f.state[11] += np.float32(0.125)
        f.run(True, fn=candidate)
        actual = f.result(True)
        other_heads = np.arange(48) != 11
        np.testing.assert_array_equal(actual[0][:, other_heads].view(np.uint32),
                                      expected[0][:, other_heads].view(np.uint32))
        np.testing.assert_array_equal(actual[1][other_heads].view(np.uint32),
                                      expected[1][other_heads].view(np.uint32))
        assert not np.array_equal(actual[1][11], expected[1][11])
    finally:
        f.close()


def test_raw_abi_rejects_overflow_without_launch():
    library = base.gdn.build_qwen4_exp_gdn(load=True)
    fn = library.hipengine_qwen4_exp_gdn_wave_norm_narrow_prefill_f32
    fn.argtypes = (ctypes.c_void_p,) * 9 + (ctypes.c_int64,) * 5 + (ctypes.c_void_p,)
    fn.restype = ctypes.c_int
    for shape in (
        (2**31 // 10240 + 1, 16, 48, 128, 128),
        (1, 1, 2**31 // 16384, 128, 128),
        (1, 2**62, 2**62, 128, 128),
    ):
        assert fn(*([1] * 9), *shape, None) != 0
