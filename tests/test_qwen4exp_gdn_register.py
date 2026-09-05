import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr
from hipengine.kernels.cpu_reference import gdn_prefill_recurrent_segments
from hipengine.kernels.cpu_reference.qwen4_exp import sigmoid_gated_rmsnorm
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from tests.test_qwen4_exp_gdn_hip import _alloc, _download, _upload


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


class Fixture:
    def __init__(self, tokens):
        self.runtime = get_hip_runtime()
        self.allocations = []
        self.tokens = tokens
        rng = np.random.default_rng(8305)
        self.q = rng.normal(0, 0.05, (tokens, 16, 128)).astype(np.float32)
        self.k = rng.normal(0, 0.05, (tokens, 16, 128)).astype(np.float32)
        self.q[:, 0] = 0
        self.k[:, 0] = 0
        self.v = rng.normal(0, 0.05, (tokens, 48, 128)).astype(np.float32)
        conv = np.concatenate(
            (self.q.reshape(tokens, -1), self.k.reshape(tokens, -1), self.v.reshape(tokens, -1)),
            axis=1,
        )
        self.gate = rng.normal(0, 0.5, (tokens, 48, 128)).astype(np.float32)
        self.alpha = rng.normal(-0.2, 0.1, (tokens, 48)).astype(np.float32)
        self.beta = rng.normal(0, 0.2, (tokens, 48)).astype(np.float32)
        self.dt = rng.normal(-1, 0.1, 48).astype(np.float32)
        self.a = -np.exp(rng.normal(-0.5, 0.1, 48)).astype(np.float32)
        self.norm = rng.normal(1, 0.05, 128).astype(np.float32)
        self.state = rng.normal(0, 0.01, (48, 128, 128)).astype(np.float32)
        self.inputs = [
            _upload(x, self.runtime, self.allocations)
            for x in (conv, self.gate, self.alpha, self.beta, self.dt, self.a, self.norm)
        ]
        self.states = [_upload(self.state, self.runtime, self.allocations) for _ in range(2)]
        self.outputs = [
            _alloc((tokens, 48, 128), np.float32, self.runtime, self.allocations) for _ in range(2)
        ]
        self.library = gdn.build_qwen4_exp_gdn(load=True)

    def run(self, candidate, split=None):
        idx = int(candidate)
        copy_host_to_device(self.states[idx], host_array_ptr(self.state), runtime=self.runtime)
        fn = gdn.qwen4_exp_gdn_register_prefill_f32 if candidate else gdn.qwen4_exp_gdn_prefill_f32
        for lo, hi in [(0, self.tokens)] if split is None else [(0, split), (split, self.tokens)]:
            ptrs = [x.ptr for x in self.inputs]
            for i, width in enumerate((10240, 6144, 48, 48)):
                ptrs[i] += lo * width * 4
            fn(
                *ptrs,
                self.states[idx].ptr,
                self.outputs[idx].ptr + lo * 6144 * 4,
                hi - lo,
                16,
                48,
                128,
                128,
                library=self.library,
                runtime=self.runtime,
            )
        self.runtime.device_synchronize()

    def result(self, candidate):
        i = int(candidate)
        return (
            _download(self.outputs[i], (self.tokens, 48, 128), np.float32, self.runtime),
            _download(self.states[i], self.state.shape, np.float32, self.runtime),
        )

    def close(self):
        for p in reversed(self.allocations):
            free(p, runtime=self.runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("tokens", [1, 17, 64, 512])
def test_register_gdn_exact(tokens):
    assert hasattr(gdn, "qwen4_exp_gdn_register_prefill_f32")
    f = Fixture(tokens)
    try:
        f.run(False)
        parent = f.result(False)
        for _ in range(2):
            f.run(True)
            for a, b in zip(f.result(True), parent):
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        if tokens > 1:
            f.run(True, split=7)
            for a, b in zip(f.result(True), parent):
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        if tokens <= 64:
            mapping = np.arange(48) % 16
            q = f.q[:, mapping].copy()
            k = f.k[:, mapping].copy()
            q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
            q /= np.sqrt(np.float32(128))
            k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
            beta = 1 / (1 + np.exp(-f.beta))
            decay = np.exp(f.a * np.log1p(np.exp(f.alpha + f.dt)))
            core, state = gdn_prefill_recurrent_segments(
                q, k, f.v, beta, decay, f.state[None], [0, tokens], [0]
            )
            expected = sigmoid_gated_rmsnorm(core, f.norm, f.gate)
            np.testing.assert_allclose(parent[0], expected, rtol=5e-5, atol=5e-5)
            np.testing.assert_allclose(parent[1], state[0], rtol=5e-5, atol=5e-5)
            logits = parent[0].astype(np.float64).reshape(-1, 128)
            reference = expected.astype(np.float64).reshape(-1, 128)

            def log_softmax(x):
                x = x - x.max(axis=-1, keepdims=True)
                return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

            lp, lq = log_softmax(reference), log_softmax(logits)
            kl = np.sum(np.exp(lp) * (lp - lq), axis=-1)
            assert float(kl.max()) <= 0.05
            assert np.mean(logits.argmax(-1) == reference.argmax(-1)) >= 0.90
    finally:
        f.close()


def test_register_gdn_registry_and_shape_guards():
    from hipengine.kernels.registry import resolve

    gdn.register_qwen4_exp_gdn_kernels()
    for variant, fn in (
        ("qwen4exp_sigmoid_register_prefill", gdn.qwen4_exp_gdn_register_prefill_f32),
        ("qwen4exp_sigmoid_strict_prefill", gdn.qwen4_exp_gdn_prefill_f32),
    ):
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="gdn_recurrence_norm_gate",
                quant="f32_state",
                variant=variant,
            )
            is fn
        )
    for dk, dv in ((64, 128), (128, 64)):
        with pytest.raises(ValueError, match="Dk=Dv=128"):
            gdn.qwen4_exp_gdn_register_prefill_f32(*([0] * 9), 16, 16, 48, dk, dv)
