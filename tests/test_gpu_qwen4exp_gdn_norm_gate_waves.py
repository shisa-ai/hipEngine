"""The eight-wave GDN norm/gate is bit-identical to its per-pair parent.

``qwen4_exp_gdn_core_norm_gate_waves_f32`` changes only launch geometry: one
workgroup handles eight (token, head) pairs, one per wave, running the
parent's arithmetic. The contract is exact parent parity.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    build_qwen4_exp_gdn,
    qwen4_exp_gdn_core_norm_gate_f32,
    qwen4_exp_gdn_core_norm_gate_waves_f32,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _run(fn, core, gate, weight, tokens, heads, dim):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn(load=True)
    padded = np.concatenate([core.ravel(), np.full(64, 7.5, dtype=np.float32)])
    buffers = []
    try:
        devs = []
        for arr in (padded, gate, weight):
            host = np.ascontiguousarray(arr)
            dev = malloc(host.nbytes, runtime=runtime)
            buffers.append(dev)
            copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
            devs.append(dev)
        fn(devs[0].ptr, devs[1].ptr, devs[2].ptr, tokens, heads, dim,
           library=library, runtime=runtime)
        runtime.device_synchronize()
        out = np.empty_like(padded)
        copy_device_to_host(host_array_ptr(out), devs[0], runtime=runtime)
    finally:
        for dev in reversed(buffers):
            free(dev, runtime=runtime)
    assert (out[core.size:] == np.float32(7.5)).all()
    return out[: core.size].reshape(core.shape)


@pytest.mark.parametrize(
    "tokens,heads,dim",
    [(1, 48, 128), (5, 48, 128), (37, 32, 128), (3, 7, 96), (16, 48, 128)],
)
def test_waves_kernel_is_bit_identical_to_the_per_pair_parent(tokens, heads, dim) -> None:
    rng = np.random.default_rng(tokens * 131 + heads)
    core = (rng.standard_normal((tokens, heads * dim)) * 3.0).astype(np.float32)
    gate = rng.standard_normal((tokens, heads * dim)).astype(np.float32)
    weight = (1.0 + 0.1 * rng.standard_normal(dim)).astype(np.float32)
    parent = _run(qwen4_exp_gdn_core_norm_gate_f32, core, gate, weight, tokens, heads, dim)
    waves = _run(qwen4_exp_gdn_core_norm_gate_waves_f32, core, gate, weight, tokens, heads, dim)
    assert np.isfinite(parent).all()
    assert np.array_equal(parent.view(np.uint32), waves.view(np.uint32))
    # Sanity against the definition (not bit-exact: rsqrt and sigmoid differ).
    blocks = core.reshape(tokens, heads, dim).astype(np.float64)
    rms = np.sqrt((blocks ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    expected = blocks / rms * weight * (1.0 / (1.0 + np.exp(-gate.reshape(tokens, heads, dim).astype(np.float64))))
    np.testing.assert_allclose(parent.reshape(tokens, heads, dim), expected, rtol=2e-4, atol=1e-5)
