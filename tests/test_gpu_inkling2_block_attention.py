import ctypes

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_block_attention import Inkling2DeviceBlockAttention


@pytest.mark.parametrize("queries,keys,heads,kv_heads,dim", [(20, 20, 16, 16, 64), (8, 37, 24, 8, 128)])
def test_noncausal_attention_matches_cpu(queries, keys, heads, kv_heads, dim):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.attention.inkling2_block_attention import launch_block_attention

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    rng = np.random.default_rng(721)
    q = rng.standard_normal((queries, heads, dim), dtype=np.float32)
    k, v = rng.standard_normal((2, keys, kv_heads, dim), dtype=np.float32)
    rk, rv = (np.repeat(a.astype(np.float64), heads // kv_heads, axis=1) for a in (k, v))
    scores = np.einsum("qhd,khd->qhk", q, rk) / np.sqrt(dim)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    ref = np.einsum("qhk,khd->qhd", p, rv)
    with Inkling2DeviceBlockAttention(runtime=get_hip_runtime(), launch=launch_block_attention) as attn:
        out = attn(q, k, v)
        np.testing.assert_allclose(out, ref, rtol=5e-6, atol=1e-6)
        np.testing.assert_array_equal(attn(q[:1], k, v), out[:1])
    np.testing.assert_array_equal(out.argmax(axis=-1), ref.argmax(axis=-1))
    rp = np.exp(ref - ref.max(axis=-1, keepdims=True)); rp /= rp.sum(axis=-1, keepdims=True)
    gp = np.exp(out.astype(np.float64) - out.max(axis=-1, keepdims=True)); gp /= gp.sum(axis=-1, keepdims=True)
    assert np.max(np.sum(rp * (np.log(rp) - np.log(gp)), axis=-1)) <= 0.05
