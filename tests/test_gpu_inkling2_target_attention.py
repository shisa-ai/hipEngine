import ctypes

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention


@pytest.mark.parametrize("capacity", [3, 9])
def test_target_attention_matches_grouped_reference_with_relative_bias_and_sink(capacity):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.attention.inkling2_target_attention import launch_target_attention

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    nq, nk, dim, extent = 6, 2, 128, 2
    owner = Inkling2DeviceAttention(runtime=get_hip_runtime(), launch=launch_target_attention,
                                    geometry=[(capacity, nq, nk, dim, extent)])
    rng = np.random.default_rng(14)
    keys, values = [], []
    try:
        for pos in range(8):
            q, k, v, bias, sink = [rng.standard_normal(shape, dtype=np.float32) for shape in
                                   ((nq, dim), (nk, dim), (nk, dim), (nq, extent), (nq,))]
            keys.append(k)
            values.append(v)
            first = max(0, pos + 1 - capacity)
            kr, vr = (np.repeat(np.stack(a[first:]).astype(np.float64), nq // nk, axis=1) for a in (keys, values))
            scores = np.einsum("hd,thd->ht", q, kr) / np.sqrt(dim)
            for i in range(min(extent, pos + 1 - first)):
                scores[:, -1 - i] += bias[:, i]
            scores = np.concatenate((scores, sink[:, None]), axis=1)
            p = np.exp(scores - scores.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            ref = np.einsum("ht,thd->hd", p[:, :-1], vr)
            out = owner.step(0, pos, q, k, v, bias, sink)
            np.testing.assert_allclose(out, ref, rtol=5e-6, atol=1e-6)
            np.testing.assert_array_equal(out.argmax(axis=-1), ref.argmax(axis=-1))
            rp = np.exp(ref - ref.max(axis=-1, keepdims=True)); rp /= rp.sum(axis=-1, keepdims=True)
            gp = np.exp(out.astype(np.float64) - out.max(axis=-1, keepdims=True)); gp /= gp.sum(axis=-1, keepdims=True)
            assert np.max(np.sum(rp * (np.log(rp) - np.log(gp)), axis=-1)) <= 0.05
    finally:
        owner.close()
    assert not owner.banks
