import ctypes

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention


@pytest.mark.parametrize("capacity", [3, 9, 33])
def test_target_attention_matches_grouped_reference_with_relative_bias_and_sink(capacity):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.attention.inkling2_target_attention import (
        launch_cache_write, launch_split_attention, launch_target_attention, launch_tiled_attention,
    )
    from hipengine.runtime.inkling2_undo import Inkling2UndoJournal

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    nq, nk, dim, extent = 6, 2, 128 if capacity != 9 else 65, 17
    runtime = get_hip_runtime()
    copies = []

    class TracedRuntime:
        def __getattr__(self, name):
            return getattr(runtime, name)

        def memcpy(self, dst, src, size, kind):
            copies.append(kind)
            runtime.memcpy(dst, src, size, kind)

    geometry = [(capacity, nq, nk, dim, extent)]
    split_rows = []

    def split(*args):
        split_rows.append(args[-1])
        launch_split_attention(*args)

    owner = Inkling2DeviceAttention(runtime=TracedRuntime(), launch=launch_tiled_attention, write=launch_cache_write,
                                    geometry=geometry, split=split)
    strict = Inkling2DeviceAttention(runtime=runtime, launch=launch_target_attention, write=launch_cache_write,
                                     geometry=geometry)
    rng = np.random.default_rng(14)
    keys, values, replay = [], [], []
    undo = Inkling2UndoJournal()
    try:
        for pos in range(capacity + 3):
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
            original = strict.step(0, pos, q, k, v, bias, sink)
            copies.clear()
            out = owner.step(0, pos, q, k, v, bias, sink, undo=undo if pos >= 3 else None)
            if pos:
                assert [kind for kind in copies if kind != HipMemcpyKind.DEVICE_TO_DEVICE] == [
                    HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST,
                ]
            if pos >= 3:
                replay.append((pos, (q, k, v, bias, sink), out))
            np.testing.assert_array_equal(out.view(np.uint32), original.view(np.uint32))
            np.testing.assert_allclose(out, ref, rtol=5e-6, atol=1e-6)
            np.testing.assert_array_equal(out.argmax(axis=-1), ref.argmax(axis=-1))
            rp = np.exp(ref - ref.max(axis=-1, keepdims=True)); rp /= rp.sum(axis=-1, keepdims=True)
            gp = np.exp(out.astype(np.float64) - out.max(axis=-1, keepdims=True)); gp /= gp.sum(axis=-1, keepdims=True)
            assert np.max(np.sum(rp * (np.log(rp) - np.log(gp)), axis=-1)) <= 0.05
        if capacity >= 16:
            assert {16, 17, capacity} <= set(split_rows)
        else:
            assert not split_rows and owner.scores == 0
        undo.finish(3)
        for pos, args, expected in replay:
            np.testing.assert_array_equal(owner.step(0, pos, *args).view(np.uint32), expected.view(np.uint32))
    finally:
        undo.finish(3)
        strict.close()
        owner.close()
    assert not owner.banks
