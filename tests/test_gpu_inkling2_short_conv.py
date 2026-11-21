import ctypes

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution
from hipengine.runtime.inkling2_mini_reference import _VisitState


@pytest.mark.parametrize("paired", [False, True])
def test_device_short_convolution_preserves_separate_visit_histories(monkeypatch, paired):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    from hipengine.kernels.hip_gfx1100.quant.inkling2_short_conv import launch_short_conv

    runtime = get_hip_runtime()
    copies = []
    memcpy = type(runtime).memcpy_async
    synchronize = type(runtime).stream_synchronize

    def record_copy(self, dst, src, size, kind, stream):
        copies.append(kind)
        return memcpy(self, dst, src, size, kind, stream)

    def record_wait(self, stream):
        copies.append("wait")
        return synchronize(self, stream)

    monkeypatch.setattr(type(runtime), "memcpy_async", record_copy)
    monkeypatch.setattr(type(runtime), "stream_synchronize", record_wait)
    owner = Inkling2DeviceConvolution(runtime=runtime, launch=launch_short_conv, channels=195, width=4)
    refs = [_VisitState(None, None), _VisitState(None, None)]
    rng = np.random.default_rng(52)
    try:
        for i in range(12):
            visit = i % 2
            inputs = [rng.standard_normal(193, dtype=np.float32) for _ in range(2)]
            taps = [rng.standard_normal((193, 4), dtype=np.float32)[:, ::-1] for _ in range(2)]
            copies.clear()
            if paired:
                actual = owner.step_kv(visit, *inputs, *taps, dynamic=bool(visit))
            else:
                actual = tuple(owner.step(visit, name, x, t, dynamic=bool(visit))
                               for name, x, t in zip(("k", "v"), inputs, taps, strict=True))
            assert copies == [HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST, "wait"] * (1 if paired else 2)
            for name, x, t, gpu in zip(("k", "v"), inputs, taps, actual, strict=True):
                ref = refs[visit].convolve(name, x, t, dynamic=bool(visit))
                np.testing.assert_allclose(gpu, ref, rtol=1e-6, atol=1e-6)
                assert np.argmax(gpu) == np.argmax(ref)
                rp = np.exp(ref.astype(np.float64) - np.max(ref))
                gp = np.exp(gpu.astype(np.float64) - np.max(gpu))
                rp /= rp.sum()
                gp /= gp.sum()
                assert np.sum(rp * (np.log(rp) - np.log(gp))) <= 0.05
        assert len(owner.histories) == 4
    finally:
        owner.close()
    assert not owner.histories


def test_static_convolution_in_place_matches_separate_calls_after_rollback():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant.inkling2_short_conv import launch_short_conv
    from hipengine.runtime.inkling2_undo import Inkling2UndoJournal

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    runtime = get_hip_runtime()
    def owner():
        return Inkling2DeviceConvolution(runtime=runtime, launch=launch_short_conv, channels=33, width=3)

    baseline, chained = owner(), owner()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 33), dtype=np.float32)
    taps = rng.standard_normal((33, 3), dtype=np.float32)[:, ::-1]
    scratch = malloc(x.nbytes, runtime=runtime)
    undo = Inkling2UndoJournal()

    def invoke(rows, pos, journal=None):
        out = np.empty_like(rows)
        runtime.memcpy(scratch.ptr, rows.ctypes.data, rows.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        chained.apply_static(0, "attn", taps, scratch.ptr, rows.shape, position=pos, undo=journal)
        runtime.memcpy(out.ctypes.data, scratch.ptr, out.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        return out.view(np.uint32)

    try:
        expected = np.stack([baseline.step(0, "attn", row, taps) for row in x]).view(np.uint32)
        np.testing.assert_array_equal(invoke(x, 0, undo), expected)
        undo.finish(2)
        np.testing.assert_array_equal(invoke(x[2], 2), expected[2])
        np.testing.assert_array_equal(invoke(x[3:], 3), expected[3:])
    finally:
        undo.finish(0)
        free(scratch, runtime=runtime)
        chained.close()
        baseline.close()
