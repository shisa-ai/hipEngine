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
