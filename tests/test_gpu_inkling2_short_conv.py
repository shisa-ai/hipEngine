import ctypes
from contextlib import closing

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution
from hipengine.runtime.inkling2_mini_reference import _VisitState


def test_static_block_convolution_uses_one_launch_and_preserves_history():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv as conv

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    runtime = get_hip_runtime()
    block_sizes = []

    def launch_block(*args):
        block_sizes.append(args[-1])
        conv.launch_short_conv_block(*args)

    rng = np.random.default_rng(19)
    x = rng.normal(size=(13, 131)).astype(np.float32)
    taps = rng.normal(size=(131, 4)).astype(np.float32)[:, ::-1]
    stage = malloc(x.nbytes, runtime=runtime)
    try:
        with (
            closing(Inkling2DeviceConvolution(runtime=runtime, launch=conv.launch_short_conv,
                                             channels=131, width=4)) as serial,
            closing(Inkling2DeviceConvolution(runtime=runtime, launch=conv.launch_short_conv, launch_block=launch_block,
                                             channels=131, width=4)) as batched,
        ):
            expected = np.stack([serial.step(0, "attn", row, taps) for row in x])
            actual = np.empty_like(x)
            runtime.memcpy(stage.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            for i, j in ((0, 2), (2, 7), (7, 12)):
                batched.apply_static(0, "attn", taps, stage.ptr + i * x[0].nbytes, x[i:j].shape, position=i)
            runtime.memcpy(actual.ctypes.data, stage.ptr, x.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            actual[-1] = batched.step(0, "attn", x[-1], taps, position=12)
            assert block_sizes == [2, 5, 5]
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    finally:
        free(stage, runtime=runtime)


@pytest.mark.parametrize("dynamic", [False, True])
def test_kv_block_convolution_matches_scalar_across_chunks(dynamic):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv as conv

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    runtime = get_hip_runtime()
    rng = np.random.default_rng(21)
    xs = [rng.standard_normal((11, 137), dtype=np.float32) for _ in range(2)]
    ts = [rng.standard_normal(((11,) if dynamic else ()) + (137, 3), dtype=np.float32)[..., ::-1]
          for _ in range(2)]
    owner = dict(runtime=runtime, launch=conv.launch_short_conv, channels=137, width=3)
    with (
        closing(Inkling2DeviceConvolution(**owner)) as serial,
        closing(Inkling2DeviceConvolution(**owner, launch_block=conv.launch_short_conv_block)) as batch,
    ):
        expected, actual = [], []
        for i in range(11):
            expected.append(serial.step_kv(0, *(x[i] for x in xs), *(t[i] if dynamic else t for t in ts),
                                           dynamic=dynamic, position=i))
        for i, j in ((0, 2), (2, 9), (9, 10)):
            result = batch.step_kv(0, *(x[i:j] for x in xs), *(t[i:j] if dynamic else t for t in ts),
                                   dynamic=dynamic, position=i)
            actual.extend(zip(*result, strict=True))
        actual.append(batch.step_kv(0, *(x[-1] for x in xs), *(t[-1] if dynamic else t for t in ts),
                                    dynamic=dynamic, position=10))
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32))


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
