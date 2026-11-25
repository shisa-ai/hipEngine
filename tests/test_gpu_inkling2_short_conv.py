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
            exchanged = np.concatenate([batched.step_static_block(3, "mlp", x[i:j], taps, position=i)
                                        for i, j in ((0, 3), (3, 10), (10, 13))])
            np.testing.assert_array_equal(exchanged.view(np.uint32), expected.view(np.uint32))
            assert block_sizes == [2, 5, 5, 3, 7, 3]
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


@pytest.mark.parametrize("heads,dim,width", [(1, 33, 1), (3, 67, 3), (8, 128, 4)])
def test_projected_dynamic_convolution_matches_expanded_taps(heads, dim, width):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from functools import partial

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv as conv

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    arch = backend.removeprefix("hip_")
    runtime = get_hip_runtime()
    channels = heads * dim
    rng = np.random.default_rng(21627)
    x = rng.standard_normal((12, channels), dtype=np.float32)
    fixed = rng.standard_normal((heads, dim, width), dtype=np.float32)
    dyn = rng.standard_normal((12, heads, width), dtype=np.float32)
    expanded = (fixed + dyn[:, :, None]).reshape(12, channels, width)
    owner = dict(runtime=runtime, channels=channels, width=width,
                 launch=partial(conv.launch_short_conv, target_arch=arch))
    scratch = malloc(x.nbytes + dyn.nbytes, runtime=runtime)
    try:
        with (
            closing(Inkling2DeviceConvolution(**owner)) as serial,
            closing(Inkling2DeviceConvolution(**owner,
                launch_dynamic=partial(conv.launch_short_conv_dynamic, target_arch=arch))) as chained,
        ):
            cpu = _VisitState(None, None)
            for slot, role in ((0, "k"), (1, "k"), (0, "v")):
                expected = np.stack([serial.step(slot, role, row, taps, dynamic=True)
                                     for row, taps in zip(x, expanded, strict=True)])
                actual = np.empty_like(expected)
                taps_ptr = scratch.ptr + x.nbytes
                runtime.memcpy(scratch.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                runtime.memcpy(taps_ptr, dyn.ctypes.data, dyn.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                for i, j in ((0, 1), (1, 8), (8, 11), (11, 12)):
                    shape = x[i].shape if j - i == 1 else x[i:j].shape
                    tap_shape = (heads * width,) if j - i == 1 else (j - i, heads * width)
                    chained.apply_dynamic(slot, role, fixed.reshape(channels, width), heads,
                                          (scratch.ptr + i * x[0].nbytes, shape),
                                          (taps_ptr + i * dyn[0].nbytes, tap_shape))
                runtime.memcpy(actual.ctypes.data, scratch.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                if slot == 0:
                    ref = np.stack([cpu.convolve(role, row, taps, dynamic=True)
                                    for row, taps in zip(x, expanded, strict=True)])
                    np.testing.assert_allclose(actual, ref, rtol=2e-6, atol=2e-6)
    finally:
        free(scratch, runtime=runtime)


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


def test_shared_static_taps_outlive_one_session_without_sharing_history(monkeypatch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant.inkling2_short_conv import launch_short_conv
    from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolutionTaps

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    runtime = get_hip_runtime()
    copies = []
    base_copy = type(runtime).memcpy

    def record(self, dst, src, size, kind):
        if kind == HipMemcpyKind.HOST_TO_DEVICE:
            copies.append(dst)
        return base_copy(self, dst, src, size, kind)

    monkeypatch.setattr(type(runtime), "memcpy", record)
    taps = np.arange(99, dtype=np.float32).reshape(33, 3) / 127
    rng = np.random.default_rng(727)
    x, y, z = rng.standard_normal((3, 33), dtype=np.float32)
    arch = backend.removeprefix("hip_")

    def launch(*args):
        launch_short_conv(*args, target_arch=arch)

    def session(cache=None):
        return Inkling2DeviceConvolution(runtime=runtime, launch=launch, channels=33, width=3, tap_cache=cache)

    slot, name = 1, "mlp"
    buf = malloc(x.nbytes, runtime=runtime)
    try:
        with closing(Inkling2DeviceConvolutionTaps(runtime)) as cache, closing(session(cache)) as second, \
                closing(session()) as control:
            def apply(owner, row):
                out = np.empty_like(row)
                runtime.memcpy(buf.ptr, row.ctypes.data, row.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                owner.apply_static(slot, name, taps, buf.ptr, row.shape)
                runtime.memcpy(out.ctypes.data, buf.ptr, out.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                return out.view(np.uint32)

            with closing(session(cache)) as first:
                apply(first, x)
                got = apply(second, y)
                assert first.histories[slot, name].ptr != second.histories[slot, name].ptr
                shared_ptr = first._static_tap_ptr(slot, name, taps)
                assert second._static_tap_ptr(slot, name, taps) == shared_ptr
            np.testing.assert_array_equal(got, apply(control, y))
            np.testing.assert_array_equal(apply(second, z), apply(control, z))
            assert copies.count(shared_ptr) == 1
        with pytest.raises(RuntimeError, match="taps are closed"):
            cache.pointer(slot, name, taps)
    finally:
        free(buf, runtime=runtime)


def _conv_device():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import resolve_backend

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return get_hip_runtime(), backend.removeprefix("hip_")


def _upload(array, runtime):
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import malloc

    array = np.ascontiguousarray(array, dtype=np.float32)
    buffer = malloc(array.nbytes, runtime=runtime)
    runtime.memcpy(buffer.ptr, array.ctypes.data, array.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
    return buffer, array


def _download(buffer, like, runtime):
    from hipengine.core.hip import HipMemcpyKind

    host = np.empty_like(like)
    runtime.memcpy(host.ctypes.data, buffer.ptr, host.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    return host


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 256])
@pytest.mark.parametrize("channels", [1024, 2304, 100])
@pytest.mark.parametrize("width", [4, 1, 3])
@pytest.mark.parametrize("per_row", [False, True])
def test_row_block_convolution_is_the_serial_kernel_bit_for_bit(rows, channels, width, per_row):
    from hipengine.core.memory import free
    from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv as conv

    runtime, arch = _conv_device()
    rng = np.random.default_rng(1000 + rows + channels + width)
    # Inputs large enough that every partial sum rounds: the order of the terms is what is compared.
    history = rng.standard_normal((channels, width), dtype=np.float32) * 1e3
    x = rng.standard_normal((rows, channels), dtype=np.float32) * 1e2
    taps = rng.standard_normal((rows, channels, width) if per_row else (channels, width), dtype=np.float32)
    scale = np.float32(1 / np.sqrt(width + 1))
    serial_history, _ = _upload(history, runtime)
    row_history, _ = _upload(history, runtime)
    source, _ = _upload(x, runtime)
    tap_buffer, _ = _upload(taps, runtime)
    serial_out, _ = _upload(np.zeros_like(x), runtime)
    row_out, _ = _upload(np.zeros_like(x), runtime)
    try:
        stride = channels * width if per_row else 0
        conv._serial_symbol("block", arch)(serial_history.ptr, source.ptr, tap_buffer.ptr, serial_out.ptr, channels,
                                           width, scale, rows, stride, 0)
        conv.launch_short_conv_block(row_history.ptr, source.ptr, tap_buffer.ptr, row_out.ptr, channels, width, scale,
                                     rows, dynamic_taps=per_row, target_arch=arch)
        runtime.stream_synchronize(0)
        expected = _download(serial_out, x, runtime)
        actual = _download(row_out, x, runtime)
        assert np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)) == 0
        expected_history = _download(serial_history, history, runtime)
        actual_history = _download(row_history, history, runtime)
        assert np.count_nonzero(actual_history.view(np.uint32) != expected_history.view(np.uint32)) == 0
        # A block leaves the newest inputs in its history; a block shorter than the history keeps the rest.
        for back in range(min(width, rows)):
            np.testing.assert_array_equal(actual_history[:, back], x[rows - 1 - back])
    finally:
        for buffer in (serial_history, row_history, source, tap_buffer, serial_out, row_out):
            free(buffer, runtime=runtime)


@pytest.mark.parametrize("rows", [1, 3, 8, 256])
@pytest.mark.parametrize(("heads", "dim", "width"), [(8, 128, 4), (3, 67, 3), (2, 32, 1)])
def test_row_dynamic_convolution_is_the_serial_kernel_bit_for_bit(rows, heads, dim, width):
    from hipengine.core.memory import free
    from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv as conv

    runtime, arch = _conv_device()
    channels = heads * dim
    rng = np.random.default_rng(2000 + rows + channels)
    history = rng.standard_normal((channels, width), dtype=np.float32) * 1e3
    x = rng.standard_normal((rows, channels), dtype=np.float32) * 1e2
    fixed = rng.standard_normal((channels, width), dtype=np.float32)
    dynamic = rng.standard_normal((rows, heads, width), dtype=np.float32)
    scale = np.float32(1 / np.sqrt(2 * width + 1))
    serial_history, _ = _upload(history, runtime)
    row_history, _ = _upload(history, runtime)
    serial_x, _ = _upload(x, runtime)
    fixed_buffer, _ = _upload(fixed, runtime)
    dynamic_buffer, _ = _upload(dynamic, runtime)
    row_out, _ = _upload(np.zeros_like(x), runtime)
    try:
        conv._serial_symbol("dynamic", arch)(serial_history.ptr, serial_x.ptr, fixed_buffer.ptr, dynamic_buffer.ptr,
                                             heads, dim, width, rows, scale, 0)
        # The row kernel reads the block it convolves, so the caller keeps the input apart from the output.
        source = _upload(x, runtime)[0]
        try:
            conv.launch_short_conv_dynamic(row_history.ptr, source.ptr, fixed_buffer.ptr, dynamic_buffer.ptr, heads,
                                           dim, width, rows, scale, out=row_out.ptr, target_arch=arch)
            runtime.stream_synchronize(0)
        finally:
            free(source, runtime=runtime)
        expected = _download(serial_x, x, runtime)
        actual = _download(row_out, x, runtime)
        assert np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)) == 0
        expected_history = _download(serial_history, history, runtime)
        actual_history = _download(row_history, history, runtime)
        assert np.count_nonzero(actual_history.view(np.uint32) != expected_history.view(np.uint32)) == 0
    finally:
        for buffer in (serial_history, row_history, serial_x, fixed_buffer, dynamic_buffer, row_out):
            free(buffer, runtime=runtime)


def test_row_kernels_refuse_what_the_serial_kernel_allowed():
    from hipengine.kernels.hip_gfx1100.quant import inkling2_short_conv as conv

    with pytest.raises(ValueError, match="apart from its input"):
        conv.launch_short_conv_block(1, 2, 3, 2, 1024, 4, 1.0, 256, target_arch="gfx1100")
    with pytest.raises(ValueError, match="needs an output"):
        conv.launch_short_conv_dynamic(1, 2, 3, 4, 8, 128, 4, 4, 1.0, target_arch="gfx1100")
    with pytest.raises(ValueError, match="one to sixteen"):
        conv.launch_short_conv_block(1, 2, 3, 5, 1024, 17, 1.0, 256, target_arch="gfx1100")
