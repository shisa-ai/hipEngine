import ctypes

import numpy as np
import pytest

_CASES = [(3, 5, 2304), (2, 7, 1000), (1, 1, 2304), (4, 33, 96)]


def _device():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import resolve_backend

    if resolve_backend("auto", execution_device=True) not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires a HIP backend with the Inkling controller kernels")
    return get_hip_runtime()


def _rms(values, eps):
    return values / np.sqrt((values * values).mean(axis=-1, keepdims=True) + eps)


@pytest.mark.parametrize("count,rows,width", _CASES)
def test_controller_input_matches_double_rms(count, rows, width):
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.norm import inkling2_controller as glue

    runtime = _device()
    rng = np.random.default_rng(811 + width)
    eps = np.float32(1e-6)
    x = rng.standard_normal((count, rows, width), dtype=np.float32) * 3
    message = rng.standard_normal((count, rows, width), dtype=np.float32)
    expected = np.concatenate((_rms(x.astype(np.float64), eps), _rms(message.astype(np.float64), eps)),
                              axis=-1).astype(np.float32)
    buffers = [malloc(a.nbytes, runtime=runtime) for a in (x, message, expected)]
    x_dev, message_dev, out_dev = buffers
    try:
        runtime.memcpy(x_dev.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        runtime.memcpy(message_dev.ptr, message.ctypes.data, message.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        actual = np.full(expected.shape, np.nan, dtype=np.float32)
        first = None
        for _ in range(2):
            glue.launch_controller_input(x_dev.ptr, message_dev.ptr, out_dev.ptr, rows, width, count, eps)
            runtime.stream_synchronize(0)
            runtime.memcpy(actual.ctypes.data, out_dev.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            if first is None:
                first = actual.copy()
        np.testing.assert_array_equal(actual, first)
        assert np.isfinite(actual).all()
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


@pytest.mark.parametrize("count,rows,width", _CASES)
def test_controller_blend_norm_matches_double_reference(count, rows, width):
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.norm import inkling2_controller as glue

    runtime = _device()
    rng = np.random.default_rng(977 + width)
    eps = np.float32(1e-6)
    shape = (count, rows, width)
    x = rng.standard_normal(shape, dtype=np.float32) * 2
    message = rng.standard_normal(shape, dtype=np.float32) * 0.5
    forget = rng.standard_normal(shape, dtype=np.float32) * 4
    gain = rng.standard_normal(shape, dtype=np.float32) * 4
    weight = rng.standard_normal((count, width), dtype=np.float32)
    merged = ((1 - np.tanh(forget.astype(np.float64))) * x + np.tanh(gain.astype(np.float64)) * message)
    expected = (_rms(merged, eps) * weight.astype(np.float64)[:, None, :]).astype(np.float32)
    arrays = (x, message, forget, gain, weight, expected)
    buffers = [malloc(a.nbytes, runtime=runtime) for a in arrays]
    *sources, out_dev = buffers
    try:
        for buf, a in zip(sources, arrays):
            runtime.memcpy(buf.ptr, a.ctypes.data, a.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        actual = np.full(expected.shape, np.nan, dtype=np.float32)
        first = None
        for _ in range(2):
            glue.launch_controller_blend_norm(*(buf.ptr for buf in sources), out_dev.ptr, rows, width, count, eps)
            runtime.stream_synchronize(0)
            runtime.memcpy(actual.ctypes.data, out_dev.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            if first is None:
                first = actual.copy()
        np.testing.assert_array_equal(actual, first)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_resident_controllers_match_the_host_chain(count):
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    from unittest.mock import Mock

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.rocblas import Rocblas
    from hipengine.kernels.backends import resolve_backend
    from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
    from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections
    from hipengine.runtime.inkling2_mini_reference import _blend_controller, _rms
    from tests.test_unit_inkling2_gguf import _file

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    rng = np.random.default_rng(234)
    reader = Mock()
    reader.info = _file(False)
    weights = Inkling2GGUFWeights(reader)
    layer = next(candidate for candidate in weights.mapping.layers if candidate.controllers)
    raw = {}

    # The reader hands float32 tensors through unchanged; only block formats arrive as bytes.
    def floats(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    for controller in layer.controllers:
        for op in ("down", "forget", "gain"):
            tensor = getattr(controller, op)
            raw[tensor.name] = floats(tensor.shape, 0.3)
    norm = layer.tensors["attn_norm.weight"]
    raw[norm.name] = floats(norm.shape, 0.1)
    reader.tensor_data.side_effect = raw.__getitem__

    eps = np.float32(1e-6)
    rows = 13
    hidden = weights.mapping.hidden_size
    inputs = [rng.standard_normal((rows, hidden), dtype=np.float32) for _ in range(count)]
    messages = [rng.standard_normal((rows, hidden), dtype=np.float32) for _ in range(count)]
    norm_weight = weights.decode(norm)
    blas = Rocblas.load()
    try:
        options = dict(runtime=get_hip_runtime(), blas=blas, backend=backend)
        with Inkling2GGUFDeviceProjections(weights, **options) as strict:
            assert not strict.supports_resident_controllers
        with Inkling2GGUFDeviceProjections(weights, fast_controllers=True, **options) as owner:
            assert owner.supports_resident_controllers
            names = [tuple(getattr(c, op).name for op in ("down", "forget", "gain"))
                     for c in layer.controllers[:count]]
            out = owner.project_controllers_normed(zip(names, inputs, messages, strict=True),
                                                   norm_weights=[norm_weight] * count, eps=eps)
            assert out.shape == (count, rows, hidden)
            for i, controller in enumerate(layer.controllers[:count]):
                gates = np.concatenate((_rms(inputs[i], eps), _rms(messages[i], eps)), axis=-1)
                mid = gates @ weights.decode(controller.down).T
                forget = mid @ weights.decode(controller.forget).T
                gain = mid @ weights.decode(controller.gain).T
                expected = _rms(_blend_controller(inputs[i], messages[i], forget, gain), eps) * norm_weight
                np.testing.assert_allclose(out[i], expected, rtol=1e-4, atol=1e-5)
                # Each controller's result does not depend on the others in the exchange.
                alone = owner.project_controllers_normed(((names[i], inputs[i], messages[i]),),
                                                         norm_weights=[norm_weight], eps=eps)[0]
                np.testing.assert_allclose(out[i], alone, rtol=1e-6, atol=1e-6)
            with pytest.raises(ValueError):
                owner.project_controllers_normed(((names[0], inputs[0], messages[0]),), norm_weights=(), eps=eps)
    finally:
        blas.close()
