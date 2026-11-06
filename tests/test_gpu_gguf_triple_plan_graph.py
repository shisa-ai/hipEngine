"""Q8T16 triple decisions retain no graph input/output resources."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason='requires ROCm/libamdhip64.so')


@pytest.mark.parametrize('rows,rowtile', [(1, False), (4, True), (7, True)])
def test_triple_owner_graph_reads_fresh_inputs(rows, rowtile):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_GGUF_Q8_0_T16
    from hipengine.runtime.gguf_launch_policy import LaunchDecisionOwner, launch_decision_session
    from hipengine.runtime.gguf_linear import _resolve_gguf_linear_triple_plan, _execute_gguf_linear_triple_plan

    rt = get_hip_runtime()
    buffers = []
    def alloc(nbytes):
        value = malloc(nbytes)
        buffers.append(value)
        return value
    def upload(array):
        value = alloc(array.nbytes)
        copy_host_to_device(value, host_array_ptr(array), array.nbytes)
        return value

    rng = np.random.default_rng(19)
    k, n = 2048, 64
    weights = []
    stream = rt.stream_create(nonblocking=True)
    graph = executable = None
    try:
        for index in range(3):
            raw = np.empty((n, k // 32, 34), dtype=np.uint8)
            raw[..., :2] = np.frombuffer(np.float16(0.03125).tobytes(), dtype=np.uint8)
            raw[..., 2:] = rng.integers(-20, 21, size=(n, k // 32, 32), dtype=np.int8).view(np.uint8)
            tiles = repack_gguf_q8_0_tile16(raw.reshape(n, -1)).tiles
            buf = upload(tiles)
            allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=buf.ptr, shape=tiles.shape), buffer=buf)
            weights.append(SimpleNamespace(
                spec=SimpleNamespace(layout=LAYOUT_GGUF_Q8_0_T16, quant_key='gguf_q8_0_t16_v1',
                                     slot_path=f'layers.0.attn_{index}', source=SimpleNamespace(shape=(n, k))),
                allocation=lambda name, allocation=allocation: allocation,
            ))
        x = alloc(rows * k * 2)
        eager = [alloc(rows * n * 2) for _ in range(3)]
        replay = [alloc(rows * n * 2) for _ in range(3)]
        owner = LaunchDecisionOwner()
        with launch_decision_session(owner):
            plan = _resolve_gguf_linear_triple_plan(*weights, rows=rows, in_features=k,
                out_features=n, use_wmma_prefill=False, rowtile_all=rowtile)
        assert plan.operation is not None
        assert len(owner) == 1
        def launch(outputs, launch_stream=0):
            assert _execute_gguf_linear_triple_plan(plan, *weights, x.ptr,
                *(out.ptr for out in outputs), stream=launch_stream, runtime=rt)
        initial = np.zeros((rows, k), dtype=np.uint16)
        copy_host_to_device(x, host_array_ptr(initial), initial.nbytes)
        launch(eager)
        rt.device_synchronize()
        rt.stream_begin_capture(stream, 2)
        launch(replay, stream)
        graph = rt.stream_end_capture(stream)
        executable = rt.graph_instantiate(graph)
        previous = None
        for iteration in range(4):
            values = rng.standard_normal((rows, k)).astype(np.float32) * (iteration + 1)
            bits = values.view(np.uint32)
            inp = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            copy_host_to_device(x, host_array_ptr(inp), inp.nbytes)
            launch(eager)
            rt.device_synchronize()
            rt.graph_launch(executable, stream)
            rt.stream_synchronize(stream)
            actuals = []
            for expected_buf, actual_buf in zip(eager, replay, strict=True):
                expected = np.empty((rows, n), dtype=np.uint16)
                actual = np.empty_like(expected)
                copy_device_to_host(host_array_ptr(expected), expected_buf, expected.nbytes)
                copy_device_to_host(host_array_ptr(actual), actual_buf, actual.nbytes)
                np.testing.assert_array_equal(actual, expected)
                actuals.append(actual)
            if previous is not None:
                assert any(not np.array_equal(a, b) for a, b in zip(previous, actuals))
            previous = actuals
    finally:
        rt.stream_synchronize(stream)
        if executable is not None:
            rt.graph_exec_destroy(executable)
        if graph is not None:
            rt.graph_destroy(graph)
        rt.stream_destroy(stream)
        for buf in reversed(buffers):
            free(buf)
