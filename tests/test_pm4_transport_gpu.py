"""Guarded registry/ownership smoke for explicit retained-PM4 graph submission."""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest


def _rocm_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("libhsa-runtime64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _rocm_available(), reason="requires ROCm HIP and public HSA runtimes"
)


def test_registry_selected_pm4_submission_reuses_one_queue_without_hip_fallback() -> None:
    if os.environ.get("HIPENGINE_HIP_ARCH", "gfx1100") != "gfx1100":
        pytest.skip("initial PM4 transport integration gate is gfx1100-only")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.pm4 import create_graph_submission, create_graph_submission_context
    from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add, smoke_add_f32

    runtime = get_hip_runtime()
    library = build_smoke_add(load=True)
    n = 257
    nbytes = n * np.dtype(np.float32).itemsize
    buffers = [malloc(nbytes) for _ in range(3)]
    stream = runtime.stream_create(nonblocking=True)
    graph = 0
    submission = None
    submission_context = None
    try:
        smoke_add_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            n,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        runtime.stream_begin_capture(stream, 2)
        smoke_add_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            n,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        graph = runtime.stream_end_capture(stream)
        submission_context = create_graph_submission_context(
            backend="hip_gfx1100",
            gfx_arch="gfx1100",
            runtime=runtime,
            transport="pm4",
        )
        assert submission_context is not None
        submission = create_graph_submission(
            backend="hip_gfx1100",
            gfx_arch="gfx1100",
            runtime=runtime,
            graph=graph,
            stream=stream,
            transport="pm4",
            submission_context=submission_context,
        )

        rng = np.random.default_rng(20260807)
        for _ in range(2):
            a = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float32)
            b = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float32)
            expected = np.ascontiguousarray(a + b)
            copy_host_to_device(buffers[0], host_array_ptr(a), a.nbytes, runtime=runtime)
            copy_host_to_device(buffers[1], host_array_ptr(b), b.nbytes, runtime=runtime)
            runtime.memset(buffers[2].ptr, 0, nbytes)
            runtime.device_synchronize()

            submission.launch(stream)
            runtime.stream_synchronize(stream)
            output = np.empty(n, dtype=np.float32)
            copy_device_to_host(host_array_ptr(output), buffers[2], output.nbytes, runtime=runtime)
            assert np.array_equal(output, expected)

        live = submission.provenance()
        assert live["transport"] == "pm4"
        assert live["stateful_registers"] is True
        assert live["local_cache_dependencies"] is True
        assert live["executable"]["stateful_registers"] is True
        assert live["executable"]["local_cache_dependencies"] is True
        assert live["launches"] == 2
        assert live["native_fallbacks"] == 0
        assert live["context"]["submissions"] == 2
        assert live["executable"]["pm4_submissions"] == 2
        assert live["executable"]["retired"] is True
        submission.close()
        closed = submission.provenance()
        assert closed["closed"] is True
        assert closed["native_fallbacks"] == 0
        assert closed["transport_context"]["children"] == 0
        submission_context.close()
        assert submission_context.provenance()["closed"] is True
    finally:
        if submission is not None:
            submission.close()
        if submission_context is not None:
            submission_context.close()
        if graph:
            runtime.graph_destroy(graph)
        runtime.stream_destroy(stream)
        for buffer in reversed(buffers):
            free(buffer)
