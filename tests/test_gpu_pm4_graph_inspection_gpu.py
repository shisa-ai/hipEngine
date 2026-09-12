"""Guarded live proof that native HIP graphs can be inspected without interposition."""

from __future__ import annotations

import ctypes

import pytest

from hipengine.kernels.backends import detect_hip_target_arches


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="requires ROCm/libamdhip64.so")


def test_smoke_add_graph_reconciles_exact_dso_hsaco_geometry_and_kernargs() -> None:
    if "gfx1100" not in detect_hip_target_arches():
        pytest.skip("initial PM4 inspection gate requires physical gfx1100")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.core.pm4 import inspect_hip_graph
    from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add, smoke_add_f32

    runtime = get_hip_runtime()
    library = build_smoke_add(load=True)
    buffers = [malloc(4) for _ in range(3)]
    stream = runtime.stream_create(nonblocking=True)
    graph = 0
    try:
        # Build/load and warm the wrapper before entering capture.
        smoke_add_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            1,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        runtime.stream_begin_capture(stream, 2)
        smoke_add_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            1,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        graph = runtime.stream_end_capture(stream)
        manifest = inspect_hip_graph(runtime, graph, gfx_arch="gfx1100", stream=stream)

        assert manifest.order == tuple(node.handle for node in manifest.nodes)
        assert len(manifest.nodes) == 1
        node = manifest.nodes[0]
        assert node.name == "hipengine_smoke_add_f32_kernel"
        assert node.loader_symbol == "hipengine_smoke_add_f32_kernel.kd"
        assert node.grid_blocks == (1, 1, 1)
        assert node.grid_workitems == (256, 1, 1)
        assert node.block == (256, 1, 1)
        assert len(node.kernarg) == 288
        assert int.from_bytes(node.kernarg[0:8], "little") == buffers[0].ptr
        assert int.from_bytes(node.kernarg[8:16], "little") == buffers[1].ptr
        assert int.from_bytes(node.kernarg[16:24], "little") == buffers[2].ptr
        assert int.from_bytes(node.kernarg[24:32], "little") == 1
        assert node.kernarg[32:36] == (1).to_bytes(4, "little")
        assert node.kernarg[44:46] == (256).to_bytes(2, "little")
        assert node.kernarg[50:96] == bytes(46)
        assert node.kernarg[96:98] == (1).to_bytes(2, "little")
        assert node.hsaco.startswith(b"\x7fELF")
        assert node.target_id.endswith("gfx1100")
    finally:
        if graph:
            runtime.graph_destroy(graph)
        runtime.stream_destroy(stream)
        for buffer in reversed(buffers):
            free(buffer)
