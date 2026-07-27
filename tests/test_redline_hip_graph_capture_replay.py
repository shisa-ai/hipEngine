"""Strict retained-PM4 graph relaunch gate for the optional Redline adapter."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


def _redline_paths() -> tuple[Path, Path, str] | None:
    library = os.environ.get("HIPENGINE_REDLINE_HIPGRAPH_LIBRARY")
    module = os.environ.get("HIPENGINE_REDLINE_HIPGRAPH_MODULE")
    digest = os.environ.get("HIPENGINE_REDLINE_HIPGRAPH_SHA256")
    if not library and not module and not digest:
        return None
    if not library or not module or not digest:
        pytest.fail("all HIPENGINE_REDLINE_HIPGRAPH_* controls are required")
    return Path(library), Path(module), digest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return (_to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def test_redline_graph_capture_replay_is_pm4_and_bit_exact() -> None:
    paths = _redline_paths()
    if paths is None:
        pytest.skip("requires an explicitly preloaded pinned Redline hipGraph DSO")
    if not _hip_available():
        pytest.skip("requires ROCm/libamdhip64.so")
    library_path, module_path, digest = paths

    from hipengine.core.hip import configure_default_graph_adapter, get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.redline_graph import RedlineHipGraphAdapter
    from hipengine.kernels.hip_gfx1100.convert.cast import (
        bf16_to_f32,
        build_cast,
        f32_to_bf16,
    )

    adapter = RedlineHipGraphAdapter.load(
        library_path=library_path,
        module_path=module_path,
        expected_sha256=digest,
        require_pm4=True,
    )
    configure_default_graph_adapter(adapter)
    runtime = get_hip_runtime()
    compiler_version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    if not compiler_version_file:
        pytest.fail("HIPENGINE_COMPILER_VERSION_FILE is required for cache-only replay")
    compiler_version = Path(compiler_version_file).read_text(encoding="utf-8").strip()
    cast_library = build_cast(
        load=True,
        require_cached=True,
        compiler_version=compiler_version,
    )

    n = 4096
    buffers = []

    def allocate(nbytes: int):
        buffer = malloc(nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    x = allocate(n * 4)
    xb = allocate(n * 2)
    x2 = allocate(n * 4)
    stream = runtime.stream_create(nonblocking=True)
    graph = 0
    graph_exec = 0
    try:
        warm = np.zeros(n, dtype=np.float32)
        copy_host_to_device(
            x, host_array_ptr(warm), warm.nbytes, runtime=runtime
        )
        f32_to_bf16(
            x.ptr, xb.ptr, n, library=cast_library, runtime=runtime
        )
        bf16_to_f32(
            xb.ptr, x2.ptr, n, library=cast_library, runtime=runtime
        )
        runtime.device_synchronize()

        runtime.stream_begin_capture(stream, mode=2)
        f32_to_bf16(
            x.ptr,
            xb.ptr,
            n,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        bf16_to_f32(
            xb.ptr,
            x2.ptr,
            n,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        graph = runtime.stream_end_capture(stream)
        graph_exec = runtime.graph_instantiate(graph)
        assert runtime.graph_exec_transport(graph_exec) == "redline_pm4"

        rng = np.random.default_rng(20260728)
        for replay in range(4):
            value = rng.standard_normal(n).astype(np.float32) * (replay + 1)
            copy_host_to_device(
                x,
                host_array_ptr(np.ascontiguousarray(value)),
                value.nbytes,
                runtime=runtime,
            )
            runtime.graph_launch(graph_exec, stream)
            runtime.stream_synchronize(stream)
            assert runtime.graph_exec_transport(graph_exec) == "redline_pm4"
            output = np.empty(n, dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(output), x2, output.nbytes, runtime=runtime
            )
            assert np.array_equal(output, _bf16_round(value))

        provenance = runtime.graph_transport_provenance()
        assert provenance["library_sha256"] == "sha256:" + digest.removeprefix("sha256:")
        assert provenance["same_backing_file"] is True
        assert provenance["selected_exec_transports"] == {"redline_pm4": 1}
    finally:
        if graph_exec:
            runtime.graph_exec_destroy(graph_exec)
        if graph:
            runtime.graph_destroy(graph)
        runtime.stream_destroy(stream)
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
