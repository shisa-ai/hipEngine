"""Guarded exact HIP/direct-AQL/retained-PM4 smoke on one gfx1100 HSACO."""

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


def test_same_hsaco_native_graph_direct_aql_and_pm4_are_bit_exact() -> None:
    if os.environ.get("HIPENGINE_HIP_ARCH", "gfx1100") != "gfx1100":
        pytest.skip("initial native PM4 gate is gfx1100-only")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.pm4 import NativePm4Context, NativePm4Error, inspect_hip_graph
    from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add, smoke_add_f32

    runtime = get_hip_runtime()
    library = build_smoke_add(load=True)
    n = 257
    nbytes = n * np.dtype(np.float32).itemsize
    buffers = [malloc(nbytes) for _ in range(3)]
    stream = runtime.stream_create(nonblocking=True)
    graph = 0
    graph_exec = 0
    context = None
    executable = None
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
        manifest = inspect_hip_graph(runtime, graph, gfx_arch="gfx1100", stream=stream)
        graph_exec = runtime.graph_instantiate(graph)

        context = NativePm4Context.create(
            pci_bdf=runtime.device_pci_bus_id(), gfx_arch="gfx1100"
        )
        executable = context.instantiate(manifest)
        with pytest.raises(NativePm4Error, match="live executables"):
            context.close()

        rng = np.random.default_rng(20260807)
        outputs: dict[str, list[np.ndarray]] = {"hipgraph": [], "aql": [], "pm4": []}
        for transport in ("hipgraph", "aql", "pm4"):
            for iteration in range(2):
                a = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float32)
                b = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float32)
                expected = a + b
                copy_host_to_device(buffers[0], host_array_ptr(a), a.nbytes, runtime=runtime)
                copy_host_to_device(buffers[1], host_array_ptr(b), b.nbytes, runtime=runtime)
                runtime.memset(buffers[2].ptr, 0, nbytes)
                runtime.device_synchronize()

                if transport == "hipgraph":
                    runtime.graph_launch(graph_exec, stream)
                    runtime.stream_synchronize(stream)
                else:
                    executable.launch(transport)

                output = np.empty(n, dtype=np.float32)
                copy_device_to_host(host_array_ptr(output), buffers[2], output.nbytes, runtime=runtime)
                assert np.array_equal(output, expected), f"{transport} iteration {iteration}"
                outputs[transport].append(output)

        provenance = executable.provenance()
        context_provenance = context.provenance()
        assert provenance["nodes"] == 1
        assert provenance["modules"] == 1
        assert provenance["pm4_dwords"] > 0
        assert provenance["aql_publication"] != 0
        assert provenance["pm4_publication"] != 0
        assert provenance["aql_submissions"] == 2
        assert provenance["pm4_submissions"] == 2
        assert provenance["last_packet_id"] == 3
        assert provenance["last_packet_count"] == 1
        assert provenance["last_timeout_ns"] == 5_000_000_000
        assert provenance["last_completion_value"] == 0
        assert provenance["last_transport"] == "pm4"
        assert provenance["retired"] is True
        assert provenance["usable"] is True
        assert len(provenance["module_records"]) == 1
        assert provenance["module_records"][0]["reader_handle"] != 0
        assert provenance["module_records"][0]["executable_handle"] != 0
        assert len(provenance["dispatch_records"]) == 1
        dispatch = provenance["dispatch_records"][0]
        assert dispatch["symbol"] == "hipengine_smoke_add_f32_kernel.kd"
        assert dispatch["kernel_object"] != 0
        assert dispatch["code_entry"] != 0
        assert dispatch["kernarg_address"] % dispatch["kernarg_align"] == 0
        assert context_provenance["process_id"] == os.getpid()
        assert context_provenance["hsa_version_major"] >= 1
        assert context_provenance["queue_type"] == "multi"
        assert context_provenance["last_doorbell_value"] == 3
        assert isinstance(context_provenance["doorbell_value"], int)
        assert context_provenance["completion_value"] == 0
        assert context_provenance["submissions"] == 4
        assert context_provenance["callback_status"] == 0
        assert context_provenance["usable"] is True
    finally:
        if executable is not None:
            executable.close()
        if context is not None:
            context.close()
        if graph_exec:
            runtime.graph_exec_destroy(graph_exec)
        if graph:
            runtime.graph_destroy(graph)
        runtime.stream_destroy(stream)
        for buffer in reversed(buffers):
            free(buffer)
