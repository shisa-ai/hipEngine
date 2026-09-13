"""Foundational HIP graph capture/replay parity gate (task #15).

The GGUF decode-graph was retired because the GDN conv/recurrent decode kernels
corrupted device state on the 3rd+ graph relaunch (a stateful in-place hazard).
The planned per-layer MoE graph is STATELESS (it recomputes from fresh inputs
each replay, no persistent recurrent state), so this test pins the property the
MoE graph relies on: a captured multi-kernel sequence replays BIT-EXACTLY across
>=3 relaunches AND reads fresh input data written between replays (the MoE reads
the eager attention's fresh output from a fixed scratch pointer each token).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="requires ROCm/libamdhip64.so")


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return (_to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def test_hip_graph_capture_replay_stateless_bit_exact_across_relaunches() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.convert.cast import (
        bf16_to_f32,
        build_cast,
        f32_to_bf16,
    )

    rt = get_hip_runtime()
    cast_lib = build_cast(load=True)
    n = 4096

    bufs = []

    def _alloc(nbytes):
        b = malloc(nbytes)
        bufs.append(b)
        return b

    x = _alloc(n * 4)      # f32 input
    xb = _alloc(n * 2)     # bf16 intermediate
    x2 = _alloc(n * 4)     # f32 output

    stream = rt.stream_create(nonblocking=True)
    graph_exec = None
    graph = None
    try:
        # Warm the kernels (force JIT build/load) BEFORE capture so no hipcc runs
        # during stream capture.
        warm = np.zeros(n, dtype=np.float32)
        copy_host_to_device(x, host_array_ptr(warm), warm.nbytes)
        f32_to_bf16(x.ptr, xb.ptr, n, library=cast_lib, runtime=rt)
        bf16_to_f32(xb.ptr, x2.ptr, n, library=cast_lib, runtime=rt)
        rt.device_synchronize()

        # Capture the 2-kernel sequence on a non-default stream.
        rt.stream_begin_capture(stream, 2)
        f32_to_bf16(x.ptr, xb.ptr, n, stream=stream, library=cast_lib, runtime=rt)
        bf16_to_f32(xb.ptr, x2.ptr, n, stream=stream, library=cast_lib, runtime=rt)
        graph = rt.stream_end_capture(stream)
        assert graph != 0, "stream capture produced no graph"
        graph_exec = rt.graph_instantiate(graph)
        assert graph_exec != 0, "graph instantiate failed"

        # Replay 4 times (>=3 to catch the GDN-style relaunch hazard), each with
        # a DIFFERENT fresh input written to the same pointer.
        rng = np.random.default_rng(20260628)
        for it in range(4):
            inp = rng.standard_normal(n).astype(np.float32) * (it + 1)
            copy_host_to_device(x, host_array_ptr(np.ascontiguousarray(inp)), inp.nbytes)
            rt.graph_launch(graph_exec, stream)
            rt.stream_synchronize(stream)
            out = np.empty(n, dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), x2, out.nbytes)
            ref = _bf16_round(inp)  # f32 -> bf16 -> f32 round-trip
            assert np.array_equal(out, ref), f"relaunch {it} diverged (max|d|={np.max(np.abs(out-ref))})"
    finally:
        if graph_exec:
            rt.graph_exec_destroy(graph_exec)
        if graph:
            rt.graph_destroy(graph)
        rt.stream_destroy(stream)
        for b in reversed(bufs):
            free(b)
