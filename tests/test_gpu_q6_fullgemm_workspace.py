"""Q6 full-GEMM scratch retirement and independent owner numerical isolation."""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    malloc, free, memory_stats, copy_host_array_to_device,
    copy_device_to_host, host_array_ptr,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_f16_rocblas_prefill as q6
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
from tests.test_gpu_gguf_k_gemv import make_q6_k_weight
from hipengine.kernels.cpu_reference import gguf_q6_k_gemv


def available():
    try:
        ctypes.CDLL('libamdhip64.so')
        ctypes.CDLL('libhipblaslt.so')
    except OSError:
        return False
    return True


@pytest.mark.skipif(not available(), reason='HIP and hipBLASLt required')
def test_fullgemm_two_owners_preserve_outputs_and_release_scratch():
    runtime = get_hip_runtime()
    initial = memory_stats()['current_allocated_bytes']
    owners = [q6.Q6FullGemmWorkspace(), q6.Q6FullGemmWorkspace()]
    streams = [runtime.stream_create(), runtime.stream_create()]
    buffers = []
    try:
        rows, k, n = 17, 256, 32
        raw = make_q6_k_weight(n, k)
        tiles = repack_gguf_q6_k_tile16_qmicro_planar(raw[np.newaxis]).tiles
        x = np.random.default_rng(42).normal(size=(rows, k)).astype(np.float16)
        for array in (x, tiles):
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_array_to_device(buf, array, runtime=runtime)
        for _ in owners:
            buffers.append(malloc(rows*n*2, runtime=runtime))
        runtime.device_synchronize()
        base = memory_stats()['current_allocated_bytes']

        def run(index, values):
            copy_host_array_to_device(buffers[0], values, runtime=runtime)
            q6.gguf_q6_k_t16_qmicro_planar_fullgemm_fp16_in_bf16_out(
                buffers[0].ptr, buffers[1].ptr, buffers[2+index].ptr,
                rows, k, n, stream=streams[index], runtime=runtime,
                workspace_owner=owners[index],
            )
            runtime.stream_synchronize(streams[index])
            bits = np.empty((rows,n), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(bits), buffers[2+index], bits.nbytes, runtime=runtime)
            return bits

        first = run(0, x)
        second = run(1, -x)
        reference = gguf_q6_k_gemv(x.astype(np.float32), raw)
        actual = (first.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_allclose(actual, reference, atol=0.05, rtol=0.03)
        assert memory_stats()['current_allocated_bytes'] > base
        owners[0].close()
        np.testing.assert_array_equal(run(1, -x), second)
        owners[1].close()
        assert memory_stats()['current_allocated_bytes'] == base
    finally:
        for owner in owners:
            owner.close()
        for stream in streams:
            runtime.stream_destroy(stream)
        for buf in reversed(buffers):
            free(buf, runtime=runtime)
    assert memory_stats()['current_allocated_bytes'] == initial
