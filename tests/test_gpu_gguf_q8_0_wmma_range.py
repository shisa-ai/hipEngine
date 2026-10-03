"""Finite wide-range activations must survive raw Q8 WMMA narrowing."""
import ctypes

import numpy as np
import pytest

from hipengine.core.memory import malloc, free, host_array_ptr, copy_host_to_device, copy_device_to_host
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    gguf_q8_0_wmma_prefill_bf16_bf16_out, gguf_q8_0_wmma_prefill_f32_f32_out,
    gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out,
)
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight
from tests.test_gpu_gguf_q8_0_wmma_prefill import _float_array_to_bf16_bits, _bf16_bits_to_float32


def _require_hip():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        pytest.skip('libamdhip64.so is not available')


@pytest.mark.parametrize('dtype,dual', [('bf16',False), ('f32',False), ('bf16',True)])
@pytest.mark.parametrize('rows', [4, 17, 33])
def test_raw_q8_wmma_preserves_finite_inputs_outside_fp16_range(dtype, dual, rows):
    _require_hip()
    from hipengine.core.hip import get_hip_runtime
    runtime = get_hip_runtime()
    rng = np.random.default_rng(311)
    x = rng.uniform(-1, 1, (rows, 64)).astype(np.float32) * 2**20
    host = _float_array_to_bf16_bits(x) if dtype == 'bf16' else x
    x = _bf16_bits_to_float32(host) if dtype == 'bf16' else host
    columns = 32 if dual else 24
    w = make_q8_0_weight(columns, 64)
    expected = gguf_q8_0_gemv(x, w)
    other = w.copy()
    other.reshape(columns, 2, 34)[:, :, 1] ^= 0x80
    if dual:
        expected = np.concatenate((expected,gguf_q8_0_gemv(x,other)),axis=1)
    output = np.empty((rows, columns * (2 if dual else 1)), dtype=np.uint16 if dtype == 'bf16' else np.float32)
    buffers = []
    try:
        for size in (host.nbytes, w.nbytes, output.nbytes, other.nbytes):
            buffers.append(malloc(size, runtime=runtime))
        a, b, c, d = buffers
        copy_host_to_device(a, host_array_ptr(host), runtime=runtime)
        copy_host_to_device(b, host_array_ptr(w), runtime=runtime)
        if dual:
            copy_host_to_device(d, host_array_ptr(other), runtime=runtime)
            gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
                a.ptr,b.ptr,d.ptr,c.ptr,rows,64,columns,columns,runtime=runtime)
        else:
            wrapper = gguf_q8_0_wmma_prefill_bf16_bf16_out if dtype == 'bf16' else gguf_q8_0_wmma_prefill_f32_f32_out
            wrapper(a.ptr,b.ptr,c.ptr,rows,64,columns,runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(output),c,runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer,runtime=runtime)
    actual = _bf16_bits_to_float32(output) if dtype == 'bf16' else output
    assert np.isfinite(actual).all()
    if dtype == 'bf16':
        expected = _bf16_bits_to_float32(_float_array_to_bf16_bits(expected))
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1.0)
