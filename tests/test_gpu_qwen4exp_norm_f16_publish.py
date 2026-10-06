"""Dual norm publication equals parent F32 plus the existing F16 conversion."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import free, malloc, copy_host_to_device, host_array_ptr
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.kernels.cpu_reference.qwen4_exp import grouped_zero_centered_rmsnorm
from tests.test_gpu_qwen4_exp_gr_hip import _upload, _alloc, _download


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("rows,hidden", [(1, 33), (3, 257), (257, 2560), (512, 2560)])
def test_norm_f16_publication_parent_bits_guards_and_cpu(rows, hidden):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused import qwen4_exp_gr as gr
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide

    runtime = get_hip_runtime()
    library = gr.build_qwen4_exp_gr(load=True)
    cast_library = wide.build_gguf_q8_0_dense_wide(load=True)
    branches = 4
    shape = (rows, branches, hidden)
    rng = np.random.default_rng(717)
    residual = float_array_to_bf16_bits(rng.normal(0, 0.3, shape).astype(np.float32))
    # Zero, subnormal BF16, large residual, signed zero and IEEE specials.
    residual.reshape(-1)[:8] = [0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0x7fc0]
    weight = rng.normal(1, 0.05, (branches, hidden)).astype(np.float32)
    # Test half rounding/overflow/underflow without changing norm reductions.
    weight.reshape(-1)[:8] = [0, -0.0, 2**-24, 2**-14, 1.00048828125, 65504, 1e10, -1e10]
    allocations = []
    stream = runtime.stream_create()
    try:
        x = _upload(residual, runtime, allocations)
        w = _upload(weight, runtime, allocations)
        parent = _alloc(shape, np.float32, runtime, allocations)
        staged = _alloc(shape, np.float16, runtime, allocations)
        guard = 64
        f32_bytes = residual.size * 4
        f16_bytes = residual.size * 2
        output = malloc(f32_bytes + 2 * guard, runtime=runtime)
        half = malloc(f16_bytes + 2 * guard, runtime=runtime)
        allocations.extend([output, half])
        for buffer in (output, half):
            poison = np.full(buffer.nbytes, 0xa5, np.uint8)
            copy_host_to_device(buffer, host_array_ptr(poison), runtime=runtime)
        out_ptr, half_ptr = output.ptr + guard, half.ptr + guard
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32(x.ptr, w.ptr, parent.ptr, rows, branches, hidden,
                                            stream=stream, library=library, runtime=runtime)
        wide.f32_to_f16(parent.ptr, staged.ptr, residual.size, stream=stream,
                        library=cast_library, runtime=runtime)
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(x.ptr, w.ptr, out_ptr, half_ptr,
                                                rows, branches, hidden, stream=stream,
                                                library=library, runtime=runtime)
        runtime.stream_synchronize(stream)
        f32 = _download(SimpleNamespace(ptr=out_ptr, nbytes=f32_bytes), shape, np.float32, runtime)
        f16 = _download(SimpleNamespace(ptr=half_ptr, nbytes=f16_bytes), shape, np.float16, runtime)
        p32 = _download(parent, shape, np.float32, runtime)
        p16 = _download(staged, shape, np.float16, runtime)
        np.testing.assert_array_equal(f32.view(np.uint32), p32.view(np.uint32))
        np.testing.assert_array_equal(f16.view(np.uint16), p16.view(np.uint16))
        for buffer in (output, half):
            data = _download(buffer, (buffer.nbytes,), np.uint8, runtime)
            assert np.all(data[:guard] == 0xa5) and np.all(data[-guard:] == 0xa5)
        np.testing.assert_array_equal(_download(x, shape, np.uint16, runtime), residual)
        np.testing.assert_array_equal(_download(w, weight.shape, np.float32, runtime), weight)
        # Independent CPU floor on the unpoisoned groups; IEEE-special groups
        # are checked by parent bits rather than a different CPU NaN payload.
        with np.errstate(all="ignore"):
            expected = grouped_zero_centered_rmsnorm(bf16_to_float32(residual), weight)
        finite_groups = np.isfinite(expected).all(axis=-1)
        assert finite_groups.any()
        np.testing.assert_allclose(f32[finite_groups], expected[finite_groups], atol=2e-5, rtol=2e-5)
        def log_softmax(values):
            values = values.astype(np.float64)
            values -= values.max(axis=-1, keepdims=True)
            return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))
        lp, lq = log_softmax(expected[finite_groups]), log_softmax(f32[finite_groups])
        assert np.max(np.sum(np.exp(lp) * (lp - lq), axis=-1)) <= 0.05
        assert np.mean(lp.argmax(-1) == lq.argmax(-1)) >= 0.90
        # Repeat publication after poison proves both planes are really written.
        runtime.memset_async(out_ptr, 0x5a, f32_bytes, stream)
        runtime.memset_async(half_ptr, 0x5a, f16_bytes, stream)
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(x.ptr, w.ptr, out_ptr, half_ptr,
                                                rows, branches, hidden, stream=stream,
                                                library=library, runtime=runtime)
        runtime.stream_synchronize(stream)
        np.testing.assert_array_equal(_download(SimpleNamespace(ptr=out_ptr, nbytes=f32_bytes), shape,
                                                np.uint32, runtime), p32.view(np.uint32))
        np.testing.assert_array_equal(_download(SimpleNamespace(ptr=half_ptr, nbytes=f16_bytes), shape,
                                                np.uint16, runtime), p16.view(np.uint16))
    finally:
        runtime.stream_destroy(stream)
        for buffer in reversed(allocations):
            free(buffer, runtime=runtime)


def test_norm_f16_publication_finite_rounding_boundaries():
    """Keep the rounding probes outside the IEEE-special/NaN groups."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused import qwen4_exp_gr as gr
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide

    runtime = get_hip_runtime()
    library = gr.build_qwen4_exp_gr(load=True)
    cast_library = wide.build_gguf_q8_0_dense_wide(load=True)
    # eps=0 and residual=1 make the exact normalized value equal gamma.
    gamma = np.array([0.0, -0.0, 2**-25, -2**-25, 2**-24, 2**-14,
                      1.00048828125, 1.00146484375, 65504, 65520, -65520], np.float32)
    hidden = len(gamma)
    x = float_array_to_bf16_bits(np.ones((1, 1, hidden), np.float32))
    allocations = []
    try:
        xd, gd = _upload(x, runtime, allocations), _upload(gamma, runtime, allocations)
        parent = _alloc((hidden,), np.float32, runtime, allocations)
        staged = _alloc((hidden,), np.float16, runtime, allocations)
        out = _alloc((hidden,), np.float32, runtime, allocations)
        half = _alloc((hidden,), np.float16, runtime, allocations)
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32(
            xd.ptr, gd.ptr, parent.ptr, 1, 1, hidden, 0.0, library=library, runtime=runtime)
        wide.f32_to_f16(parent.ptr, staged.ptr, hidden, library=cast_library, runtime=runtime)
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
            xd.ptr, gd.ptr, out.ptr, half.ptr, 1, 1, hidden, 0.0, library=library, runtime=runtime)
        runtime.device_synchronize()
        p32 = _download(parent, (hidden,), np.float32, runtime)
        np.testing.assert_array_equal(p32.view(np.uint32), gamma.view(np.uint32))
        np.testing.assert_array_equal(_download(out, (hidden,), np.uint32, runtime), p32.view(np.uint32))
        expected_bits = np.array([0, 0x8000, 0, 0x8000, 1, 0x400,
                                  0x3c00, 0x3c02, 0x7bff, 0x7c00, 0xfc00], np.uint16)
        np.testing.assert_array_equal(_download(staged, (hidden,), np.uint16, runtime), expected_bits)
        np.testing.assert_array_equal(_download(half, (hidden,), np.uint16, runtime), expected_bits)
    finally:
        for buffer in reversed(allocations):
            free(buffer, runtime=runtime)
