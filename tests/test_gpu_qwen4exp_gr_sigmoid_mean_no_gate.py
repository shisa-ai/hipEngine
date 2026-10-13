"""The no-gate fused sigmoid gated-mean matches the separate pair bit for bit.

``qwen4_exp_gated_mean_sigmoid_f32(store_gate=False)`` applies the sigmoid
inside the gated-mean pass and leaves the logits buffer untouched. The
contract is exact parity of the mixed output with ``sigmoid`` followed by
``gated_mean``.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    build_qwen4_exp_gr,
    qwen4_exp_gated_mean_f32,
    qwen4_exp_gated_mean_sigmoid_f32,
    qwen4_exp_sigmoid_f32,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


@pytest.mark.parametrize(
    "rows,branches,hidden", [(1, 4, 6), (7, 4, 300), (33, 4, 2560), (512, 4, 2560), (5, 3, 257)]
)
def test_no_gate_fused_epilogue_is_bit_identical_and_keeps_logits(rows, branches, hidden):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gr(load=True)
    rng = np.random.default_rng(rows * 131 + hidden)
    normalized = rng.normal(0, 1.0, (rows, branches, hidden)).astype(np.float32)
    logits = rng.normal(0, 4.0, (rows, branches, hidden)).astype(np.float32)
    logits[0, 0, :3] = (0.0, -80.0, 80.0)
    buffers = []

    def upload(arr):
        host = np.ascontiguousarray(arr)
        dev = malloc(host.nbytes, runtime=runtime)
        buffers.append(dev)
        copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
        return dev

    def download(dev, like):
        out = np.empty_like(like)
        copy_device_to_host(host_array_ptr(out), dev, runtime=runtime)
        return out

    mixed_like = np.full((rows, hidden), -7.0, dtype=np.float32)
    try:
        d_norm = upload(normalized)
        d_logits = upload(logits)
        d_gate = upload(logits)
        d_fused = upload(mixed_like)
        d_pair = upload(mixed_like)
        qwen4_exp_gated_mean_sigmoid_f32(
            d_norm.ptr, d_logits.ptr, d_fused.ptr, rows, branches, hidden,
            library=library, runtime=runtime, store_gate=False,
        )
        qwen4_exp_sigmoid_f32(
            d_gate.ptr, d_gate.ptr, rows * branches * hidden,
            library=library, runtime=runtime,
        )
        qwen4_exp_gated_mean_f32(
            d_norm.ptr, d_gate.ptr, d_pair.ptr, rows, branches, hidden,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        fused = download(d_fused, mixed_like)
        pair = download(d_pair, mixed_like)
        kept = download(d_logits, logits)
    finally:
        for dev in reversed(buffers):
            free(dev, runtime=runtime)
    assert np.isfinite(pair).all()
    np.testing.assert_array_equal(fused.view(np.uint32), pair.view(np.uint32))
    np.testing.assert_array_equal(kept.view(np.uint32), logits.view(np.uint32))
