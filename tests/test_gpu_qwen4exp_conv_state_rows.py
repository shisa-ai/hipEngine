"""Per-token conv states of the Qwen4Exp exact conv prefill.

Contract: with ``state_rows_ptr`` the conv output and final state are
unchanged, and ``state_rows[t]`` equals the final state a prefill of the
first ``t + 1`` tokens leaves, bit for bit (what a speculative verifier
commits on a partial accept).
"""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen4_exp_linear_attn_conv_prefill_exact_f32,
)


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")

CHANNELS = 10240


def _run(hidden, state0, weight, rows, *, capture=False):
    runtime = get_hip_runtime()
    bufs = {
        "hidden": malloc(hidden[:rows].nbytes, runtime=runtime),
        "state": malloc(state0.nbytes, runtime=runtime),
        "weight": malloc(weight.nbytes, runtime=runtime),
        "out": malloc(rows * CHANNELS * 4, runtime=runtime),
        "rows": malloc(max(rows, 1) * state0.nbytes, runtime=runtime),
    }
    try:
        copy_host_to_device(bufs["hidden"], host_array_ptr(np.ascontiguousarray(hidden[:rows])),
                            runtime=runtime)
        copy_host_to_device(bufs["state"], host_array_ptr(state0), runtime=runtime)
        copy_host_to_device(bufs["weight"], host_array_ptr(weight), runtime=runtime)
        qwen4_exp_linear_attn_conv_prefill_exact_f32(
            bufs["hidden"].ptr, bufs["state"].ptr, bufs["weight"].ptr, bufs["out"].ptr,
            rows, CHANNELS, 4, state_rows_ptr=bufs["rows"].ptr if capture else 0)
        runtime.device_synchronize()
        out = np.empty((rows, CHANNELS), np.float32)
        final = np.empty_like(state0)
        per_row = np.empty((rows,) + state0.shape, np.float32)
        copy_device_to_host(host_array_ptr(out), bufs["out"], runtime=runtime)
        copy_device_to_host(host_array_ptr(final), bufs["state"], runtime=runtime)
        if capture:
            copy_device_to_host(host_array_ptr(per_row), bufs["rows"], runtime=runtime)
        return out, final, per_row
    finally:
        for buf in bufs.values():
            free(buf, runtime=runtime)


@pytest.mark.parametrize("rows", [1, 2, 4, 6])
def test_conv_state_rows_are_each_prefix_final_state(rows):
    rng = np.random.default_rng(rows)
    hidden = rng.normal(0, 1, (rows, CHANNELS)).astype(np.float32)
    state0 = rng.normal(0, 1, (CHANNELS, 4)).astype(np.float32)
    weight = rng.normal(0, 0.5, (CHANNELS, 4)).astype(np.float32)
    plain_out, plain_final, _ = _run(hidden, state0, weight, rows)
    out, final, per_row = _run(hidden, state0, weight, rows, capture=True)
    assert np.array_equal(out.view(np.uint32), plain_out.view(np.uint32))
    assert np.array_equal(final.view(np.uint32), plain_final.view(np.uint32))
    for t in range(rows):
        _, prefix_final, _ = _run(hidden, state0, weight, t + 1)
        assert np.array_equal(per_row[t].view(np.uint32), prefix_final.view(np.uint32)), t
