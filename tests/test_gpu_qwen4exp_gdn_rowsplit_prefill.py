"""Row-split GDN prefill recurrence against the column-warp owner.

The row-split kernel is not bit-identical to the column-warp owners: it decays
the state first, applies the key and query scales to the two dot products and
sums in a different order. The contract here is FP32 round-off agreement on
the unnormalized-then-gated output and on the final state, at every tail
shape, plus run-to-run determinism.
"""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    build_qwen4_exp_gdn_dpp,
    qwen4_exp_gdn_prefill_columnwarps_f32,
    qwen4_exp_gdn_prefill_rowsplit_f32,
)


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def _run(function, host, state0, rows, v_heads, library, runtime):
    k_heads, dim = 16, 128
    dev = [malloc(a.nbytes, runtime=runtime) for a in host]
    state = malloc(state0.nbytes, runtime=runtime)
    out = malloc(rows * v_heads * dim * 4, runtime=runtime)
    try:
        for d, a in zip(dev, host):
            copy_host_to_device(d, host_array_ptr(a), runtime=runtime)
        copy_host_to_device(state, host_array_ptr(state0), runtime=runtime)
        function(*[d.ptr for d in dev], state.ptr, out.ptr, rows, k_heads, v_heads,
                 dim, dim, library=library, runtime=runtime)
        runtime.device_synchronize()
        core = np.empty((rows, v_heads * dim), np.float32)
        final = np.empty_like(state0)
        copy_device_to_host(host_array_ptr(core), out, runtime=runtime)
        copy_device_to_host(host_array_ptr(final), state, runtime=runtime)
        return core, final
    finally:
        for buf in (*dev, state, out):
            free(buf, runtime=runtime)


@pytest.mark.parametrize(
    "v_heads,rows", [(48, 1), (48, 3), (48, 5), (48, 17), (32, 64), (48, 513)]
)
def test_rowsplit_matches_the_column_warp_owner_to_round_off(v_heads, rows):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn_dpp(multi_column=True)
    rng = np.random.default_rng(31 * v_heads + rows)
    k_heads, dim = 16, 128
    qkv_width = 2 * k_heads * dim + v_heads * dim
    host = [
        rng.normal(0.0, 0.05, (rows, qkv_width)).astype(np.float32),
        rng.normal(0.0, 0.5, (rows, v_heads * dim)).astype(np.float32),
        rng.normal(-0.2, 0.1, (rows, v_heads)).astype(np.float32),
        rng.normal(0.0, 0.2, (rows, v_heads)).astype(np.float32),
        rng.normal(-1.0, 0.1, v_heads).astype(np.float32),
        (-np.exp(rng.normal(-0.5, 0.1, v_heads))).astype(np.float32),
        rng.normal(1.0, 0.05, dim).astype(np.float32),
    ]
    state0 = rng.normal(0.0, 0.01, (v_heads, dim, dim)).astype(np.float32)
    parent, parent_state = _run(
        qwen4_exp_gdn_prefill_columnwarps_f32, host, state0, rows, v_heads,
        library, runtime)
    core, state = _run(
        qwen4_exp_gdn_prefill_rowsplit_f32, host, state0, rows, v_heads,
        library, runtime)
    repeat, repeat_state = _run(
        qwen4_exp_gdn_prefill_rowsplit_f32, host, state0, rows, v_heads,
        library, runtime)
    assert np.isfinite(core).all() and np.isfinite(state).all()
    np.testing.assert_array_equal(repeat.view(np.uint32), core.view(np.uint32))
    np.testing.assert_array_equal(repeat_state.view(np.uint32), state.view(np.uint32))
    assert np.linalg.norm(core - parent) <= 2e-6 * np.linalg.norm(parent)
    assert np.linalg.norm(state - parent_state) <= 2e-6 * np.linalg.norm(parent_state)
    assert np.abs(core - parent).max() <= 1e-5 * max(1.0, np.abs(parent).max())
