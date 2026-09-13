"""DPP must preserve the parent reduction and carried state on gfx1151."""

import ctypes

import numpy as np
import pytest


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


@pytest.mark.skipif(not hip_available(), reason="HIP runtime unavailable")
@pytest.mark.parametrize("rows", [16, 17, 64, 512, 1024])
@pytest.mark.parametrize("multi_column", [False, True])
def test_dpp_prefill_matches_all_outputs_and_carried_state(rows, multi_column):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn, build_qwen4_exp_gdn_dpp,
        qwen4_exp_gdn_prefill_tiled16_f32,
    )
    from tests.test_gpu_qwen4_exp_gdn_tiled16_prefill import _upload, _alloc, _download

    runtime = get_hip_runtime()
    from hipengine.kernels.backends import detect_hip_target_arches
    if "gfx1151" not in detect_hip_target_arches():
        pytest.skip("DPP variant qualified only for gfx1151")
    parent = build_qwen4_exp_gdn()
    candidate = build_qwen4_exp_gdn_dpp(multi_column=multi_column)
    rng = np.random.default_rng(73194 + rows)
    heads, dim = 48, 128
    arrays = [
        rng.normal(0, .05, (rows, 10240)).astype(np.float32),
        rng.normal(0, .5, (rows, heads * dim)).astype(np.float32),
        rng.normal(-.2, .1, (rows, heads)).astype(np.float32),
        rng.normal(0, .2, (rows, heads)).astype(np.float32),
        rng.normal(-1, .1, heads).astype(np.float32),
        -np.exp(rng.normal(-.5, .1, heads)).astype(np.float32),
        rng.normal(1, .05, dim).astype(np.float32),
    ]
    initial = rng.normal(0, .01, (heads, dim, dim)).astype(np.float32)
    allocations = []
    try:
        inputs = [_upload(a, runtime, allocations) for a in arrays]
        states = [_upload(initial, runtime, allocations) for _ in range(2)]
        outputs = [_alloc((rows, heads * dim), np.float32, runtime, allocations) for _ in range(2)]
        for iteration in range(2):
            for lib, state, output in zip((parent, candidate), states, outputs):
                qwen4_exp_gdn_prefill_tiled16_f32(
                    *(a.ptr for a in inputs), state.ptr, output.ptr,
                    rows, 16, heads, dim, dim, library=lib, runtime=runtime,
                )
            runtime.device_synchronize()
            parent_out = _download(outputs[0], (rows, heads * dim), np.float32, runtime)
            candidate_out = _download(outputs[1], (rows, heads * dim), np.float32, runtime)
            parent_state = _download(states[0], initial.shape, np.float32, runtime)
            candidate_state = _download(states[1], initial.shape, np.float32, runtime)
            if multi_column:
                np.testing.assert_allclose(candidate_out, parent_out, rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(candidate_state, parent_state, rtol=2e-4, atol=2e-5)
            else:
                assert parent_out.tobytes() == candidate_out.tobytes()
                assert parent_state.tobytes() == candidate_state.tobytes()
            if rows == 16 and iteration == 0:
                from hipengine.kernels.cpu_reference import gdn_prefill_recurrent_segments
                from hipengine.kernels.cpu_reference.qwen4_exp import sigmoid_gated_rmsnorm

                conv, gate, alpha, beta_logits, dt, a, norm = arrays
                mapping = np.arange(heads) % 16
                query = conv[:, :2048].reshape(rows, 16, dim)[:, mapping].copy()
                key = conv[:, 2048:4096].reshape(rows, 16, dim)[:, mapping].copy()
                query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + np.float32(1e-6))
                query /= np.sqrt(np.float32(dim))
                key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + np.float32(1e-6))
                core, cpu_state = gdn_prefill_recurrent_segments(
                    query, key, conv[:, 4096:].reshape(rows, heads, dim),
                    1 / (1 + np.exp(-beta_logits)),
                    np.exp(a * np.log1p(np.exp(alpha + dt))),
                    initial[None], [0, rows], [0],
                )
                expected = sigmoid_gated_rmsnorm(core, norm, gate.reshape(rows, heads, dim))
                actual = _download(outputs[1], (rows, heads, dim), np.float32, runtime)
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(
                    _download(states[1], initial.shape, np.float32, runtime),
                    cpu_state[0], rtol=2e-4, atol=2e-5,
                )
        if multi_column:
            repeated = []
            for _ in range(3):
                copy_host_to_device(states[1], host_array_ptr(initial), runtime=runtime)
                qwen4_exp_gdn_prefill_tiled16_f32(
                    *(a.ptr for a in inputs), states[1].ptr, outputs[1].ptr,
                    rows, 16, heads, dim, dim, library=candidate, runtime=runtime,
                )
                runtime.device_synchronize()
                repeated.append((
                    _download(outputs[1], (rows, heads * dim), np.float32, runtime).tobytes(),
                    _download(states[1], initial.shape, np.float32, runtime).tobytes(),
                ))
            assert repeated[0] == repeated[1] == repeated[2]
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
