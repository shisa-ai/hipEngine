"""PF-5 RED/parity tests for the token-tile-16 GDN prefill candidate.

The candidate adapts cooperative staging from halo-box PR11
``gated_delta_net_kda_tiled_128_cuda`` (``a7ad7b7f``) to the in-tree
Qwen4Exp arithmetic and binding Hk=16/Hv=48/D=128 shape. The external tile-16
branch itself is H=32 KDA and is inactive on this model.

Arithmetic class: T0. The candidate must preserve the production columnwarp
owner's per-lane r=0..3 accumulation order, XOR shuffle reductions, recurrent
update order, and FP32 outputs/state bit-for-bit. The registered serial
``qwen4exp_sigmoid_strict_prefill`` route remains the strict fallback; the
columnwarp owner remains the immediate production rollback.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_gdn_tiled16_prefill_registry_contract() -> None:
    """The exact candidate, parent, and strict fallback keys must resolve."""
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        qwen4_exp_gdn_prefill_columnwarps_f32,
        qwen4_exp_gdn_prefill_f32,
        qwen4_exp_gdn_prefill_tiled16_f32,
        register_qwen4_exp_gdn_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_qwen4_exp_gdn_kernels()
    candidate = resolve(
        backend="hip_gfx1100",
        layer="gdn_recurrence_norm_gate",
        quant="f32_state",
        variant="qwen4exp_gdn_tiled16_prefill",
        missing="none",
    )
    parent = resolve(
        backend="hip_gfx1100",
        layer="gdn_recurrence_norm_gate",
        quant="f32_state",
        variant="qwen4exp_gdn_columnwarps_prefill",
        missing="none",
    )
    strict = resolve(
        backend="hip_gfx1100",
        layer="gdn_recurrence_norm_gate",
        quant="f32_state",
        variant="qwen4exp_sigmoid_strict_prefill",
        missing="none",
    )
    assert candidate is qwen4_exp_gdn_prefill_tiled16_f32
    assert parent is qwen4_exp_gdn_prefill_columnwarps_f32
    assert strict is qwen4_exp_gdn_prefill_f32


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("v_heads", "rows"),
    [(32, 16), (48, 16), (48, 17), (48, 64)],
)
def test_qwen4_exp_gdn_tiled16_prefill_exact_vs_columnwarps(
    v_heads: int, rows: int
) -> None:
    """Tile-16 output and state must be exact, including binding Hv=48 tails."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn,
        qwen4_exp_gdn_prefill_columnwarps_f32,
        qwen4_exp_gdn_prefill_tiled16_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn(load=True)
    rng = np.random.default_rng(101_600 + 100 * v_heads + rows)
    k_heads = 16
    head_dim = 128
    qkv_width = 2 * k_heads * head_dim + v_heads * head_dim
    core_width = v_heads * head_dim
    conv = rng.normal(0.0, 0.05, size=(rows, qkv_width)).astype(np.float32)
    gate = rng.normal(0.0, 0.5, size=(rows, core_width)).astype(np.float32)
    alpha = rng.normal(-0.2, 0.1, size=(rows, v_heads)).astype(np.float32)
    beta_logits = rng.normal(0.0, 0.2, size=(rows, v_heads)).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.1, size=v_heads).astype(np.float32)
    a = -np.exp(rng.normal(-0.5, 0.1, size=v_heads)).astype(np.float32)
    norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    initial_state = rng.normal(
        0.0, 0.01, size=(v_heads, head_dim, head_dim)
    ).astype(np.float32)

    allocations = []
    try:
        d_conv = _upload(conv, runtime, allocations)
        d_gate = _upload(gate, runtime, allocations)
        d_alpha = _upload(alpha, runtime, allocations)
        d_beta = _upload(beta_logits, runtime, allocations)
        d_dt = _upload(dt_bias, runtime, allocations)
        d_a = _upload(a, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_parent_state = _upload(initial_state, runtime, allocations)
        d_candidate_state = _upload(initial_state, runtime, allocations)
        d_parent = _alloc((rows, core_width), np.float32, runtime, allocations)
        d_candidate = _alloc((rows, core_width), np.float32, runtime, allocations)
        for function, state, output in (
            (
                qwen4_exp_gdn_prefill_columnwarps_f32,
                d_parent_state,
                d_parent,
            ),
            (
                qwen4_exp_gdn_prefill_tiled16_f32,
                d_candidate_state,
                d_candidate,
            ),
        ):
            function(
                d_conv.ptr,
                d_gate.ptr,
                d_alpha.ptr,
                d_beta.ptr,
                d_dt.ptr,
                d_a.ptr,
                d_norm.ptr,
                state.ptr,
                output.ptr,
                rows,
                k_heads,
                v_heads,
                head_dim,
                head_dim,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        parent = _download(d_parent, (rows, core_width), np.float32, runtime)
        candidate = _download(
            d_candidate, (rows, core_width), np.float32, runtime
        )
        parent_state = _download(
            d_parent_state, initial_state.shape, np.float32, runtime
        )
        candidate_state = _download(
            d_candidate_state, initial_state.shape, np.float32, runtime
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(candidate, parent)
    np.testing.assert_array_equal(candidate_state, parent_state)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
