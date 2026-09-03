"""PF-5 RED tests: 32-warp GDN prefill variant (M6 geometry port).

Declaration entry:
``worklog/entries/20260903T152319.089915Z-lhl-qwen4exp-pf5-gdn-m6-declaration-55a7d9.md``

- **Arithmetic class:** T0 candidate. The lever changes launch geometry only
  (32 warps/block for the S_v=128 GDN recurrence family; halo-box PR11
  ``gated_delta_net.cu`` lineage). FP32 accumulation order is preserved, so
  the binding contract is **bit-exact** parity against the registered strict
  owner ``qwen4_exp_gdn_prefill_f32`` (variant
  ``qwen4exp_sigmoid_strict_prefill``). If implementation forces a
  reassociation, the class escalates to T1 and the unit stops.
- **Strict fallback:** the unmodified ``gdn_recurrence_norm_gate``
  (``f32_state``) owner, retained under its registry key.

RED semantics (unmodified path): the candidate function
``qwen4_exp_gdn_prefill_w32_f32`` and its registry variant
``qwen4exp_gdn_w32_prefill`` do not exist yet, so both tests below fail
(ImportError / MissingKernelError). They must pass after the port lands.
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


def test_qwen4_exp_gdn_w32_prefill_registry_contract() -> None:
    """The w32 prefill variant must be registered on its four-axis key.

    RED on the unmodified path: the candidate function is not implemented, so
    the import fails (and/or resolve raises MissingKernelError). Note the
    resolver's fallback chain must NOT satisfy this key: the exact variant
    must resolve to the candidate function itself, never to the strict owner
    or a broadened candidate.
    """
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        qwen4_exp_gdn_prefill_w32_f32,
        register_qwen4_exp_gdn_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_qwen4_exp_gdn_kernels()
    resolved = resolve(
        backend="hip_gfx1100",
        layer="gdn_recurrence_norm_gate",
        quant="f32_state",
        variant="qwen4exp_gdn_w32_prefill",
        missing="none",
    )
    assert resolved is qwen4_exp_gdn_prefill_w32_f32


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("v_heads", [32, 48, 64])
def test_qwen4_exp_gdn_w32_prefill_exact_vs_strict_owner(v_heads: int) -> None:
    """The w32 prefill variant must be bit-exact vs the strict owner (T0).

    Production-like geometry: multi-row prefill (rows=16), H in {32, 48, 64}
    value heads with 16 key heads, S_k = S_v = 128. Oracle = the current
    strict owner ``qwen4_exp_gdn_prefill_f32`` outputs and final recurrent
    state, bit-for-bit.

    RED on the unmodified path: the candidate function does not exist, so the
    import fails before any GPU work runs.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn,
        qwen4_exp_gdn_prefill_f32,
        qwen4_exp_gdn_prefill_w32_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn(load=True)
    rng = np.random.default_rng(9055 + v_heads)
    rows = 16
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
        d_strict_state = _upload(initial_state, runtime, allocations)
        d_w32_state = _upload(initial_state, runtime, allocations)
        d_strict = _alloc((rows, core_width), np.float32, runtime, allocations)
        d_w32 = _alloc((rows, core_width), np.float32, runtime, allocations)
        for fn, d_state, d_out in (
            (qwen4_exp_gdn_prefill_f32, d_strict_state, d_strict),
            (qwen4_exp_gdn_prefill_w32_f32, d_w32_state, d_w32),
        ):
            fn(
                d_conv.ptr,
                d_gate.ptr,
                d_alpha.ptr,
                d_beta.ptr,
                d_dt.ptr,
                d_a.ptr,
                d_norm.ptr,
                d_state.ptr,
                d_out.ptr,
                rows,
                k_heads,
                v_heads,
                head_dim,
                head_dim,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        strict_out = _download(d_strict, (rows, core_width), np.float32, runtime)
        w32_out = _download(d_w32, (rows, core_width), np.float32, runtime)
        strict_state = _download(
            d_strict_state, initial_state.shape, np.float32, runtime
        )
        w32_state = _download(d_w32_state, initial_state.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(w32_out, strict_out)
    np.testing.assert_array_equal(w32_state, strict_state)


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
