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
from hipengine.kernels.cpu_reference import gdn_prefill_recurrent_segments
from hipengine.kernels.cpu_reference.qwen4_exp import sigmoid_gated_rmsnorm


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_gdn_build_and_registry_contract() -> None:
    from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
        qwen4_exp_linear_attn_conv_prefill_exact_f32,
        register_qwen35_linear_attn_conv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        plan_qwen4_exp_gdn_build,
        qwen4_exp_gdn_decode_f32,
        qwen4_exp_gdn_prefill_f32,
        register_qwen4_exp_gdn_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_gdn_build()
    assert artifact.output_path.name == "qwen4_exp_gdn.so"
    register_qwen4_exp_gdn_kernels()
    register_qwen35_linear_attn_conv_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_conv_prefill",
            quant="qwen4_exp",
            variant="f32_decode_exact_k4",
        )
        is qwen4_exp_linear_attn_conv_prefill_exact_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrence_norm_gate",
            quant="f32_state",
            variant="qwen4exp_sigmoid_strict",
        )
        is qwen4_exp_gdn_decode_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrence_norm_gate",
            quant="f32_state",
            variant="qwen4exp_sigmoid_strict_prefill",
        )
        is qwen4_exp_gdn_prefill_f32
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gdn_decode_matches_cpu_at_production_geometry() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn,
        qwen4_exp_gdn_decode_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn(load=True)
    rng = np.random.default_rng(4035)
    k_heads, v_heads, head_dim = 16, 48, 128
    value_dim = 128
    q_raw = rng.normal(0.0, 0.05, size=(k_heads, head_dim)).astype(np.float32)
    k_raw = rng.normal(0.0, 0.05, size=(k_heads, head_dim)).astype(np.float32)
    # HF Qwen4Exp l2norm adds eps to the sum. A zero head must stay finite,
    # rather than evaluating 0 * rsqrt(0) in the fused decode kernel.
    q_raw[0] = 0.0
    k_raw[0] = 0.0
    value = rng.normal(0.0, 0.05, size=(v_heads, value_dim)).astype(np.float32)
    conv = np.concatenate((q_raw.reshape(-1), k_raw.reshape(-1), value.reshape(-1)))
    gate = rng.normal(0.0, 0.5, size=(v_heads, value_dim)).astype(np.float32)
    alpha = rng.normal(-0.2, 0.1, size=(v_heads,)).astype(np.float32)
    beta_logits = rng.normal(0.0, 0.2, size=(v_heads,)).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.1, size=(v_heads,)).astype(np.float32)
    log_a = rng.normal(-0.5, 0.1, size=(v_heads,)).astype(np.float32)
    # Frozen GGUF conversion stores -exp(A_log), not the source A_log value.
    a = -np.exp(log_a).astype(np.float32)
    norm = rng.normal(1.0, 0.05, size=(value_dim,)).astype(np.float32)
    state = rng.normal(
        0.0,
        0.01,
        size=(v_heads, head_dim, value_dim),
    ).astype(np.float32)

    mapping = np.arange(v_heads) % k_heads
    query = q_raw[mapping]
    key = k_raw[mapping]
    query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + np.float32(1e-6))
    query /= np.sqrt(np.float32(head_dim))
    key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + np.float32(1e-6))
    beta = 1.0 / (1.0 + np.exp(-beta_logits))
    decay = np.exp(a * np.log1p(np.exp(alpha + dt_bias)))
    core, expected_state = gdn_prefill_recurrent_segments(
        query[None],
        key[None],
        value[None],
        beta[None],
        decay[None],
        state[None],
        [0, 1],
        [0],
    )
    expected = sigmoid_gated_rmsnorm(core, norm, gate[None])[0]

    allocations = []
    try:
        d_conv = _upload(conv, runtime, allocations)
        d_gate = _upload(gate, runtime, allocations)
        d_alpha = _upload(alpha, runtime, allocations)
        d_beta = _upload(beta_logits, runtime, allocations)
        d_dt = _upload(dt_bias, runtime, allocations)
        d_a = _upload(a, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_state = _upload(state, runtime, allocations)
        d_output = _alloc(expected.shape, np.float32, runtime, allocations)
        qwen4_exp_gdn_decode_f32(
            d_conv.ptr,
            d_gate.ptr,
            d_alpha.ptr,
            d_beta.ptr,
            d_dt.ptr,
            d_a.ptr,
            d_norm.ptr,
            d_state.ptr,
            d_output.ptr,
            k_heads,
            v_heads,
            head_dim,
            value_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, expected.shape, np.float32, runtime)
        actual_state = _download(d_state, state.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(actual_state, expected_state[0], rtol=5e-5, atol=5e-5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_bulk_conv_prefill_matches_serial_decode_bits() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
        build_qwen35_linear_attn_conv,
        qwen35_linear_attn_conv_decode_f32,
        qwen4_exp_linear_attn_conv_prefill_exact_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(load=True)
    rng = np.random.default_rng(40381)
    rows, channels, kernel_size = 17, 256, 4
    hidden = rng.normal(0.0, 0.2, size=(rows, channels)).astype(np.float32)
    weight = rng.normal(0.0, 0.1, size=(channels, kernel_size)).astype(np.float32)
    state = rng.normal(0.0, 0.1, size=(channels, kernel_size)).astype(np.float32)
    allocations = []
    try:
        d_hidden = _upload(hidden, runtime, allocations)
        d_weight = _upload(weight, runtime, allocations)
        d_serial_state = _upload(state, runtime, allocations)
        d_bulk_state = _upload(state, runtime, allocations)
        d_serial = _alloc(hidden.shape, np.float32, runtime, allocations)
        d_bulk = _alloc(hidden.shape, np.float32, runtime, allocations)
        for row in range(rows):
            qwen35_linear_attn_conv_decode_f32(
                d_hidden.ptr + row * channels * 4,
                d_serial_state.ptr,
                d_weight.ptr,
                d_serial.ptr + row * channels * 4,
                channels,
                kernel_size,
                library=library,
                runtime=runtime,
            )
        qwen4_exp_linear_attn_conv_prefill_exact_f32(
            d_hidden.ptr,
            d_bulk_state.ptr,
            d_weight.ptr,
            d_bulk.ptr,
            rows,
            channels,
            kernel_size,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        serial = _download(d_serial, hidden.shape, np.float32, runtime)
        bulk = _download(d_bulk, hidden.shape, np.float32, runtime)
        serial_state = _download(
            d_serial_state, state.shape, np.float32, runtime
        )
        bulk_state = _download(d_bulk_state, state.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_array_equal(bulk, serial)
    np.testing.assert_array_equal(bulk_state, serial_state)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gdn_state_layout_roundtrip_is_exact_at_actual_shape() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        qwen4_exp_gdn_state_strict_to_transposed_f32,
        qwen4_exp_gdn_state_transposed_to_strict_f32,
    )
    runtime = get_hip_runtime()
    rng = np.random.default_rng(4051)
    heads, key_dim, value_dim = 48, 128, 128
    strict = rng.normal(0, 0.02, (heads, key_dim, value_dim)).astype(np.float32)
    allocations = []
    try:
        d_strict = _upload(strict, runtime, allocations)
        d_transposed = _alloc((heads, value_dim, key_dim), np.float32, runtime, allocations)
        d_roundtrip = _alloc(strict.shape, np.float32, runtime, allocations)
        qwen4_exp_gdn_state_strict_to_transposed_f32(d_strict.ptr, d_transposed.ptr, heads, key_dim, value_dim, runtime=runtime)
        qwen4_exp_gdn_state_transposed_to_strict_f32(d_transposed.ptr, d_roundtrip.ptr, heads, key_dim, value_dim, runtime=runtime)
        runtime.device_synchronize()
        transposed = _download(d_transposed, (heads, value_dim, key_dim), np.float32, runtime)
        roundtrip = _download(d_roundtrip, strict.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations): free(allocation, runtime=runtime)
    np.testing.assert_array_equal(transposed, np.transpose(strict, (0, 2, 1)))
    np.testing.assert_array_equal(roundtrip, strict)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gdn_transposed_decode_matches_strict_envelope() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        qwen4_exp_gdn_decode_f32, qwen4_exp_gdn_decode_transposed_f32,
        qwen4_exp_gdn_prefill_prepare_f32, qwen4_exp_gdn_prefill_sigmoid_gate_f32,
        qwen4_exp_gdn_state_transpose_inplace_f32,
    )
    runtime=get_hip_runtime(); rng=np.random.default_rng(4052); kh,vh,d=16,48,128
    qkv=2*kh*d+vh*d; core=vh*d
    conv=rng.normal(0,.05,qkv).astype(np.float32); gate=rng.normal(0,.05,core).astype(np.float32)
    alpha=rng.normal(0,.05,vh).astype(np.float32); beta=rng.normal(0,.05,vh).astype(np.float32)
    dt=rng.normal(0,.05,vh).astype(np.float32); a=-np.abs(rng.normal(.2,.05,vh)).astype(np.float32)
    norm=rng.normal(1,.05,d).astype(np.float32); state=rng.normal(0,.02,(vh,d,d)).astype(np.float32)
    allocations=[]
    try:
        ds=[_upload(x,runtime,allocations) for x in (conv,gate,alpha,beta,dt,a,norm,state,state)]
        dc, dg, da, db, ddt, d_a, dn, strict_state, trans_state=ds
        strict_out=_alloc((core,),np.float32,runtime,allocations); trans_out=_alloc((core,),np.float32,runtime,allocations)
        prepared=_alloc((qkv,),np.float32,runtime,allocations); prep_beta=_alloc((vh,),np.float32,runtime,allocations); prep_decay=_alloc((vh,),np.float32,runtime,allocations); trans_core=_alloc((core,),np.float32,runtime,allocations)
        qwen4_exp_gdn_decode_f32(dc.ptr,dg.ptr,da.ptr,db.ptr,ddt.ptr,d_a.ptr,dn.ptr,strict_state.ptr,strict_out.ptr,kh,vh,d,d,runtime=runtime)
        qwen4_exp_gdn_prefill_prepare_f32(dc.ptr,da.ptr,db.ptr,ddt.ptr,d_a.ptr,prepared.ptr,prepared.ptr+kh*d*4,prepared.ptr+2*kh*d*4,prep_beta.ptr,prep_decay.ptr,1,kh,vh,d,d,runtime=runtime)
        qwen4_exp_gdn_state_transpose_inplace_f32(trans_state.ptr,vh,d,runtime=runtime)
        qwen4_exp_gdn_decode_transposed_f32(prepared.ptr,prepared.ptr+kh*d*4,prepared.ptr+2*kh*d*4,prep_beta.ptr,prep_decay.ptr,trans_state.ptr,trans_core.ptr,kh,vh,d,d,runtime=runtime)
        qwen4_exp_gdn_prefill_sigmoid_gate_f32(trans_core.ptr,dg.ptr,dn.ptr,1,vh,d,runtime=runtime)
        runtime.memcpy(trans_out.ptr,trans_core.ptr,trans_out.nbytes,3)
        qwen4_exp_gdn_state_transpose_inplace_f32(trans_state.ptr,vh,d,runtime=runtime)
        runtime.device_synchronize()
        strict=_download(strict_out,(core,),np.float32,runtime); trans=_download(trans_out,(core,),np.float32,runtime)
        strict_matrix=_download(strict_state,state.shape,np.float32,runtime); trans_matrix=_download(trans_state,state.shape,np.float32,runtime)
    finally:
        for allocation in reversed(allocations): free(allocation,runtime=runtime)
    np.testing.assert_allclose(trans,strict,rtol=5e-4,atol=5e-5)
    np.testing.assert_allclose(trans_matrix,strict_matrix,rtol=2e-5,atol=2e-7)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gdn_prefill_is_exact_to_serial_decode() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn,
        qwen4_exp_gdn_decode_f32,
        qwen4_exp_gdn_prefill_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn(load=True)
    rng = np.random.default_rng(4038)
    rows, k_heads, v_heads, head_dim = 5, 2, 4, 8
    qkv_width = 2 * k_heads * head_dim + v_heads * head_dim
    core_width = v_heads * head_dim
    conv = rng.normal(0.0, 0.05, size=(rows, qkv_width)).astype(np.float32)
    gate = rng.normal(0.0, 0.5, size=(rows, core_width)).astype(np.float32)
    alpha = rng.normal(-0.2, 0.1, size=(rows, v_heads)).astype(np.float32)
    beta = rng.normal(0.0, 0.2, size=(rows, v_heads)).astype(np.float32)
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
        d_beta = _upload(beta, runtime, allocations)
        d_dt = _upload(dt_bias, runtime, allocations)
        d_a = _upload(a, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_serial_state = _upload(initial_state, runtime, allocations)
        d_prefill_state = _upload(initial_state, runtime, allocations)
        d_serial = _alloc((rows, core_width), np.float32, runtime, allocations)
        d_prefill = _alloc((rows, core_width), np.float32, runtime, allocations)
        for row in range(rows):
            qwen4_exp_gdn_decode_f32(
                d_conv.ptr + row * qkv_width * 4,
                d_gate.ptr + row * core_width * 4,
                d_alpha.ptr + row * v_heads * 4,
                d_beta.ptr + row * v_heads * 4,
                d_dt.ptr,
                d_a.ptr,
                d_norm.ptr,
                d_serial_state.ptr,
                d_serial.ptr + row * core_width * 4,
                k_heads,
                v_heads,
                head_dim,
                head_dim,
                library=library,
                runtime=runtime,
            )
        qwen4_exp_gdn_prefill_f32(
            d_conv.ptr,
            d_gate.ptr,
            d_alpha.ptr,
            d_beta.ptr,
            d_dt.ptr,
            d_a.ptr,
            d_norm.ptr,
            d_prefill_state.ptr,
            d_prefill.ptr,
            rows,
            k_heads,
            v_heads,
            head_dim,
            head_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        serial = _download(d_serial, (rows, core_width), np.float32, runtime)
        prefill = _download(d_prefill, (rows, core_width), np.float32, runtime)
        serial_state = _download(
            d_serial_state, initial_state.shape, np.float32, runtime
        )
        prefill_state = _download(
            d_prefill_state, initial_state.shape, np.float32, runtime
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(prefill, serial)
    np.testing.assert_array_equal(prefill_state, serial_state)


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
