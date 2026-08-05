from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear import dense_gemv as dense_gemv_module
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_gemv_bf16_f32w_bf16_out,
)
from hipengine.kernels.hip_gfx1100.linear_attn import gdn as gdn_module
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

_COMPOSITE_LAYER = (
    "linear_attn_alpha_beta+gdn_chain_recurrent_rmsnorm_gate+cast+snapshot"
)
_COMPOSITE_QUANT = "f32+gguf_q5_k_t16_v1"
_COMPOSITE_VARIANT = (
    "bf16_k5120_n48_hk16_hv48_d128_exact_state_rows_tloop_f32_bf16_out"
)
_COMPOSITE_KEY = KernelKey(
    "hip_gfx1100",
    _COMPOSITE_LAYER,
    _COMPOSITE_QUANT,
    _COMPOSITE_VARIANT,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


def _composite_wrapper():
    wrapper = getattr(
        gdn_module,
        "qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out",
        None,
    )
    assert callable(wrapper), "dependent alpha/beta-to-GDN wrapper is not implemented"
    return wrapper


def test_dependent_pair_gdn_registers_only_on_screened_gfx1100_backend() -> None:
    gdn_module.register_qwen35_linear_attn_gdn_kernels(replace=True)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)

    assert resolve(
        backend=_COMPOSITE_KEY.backend,
        layer=_COMPOSITE_KEY.layer,
        quant=_COMPOSITE_KEY.quant,
        variant=_COMPOSITE_KEY.variant,
    ) is _composite_wrapper()
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _COMPOSITE_KEY.layer,
            _COMPOSITE_KEY.quant,
            _COMPOSITE_KEY.variant,
        )
    )


def test_dependent_pair_gdn_wrapper_uses_one_symbol_and_validates_shape() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []
    symbol = "hipengine_qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out"

    class Launch:
        argtypes = None
        restype = None

        def __call__(self, *args):
            calls.append((symbol, args))
            return 0

    wrapper = _composite_wrapper()
    runtime = SimpleNamespace(check=lambda err: pytest.fail(f"unexpected HIP error {err}"))
    library = SimpleNamespace(**{symbol: Launch()})
    pointers = tuple(range(10, 25))
    for rows in (1, 2, 3, 4):
        wrapper(
            *pointers,
            1.0e-6,
            rows,
            5120,
            16,
            48,
            128,
            128,
            stream=17,
            library=library,
            runtime=runtime,
        )

    assert [symbol for symbol, _args in calls] == [
        "hipengine_qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out"
    ] * 4
    def value(item: object) -> int:
        return int(getattr(item, "value", item))

    assert [value(args[16]) for _symbol, args in calls] == [1, 2, 3, 4]
    assert all(tuple(value(item) for item in args[17:22]) == (5120, 16, 48, 128, 128) for _, args in calls)
    assert all(value(args[22]) == 17 for _, args in calls)

    invalid = (
        ((0, 5120, 16, 48, 128, 128), "rows must be between 1 and 4"),
        ((5, 5120, 16, 48, 128, 128), "rows must be between 1 and 4"),
        ((2, 4096, 16, 48, 128, 128), "in_features must equal 5120"),
        ((2, 5120, 8, 48, 128, 128), "num_k_heads must equal 16"),
        ((2, 5120, 16, 32, 128, 128), "num_v_heads must equal 48"),
        ((2, 5120, 16, 48, 64, 128), "head_k_dim must equal 128"),
        ((2, 5120, 16, 48, 128, 64), "head_v_dim must equal 128"),
    )
    for dimensions, message in invalid:
        with pytest.raises(ValueError, match=message):
            wrapper(*pointers, 1.0e-6, *dimensions)
    with pytest.raises(ValueError, match="threads must equal 256"):
        wrapper(*pointers, 1.0e-6, 2, 5120, 16, 48, 128, 128, threads=128)


def test_dependent_pair_gdn_stays_primitive_only_after_runtime_rejection() -> None:
    from hipengine.runtime import gguf_linear
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    assert not hasattr(gguf_linear, "resolve_gguf_linear_pair_gdn_snapshot")
    assert not hasattr(
        Qwen35GGUFFullStackRunner,
        "_try_run_linear_attention_alpha_beta_gdn_rows_exact",
    )


def _softmax_kl_top1(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    expected64 = np.asarray(expected, dtype=np.float64)
    actual64 = np.asarray(actual, dtype=np.float64)
    p = np.exp(expected64 - np.max(expected64, axis=-1, keepdims=True))
    p /= np.sum(p, axis=-1, keepdims=True)
    q = np.exp(actual64 - np.max(actual64, axis=-1, keepdims=True))
    q /= np.sum(q, axis=-1, keepdims=True)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    top1 = np.mean(np.argmax(expected64, axis=-1) == np.argmax(actual64, axis=-1))
    return float(np.max(kl)), float(top1)


def _cpu_gdn(
    conv_out: np.ndarray,
    gate_bits: np.ndarray,
    alpha_bits: np.ndarray,
    beta_bits: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    base_state: np.ndarray,
) -> np.ndarray:
    rows, channels = conv_out.shape
    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = head_v_dim = 128
    key_dim = num_k_heads * head_k_dim
    assert channels == 2 * key_dim + num_v_heads * head_v_dim
    gate = bf16_to_float32(gate_bits).reshape(rows, num_v_heads, head_v_dim)
    alpha = bf16_to_float32(alpha_bits).reshape(rows, num_v_heads)
    beta_raw = bf16_to_float32(beta_bits).reshape(rows, num_v_heads)
    state = np.array(base_state, dtype=np.float32, copy=True)
    out = np.empty((rows, num_v_heads, head_v_dim), dtype=np.float32)
    for row in range(rows):
        for v_head in range(num_v_heads):
            k_head = v_head % num_k_heads
            q = conv_out[row, k_head * head_k_dim : (k_head + 1) * head_k_dim]
            k = conv_out[
                row,
                key_dim + k_head * head_k_dim : key_dim + (k_head + 1) * head_k_dim,
            ]
            value = conv_out[
                row,
                2 * key_dim + v_head * head_v_dim : 2 * key_dim + (v_head + 1) * head_v_dim,
            ]
            q_scale = np.float32(1.0 / max(float(np.linalg.norm(q)), 1.0e-6)) * np.float32(
                1.0 / np.sqrt(float(head_k_dim))
            )
            k_scale = np.float32(1.0 / max(float(np.linalg.norm(k)), 1.0e-6))
            q_norm = np.asarray(q * q_scale, dtype=np.float32)
            k_norm = np.asarray(k * k_scale, dtype=np.float32)
            beta = np.float32(1.0 / (1.0 + np.exp(-np.float64(beta_raw[row, v_head]))))
            softplus = np.float32(
                np.log1p(np.exp(-abs(float(alpha[row, v_head] + dt_bias[v_head]))))
                + max(float(alpha[row, v_head] + dt_bias[v_head]), 0.0)
            )
            decay = np.float32(np.exp(-np.exp(np.float64(a_log[v_head])) * softplus))
            old = state[v_head]
            decayed = np.asarray(old * decay, dtype=np.float32)
            kv_mem = np.sum(k_norm[:, None] * decayed, axis=0, dtype=np.float32)
            delta = np.asarray((value - kv_mem) * beta, dtype=np.float32)
            new = np.asarray(decayed + k_norm[:, None] * delta[None, :], dtype=np.float32)
            state[v_head] = new
            raw = np.sum(q_norm[:, None] * new, axis=0, dtype=np.float32)
            inv_rms = np.float32(
                1.0 / np.sqrt(float(np.mean(raw * raw, dtype=np.float32)) + 1.0e-6)
            )
            gate_value = gate[row, v_head]
            silu_gate = np.asarray(gate_value / (1.0 + np.exp(-gate_value)), dtype=np.float32)
            out[row, v_head] = np.asarray(
                raw * inv_rms * norm_weight * silu_gate,
                dtype=np.float32,
            )
    return out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (1, 2, 3, 4))
def test_dependent_pair_gdn_is_scalar_bit_exact_and_passes_cpu_gate(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    dense_library = dense_gemv_module.build_dense_gemv(load=True)
    gdn_library = gdn_module.build_qwen35_linear_attn_gdn(load=True)
    composite = _composite_wrapper()
    in_features = 5120
    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = head_v_dim = 128
    channels = 10240
    rng = np.random.default_rng(0xA6D0 + rows)
    norm_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.20, size=(rows, in_features)).astype(np.float32)
    )
    alpha_weight = rng.normal(0.0, 0.005, size=(num_v_heads, in_features)).astype(np.float32)
    beta_weight = rng.normal(0.0, 0.005, size=(num_v_heads, in_features)).astype(np.float32)
    conv_out = rng.normal(0.0, 0.25, size=(rows, channels)).astype(np.float32)
    gate_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.25, size=(rows, num_v_heads, head_v_dim)).astype(np.float32)
    )
    dt_bias = rng.normal(-0.2, 0.03, size=(num_v_heads,)).astype(np.float32)
    a_log = rng.normal(-1.0, 0.04, size=(num_v_heads,)).astype(np.float32)
    norm_weight = rng.normal(1.0, 0.02, size=(head_v_dim,)).astype(np.float32)
    base_state = rng.normal(
        0.0,
        0.01,
        size=(num_v_heads, head_k_dim, head_v_dim),
    ).astype(np.float32)

    linear_shape = (rows, num_v_heads)
    state_shape = (rows, num_v_heads, head_k_dim, head_v_dim)
    out_shape = (rows, num_v_heads, head_v_dim)
    host = {
        "control_alpha": np.empty(linear_shape, dtype=np.uint16),
        "control_beta": np.empty(linear_shape, dtype=np.uint16),
        "candidate_alpha": np.empty(linear_shape, dtype=np.uint16),
        "candidate_beta": np.empty(linear_shape, dtype=np.uint16),
        "control_state": np.empty(state_shape, dtype=np.float32),
        "candidate_state": np.empty(state_shape, dtype=np.float32),
        "control_snapshot": np.empty(base_state.shape, dtype=np.float32),
        "candidate_snapshot": np.empty(base_state.shape, dtype=np.float32),
        "control_out": np.empty(out_shape, dtype=np.float32),
        "candidate_out": np.empty(out_shape, dtype=np.float32),
        "control_bf16": np.empty(out_shape, dtype=np.uint16),
        "candidate_bf16": np.empty(out_shape, dtype=np.uint16),
    }
    buffers = []

    def device(array: np.ndarray):
        allocation = malloc(array.nbytes, runtime=runtime)
        buffers.append(allocation)
        copy_host_to_device(allocation, host_array_ptr(np.ascontiguousarray(array)), runtime=runtime)
        return allocation

    def empty(array: np.ndarray):
        allocation = malloc(array.nbytes, runtime=runtime)
        buffers.append(allocation)
        return allocation

    try:
        dnorm = device(norm_bits)
        dwa = device(alpha_weight)
        dwb = device(beta_weight)
        dconv = device(conv_out)
        dgate = device(gate_bits)
        ddt = device(dt_bias)
        da_log = device(a_log)
        dnorm_weight = device(norm_weight)
        dbase = device(base_state)
        device_outputs = {name: empty(array) for name, array in host.items()}

        dense_gemv_bf16_f32w_bf16_out(
            dnorm.ptr,
            dwa.ptr,
            device_outputs["control_alpha"].ptr,
            rows,
            in_features,
            num_v_heads,
            threads=256,
            library=dense_library,
            runtime=runtime,
        )
        dense_gemv_bf16_f32w_bf16_out(
            dnorm.ptr,
            dwb.ptr,
            device_outputs["control_beta"].ptr,
            rows,
            in_features,
            num_v_heads,
            threads=256,
            library=dense_library,
            runtime=runtime,
        )
        gdn_module.qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_f32_bf16_out(
            dconv.ptr,
            dgate.ptr,
            device_outputs["control_alpha"].ptr,
            device_outputs["control_beta"].ptr,
            ddt.ptr,
            da_log.ptr,
            dnorm_weight.ptr,
            dbase.ptr,
            device_outputs["control_state"].ptr,
            device_outputs["control_snapshot"].ptr,
            0,
            device_outputs["control_out"].ptr,
            device_outputs["control_bf16"].ptr,
            1.0e-6,
            rows,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
            runtime=runtime,
        )
        composite(
            dnorm.ptr,
            dwa.ptr,
            dwb.ptr,
            device_outputs["candidate_alpha"].ptr,
            device_outputs["candidate_beta"].ptr,
            dconv.ptr,
            dgate.ptr,
            ddt.ptr,
            da_log.ptr,
            dnorm_weight.ptr,
            dbase.ptr,
            device_outputs["candidate_state"].ptr,
            device_outputs["candidate_snapshot"].ptr,
            device_outputs["candidate_out"].ptr,
            device_outputs["candidate_bf16"].ptr,
            1.0e-6,
            rows,
            in_features,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for name, array in host.items():
            copy_device_to_host(host_array_ptr(array), device_outputs[name], runtime=runtime)
    finally:
        for allocation in reversed(buffers):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(host["candidate_alpha"], host["control_alpha"])
    np.testing.assert_array_equal(host["candidate_beta"], host["control_beta"])
    np.testing.assert_array_equal(
        host["candidate_state"].view(np.uint32),
        host["control_state"].view(np.uint32),
    )
    np.testing.assert_array_equal(
        host["candidate_snapshot"].view(np.uint32),
        host["control_snapshot"].view(np.uint32),
    )
    np.testing.assert_array_equal(
        host["candidate_out"].view(np.uint32),
        host["control_out"].view(np.uint32),
    )
    np.testing.assert_array_equal(host["candidate_bf16"], host["control_bf16"])

    norm = bf16_to_float32(norm_bits)
    expected_pair = np.concatenate(
        (norm @ alpha_weight.T, norm @ beta_weight.T),
        axis=-1,
    )
    actual_pair = np.concatenate(
        (
            bf16_to_float32(host["candidate_alpha"]),
            bf16_to_float32(host["candidate_beta"]),
        ),
        axis=-1,
    )
    pair_kl, pair_top1 = _softmax_kl_top1(expected_pair, actual_pair)
    assert pair_kl <= 0.05
    assert pair_top1 >= 0.90

    expected_gdn = _cpu_gdn(
        conv_out,
        gate_bits,
        host["control_alpha"],
        host["control_beta"],
        dt_bias,
        a_log,
        norm_weight,
        base_state,
    )
    gdn_kl, gdn_top1 = _softmax_kl_top1(expected_gdn, host["candidate_out"])
    assert gdn_kl <= 0.05
    assert gdn_top1 >= 0.90
