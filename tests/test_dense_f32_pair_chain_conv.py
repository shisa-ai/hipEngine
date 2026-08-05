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
from hipengine.kernels.hip_gfx1100.linear_attn import conv as conv_module
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime import gguf_linear as gguf_linear_module

_COMPOSITE_LAYER = "linear_attn_alpha_beta+chain_conv+snapshot"
_COMPOSITE_VARIANT = "bf16_k5120_n48_c10240_k4_exact_state_rows_tloop"
_COMPOSITE_KEY = KernelKey(
    "hip_gfx1100",
    _COMPOSITE_LAYER,
    "f32",
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
        conv_module,
        "qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w_tloop",
        None,
    )
    assert callable(wrapper), "alpha/beta plus snapshot chain-Conv wrapper is not implemented"
    return wrapper


def test_composite_registers_only_on_screened_gfx1100_backend() -> None:
    conv_module.register_qwen35_linear_attn_conv_kernels(replace=True)
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


def test_composite_wrapper_uses_one_symbol_and_validates_screened_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def signed(_library, symbol, _argtypes, _restype):
        def launch(*args):
            calls.append((symbol, args))
            return 0

        return launch

    monkeypatch.setattr(conv_module, "signed_kernel_fn", signed)
    wrapper = _composite_wrapper()
    runtime = SimpleNamespace(check=lambda err: pytest.fail(f"unexpected HIP error {err}"))
    pointers = tuple(range(10, 21))
    for rows in (1, 2, 3, 4):
        wrapper(
            *pointers,
            rows,
            5120,
            48,
            10240,
            4,
            stream=17,
            library=object(),
            runtime=runtime,
        )

    assert [symbol for symbol, _args in calls] == [
        "hipengine_qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w"
    ] * 4
    assert [int(args[11]) for _symbol, args in calls] == [1, 2, 3, 4]
    assert all(tuple(int(value) for value in args[12:16]) == (5120, 48, 10240, 4) for _, args in calls)
    assert all(int(args[16]) == 17 for _, args in calls)

    invalid = (
        ((0, 5120, 48, 10240, 4), "rows must be between 1 and 4"),
        ((5, 5120, 48, 10240, 4), "rows must be between 1 and 4"),
        ((2, 4096, 48, 10240, 4), "in_features must equal 5120"),
        ((2, 5120, 32, 10240, 4), "out_features must equal 48"),
        ((2, 5120, 48, 8192, 4), "channels must equal 10240"),
        ((2, 5120, 48, 10240, 3), "kernel_size must equal 4"),
    )
    for dimensions, message in invalid:
        with pytest.raises(ValueError, match=message):
            wrapper(*pointers, *dimensions)
    with pytest.raises(ValueError, match="threads must equal 256"):
        wrapper(*pointers, 2, 5120, 48, 10240, 4, threads=128)


def test_composite_stays_primitive_only_after_runtime_wall_rejection() -> None:
    assert not hasattr(
        gguf_linear_module,
        "launch_gguf_linear_pair_chain_conv_snapshot",
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


def _cpu_chain_conv(
    hidden_bits: np.ndarray,
    base_state: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = bf16_to_float32(hidden_bits)
    state = np.array(base_state, dtype=np.float32, copy=True)
    rows = hidden.shape[0]
    state_rows = np.empty((rows, *state.shape), dtype=np.float32)
    out = np.empty(hidden.shape, dtype=np.float32)
    for row in range(rows):
        state[:, :-1] = state[:, 1:].copy()
        state[:, -1] = hidden[row]
        state_rows[row] = state
        acc = np.zeros(hidden.shape[1], dtype=np.float32)
        for kernel_index in range(weight.shape[1]):
            product = np.asarray(state[:, kernel_index] * weight[:, kernel_index], dtype=np.float32)
            acc = np.asarray(acc + product, dtype=np.float32)
        out[row] = np.asarray(acc / (np.float32(1.0) + np.exp(-acc)), dtype=np.float32)
    return state_rows, out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (1, 2, 3, 4))
def test_composite_is_three_primitive_bit_exact_and_passes_cpu_gate(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    dense_library = dense_gemv_module.build_dense_gemv(load=True)
    conv_library = conv_module.build_qwen35_linear_attn_conv(load=True)
    composite = _composite_wrapper()
    in_features = 5120
    out_features = 48
    channels = 10240
    kernel_size = 4
    rng = np.random.default_rng(0xC04A + rows)
    norm_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.25, size=(rows, in_features)).astype(np.float32)
    )
    weight_a = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
    weight_b = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
    hidden_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.08, size=(rows, channels)).astype(np.float32)
    )
    base_state = rng.normal(0.0, 0.04, size=(channels, kernel_size)).astype(np.float32)
    conv_weight = rng.normal(0.0, 0.06, size=(channels, kernel_size)).astype(np.float32)

    linear_shape = (rows, out_features)
    state_shape = (rows, channels, kernel_size)
    conv_shape = (rows, channels)
    control_a = np.empty(linear_shape, dtype=np.uint16)
    control_b = np.empty(linear_shape, dtype=np.uint16)
    candidate_a = np.empty(linear_shape, dtype=np.uint16)
    candidate_b = np.empty(linear_shape, dtype=np.uint16)
    control_state = np.empty(state_shape, dtype=np.float32)
    candidate_state = np.empty(state_shape, dtype=np.float32)
    control_snapshot = np.empty(base_state.shape, dtype=np.float32)
    candidate_snapshot = np.empty(base_state.shape, dtype=np.float32)
    control_conv = np.empty(conv_shape, dtype=np.float32)
    candidate_conv = np.empty(conv_shape, dtype=np.float32)
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
        dwa = device(weight_a)
        dwb = device(weight_b)
        dhidden = device(hidden_bits)
        dbase = device(base_state)
        dconv_weight = device(conv_weight)
        dcontrol_a = empty(control_a)
        dcontrol_b = empty(control_b)
        dcandidate_a = empty(candidate_a)
        dcandidate_b = empty(candidate_b)
        dcontrol_state = empty(control_state)
        dcandidate_state = empty(candidate_state)
        dcontrol_snapshot = empty(control_snapshot)
        dcandidate_snapshot = empty(candidate_snapshot)
        dcontrol_conv = empty(control_conv)
        dcandidate_conv = empty(candidate_conv)

        dense_gemv_bf16_f32w_bf16_out(
            dnorm.ptr,
            dwa.ptr,
            dcontrol_a.ptr,
            rows,
            in_features,
            out_features,
            library=dense_library,
            runtime=runtime,
        )
        dense_gemv_bf16_f32w_bf16_out(
            dnorm.ptr,
            dwb.ptr,
            dcontrol_b.ptr,
            rows,
            in_features,
            out_features,
            library=dense_library,
            runtime=runtime,
        )
        conv_module.qwen35_linear_attn_chain_conv_decode_bf16_snapshot_tloop(
            dhidden.ptr,
            dbase.ptr,
            dcontrol_state.ptr,
            dcontrol_snapshot.ptr,
            dconv_weight.ptr,
            dcontrol_conv.ptr,
            rows,
            channels,
            kernel_size,
            library=conv_library,
            runtime=runtime,
        )
        composite(
            dnorm.ptr,
            dwa.ptr,
            dwb.ptr,
            dcandidate_a.ptr,
            dcandidate_b.ptr,
            dhidden.ptr,
            dbase.ptr,
            dcandidate_state.ptr,
            dcandidate_snapshot.ptr,
            dconv_weight.ptr,
            dcandidate_conv.ptr,
            rows,
            in_features,
            out_features,
            channels,
            kernel_size,
            library=conv_library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        for host, allocation in (
            (control_a, dcontrol_a),
            (control_b, dcontrol_b),
            (candidate_a, dcandidate_a),
            (candidate_b, dcandidate_b),
            (control_state, dcontrol_state),
            (candidate_state, dcandidate_state),
            (control_snapshot, dcontrol_snapshot),
            (candidate_snapshot, dcandidate_snapshot),
            (control_conv, dcontrol_conv),
            (candidate_conv, dcandidate_conv),
        ):
            copy_device_to_host(host_array_ptr(host), allocation, runtime=runtime)
    finally:
        for allocation in reversed(buffers):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(candidate_a, control_a)
    np.testing.assert_array_equal(candidate_b, control_b)
    np.testing.assert_array_equal(candidate_state.view(np.uint32), control_state.view(np.uint32))
    np.testing.assert_array_equal(candidate_snapshot.view(np.uint32), control_snapshot.view(np.uint32))
    np.testing.assert_array_equal(candidate_conv.view(np.uint32), control_conv.view(np.uint32))

    norm = bf16_to_float32(norm_bits)
    expected_pair = np.concatenate((norm @ weight_a.T, norm @ weight_b.T), axis=-1)
    actual_pair = np.concatenate(
        (bf16_to_float32(candidate_a), bf16_to_float32(candidate_b)),
        axis=-1,
    )
    pair_kl, pair_top1 = _softmax_kl_top1(expected_pair, actual_pair)
    assert pair_kl <= 0.05
    assert pair_top1 >= 0.90

    expected_state, expected_conv = _cpu_chain_conv(hidden_bits, base_state, conv_weight)
    np.testing.assert_array_equal(candidate_state.view(np.uint32), expected_state.view(np.uint32))
    np.testing.assert_array_equal(candidate_snapshot.view(np.uint32), base_state.view(np.uint32))
    np.testing.assert_allclose(candidate_conv, expected_conv, rtol=2.0e-6, atol=2.0e-7)
    conv_kl, conv_top1 = _softmax_kl_top1(expected_conv, candidate_conv)
    assert conv_kl <= 0.05
    assert conv_top1 >= 0.90
