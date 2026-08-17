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
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100.linear import dense_gemv as dense_gemv_module
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_gemv_bf16_f32w_bf16_out,
    dense_pair_gemv_bf16_f32w_bf16_out,
)
from hipengine.kernels.hip_gfx1100.linear_attn import conv as conv_module
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_F32
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime import gguf_linear as gguf_linear_module
from hipengine.runtime import qwen35_gguf_runner as qgr

_COMPOSITE_LAYER = "linear_attn_alpha_beta+chain_conv+snapshot"
_COMPOSITE_VARIANT = "bf16_k5120_n48_c10240_k4_exact_state_rows_tloop"
_COMPOSITE_KEY = KernelKey(
    "hip_gfx1100",
    _COMPOSITE_LAYER,
    "f32",
    _COMPOSITE_VARIANT,
)
_SERIAL_LAYER = "linear_attn_alpha_beta+conv_decode"
_SERIAL_VARIANT = "bf16_k5120_n48_c10240_k4_c1"
_SERIAL_KEY = KernelKey("hip_gfx1100", _SERIAL_LAYER, "f32", _SERIAL_VARIANT)
_GFX1151_SERIAL_KEY = KernelKey(
    "hip_gfx1151",
    _SERIAL_KEY.layer,
    _SERIAL_KEY.quant,
    _SERIAL_KEY.variant,
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


def _serial_wrapper():
    wrapper = getattr(
        conv_module,
        "qwen35_linear_attn_alpha_beta_conv_decode_bf16_f32w",
        None,
    )
    assert callable(wrapper), "serial alpha/beta plus Conv wrapper is not implemented"
    return wrapper


def _fake_weight(ptr: int, *, backend: str = "hip_gfx1151", quant_key: str = "f32"):
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=int(ptr)))
    return SimpleNamespace(
        backend=backend,
        spec=SimpleNamespace(layout=LAYOUT_DENSE_F32, quant_key=quant_key),
        allocation=lambda _name="raw": allocation,
    )


def test_composite_and_serial_registration_are_independently_screened() -> None:
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
    for key in (_SERIAL_KEY, _GFX1151_SERIAL_KEY):
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is _serial_wrapper()
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_F32_ALPHA_BETA_CONV_DECODE_SHAPES",
        (),
    ) == frozenset({(1, 5_120, 48, 10_240, 4)})
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_DENSE_F32_ALPHA_BETA_CONV_DECODE_SHAPES",
        (),
    ) == ()


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


def test_serial_wrapper_uses_one_symbol_and_validates_exact_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def signed(_library, symbol, _argtypes, _restype):
        def launch(*args):
            calls.append((symbol, args))
            return 0

        return launch

    monkeypatch.setattr(conv_module, "signed_kernel_fn", signed)
    wrapper = _serial_wrapper()
    runtime = SimpleNamespace(check=lambda err: pytest.fail(f"unexpected HIP error {err}"))
    wrapper(
        *range(10, 19),
        5_120,
        48,
        10_240,
        4,
        stream=17,
        library=object(),
        runtime=runtime,
    )

    assert [symbol for symbol, _args in calls] == [
        "hipengine_qwen35_linear_attn_alpha_beta_conv_decode_bf16_f32w"
    ]
    assert tuple(int(value) for value in calls[0][1][9:13]) == (5_120, 48, 10_240, 4)
    assert int(calls[0][1][13]) == 17

    invalid = (
        ((4_096, 48, 10_240, 4), "in_features must equal 5120"),
        ((5_120, 32, 10_240, 4), "out_features must equal 48"),
        ((5_120, 48, 8_192, 4), "channels must equal 10240"),
        ((5_120, 48, 10_240, 3), "kernel_size must equal 4"),
    )
    for dimensions, message in invalid:
        with pytest.raises(ValueError, match=message):
            wrapper(*range(10, 19), *dimensions)
    with pytest.raises(ValueError, match="threads must equal 256"):
        wrapper(*range(10, 19), 5_120, 48, 10_240, 4, threads=128)


def test_serial_route_is_capability_shape_and_registry_qualified() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    original = resolve(
        backend=_GFX1151_SERIAL_KEY.backend,
        layer=_GFX1151_SERIAL_KEY.layer,
        quant=_GFX1151_SERIAL_KEY.quant,
        variant=_GFX1151_SERIAL_KEY.variant,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def serial(*args, **kwargs):
        calls.append((args, kwargs))

    route = getattr(qgr, "_try_launch_dense_f32_alpha_beta_conv_decode", None)
    assert callable(route), "serial alpha/beta plus Conv runtime route is not implemented"
    register(_GFX1151_SERIAL_KEY, serial, replace=True)
    try:
        common = dict(
            weight_a=_fake_weight(20),
            weight_b=_fake_weight(30),
            norm_ptr=10,
            out_a_ptr=40,
            out_b_ptr=50,
            hidden_states_ptr=60,
            conv_state_ptr=70,
            conv_weight_ptr=80,
            conv_out_ptr=90,
            in_features=5_120,
            out_features=48,
            channels=10_240,
            kernel_size=4,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
        )
        assert route(rows=1, **common)
        assert not route(rows=2, **common)
        mismatched = {**common, "out_features": 47}
        assert not route(rows=1, **mismatched)
        assert not route(
            rows=1,
            weight_a=_fake_weight(20, quant_key="gguf_q4_k"),
            **{key: value for key, value in common.items() if key != "weight_a"},
        )
    finally:
        register(_GFX1151_SERIAL_KEY, original, replace=True)

    assert calls == [
        (
            (10, 20, 30, 40, 50, 60, 70, 80, 90, 5_120, 48, 10_240, 4),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_serial_composite_is_pair_plus_inplace_conv_bit_exact_and_passes_cpu_gate() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    dense_library = dense_gemv_module.build_dense_gemv(load=True)
    conv_library = conv_module.build_qwen35_linear_attn_conv(load=True)
    serial = _serial_wrapper()
    in_features = 5_120
    out_features = 48
    channels = 10_240
    kernel_size = 4
    rng = np.random.default_rng(0xC138)
    norm_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.25, size=(1, in_features)).astype(np.float32)
    )
    weight_a = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
    weight_b = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
    hidden_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.08, size=(1, channels)).astype(np.float32)
    )
    base_state = rng.normal(0.0, 0.04, size=(channels, kernel_size)).astype(np.float32)
    conv_weight = rng.normal(0.0, 0.06, size=(channels, kernel_size)).astype(np.float32)

    arrays = {
        "control_a": np.empty((1, out_features), dtype=np.uint16),
        "control_b": np.empty((1, out_features), dtype=np.uint16),
        "candidate_a": np.empty((1, out_features), dtype=np.uint16),
        "candidate_b": np.empty((1, out_features), dtype=np.uint16),
        "control_state": np.array(base_state, copy=True),
        "candidate_state": np.array(base_state, copy=True),
        "control_conv": np.empty((1, channels), dtype=np.float32),
        "candidate_conv": np.empty((1, channels), dtype=np.float32),
    }
    buffers = []

    def device(array: np.ndarray):
        allocation = malloc(array.nbytes, runtime=runtime)
        buffers.append(allocation)
        copy_host_to_device(
            allocation,
            host_array_ptr(np.ascontiguousarray(array)),
            runtime=runtime,
        )
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
        dconv_weight = device(conv_weight)
        dcontrol_state = device(arrays["control_state"])
        dcandidate_state = device(arrays["candidate_state"])
        dcontrol_a = empty(arrays["control_a"])
        dcontrol_b = empty(arrays["control_b"])
        dcandidate_a = empty(arrays["candidate_a"])
        dcandidate_b = empty(arrays["candidate_b"])
        dcontrol_conv = empty(arrays["control_conv"])
        dcandidate_conv = empty(arrays["candidate_conv"])

        dense_pair_gemv_bf16_f32w_bf16_out(
            dnorm.ptr,
            dwa.ptr,
            dwb.ptr,
            dcontrol_a.ptr,
            dcontrol_b.ptr,
            1,
            in_features,
            out_features,
            library=dense_library,
            runtime=runtime,
        )
        conv_module.qwen35_linear_attn_conv_decode_bf16(
            dhidden.ptr,
            dcontrol_state.ptr,
            dconv_weight.ptr,
            dcontrol_conv.ptr,
            channels,
            kernel_size,
            library=conv_library,
            runtime=runtime,
        )
        serial(
            dnorm.ptr,
            dwa.ptr,
            dwb.ptr,
            dcandidate_a.ptr,
            dcandidate_b.ptr,
            dhidden.ptr,
            dcandidate_state.ptr,
            dconv_weight.ptr,
            dcandidate_conv.ptr,
            in_features,
            out_features,
            channels,
            kernel_size,
            library=conv_library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        for name, allocation in (
            ("control_a", dcontrol_a),
            ("control_b", dcontrol_b),
            ("candidate_a", dcandidate_a),
            ("candidate_b", dcandidate_b),
            ("control_state", dcontrol_state),
            ("candidate_state", dcandidate_state),
            ("control_conv", dcontrol_conv),
            ("candidate_conv", dcandidate_conv),
        ):
            copy_device_to_host(host_array_ptr(arrays[name]), allocation, runtime=runtime)
    finally:
        for allocation in reversed(buffers):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(arrays["candidate_a"], arrays["control_a"])
    np.testing.assert_array_equal(arrays["candidate_b"], arrays["control_b"])
    np.testing.assert_array_equal(
        arrays["candidate_state"].view(np.uint32),
        arrays["control_state"].view(np.uint32),
    )
    np.testing.assert_array_equal(
        arrays["candidate_conv"].view(np.uint32),
        arrays["control_conv"].view(np.uint32),
    )

    norm = bf16_to_float32(norm_bits)
    expected_pair = np.concatenate((norm @ weight_a.T, norm @ weight_b.T), axis=-1)
    actual_pair = np.concatenate(
        (bf16_to_float32(arrays["candidate_a"]), bf16_to_float32(arrays["candidate_b"])),
        axis=-1,
    )
    pair_kl, pair_top1 = _softmax_kl_top1(expected_pair, actual_pair)
    assert pair_kl <= 0.05
    assert pair_top1 >= 0.90

    expected_state, expected_conv = _cpu_chain_conv(hidden_bits, base_state, conv_weight)
    np.testing.assert_array_equal(
        arrays["candidate_state"].view(np.uint32),
        expected_state[-1].view(np.uint32),
    )
    np.testing.assert_allclose(arrays["candidate_conv"], expected_conv, rtol=2.0e-6, atol=2.0e-7)
    conv_kl, conv_top1 = _softmax_kl_top1(expected_conv, arrays["candidate_conv"])
    assert conv_kl <= 0.05
    assert conv_top1 >= 0.90
