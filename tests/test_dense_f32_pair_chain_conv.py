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
from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    register,
    resolve,
    unregister,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime import gguf_linear as gguf_linear_module
from hipengine.runtime import qwen35_gguf_runner as runner_module

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


def _composite_dispatch():
    dispatch = getattr(
        gguf_linear_module,
        "launch_gguf_linear_pair_chain_conv_snapshot",
        None,
    )
    assert callable(dispatch), "GGUF alpha/beta plus snapshot chain-Conv dispatch is not implemented"
    return dispatch


def _fake_weight(
    ptr: int,
    *,
    backend: str = "hip_gfx1100",
    layout: str = LAYOUT_DENSE_F32,
    quant_key: str = "f32",
):
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=int(ptr)))
    return SimpleNamespace(
        backend=backend,
        spec=SimpleNamespace(layout=layout, quant_key=quant_key),
        allocation=lambda _name="raw": allocation,
    )


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


def test_composite_dispatch_resolves_dense_f32_pair_and_falls_back_on_every_miss() -> None:
    dispatch = _composite_dispatch()
    backend = "test_dense_f32_pair_chain_conv"
    key = KernelKey(backend, _COMPOSITE_LAYER, "f32", _COMPOSITE_VARIANT)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def owner(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, owner, replace=True)
    try:
        assert dispatch(
            _fake_weight(20, backend=backend),
            _fake_weight(30, backend=backend),
            norm_ptr=10,
            out_a_ptr=40,
            out_b_ptr=50,
            hidden_states_ptr=60,
            base_conv_state_ptr=70,
            chain_conv_state_ptr=80,
            initial_conv_state_snapshot_ptr=90,
            conv_weight_ptr=100,
            conv_out_ptr=110,
            rows=4,
            in_features=5120,
            out_features=48,
            channels=10240,
            kernel_size=4,
            backend=backend,
            stream=7,
            runtime="runtime-sentinel",
        )
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[:11] == (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110)
        assert args[11:] == (4, 5120, 48, 10240, 4)
        assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}

        assert not dispatch(
            _fake_weight(20, backend=backend, layout=LAYOUT_DENSE_BF16, quant_key="bf16"),
            _fake_weight(30, backend=backend),
            norm_ptr=10,
            out_a_ptr=40,
            out_b_ptr=50,
            hidden_states_ptr=60,
            base_conv_state_ptr=70,
            chain_conv_state_ptr=80,
            initial_conv_state_snapshot_ptr=90,
            conv_weight_ptr=100,
            conv_out_ptr=110,
            rows=4,
            in_features=5120,
            out_features=48,
            channels=10240,
            kernel_size=4,
            backend=backend,
        )
        assert not dispatch(
            _fake_weight(20, backend=backend),
            _fake_weight(30, backend=backend),
            norm_ptr=10,
            out_a_ptr=40,
            out_b_ptr=50,
            hidden_states_ptr=60,
            base_conv_state_ptr=70,
            chain_conv_state_ptr=80,
            initial_conv_state_snapshot_ptr=90,
            conv_weight_ptr=100,
            conv_out_ptr=110,
            rows=5,
            in_features=5120,
            out_features=48,
            channels=10240,
            kernel_size=4,
            backend=backend,
        )
        assert len(calls) == 1

        unregister(key)
        assert not dispatch(
            _fake_weight(20, backend=backend),
            _fake_weight(30, backend=backend),
            norm_ptr=10,
            out_a_ptr=40,
            out_b_ptr=50,
            hidden_states_ptr=60,
            base_conv_state_ptr=70,
            chain_conv_state_ptr=80,
            initial_conv_state_snapshot_ptr=90,
            conv_weight_ptr=100,
            conv_out_ptr=110,
            rows=4,
            in_features=5120,
            out_features=48,
            channels=10240,
            kernel_size=4,
            backend=backend,
        )
    finally:
        if is_registered(key):
            unregister(key)


def _runner_fixture():
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    pointers = {
        "attn_norm": 0x1000,
        "attn_qkv": 0x1100,
        "attn_gate": 0x1200,
        "ssm_alpha": 0x1300,
        "ssm_beta": 0x1400,
        "ssm_conv1d": 0x1500,
        "ssm_out": 0x1600,
        "ssm_dt_bias": 0x1700,
        "ssm_a": 0x1800,
        "ssm_norm": 0x1900,
    }
    layer_weights = {name: _fake_weight(ptr) for name, ptr in pointers.items()}
    layer = SimpleNamespace(weight=layer_weights.__getitem__)
    config = SimpleNamespace(
        hidden_size=5120,
        ssm_group_count=16,
        ssm_state_size=128,
        ssm_inner_size=6144,
        ssm_time_step_rank=48,
        ssm_conv_kernel=4,
        rms_norm_eps=1.0e-6,
    )
    runner = object.__new__(Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1100"
    runner.runtime = SimpleNamespace()
    runner.weights = SimpleNamespace(config=config, layer=lambda _layer_id: layer)
    runner._gdn_chain_output_fusion_for_weight = lambda _weight: object()
    runner._gdn_chain_snapshot_output_fusion_for_weight = lambda _weight: object()
    runner._gdn_decode_output_cast_for_weight = lambda _weight, rows: None

    def buf(ptr: int, nbytes: int = 4096):
        return SimpleNamespace(ptr=ptr, nbytes=nbytes)

    scratch = SimpleNamespace(
        norm=buf(0x2000),
        linear_qkv=buf(0x2100),
        linear_z=buf(0x2200),
        linear_alpha=buf(0x2300),
        linear_beta=buf(0x2400),
        conv_out=buf(0x2500),
        recurrent_out=buf(0x2600),
        recurrent_bf16=buf(0x2700),
    )
    decode_scratch = SimpleNamespace(
        layer_conv_states=(buf(0x3000, 163840),),
        layer_recurrent_states=(buf(0x3100, 3145728),),
    )
    linear_state_rows = (buf(0x4000), buf(0x4100))
    initial_state_snapshot = (buf(0x5000), buf(0x5100))
    return runner, layer, scratch, decode_scratch, linear_state_rows, initial_state_snapshot


@pytest.mark.parametrize("composite_result", (True, False))
def test_chain_runner_owns_composite_once_or_complete_three_primitive_fallback(
    monkeypatch: pytest.MonkeyPatch,
    composite_result: bool,
) -> None:
    runner, layer, scratch, decode_scratch, linear_state_rows, initial_state_snapshot = _runner_fixture()
    composite_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    alpha_beta_calls: list[tuple[object, ...]] = []
    journal_calls: list[dict[str, object]] = []
    linear_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def composite(*args, **kwargs):
        composite_calls.append((args, kwargs))
        return composite_result

    def alpha_beta(*args, **kwargs):
        alpha_beta_calls.append((*args, kwargs))
        return "fallback"

    def journal(*_args, **kwargs):
        journal_calls.append(kwargs)
        return True

    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear_pair_chain_conv_snapshot",
        composite,
        raising=False,
    )
    monkeypatch.setattr(runner_module, "launch_gguf_linear_pair", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear",
        lambda *args, **kwargs: linear_calls.append((args, kwargs)),
    )
    runner._run_linear_attention_alpha_beta_rows = alpha_beta
    runner._try_run_linear_attention_chain_journal_rows_exact = journal

    runner._run_linear_attention_attn_chain_rows_exact(
        0,
        hidden_ptr=0x6000,
        attn_out_ptr=0x6100,
        scratch=scratch,
        rows=4,
        decode_scratch=decode_scratch,
        linear_state_rows=linear_state_rows,
        initial_state_snapshot=initial_state_snapshot,
        input_norm_ptr=scratch.norm.ptr,
        commit_final_linear_state=False,
        stream=19,
    )

    assert len(composite_calls) == 1
    args, kwargs = composite_calls[0]
    assert args[:2] == (layer.weight("ssm_alpha"), layer.weight("ssm_beta"))
    assert kwargs == {
        "norm_ptr": scratch.norm.ptr,
        "out_a_ptr": scratch.linear_alpha.ptr,
        "out_b_ptr": scratch.linear_beta.ptr,
        "hidden_states_ptr": scratch.linear_qkv.ptr,
        "base_conv_state_ptr": decode_scratch.layer_conv_states[0].ptr,
        "chain_conv_state_ptr": linear_state_rows[0].ptr,
        "initial_conv_state_snapshot_ptr": initial_state_snapshot[0].ptr,
        "conv_weight_ptr": layer.weight("ssm_conv1d").allocation().tensor.ptr,
        "conv_out_ptr": scratch.conv_out.ptr,
        "rows": 4,
        "in_features": 5120,
        "out_features": 48,
        "channels": 10240,
        "kernel_size": 4,
        "backend": "hip_gfx1100",
        "stream": 19,
        "runtime": runner.runtime,
    }
    assert len(alpha_beta_calls) == (0 if composite_result else 1)
    assert len(journal_calls) == 1
    assert journal_calls[0]["conv_ready"] is composite_result
    assert len(linear_calls) == 1
    assert linear_calls[0][0][0] == layer.weight("ssm_out")

    # Without producer-folded snapshot storage, do not launch the composite at
    # all: alpha/beta and ordinary chain Conv keep complete ownership.
    composite_calls.clear()
    alpha_beta_calls.clear()
    journal_calls.clear()
    linear_calls.clear()
    runner._run_linear_attention_attn_chain_rows_exact(
        0,
        hidden_ptr=0x6000,
        attn_out_ptr=0x6100,
        scratch=scratch,
        rows=4,
        decode_scratch=decode_scratch,
        linear_state_rows=linear_state_rows,
        initial_state_snapshot=None,
        input_norm_ptr=scratch.norm.ptr,
        commit_final_linear_state=False,
        stream=19,
    )
    assert composite_calls == []
    assert len(alpha_beta_calls) == 1
    assert len(journal_calls) == 1
    assert journal_calls[0]["conv_ready"] is False
    assert len(linear_calls) == 1


def test_chain_journal_skips_only_the_snapshot_conv_when_composite_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, layer, scratch, decode_scratch, linear_state_rows, initial_state_snapshot = _runner_fixture()
    calls: list[str] = []
    plan = SimpleNamespace(
        available=True,
        conv=lambda *_args, **_kwargs: calls.append("conv"),
        gdn=lambda *_args, **_kwargs: calls.append("gdn"),
        conv_snapshot=lambda *_args, **_kwargs: calls.append("conv_snapshot"),
        gdn_snapshot=lambda *_args, **_kwargs: calls.append("gdn_snapshot"),
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_gguf_linear_attention_chain_journal_plan",
        lambda _backend: plan,
    )
    runtime = SimpleNamespace(memcpy_async=lambda *_args: pytest.fail("unexpected commit copy"))

    assert runner._try_run_linear_attention_chain_journal_rows_exact(
        layer,
        scratch,
        decode_scratch.layer_conv_states[0],
        decode_scratch.layer_recurrent_states[0],
        rows=4,
        linear_state_rows=linear_state_rows,
        initial_state_snapshot=initial_state_snapshot,
        commit_final_linear_state=False,
        stream=23,
        runtime=runtime,
        conv_ready=True,
    )
    assert calls == ["gdn_snapshot"]


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
