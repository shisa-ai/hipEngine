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
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve, unregister
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q5_K_T16,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime import gguf_linear as gguf_linear_module
from hipengine.runtime import qwen35_gguf_runner as runner_module

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


def _composite_resolver():
    resolver = getattr(
        gguf_linear_module,
        "resolve_gguf_linear_pair_gdn_snapshot",
        None,
    )
    assert callable(resolver), "dependent alpha/beta-to-GDN resolver is not implemented"
    return resolver


def _fake_weight(
    ptr: int,
    *,
    backend: str = "hip_gfx1100",
    layout: str = LAYOUT_DENSE_F32,
    quant_key: str = "f32",
):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=int(ptr))),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=int(ptr) + 1)),
    }
    return SimpleNamespace(
        backend=backend,
        spec=SimpleNamespace(layout=layout, quant_key=quant_key),
        allocations=allocations,
        allocation=lambda name="raw": allocations[name],
    )


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


def test_dependent_pair_gdn_resolver_derives_both_quant_axes_and_fails_closed() -> None:
    resolver = _composite_resolver()
    backend = "test_dense_f32_pair_gdn"
    key = KernelKey(backend, _COMPOSITE_LAYER, _COMPOSITE_QUANT, _COMPOSITE_VARIANT)

    def owner(*_args, **_kwargs):
        return None

    register(key, owner, replace=True)
    alpha = _fake_weight(20, backend=backend)
    beta = _fake_weight(30, backend=backend)
    ssm_out = _fake_weight(
        40,
        backend=backend,
        layout=LAYOUT_GGUF_Q5_K_T16,
        quant_key="gguf_q5_k_t16_v1",
    )
    try:
        assert resolver(
            alpha,
            beta,
            ssm_out,
            rows=4,
            in_features=5120,
            out_features=48,
            ssm_in_features=6144,
            ssm_out_features=5120,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            backend=backend,
        ) is owner
        assert resolver(
            _fake_weight(
                20,
                backend=backend,
                layout=LAYOUT_GGUF_Q5_K_T16,
                quant_key="gguf_q5_k_t16_v1",
            ),
            beta,
            ssm_out,
            rows=4,
            in_features=5120,
            out_features=48,
            ssm_in_features=6144,
            ssm_out_features=5120,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            backend=backend,
        ) is None
        assert resolver(
            alpha,
            beta,
            ssm_out,
            rows=5,
            in_features=5120,
            out_features=48,
            ssm_in_features=6144,
            ssm_out_features=5120,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            backend=backend,
        ) is None
        unregister(key)
        assert resolver(
            alpha,
            beta,
            ssm_out,
            rows=4,
            in_features=5120,
            out_features=48,
            ssm_in_features=6144,
            ssm_out_features=5120,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            backend=backend,
        ) is None
    finally:
        if is_registered(key):
            unregister(key)


def _runner_fixture():
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    layer_weights = {
        "attn_norm": _fake_weight(0x1000),
        "attn_qkv": _fake_weight(0x1100),
        "attn_gate": _fake_weight(0x1200),
        "ssm_alpha": _fake_weight(0x1300),
        "ssm_beta": _fake_weight(0x1400),
        "ssm_conv1d": _fake_weight(0x1500),
        "ssm_out": _fake_weight(
            0x1600,
            layout=LAYOUT_GGUF_Q5_K_T16,
            quant_key="gguf_q5_k_t16_v1",
        ),
        "ssm_dt_bias": _fake_weight(0x1700),
        "ssm_a": _fake_weight(0x1800),
        "ssm_norm": _fake_weight(0x1900),
    }
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


def test_dependent_pair_gdn_method_launches_snapshot_conv_then_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, layer, scratch, decode_scratch, linear_state_rows, initial_state_snapshot = _runner_fixture()
    method = getattr(
        runner,
        "_try_run_linear_attention_alpha_beta_gdn_rows_exact",
        None,
    )
    assert callable(method), "dependent runner method is not implemented"
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def owner(*args, **kwargs):
        calls.append(("owner", args, kwargs))

    plan = SimpleNamespace(
        available=True,
        conv=lambda *_args, **_kwargs: pytest.fail("unexpected ordinary Conv"),
        gdn=lambda *_args, **_kwargs: pytest.fail("unexpected ordinary GDN"),
        conv_snapshot=lambda *args, **kwargs: calls.append(("conv", args, kwargs)),
        gdn_snapshot=lambda *_args, **_kwargs: pytest.fail("unexpected snapshot GDN"),
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_gguf_linear_pair_gdn_snapshot",
        lambda *_args, **_kwargs: owner,
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_gguf_linear_attention_chain_journal_plan",
        lambda _backend: plan,
    )
    runtime = SimpleNamespace(memcpy_async=lambda *_args: pytest.fail("unexpected commit copy"))

    assert method(
        layer,
        scratch,
        decode_scratch.layer_conv_states[0],
        decode_scratch.layer_recurrent_states[0],
        rows=4,
        linear_state_rows=linear_state_rows,
        initial_state_snapshot=initial_state_snapshot,
        ssm_out_weight=layer.weight("ssm_out"),
        commit_final_linear_state=False,
        stream=23,
        runtime=runtime,
    )
    assert [name for name, _args, _kwargs in calls] == ["conv", "owner"]
    conv_args, conv_kwargs = calls[0][1:]
    assert conv_args[:4] == (
        scratch.linear_qkv.ptr,
        decode_scratch.layer_conv_states[0].ptr,
        linear_state_rows[0].ptr,
        initial_state_snapshot[0].ptr,
    )
    assert conv_kwargs == {"stream": 23, "runtime": runtime}
    owner_args, owner_kwargs = calls[1][1:]
    assert owner_args[:15] == (
        scratch.norm.ptr,
        layer.weight("ssm_alpha").allocation().tensor.ptr,
        layer.weight("ssm_beta").allocation().tensor.ptr,
        scratch.linear_alpha.ptr,
        scratch.linear_beta.ptr,
        scratch.conv_out.ptr,
        scratch.linear_z.ptr,
        layer.weight("ssm_dt_bias").allocation().tensor.ptr,
        layer.weight("ssm_a").allocation().tensor.ptr,
        layer.weight("ssm_norm").allocation().tensor.ptr,
        decode_scratch.layer_recurrent_states[0].ptr,
        linear_state_rows[1].ptr,
        initial_state_snapshot[1].ptr,
        scratch.recurrent_out.ptr,
        scratch.recurrent_bf16.ptr,
    )
    assert owner_args[15:] == (1.0e-6, 4, 5120, 16, 48, 128, 128)
    assert owner_kwargs == {"stream": 23, "runtime": runtime}

    calls.clear()
    monkeypatch.setattr(
        runner_module,
        "resolve_gguf_linear_pair_gdn_snapshot",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    assert not method(
        layer,
        scratch,
        decode_scratch.layer_conv_states[0],
        decode_scratch.layer_recurrent_states[0],
        rows=4,
        linear_state_rows=linear_state_rows,
        initial_state_snapshot=initial_state_snapshot,
        ssm_out_weight=layer.weight("ssm_out"),
        commit_final_linear_state=False,
        stream=23,
        runtime=runtime,
    )
    assert calls == []


@pytest.mark.parametrize("dependent_result", (True, False))
def test_staged_runner_owns_dependent_once_or_complete_existing_fallback(
    monkeypatch: pytest.MonkeyPatch,
    dependent_result: bool,
) -> None:
    runner, layer, scratch, decode_scratch, linear_state_rows, initial_state_snapshot = _runner_fixture()
    dependent_calls: list[dict[str, object]] = []
    alpha_beta_calls: list[tuple[object, ...]] = []
    journal_calls: list[dict[str, object]] = []
    linear_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def dependent(*_args, **kwargs):
        dependent_calls.append(kwargs)
        return dependent_result

    def alpha_beta(*args, **kwargs):
        alpha_beta_calls.append((*args, kwargs))
        return "fallback"

    def journal(*_args, **kwargs):
        journal_calls.append(kwargs)
        return True

    runner._try_run_linear_attention_alpha_beta_gdn_rows_exact = dependent
    runner._run_linear_attention_alpha_beta_rows = alpha_beta
    runner._try_run_linear_attention_chain_journal_rows_exact = journal
    monkeypatch.setattr(runner_module, "launch_gguf_linear_pair", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear",
        lambda *args, **kwargs: linear_calls.append((args, kwargs)),
    )

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

    assert len(dependent_calls) == 1
    assert len(alpha_beta_calls) == (0 if dependent_result else 1)
    assert len(journal_calls) == (0 if dependent_result else 1)
    assert len(linear_calls) == 1
    assert linear_calls[0][0][0] == layer.weight("ssm_out")
    assert linear_calls[0][1]["activation_dtype"] == "bf16"


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
