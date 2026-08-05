"""Exact fused gate/up row reuse plus SiLU for resident Q4_K pack8 weights."""

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
from hipengine.kernels.cpu_reference import gguf_q4_k_pack8_gemv
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_Q4_K_PACK8, LAYOUT_RAW_GGUF
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from hipengine.runtime import qwen35_gguf_runner as qwen35_runner
from hipengine.runtime.gguf_linear import (
    launch_gguf_linear_pair_silu,
    native_batch_decode_session,
    q4k_rowtile_session,
)
from tests.test_gguf_q4_k_gemv import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).copy()
    words += 0x7FFF + ((words >> 16) & 1)
    return (words >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probs = np.exp(shifted, dtype=np.float64)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def test_pack8_dual_rowtile_silu_registry_contract() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out


def test_pack8_dual_rowtile_silu_wrapper_rejects_unsupported_launches() -> None:
    for rows in (1, 5):
        with pytest.raises(ValueError, match="rows must be 2, 3, or 4"):
            gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                rows,
                256,
                16,
            )
    with pytest.raises(ValueError, match="threads must be 0 or 64"):
        gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            4,
            256,
            16,
            threads=32,
        )


def _fake_pack8_weight(
    base_ptr: int,
    *,
    layout: str = LAYOUT_Q4_K_PACK8,
    decode_tiles: bool = False,
):
    allocations = {
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=base_ptr + 1)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=base_ptr + 2)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=base_ptr + 3)),
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=base_ptr + 4)),
    }
    if decode_tiles:
        allocations["decode_tiles"] = SimpleNamespace(
            tensor=SimpleNamespace(ptr=base_ptr + 5)
        )
    return SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=layout, quant_key="gguf_q4_k"),
        allocation=lambda name="raw": allocations[name],
    )


def _launch_runtime_candidate(
    weight_a,
    weight_b,
    *,
    rows: int = 4,
    in_features: int = 5120,
    out_features: int = 17408,
) -> bool:
    return launch_gguf_linear_pair_silu(
        weight_a,
        weight_b,
        x_ptr=100,
        out_ptr=400,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        backend="hip_gfx1100",
        stream=7,
        libraries={
            "gguf_q4_k": "library-sentinel",
            "gguf_q4_k_t16_v1": "t16-library-sentinel",
        },
        runtime="runtime-sentinel",
        use_gemv_decode=True,
    )


def test_pack8_dual_rowtile_silu_runtime_policy_is_native_shape_bounded() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []
    register(key, lambda *args, **kwargs: calls.append((args, kwargs)), replace=True)
    weight_a = _fake_pack8_weight(10)
    weight_b = _fake_pack8_weight(20)
    try:
        assert not _launch_runtime_candidate(weight_a, weight_b)
        with native_batch_decode_session(True):
            for rows in (2, 3, 4):
                assert _launch_runtime_candidate(weight_a, weight_b, rows=rows)
            for rows in (1, 5):
                assert not _launch_runtime_candidate(weight_a, weight_b, rows=rows)
            assert not _launch_runtime_candidate(weight_a, weight_b, in_features=5376)
            assert not _launch_runtime_candidate(weight_a, weight_b, out_features=17152)
            assert not _launch_runtime_candidate(
                _fake_pack8_weight(30, layout=LAYOUT_RAW_GGUF),
                weight_b,
            )
            with q4k_rowtile_session(False):
                assert not _launch_runtime_candidate(weight_a, weight_b)
    finally:
        register(key, original, replace=True)

    assert [args[8] for args, _kwargs in calls] == [2, 3, 4]
    for args, kwargs in calls:
        assert args[:8] == (100, 11, 12, 13, 21, 22, 23, 400)
        assert args[9:] == (5120, 17408)
        assert kwargs == {
            "stream": 7,
            "runtime": "runtime-sentinel",
            "library": "library-sentinel",
        }


def test_q4_t16_dual_rowtile_silu_sidecar_precedes_pack8() -> None:
    t16_key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_rowtile_bf16_bf16_out",
    )
    pack8_key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    original_t16 = resolve(
        backend=t16_key.backend,
        layer=t16_key.layer,
        quant=t16_key.quant,
        variant=t16_key.variant,
    )
    original_pack8 = resolve(
        backend=pack8_key.backend,
        layer=pack8_key.layer,
        quant=pack8_key.quant,
        variant=pack8_key.variant,
    )
    calls: list[tuple[str, tuple, dict]] = []
    register(
        t16_key,
        lambda *args, **kwargs: calls.append(("t16", args, kwargs)),
        replace=True,
    )
    register(
        pack8_key,
        lambda *args, **kwargs: calls.append(("pack8", args, kwargs)),
        replace=True,
    )
    try:
        with native_batch_decode_session(True):
            for rows in (2, 3, 4):
                assert _launch_runtime_candidate(
                    _fake_pack8_weight(10, decode_tiles=True),
                    _fake_pack8_weight(20, decode_tiles=True),
                    rows=rows,
                )
            assert _launch_runtime_candidate(
                _fake_pack8_weight(10, decode_tiles=True),
                _fake_pack8_weight(20),
            )
    finally:
        register(t16_key, original_t16, replace=True)
        register(pack8_key, original_pack8, replace=True)

    assert [name for name, _args, _kwargs in calls] == [
        "t16",
        "t16",
        "t16",
        "pack8",
    ]
    for _name, args, kwargs in calls[:3]:
        assert args[:4] == (100, 15, 25, 400)
        assert args[5:] == (5120, 17408)
        assert kwargs == {
            "stream": 7,
            "runtime": "runtime-sentinel",
            "library": "t16-library-sentinel",
        }


def test_pack8_dual_rowtile_silu_runtime_policy_fails_closed_on_missing_key() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    register(key, None, replace=True)
    try:
        with native_batch_decode_session(True):
            assert not _launch_runtime_candidate(
                _fake_pack8_weight(10),
                _fake_pack8_weight(20),
            )
    finally:
        register(key, original, replace=True)


def _fake_dense_runner():
    slots = {
        name: SimpleNamespace(name=name)
        for name in ("post_attention_norm", "ffn_gate", "ffn_up", "ffn_down")
    }
    slots["post_attention_norm"].allocation = lambda: SimpleNamespace(
        tensor=SimpleNamespace(ptr=300)
    )
    layer = SimpleNamespace(weight=lambda name: slots[name])
    weights = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=5120,
            feed_forward_length=17408,
            is_moe=False,
            rms_norm_eps=1.0e-6,
        ),
        layer=lambda layer_id: layer,
    )
    runner = object.__new__(qwen35_runner.Qwen35GGUFFullStackRunner)
    runner.weights = weights
    runner.runtime = SimpleNamespace()
    scratch = SimpleNamespace(
        post_norm=SimpleNamespace(ptr=400),
        residual=SimpleNamespace(ptr=500),
        ffn_gate_up=SimpleNamespace(ptr=600),
        ffn_intermediate=SimpleNamespace(ptr=700),
        ffn_down=SimpleNamespace(ptr=800),
    )
    return runner, scratch


def test_dense_runner_consumes_native_dual_rowtile_silu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        qwen35_runner,
        "gguf_add_rmsnorm_bf16_f32_weight",
        lambda *args, **kwargs: calls.append(("norm", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair_silu",
        lambda *args, **kwargs: calls.append(("pair_silu", args, kwargs)) or True,
        raising=False,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair",
        lambda *args, **kwargs: pytest.fail("fused route must skip the unfused pair"),
    )
    monkeypatch.setattr(
        qwen35_runner,
        "silu_mul_separate_out_bf16",
        lambda *args, **kwargs: pytest.fail("fused route must skip separate SiLU"),
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_residual",
        lambda *args, **kwargs: False,
        raising=False,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear",
        lambda *args, **kwargs: calls.append(("linear", args, kwargs)),
    )
    monkeypatch.setattr(
        qwen35_runner,
        "gguf_bf16_add",
        lambda *args, **kwargs: calls.append(("add", args, kwargs)),
    )

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=900,
        scratch=scratch,
        rows=4,
    )

    assert [name for name, _args, _kwargs in calls] == [
        "norm",
        "pair_silu",
        "linear",
        "add",
    ]
    pair_args = calls[1][1]
    pair_kwargs = calls[1][2]
    assert pair_args[2:4] == (400, 700)
    assert pair_kwargs["rows"] == 4
    assert pair_kwargs["in_features"] == 5120
    assert pair_kwargs["out_features"] == 17408
    assert pair_kwargs["runtime"] is runner.runtime
    assert calls[2][1][1:3] == (700, 800)


def test_dense_runner_fuses_down_residual_when_candidate_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(qwen35_runner, "gguf_add_rmsnorm_bf16_f32_weight", lambda *a, **k: None)
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair_silu",
        lambda *args, **kwargs: calls.append(("pair_silu", args, kwargs)) or True,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_residual",
        lambda *args, **kwargs: calls.append(("down_residual", args, kwargs)) or True,
        raising=False,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear",
        lambda *args, **kwargs: pytest.fail("fused down-residual must skip projection fallback"),
    )
    monkeypatch.setattr(
        qwen35_runner,
        "gguf_bf16_add",
        lambda *args, **kwargs: pytest.fail("fused down-residual must skip standalone add"),
    )

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=900,
        scratch=scratch,
        rows=4,
    )

    assert [name for name, _args, _kwargs in calls] == [
        "pair_silu",
        "down_residual",
    ]
    fused_args = calls[1][1]
    fused_kwargs = calls[1][2]
    assert fused_args[1:4] == (700, 500, 900)
    assert fused_args[4:] == (4, 17_408, 5_120)
    assert fused_kwargs["runtime"] is runner.runtime


def test_dense_runner_multirow_retains_unfused_chain_when_candidate_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[str] = []
    monkeypatch.setattr(qwen35_runner, "gguf_add_rmsnorm_bf16_f32_weight", lambda *a, **k: None)
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair_silu",
        lambda *args, **kwargs: calls.append("pair_silu") or False,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair",
        lambda *args, **kwargs: calls.append("pair") or True,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "silu_mul_separate_out_bf16",
        lambda *args, **kwargs: calls.append("silu"),
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_residual",
        lambda *args, **kwargs: calls.append("down_residual") or False,
        raising=False,
    )
    monkeypatch.setattr(qwen35_runner, "launch_gguf_linear", lambda *a, **k: calls.append("down"))
    monkeypatch.setattr(qwen35_runner, "gguf_bf16_add", lambda *a, **k: calls.append("add"))

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=900,
        scratch=scratch,
        rows=4,
    )

    assert calls == ["pair_silu", "pair", "silu", "down_residual", "down", "add"]


def test_dense_runner_c1_retains_unfused_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[str] = []
    monkeypatch.setattr(qwen35_runner, "gguf_add_rmsnorm_bf16_f32_weight", lambda *a, **k: None)
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair_silu",
        lambda *args, **kwargs: pytest.fail("c1 must retain its existing owner"),
        raising=False,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "launch_gguf_linear_pair",
        lambda *args, **kwargs: calls.append("pair") or True,
    )
    monkeypatch.setattr(
        qwen35_runner,
        "silu_mul_separate_out_bf16",
        lambda *args, **kwargs: calls.append("silu"),
    )
    monkeypatch.setattr(qwen35_runner, "launch_gguf_linear", lambda *a, **k: calls.append("down"))
    monkeypatch.setattr(qwen35_runner, "gguf_bf16_add", lambda *a, **k: calls.append("add"))

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=900,
        scratch=scratch,
        rows=1,
    )

    assert calls == ["pair", "silu", "down", "add"]


def _run_fused_chain(
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = repack_gguf_q4_k_pack8(make_q4_k_weight(out_features, in_features))
    up = repack_gguf_q4_k_pack8(make_q4_k_weight(out_features, in_features))
    # Make the two projections distinct without changing the packed layout.
    up.qweight[:] ^= np.int32(0x13579BDF)
    rng = np.random.default_rng(0xD28 + rows * 17 + in_features + out_features)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.15, size=(rows, in_features)).astype(np.float32)
    )
    control = np.empty((rows, out_features), dtype=np.uint16)
    candidate = np.empty_like(control)
    arrays = (
        x_bits,
        gate.qweight,
        gate.scales,
        gate.mins,
        up.qweight,
        up.scales,
        up.mins,
    )
    inputs = [malloc(array.nbytes) for array in arrays]
    output_nbytes = control.nbytes
    gate_d = malloc(output_nbytes)
    up_d = malloc(output_nbytes)
    control_d = malloc(output_nbytes)
    candidate_d = malloc(output_nbytes)
    library = build_gguf_q4_k_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, gate_q_d, gate_s_d, gate_m_d, up_q_d, up_s_d, up_m_d = inputs
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            x_d.ptr,
            gate_q_d.ptr,
            gate_s_d.ptr,
            gate_m_d.ptr,
            gate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            x_d.ptr,
            up_q_d.ptr,
            up_s_d.ptr,
            up_m_d.ptr,
            up_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        silu_mul_separate_out_bf16(
            gate_d.ptr,
            up_d.ptr,
            control_d.ptr,
            rows=rows,
            features=out_features,
        )
        gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
            x_d.ptr,
            gate_q_d.ptr,
            gate_s_d.ptr,
            gate_m_d.ptr,
            up_q_d.ptr,
            up_s_d.ptr,
            up_m_d.ptr,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(control), control_d, control.nbytes)
        copy_device_to_host(host_array_ptr(candidate), candidate_d, candidate.nbytes)
    finally:
        for allocation in (candidate_d, control_d, up_d, gate_d, *inputs):
            free(allocation)

    gate_cpu = gguf_q4_k_pack8_gemv(
        _bf16_f32(x_bits), gate.qweight, gate.scales, gate.mins
    )
    up_cpu = gguf_q4_k_pack8_gemv(
        _bf16_f32(x_bits), up.qweight, up.scales, up.mins
    )
    gate_cpu = _bf16_f32(_bf16_bits(gate_cpu))
    up_cpu = _bf16_f32(_bf16_bits(up_cpu))
    with np.errstate(over="ignore"):
        cpu = gate_cpu * (1.0 / (1.0 + np.exp(-gate_cpu))) * up_cpu
    return control, candidate, cpu


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4))
@pytest.mark.parametrize("in_features,out_features", ((256, 16), (512, 64)))
def test_pack8_dual_rowtile_silu_is_bit_exact_and_passes_cpu_gate(
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    control, candidate, cpu = _run_fused_chain(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    np.testing.assert_array_equal(candidate, control)

    gpu = _bf16_f32(candidate)
    np.testing.assert_allclose(gpu, cpu, rtol=0.04, atol=0.04)
    p = _softmax(cpu)
    q = _softmax(gpu)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    top1 = np.mean(np.argmax(cpu, axis=-1) == np.argmax(gpu, axis=-1))
    assert float(np.max(kl)) <= 0.05
    assert float(top1) >= 0.90
