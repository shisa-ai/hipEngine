"""Exact dense GGUF FFN-tail rounded add plus next-input RMSNorm."""

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
from hipengine.kernels.cpu_reference import rmsnorm
from hipengine.kernels.hip_gfx1100 import fused
from hipengine.kernels.hip_gfx1100.fused import gguf_ops
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime import qwen35_gguf_runner as runner_module

_KEY = KernelKey(
    "hip_gfx1100",
    "add+rmsnorm",
    "gguf_f32_weight",
    "rounded_bf16_out",
)
_WRAPPER = "gguf_rounded_add_rmsnorm_bf16_f32_weight"
_EPS = 1.0e-6


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _candidate():
    return getattr(gguf_ops, _WRAPPER, None)


def _upload(buffers: list, array: np.ndarray, *, runtime):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(
        buffer,
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )
    return buffer


def _allocate(buffers: list, shape: tuple[int, ...], *, runtime):
    array = np.empty(shape, dtype=np.uint16)
    buffer = malloc(array.nbytes, runtime=runtime)
    buffers.append(buffer)
    return buffer


def _download(buffer, shape: tuple[int, ...], *, runtime) -> np.ndarray:
    out = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes, runtime=runtime)
    return out


def _free_all(buffers: list, *, runtime) -> None:
    for buffer in reversed(buffers):
        free(buffer, runtime=runtime)


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
            hidden_size=5_120,
            feed_forward_length=17_408,
            is_moe=False,
            rms_norm_eps=_EPS,
        ),
        layer=lambda _layer_id: layer,
    )
    runner = object.__new__(runner_module.Qwen35GGUFFullStackRunner)
    runner.weights = weights
    runner.runtime = "runtime-sentinel"
    runner.backend = "hip_gfx1100"
    scratch = SimpleNamespace(
        norm=SimpleNamespace(ptr=350),
        post_norm=SimpleNamespace(ptr=400),
        residual=SimpleNamespace(ptr=500),
        ffn_gate_up=SimpleNamespace(ptr=600),
        ffn_intermediate=SimpleNamespace(ptr=700),
        ffn_down=SimpleNamespace(ptr=800),
        attn_out=SimpleNamespace(ptr=850),
    )
    return runner, scratch


def _patch_dense_prefix(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    monkeypatch.setattr(
        runner_module,
        "gguf_add_rmsnorm_bf16_f32_weight",
        lambda *args, **kwargs: calls.append(("post_norm", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear_pair_silu",
        lambda *args, **kwargs: calls.append(("pair_silu", args, kwargs)) or True,
    )
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear_pair",
        lambda *args, **kwargs: pytest.fail("fused gate/up+SiLU must own this fixture"),
    )
    monkeypatch.setattr(
        runner_module,
        "silu_mul_separate_out_bf16",
        lambda *args, **kwargs: pytest.fail("fused gate/up+SiLU must skip separate SiLU"),
    )


def test_rounded_add_rmsnorm_registry_export_and_peer_exclusion() -> None:
    candidate = _candidate()
    assert callable(candidate), "rounded add+RMSNorm wrapper must be admitted"
    assert getattr(fused, _WRAPPER, None) is candidate
    gguf_ops.register_gguf_ops(replace=True)
    assert resolve(
        backend=_KEY.backend,
        layer=_KEY.layer,
        quant=_KEY.quant,
        variant=_KEY.variant,
    ) is candidate

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels()
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _KEY.layer,
            _KEY.quant,
            _KEY.variant,
        )
    )


def test_rounded_add_rmsnorm_wrapper_rejects_unqualified_boundaries() -> None:
    candidate = _candidate()
    assert callable(candidate), "rounded add+RMSNorm wrapper must be admitted"
    pointers = (1,) * 5
    for rows in (1, 9):
        with pytest.raises(ValueError, match="rows must be between 2 and 8"):
            candidate(*pointers, rows, 5_120, _EPS)
    with pytest.raises(ValueError, match="threads must be exactly 256"):
        candidate(*pointers, 4, 5_120, _EPS, threads=128)
    for index, name in enumerate(
        ("residual_ptr", "add_ptr", "weight_ptr", "norm_out_ptr", "residual_out_ptr")
    ):
        nulls = list(pointers)
        nulls[index] = 0
        with pytest.raises(ValueError, match=rf"{name} must be non-zero"):
            candidate(*nulls, 4, 5_120, _EPS)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4, 5, 6, 7, 8))
def test_rounded_add_rmsnorm_is_bit_exact_to_add_then_rmsnorm(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    candidate = _candidate()
    assert callable(candidate), "rounded add+RMSNorm wrapper must be admitted"
    runtime = get_hip_runtime()
    library = gguf_ops.build_gguf_ops(load=True)
    hidden = 5_120
    rng = np.random.default_rng(0x364200 + rows)
    residual = float_array_to_bf16_bits(
        rng.normal(0.0, 0.31, size=(rows, hidden)).astype(np.float32)
    )
    add = float_array_to_bf16_bits(
        rng.normal(0.0, 0.23, size=(rows, hidden)).astype(np.float32)
    )
    weight = rng.uniform(0.5, 1.5, size=hidden).astype(np.float32)
    shape = (rows, hidden)
    buffers: list = []
    try:
        residual_d = _upload(buffers, residual, runtime=runtime)
        add_d = _upload(buffers, add, runtime=runtime)
        weight_d = _upload(buffers, weight, runtime=runtime)
        control_residual_d = _allocate(buffers, shape, runtime=runtime)
        control_norm_d = _allocate(buffers, shape, runtime=runtime)
        candidate_residual_d = _allocate(buffers, shape, runtime=runtime)
        candidate_norm_d = _allocate(buffers, shape, runtime=runtime)
        unrounded_residual_d = _allocate(buffers, shape, runtime=runtime)
        unrounded_norm_d = _allocate(buffers, shape, runtime=runtime)

        gguf_ops.gguf_bf16_add(
            residual_d.ptr,
            add_d.ptr,
            control_residual_d.ptr,
            rows * hidden,
            library=library,
            runtime=runtime,
        )
        gguf_ops.gguf_rmsnorm_bf16_f32_weight(
            control_residual_d.ptr,
            weight_d.ptr,
            control_norm_d.ptr,
            rows,
            hidden,
            _EPS,
            library=library,
            runtime=runtime,
        )
        candidate(
            residual_d.ptr,
            add_d.ptr,
            weight_d.ptr,
            candidate_norm_d.ptr,
            candidate_residual_d.ptr,
            rows,
            hidden,
            _EPS,
            library=library,
            runtime=runtime,
        )
        gguf_ops.gguf_add_rmsnorm_bf16_f32_weight(
            residual_d.ptr,
            add_d.ptr,
            weight_d.ptr,
            unrounded_norm_d.ptr,
            unrounded_residual_d.ptr,
            rows,
            hidden,
            _EPS,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        control_residual = _download(control_residual_d, shape, runtime=runtime)
        control_norm = _download(control_norm_d, shape, runtime=runtime)
        candidate_residual = _download(candidate_residual_d, shape, runtime=runtime)
        candidate_norm = _download(candidate_norm_d, shape, runtime=runtime)
        unrounded_residual = _download(unrounded_residual_d, shape, runtime=runtime)
        unrounded_norm = _download(unrounded_norm_d, shape, runtime=runtime)
    finally:
        _free_all(buffers, runtime=runtime)

    np.testing.assert_array_equal(candidate_residual, control_residual)
    np.testing.assert_array_equal(candidate_norm, control_norm)
    np.testing.assert_array_equal(unrounded_residual, control_residual)
    assert np.count_nonzero(unrounded_norm != control_norm) > 0

    # Bind the new kernel directly to the independent CPU-reference RMSNorm
    # gate in addition to proving bit identity with the registered HIP chain.
    cpu_residual_bits = float_array_to_bf16_bits(
        bf16_to_float32(residual) + bf16_to_float32(add)
    )
    np.testing.assert_array_equal(candidate_residual, cpu_residual_bits)
    cpu_norm = rmsnorm(bf16_to_float32(cpu_residual_bits), weight, eps=_EPS)
    gpu_norm = bf16_to_float32(candidate_norm)
    head_rng = np.random.default_rng(0x3642C0 + rows)
    head = head_rng.normal(
        0.0,
        1.0 / np.sqrt(hidden),
        size=(64, hidden),
    ).astype(np.float32)
    cpu_logits = (cpu_norm @ head.T).astype(np.float64)
    gpu_logits = (gpu_norm @ head.T).astype(np.float64)
    cpu_top1 = np.argmax(cpu_logits, axis=-1)
    gpu_top1 = np.argmax(gpu_logits, axis=-1)
    cpu_logits -= np.max(cpu_logits, axis=-1, keepdims=True)
    gpu_logits -= np.max(gpu_logits, axis=-1, keepdims=True)
    cpu_prob = np.exp(cpu_logits)
    gpu_prob = np.exp(gpu_logits)
    cpu_prob /= np.sum(cpu_prob, axis=-1, keepdims=True)
    gpu_prob /= np.sum(gpu_prob, axis=-1, keepdims=True)
    row_kl = np.sum(
        cpu_prob
        * (np.log(cpu_prob + 1.0e-300) - np.log(gpu_prob + 1.0e-300)),
        axis=-1,
    )
    top1 = np.mean(cpu_top1 == gpu_top1)
    assert float(np.max(row_kl)) <= 0.05
    assert float(top1) >= 0.90


def test_dense_runner_uses_projection_then_rounded_next_rmsnorm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[tuple[str, tuple, dict]] = []
    _patch_dense_prefix(monkeypatch, calls)

    def rounded(*args, **kwargs):
        calls.append(("rounded_next_rms", args, kwargs))

    monkeypatch.setattr(runner, "_rounded_add_rmsnorm_fn", lambda: rounded, raising=False)
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear_residual",
        lambda *args, **kwargs: pytest.fail(
            "next-RMSNorm successor must leave the projection body unchanged"
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear",
        lambda *args, **kwargs: calls.append(("down", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_module,
        "gguf_bf16_add",
        lambda *args, **kwargs: pytest.fail("rounded composite must replace standalone add"),
    )
    monkeypatch.setattr(
        runner_module,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda *args, **kwargs: pytest.fail("rounded composite must replace next RMSNorm"),
    )

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=900,
        scratch=scratch,
        rows=4,
        next_norm_weight_ptr=1_000,
        next_norm_out_ptr=1_100,
        stream=7,
    )

    assert [name for name, _args, _kwargs in calls] == [
        "post_norm",
        "pair_silu",
        "down",
        "rounded_next_rms",
    ]
    assert calls[2][1][1:3] == (700, 800)
    rounded_args, rounded_kwargs = calls[3][1:]
    assert rounded_args == (500, 800, 1_000, 1_100, 900, 4, 5_120, _EPS)
    assert rounded_kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_dense_runner_registry_miss_keeps_projection_add_rmsnorm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[tuple[str, tuple, dict]] = []
    _patch_dense_prefix(monkeypatch, calls)
    monkeypatch.setattr(runner, "_rounded_add_rmsnorm_fn", lambda: None, raising=False)
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear_residual",
        lambda *args, **kwargs: pytest.fail(
            "composite miss must keep the explicit projection+add+RMSNorm fallback"
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear",
        lambda *args, **kwargs: calls.append(("down", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_module,
        "gguf_bf16_add",
        lambda *args, **kwargs: calls.append(("add", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_module,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda *args, **kwargs: calls.append(("next_rms", args, kwargs)),
    )

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=900,
        scratch=scratch,
        rows=4,
        next_norm_weight_ptr=1_000,
        next_norm_out_ptr=1_100,
        stream=7,
    )

    assert [name for name, _args, _kwargs in calls] == [
        "post_norm",
        "pair_silu",
        "down",
        "add",
        "next_rms",
    ]
    assert calls[3][1][:3] == (500, 800, 900)
    assert calls[4][1][:3] == (900, 1_000, 1_100)


def test_dense_native_layer_forwards_prefused_input_and_next_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_dense_runner()
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        runner,
        "_run_linear_attention_attn_chain_rows_exact",
        lambda *args, **kwargs: calls.append(("attention", args, kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "_run_post_attention_ffn_rows",
        lambda *args, **kwargs: calls.append(("ffn", args, kwargs)),
    )

    runner._run_native_attention_bulk_ffn_layer_rows(
        0,
        LINEAR_ATTENTION,
        hidden_ptr=100,
        out_ptr=900,
        scratch=scratch,
        rows=4,
        decode_scratch=SimpleNamespace(),
        input_norm_ptr=350,
        next_norm_weight_ptr=1_000,
        next_norm_out_ptr=1_100,
        stream=7,
    )

    assert [name for name, _args, _kwargs in calls] == ["attention", "ffn"]
    assert calls[0][2]["input_norm_ptr"] == 350
    assert calls[1][2]["next_norm_weight_ptr"] == 1_000
    assert calls[1][2]["next_norm_out_ptr"] == 1_100
