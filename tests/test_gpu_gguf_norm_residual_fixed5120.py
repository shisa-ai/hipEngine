from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.ops import rmsnorm
from hipengine.kernels.hip_gfx1100.fused import gguf_ops
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

_HIDDEN = 5_120
_EPS = 1.0e-6
_RMS_WRAPPER = "gguf_rmsnorm_bf16_f32_weight_fixed5120_wave256"
_ADD_WRAPPER = "gguf_add_rmsnorm_bf16_f32_weight_fixed5120_wave256"
_ROUNDED_WRAPPER = "gguf_rounded_add_rmsnorm_bf16_f32_weight_fixed5120_wave256"
_KEYS = (
    KernelKey("hip_gfx1100", "rmsnorm", "gguf_f32_weight", "bf16_out_fixed5120_wave256"),
    KernelKey("hip_gfx1100", "add_rmsnorm", "gguf_f32_weight", "bf16_out_fixed5120_wave256"),
    KernelKey(
        "hip_gfx1100",
        "add+rmsnorm",
        "gguf_f32_weight",
        "rounded_bf16_out_fixed5120_wave256",
    ),
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _wrappers():
    return (
        getattr(gguf_ops, _RMS_WRAPPER, None),
        getattr(gguf_ops, _ADD_WRAPPER, None),
        getattr(gguf_ops, _ROUNDED_WRAPPER, None),
    )


def test_fixed5120_wave256_registry_and_preload_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    rms_fn, add_fn, rounded_fn = _wrappers()
    assert callable(rms_fn)
    assert callable(add_fn)
    assert callable(rounded_fn)
    gguf_ops.register_gguf_ops(replace=True)
    assert all(is_registered(key) for key in _KEYS)

    calls = []

    def fail_build(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid fixed shape must fail before build")

    monkeypatch.setattr(gguf_ops, "build_gguf_ops", fail_build)
    # 2026-09-13: both fixed-5120 wrappers now accept the verifier row slab.
    with pytest.raises(ValueError, match="rows must be between 1 and 8"):
        rms_fn(1, 2, 3, 9, _HIDDEN, _EPS)
    with pytest.raises(ValueError, match="rows must be between 1 and 8"):
        add_fn(1, 2, 3, 4, 5, 9, _HIDDEN, _EPS)
    # The rounded layer keeps its declared 2 <= rows <= 8 envelope.
    with pytest.raises(ValueError, match="rows must be 2 to 8"):
        rounded_fn(1, 2, 3, 4, 5, 1, _HIDDEN, _EPS)
    with pytest.raises(ValueError, match="rows must be 2 to 8"):
        rounded_fn(1, 2, 3, 4, 5, 9, _HIDDEN, _EPS)
    with pytest.raises(ValueError, match="hidden_size must be exactly 5120"):
        rounded_fn(1, 2, 3, 4, 5, 3, 1_024, _EPS)
    with pytest.raises(ValueError, match="hidden_size must be exactly 5120"):
        rms_fn(1, 2, 3, 1, 1_024, _EPS)
    with pytest.raises(ValueError, match="threads must be exactly 256"):
        add_fn(1, 2, 3, 4, 5, 1, _HIDDEN, _EPS, threads=128)
    for index, name in enumerate(
        ("x_ptr", "add_ptr", "weight_ptr", "norm_out_ptr", "residual_out_ptr")
    ):
        pointers = [1, 1, 1, 1, 1]
        pointers[index] = 0
        with pytest.raises(ValueError, match=rf"{name} must be non-zero"):
            add_fn(*pointers, 1, _HIDDEN, _EPS)
    assert calls == []


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_fixed5120_wave256_matches_current_and_cpu_oracles() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = gguf_ops.build_gguf_ops(load=True)
    rms_fn, add_fn, rounded_fn = _wrappers()
    assert callable(rms_fn) and callable(add_fn) and callable(rounded_fn)
    rng = np.random.default_rng(2026081601)
    edge_bits = np.asarray(
        [
            0x0000,
            0x8000,
            0x0001,
            0x007F,
            0x0080,
            0x3F7F,
            0x3F80,
            0x3F81,
            0xBF7F,
            0xBF80,
            0xBF81,
            0x4100,
            0xC100,
            0x4700,
            0xC700,
        ],
        dtype=np.uint16,
    )
    cases = [
        (
            float_array_to_bf16_bits(rng.normal(0, 0.8, _HIDDEN).astype(np.float32)),
            float_array_to_bf16_bits(rng.normal(0, 0.8, _HIDDEN).astype(np.float32)),
        ),
        (np.resize(edge_bits, _HIDDEN).copy(), np.resize(edge_bits[::-1], _HIDDEN).copy()),
    ]
    for case_index, (x, add) in enumerate(cases):
        weight = rng.uniform(0.25, 1.75, _HIDDEN).astype(np.float32)
        buffers = []
        try:

            def upload(array):
                array = np.ascontiguousarray(array)
                buf = malloc(array.nbytes, runtime=runtime)
                buffers.append(buf)
                copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
                return buf

            def output():
                buf = malloc(_HIDDEN * 2, runtime=runtime)
                buffers.append(buf)
                return buf

            def download(buf):
                array = np.empty(_HIDDEN, dtype=np.uint16)
                copy_device_to_host(host_array_ptr(array), buf, array.nbytes, runtime=runtime)
                return array

            x_d, add_d, weight_d = upload(x), upload(add), upload(weight)
            control_norm_d, control_residual_d = output(), output()
            candidate_norm_d, candidate_residual_d = output(), output()

            gguf_ops.gguf_rmsnorm_bf16_f32_weight(
                x_d.ptr,
                weight_d.ptr,
                control_norm_d.ptr,
                1,
                _HIDDEN,
                _EPS,
                threads=256,
                library=library,
                runtime=runtime,
            )
            rms_fn(
                x_d.ptr,
                weight_d.ptr,
                candidate_norm_d.ptr,
                1,
                _HIDDEN,
                _EPS,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            control_rms = download(control_norm_d)
            candidate_rms = download(candidate_norm_d)
            quality = evaluate_logits(
                rmsnorm(bf16_to_float32(x), weight, _EPS),
                bf16_to_float32(candidate_rms),
            )
            assert quality.kl_max <= 0.05
            assert quality.top1_agreement >= 0.90
            assert np.isfinite(bf16_to_float32(candidate_rms)).all()
            if case_index == 0:
                assert np.array_equal(control_rms, candidate_rms)
            else:
                assert np.array_equal(control_rms & 0x7FFF, candidate_rms & 0x7FFF)

            gguf_ops.gguf_add_rmsnorm_bf16_f32_weight(
                x_d.ptr,
                add_d.ptr,
                weight_d.ptr,
                control_norm_d.ptr,
                control_residual_d.ptr,
                1,
                _HIDDEN,
                _EPS,
                threads=256,
                library=library,
                runtime=runtime,
            )
            add_fn(
                x_d.ptr,
                add_d.ptr,
                weight_d.ptr,
                candidate_norm_d.ptr,
                candidate_residual_d.ptr,
                1,
                _HIDDEN,
                _EPS,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            control_norm = download(control_norm_d)
            control_residual = download(control_residual_d)
            candidate_norm = download(candidate_norm_d)
            candidate_residual = download(candidate_residual_d)
            unrounded = bf16_to_float32(x) + bf16_to_float32(add)
            quality = evaluate_logits(
                rmsnorm(unrounded, weight, _EPS),
                bf16_to_float32(candidate_norm),
            )
            assert quality.kl_max <= 0.05
            assert quality.top1_agreement >= 0.90
            assert np.isfinite(bf16_to_float32(candidate_norm)).all()
            if case_index == 0:
                assert np.array_equal(control_norm, candidate_norm)
                assert np.array_equal(control_residual, candidate_residual)
            else:
                assert np.array_equal(control_norm & 0x7FFF, candidate_norm & 0x7FFF)
                assert np.array_equal(
                    control_residual & 0x7FFF,
                    candidate_residual & 0x7FFF,
                )
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_fixed5120_rounded_leaf_is_bit_identical_to_the_generic_owner() -> None:
    """The register-cached rounded leaf must reproduce the generic rounded owner.

    Both consume bf16(bf16(x) + bf16(add)) for the reduction and the residual
    output, so the row slab and the unrolled second pass may only change which
    row a block owns, never the arithmetic.
    """

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = gguf_ops.build_gguf_ops(load=True)
    _, _, rounded_fn = _wrappers()
    assert callable(rounded_fn)
    rng = np.random.default_rng(2026091301)
    weight = rng.uniform(0.25, 1.75, _HIDDEN).astype(np.float32)
    for rows in (2, 3, 4, 5, 6, 8):
        x = float_array_to_bf16_bits(rng.normal(0, 0.8, (rows, _HIDDEN)).astype(np.float32))
        add = float_array_to_bf16_bits(rng.normal(0, 0.8, (rows, _HIDDEN)).astype(np.float32))
        buffers = []
        try:

            def upload(array):
                array = np.ascontiguousarray(array)
                buf = malloc(array.nbytes, runtime=runtime)
                buffers.append(buf)
                copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
                return buf

            def output():
                buf = malloc(rows * _HIDDEN * 2, runtime=runtime)
                buffers.append(buf)
                return buf

            def download(buf):
                array = np.empty(rows * _HIDDEN, dtype=np.uint16)
                copy_device_to_host(host_array_ptr(array), buf, array.nbytes, runtime=runtime)
                return array

            x_d, add_d, weight_d = upload(x), upload(add), upload(weight)
            control_norm_d, control_residual_d = output(), output()
            candidate_norm_d, candidate_residual_d = output(), output()
            gguf_ops.gguf_rounded_add_rmsnorm_bf16_f32_weight(
                x_d.ptr,
                add_d.ptr,
                weight_d.ptr,
                control_norm_d.ptr,
                control_residual_d.ptr,
                rows,
                _HIDDEN,
                _EPS,
                threads=256,
                library=library,
                runtime=runtime,
            )
            rounded_fn(
                x_d.ptr,
                add_d.ptr,
                weight_d.ptr,
                candidate_norm_d.ptr,
                candidate_residual_d.ptr,
                rows,
                _HIDDEN,
                _EPS,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            control_norm = download(control_norm_d)
            control_residual = download(control_residual_d)
            candidate_norm = download(candidate_norm_d)
            candidate_residual = download(candidate_residual_d)
            assert np.array_equal(control_norm, candidate_norm)
            assert np.array_equal(control_residual, candidate_residual)
            # The rounded residual must be the round-tripped sum, not the f32 one.
            rounded = bf16_to_float32(control_residual)
            exact = bf16_to_float32(x) + bf16_to_float32(add)
            assert not np.array_equal(rounded, exact)
            quality = evaluate_logits(
                rmsnorm(rounded.reshape(rows, _HIDDEN), weight, _EPS).reshape(-1),
                bf16_to_float32(candidate_norm),
            )
            assert quality.kl_max <= 0.05
            assert quality.top1_agreement >= 0.90
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
