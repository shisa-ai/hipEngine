"""Exact RED/GREEN gate for staged-F32 Laguna add+RMSNorm."""

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
from hipengine.kernels.cpu_reference import ops as cpu_ops
from hipengine.kernels.hip_gfx1100 import fused
from hipengine.kernels.hip_gfx1100.fused import gguf_ops
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

_VARIANT = "bf16_out_staged_f32_local256"
_WRAPPER = "gguf_add_rmsnorm_bf16_f32_weight_staged_f32_local256"
_KEY = KernelKey(
    "hip_gfx1100",
    "add_rmsnorm",
    "gguf_f32_weight",
    _VARIANT,
)
_THREADS = 256
_EPS = 1.0e-6


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def hip_context():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime(), gguf_ops.build_gguf_ops(load=True)


def _candidate():
    return getattr(gguf_ops, _WRAPPER, None)


def _upload(buffers: list, array: np.ndarray, *, runtime):
    array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
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


def _case(hidden_size: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026072604 + hidden_size + sum(kind.encode()))
    if kind == "random":
        x = float_array_to_bf16_bits(
            rng.normal(0.0, 0.7, size=hidden_size).astype(np.float32)
        )
        add = float_array_to_bf16_bits(
            rng.normal(0.0, 0.7, size=hidden_size).astype(np.float32)
        )
    elif kind == "bf16_edges":
        edge_bits = np.asarray(
            (
                0x0000,
                0x8000,
                0x0001,
                0x8001,
                0x007F,
                0x807F,
                0x0080,
                0x8080,
                0x3F7F,
                0x3F80,
                0x3F81,
                0xBF7F,
                0xBF80,
                0xBF81,
                0x4700,
                0xC700,
                0x7F7F,
                0xFF7F,
            ),
            dtype=np.uint16,
        )
        x = np.resize(edge_bits, hidden_size).copy()
        add = np.resize(edge_bits[::-1], hidden_size).copy()
    elif kind == "signed_zero":
        x = np.zeros(hidden_size, dtype=np.uint16)
        add = np.zeros(hidden_size, dtype=np.uint16)
        x[1::2] = np.uint16(0x8000)
        add[::2] = np.uint16(0x8000)
    else:
        raise AssertionError(kind)
    weight = rng.uniform(0.25, 1.75, size=hidden_size).astype(np.float32)
    return x, add, weight


def _launch_pair(
    x: np.ndarray,
    add: np.ndarray,
    weight: np.ndarray,
    *,
    runtime,
    library,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = _candidate()
    assert callable(candidate), "staged-F32 add+RMSNorm wrapper must be admitted"
    hidden_size = int(x.size)
    buffers: list = []
    try:
        x_d = _upload(buffers, x, runtime=runtime)
        add_d = _upload(buffers, add, runtime=runtime)
        weight_d = _upload(buffers, weight, runtime=runtime)
        control_norm_d = _allocate(buffers, (hidden_size,), runtime=runtime)
        control_residual_d = _allocate(buffers, (hidden_size,), runtime=runtime)
        candidate_norm_d = _allocate(buffers, (hidden_size,), runtime=runtime)
        candidate_residual_d = _allocate(buffers, (hidden_size,), runtime=runtime)

        gguf_ops.gguf_add_rmsnorm_bf16_f32_weight(
            x_d.ptr,
            add_d.ptr,
            weight_d.ptr,
            control_norm_d.ptr,
            control_residual_d.ptr,
            1,
            hidden_size,
            _EPS,
            threads=_THREADS,
            library=library,
            runtime=runtime,
        )
        candidate(
            x_d.ptr,
            add_d.ptr,
            weight_d.ptr,
            candidate_norm_d.ptr,
            candidate_residual_d.ptr,
            1,
            hidden_size,
            _EPS,
            threads=_THREADS,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return (
            _download(control_norm_d, (hidden_size,), runtime=runtime),
            _download(control_residual_d, (hidden_size,), runtime=runtime),
            _download(candidate_norm_d, (hidden_size,), runtime=runtime),
            _download(candidate_residual_d, (hidden_size,), runtime=runtime),
        )
    finally:
        _free_all(buffers, runtime=runtime)


def test_staged_f32_registry_package_export_and_preload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    assert callable(candidate), "staged-F32 add+RMSNorm wrapper must be admitted"
    assert getattr(fused, _WRAPPER, None) is candidate

    gguf_ops.register_gguf_ops(replace=True)
    assert resolve(
        backend=_KEY.backend,
        layer=_KEY.layer,
        quant=_KEY.quant,
        variant=_KEY.variant,
    ) is candidate
    assert resolve(
        backend="hip_gfx1100",
        layer="add_rmsnorm",
        quant="gguf_f32_weight",
        variant="bf16_out",
    ) is gguf_ops.gguf_add_rmsnorm_bf16_f32_weight

    build_calls: list[bool] = []

    def fail_build(*, load: bool = True, **_kwargs):
        build_calls.append(load)
        raise AssertionError("validation must reject before library load")

    monkeypatch.setattr(gguf_ops, "build_gguf_ops", fail_build)
    pointers = (1,) * 5
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        candidate(*pointers, 2, 3_072, _EPS)
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        candidate(*pointers, 1, 0, _EPS)
    with pytest.raises(ValueError, match="hidden_size must be <= 4096"):
        candidate(*pointers, 1, 4_352, _EPS)
    with pytest.raises(ValueError, match="hidden_size must be divisible by 256"):
        candidate(*pointers, 1, 3_073, _EPS)
    with pytest.raises(ValueError, match="threads must be exactly 256"):
        candidate(*pointers, 1, 3_072, _EPS, threads=128)
    pointer_names = (
        "x_ptr",
        "add_ptr",
        "weight_ptr",
        "norm_out_ptr",
        "residual_out_ptr",
    )
    for index, pointer_name in enumerate(pointer_names):
        null_pointers = list(pointers)
        null_pointers[index] = 0
        with pytest.raises(ValueError, match=rf"{pointer_name} must be non-zero"):
            candidate(*null_pointers, 1, 3_072, _EPS)
    assert build_calls == []


def test_staged_f32_key_is_excluded_from_unvalidated_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    for unvalidated_backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert resolve(
            backend=unvalidated_backend,
            layer=_KEY.layer,
            quant=_KEY.quant,
            variant=_KEY.variant,
            missing="none",
        ) is None

    registered: list[KernelKey] = []
    monkeypatch.setattr(backend, "import_module", lambda _name: None)
    monkeypatch.setattr(backend, "registered_keys", lambda: (_KEY,))
    monkeypatch.setattr(backend, "is_registered", lambda _key: False)
    monkeypatch.setattr(backend, "resolve", lambda **_kwargs: object())
    monkeypatch.setattr(
        backend,
        "register",
        lambda key, _kernel, *, replace=False: registered.append(key),
    )

    backend.register_gfx1151_kernels()

    assert all(
        (key.layer, key.quant, key.variant) != (_KEY.layer, _KEY.quant, _KEY.variant)
        for key in registered
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("hidden_size", "kind"),
    (
        (256, "bf16_edges"),
        (1_024, "signed_zero"),
        (3_072, "random"),
        (4_096, "bf16_edges"),
    ),
)
def test_staged_f32_is_bf16_bit_exact_to_registered_control(
    hip_context,
    hidden_size: int,
    kind: str,
) -> None:
    runtime, library = hip_context
    control_norm, control_residual, candidate_norm, candidate_residual = _launch_pair(
        *_case(hidden_size, kind),
        runtime=runtime,
        library=library,
    )
    np.testing.assert_array_equal(candidate_norm, control_norm)
    np.testing.assert_array_equal(candidate_residual, control_residual)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_staged_f32_passes_independent_cpu_gate_and_keeps_unrounded_norm_boundary(
    hip_context,
) -> None:
    runtime, library = hip_context
    x_bits, add_bits, weight = _case(1_024, "random")
    control_norm, control_residual, candidate_norm, candidate_residual = _launch_pair(
        x_bits,
        add_bits,
        weight,
        runtime=runtime,
        library=library,
    )
    np.testing.assert_array_equal(candidate_norm, control_norm)
    np.testing.assert_array_equal(candidate_residual, control_residual)

    unrounded_sum = (
        bf16_to_float32(x_bits) + bf16_to_float32(add_bits)
    ).astype(np.float32)
    cpu_norm = cpu_ops.rmsnorm(unrounded_sum[None, :], weight, eps=_EPS).astype(np.float32)
    candidate_norm_f32 = bf16_to_float32(candidate_norm)[None, :]
    quality = evaluate_logits(cpu_norm, candidate_norm_f32)
    assert quality.kl_max <= 0.05
    assert quality.top1_agreement >= 0.90

    rounded_sum = bf16_to_float32(float_array_to_bf16_bits(unrounded_sum))
    rounded_boundary_norm = float_array_to_bf16_bits(
        cpu_ops.rmsnorm(rounded_sum[None, :], weight, eps=_EPS).astype(np.float32)
    )[0]
    assert np.any(candidate_norm != rounded_boundary_norm)
