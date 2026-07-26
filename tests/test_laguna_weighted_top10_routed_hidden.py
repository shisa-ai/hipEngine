"""Exact RED/GREEN gate for Laguna top-10 weighted+routed/hidden production."""

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
from hipengine.kernels.hip_gfx1100.fused import gguf_ops, paro_combine
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

_ROWS = 1
_TOP_K = 10
_HIDDEN = 3_072
_THREADS = 32
_VARIANT = "laguna_top10_routed_hidden_out"
_WRAPPER = "laguna_weighted_top10_routed_hidden_bf16_out"
_CPU_REFERENCE = "laguna_weighted_top10_routed_hidden"
_KEY = KernelKey(
    "hip_gfx1100",
    "weighted_sum+moe_tail",
    "bf16",
    _VARIANT,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def libraries():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return paro_combine.build_paro_combine(load=True), gguf_ops.build_gguf_ops(load=True)


def _candidate():
    return getattr(paro_combine, _WRAPPER, None)


def _cpu_reference():
    return getattr(cpu_ops, _CPU_REFERENCE, None)


def _upload(buffers: list, array: np.ndarray):
    array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
    return buffer


def _allocate(buffers: list, shape: tuple[int, ...]):
    array = np.empty(shape, dtype=np.uint16)
    buffer = malloc(array.nbytes)
    buffers.append(buffer)
    return buffer


def _download(buffer, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
    return out


def _free_all(buffers: list) -> None:
    for buffer in reversed(buffers):
        free(buffer)


def _case(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026072602 + sum(name.encode()))
    if name == "random":
        expert = rng.normal(0.0, 0.7, size=(_TOP_K, _HIDDEN)).astype(np.float32)
        weights = rng.uniform(0.01, 0.25, size=_TOP_K).astype(np.float32)
        shared = rng.normal(0.0, 0.7, size=_HIDDEN).astype(np.float32)
        post = rng.normal(0.0, 0.7, size=_HIDDEN).astype(np.float32)
    elif name == "rounding_edges":
        bf16_edges = np.asarray(
            [
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
                0x4F00,
                0xCF00,
            ],
            dtype=np.uint16,
        )
        expert = bf16_to_float32(
            np.resize(bf16_edges, (_TOP_K, _HIDDEN)).copy()
        )
        weights = np.asarray(
            [0.0, -0.0, 2.0**-24, -(2.0**-24), 0.125, -0.25, 0.5, 0.75, -1.0, 1.5],
            dtype=np.float32,
        )
        shared = bf16_to_float32(np.resize(bf16_edges[::-1], _HIDDEN).copy())
        post = bf16_to_float32(np.resize(np.roll(bf16_edges, 5), _HIDDEN).copy())
    elif name == "signed_zero":
        expert = np.zeros((_TOP_K, _HIDDEN), dtype=np.float32)
        expert[1::2] = np.float32(-0.0)
        weights = np.zeros(_TOP_K, dtype=np.float32)
        weights[::2] = np.float32(-0.0)
        shared = np.zeros(_HIDDEN, dtype=np.float32)
        shared[1::2] = np.float32(-0.0)
        post = np.zeros(_HIDDEN, dtype=np.float32)
        post[::2] = np.float32(-0.0)
    else:
        raise AssertionError(name)
    norm_weight = rng.uniform(0.25, 1.75, size=_HIDDEN).astype(np.float32)
    return (
        float_array_to_bf16_bits(expert),
        weights,
        float_array_to_bf16_bits(shared),
        float_array_to_bf16_bits(post),
        norm_weight,
    )


def test_laguna_weighted_top10_registry_and_preload_validation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    cpu_reference = _cpu_reference()
    assert callable(candidate), "Laguna weighted+routed/hidden wrapper must be admitted"
    assert callable(cpu_reference), "independent NumPy oracle must be admitted"

    paro_combine.register_paro_combine_kernels(replace=True)
    assert resolve(
        backend=_KEY.backend,
        layer=_KEY.layer,
        quant=_KEY.quant,
        variant=_KEY.variant,
    ) is candidate
    assert resolve(
        backend="hip_gfx1100",
        layer="weighted_sum",
        quant="bf16",
        variant="out",
    ) is paro_combine.weighted_sum_out_bf16_f32w

    build_calls: list[bool] = []

    def fail_build(*, load: bool = True, **_kwargs):
        build_calls.append(load)
        raise AssertionError("validation must reject before library load")

    monkeypatch.setattr(paro_combine, "build_paro_combine", fail_build)
    pointers = (1,) * 6
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        candidate(*pointers, 2, _TOP_K, _HIDDEN)
    with pytest.raises(ValueError, match="top_k must be exactly 10"):
        candidate(*pointers, _ROWS, _TOP_K - 1, _HIDDEN)
    with pytest.raises(ValueError, match="features must be exactly 3072"):
        candidate(*pointers, _ROWS, _TOP_K, _HIDDEN - 1)
    with pytest.raises(ValueError, match="threads must be 32"):
        candidate(*pointers, _ROWS, _TOP_K, _HIDDEN, threads=64)
    pointer_names = (
        "expert_down_ptr",
        "routing_weights_ptr",
        "shared_ptr",
        "post_attention_ptr",
        "routed_out_ptr",
        "hidden_out_ptr",
    )
    for index, pointer_name in enumerate(pointer_names):
        null_pointers = list(pointers)
        null_pointers[index] = 0
        with pytest.raises(ValueError, match=rf"{pointer_name} must be non-zero"):
            candidate(*null_pointers, _ROWS, _TOP_K, _HIDDEN)
    assert build_calls == []


def test_laguna_weighted_top10_key_is_excluded_from_unvalidated_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    for unvalidated_backend in ("hip_gfx1151", "cuda_sm86"):
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

    assert registered == []


def test_laguna_weighted_top10_cpu_reference_matches_hand_checked_fixture() -> None:
    cpu_reference = _cpu_reference()
    assert callable(cpu_reference), "independent NumPy oracle must be admitted"
    expert = np.zeros((_TOP_K, 3), dtype=np.float32)
    expert[0] = (1.0, 2.0, 3.0)
    expert[1] = (2.0, -4.0, 1.0)
    weights = np.zeros(_TOP_K, dtype=np.float32)
    weights[:2] = (0.5, 0.25)
    routed, hidden = cpu_reference(
        expert,
        weights,
        np.asarray((0.5, -0.5, 0.25), dtype=np.float32),
        np.asarray((-0.5, 0.5, 1.0), dtype=np.float32),
    )

    np.testing.assert_array_equal(routed, np.asarray((1.0, 0.0, 1.75), dtype=np.float32))
    np.testing.assert_array_equal(hidden, np.asarray((1.0, 0.0, 3.0), dtype=np.float32))


def test_laguna_weighted_top10_cpu_reference_preserves_all_three_bf16_boundaries() -> None:
    cpu_reference = _cpu_reference()
    assert callable(cpu_reference), "independent NumPy oracle must be admitted"
    hidden = 17
    rng = np.random.default_rng(2026072603)
    expert_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.8, size=(_TOP_K, hidden)).astype(np.float32)
    )
    weights = rng.uniform(-0.4, 0.4, size=_TOP_K).astype(np.float32)
    shared_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.8, size=hidden).astype(np.float32)
    )
    post_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.8, size=hidden).astype(np.float32)
    )
    routed, hidden_out = cpu_reference(
        bf16_to_float32(expert_bits),
        weights,
        bf16_to_float32(shared_bits),
        bf16_to_float32(post_bits),
    )

    routed_bits = float_array_to_bf16_bits(routed)
    first_add = bf16_to_float32(
        float_array_to_bf16_bits(routed + bf16_to_float32(shared_bits))
    )
    expected_hidden_bits = float_array_to_bf16_bits(
        bf16_to_float32(post_bits) + first_add
    )
    collapsed_hidden_bits = float_array_to_bf16_bits(
        bf16_to_float32(post_bits) + routed + bf16_to_float32(shared_bits)
    )
    np.testing.assert_array_equal(float_array_to_bf16_bits(hidden_out), expected_hidden_bits)
    assert routed_bits.shape == (hidden,)
    assert np.any(expected_hidden_bits != collapsed_hidden_bits)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("name", ("random", "rounding_edges", "signed_zero"))
def test_laguna_weighted_top10_is_bit_exact_to_registered_fallback_and_passes_cpu_gate(
    libraries,
    name: str,
) -> None:
    candidate = _candidate()
    cpu_reference = _cpu_reference()
    assert callable(candidate), "Laguna weighted+routed/hidden wrapper must be admitted"
    assert callable(cpu_reference), "independent NumPy oracle must be admitted"
    combine_library, gguf_library = libraries
    expert_bits, weights, shared_bits, post_bits, norm_weight = _case(name)
    buffers: list = []
    try:
        expert_d = _upload(buffers, expert_bits)
        weights_d = _upload(buffers, weights)
        shared_d = _upload(buffers, shared_bits)
        post_d = _upload(buffers, post_bits)
        norm_weight_d = _upload(buffers, norm_weight)
        control_routed_d = _allocate(buffers, (_HIDDEN,))
        control_hidden_d = _allocate(buffers, (_HIDDEN,))
        control_norm_d = _allocate(buffers, (_HIDDEN,))
        candidate_routed_d = _allocate(buffers, (_HIDDEN,))
        candidate_hidden_d = _allocate(buffers, (_HIDDEN,))
        candidate_norm_d = _allocate(buffers, (_HIDDEN,))

        paro_combine.weighted_sum_out_bf16_f32w(
            expert_d.ptr,
            weights_d.ptr,
            control_routed_d.ptr,
            _TOP_K,
            _HIDDEN,
            library=combine_library,
        )
        paro_combine.laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
            control_routed_d.ptr,
            shared_d.ptr,
            post_d.ptr,
            norm_weight_d.ptr,
            control_norm_d.ptr,
            control_hidden_d.ptr,
            _HIDDEN,
            library=combine_library,
        )
        candidate(
            expert_d.ptr,
            weights_d.ptr,
            shared_d.ptr,
            post_d.ptr,
            candidate_routed_d.ptr,
            candidate_hidden_d.ptr,
            _ROWS,
            _TOP_K,
            _HIDDEN,
            library=combine_library,
        )
        gguf_ops.gguf_rmsnorm_bf16_f32_weight(
            candidate_hidden_d.ptr,
            norm_weight_d.ptr,
            candidate_norm_d.ptr,
            _ROWS,
            _HIDDEN,
            1e-6,
            library=gguf_library,
        )

        control_routed = _download(control_routed_d, (_HIDDEN,))
        control_hidden = _download(control_hidden_d, (_HIDDEN,))
        control_norm = _download(control_norm_d, (_HIDDEN,))
        actual_routed = _download(candidate_routed_d, (_HIDDEN,))
        actual_hidden = _download(candidate_hidden_d, (_HIDDEN,))
        actual_norm = _download(candidate_norm_d, (_HIDDEN,))
    finally:
        _free_all(buffers)

    np.testing.assert_array_equal(actual_routed, control_routed)
    np.testing.assert_array_equal(actual_hidden, control_hidden)
    np.testing.assert_array_equal(actual_norm, control_norm)

    cpu_routed, cpu_hidden = cpu_reference(
        bf16_to_float32(expert_bits),
        weights,
        bf16_to_float32(shared_bits),
        bf16_to_float32(post_bits),
    )
    routed_result = evaluate_logits(
        np.asarray(cpu_routed, dtype=np.float32)[None, :],
        bf16_to_float32(actual_routed)[None, :],
    )
    hidden_result = evaluate_logits(
        np.asarray(cpu_hidden, dtype=np.float32)[None, :],
        bf16_to_float32(actual_hidden)[None, :],
    )
    assert routed_result.kl_mean <= 0.05
    assert routed_result.top1_agreement >= 0.90
    assert hidden_result.kl_mean <= 0.05
    assert hidden_result.top1_agreement >= 0.90
