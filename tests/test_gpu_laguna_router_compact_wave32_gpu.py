from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.laguna import (
    laguna_sigmoid_correction_topk_from_logits,
)
from hipengine.kernels.hip_gfx1100.moe import laguna_router
from hipengine.kernels.registry import KernelKey, resolve

_EXPERTS = 256
_TOP_K = 10
_SCALE = 2.5
_VARIANT = "correction_bias_compact_wave32"
_WRAPPER = "laguna_sigmoid_correction_topk_compact_wave32_f32"
_KEY = KernelKey(
    "hip_gfx1100",
    "laguna_sigmoid_router_topk",
    "f32",
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
def router_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return laguna_router.build_laguna_router(load=True)


def _candidate():
    return getattr(laguna_router, _WRAPPER, None)


def _case(name: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026072601 + sum(name.encode()))
    if name == "random":
        return (
            rng.normal(0.0, 4.0, size=_EXPERTS).astype(np.float32),
            rng.normal(0.0, 0.2, size=_EXPERTS).astype(np.float32),
        )
    if name == "all_ties":
        return np.zeros(_EXPERTS, dtype=np.float32), np.zeros(_EXPERTS, dtype=np.float32)
    if name == "cross_lane_cross_item_ties":
        logits = rng.normal(-3.0, 0.25, size=_EXPERTS).astype(np.float32)
        correction = rng.normal(0.0, 0.01, size=_EXPERTS).astype(np.float32)
        tied = np.asarray((1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 224, 255))
        logits[tied] = np.float32(8.0)
        correction[tied] = np.float32(0.25)
        return logits, correction
    if name == "finite_extremes":
        limit = np.finfo(np.float32).max
        logits = np.linspace(-100.0, 100.0, _EXPERTS, dtype=np.float32)
        logits[[0, 1, 254, 255]] = (-limit, -1.0e20, 1.0e20, limit)
        correction = np.linspace(-4.0, 4.0, _EXPERTS, dtype=np.float32)
        return logits, correction
    if name == "signed_zero":
        logits = np.zeros(_EXPERTS, dtype=np.float32)
        correction = np.zeros(_EXPERTS, dtype=np.float32)
        logits[1::2] = np.float32(-0.0)
        correction[::2] = np.float32(-0.0)
        return logits, correction
    raise AssertionError(name)


def _empty_outputs() -> tuple[np.ndarray, ...]:
    return (
        np.full(_EXPERTS, np.nan, dtype=np.float32),
        np.full(_EXPERTS, np.nan, dtype=np.float32),
        np.full(_TOP_K, -1, dtype=np.int64),
        np.full(_TOP_K, np.nan, dtype=np.float32),
        np.full(_TOP_K, np.nan, dtype=np.float32),
    )


def _launch(
    fn,
    logits: np.ndarray,
    correction: np.ndarray,
    *,
    library,
) -> tuple[np.ndarray, ...]:
    outputs = _empty_outputs()
    arrays = (logits, correction, *outputs)
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        fn(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            buffers[5].ptr,
            buffers[6].ptr,
            1,
            _EXPERTS,
            _TOP_K,
            _SCALE,
            library=library,
        )
        for array, buffer in zip(outputs, buffers[2:], strict=True):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
        return outputs
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def test_compact_wave32_registry_and_validation_contract() -> None:
    candidate = _candidate()
    assert callable(candidate), "compact-wave32 wrapper must be admitted"

    laguna_router.register_laguna_router_kernels(replace=True)
    assert resolve(
        backend=_KEY.backend,
        layer=_KEY.layer,
        quant=_KEY.quant,
        variant=_KEY.variant,
    ) is candidate
    assert resolve(
        backend="hip_gfx1100",
        layer="laguna_sigmoid_router_topk",
        quant="f32",
        variant="correction_bias",
    ) is laguna_router.laguna_sigmoid_correction_topk_f32

    pointers = (0,) * 7
    with pytest.raises(ValueError, match="tokens == 1"):
        candidate(*pointers, 2, _EXPERTS, _TOP_K, _SCALE)
    with pytest.raises(ValueError, match="num_experts == 256"):
        candidate(*pointers, 1, _EXPERTS - 1, _TOP_K, _SCALE)
    with pytest.raises(ValueError, match="top_k == 10"):
        candidate(*pointers, 1, _EXPERTS, _TOP_K - 1, _SCALE)
    with pytest.raises(ValueError, match="routed_scaling_factor"):
        candidate(*pointers, 1, _EXPERTS, _TOP_K, 0.0)
    with pytest.raises(ValueError, match="requires 32 threads"):
        candidate(*pointers, 1, _EXPERTS, _TOP_K, _SCALE, threads=64)


def test_compact_wave32_is_excluded_from_gfx1151_aliasing(
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

    assert not any(
        key.layer == _KEY.layer
        and key.quant == _KEY.quant
        and key.variant == _KEY.variant
        for key in registered
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "name",
    (
        "random",
        "all_ties",
        "cross_lane_cross_item_ties",
        "finite_extremes",
        "signed_zero",
    ),
)
def test_compact_wave32_is_field_bit_exact_and_passes_cpu_gate(
    router_library,
    name: str,
) -> None:
    candidate = _candidate()
    assert callable(candidate), "compact-wave32 wrapper must be admitted"
    logits, correction = _case(name)
    control = _launch(
        laguna_router.laguna_sigmoid_correction_topk_f32,
        logits,
        correction,
        library=router_library,
    )
    actual = _launch(candidate, logits, correction, library=router_library)

    for expected_field, actual_field in zip(control, actual, strict=True):
        np.testing.assert_array_equal(actual_field, expected_field)

    cpu = laguna_sigmoid_correction_topk_from_logits(
        logits[None, :],
        correction,
        experts_used=_TOP_K,
        routed_scaling_factor=_SCALE,
    )
    np.testing.assert_array_equal(actual[2], cpu.selected_experts[0])
    top1_agreement = float(actual[2][0] == cpu.selected_experts[0, 0])
    candidate_distribution = actual[0].astype(np.float64)
    cpu_distribution = cpu.routing_scores[0].astype(np.float64)
    candidate_distribution /= candidate_distribution.sum()
    cpu_distribution /= cpu_distribution.sum()
    nonzero = candidate_distribution > 0.0
    kl = float(
        np.sum(
            candidate_distribution[nonzero]
            * np.log(candidate_distribution[nonzero] / cpu_distribution[nonzero])
        )
    )
    assert kl <= 0.05
    assert top1_agreement >= 0.90
