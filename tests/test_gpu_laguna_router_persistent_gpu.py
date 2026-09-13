from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.moe.laguna_router import (
    build_laguna_router,
    laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10,
    laguna_sigmoid_correction_topk_f32,
    register_laguna_router_kernels,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    build_qwen35_router,
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.registry import resolve

_EXPERTS = 256
_TOP_K = 10
_SCALE = 2.5


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
    return build_laguna_router(load=True), build_qwen35_router(load=True)


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    bits = value.view(np.uint32).copy()
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(value.shape)


def _case(name: str, hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026072600 + hidden_size + sum(name.encode()))
    if name in {"random", "production_projection"}:
        hidden = _f32_to_bf16_u16(
            rng.normal(0.0, 0.04, size=hidden_size).astype(np.float32)
        )
        weight = rng.normal(0.0, 0.03, size=(_EXPERTS, hidden_size)).astype(np.float32)
        correction = rng.normal(0.0, 0.2, size=_EXPERTS).astype(np.float32)
        return hidden, weight, correction

    hidden_f32 = np.zeros(hidden_size, dtype=np.float32)
    hidden_f32[0] = np.float32(1.0)
    hidden = _f32_to_bf16_u16(hidden_f32)
    weight = np.zeros((_EXPERTS, hidden_size), dtype=np.float32)
    correction = np.zeros(_EXPERTS, dtype=np.float32)
    if name == "all_ties":
        return hidden, weight, correction
    if name == "finite_extremes":
        weight[:, 0] = np.linspace(-100.0, 100.0, _EXPERTS, dtype=np.float32)
        correction[::2] = np.float32(4.0)
        correction[1::2] = np.float32(-4.0)
        return hidden, weight, correction
    if name == "cross_wave_ties":
        weight[:, 0] = rng.normal(0.0, 1.0, size=_EXPERTS).astype(np.float32)
        tied = np.asarray([4, 39, 62, 97, 138, 145, 176, 201, 229, 244, 31, 255])
        weight[tied, 0] = np.float32(8.0)
        correction[tied] = np.float32(0.25)
        return hidden, weight, correction
    raise AssertionError(name)


def _empty_outputs() -> tuple[np.ndarray, ...]:
    return (
        np.full(_EXPERTS, np.nan, dtype=np.float32),
        np.full(_EXPERTS, np.nan, dtype=np.float32),
        np.full(_EXPERTS, np.nan, dtype=np.float32),
        np.full(_TOP_K, -1, dtype=np.int64),
        np.full(_TOP_K, np.nan, dtype=np.float32),
        np.full(_TOP_K, np.nan, dtype=np.float32),
    )


def test_persistent_wave_top10_registry_and_validation_contract() -> None:
    register_laguna_router_kernels(replace=True)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_router_topk",
            quant="f32",
            variant="bf16_hidden_correction_bias_persistent_wave_top10",
        )
        is laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10
    )

    fn = laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10
    with pytest.raises(ValueError, match="tokens == 1"):
        fn(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 17, _EXPERTS, _TOP_K, _SCALE)
    with pytest.raises(ValueError, match="hidden_size must be <= 3072"):
        fn(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3073, _EXPERTS, _TOP_K, _SCALE)
    with pytest.raises(ValueError, match="num_experts == 256"):
        fn(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 17, 255, _TOP_K, _SCALE)
    with pytest.raises(ValueError, match="top_k == 10"):
        fn(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 17, _EXPERTS, 9, _SCALE)
    with pytest.raises(ValueError, match="completion_counter_ptr must be nonzero"):
        fn(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 17, _EXPERTS, _TOP_K, _SCALE)
    with pytest.raises(ValueError, match="requires 256 threads"):
        fn(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 17, _EXPERTS, _TOP_K, _SCALE, threads=128)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("name", "hidden_size"),
    (
        ("random", 17),
        ("production_projection", 3_072),
        ("all_ties", 17),
        ("finite_extremes", 17),
        ("cross_wave_ties", 17),
    ),
)
def test_persistent_wave_top10_is_bit_exact_to_registered_split(
    libraries,
    name: str,
    hidden_size: int,
) -> None:
    router_library, logits_library = libraries
    runtime = get_hip_runtime()
    hidden, weight, correction = _case(name, hidden_size)
    control = _empty_outputs()
    candidate = _empty_outputs()
    counter = np.asarray([0], dtype=np.int32)
    arrays = (hidden, weight, correction, *control, *candidate, counter)
    buffers = [malloc(array.nbytes, runtime=runtime) for array in arrays]
    hidden_buf, weight_buf, correction_buf = buffers[:3]
    control_bufs = buffers[3:9]
    candidate_bufs = buffers[9:15]
    counter_buf = buffers[15]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        qwen35_router_logits_bf16_f32w_auto_256(
            hidden_buf.ptr,
            weight_buf.ptr,
            control_bufs[0].ptr,
            1,
            hidden_size,
            _EXPERTS,
            library=logits_library,
            runtime=runtime,
        )
        laguna_sigmoid_correction_topk_f32(
            control_bufs[0].ptr,
            correction_buf.ptr,
            control_bufs[1].ptr,
            control_bufs[2].ptr,
            control_bufs[3].ptr,
            control_bufs[4].ptr,
            control_bufs[5].ptr,
            1,
            _EXPERTS,
            _TOP_K,
            _SCALE,
            library=router_library,
            runtime=runtime,
        )

        fn = laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10
        for _ in range(2):
            fn(
                hidden_buf.ptr,
                weight_buf.ptr,
                correction_buf.ptr,
                candidate_bufs[0].ptr,
                candidate_bufs[1].ptr,
                candidate_bufs[2].ptr,
                candidate_bufs[3].ptr,
                candidate_bufs[4].ptr,
                candidate_bufs[5].ptr,
                counter_buf.ptr,
                1,
                hidden_size,
                _EXPERTS,
                _TOP_K,
                _SCALE,
                library=router_library,
                runtime=runtime,
            )
            for actual, buffer in zip(candidate, candidate_bufs, strict=True):
                copy_device_to_host(host_array_ptr(actual), buffer, actual.nbytes, runtime=runtime)
            runtime.memcpy(
                host_array_ptr(counter),
                counter_buf.ptr,
                counter.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            for expected, buffer in zip(control, control_bufs, strict=True):
                copy_device_to_host(host_array_ptr(expected), buffer, expected.nbytes, runtime=runtime)
            for expected, actual in zip(control, candidate, strict=True):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(counter, np.asarray([0], dtype=np.int32))
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
