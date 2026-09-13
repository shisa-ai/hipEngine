from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.moonshine import moonshine_layernorm
from hipengine.kernels.registry import resolve


def _gfx1151_available() -> bool:
    if os.environ.get("HIPENGINE_HIP_ARCH") != "gfx1151":
        return False
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_moonshine_layernorm_registry_resolves_explicit_gfx1151_key() -> None:
    from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
        register_moonshine_layernorm_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_moonshine_layernorm_kernels()
    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend="hip_gfx1151",
        layer="moonshine_layernorm",
        quant="fp16",
        variant="fp32_stats",
    ) is moonshine_layernorm_fp16


def test_moonshine_layernorm_wrapper_keeps_raw_pointer_abi() -> None:
    from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_layernorm_fp16 = FakeKernel()

    library = FakeLibrary()
    moonshine_layernorm_fp16(
        1,
        2,
        3,
        7,
        416,
        eps=1.0e-5,
        threads=256,
        stream=9,
        library=library,
        runtime=object(),
    )
    assert library.hipengine_moonshine_layernorm_fp16.calls == [
        (1, 2, 3, 7, 416, pytest.approx(1.0e-5), 256, 9)
    ]


def test_moonshine_layernorm_rejects_invalid_contract_before_build() -> None:
    from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
    )

    with pytest.raises(ValueError, match="rows"):
        moonshine_layernorm_fp16(1, 2, 3, 0, 416)
    with pytest.raises(ValueError, match="eps"):
        moonshine_layernorm_fp16(1, 2, 3, 1, 416, eps=0.0)
    with pytest.raises(ValueError, match="threads"):
        moonshine_layernorm_fp16(1, 2, 3, 1, 416, threads=48)


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_layernorm_hidden416_matches_fp32_stats_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_layernorm_fp16,
    )

    rng = np.random.default_rng(0x1A92)
    rows, hidden = 7, 416
    inputs = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    weights = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    expected = moonshine_layernorm(inputs, weights)
    runtime = get_hip_runtime()
    library = build_moonshine_layernorm(load=True)
    allocations = []
    try:
        device_input = _upload(inputs, runtime, allocations)
        device_weight = _upload(weights, runtime, allocations)
        device_output = malloc(expected.nbytes, runtime=runtime)
        allocations.append(device_output)
        moonshine_layernorm_fp16(
            device_input.ptr,
            device_weight.ptr,
            device_output.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(actual), device_output, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=3.0e-3, atol=3.0e-3)
    assert np.isfinite(actual).all()


def test_moonshine_residual_layernorm_registry_and_raw_pointer_abi() -> None:
    from hipengine.kernels.cpu_reference.moonshine import (
        register_moonshine_cpu_reference_kernels,
    )
    from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
        moonshine_residual_layernorm_fp16,
        register_moonshine_layernorm_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_residual_layernorm_fp16 = FakeKernel()

    register_moonshine_cpu_reference_kernels()
    register_moonshine_layernorm_kernels()
    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend="hip_gfx1151",
        layer="moonshine_residual+moonshine_layernorm",
        quant="fp16",
        variant="rounded_fp32_stats",
    ) is moonshine_residual_layernorm_fp16

    library = FakeLibrary()
    moonshine_residual_layernorm_fp16(
        1,
        2,
        3,
        4,
        5,
        1,
        416,
        eps=1.0e-5,
        threads=256,
        stream=9,
        library=library,
        runtime=object(),
    )
    assert library.hipengine_moonshine_residual_layernorm_fp16.calls == [
        (1, 2, 3, 4, 5, 1, 416, pytest.approx(1.0e-5), 256, 9)
    ]


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_residual_layernorm_matches_unfused_boundaries() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.moonshine import moonshine_residual_layernorm
    from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_residual_layernorm_fp16,
    )

    rng = np.random.default_rng(0xADD10)
    rows, hidden = 7, 416
    residual = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    update = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float16)
    weight = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    expected_residual, expected_norm = moonshine_residual_layernorm(
        residual, update, weight
    )
    runtime = get_hip_runtime()
    library = build_moonshine_layernorm(load=True)
    allocations = []
    try:
        device_residual = _upload(residual, runtime, allocations)
        device_update = _upload(update, runtime, allocations)
        device_weight = _upload(weight, runtime, allocations)
        residual_output = malloc(expected_residual.nbytes, runtime=runtime)
        norm_output = malloc(expected_norm.nbytes, runtime=runtime)
        allocations.extend((residual_output, norm_output))
        moonshine_residual_layernorm_fp16(
            device_residual.ptr,
            device_update.ptr,
            device_weight.ptr,
            residual_output.ptr,
            norm_output.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_residual = np.empty_like(expected_residual)
        actual_norm = np.empty_like(expected_norm)
        copy_device_to_host(
            host_array_ptr(actual_residual), residual_output, runtime=runtime
        )
        copy_device_to_host(host_array_ptr(actual_norm), norm_output, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual_residual, expected_residual)
    np.testing.assert_allclose(actual_norm, expected_norm, rtol=3.0e-3, atol=3.0e-3)
    assert np.isfinite(actual_norm).all()


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device
