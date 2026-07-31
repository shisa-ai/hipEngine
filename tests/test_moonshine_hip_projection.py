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
from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_projection,
    moonshine_triple_projection,
)
from hipengine.kernels.registry import resolve


def _hip_available() -> bool:
    if os.environ.get("HIPENGINE_HIP_ARCH") != "gfx1151":
        return False
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_moonshine_projection_registry_resolves_explicit_gfx1151_keys() -> None:
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_triple,
        register_moonshine_projection_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_moonshine_projection_kernels()
    register_gfx1151_kernels(replace=True)
    expected = {
        ("moonshine_projection", "single_fp32_accum"): moonshine_f16_projection,
        ("moonshine_projection_rows", "single_fp32_accum"): moonshine_f16_projection,
        ("moonshine_projection_bias", "single_fp32_accum"): moonshine_f16_projection_bias,
        ("moonshine_projection_pair", "pair_fp32_accum"): moonshine_f16_projection_pair,
        ("moonshine_qkv_proj", "triple_fp32_accum"): moonshine_f16_projection_triple,
    }
    for (layer, variant), function in expected.items():
        assert resolve(
            backend="hip_gfx1151",
            layer=layer,
            quant="fp16",
            variant=variant,
        ) is function


def test_moonshine_projection_wrappers_keep_raw_pointer_abi() -> None:
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_triple,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_f16_projection = FakeKernel()
        hipengine_moonshine_f16_projection_bias = FakeKernel()
        hipengine_moonshine_f16_projection_pair = FakeKernel()
        hipengine_moonshine_f16_projection_triple = FakeKernel()

    library = FakeLibrary()
    common = {"threads": 256, "stream": 7, "library": library, "runtime": object()}
    moonshine_f16_projection(1, 2, 3, 1, 416, 416, **common)
    moonshine_f16_projection_bias(1, 2, 3, 4, 1, 416, 416, **common)
    moonshine_f16_projection_pair(1, 2, 3, 4, 5, 1, 416, 416, 416, **common)
    moonshine_f16_projection_triple(
        1, 2, 3, 4, 5, 6, 7, 1, 416, 416, 416, 416, **common
    )
    assert library.hipengine_moonshine_f16_projection.calls == [
        (1, 2, 3, 1, 416, 416, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_bias.calls == [
        (1, 2, 3, 4, 1, 416, 416, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_pair.calls == [
        (1, 2, 3, 4, 5, 1, 416, 416, 416, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_triple.calls == [
        (1, 2, 3, 4, 5, 6, 7, 1, 416, 416, 416, 416, 256, 7)
    ]


def test_moonshine_projection_rejects_invalid_shapes_before_build() -> None:
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        moonshine_f16_projection,
    )

    with pytest.raises(ValueError, match="rows"):
        moonshine_f16_projection(1, 2, 3, 0, 416, 416)
    with pytest.raises(ValueError, match="threads"):
        moonshine_f16_projection(1, 2, 3, 1, 416, 416, threads=48)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_moonshine_projection_single_pair_triple_match_cpu_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_triple,
    )

    rng = np.random.default_rng(0x92B)
    hidden = 416
    x_one = rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float16)
    x_rows = rng.normal(0.0, 0.05, size=(40, hidden)).astype(np.float16)
    weights = tuple(
        rng.normal(0.0, 0.04, size=(hidden, hidden)).astype(np.float16)
        for _ in range(3)
    )
    bias = rng.normal(0.0, 0.03, size=(hidden,)).astype(np.float16)
    expected_one = moonshine_triple_projection(x_one, *weights)
    expected_bias = moonshine_projection(x_one, weights[0], bias)
    expected_rows = tuple(moonshine_projection(x_rows, weight) for weight in weights[:2])

    runtime = get_hip_runtime()
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        dx_one = _upload(x_one, runtime, allocations)
        dx_rows = _upload(x_rows, runtime, allocations)
        device_weights = tuple(_upload(weight, runtime, allocations) for weight in weights)
        device_bias = _upload(bias, runtime, allocations)
        single = _alloc((1, hidden), runtime, allocations)
        biased = _alloc((1, hidden), runtime, allocations)
        triple = tuple(_alloc((1, hidden), runtime, allocations) for _ in range(3))
        pair = tuple(_alloc((40, hidden), runtime, allocations) for _ in range(2))

        moonshine_f16_projection(
            dx_one.ptr, device_weights[0].ptr, single.ptr, 1, hidden, hidden,
            library=library, runtime=runtime,
        )
        moonshine_f16_projection_bias(
            dx_one.ptr, device_weights[0].ptr, device_bias.ptr, biased.ptr,
            1, hidden, hidden, library=library, runtime=runtime,
        )
        moonshine_f16_projection_triple(
            dx_one.ptr,
            *(weight.ptr for weight in device_weights),
            *(output.ptr for output in triple),
            1,
            hidden,
            hidden,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_pair(
            dx_rows.ptr,
            device_weights[0].ptr,
            device_weights[1].ptr,
            pair[0].ptr,
            pair[1].ptr,
            40,
            hidden,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_single = _download(single, (1, hidden), runtime)
        actual_bias = _download(biased, (1, hidden), runtime)
        actual_triple = tuple(_download(output, (1, hidden), runtime) for output in triple)
        actual_pair = tuple(_download(output, (40, hidden), runtime) for output in pair)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_single, expected_one[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual_bias, expected_bias, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_triple, expected_one, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_pair, expected_rows, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    assert all(np.isfinite(value).all() for value in (*actual_triple, *actual_pair))


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(np.float16).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
