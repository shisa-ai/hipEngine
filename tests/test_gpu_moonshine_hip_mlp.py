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
from hipengine.kernels.registry import resolve


def _gfx1151_available() -> bool:
    if os.environ.get("HIPENGINE_HIP_ARCH") != "gfx1151":
        return False
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_moonshine_gated_silu_cpu_oracle_is_hand_checkable() -> None:
    from hipengine.kernels.cpu_reference.moonshine import moonshine_gated_silu

    values = np.asarray([[2.0, -3.0, 0.0, 1.0]], dtype=np.float16)
    actual = moonshine_gated_silu(values)
    gate = values[:, 2:].astype(np.float32)
    expected = (
        values[:, :2].astype(np.float32)
        * (gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32)))
    ).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)


def test_moonshine_gated_silu_registry_resolves_cpu_and_gfx1151() -> None:
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_gated_silu,
        register_moonshine_cpu_reference_kernels,
    )
    from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
        moonshine_gated_silu_fp16,
        register_moonshine_mlp_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_moonshine_cpu_reference_kernels()
    register_moonshine_mlp_kernels()
    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend="cpu_reference",
        layer="moonshine_gated_silu",
        quant="fp16",
        variant="value_gate_split",
    ) is moonshine_gated_silu
    assert resolve(
        backend="hip_gfx1151",
        layer="moonshine_gated_silu",
        quant="fp16",
        variant="value_gate_split",
    ) is moonshine_gated_silu_fp16


def test_moonshine_gated_silu_wrapper_keeps_raw_pointer_abi() -> None:
    from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
        moonshine_gated_silu_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_gated_silu_fp16 = FakeKernel()

    library = FakeLibrary()
    moonshine_gated_silu_fp16(
        1, 2, 7, 1664, threads=256, stream=9, library=library, runtime=object()
    )
    assert library.hipengine_moonshine_gated_silu_fp16.calls == [
        (1, 2, 7, 1664, 256, 9)
    ]


def test_moonshine_gated_silu_rejects_invalid_contract_before_build() -> None:
    from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
        moonshine_gated_silu_fp16,
    )

    with pytest.raises(ValueError, match="rows"):
        moonshine_gated_silu_fp16(1, 2, 0, 1664)
    with pytest.raises(ValueError, match="intermediate_size"):
        moonshine_gated_silu_fp16(1, 2, 1, 0)
    with pytest.raises(ValueError, match="threads"):
        moonshine_gated_silu_fp16(1, 2, 1, 1664, threads=48)


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_gated_silu_production_shape_matches_cpu_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.moonshine import moonshine_gated_silu
    from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
        build_moonshine_mlp,
        moonshine_gated_silu_fp16,
    )

    rng = np.random.default_rng(0x5110)
    rows, intermediate = 7, 1664
    inputs = rng.normal(0.0, 0.4, size=(rows, 2 * intermediate)).astype(np.float16)
    expected = moonshine_gated_silu(inputs)
    runtime = get_hip_runtime()
    library = build_moonshine_mlp(load=True)
    allocations = []
    try:
        device_input = _upload(inputs, runtime, allocations)
        device_output = _empty(expected.shape, runtime, allocations)
        moonshine_gated_silu_fp16(
            device_input.ptr,
            device_output.ptr,
            rows,
            intermediate,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(device_output, expected.shape, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    assert np.isfinite(actual).all()


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_unfused_gated_mlp_chain_matches_decoder_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_decoder_mlp,
        moonshine_residual,
    )
    from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_residual_fp16,
    )
    from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
        build_moonshine_mlp,
        moonshine_gated_silu_fp16,
    )
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_projection_bias,
    )

    rng = np.random.default_rng(0xDEC0DE)
    rows, hidden, intermediate = 1, 416, 1664
    normalized = rng.normal(0.0, 0.06, size=(rows, hidden)).astype(np.float16)
    residual = rng.normal(0.0, 0.08, size=(rows, hidden)).astype(np.float16)
    fc1_weight = rng.normal(
        0.0, 0.025, size=(2 * intermediate, hidden)
    ).astype(np.float16)
    fc1_bias = rng.normal(0.0, 0.02, size=(2 * intermediate,)).astype(np.float16)
    fc2_weight = rng.normal(0.0, 0.025, size=(hidden, intermediate)).astype(np.float16)
    fc2_bias = rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float16)
    expected_mlp = moonshine_decoder_mlp(
        normalized,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    expected = moonshine_residual(residual, expected_mlp)

    runtime = get_hip_runtime()
    projection = build_moonshine_projection(load=True)
    mlp = build_moonshine_mlp(load=True)
    glue = build_moonshine_glue(load=True)
    allocations = []
    try:
        device_normalized = _upload(normalized, runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_fc1_weight = _upload(fc1_weight, runtime, allocations)
        device_fc1_bias = _upload(fc1_bias, runtime, allocations)
        device_fc2_weight = _upload(fc2_weight, runtime, allocations)
        device_fc2_bias = _upload(fc2_bias, runtime, allocations)
        fc1_output = _empty((rows, 2 * intermediate), runtime, allocations)
        intermediate_output = _empty((rows, intermediate), runtime, allocations)
        fc2_output = _empty((rows, hidden), runtime, allocations)
        final_output = _empty((rows, hidden), runtime, allocations)

        moonshine_f16_projection_bias(
            device_normalized.ptr,
            device_fc1_weight.ptr,
            device_fc1_bias.ptr,
            fc1_output.ptr,
            rows,
            hidden,
            2 * intermediate,
            library=projection,
            runtime=runtime,
        )
        moonshine_gated_silu_fp16(
            fc1_output.ptr,
            intermediate_output.ptr,
            rows,
            intermediate,
            library=mlp,
            runtime=runtime,
        )
        moonshine_f16_projection_bias(
            intermediate_output.ptr,
            device_fc2_weight.ptr,
            device_fc2_bias.ptr,
            fc2_output.ptr,
            rows,
            intermediate,
            hidden,
            library=projection,
            runtime=runtime,
        )
        moonshine_residual_fp16(
            device_residual.ptr,
            fc2_output.ptr,
            final_output.ptr,
            rows * hidden,
            library=glue,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(final_output, expected.shape, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=5.0e-3, atol=5.0e-3)
    assert np.isfinite(actual).all()


def test_moonshine_fused_mlp_projection_registry_and_raw_pointer_abis() -> None:
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_mlp_fc1_gated_silu,
        moonshine_projection_bias_residual,
        register_moonshine_cpu_reference_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        moonshine_f16_projection_bias_gated_silu,
        moonshine_f16_projection_bias_residual,
        register_moonshine_projection_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_f16_projection_bias_gated_silu = FakeKernel()
        hipengine_moonshine_f16_projection_bias_residual = FakeKernel()

    register_moonshine_cpu_reference_kernels()
    register_moonshine_projection_kernels()
    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend="hip_gfx1151",
        layer="moonshine_mlp_fc1",
        quant="fp16",
        variant="bias_gated_silu_fp32_accum",
    ) is moonshine_f16_projection_bias_gated_silu
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_mlp_fc1",
        quant="fp16",
        variant="bias_gated_silu_fp32_accum",
    ) is moonshine_mlp_fc1_gated_silu
    assert resolve(
        backend="hip_gfx1151",
        layer="moonshine_mlp_fc2_residual",
        quant="fp16",
        variant="bias_rounded_residual_fp32_accum",
    ) is moonshine_f16_projection_bias_residual
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_mlp_fc2_residual",
        quant="fp16",
        variant="bias_rounded_residual_fp32_accum",
    ) is moonshine_projection_bias_residual

    library = FakeLibrary()
    common = {"stream": 7, "library": library, "runtime": object()}
    moonshine_f16_projection_bias_gated_silu(
        1, 2, 3, 4, 1, 416, 1664, threads=32, **common
    )
    moonshine_f16_projection_bias_residual(
        1, 2, 3, 4, 5, 1, 1664, 416, threads=64, **common
    )
    assert library.hipengine_moonshine_f16_projection_bias_gated_silu.calls == [
        (1, 2, 3, 4, 1, 416, 1664, 32, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_bias_residual.calls == [
        (1, 2, 3, 4, 5, 1, 1664, 416, 64, 7)
    ]


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_fused_mlp_projection_chain_matches_decoder_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_decoder_mlp,
        moonshine_residual,
    )
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_projection_bias_gated_silu,
        moonshine_f16_projection_bias_residual,
    )

    rng = np.random.default_rng(0xF05ED)
    rows, hidden, intermediate = 1, 416, 1664
    normalized = rng.normal(0.0, 0.06, size=(rows, hidden)).astype(np.float16)
    residual = rng.normal(0.0, 0.08, size=(rows, hidden)).astype(np.float16)
    fc1_weight = rng.normal(
        0.0, 0.025, size=(2 * intermediate, hidden)
    ).astype(np.float16)
    fc1_bias = rng.normal(0.0, 0.02, size=(2 * intermediate,)).astype(np.float16)
    fc2_weight = rng.normal(0.0, 0.025, size=(hidden, intermediate)).astype(np.float16)
    fc2_bias = rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float16)
    expected = moonshine_residual(
        residual,
        moonshine_decoder_mlp(
            normalized,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        ),
    )
    runtime = get_hip_runtime()
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        device_normalized = _upload(normalized, runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_fc1_weight = _upload(fc1_weight, runtime, allocations)
        device_fc1_bias = _upload(fc1_bias, runtime, allocations)
        device_fc2_weight = _upload(fc2_weight, runtime, allocations)
        device_fc2_bias = _upload(fc2_bias, runtime, allocations)
        intermediate_output = _empty((rows, intermediate), runtime, allocations)
        final_output = _empty((rows, hidden), runtime, allocations)
        moonshine_f16_projection_bias_gated_silu(
            device_normalized.ptr,
            device_fc1_weight.ptr,
            device_fc1_bias.ptr,
            intermediate_output.ptr,
            rows,
            hidden,
            intermediate,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_bias_residual(
            intermediate_output.ptr,
            device_fc2_weight.ptr,
            device_fc2_bias.ptr,
            device_residual.ptr,
            final_output.ptr,
            rows,
            intermediate,
            hidden,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(final_output, expected.shape, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _empty(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * 2, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
