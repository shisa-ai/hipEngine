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
    moonshine_decoder_mlp,
    moonshine_gated_silu,
    moonshine_residual,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def _cuda_sm120a_enabled() -> bool:
    if os.environ.get("HIPENGINE_RUN_CUDA_SM120A") != "1":
        return False
    if os.environ.get("HIPENGINE_CUDA_ARCH") != "sm_120a":
        return False
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError:
        return False
    return True


def setup_function() -> None:
    clear_registry_for_tests()


def test_moonshine_cuda_mlp_registry_resolves_explicit_keys() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
        moonshine_gated_silu_fp16,
        register_moonshine_mlp_kernels,
    )
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_gated_silu,
        register_moonshine_cpu_reference_kernels,
    )

    register_moonshine_cpu_reference_kernels()
    register_moonshine_mlp_kernels()
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_gated_silu",
        quant="fp16",
        variant="value_gate_split",
    ) is moonshine_gated_silu_fp16
    assert resolve(
        backend="cpu_reference",
        layer="moonshine_gated_silu",
        quant="fp16",
        variant="value_gate_split",
    ) is moonshine_gated_silu


def test_moonshine_cuda_mlp_build_plan_targets_sm120a(tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
        plan_moonshine_mlp_build,
    )

    artifact = plan_moonshine_mlp_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc Moonshine test version",
    )
    assert artifact.family == "cuda_sm120a_moonshine_mlp"
    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)
    assert artifact.output_path.name == "moonshine_mlp.so"
    assert not artifact.cache_dir.exists()


def test_moonshine_cuda_mlp_wrapper_keeps_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
        moonshine_gated_silu_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_gated_silu_fp16 = FakeKernel()

    library = FakeLibrary()
    moonshine_gated_silu_fp16(
        1,
        2,
        3,
        1664,
        threads=256,
        stream=7,
        library=library,
        runtime=object(),
    )
    assert library.hipengine_cuda_sm120a_moonshine_gated_silu_fp16.calls == [
        (1, 2, 3, 1664, 256, 7)
    ]


def test_moonshine_cuda_mlp_rejects_invalid_contract_before_build() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
        moonshine_gated_silu_fp16,
    )

    with pytest.raises(ValueError, match="rows"):
        moonshine_gated_silu_fp16(1, 2, 0, 1664)
    with pytest.raises(ValueError, match="intermediate_size"):
        moonshine_gated_silu_fp16(1, 2, 3, 0)
    with pytest.raises(ValueError, match="threads"):
        moonshine_gated_silu_fp16(1, 2, 3, 1664, threads=48)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_mlp_gated_silu_matches_cpu_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
        build_moonshine_mlp,
        moonshine_gated_silu_fp16,
    )

    rng = np.random.default_rng(0xC1D0)
    intermediate = 1664
    row_buckets = (1, 7, 40, 207, 1248)
    expected = {}
    inputs = {}
    for rows in row_buckets:
        fc1 = rng.normal(0.0, 0.05, size=(rows, 2 * intermediate)).astype(np.float16)
        inputs[rows] = fc1
        expected[rows] = moonshine_gated_silu(fc1)

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_mlp(load=True)
    allocations = []
    try:
        device_inputs = {rows: _upload(fc1, runtime, allocations) for rows, fc1 in inputs.items()}
        device_outputs = {
            rows: _alloc((rows, intermediate), runtime, allocations) for rows in row_buckets
        }
        for rows in row_buckets:
            moonshine_gated_silu_fp16(
                device_inputs[rows].ptr,
                device_outputs[rows].ptr,
                rows,
                intermediate,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        actual = {
            rows: _download(device_outputs[rows], (rows, intermediate), runtime)
            for rows in row_buckets
        }
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    for rows in row_buckets:
        np.testing.assert_allclose(actual[rows], expected[rows], rtol=1e-3, atol=1e-3)
        assert np.isfinite(actual[rows]).all()


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_unfused_gated_mlp_chain_matches_decoder_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_residual_fp16,
    )
    from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
        build_moonshine_mlp,
        moonshine_gated_silu_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_projection_bias,
    )

    rng = np.random.default_rng(0xC1D1)
    hidden = 416
    intermediate = 1664
    x = rng.normal(0.0, 0.05, size=(40, hidden)).astype(np.float16)
    fc1_weight = rng.normal(0.0, 0.04, size=(2 * intermediate, hidden)).astype(np.float16)
    fc1_bias = rng.normal(0.0, 0.03, size=(2 * intermediate,)).astype(np.float16)
    fc2_weight = rng.normal(0.0, 0.04, size=(hidden, intermediate)).astype(np.float16)
    fc2_bias = rng.normal(0.0, 0.03, size=(hidden,)).astype(np.float16)
    expected = moonshine_residual(
        x,
        moonshine_decoder_mlp(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias),
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    projection = build_moonshine_projection(load=True)
    mlp = build_moonshine_mlp(load=True)
    glue = build_moonshine_glue(load=True)
    allocations = []
    try:
        dx = _upload(x, runtime, allocations)
        dfc1_weight = _upload(fc1_weight, runtime, allocations)
        dfc1_bias = _upload(fc1_bias, runtime, allocations)
        dfc2_weight = _upload(fc2_weight, runtime, allocations)
        dfc2_bias = _upload(fc2_bias, runtime, allocations)
        fc1_out = _alloc((40, 2 * intermediate), runtime, allocations)
        activated = _alloc((40, intermediate), runtime, allocations)
        mlp_out = _alloc((40, hidden), runtime, allocations)
        residual_out = _alloc((40, hidden), runtime, allocations)

        moonshine_f16_projection_bias(
            dx.ptr,
            dfc1_weight.ptr,
            dfc1_bias.ptr,
            fc1_out.ptr,
            40,
            hidden,
            2 * intermediate,
            library=projection,
            runtime=runtime,
        )
        moonshine_gated_silu_fp16(
            fc1_out.ptr,
            activated.ptr,
            40,
            intermediate,
            library=mlp,
            runtime=runtime,
        )
        moonshine_f16_projection_bias(
            activated.ptr,
            dfc2_weight.ptr,
            dfc2_bias.ptr,
            mlp_out.ptr,
            40,
            intermediate,
            hidden,
            library=projection,
            runtime=runtime,
        )
        moonshine_residual_fp16(
            dx.ptr,
            mlp_out.ptr,
            residual_out.ptr,
            40 * hidden,
            library=glue,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(residual_out, (40, hidden), runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
    assert np.isfinite(actual).all()


def test_moonshine_cuda_fused_mlp_registry_resolves_explicit_keys() -> None:
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        moonshine_f16_projection_bias_gated_silu,
        moonshine_f16_projection_bias_residual,
        register_moonshine_projection_kernels,
    )
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_mlp_fc1_gated_silu,
        moonshine_projection_bias_residual,
        register_moonshine_cpu_reference_kernels,
    )

    register_moonshine_cpu_reference_kernels()
    register_moonshine_projection_kernels()
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_mlp_fc1",
        quant="fp16",
        variant="bias_gated_silu_fp32_accum",
    ) is moonshine_f16_projection_bias_gated_silu
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_mlp_fc2_residual",
        quant="fp16",
        variant="bias_rounded_residual_fp32_accum",
    ) is moonshine_f16_projection_bias_residual
    assert resolve(
        backend="cpu_reference",
        layer="moonshine_mlp_fc1",
        quant="fp16",
        variant="bias_gated_silu_fp32_accum",
    ) is moonshine_mlp_fc1_gated_silu
    assert resolve(
        backend="cpu_reference",
        layer="moonshine_mlp_fc2_residual",
        quant="fp16",
        variant="bias_rounded_residual_fp32_accum",
    ) is moonshine_projection_bias_residual


def test_moonshine_cuda_fused_mlp_wrappers_keep_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        moonshine_f16_projection_bias_gated_silu,
        moonshine_f16_projection_bias_residual,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_f16_projection_bias_gated_silu = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual = FakeKernel()

    library = FakeLibrary()
    common = {"stream": 7, "library": library, "runtime": object()}
    moonshine_f16_projection_bias_gated_silu(
        1, 2, 3, 4, 1, 416, 1664, **common
    )
    moonshine_f16_projection_bias_residual(
        1, 2, 3, 4, 5, 1, 1664, 416, **common
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_f16_projection_bias_gated_silu.calls
        == [(1, 2, 3, 4, 1, 416, 1664, 32, 7)]
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual.calls
        == [(1, 2, 3, 4, 5, 1, 1664, 416, 256, 7)]
    )


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_fused_mlp_chain_matches_decoder_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
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

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        device_normalized = _upload(normalized, runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_fc1_weight = _upload(fc1_weight, runtime, allocations)
        device_fc1_bias = _upload(fc1_bias, runtime, allocations)
        device_fc2_weight = _upload(fc2_weight, runtime, allocations)
        device_fc2_bias = _upload(fc2_bias, runtime, allocations)
        intermediate_output = _alloc((rows, intermediate), runtime, allocations)
        final_output = _alloc((rows, hidden), runtime, allocations)
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


def _alloc(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(np.float16).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
