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
    moonshine_layernorm,
    moonshine_residual_layernorm,
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


def test_moonshine_cuda_layernorm_registry_resolves_explicit_keys() -> None:
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
        moonshine_residual_layernorm_fp16,
        register_moonshine_layernorm_kernels,
    )

    register_moonshine_layernorm_kernels()
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_layernorm",
        quant="fp16",
        variant="fp32_stats",
    ) is moonshine_layernorm_fp16
    assert resolve(
        backend="cuda_sm120a",
        layer="moonshine_residual+moonshine_layernorm",
        quant="fp16",
        variant="rounded_fp32_stats",
    ) is moonshine_residual_layernorm_fp16


def test_moonshine_cuda_layernorm_build_plan_targets_architecture_qualified_sm120a(
    tmp_path,
) -> None:
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        plan_moonshine_layernorm_build,
    )

    artifact = plan_moonshine_layernorm_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc Moonshine test version",
    )

    assert artifact.family == "cuda_sm120a_moonshine_layernorm"
    assert artifact.profile.name == "decode"
    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)
    assert artifact.output_path.name == "moonshine_layernorm.so"
    assert not artifact.cache_dir.exists()


def test_moonshine_cuda_layernorm_wrapper_keeps_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
        moonshine_residual_layernorm_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_layernorm_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16 = FakeKernel()

    library = FakeLibrary()
    common = {"threads": 256, "stream": 9, "library": library, "runtime": object()}
    moonshine_layernorm_fp16(1, 2, 3, 7, 416, **common)
    moonshine_residual_layernorm_fp16(1, 2, 3, 4, 5, 7, 416, **common)
    assert library.hipengine_cuda_sm120a_moonshine_layernorm_fp16.calls == [
        (1, 2, 3, 7, 416, pytest.approx(1.0e-5), 256, 9)
    ]
    assert (
        library.hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16.calls
        == [(1, 2, 3, 4, 5, 7, 416, pytest.approx(1.0e-5), 256, 9)]
    )


def test_moonshine_cuda_layernorm_auto_selects_measured_per_bucket_threads() -> None:
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
        moonshine_residual_layernorm_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_layernorm_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16 = FakeKernel()

    library = FakeLibrary()
    # Decoder/short buckets keep 256 threads; the large encoder bucket selects 128.
    for rows in (1, 7, 40, 207, 767):
        moonshine_layernorm_fp16(
            1, 2, 3, rows, 416, library=library, runtime=object()
        )
        moonshine_residual_layernorm_fp16(
            1, 2, 3, 4, 5, rows, 416, library=library, runtime=object()
        )
    for rows in (768, 1_248):
        moonshine_layernorm_fp16(
            1, 2, 3, rows, 416, library=library, runtime=object()
        )
        moonshine_residual_layernorm_fp16(
            1, 2, 3, 4, 5, rows, 416, library=library, runtime=object()
        )
    assert all(
        call[6] == 256
        for call in library.hipengine_cuda_sm120a_moonshine_layernorm_fp16.calls[:5]
    )
    assert all(
        call[8] == 256
        for call in library.hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16.calls[
            :5
        ]
    )
    assert library.hipengine_cuda_sm120a_moonshine_layernorm_fp16.calls[5][6] == 128
    assert library.hipengine_cuda_sm120a_moonshine_layernorm_fp16.calls[6][6] == 128
    assert (
        library.hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16.calls[5][8]
        == 128
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16.calls[6][8]
        == 128
    )


def test_moonshine_cuda_layernorm_rejects_invalid_contract_before_build() -> None:
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        moonshine_layernorm_fp16,
    )

    with pytest.raises(ValueError, match="rows"):
        moonshine_layernorm_fp16(1, 2, 3, 0, 416)
    with pytest.raises(ValueError, match="eps"):
        moonshine_layernorm_fp16(1, 2, 3, 1, 416, eps=0.0)
    with pytest.raises(ValueError, match="threads"):
        moonshine_layernorm_fp16(1, 2, 3, 1, 416, threads=48)


@pytest.mark.parametrize("rows", [1, 7, 40, 207, 1_248])
@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_layernorm_hidden416_matches_fp32_stats_oracle(
    rows: int,
) -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_layernorm_fp16,
    )

    rng = np.random.default_rng(0x1A92 + rows)
    hidden = 416
    inputs = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    weights = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    expected = moonshine_layernorm(inputs, weights)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
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


@pytest.mark.parametrize("rows", [1, 7, 40, 207, 1_248])
@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_residual_layernorm_matches_unfused_boundaries(
    rows: int,
) -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_residual_layernorm_fp16,
    )

    rng = np.random.default_rng(0xADD10 + rows)
    hidden = 416
    residual = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    update = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float16)
    weight = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    expected_residual, expected_norm = moonshine_residual_layernorm(
        residual, update, weight
    )
    runtime = get_cuda_runtime()
    runtime.set_device(0)
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


@pytest.mark.parametrize("threads", [32, 64, 128, 256])
@pytest.mark.parametrize("hidden_size", [52, 416])
@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_layernorm_thread_sweep_matches_oracle(
    threads: int, hidden_size: int, rows: int
) -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_layernorm_fp16,
    )

    rng = np.random.default_rng(0x31B5 + threads + hidden_size + rows)
    inputs = rng.normal(0.0, 0.6, size=(rows, hidden_size)).astype(np.float16)
    weights = rng.normal(1.0, 0.08, size=(hidden_size,)).astype(np.float16)
    expected = moonshine_layernorm(inputs, weights)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
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
            hidden_size,
            threads=threads,
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


@pytest.mark.parametrize("threads", [32, 128, 256])
@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_residual_layernorm_thread_sweep_boundary_byte_exact(
    threads: int, rows: int
) -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_residual_layernorm_fp16,
    )

    rng = np.random.default_rng(0x31C2 + threads + rows)
    hidden = 416
    residual = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    update = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float16)
    weight = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    expected_residual, expected_norm = moonshine_residual_layernorm(
        residual, update, weight
    )
    runtime = get_cuda_runtime()
    runtime.set_device(0)
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
            threads=threads,
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


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_layernorm_poisoned_output_fully_written_and_epsilon() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_layernorm_fp16,
    )

    rng = np.random.default_rng(0x50A5)
    rows, hidden = 3, 416
    inputs = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    weights = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    eps = 1.0e-4
    expected = moonshine_layernorm(inputs, weights, eps=eps)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_layernorm(load=True)
    allocations = []
    try:
        device_input = _upload(inputs, runtime, allocations)
        device_weight = _upload(weights, runtime, allocations)
        device_output = malloc(expected.nbytes, runtime=runtime)
        allocations.append(device_output)
        poison = np.full((rows, hidden), np.nan, dtype=np.float16)
        copy_host_to_device(
            device_output, host_array_ptr(poison), runtime=runtime
        )
        moonshine_layernorm_fp16(
            device_input.ptr,
            device_weight.ptr,
            device_output.ptr,
            rows,
            hidden,
            eps=eps,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(actual), device_output, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    # Every output element must be overwritten (no poison NaN remains) and the
    # explicit epsilon must round-trip through the wrapper into the oracle.
    assert not np.isnan(actual).any()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_layernorm_constant_and_extreme_rows_finite() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_layernorm_fp16,
    )

    hidden = 416
    constant = np.full((2, hidden), 3.5, dtype=np.float16)
    weights = np.random.default_rng(0xE47E).normal(
        1.0, 0.08, size=(hidden,)
    ).astype(np.float16)
    extreme = np.array(
        [[65504.0, -65504.0, 1.0, -1.0] + [0.0] * (hidden - 4)], dtype=np.float16
    ).repeat(2, axis=0)
    expected_constant = moonshine_layernorm(constant, weights)
    expected_extreme = moonshine_layernorm(extreme, weights)

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_layernorm(load=True)
    allocations = []
    try:
        device_weight = _upload(weights, runtime, allocations)
        constant_out = malloc(constant.nbytes, runtime=runtime)
        extreme_out = malloc(extreme.nbytes, runtime=runtime)
        allocations.extend((constant_out, extreme_out))
        for inputs, output, rows in (
            (constant, constant_out, 2),
            (extreme, extreme_out, 2),
        ):
            device_input = _upload(inputs, runtime, allocations)
            moonshine_layernorm_fp16(
                device_input.ptr,
                device_weight.ptr,
                output.ptr,
                rows,
                hidden,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        actual_constant = np.empty_like(expected_constant)
        actual_extreme = np.empty_like(expected_extreme)
        copy_device_to_host(
            host_array_ptr(actual_constant), constant_out, runtime=runtime
        )
        copy_device_to_host(host_array_ptr(actual_extreme), extreme_out, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    # Constant rows have zero centered variance: the normalized output is exactly
    # zero.  Extreme finite rows must stay finite and match the oracle.
    np.testing.assert_array_equal(actual_constant, expected_constant)
    np.testing.assert_array_equal(actual_extreme, expected_extreme)
    assert np.isfinite(actual_constant).all()
    assert np.isfinite(actual_extreme).all()


def _fixture_inputs_available() -> bool:
    fixture_dir = os.environ.get(
        "HIPENGINE_MOONSHINE_FIXTURE_DIR",
        "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures",
    )
    checkpoint = os.environ.get(
        "HIPENGINE_MOONSHINE_CHECKPOINT",
        "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d/model.safetensors",
    )
    return all(
        os.path.isfile(os.path.join(fixture_dir, f"{name}.npz"))
        for name in ("audio-konichiwa-fp16", "synthetic-1s-seed1234-fp16")
    ) and os.path.isfile(checkpoint)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixture_inputs_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_moonshine_cuda_layernorm_byte_exact_on_model_derived_fixtures() -> None:
    from safetensors import safe_open

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
        build_moonshine_layernorm,
        moonshine_layernorm_fp16,
    )

    fixture_dir = os.environ.get(
        "HIPENGINE_MOONSHINE_FIXTURE_DIR",
        "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures",
    )
    checkpoint = os.environ.get(
        "HIPENGINE_MOONSHINE_CHECKPOINT",
        "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d/model.safetensors",
    )
    positions = (0, 1, 8, 32, 64, 128, 193)
    with safe_open(checkpoint, framework="np") as store:
        final_norm_weight = (
            store.get_tensor("model.decoder.norm.weight").astype(np.float16)
        )
    hidden = int(final_norm_weight.shape[0])

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_layernorm(load=True)
    allocations = []
    try:
        device_weight = _upload(final_norm_weight, runtime, allocations)
        device_output = malloc(1 * hidden * 2, runtime=runtime)
        allocations.append(device_output)
        for name in ("audio-konichiwa-fp16", "synthetic-1s-seed1234-fp16"):
            with np.load(os.path.join(fixture_dir, f"{name}.npz")) as fixture:
                for pos in positions:
                    boundary = fixture[f"decoder.position_{pos}.layer_7.after_mlp"]
                    reference = fixture[f"decoder.position_{pos}.final_hidden"]
                    device_input = _upload(boundary, runtime, allocations)
                    moonshine_layernorm_fp16(
                        device_input.ptr,
                        device_weight.ptr,
                        device_output.ptr,
                        1,
                        hidden,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    actual = np.empty((1, 1, hidden), dtype=np.float16)
                    copy_device_to_host(
                        host_array_ptr(actual), device_output, runtime=runtime
                    )
                    np.testing.assert_array_equal(actual, reference)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device
