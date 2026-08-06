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
    moonshine_cross_attention,
    moonshine_self_attention,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve

SELF_PAST_LENGTHS = (0, 1, 2, 8, 32, 64, 128, 193)
CROSS_LENGTHS = (40, 207, 1248)
HEADS = 8
HEAD_DIM = 52
SELF_CAPACITY = 194


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


def test_moonshine_cuda_attention_registry_resolves_explicit_keys() -> None:
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        register_moonshine_attention_kernels,
    )
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_grouped_fp16,
        moonshine_cross_attention_parallel_fp16,
        moonshine_self_attention_fp16,
        moonshine_self_attention_parallel_fp16,
    )

    register_moonshine_attention_kernels()
    cases = (
        (
            "moonshine_self_attention",
            "fixed_cache_logical_dim",
            moonshine_self_attention_fp16,
        ),
        (
            "moonshine_self_attention",
            "fixed_cache_parallel_tokens",
            moonshine_self_attention_parallel_fp16,
        ),
        (
            "moonshine_cross_attention",
            "resident_masked_logical_dim",
            moonshine_cross_attention_fp16,
        ),
        (
            "moonshine_cross_attention",
            "resident_masked_grouped_heads",
            moonshine_cross_attention_grouped_fp16,
        ),
        (
            "moonshine_cross_attention",
            "resident_masked_parallel_tokens",
            moonshine_cross_attention_parallel_fp16,
        ),
    )
    for layer, variant, kernel in cases:
        assert (
            resolve(backend="cuda_sm120a", layer=layer, quant="fp16", variant=variant)
            is kernel
        )


def test_moonshine_cuda_attention_build_plan_targets_sm120a(tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        plan_moonshine_attention_build,
    )

    artifact = plan_moonshine_attention_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc Moonshine test version",
    )
    assert artifact.family == "cuda_sm120a_moonshine_attention"
    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)
    assert artifact.output_path.name == "moonshine_attention.so"
    assert not artifact.cache_dir.exists()


def test_moonshine_cuda_attention_wrappers_keep_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_grouped_fp16,
        moonshine_cross_attention_parallel_fp16,
        moonshine_self_attention_fp16,
        moonshine_self_attention_parallel_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_self_attention_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_cross_attention_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_cross_attention_grouped_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_cross_attention_parallel_fp16 = FakeKernel()

    library = FakeLibrary()
    common = {"stream": 7, "library": library, "runtime": object()}
    scale = 52.0**-0.5
    moonshine_self_attention_fp16(1, 2, 3, 4, 5, 8, 52, 194, **common)
    moonshine_self_attention_parallel_fp16(1, 2, 3, 4, 5, 8, 52, 194, threads=128, **common)
    moonshine_cross_attention_fp16(1, 2, 3, 4, 5, 8, 52, 40, **common)
    moonshine_cross_attention_grouped_fp16(1, 2, 3, 4, 5, 8, 52, 40, **common)
    moonshine_cross_attention_parallel_fp16(1, 2, 3, 4, 5, 8, 52, 40, threads=128, **common)
    assert (
        library.hipengine_cuda_sm120a_moonshine_self_attention_fp16.calls
        == [(1, 2, 3, 4, 5, 8, 52, 194, scale, 32, 7)]
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16.calls
        == [(1, 2, 3, 4, 5, 8, 52, 194, scale, 128, 7)]
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_cross_attention_fp16.calls
        == [(1, 2, 3, 4, 5, 8, 52, 40, scale, 32, 7)]
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_cross_attention_grouped_fp16.calls
        == [(1, 2, 3, 4, 5, 8, 52, 40, scale, 256, 7)]
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_cross_attention_parallel_fp16.calls
        == [(1, 2, 3, 4, 5, 8, 52, 40, scale, 128, 7)]
    )


def test_moonshine_cuda_attention_rejects_invalid_shapes_before_build() -> None:
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        moonshine_cross_attention_grouped_fp16,
        moonshine_self_attention_fp16,
    )

    with pytest.raises(ValueError, match="heads"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, 4, 52, 194)
    with pytest.raises(ValueError, match="head_dim"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, 8, 64, 194)
    with pytest.raises(ValueError, match="threads"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, 8, 52, 194, threads=64)
    with pytest.raises(ValueError, match="threads"):
        moonshine_cross_attention_grouped_fp16(1, 2, 3, 4, 5, 8, 52, 40, threads=128)
    with pytest.raises(ValueError, match="length"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, 8, 52, 0)


def test_moonshine_cuda_attention_schedule_helpers_reflect_measured_buckets() -> None:
    """General cache-position buckets from the exclusive-GPU0 batch screen.

    Cross attention is flat parallel t256 at every production encoder length;
    self attention is one-wave t32 below 8 visible tokens and parallel t256
    from 8 upward (tied at the boundary, 1.6x+ faster from ~16).
    """
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        _default_cross_threads,
        _default_self_threads,
    )

    for length in (40, 207, 1248):
        assert _default_cross_threads(length) == 256
    for visible in (1, 2, 7):
        assert _default_self_threads(visible) == 32
    for visible in (8, 33, 65, 129, 194):
        assert _default_self_threads(visible) == 256


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_self_attention_matches_cpu_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        build_moonshine_attention,
        moonshine_self_attention_fp16,
        moonshine_self_attention_parallel_fp16,
    )

    rng = np.random.default_rng(0xC1E5E1F)
    query = rng.normal(0.0, 0.4, size=(1, HEADS, 1, HEAD_DIM)).astype(np.float16)
    key_cache = rng.normal(0.0, 0.4, size=(1, HEADS, SELF_CAPACITY, HEAD_DIM)).astype(np.float16)
    value_cache = rng.normal(0.0, 0.4, size=(1, HEADS, SELF_CAPACITY, HEAD_DIM)).astype(np.float16)

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_attention(load=True)
    allocations = []
    try:
        device_query = _upload(query.reshape(1, HEADS * HEAD_DIM), runtime, allocations)
        device_key = _upload(key_cache.reshape(1, HEADS * SELF_CAPACITY * HEAD_DIM), runtime, allocations)
        device_value = _upload(value_cache.reshape(1, HEADS * SELF_CAPACITY * HEAD_DIM), runtime, allocations)
        device_output = _alloc((1, HEADS * HEAD_DIM), runtime, allocations)
        device_position = _upload(np.array([0], dtype=np.int64), runtime, allocations)

        for pos in SELF_PAST_LENGTHS:
            expected = moonshine_self_attention(
                query, key_cache, value_cache, position=pos
            )  # (1, 8, 1, 52)
            copy_host_to_device(
                device_position,
                host_array_ptr(np.array([pos], dtype=np.int64)),
                runtime=runtime,
            )

            moonshine_self_attention_fp16(
                device_query.ptr,
                device_key.ptr,
                device_value.ptr,
                device_position.ptr,
                device_output.ptr,
                HEADS,
                HEAD_DIM,
                SELF_CAPACITY,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            actual = _download(device_output, (1, HEADS * HEAD_DIM), runtime).reshape(1, HEADS, 1, HEAD_DIM)
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
            assert np.isfinite(actual).all()

            # parallel variant agrees with the one-wave result across thread counts.
            for threads in (64, 128, 256):
                moonshine_self_attention_parallel_fp16(
                    device_query.ptr,
                    device_key.ptr,
                    device_value.ptr,
                    device_position.ptr,
                    device_output.ptr,
                    HEADS,
                    HEAD_DIM,
                    SELF_CAPACITY,
                    threads=threads,
                    library=library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                actual_parallel = _download(device_output, (1, HEADS * HEAD_DIM), runtime).reshape(1, HEADS, 1, HEAD_DIM)
                np.testing.assert_allclose(actual_parallel, actual, rtol=2e-3, atol=2e-3)
                np.testing.assert_allclose(actual_parallel, expected, rtol=2e-3, atol=2e-3)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_cross_attention_matches_cpu_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        build_moonshine_attention,
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_grouped_fp16,
        moonshine_cross_attention_parallel_fp16,
    )

    rng = np.random.default_rng(0xC1E5C2F)
    query = rng.normal(0.0, 0.4, size=(1, HEADS, 1, HEAD_DIM)).astype(np.float16)

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_attention(load=True)
    allocations = []
    try:
        device_query = _upload(query.reshape(1, HEADS * HEAD_DIM), runtime, allocations)
        for length in CROSS_LENGTHS:
            key = rng.normal(0.0, 0.4, size=(1, HEADS, length, HEAD_DIM)).astype(np.float16)
            value = rng.normal(0.0, 0.4, size=(1, HEADS, length, HEAD_DIM)).astype(np.float16)
            # Mask: every third frame invisible (and force some pad at the tail).
            mask = np.ones((length,), dtype=np.int32)
            mask[2::3] = 0
            mask[-1] = 0

            expected = moonshine_cross_attention(
                query, key, value, mask=mask
            )  # (1, 8, 1, 52)

            device_key = _upload(key.reshape(1, HEADS * length * HEAD_DIM), runtime, allocations)
            device_value = _upload(value.reshape(1, HEADS * length * HEAD_DIM), runtime, allocations)
            device_mask = _upload(mask, runtime, allocations)
            device_output = _alloc((1, HEADS * HEAD_DIM), runtime, allocations)

            for launch in (
                lambda: moonshine_cross_attention_fp16(
                    device_query.ptr, device_key.ptr, device_value.ptr,
                    device_mask.ptr, device_output.ptr, HEADS, HEAD_DIM, length,
                    library=library, runtime=runtime,
                ),
                lambda: moonshine_cross_attention_grouped_fp16(
                    device_query.ptr, device_key.ptr, device_value.ptr,
                    device_mask.ptr, device_output.ptr, HEADS, HEAD_DIM, length,
                    library=library, runtime=runtime,
                ),
                lambda: moonshine_cross_attention_parallel_fp16(
                    device_query.ptr, device_key.ptr, device_value.ptr,
                    device_mask.ptr, device_output.ptr, HEADS, HEAD_DIM, length,
                    threads=64, library=library, runtime=runtime,
                ),
                lambda: moonshine_cross_attention_parallel_fp16(
                    device_query.ptr, device_key.ptr, device_value.ptr,
                    device_mask.ptr, device_output.ptr, HEADS, HEAD_DIM, length,
                    threads=128, library=library, runtime=runtime,
                ),
                lambda: moonshine_cross_attention_parallel_fp16(
                    device_query.ptr, device_key.ptr, device_value.ptr,
                    device_mask.ptr, device_output.ptr, HEADS, HEAD_DIM, length,
                    threads=256, library=library, runtime=runtime,
                ),
            ):
                launch()
                runtime.device_synchronize()
                actual = _download(device_output, (1, HEADS * HEAD_DIM), runtime).reshape(1, HEADS, 1, HEAD_DIM)
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
                assert np.isfinite(actual).all()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


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
