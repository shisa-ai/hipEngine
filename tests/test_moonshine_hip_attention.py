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
from hipengine.kernels.cpu_reference.moonshine import moonshine_attention
from hipengine.kernels.registry import resolve

SELF_PAST_LENGTHS = (0, 1, 2, 8, 32, 64, 128, 193)
CROSS_LENGTHS = (40, 207, 1248)
HEADS = 8
HEAD_DIM = 52
SELF_CAPACITY = 194


def _gfx1151_available() -> bool:
    if os.environ.get("HIPENGINE_HIP_ARCH") != "gfx1151":
        return False
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_moonshine_attention_registry_resolves_hip_and_cpu_fallbacks() -> None:
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_cross_attention,
        moonshine_self_attention,
        register_moonshine_cpu_reference_kernels,
    )
    from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_grouped_fp16,
        moonshine_cross_attention_parallel_fp16,
        moonshine_self_attention_branch_fp16,
        moonshine_self_attention_fp16,
        moonshine_self_attention_parallel_fp16,
        register_moonshine_attention_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_moonshine_cpu_reference_kernels()
    register_moonshine_attention_kernels()
    register_gfx1151_kernels(replace=True)
    cases = (
        (
            "moonshine_self_attention",
            "fixed_cache_logical_dim",
            moonshine_self_attention_fp16,
            moonshine_self_attention,
        ),
        (
            "moonshine_self_attention",
            "fixed_cache_branch_online",
            moonshine_self_attention_branch_fp16,
            moonshine_self_attention,
        ),
        (
            "moonshine_self_attention",
            "fixed_cache_parallel_tokens",
            moonshine_self_attention_parallel_fp16,
            moonshine_self_attention,
        ),
        (
            "moonshine_cross_attention",
            "resident_masked_logical_dim",
            moonshine_cross_attention_fp16,
            moonshine_cross_attention,
        ),
        (
            "moonshine_cross_attention",
            "resident_masked_grouped_heads",
            moonshine_cross_attention_grouped_fp16,
            moonshine_cross_attention,
        ),
        (
            "moonshine_cross_attention",
            "resident_masked_parallel_tokens",
            moonshine_cross_attention_parallel_fp16,
            moonshine_cross_attention,
        ),
    )
    for layer, variant, hip_kernel, cpu_kernel in cases:
        assert resolve(
            backend="hip_gfx1151",
            layer=layer,
            quant="fp16",
            variant=variant,
        ) is hip_kernel
        assert resolve(
            backend="cuda_sm120a",
            layer=layer,
            quant="fp16",
            variant=variant,
        ) is cpu_kernel


def test_moonshine_attention_wrappers_keep_raw_pointer_abis() -> None:
    from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_grouped_fp16,
        moonshine_cross_attention_parallel_fp16,
        moonshine_self_attention_branch_fp16,
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
        hipengine_moonshine_self_attention_fp16 = FakeKernel()
        hipengine_moonshine_self_attention_branch_fp16 = FakeKernel()
        hipengine_moonshine_self_attention_parallel_fp16 = FakeKernel()
        hipengine_moonshine_cross_attention_fp16 = FakeKernel()
        hipengine_moonshine_cross_attention_grouped_fp16 = FakeKernel()
        hipengine_moonshine_cross_attention_parallel_fp16 = FakeKernel()

    library = FakeLibrary()
    common = {"threads": 32, "stream": 7, "library": library, "runtime": object()}
    moonshine_self_attention_fp16(1, 2, 3, 4, 5, HEADS, HEAD_DIM, SELF_CAPACITY, **common)
    moonshine_self_attention_branch_fp16(
        1, 2, 3, 4, 5, HEADS, HEAD_DIM, SELF_CAPACITY, **common
    )
    moonshine_self_attention_parallel_fp16(
        1,
        2,
        3,
        4,
        5,
        HEADS,
        HEAD_DIM,
        SELF_CAPACITY,
        threads=128,
        stream=7,
        library=library,
        runtime=object(),
    )
    moonshine_cross_attention_fp16(1, 2, 3, 4, 5, HEADS, HEAD_DIM, 1248, **common)
    moonshine_cross_attention_grouped_fp16(
        1,
        2,
        3,
        4,
        5,
        HEADS,
        HEAD_DIM,
        1248,
        stream=7,
        library=library,
        runtime=object(),
    )
    moonshine_cross_attention_parallel_fp16(
        1,
        2,
        3,
        4,
        5,
        HEADS,
        HEAD_DIM,
        1248,
        threads=128,
        stream=7,
        library=library,
        runtime=object(),
    )
    self_call = library.hipengine_moonshine_self_attention_fp16.calls[0]
    cross_call = library.hipengine_moonshine_cross_attention_fp16.calls[0]
    assert self_call[:8] == (1, 2, 3, 4, 5, HEADS, HEAD_DIM, SELF_CAPACITY)
    assert self_call[8] == pytest.approx(HEAD_DIM**-0.5)
    assert self_call[9:] == (32, 7)
    branch_call = library.hipengine_moonshine_self_attention_branch_fp16.calls[0]
    assert branch_call == self_call
    self_parallel_call = library.hipengine_moonshine_self_attention_parallel_fp16.calls[0]
    assert self_parallel_call[:8] == self_call[:8]
    assert self_parallel_call[8] == pytest.approx(HEAD_DIM**-0.5)
    assert self_parallel_call[9:] == (128, 7)
    assert cross_call[:8] == (1, 2, 3, 4, 5, HEADS, HEAD_DIM, 1248)
    assert cross_call[8] == pytest.approx(HEAD_DIM**-0.5)
    assert cross_call[9:] == (32, 7)
    grouped_call = library.hipengine_moonshine_cross_attention_grouped_fp16.calls[0]
    assert grouped_call[:8] == cross_call[:8]
    assert grouped_call[8] == pytest.approx(HEAD_DIM**-0.5)
    assert grouped_call[9:] == (256, 7)
    parallel_call = library.hipengine_moonshine_cross_attention_parallel_fp16.calls[0]
    assert parallel_call[:8] == cross_call[:8]
    assert parallel_call[8] == pytest.approx(HEAD_DIM**-0.5)
    assert parallel_call[9:] == (128, 7)


def test_moonshine_attention_rejects_non_contract_shapes_before_build() -> None:
    from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_parallel_fp16,
        moonshine_self_attention_fp16,
        moonshine_self_attention_parallel_fp16,
    )

    with pytest.raises(ValueError, match="heads"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, 4, HEAD_DIM, SELF_CAPACITY)
    with pytest.raises(ValueError, match="head_dim"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, HEADS, 56, SELF_CAPACITY)
    with pytest.raises(ValueError, match="capacity"):
        moonshine_self_attention_fp16(1, 2, 3, 4, 5, HEADS, HEAD_DIM, 0)
    with pytest.raises(ValueError, match="encoder_length"):
        moonshine_cross_attention_fp16(1, 2, 3, 4, 5, HEADS, HEAD_DIM, 0)
    with pytest.raises(ValueError, match="scale"):
        moonshine_cross_attention_fp16(
            1, 2, 3, 4, 5, HEADS, HEAD_DIM, 40, scale=float("nan")
        )
    with pytest.raises(ValueError, match="threads"):
        moonshine_cross_attention_parallel_fp16(
            1, 2, 3, 4, 5, HEADS, HEAD_DIM, 40, threads=32
        )
    with pytest.raises(ValueError, match="threads"):
        moonshine_self_attention_parallel_fp16(
            1, 2, 3, 4, 5, HEADS, HEAD_DIM, SELF_CAPACITY, threads=32
        )


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_self_attention_matches_cpu_at_all_past_lengths() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
        build_moonshine_attention,
        moonshine_self_attention_branch_fp16,
        moonshine_self_attention_fp16,
        moonshine_self_attention_parallel_fp16,
    )

    rng = np.random.default_rng(0x5E1F)
    runtime = get_hip_runtime()
    library = build_moonshine_attention(load=True)
    for past_length in SELF_PAST_LENGTHS:
        visible_length = past_length + 1
        query = rng.normal(0.0, 0.08, size=(1, HEADS, 1, HEAD_DIM)).astype(np.float16)
        key_cache = np.zeros((1, HEADS, SELF_CAPACITY, HEAD_DIM), dtype=np.float16)
        value_cache = np.zeros_like(key_cache)
        key_cache[:, :, :visible_length] = rng.normal(
            0.0, 0.08, size=(1, HEADS, visible_length, HEAD_DIM)
        ).astype(np.float16)
        value_cache[:, :, :visible_length] = rng.normal(
            0.0, 0.10, size=(1, HEADS, visible_length, HEAD_DIM)
        ).astype(np.float16)
        expected = moonshine_attention(
            query,
            key_cache[:, :, :visible_length],
            value_cache[:, :, :visible_length],
        )
        allocations = []
        try:
            device_query = _upload(query, runtime, allocations)
            device_key = _upload(key_cache, runtime, allocations)
            device_value = _upload(value_cache, runtime, allocations)
            device_position = _upload(
                np.asarray([past_length], dtype=np.int64), runtime, allocations
            )
            device_output = _empty(expected.shape, np.float16, runtime, allocations)
            candidates = (
                (moonshine_self_attention_fp16, {}),
                (moonshine_self_attention_branch_fp16, {}),
                (moonshine_self_attention_parallel_fp16, {"threads": 64}),
                (moonshine_self_attention_parallel_fp16, {"threads": 128}),
                (moonshine_self_attention_parallel_fp16, {"threads": 256}),
            )
            actuals = []
            for launch, options in candidates:
                launch(
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
                    **options,
                )
                runtime.device_synchronize()
                actuals.append(
                    _download(device_output, expected.shape, np.float16, runtime)
                )
        finally:
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)
        for actual in actuals:
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=5.0e-3, atol=5.0e-3)


@pytest.mark.skipif(not _gfx1151_available(), reason="gfx1151 HIP gate is not enabled")
def test_moonshine_masked_cross_attention_matches_cpu_at_all_buckets() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
        build_moonshine_attention,
        moonshine_cross_attention_fp16,
        moonshine_cross_attention_grouped_fp16,
        moonshine_cross_attention_parallel_fp16,
    )

    rng = np.random.default_rng(0xC2055)
    runtime = get_hip_runtime()
    library = build_moonshine_attention(load=True)
    for encoder_length in CROSS_LENGTHS:
        query = rng.normal(0.0, 0.08, size=(1, HEADS, 1, HEAD_DIM)).astype(np.float16)
        key = rng.normal(
            0.0, 0.08, size=(1, HEADS, encoder_length, HEAD_DIM)
        ).astype(np.float16)
        value = rng.normal(
            0.0, 0.10, size=(1, HEADS, encoder_length, HEAD_DIM)
        ).astype(np.float16)
        mask = np.ones((1, encoder_length), dtype=np.int32)
        mask[:, -(encoder_length // 7) :] = 0
        expected = moonshine_attention(query, key, value, mask=mask)
        allocations = []
        try:
            device_query = _upload(query, runtime, allocations)
            device_key = _upload(key, runtime, allocations)
            device_value = _upload(value, runtime, allocations)
            device_mask = _upload(mask, runtime, allocations)
            device_output = _empty(expected.shape, np.float16, runtime, allocations)
            candidates = (
                (moonshine_cross_attention_fp16, {}),
                (moonshine_cross_attention_grouped_fp16, {}),
                (moonshine_cross_attention_parallel_fp16, {"threads": 64}),
                (moonshine_cross_attention_parallel_fp16, {"threads": 128}),
                (moonshine_cross_attention_parallel_fp16, {"threads": 256}),
            )
            actuals = []
            for launch, options in candidates:
                launch(
                    device_query.ptr,
                    device_key.ptr,
                    device_value.ptr,
                    device_mask.ptr,
                    device_output.ptr,
                    HEADS,
                    HEAD_DIM,
                    encoder_length,
                    library=library,
                    runtime=runtime,
                    **options,
                )
                runtime.device_synchronize()
                actuals.append(
                    _download(device_output, expected.shape, np.float16, runtime)
                )
        finally:
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)
        for actual in actuals:
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=5.0e-3, atol=5.0e-3)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _empty(shape: tuple[int, ...], dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], dtype, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
