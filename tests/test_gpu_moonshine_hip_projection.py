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
        moonshine_f16_lm_head_projection,
        moonshine_f16_lm_head_projection_wave8,
        moonshine_f16_lm_head_projection_wave8_top1,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
        moonshine_f16_projection_triple,
        register_moonshine_projection_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_moonshine_projection_kernels()
    register_gfx1151_kernels(replace=True)
    expected = {
        ("moonshine_projection", "single_fp32_accum"): moonshine_f16_projection,
        ("moonshine_lm_head", "tied_fp32_accum"): moonshine_f16_lm_head_projection,
        ("moonshine_lm_head", "tied_wave8_fp32_accum"): moonshine_f16_lm_head_projection_wave8,
        (
            "moonshine_lm_head",
            "tied_wave8_top1_logits_fp32_accum",
        ): moonshine_f16_lm_head_projection_wave8_top1,
        ("moonshine_projection_rows", "single_fp32_accum"): moonshine_f16_projection,
        ("moonshine_projection_bias", "single_fp32_accum"): moonshine_f16_projection_bias,
        ("moonshine_projection_pair", "pair_fp32_accum"): moonshine_f16_projection_pair,
        (
            "moonshine_cross_kv_precompute",
            "pair_head_major_fp32_accum",
        ): moonshine_f16_projection_pair_head_major,
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
        moonshine_f16_lm_head_projection,
        moonshine_f16_lm_head_projection_wave8,
        moonshine_f16_lm_head_projection_wave8_top1,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
        moonshine_f16_projection_triple,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_f16_lm_head_projection = FakeKernel()
        hipengine_moonshine_f16_lm_head_projection_wave8 = FakeKernel()
        hipengine_moonshine_f16_lm_head_projection_wave8_top1 = FakeKernel()
        hipengine_moonshine_f16_projection = FakeKernel()
        hipengine_moonshine_f16_projection_bias = FakeKernel()
        hipengine_moonshine_f16_projection_pair = FakeKernel()
        hipengine_moonshine_f16_projection_pair_head_major = FakeKernel()
        hipengine_moonshine_f16_projection_triple = FakeKernel()

    library = FakeLibrary()
    common = {"threads": 256, "stream": 7, "library": library, "runtime": object()}
    moonshine_f16_projection(1, 2, 3, 1, 416, 416, **common)
    moonshine_f16_lm_head_projection(1, 2, 3, 1, 416, 36_864, **common)
    moonshine_f16_lm_head_projection_wave8(
        1, 2, 3, 1, 416, 36_864, stream=7, library=library, runtime=object()
    )
    moonshine_f16_lm_head_projection_wave8_top1(
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        416,
        36_864,
        stream=7,
        library=library,
        runtime=object(),
    )
    moonshine_f16_projection_bias(1, 2, 3, 4, 1, 416, 416, **common)
    moonshine_f16_projection_pair(1, 2, 3, 4, 5, 1, 416, 416, 416, **common)
    moonshine_f16_projection_pair_head_major(
        1, 2, 3, 4, 5, 40, 416, 416, 416, 52, **common
    )
    moonshine_f16_projection_triple(
        1, 2, 3, 4, 5, 6, 7, 1, 416, 416, 416, 416, **common
    )
    assert library.hipengine_moonshine_f16_projection.calls == [
        (1, 2, 3, 1, 416, 416, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_lm_head_projection.calls == [
        (1, 2, 3, 1, 416, 36_864, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_lm_head_projection_wave8.calls == [
        (1, 2, 3, 1, 416, 36_864, 7)
    ]
    assert library.hipengine_moonshine_f16_lm_head_projection_wave8_top1.calls == [
        (1, 2, 3, 4, 5, 6, 1, 416, 36_864, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_bias.calls == [
        (1, 2, 3, 4, 1, 416, 416, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_pair.calls == [
        (1, 2, 3, 4, 5, 1, 416, 416, 416, 256, 7)
    ]
    assert library.hipengine_moonshine_f16_projection_pair_head_major.calls == [
        (1, 2, 3, 4, 5, 40, 416, 416, 416, 52, 256, 7)
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
        moonshine_f16_lm_head_projection,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
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
        lm_head = _alloc((1, hidden), runtime, allocations)
        biased = _alloc((1, hidden), runtime, allocations)
        triple = tuple(_alloc((1, hidden), runtime, allocations) for _ in range(3))
        pair = tuple(_alloc((40, hidden), runtime, allocations) for _ in range(2))
        head_major = tuple(_alloc((8, 40, 52), runtime, allocations) for _ in range(2))

        moonshine_f16_projection(
            dx_one.ptr, device_weights[0].ptr, single.ptr, 1, hidden, hidden,
            library=library, runtime=runtime,
        )
        moonshine_f16_lm_head_projection(
            dx_one.ptr, device_weights[0].ptr, lm_head.ptr, 1, hidden, hidden,
            library=library, runtime=runtime,
        )
        moonshine_f16_projection_bias(
            dx_one.ptr, device_weights[0].ptr, device_bias.ptr, biased.ptr,
            1, hidden, hidden, library=library, runtime=runtime,
        )
        moonshine_f16_projection_pair_head_major(
            dx_rows.ptr,
            device_weights[0].ptr,
            device_weights[1].ptr,
            head_major[0].ptr,
            head_major[1].ptr,
            40,
            hidden,
            hidden,
            hidden,
            52,
            library=library,
            runtime=runtime,
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
        actual_lm_head = _download(lm_head, (1, hidden), runtime)
        actual_bias = _download(biased, (1, hidden), runtime)
        actual_triple = tuple(_download(output, (1, hidden), runtime) for output in triple)
        actual_pair = tuple(_download(output, (40, hidden), runtime) for output in pair)
        actual_head_major = tuple(
            _download(output, (8, 40, 52), runtime) for output in head_major
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_single, expected_one[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(actual_lm_head, actual_single)
    np.testing.assert_allclose(actual_bias, expected_bias, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_triple, expected_one, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_pair, expected_rows, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_head_major, expected_rows, strict=True):
        expected_layout = expected.reshape(40, 8, 52).transpose(1, 0, 2)
        np.testing.assert_allclose(actual, expected_layout, rtol=2e-3, atol=2e-3)
    assert all(
        np.isfinite(value).all()
        for value in (*actual_triple, *actual_pair, *actual_head_major)
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("vocab_size", [13, 36_864])
def test_moonshine_wave8_top1_matches_unfused_full_logits_and_stable_token(
    vocab_size: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_argmax_fp16,
    )
    from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection_wave8,
        moonshine_f16_lm_head_projection_wave8_top1,
        moonshine_lm_head_partial_count,
    )

    hidden = 416
    if vocab_size == 13:
        inputs = np.zeros((1, hidden), dtype=np.float16)
        weights = np.zeros((vocab_size, hidden), dtype=np.float16)
    else:
        rng = np.random.default_rng(0x1151)
        inputs = rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float16)
        weights = rng.normal(0.0, 0.04, size=(vocab_size, hidden)).astype(
            np.float16
        )
    expected_logits = moonshine_projection(inputs, weights)
    expected_token = int(np.argmax(expected_logits[0]))
    partial_count = moonshine_lm_head_partial_count(vocab_size)

    runtime = get_hip_runtime()
    projection_library = build_moonshine_projection(load=True)
    glue_library = build_moonshine_glue(load=True)
    allocations = []
    try:
        device_input = _upload(inputs, runtime, allocations)
        device_weights = _upload(weights, runtime, allocations)
        baseline_logits = _alloc_dtype(
            (1, vocab_size), np.float16, runtime, allocations
        )
        candidate_logits = _alloc_dtype(
            (1, vocab_size), np.float16, runtime, allocations
        )
        baseline_token = _alloc_dtype((1,), np.int64, runtime, allocations)
        candidate_token = _alloc_dtype((1,), np.int64, runtime, allocations)
        partial_values = _alloc_dtype(
            (partial_count,), np.float16, runtime, allocations
        )
        partial_indices = _alloc_dtype(
            (partial_count,), np.int64, runtime, allocations
        )

        moonshine_f16_lm_head_projection_wave8(
            device_input.ptr,
            device_weights.ptr,
            baseline_logits.ptr,
            1,
            hidden,
            vocab_size,
            library=projection_library,
            runtime=runtime,
        )
        moonshine_argmax_fp16(
            baseline_logits.ptr,
            baseline_token.ptr,
            vocab_size,
            library=glue_library,
            runtime=runtime,
        )
        moonshine_f16_lm_head_projection_wave8_top1(
            device_input.ptr,
            device_weights.ptr,
            candidate_logits.ptr,
            partial_values.ptr,
            partial_indices.ptr,
            candidate_token.ptr,
            1,
            hidden,
            vocab_size,
            library=projection_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        baseline_logits_host = _download(
            baseline_logits, (1, vocab_size), runtime
        )
        candidate_logits_host = _download(
            candidate_logits, (1, vocab_size), runtime
        )
        baseline_token_host = np.empty((1,), dtype=np.int64)
        candidate_token_host = np.empty((1,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(baseline_token_host), baseline_token, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(candidate_token_host), candidate_token, runtime=runtime
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(candidate_logits_host, baseline_logits_host)
    np.testing.assert_allclose(
        candidate_logits_host, expected_logits, rtol=2e-3, atol=2e-3
    )
    assert int(candidate_token_host[0]) == int(baseline_token_host[0])
    assert int(candidate_token_host[0]) == expected_token


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape: tuple[int, ...], runtime, allocations):
    return _alloc_dtype(shape, np.float16, runtime, allocations)


def _alloc_dtype(shape: tuple[int, ...], dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
