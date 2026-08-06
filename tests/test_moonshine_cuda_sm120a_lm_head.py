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
    moonshine_lm_head_argmax,
    moonshine_tied_lm_logits,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve

_HIDDEN = 416
_VOCAB = 36_864
_SCRATCH_ROWS = 16


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


def test_moonshine_cuda_lm_head_registry_resolves_explicit_key() -> None:
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        moonshine_lm_head_argmax_fp16,
        register_moonshine_lm_head_kernels,
    )

    register_moonshine_lm_head_kernels()
    assert (
        resolve(
            backend="cuda_sm120a",
            layer="moonshine_lm_head",
            quant="fp16",
            variant="fused_argmax_fp32_accum",
        )
        is moonshine_lm_head_argmax_fp16
    )


def test_moonshine_cuda_lm_head_build_plan_targets_sm120a(tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        plan_moonshine_lm_head_build,
    )

    artifact = plan_moonshine_lm_head_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc Moonshine test version",
    )
    assert artifact.family == "cuda_sm120a_moonshine_lm_head"
    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)
    assert artifact.output_path.name == "lm_head.so"
    assert not artifact.cache_dir.exists()


def test_moonshine_cuda_lm_head_wrapper_keeps_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        lm_head_argmax_scratch_elements,
        moonshine_lm_head_argmax_fp16,
    )

    assert lm_head_argmax_scratch_elements(36_864, 16) == 2_304
    assert lm_head_argmax_scratch_elements(40, 16) == 3

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16 = FakeKernel()

    library = FakeLibrary()
    moonshine_lm_head_argmax_fp16(
        1,
        2,
        3,
        4,
        5,
        6,
        _HIDDEN,
        _VOCAB,
        rows_per_block=_SCRATCH_ROWS,
        stream=9,
        library=library,
        runtime=object(),
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16.calls
        == [(1, 2, 3, 4, 5, 6, _HIDDEN, _VOCAB, _SCRATCH_ROWS, 9)]
    )


def test_moonshine_cuda_lm_head_rejects_invalid_shapes_before_build() -> None:
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        moonshine_lm_head_argmax_fp16,
    )

    with pytest.raises(ValueError, match="in_features"):
        moonshine_lm_head_argmax_fp16(1, 2, 3, 4, 5, 6, 0, _VOCAB)
    with pytest.raises(ValueError, match="vocab_size"):
        moonshine_lm_head_argmax_fp16(1, 2, 3, 4, 5, 6, _HIDDEN, 0)
    with pytest.raises(ValueError, match="rows_per_block"):
        moonshine_lm_head_argmax_fp16(1, 2, 3, 4, 5, 6, _HIDDEN, _VOCAB, rows_per_block=0)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_lm_head_argmax_matches_cpu_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_scratch_elements,
        moonshine_lm_head_argmax_fp16,
    )

    rng = np.random.default_rng(0xC1F0AC1)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_lm_head(load=True)
    allocations = []
    try:
        for _ in range(3):
            hidden = rng.normal(0.0, 0.05, size=(_HIDDEN,)).astype(np.float16)
            weight = rng.normal(0.0, 0.03, size=(_VOCAB, _HIDDEN)).astype(np.float16)
            expected_logits, expected_index = moonshine_lm_head_argmax(hidden, weight)
            expected_index = int(expected_index)

            d_input = _upload(hidden, runtime, allocations)
            d_weight = _upload(weight, runtime, allocations)
            num_blocks = lm_head_argmax_scratch_elements(_VOCAB, _SCRATCH_ROWS)
            d_values = _alloc_f32(num_blocks, runtime, allocations)
            d_indices = _alloc_i64(num_blocks, runtime, allocations)
            d_out_index = _alloc_i64(1, runtime, allocations)
            d_out_value = _alloc_f32(1, runtime, allocations)

            moonshine_lm_head_argmax_fp16(
                d_input.ptr,
                d_weight.ptr,
                d_values.ptr,
                d_indices.ptr,
                d_out_index.ptr,
                d_out_value.ptr,
                _HIDDEN,
                _VOCAB,
                rows_per_block=_SCRATCH_ROWS,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            actual_index = int(_download_i64(d_out_index, 1, runtime)[0])
            actual_value = float(_download_f32(d_out_value, 1, runtime)[0])
            assert actual_index == expected_index
            np.testing.assert_allclose(
                actual_value,
                float(expected_logits[expected_index]),
                rtol=2e-3,
                atol=2e-3,
            )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_lm_head_argmax_matches_two_step_gpu_path() -> None:
    """The fused kernel is byte-exact with projection -> stable argmax (C1f)."""
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_argmax_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_scratch_elements,
        moonshine_lm_head_argmax_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection,
    )

    rng = np.random.default_rng(0xC1F0C1F)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library_lm = build_moonshine_lm_head(load=True)
    library_proj = build_moonshine_projection(load=True)
    library_glue = build_moonshine_glue(load=True)
    allocations = []
    try:
        for rows_per_block in (8, 16, 32):
            hidden = rng.normal(0.0, 0.05, size=(_HIDDEN,)).astype(np.float16)
            weight = rng.normal(0.0, 0.03, size=(_VOCAB, _HIDDEN)).astype(np.float16)

            d_input = _upload(hidden, runtime, allocations)
            d_weight = _upload(weight, runtime, allocations)
            d_logits = _alloc_f16((_VOCAB,), runtime, allocations)
            d_argmax = _alloc_i64(1, runtime, allocations)
            moonshine_f16_lm_head_projection(
                d_input.ptr,
                d_weight.ptr,
                d_logits.ptr,
                1,
                _HIDDEN,
                _VOCAB,
                library=library_proj,
                runtime=runtime,
            )
            moonshine_argmax_fp16(
                d_logits.ptr,
                d_argmax.ptr,
                _VOCAB,
                library=library_glue,
                runtime=runtime,
            )
            runtime.device_synchronize()
            two_step_index = int(_download_i64(d_argmax, 1, runtime)[0])
            logits = _download_f16(d_logits, (_VOCAB,), runtime)
            two_step_value = float(logits[two_step_index])

            num_blocks = lm_head_argmax_scratch_elements(_VOCAB, rows_per_block)
            d_values = _alloc_f32(num_blocks, runtime, allocations)
            d_indices = _alloc_i64(num_blocks, runtime, allocations)
            d_out_index = _alloc_i64(1, runtime, allocations)
            d_out_value = _alloc_f32(1, runtime, allocations)
            moonshine_lm_head_argmax_fp16(
                d_input.ptr,
                d_weight.ptr,
                d_values.ptr,
                d_indices.ptr,
                d_out_index.ptr,
                d_out_value.ptr,
                _HIDDEN,
                _VOCAB,
                rows_per_block=rows_per_block,
                library=library_lm,
                runtime=runtime,
            )
            runtime.device_synchronize()
            fused_index = int(_download_i64(d_out_index, 1, runtime)[0])
            fused_value = float(_download_f32(d_out_value, 1, runtime)[0])

            assert fused_index == two_step_index, (
                f"rows_per_block={rows_per_block}: fused {fused_index} != two-step {two_step_index}"
            )
            np.testing.assert_allclose(fused_value, two_step_value, rtol=1e-6, atol=1e-6)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_lm_head_argmax_deterministic_ties() -> None:
    """Lowest index wins on FP16-visible ties; result is stable across launches."""
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_scratch_elements,
        moonshine_lm_head_argmax_fp16,
    )

    rng = np.random.default_rng(0xC1F0C1F)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_lm_head(load=True)
    allocations = []

    # All rows share an identical weight row: every logit is FP16-identical, so
    # the stable argmax must select the lowest index (0).
    hidden = rng.normal(0.0, 0.05, size=(_HIDDEN,)).astype(np.float16)
    row = rng.normal(0.0, 0.03, size=(_HIDDEN,)).astype(np.float16)
    weight = np.broadcast_to(row, (_VOCAB, _HIDDEN)).copy().astype(np.float16)
    expected_logits, expected_index = moonshine_lm_head_argmax(hidden, weight)
    assert expected_index == 0

    try:
        d_input = _upload(hidden, runtime, allocations)
        d_weight = _upload(weight, runtime, allocations)
        num_blocks = lm_head_argmax_scratch_elements(_VOCAB, _SCRATCH_ROWS)
        d_values = _alloc_f32(num_blocks, runtime, allocations)
        d_indices = _alloc_i64(num_blocks, runtime, allocations)
        d_out_index = _alloc_i64(1, runtime, allocations)
        d_out_value = _alloc_f32(1, runtime, allocations)

        results = set()
        for _ in range(10):
            moonshine_lm_head_argmax_fp16(
                d_input.ptr,
                d_weight.ptr,
                d_values.ptr,
                d_indices.ptr,
                d_out_index.ptr,
                d_out_value.ptr,
                _HIDDEN,
                _VOCAB,
                rows_per_block=_SCRATCH_ROWS,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            results.add(int(_download_i64(d_out_index, 1, runtime)[0]))
        assert results == {0}, f"tied argmax not stable/lowest: {results}"

        # Second case: rows 5 and 9 are identical and strictly maximal; the
        # stable argmax must pick 5 (the lowest of the tied maxes).
        weight2 = rng.normal(0.0, 0.03, size=(_VOCAB, _HIDDEN)).astype(np.float16)
        dominant = rng.normal(2.0, 0.01, size=(_HIDDEN,)).astype(np.float16)
        weight2[5] = dominant
        weight2[9] = dominant
        _, expected_index2 = moonshine_lm_head_argmax(hidden, weight2)
        assert expected_index2 == 5

        d_weight2 = _upload(weight2, runtime, allocations)
        results2 = set()
        for _ in range(10):
            moonshine_lm_head_argmax_fp16(
                d_input.ptr,
                d_weight2.ptr,
                d_values.ptr,
                d_indices.ptr,
                d_out_index.ptr,
                d_out_value.ptr,
                _HIDDEN,
                _VOCAB,
                rows_per_block=_SCRATCH_ROWS,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            results2.add(int(_download_i64(d_out_index, 1, runtime)[0]))
        assert results2 == {5}, f"tie did not resolve to lowest index: {results2}"
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc_f16(shape, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(np.float16).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _alloc_f32(count: int, runtime, allocations):
    device = malloc(count * np.dtype(np.float32).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _alloc_i64(count: int, runtime, allocations):
    device = malloc(count * np.dtype(np.int64).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download_f16(device, shape, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _download_f32(device, count, runtime) -> np.ndarray:
    host = np.empty(count, dtype=np.float32)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _download_i64(device, count, runtime) -> np.ndarray:
    host = np.empty(count, dtype=np.int64)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
