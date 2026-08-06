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
    moonshine_apply_partial_rope,
    moonshine_fixed_cache_write,
    moonshine_residual,
    moonshine_rope_tables,
    moonshine_stable_argmax,
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


def test_moonshine_cuda_glue_registry_resolves_primitives_and_composite() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        moonshine_argmax_fp16,
        moonshine_embedding_lookup_fp16,
        moonshine_partial_rope_cache_append_fp16,
        moonshine_partial_rope_fp16,
        moonshine_residual_fp16,
        moonshine_self_cache_append_fp16,
        register_moonshine_glue_kernels,
    )

    register_moonshine_glue_kernels()
    expected = {
        ("moonshine_argmax", "lowest_id"): moonshine_argmax_fp16,
        ("moonshine_embedding", "lookup_i64"): moonshine_embedding_lookup_fp16,
        ("moonshine_residual", "rounded"): moonshine_residual_fp16,
        ("moonshine_partial_rope", "interleaved"): moonshine_partial_rope_fp16,
        ("moonshine_self_cache", "fixed"): moonshine_self_cache_append_fp16,
        (
            "moonshine_partial_rope+moonshine_self_cache",
            "interleaved_fixed_append",
        ): moonshine_partial_rope_cache_append_fp16,
    }
    for (layer, variant), function in expected.items():
        assert resolve(
            backend="cuda_sm120a",
            layer=layer,
            quant="fp16",
            variant=variant,
        ) is function


def test_moonshine_cuda_glue_build_plan_targets_architecture_qualified_sm120a(
    tmp_path,
) -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        plan_moonshine_glue_build,
    )

    artifact = plan_moonshine_glue_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc Moonshine test version",
    )

    assert artifact.family == "cuda_sm120a_moonshine_glue"
    assert artifact.profile.name == "decode"
    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)
    assert artifact.output_path.name == "moonshine_glue.so"
    assert not artifact.cache_dir.exists()


def test_moonshine_cuda_glue_wrappers_keep_raw_pointer_abis() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        moonshine_argmax_fp16,
        moonshine_embedding_lookup_fp16,
        moonshine_partial_rope_cache_append_fp16,
        moonshine_partial_rope_fp16,
        moonshine_residual_fp16,
        moonshine_self_cache_append_fp16,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_argmax_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_residual_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_partial_rope_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_self_cache_append_fp16 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_fp16 = FakeKernel()

    library = FakeLibrary()
    common = {"threads": 256, "stream": 7, "library": library, "runtime": object()}
    moonshine_argmax_fp16(1, 2, 36_864, **common)
    moonshine_embedding_lookup_fp16(1, 2, 3, 416, 36_864, **common)
    moonshine_residual_fp16(1, 2, 3, 416, **common)
    moonshine_partial_rope_fp16(1, 2, 3, 4, 5, 6, 7, 8, 52, 32, 194, **common)
    moonshine_self_cache_append_fp16(1, 2, 3, 4, 5, 8, 52, 194, **common)
    moonshine_partial_rope_cache_append_fp16(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 52, 32, 194, 194, **common
    )
    assert library.hipengine_cuda_sm120a_moonshine_argmax_fp16.calls == [
        (1, 2, 36_864, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16.calls == [
        (1, 2, 3, 416, 36_864, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_residual_fp16.calls == [
        (1, 2, 3, 416, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_partial_rope_fp16.calls == [
        (1, 2, 3, 4, 5, 6, 7, 8, 52, 32, 194, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_self_cache_append_fp16.calls == [
        (1, 2, 3, 4, 5, 8, 52, 194, 256, 7)
    ]
    assert (
        library.hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_fp16.calls
        == [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 52, 32, 194, 194, 256, 7)]
    )


def test_moonshine_cuda_glue_rejects_invalid_shapes_before_build() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        moonshine_argmax_fp16,
        moonshine_embedding_lookup_fp16,
        moonshine_partial_rope_fp16,
        moonshine_residual_fp16,
    )

    with pytest.raises(ValueError, match="vocab_size"):
        moonshine_argmax_fp16(1, 2, 0)
    with pytest.raises(ValueError, match="hidden_size"):
        moonshine_embedding_lookup_fp16(1, 2, 3, 0, 36_864)
    with pytest.raises(ValueError, match="elements"):
        moonshine_residual_fp16(1, 2, 3, 0)
    with pytest.raises(ValueError, match="rotary_dim"):
        moonshine_partial_rope_fp16(1, 2, 3, 4, 5, 6, 7, 8, 52, 33, 194)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_embedding_residual_and_argmax_match_cpu() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_argmax_fp16,
        moonshine_embedding_lookup_fp16,
        moonshine_residual_fp16,
    )

    rng = np.random.default_rng(0x61E)
    hidden, vocab, token = 416, 37, 23
    embedding = rng.normal(0.0, 0.1, size=(vocab, hidden)).astype(np.float16)
    residual = rng.normal(0.0, 0.2, size=(hidden,)).astype(np.float16)
    delta = rng.normal(0.0, 0.2, size=(hidden,)).astype(np.float16)
    expected_residual = moonshine_residual(residual, delta)
    logits = rng.normal(0.0, 0.2, size=(36_864,)).astype(np.float16)
    logits[7] = np.float16(4.0)
    logits[11] = np.float16(4.0)
    expected_argmax = moonshine_stable_argmax(logits)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_glue(load=True)
    allocations = []
    try:
        device_embedding = _upload(embedding, runtime, allocations)
        device_token = _upload(np.asarray([token], dtype=np.int64), runtime, allocations)
        device_hidden = _empty((hidden,), np.float16, runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_delta = _upload(delta, runtime, allocations)
        device_sum = _empty((hidden,), np.float16, runtime, allocations)
        device_logits = _upload(logits, runtime, allocations)
        device_argmax = _empty((1,), np.int64, runtime, allocations)
        moonshine_embedding_lookup_fp16(
            device_embedding.ptr,
            device_token.ptr,
            device_hidden.ptr,
            hidden,
            vocab,
            library=library,
            runtime=runtime,
        )
        moonshine_argmax_fp16(
            device_logits.ptr,
            device_argmax.ptr,
            logits.size,
            library=library,
            runtime=runtime,
        )
        moonshine_residual_fp16(
            device_residual.ptr,
            device_delta.ptr,
            device_sum.ptr,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_hidden = _download(device_hidden, (hidden,), np.float16, runtime)
        actual_sum = _download(device_sum, (hidden,), np.float16, runtime)
        actual_argmax = _download(device_argmax, (1,), np.int64, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual_hidden, embedding[token])
    np.testing.assert_array_equal(actual_sum, expected_residual)
    np.testing.assert_array_equal(actual_argmax, expected_argmax.reshape(1))


@pytest.mark.parametrize("position", [0, 1, 63, 193])
@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_rope_cache_composite_matches_unfused_and_cpu(
    position: int,
) -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_partial_rope_cache_append_fp16,
        moonshine_partial_rope_fp16,
        moonshine_self_cache_append_fp16,
    )

    rng = np.random.default_rng(0xCACE + position)
    heads, head_dim, rotary_dim, capacity = 8, 52, 32, 194
    query = rng.normal(0.0, 0.1, size=(1, heads, 1, head_dim)).astype(np.float16)
    key = rng.normal(0.0, 0.1, size=(1, heads, 1, head_dim)).astype(np.float16)
    value = rng.normal(0.0, 0.1, size=(1, heads, 1, head_dim)).astype(np.float16)
    cos, sin = moonshine_rope_tables(capacity, rotary_dim=rotary_dim)
    expected_query, expected_key = moonshine_apply_partial_rope(
        query,
        key,
        cos,
        sin,
        position_ids=np.asarray([position], dtype=np.int64),
        rotary_dim=rotary_dim,
    )
    expected_key_cache = np.zeros((1, heads, capacity, head_dim), dtype=np.float16)
    expected_value_cache = np.zeros_like(expected_key_cache)
    moonshine_fixed_cache_write(
        expected_key_cache,
        expected_value_cache,
        expected_key,
        value,
        position=position,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_glue(load=True)
    allocations = []
    try:
        device_query = _upload(query.reshape(heads, head_dim), runtime, allocations)
        device_key = _upload(key.reshape(heads, head_dim), runtime, allocations)
        device_value = _upload(value.reshape(heads, head_dim), runtime, allocations)
        device_cos = _upload(cos, runtime, allocations)
        device_sin = _upload(sin, runtime, allocations)
        device_position = _upload(np.asarray([position], dtype=np.int64), runtime, allocations)
        separate_query = _empty((heads, head_dim), np.float16, runtime, allocations)
        separate_key = _empty((heads, head_dim), np.float16, runtime, allocations)
        separate_k_cache = _zero_cache(heads, capacity, head_dim, runtime, allocations)
        separate_v_cache = _zero_cache(heads, capacity, head_dim, runtime, allocations)
        fused_query = _empty((heads, head_dim), np.float16, runtime, allocations)
        fused_key = _empty((heads, head_dim), np.float16, runtime, allocations)
        fused_k_cache = _zero_cache(heads, capacity, head_dim, runtime, allocations)
        fused_v_cache = _zero_cache(heads, capacity, head_dim, runtime, allocations)

        moonshine_partial_rope_fp16(
            device_query.ptr,
            device_key.ptr,
            device_cos.ptr,
            device_sin.ptr,
            device_position.ptr,
            separate_query.ptr,
            separate_key.ptr,
            heads,
            head_dim,
            rotary_dim,
            capacity,
            library=library,
            runtime=runtime,
        )
        moonshine_self_cache_append_fp16(
            separate_key.ptr,
            device_value.ptr,
            device_position.ptr,
            separate_k_cache.ptr,
            separate_v_cache.ptr,
            heads,
            head_dim,
            capacity,
            library=library,
            runtime=runtime,
        )
        moonshine_partial_rope_cache_append_fp16(
            device_query.ptr,
            device_key.ptr,
            device_value.ptr,
            device_cos.ptr,
            device_sin.ptr,
            device_position.ptr,
            fused_query.ptr,
            fused_key.ptr,
            fused_k_cache.ptr,
            fused_v_cache.ptr,
            heads,
            head_dim,
            rotary_dim,
            capacity,
            capacity,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        row_shape = (heads, head_dim)
        cache_shape = (heads, capacity, head_dim)
        actual_separate_query = _download(separate_query, row_shape, np.float16, runtime)
        actual_separate_key = _download(separate_key, row_shape, np.float16, runtime)
        actual_fused_query = _download(fused_query, row_shape, np.float16, runtime)
        actual_fused_key = _download(fused_key, row_shape, np.float16, runtime)
        actual_separate_k_cache = _download(
            separate_k_cache, cache_shape, np.float16, runtime
        )
        actual_separate_v_cache = _download(
            separate_v_cache, cache_shape, np.float16, runtime
        )
        actual_fused_k_cache = _download(fused_k_cache, cache_shape, np.float16, runtime)
        actual_fused_v_cache = _download(fused_v_cache, cache_shape, np.float16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual_fused_query, actual_separate_query)
    np.testing.assert_array_equal(actual_fused_key, actual_separate_key)
    np.testing.assert_array_equal(actual_fused_k_cache, actual_separate_k_cache)
    np.testing.assert_array_equal(actual_fused_v_cache, actual_separate_v_cache)
    np.testing.assert_allclose(
        actual_fused_query,
        expected_query.reshape(heads, head_dim),
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        actual_fused_key,
        expected_key.reshape(heads, head_dim),
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    np.testing.assert_array_equal(actual_fused_k_cache, expected_key_cache.reshape(cache_shape))
    np.testing.assert_array_equal(
        actual_fused_v_cache,
        expected_value_cache.reshape(cache_shape),
    )


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


def _zero_cache(heads: int, capacity: int, head_dim: int, runtime, allocations):
    return _upload(
        np.zeros((heads, capacity, head_dim), dtype=np.float16),
        runtime,
        allocations,
    )


def _download(device, shape: tuple[int, ...], dtype, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
