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


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_lm_head_argmax_wave8_matches_two_step_wave8_path() -> None:
    """The fused wave8 kernel equals wave8 projection -> stable argmax (C6/RR-8)."""
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_argmax_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_wave8_scratch_elements,
        moonshine_lm_head_argmax_wave8_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection_wave8,
    )

    rng = np.random.default_rng(0xC6BEA7)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library_lm = build_moonshine_lm_head(load=True)
    library_proj = build_moonshine_projection(load=True)
    library_glue = build_moonshine_glue(load=True)
    allocations = []
    try:
        for _ in range(4):
            hidden = rng.normal(0.0, 0.05, size=(_HIDDEN,)).astype(np.float16)
            weight = rng.normal(0.0, 0.03, size=(_VOCAB, _HIDDEN)).astype(np.float16)

            d_input = _upload(hidden, runtime, allocations)
            d_weight = _upload(weight, runtime, allocations)
            d_logits = _alloc_f16((_VOCAB,), runtime, allocations)
            d_argmax = _alloc_i64(1, runtime, allocations)
            moonshine_f16_lm_head_projection_wave8(
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
                d_logits.ptr, d_argmax.ptr, _VOCAB, library=library_glue, runtime=runtime
            )
            runtime.device_synchronize()
            two_step_index = int(_download_i64(d_argmax, 1, runtime)[0])
            logits = _download_f16(d_logits, (_VOCAB,), runtime)
            two_step_value = float(logits[two_step_index])

            num_blocks = lm_head_argmax_wave8_scratch_elements(_VOCAB)
            d_values = _alloc_f32(num_blocks, runtime, allocations)
            d_indices = _alloc_i64(num_blocks, runtime, allocations)
            d_out_index = _alloc_i64(1, runtime, allocations)
            d_out_value = _alloc_f32(1, runtime, allocations)
            moonshine_lm_head_argmax_wave8_fp16(
                d_input.ptr,
                d_weight.ptr,
                d_values.ptr,
                d_indices.ptr,
                d_out_index.ptr,
                d_out_value.ptr,
                _HIDDEN,
                _VOCAB,
                library=library_lm,
                runtime=runtime,
            )
            runtime.device_synchronize()
            fused_index = int(_download_i64(d_out_index, 1, runtime)[0])
            fused_value = float(_download_f32(d_out_value, 1, runtime)[0])

            assert fused_index == two_step_index, (
                f"wave8 fused {fused_index} != two-step wave8 {two_step_index}"
            )
            np.testing.assert_allclose(fused_value, two_step_value, rtol=1e-6, atol=1e-6)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_lm_head_argmax_wave8_matches_exact_fused_within_reassociation() -> None:
    """Fused wave8 vs exact fused: token flips only inside the reassociation delta.

    The wave8 and exact 256-thread fused stages differ only in FP32 accumulation
    order, so their FP16 logits differ by at most a few ULP.  The logit-margin
    gate (review §7.3) holds: when the selected token differs, the exact
    top1-top2 margin must be smaller than the observed max logit delta.  We also
    assert the max logit delta at the argmax stays within a few FP16 ULP.
    """
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_scratch_elements,
        lm_head_argmax_wave8_scratch_elements,
        moonshine_lm_head_argmax_fp16,
        moonshine_lm_head_argmax_wave8_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection,
    )

    rng = np.random.default_rng(0xC6BEAD)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_lm_head(load=True)
    library_proj = build_moonshine_projection(load=True)
    allocations = []
    max_delta = 0.0
    flips = 0
    draws = 12
    try:
        for _ in range(draws):
            hidden = rng.normal(0.0, 0.05, size=(_HIDDEN,)).astype(np.float16)
            weight = rng.normal(0.0, 0.03, size=(_VOCAB, _HIDDEN)).astype(np.float16)

            d_input = _upload(hidden, runtime, allocations)
            d_weight = _upload(weight, runtime, allocations)

            nb_exact = lm_head_argmax_scratch_elements(_VOCAB, 8)
            nb_wave8 = lm_head_argmax_wave8_scratch_elements(_VOCAB)
            ev = _alloc_f32(nb_exact, runtime, allocations)
            ei = _alloc_i64(nb_exact, runtime, allocations)
            wv = _alloc_f32(nb_wave8, runtime, allocations)
            wi = _alloc_i64(nb_wave8, runtime, allocations)
            d_exact_i = _alloc_i64(1, runtime, allocations)
            d_exact_v = _alloc_f32(1, runtime, allocations)
            d_wave8_i = _alloc_i64(1, runtime, allocations)
            d_wave8_v = _alloc_f32(1, runtime, allocations)
            d_logits = _alloc_f16((_VOCAB,), runtime, allocations)

            moonshine_lm_head_argmax_fp16(
                d_input.ptr, d_weight.ptr, ev.ptr, ei.ptr, d_exact_i.ptr, d_exact_v.ptr,
                _HIDDEN, _VOCAB, rows_per_block=8, library=library, runtime=runtime,
            )
            moonshine_lm_head_argmax_wave8_fp16(
                d_input.ptr, d_weight.ptr, wv.ptr, wi.ptr, d_wave8_i.ptr, d_wave8_v.ptr,
                _HIDDEN, _VOCAB, library=library, runtime=runtime,
            )
            moonshine_f16_lm_head_projection(
                d_input.ptr, d_weight.ptr, d_logits.ptr, 1, _HIDDEN, _VOCAB,
                library=library_proj, runtime=runtime,
            )
            runtime.device_synchronize()
            exact_i = int(_download_i64(d_exact_i, 1, runtime)[0])
            wave8_i = int(_download_i64(d_wave8_i, 1, runtime)[0])
            exact_v = float(_download_f32(d_exact_v, 1, runtime)[0])
            wave8_v = float(_download_f32(d_wave8_v, 1, runtime)[0])
            logits = _download_f16(d_logits, (_VOCAB,), runtime)
            # Exact path's own top-1 must equal the exact fused selection.
            assert int(np.argmax(logits)) == exact_i
            top2 = np.partition(logits.astype(np.float32), -2)[-2:]
            top1, top2v = float(top2[1]), float(top2[0])
            margin = top1 - top2v
            max_delta = max(max_delta, abs(exact_v - wave8_v))
            if exact_i != wave8_i:
                flips += 1
                assert margin < max(0.01, max_delta), (
                    f"token flip outside reassociation delta: margin={margin} "
                    f"exact={exact_i} wave8={wave8_i}"
                )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    # The reassociation delta must stay within a few FP16 ULP of the logit
    # magnitude (FP16 ULP at |logit|~1 is 2^-10..2^-9).
    assert max_delta <= 0.02, f"wave8-vs-exact logit delta {max_delta} too large"


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_lm_head_argmax_wave8_batch_matches_single_row() -> None:
    """Static-B wave8 fused equals B sequential single-row wave8 calls (C6/RR-8)."""
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_wave8_scratch_elements,
        moonshine_lm_head_argmax_wave8_batch_fp16,
        moonshine_lm_head_argmax_wave8_fp16,
    )

    rng = np.random.default_rng(0xC6BEEF)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_lm_head(load=True)
    allocations = []
    batch = 4
    try:
        hidden = rng.normal(0.0, 0.05, size=(batch, _HIDDEN)).astype(np.float16)
        weight = rng.normal(0.0, 0.03, size=(_VOCAB, _HIDDEN)).astype(np.float16)
        d_weight = _upload(weight, runtime, allocations)

        num_blocks = lm_head_argmax_wave8_scratch_elements(_VOCAB)
        bv = _alloc_f32(batch * num_blocks, runtime, allocations)
        bi = _alloc_i64(batch * num_blocks, runtime, allocations)
        d_batch_i = _alloc_i64(batch, runtime, allocations)
        d_batch_v = _alloc_f32(batch, runtime, allocations)
        d_batch = _upload(hidden, runtime, allocations)
        moonshine_lm_head_argmax_wave8_batch_fp16(
            d_batch.ptr, d_weight.ptr, bv.ptr, bi.ptr, d_batch_i.ptr, d_batch_v.ptr,
            _HIDDEN, _VOCAB, batch, library=library, runtime=runtime,
        )

        expected_i = np.empty(batch, dtype=np.int64)
        expected_v = np.empty(batch, dtype=np.float32)
        for row in range(batch):
            d_input = _upload(hidden[row : row + 1], runtime, allocations)
            wv = _alloc_f32(num_blocks, runtime, allocations)
            wi = _alloc_i64(num_blocks, runtime, allocations)
            d_i = _alloc_i64(1, runtime, allocations)
            d_v = _alloc_f32(1, runtime, allocations)
            moonshine_lm_head_argmax_wave8_fp16(
                d_input.ptr, d_weight.ptr, wv.ptr, wi.ptr, d_i.ptr, d_v.ptr,
                _HIDDEN, _VOCAB, library=library, runtime=runtime,
            )
            expected_i[row] = int(_download_i64(d_i, 1, runtime)[0])
            expected_v[row] = float(_download_f32(d_v, 1, runtime)[0])
        runtime.device_synchronize()
        actual_i = _download_i64(d_batch_i, batch, runtime)
        actual_v = _download_f32(d_batch_v, batch, runtime)
        np.testing.assert_array_equal(actual_i, expected_i)
        np.testing.assert_allclose(actual_v, expected_v, rtol=1e-6, atol=1e-6)
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
