from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.sampling import (
    plan_sampler_build,
    register_sampler_kernels,
    sample_topk_temperature_f32_rows_i32,
)
from hipengine.kernels.registry import resolve


_NEG_INF = np.float32(-3.4028234663852886e38)
_MASK64 = (1 << 64) - 1
_SPLITMIX_INC = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB


def _has_gfx1100() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


def test_sampler_build_plan_uses_native_arch(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    plan = plan_sampler_build(compiler_version="hipcc:test")

    assert "--offload-arch=gfx1151" in plan.command
    assert plan.target_arch == "gfx1151"
    assert plan.output_path.name == "sampler.so"


def test_sampler_registers_for_gfx1151_alias() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_sampler_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert (
        resolve(backend="hip_gfx1151", layer="sampler", quant="f32", variant="topk_temperature_rows_i32")
        is sample_topk_temperature_f32_rows_i32
    )


def test_sampler_wrapper_validates_shapes_before_loading_hip() -> None:
    with pytest.raises(ValueError, match="rows"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=0, vocab_size=16, top_k=4)
    with pytest.raises(ValueError, match="vocab_size"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=0, top_k=4)
    with pytest.raises(ValueError, match="top_k"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=65)
    with pytest.raises(ValueError, match="threads"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=4, threads=256)
    with pytest.raises(ValueError, match="step_index"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=4, step_index=-1)


def _splitmix64(value: int) -> int:
    z = (value + _SPLITMIX_INC) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX_MUL1) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_MUL2) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _uniform01(row_seed: int, step_index: int, row: int) -> np.float32:
    row_component = (((row + 1) & _MASK64) * _SPLITMIX_MUL1) & _MASK64
    step_component = (((step_index + 1) & _MASK64) * _SPLITMIX_INC) & _MASK64
    bits = _splitmix64((int(row_seed) ^ row_component ^ step_component) & _MASK64)
    return np.float32((bits >> 11) * (1.0 / 9007199254740992.0))


def _cpu_reference(logits: np.ndarray, temperatures: np.ndarray, seeds: np.ndarray, *, top_k: int, step_index: int):
    rows, _vocab = logits.shape
    selected = np.full((rows,), -1, dtype=np.int32)
    selected_logprobs = np.full((rows,), _NEG_INF, dtype=np.float32)
    top_indices = np.full((rows, top_k), -1, dtype=np.int32)
    top_logprobs = np.full((rows, top_k), _NEG_INF, dtype=np.float32)

    for row in range(rows):
        row_logits = logits[row]
        finite_ids = np.flatnonzero(np.isfinite(row_logits)).astype(np.int64, copy=False)
        order = np.lexsort((finite_ids, -row_logits[finite_ids]))
        candidates = finite_ids[order][: min(top_k, finite_ids.size)]
        if candidates.size == 0:
            continue
        top_indices[row, : candidates.size] = candidates.astype(np.int32)
        temp = np.float32(temperatures[row])
        if not np.isfinite(temp) or not (temp > np.float32(0.0)):
            selected[row] = np.int32(candidates[0])
            selected_logprobs[row] = np.float32(0.0)
            top_logprobs[row, 0] = np.float32(0.0)
            continue

        scaled = (row_logits[candidates].astype(np.float32) / temp).astype(np.float32)
        max_scaled = np.float32(scaled[0])
        weights = np.empty_like(scaled, dtype=np.float32)
        weight_sum = np.float32(0.0)
        for idx, value in enumerate(scaled):
            weight = np.float32(np.exp(np.float32(value - max_scaled)))
            weights[idx] = weight
            weight_sum = np.float32(weight_sum + weight)
        log_denom = np.float32(np.log(weight_sum) + max_scaled)
        logprobs = (scaled - log_denom).astype(np.float32)
        top_logprobs[row, : candidates.size] = logprobs

        threshold = np.float32(_uniform01(int(seeds[row]), step_index, row) * weight_sum)
        cumulative = np.float32(0.0)
        selected_pos = candidates.size - 1
        for idx, weight in enumerate(weights):
            cumulative = np.float32(cumulative + weight)
            if threshold <= cumulative:
                selected_pos = idx
                break
        selected[row] = np.int32(candidates[selected_pos])
        selected_logprobs[row] = logprobs[selected_pos]

    return selected, selected_logprobs, top_indices, top_logprobs


@pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")
def test_topk_temperature_sampler_matches_cpu_reference_and_is_deterministic() -> None:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.sampling import build_sampler

    rows = 3
    vocab_size = 257
    top_k = 16  # Deliberately beyond the older lm_head top-k helper's k<=8 cap.
    step_index = 7
    rng = np.random.default_rng(0x5A6D1E)
    logits = (rng.standard_normal((rows, vocab_size), dtype=np.float32) * np.float32(1.25)).astype(np.float32)
    logits[0, 3] = np.float32(10.0)
    logits[0, 7] = np.float32(10.0)  # Tie: lower id must sort first.
    logits[0, 11] = np.float32(9.5)
    logits[1, 20] = np.float32(6.0)
    logits[1, 21] = np.float32(5.75)
    logits[2, 5] = np.float32(np.nan)  # Non-finite logits are ignored like the host sampler.
    logits[2, 6] = np.float32(-np.inf)
    temperatures = np.array([0.7, 1.0, 1.8], dtype=np.float32)
    seeds = np.array([0x1234_5678, 0xCAFE_BABE, 0xDEAD_BEEF_1234], dtype=np.uint64)
    expected = _cpu_reference(logits, temperatures, seeds, top_k=top_k, step_index=step_index)

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        lib = build_sampler(load=True, compiler_version=compiler_version)

    bufs = []

    def upload(array: np.ndarray):
        arr = np.ascontiguousarray(array)
        buf = malloc(max(arr.nbytes, 4))
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        return buf

    def alloc(nbytes: int):
        buf = malloc(max(nbytes, 4))
        bufs.append(buf)
        return buf

    def download(buf, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(out), buf, out.nbytes)
        return out

    try:
        logits_d = upload(logits)
        temperatures_d = upload(temperatures)
        seeds_d = upload(seeds)
        selected_d = alloc(rows * np.dtype(np.int32).itemsize)
        selected_logprobs_d = alloc(rows * np.dtype(np.float32).itemsize)
        top_indices_d = alloc(rows * top_k * np.dtype(np.int32).itemsize)
        top_logprobs_d = alloc(rows * top_k * np.dtype(np.float32).itemsize)

        sample_topk_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            top_indices_d.ptr,
            top_logprobs_d.ptr,
            rows,
            vocab_size,
            top_k,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        first = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(top_indices_d, (rows, top_k), np.int32),
            download(top_logprobs_d, (rows, top_k), np.float32),
        )

        # Same row seeds + step must be deterministic across launches.
        sample_topk_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            top_indices_d.ptr,
            top_logprobs_d.ptr,
            rows,
            vocab_size,
            top_k,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        second = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(top_indices_d, (rows, top_k), np.int32),
            download(top_logprobs_d, (rows, top_k), np.float32),
        )

        assert np.array_equal(first[0], expected[0])
        np.testing.assert_allclose(first[1], expected[1], rtol=0, atol=2e-5)
        assert np.array_equal(first[2], expected[2])
        np.testing.assert_allclose(first[3], expected[3], rtol=0, atol=2e-5)
        for observed, repeated in zip(first, second, strict=True):
            if observed.dtype.kind in {"i", "u"}:
                assert np.array_equal(observed, repeated)
            else:
                np.testing.assert_allclose(observed, repeated, rtol=0, atol=0)
    finally:
        for buf in reversed(bufs):
            free(buf)
