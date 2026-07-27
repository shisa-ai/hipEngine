"""Exact Laguna D9 MoE-tail wave-0 RMS-tree primitive contract."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import laguna_aggregate_moe_tail_next_rmsnorm
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

_SOURCE = Path("hipengine/kernels/hip_gfx1100/fused/paro_combine.hip")
_RUNTIME = Path("hipengine/runtime/laguna_gguf_runner.py")
_LAYER = "moe_tail+next_rmsnorm"
_QUANT = "bf16"
_VARIANT = "laguna_aggregate_wave0_tree_gguf_f32_weight_out"
_KERNEL = "laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_out_kernel"
_SYMBOL = "hipengine_laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out"

_BASELINE_REDUCTION = """  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(features) + eps);
"""
_WAVE0_REDUCTION = """  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  if (tid < 32) {
    float wave0 = partial[tid];
    wave0 += partial[tid + 128];
    float wave1 = partial[tid + 32];
    wave1 += partial[tid + 160];
    float wave2 = partial[tid + 64];
    wave2 += partial[tid + 192];
    float wave3 = partial[tid + 96];
    wave3 += partial[tid + 224];
    wave0 += wave2;
    wave1 += wave3;
    wave0 += wave1;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      wave0 += __shfl_down(wave0, stride, 32);
    }
    if (tid == 0) {
      partial[0] = wave0;
    }
  }
  __syncthreads();
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(features) + eps);
"""


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _function_block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {marker}")


def _upload(runtime, buffers, array: np.ndarray):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    contiguous = np.ascontiguousarray(array)
    buffer = malloc(max(4, contiguous.nbytes), runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
    return buffer


def _allocate(runtime, buffers, nbytes: int):
    from hipengine.core.memory import malloc

    buffer = malloc(max(4, nbytes), runtime=runtime)
    buffers.append(buffer)
    return buffer


def _download(runtime, buffer, count: int) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    out = np.empty(count, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, runtime=runtime)
    return out


def _free_all(runtime, buffers) -> None:
    from hipengine.core.memory import free

    for buffer in reversed(buffers):
        free(buffer, runtime=runtime)


def _fixture(hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x5700 + hidden)
    routed = float_array_to_bf16_bits(
        rng.normal(0.0, 0.7, size=hidden).astype(np.float32)
    )
    shared = float_array_to_bf16_bits(
        rng.normal(0.0, 0.7, size=hidden).astype(np.float32)
    )
    post = float_array_to_bf16_bits(
        rng.normal(0.0, 0.7, size=hidden).astype(np.float32)
    )
    edges = np.array(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F00, 0xBF00],
        dtype=np.uint16,
    )
    take = min(hidden, edges.size)
    routed[:take] = edges[:take]
    shared[:take] = edges[::-1][:take]
    post[:take] = np.roll(edges, 3)[:take]
    norm_weight = rng.uniform(-1.75, 1.75, size=hidden).astype(np.float32)
    if hidden >= 6:
        norm_weight[:6] = np.array(
            [
                0.0,
                -0.0,
                np.nextafter(np.float32(0.0), np.float32(1.0)),
                np.nextafter(np.float32(0.0), np.float32(-1.0)),
                1.0,
                -1.0,
            ],
            dtype=np.float32,
        )
    return routed, shared, post, norm_weight


def _quality(actual_bits: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    actual = bf16_to_float32(actual_bits).astype(np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    actual_shift = actual - float(np.max(actual))
    expected_shift = expected64 - float(np.max(expected64))
    actual_prob = np.exp(actual_shift)
    expected_prob = np.exp(expected_shift)
    actual_prob /= np.sum(actual_prob)
    expected_prob /= np.sum(expected_prob)
    kl = float(
        np.sum(
            expected_prob
            * np.log(np.maximum(expected_prob, 1e-300) / np.maximum(actual_prob, 1e-300))
        )
    )
    top1 = float(int(np.argmax(actual)) == int(np.argmax(expected64)))
    return kl, top1


def test_wave0_tree_source_is_exact_retained_d9_plus_frozen_reduction() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    baseline = _function_block(
        source,
        "__global__ void laguna_aggregate_moe_tail_next_rmsnorm_out_kernel(",
    )
    candidate = _function_block(source, f"__global__ void {_KERNEL}(")
    expected = baseline.replace(
        "laguna_aggregate_moe_tail_next_rmsnorm_out_kernel",
        _KERNEL,
        1,
    ).replace(_BASELINE_REDUCTION, _WAVE0_REDUCTION, 1)
    assert candidate == expected
    assert baseline.count(_BASELINE_REDUCTION) == 1
    assert candidate.count(_WAVE0_REDUCTION) == 1
    assert candidate.count("__syncthreads();") == 2
    assert candidate.count("__shfl_down(wave0, stride, 32)") == 1
    assert source.count(f'extern "C" int {_SYMBOL}(') == 1


def test_wave0_tree_wrapper_validates_before_build(monkeypatch: pytest.MonkeyPatch) -> None:
    import hipengine.kernels.hip_gfx1100.fused.paro_combine as module

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_paro_combine", fail_build)
    launch = module.laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out
    valid = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000]
    for index in range(len(valid)):
        pointers = list(valid)
        pointers[index] = 0
        with pytest.raises(ValueError, match="non-zero"):
            launch(*pointers, 3072)
    with pytest.raises(ValueError, match="features must be positive"):
        launch(*valid, 0)


def test_wave0_tree_package_registry_backend_scope_fallbacks_and_no_runtime_owner() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100 import fused
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import register_gguf_ops
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out,
        laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out,
        register_paro_combine_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    candidate_key = KernelKey("hip_gfx1100", _LAYER, _QUANT, _VARIANT)
    retained_key = KernelKey(
        "hip_gfx1100",
        _LAYER,
        _QUANT,
        "laguna_aggregate_gguf_f32_weight_out",
    )
    assert (
        fused.laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out
        is laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out
    )
    assert (
        "laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out"
        in fused.__all__
    )
    register_paro_combine_kernels()
    register_gguf_ops()
    assert (
        resolve(
            backend=candidate_key.backend,
            layer=candidate_key.layer,
            quant=candidate_key.quant,
            variant=candidate_key.variant,
        )
        is laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out
    )
    assert (
        resolve(
            backend=retained_key.backend,
            layer=retained_key.layer,
            quant=retained_key.quant,
            variant=retained_key.variant,
        )
        is laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out
    )
    assert is_registered(KernelKey("hip_gfx1100", "elementwise", "bf16", "add"))
    assert is_registered(
        KernelKey("hip_gfx1100", "rmsnorm", "gguf_f32_weight", "bf16_out")
    )

    load_backend_kernel_package("hip_gfx1151")
    for backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert not is_registered(KernelKey(backend, _LAYER, _QUANT, _VARIANT))

    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert _VARIANT not in runtime_source
    assert _SYMBOL not in runtime_source
    import hipengine.kernels.hip_gfx1100 as backend

    assert not hasattr(backend, "LAGUNA_MOE_TAIL_WAVE0_TREE")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_wave0_tree_matches_retained_unfused_and_cpu_for_edge_fixtures() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        build_gguf_ops,
        gguf_bf16_add,
        gguf_rmsnorm_bf16_f32_weight,
    )
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        build_paro_combine,
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out,
        laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out,
    )

    runtime = get_hip_runtime()
    combine_library = build_paro_combine(
        load=True,
        require_cached=_require_cached_build(),
    )
    gguf_library = build_gguf_ops(
        load=True,
        require_cached=_require_cached_build(),
    )
    for hidden in (17, 3072):
        routed_bits, shared_bits, post_bits, norm_weight = _fixture(hidden)
        expected_hidden, expected_norm = laguna_aggregate_moe_tail_next_rmsnorm(
            bf16_to_float32(routed_bits),
            bf16_to_float32(shared_bits),
            bf16_to_float32(post_bits),
            norm_weight,
        )
        buffers = []
        try:
            routed_d = _upload(runtime, buffers, routed_bits)
            shared_d = _upload(runtime, buffers, shared_bits)
            post_d = _upload(runtime, buffers, post_bits)
            weight_d = _upload(runtime, buffers, norm_weight)
            nbytes = hidden * np.dtype(np.uint16).itemsize
            first_add_d = _allocate(runtime, buffers, nbytes)
            fallback_hidden_d = _allocate(runtime, buffers, nbytes)
            fallback_norm_d = _allocate(runtime, buffers, nbytes)
            retained_hidden_d = _allocate(runtime, buffers, nbytes)
            retained_norm_d = _allocate(runtime, buffers, nbytes)
            candidate_hidden_d = _allocate(runtime, buffers, nbytes)
            candidate_norm_d = _allocate(runtime, buffers, nbytes)

            gguf_bf16_add(
                routed_d.ptr,
                shared_d.ptr,
                first_add_d.ptr,
                hidden,
                library=gguf_library,
                runtime=runtime,
            )
            gguf_bf16_add(
                post_d.ptr,
                first_add_d.ptr,
                fallback_hidden_d.ptr,
                hidden,
                library=gguf_library,
                runtime=runtime,
            )
            gguf_rmsnorm_bf16_f32_weight(
                fallback_hidden_d.ptr,
                weight_d.ptr,
                fallback_norm_d.ptr,
                1,
                hidden,
                1e-6,
                library=gguf_library,
                runtime=runtime,
            )
            laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
                routed_d.ptr,
                shared_d.ptr,
                post_d.ptr,
                weight_d.ptr,
                retained_norm_d.ptr,
                retained_hidden_d.ptr,
                hidden,
                library=combine_library,
                runtime=runtime,
            )
            laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out(
                routed_d.ptr,
                shared_d.ptr,
                post_d.ptr,
                weight_d.ptr,
                candidate_norm_d.ptr,
                candidate_hidden_d.ptr,
                hidden,
                library=combine_library,
                runtime=runtime,
            )
            runtime.device_synchronize()

            fallback_hidden = _download(runtime, fallback_hidden_d, hidden)
            fallback_norm = _download(runtime, fallback_norm_d, hidden)
            retained_hidden = _download(runtime, retained_hidden_d, hidden)
            retained_norm = _download(runtime, retained_norm_d, hidden)
            candidate_hidden = _download(runtime, candidate_hidden_d, hidden)
            candidate_norm = _download(runtime, candidate_norm_d, hidden)
        finally:
            _free_all(runtime, buffers)

        np.testing.assert_array_equal(candidate_hidden, retained_hidden)
        np.testing.assert_array_equal(candidate_norm, retained_norm)
        np.testing.assert_array_equal(candidate_hidden, fallback_hidden)
        np.testing.assert_array_equal(candidate_norm, fallback_norm)
        np.testing.assert_array_equal(
            candidate_hidden,
            float_array_to_bf16_bits(expected_hidden),
        )
        np.testing.assert_array_equal(
            candidate_norm,
            float_array_to_bf16_bits(expected_norm),
        )
        kl, top1 = _quality(candidate_norm, expected_norm)
        assert kl <= 0.05
        assert top1 >= 0.90
