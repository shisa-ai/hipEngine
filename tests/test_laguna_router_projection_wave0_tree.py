"""Exact Laguna c=1 BF16-hidden/F32-weight router wave-0-tree primitive."""

from __future__ import annotations

import ctypes
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

_SOURCE = Path("hipengine/kernels/hip_gfx1100/moe/router.hip")
_RUNTIME = Path("hipengine/runtime/laguna_moe.py")
_LAYER = "router_logits"
_QUANT = "f32"
_VARIANT = "bf16_hidden_wave0_tree"
_RETAINED_VARIANT = "bf16_hidden"
_KERNEL = "qwen35_router_logits_bf16_f32w_wave0_tree_kernel"
_SYMBOL = "hipengine_qwen35_router_logits_bf16_f32w_wave0_tree"
_WRAPPER = "qwen35_router_logits_bf16_f32w_wave0_tree"

_ACCUMULATION = """  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = expert * hidden_size;
  constexpr int64_t kThreads = 256;
  constexpr int64_t kItems = 8;
  constexpr int64_t kStride = kThreads * kItems;
  for (int64_t k = static_cast<int64_t>(threadIdx.x) * kItems;
       k + 7 < hidden_size;
       k += kStride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += bf16_bits_to_float(hidden[h_off + 0]) * weight[w_off + 0];
    acc += bf16_bits_to_float(hidden[h_off + 1]) * weight[w_off + 1];
    acc += bf16_bits_to_float(hidden[h_off + 2]) * weight[w_off + 2];
    acc += bf16_bits_to_float(hidden[h_off + 3]) * weight[w_off + 3];
    acc += bf16_bits_to_float(hidden[h_off + 4]) * weight[w_off + 4];
    acc += bf16_bits_to_float(hidden[h_off + 5]) * weight[w_off + 5];
    acc += bf16_bits_to_float(hidden[h_off + 6]) * weight[w_off + 6];
    acc += bf16_bits_to_float(hidden[h_off + 7]) * weight[w_off + 7];
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x;
       k < hidden_size;
       k += kThreads) {
    acc += bf16_bits_to_float(hidden[hidden_row + k]) * weight[weight_row + k];
  }
"""
_WAVE0_REDUCTION = """  partial[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = static_cast<int>(threadIdx.x);
    float value0 = partial[lane];
    value0 += partial[lane + 128];
    float value64 = partial[lane + 64];
    value64 += partial[lane + 192];
    value0 += value64;
    float value32 = partial[lane + 32];
    value32 += partial[lane + 160];
    float value96 = partial[lane + 96];
    value96 += partial[lane + 224];
    value32 += value96;
    value0 += value32;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value0 += __shfl_down(value0, offset);
    }
    if (lane == 0) {
      logits[token * num_experts + expert] = value0;
    }
  }
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


def _candidate():
    import hipengine.kernels.hip_gfx1100.moe.router as module

    return getattr(module, _WRAPPER, None)


def test_router_projection_wave0_tree_source_preserves_dot_and_exact_tree() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    candidate = _function_block(source, f"void {_KERNEL}(")
    assert candidate.count(_ACCUMULATION) == 1
    assert candidate.count(_WAVE0_REDUCTION) == 1
    assert candidate.count("__syncthreads();") == 1
    assert candidate.count("__shfl_down(value0, offset)") == 1
    assert source.count(
        "__global__ __launch_bounds__(256, 1)\n"
        f"void {_KERNEL}("
    ) == 1
    assert source.count(f'extern "C" int {_SYMBOL}(') == 1


def test_router_projection_wave0_tree_wrapper_validates_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1100.moe.router as module

    candidate = _candidate()
    assert callable(candidate), "wave-0-tree wrapper must be admitted"

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_qwen35_router", fail_build)
    with pytest.raises(ValueError, match="tokens == 1"):
        candidate(0x1000, 0x2000, 0x3000, 2, 3072, 256)
    with pytest.raises(ValueError, match="threads == 256"):
        candidate(0x1000, 0x2000, 0x3000, 1, 3072, 256, threads=128)
    for index in range(3):
        pointers = [0x1000, 0x2000, 0x3000]
        pointers[index] = 0
        with pytest.raises(ValueError, match="non-zero"):
            candidate(*pointers, 1, 3072, 256)
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        candidate(0x1000, 0x2000, 0x3000, 1, 0, 256)
    with pytest.raises(ValueError, match="num_rows must be positive"):
        candidate(0x1000, 0x2000, 0x3000, 1, 3072, 0)


def test_router_projection_wave0_tree_package_registry_scope_and_fallback() -> None:
    from hipengine.kernels.hip_gfx1100 import moe
    from hipengine.kernels.hip_gfx1100.moe.router import (
        qwen35_router_logits_bf16_f32w_auto_256,
        register_qwen35_router_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    candidate = _candidate()
    assert callable(candidate), "wave-0-tree wrapper must be admitted"
    assert getattr(moe, _WRAPPER, None) is candidate
    assert _WRAPPER in moe.__all__
    register_qwen35_router_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer=_LAYER,
        quant=_QUANT,
        variant=_VARIANT,
    ) is candidate
    assert resolve(
        backend="hip_gfx1100",
        layer=_LAYER,
        quant=_QUANT,
        variant=_RETAINED_VARIANT,
    ) is qwen35_router_logits_bf16_f32w_auto_256

    import hipengine.kernels.hip_gfx1151 as gfx1151

    gfx1151.register_gfx1151_kernels(replace=True)
    for backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert not is_registered(KernelKey(backend, _LAYER, _QUANT, _VARIANT))

    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert _VARIANT not in runtime_source
    assert _SYMBOL not in runtime_source
    import hipengine.kernels.hip_gfx1100 as backend

    assert not hasattr(backend, "LAGUNA_ROUTER_PROJECTION_WAVE0_TREE")
    assert _VARIANT not in inspect.getsource(backend)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_router_projection_wave0_tree_matches_retained_and_cpu_edges() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.moe.router import (
        build_qwen35_router,
        qwen35_router_logits_bf16_f32w_auto_256,
    )

    candidate = _candidate()
    assert callable(candidate), "wave-0-tree wrapper must be admitted"
    library = build_qwen35_router(load=True, require_cached=_require_cached_build())
    rng = np.random.default_rng(0x1100)
    for hidden_size, num_rows in ((17, 19), (3072, 17)):
        hidden_f32 = rng.normal(0.0, 0.25, size=hidden_size).astype(np.float32)
        hidden_bits = float_array_to_bf16_bits(hidden_f32)
        edges = np.asarray(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x4780, 0xC780],
            dtype=np.uint16,
        )
        hidden_bits[: min(hidden_size, edges.size)] = edges[: min(hidden_size, edges.size)]
        weight = rng.normal(0.0, 0.2, size=(num_rows, hidden_size)).astype(np.float32)
        weight.reshape(-1)[:8] = np.asarray(
            [0.0, -0.0, 2.0**-60, -(2.0**-60), 1.0, -1.0, 2.0**12, -(2.0**12)],
            dtype=np.float32,
        )
        control = np.full(num_rows, np.nan, dtype=np.float32)
        actual = np.full(num_rows, np.nan, dtype=np.float32)
        arrays = (hidden_bits, weight, control, actual)
        buffers = [malloc(array.nbytes) for array in arrays]
        try:
            for array, buffer in zip(arrays, buffers, strict=True):
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
            qwen35_router_logits_bf16_f32w_auto_256(
                buffers[0].ptr,
                buffers[1].ptr,
                buffers[2].ptr,
                1,
                hidden_size,
                num_rows,
                library=library,
            )
            candidate(
                buffers[0].ptr,
                buffers[1].ptr,
                buffers[3].ptr,
                1,
                hidden_size,
                num_rows,
                library=library,
            )
            copy_device_to_host(host_array_ptr(control), buffers[2], control.nbytes)
            copy_device_to_host(host_array_ptr(actual), buffers[3], actual.nbytes)
        finally:
            for buffer in reversed(buffers):
                free(buffer)

        np.testing.assert_array_equal(actual.view(np.uint32), control.view(np.uint32))
        assert np.isfinite(actual).all()
        cpu = bf16_to_float32(hidden_bits).astype(np.float64) @ weight.astype(np.float64).T
        actual64 = actual.astype(np.float64)
        actual_prob = np.exp(actual64 - float(np.max(actual64)))
        cpu_prob = np.exp(cpu - float(np.max(cpu)))
        actual_prob /= actual_prob.sum()
        cpu_prob /= cpu_prob.sum()
        kl = float(np.sum(cpu_prob * np.log(np.maximum(cpu_prob, 1e-300) / np.maximum(actual_prob, 1e-300))))
        top1 = float(int(np.argmax(actual64)) == int(np.argmax(cpu)))
        assert kl <= 0.05
        assert top1 >= 0.90
