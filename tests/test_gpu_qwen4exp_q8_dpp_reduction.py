"""Exactness gate for the DPP wave reduction in the Q8_0 coltile prefill leaf.

``coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out`` is an implementation
swap of the wave reduction only. The pipelined leaf reduces each accumulator over
32 lanes with five ``__shfl_down`` levels at offsets 16, 8, 4, 2, 1, which compile
to ``ds_bpermute_b32`` - an LDS round trip each, 160 per thread plus 113
``s_waitcnt lgkmcnt``. The DPP variant folds the same pairs with one
``v_permlanex16_b32`` and four VALU data-parallel ops:

* ``v_permlanex16`` pairs lane i with lane i+16, exactly the offset-16 level;
* ``v_mov_dpp row_shr:8`` then ``row_shr:4`` then ``row_shr:2`` pair with
  i-8, i-4, i-2, which is what ``__shfl_down`` at 8, 4, 2 pairs with (a lane's
  own value plus a partner's is the same sum whichever direction the partner is
  named from);
* ``v_add_f32_dpp row_shl:1`` pairs lane 0 with lane 1, the offset-1 level.

So the addition tree is the same tree, and the candidate must reproduce the
pipelined leaf bit for bit. That is what this file checks, on the raw ``uint32``
bits, at shapes that include a partial row tile, a single K iteration, a K that is
not a multiple of the 128-thread stride, a single column tile and the production
width. The narrow shapes also carry the CPU-reference outer gate.

Evidence: ``benchmarks/results/2026-10-04-qwen4exp-q8-prefill-dpp-reduction/``.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
DPP = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
WAVE_SCALE = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out"
ROWFIRST = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_rowfirst_f32_f32_out"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry_resolves_the_dpp_leaf() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    for name in (PARENT, DPP, ROWFIRST):
        assert (
            resolve(
                backend="hip_gfx1151",
                layer="linear",
                quant="gguf_q8_0",
                variant=name.removeprefix("gguf_q8_0_gemv_"),
            )
            is getattr(q8, name)
        )


def test_dispatch_prefers_the_dpp_leaf() -> None:
    """The rewrite chain must pick DPP, and degrade to the parent when absent."""
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, unregister
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _raw_k_coltile_pipelined_dispatch,
    )

    register_gfx1151_kernels(replace=True)
    backend = "hip_gfx1151"
    base = GGUFLinearDispatch(
        KernelKey(backend, "linear", "gguf_q8_0", "coltile8_rowbatch4_wave_scale_f32_f32_out"),
        "raw",
    )
    decode_shape = {"rows": 1, "in_features": 2560, "out_features": 10240}
    qualified_shape = {"rows": 4096, "in_features": 2560, "out_features": 10240}
    assert _raw_k_coltile_pipelined_dispatch(
        base, **decode_shape
    ).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
    )
    # The qualified prefill window claims the checked full-row sibling; the
    # same launch with the dedicated opt-out falls back to the plain DPP leaf.
    os.environ.pop("HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS", None)
    try:
        assert _raw_k_coltile_pipelined_dispatch(
            base, **qualified_shape
        ).key.variant == (
            "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
        )
        os.environ["HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS"] = "0"
        assert _raw_k_coltile_pipelined_dispatch(
            base, **qualified_shape
        ).key.variant == (
            "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
        )
    finally:
        os.environ.pop("HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS", None)

    dpp_key = KernelKey(
        backend, "linear", "gguf_q8_0", "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
    )
    unregister(dpp_key)
    try:
        assert _raw_k_coltile_pipelined_dispatch(base, **decode_shape).key.variant == (
            "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
        )
    finally:
        register_gfx1151_kernels(replace=True)

    # A non-raw ABI and an unrelated variant are left alone.
    assert _raw_k_coltile_pipelined_dispatch(
        GGUFLinearDispatch(base.key, "bf16"), **decode_shape
    ).key.variant == base.key.variant
    other = GGUFLinearDispatch(
        KernelKey(backend, "linear", "gguf_q8_0", "coltile8_rowbatch4_f32_f32_out"), "raw"
    )
    assert _raw_k_coltile_pipelined_dispatch(
        other, **decode_shape
    ).key.variant == other.key.variant


def _make_operands(rows: int, k: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, (n, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(n, k // 32, 2)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    return x, raw.reshape(n, -1)


_CASES = [
    (1, 32, 8),
    (5, 32, 8),
    (7, 96, 24),
    (3, 160, 32),
    (17, 2560, 32),
    (129, 2560, 256),
    (512, 2560, 6144),
    (1023, 2560, 10240),
]


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n", _CASES)
def test_dpp_matches_the_shuffle_tree_bit_exactly(rows: int, k: int, n: int) -> None:
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=20261004)
    allocations = []
    try:
        dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
        outputs = [_alloc((rows, n), np.float32, runtime, allocations) for _ in range(2)]
        getattr(q8, PARENT)(dx.ptr, dw.ptr, outputs[0].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        parent = _download(outputs[0], (rows, n), np.float32, runtime)
        for _ in range(2):
            getattr(q8, DPP)(dx.ptr, dw.ptr, outputs[1].ptr, rows, k, n, runtime=runtime)
            runtime.device_synchronize()
            actual = _download(outputs[1], (rows, n), np.float32, runtime)
            np.testing.assert_array_equal(actual.view(np.uint32), parent.view(np.uint32))
        # The DPP leaf must also agree with the wave-scale grandparent, so the
        # strict chain is intact end to end rather than only against its sibling.
        getattr(q8, WAVE_SCALE)(dx.ptr, dw.ptr, outputs[1].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        grandparent = _download(outputs[1], (rows, n), np.float32, runtime)
        np.testing.assert_array_equal(
            grandparent.view(np.uint32), parent.view(np.uint32)
        )
        if n <= 32:
            expected = x @ dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0).T
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

            def log_softmax(a):
                a = a.astype(np.float64)
                a -= a.max(-1, keepdims=True)
                return a - np.log(np.exp(a).sum(-1, keepdims=True))

            p, q = log_softmax(expected), log_softmax(actual)
            assert np.max(np.sum(np.exp(p) * (p - q), axis=-1)) <= 0.05
            assert np.mean(expected.argmax(-1) == actual.argmax(-1)) >= 0.9
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize(
    "rows,k,n",
    [(17, 2560, 32), (1023, 2560, 10240)],
)
def test_rowfirst_grid_order_matches_the_default_order(
    rows: int, k: int, n: int
) -> None:
    """Swapping the grid axes must not change a single output bit."""
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=20261005)
    allocations = []
    try:
        dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
        outputs = [_alloc((rows, n), np.float32, runtime, allocations) for _ in range(2)]
        getattr(q8, PARENT)(dx.ptr, dw.ptr, outputs[0].ptr, rows, k, n, runtime=runtime)
        getattr(q8, ROWFIRST)(dx.ptr, dw.ptr, outputs[1].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        for i in (0, 1):
            got = _download(outputs[i], (rows, n), np.float32, runtime)
            if i == 0:
                reference = got
            else:
                np.testing.assert_array_equal(got.view(np.uint32), reference.view(np.uint32))
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
