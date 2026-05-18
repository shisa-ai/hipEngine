"""Correctness tests for the GGUF Q8_0 batched WMMA prefill kernel.

The kernel is the first batched prefill GEMM from the P8 plan
(see ``docs/GGUF.md`` "P8: real batched prefill GEMM"). It mirrors the
PARO ``awq_fusedw4_prefill_fp16_kernel`` template line-by-line and
replaces the inner K-loop dequant block with Q8_0 byte decoding.

These tests cover three layers:

1. **No-GPU surface** — registry binding, build plan, wrapper contract,
   default tile heuristic. Mirrors the existing ``test_gguf_k_gemv.py``
   build-plan + contract pattern.
2. **GPU correctness** — synthetic Q8_0 weights → WMMA prefill on
   gfx1100 → compared against the ``gguf_q8_0_gemv`` CPU reference with
   ``np.testing.assert_allclose``. Sweeps the rows / shape / dtype matrix
   the task description calls out, plus explicit tile-size selection.
3. **Dispatch integration** — confirms ``launch_gguf_linear`` with
   ``use_wmma_prefill=True`` actually fires the WMMA path and produces
   bf16 output within one bf16 ULP of the CPU reference.
"""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    _ALLOWED_TILES,
    _default_tiles,
    build_gguf_q8_0_prefill,
    gguf_q8_0_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_wmma_prefill_bf16_f32_out,
    gguf_q8_0_wmma_prefill_bf16_fp16_out,
    gguf_q8_0_wmma_prefill_f32_bf16_out,
    gguf_q8_0_wmma_prefill_f32_f32_out,
    gguf_q8_0_wmma_prefill_f32_fp16_out,
    gguf_q8_0_wmma_prefill_fp16_bf16_out,
    gguf_q8_0_wmma_prefill_fp16_f32_out,
    gguf_q8_0_wmma_prefill_fp16_fp16_out,
    plan_gguf_q8_0_prefill_build,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import launch_gguf_linear

# Reuse the synthetic Q8_0 generator from the existing test file. Plain
# module-level function, safe to import at top-level.
from tests.test_gguf_k_gemv import make_q8_0_weight


def _hip_available() -> bool:
    """Whether the HIP runtime libamdhip64.so can be loaded.

    Mirrors the helper in ``tests/test_gguf_q6_k_embedding.py``.
    """

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# 1. No-GPU surface: build plan, registry, contracts.
# ---------------------------------------------------------------------------


_ALL_DTYPE_WRAPPERS: dict[tuple[str, str], Any] = {
    ("bf16", "bf16"): gguf_q8_0_wmma_prefill_bf16_bf16_out,
    ("bf16", "fp16"): gguf_q8_0_wmma_prefill_bf16_fp16_out,
    ("bf16", "f32"): gguf_q8_0_wmma_prefill_bf16_f32_out,
    ("fp16", "bf16"): gguf_q8_0_wmma_prefill_fp16_bf16_out,
    ("fp16", "fp16"): gguf_q8_0_wmma_prefill_fp16_fp16_out,
    ("fp16", "f32"): gguf_q8_0_wmma_prefill_fp16_f32_out,
    ("f32", "bf16"): gguf_q8_0_wmma_prefill_f32_bf16_out,
    ("f32", "fp16"): gguf_q8_0_wmma_prefill_f32_fp16_out,
    ("f32", "f32"): gguf_q8_0_wmma_prefill_f32_f32_out,
}


def test_gguf_q8_0_wmma_prefill_registry_and_build_plan() -> None:
    """Every dtype combo binds to its wrapper, build plan resolves cleanly."""

    for (in_dtype, out_dtype), wrapper in _ALL_DTYPE_WRAPPERS.items():
        variant = f"wmma_prefill_{in_dtype}_{out_dtype}_out"
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant="gguf_q8_0",
                variant=variant,
            )
            is wrapper
        ), variant

    artifact = plan_gguf_q8_0_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q8_0_prefill.so"
    assert "gguf_q8_0_prefill" in str(artifact.output_path)
    assert any(path.name == "gguf_q8_0_prefill.hip" for path in artifact.sources)

    dry_run = build_gguf_q8_0_prefill(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_gguf_q8_0_wmma_prefill_default_tiles_match_paro_heuristic() -> None:
    """P9.C1 tuned heuristic: shape-aware (tile_m, tile_n) defaults.

    Pinning test for the per-shape dispatch decision. Each assertion below
    is anchored to a microbench-best tile measured on RX 7900 XTX / gfx1100
    at rows=512, BF16/BF16. Changing the heuristic must also change these
    assertions deliberately (with new microbench evidence).
    """

    # P9.C1 sweep: at rows=512 the optimal TN is 32 across all out_features.
    # rows < 32 falls back to TN=16 because the bigger TN under-utilises the
    # WMMA tile.
    assert _default_tiles(rows=512, in_features=2048, out_features=8192) == (16, 32)
    assert _default_tiles(rows=512, in_features=2048, out_features=4096) == (16, 32)
    assert _default_tiles(rows=512, in_features=4096, out_features=2048) == (64, 32)
    assert _default_tiles(rows=512, in_features=2048, out_features=2048) == (32, 32)
    assert _default_tiles(rows=512, in_features=2048, out_features=512) == (16, 32)
    assert _default_tiles(rows=32, in_features=2048, out_features=8192) == (16, 32)
    assert _default_tiles(rows=32, in_features=4096, out_features=2048) == (64, 32)
    assert _default_tiles(rows=31, in_features=2048, out_features=8192) == (16, 16)
    assert _default_tiles(rows=31, in_features=4096, out_features=2048) == (64, 16)
    assert _default_tiles(rows=8, in_features=2048, out_features=2048) == (32, 16)
    # tile_m falls back to 16 when out_features < 32 (rare; lm_head etc.).
    assert _default_tiles(rows=512, in_features=2048, out_features=16) == (16, 32)

    for tm, tn in _ALLOWED_TILES:
        assert tm in {16, 32, 64}
        assert tn in {16, 32}


def test_gguf_q8_0_wmma_prefill_wrapper_validates_contract() -> None:
    """Wrappers reject invalid shapes/tiles before allocating any GPU work."""

    # Bad in_features (not divisible by Q8_0 block size 32)
    with pytest.raises(ValueError, match="Q8_0 block"):
        gguf_q8_0_wmma_prefill_bf16_f32_out(
            1, 2, 3, rows=4, in_features=31, out_features=16
        )

    # Bad rows
    with pytest.raises(ValueError, match="rows"):
        gguf_q8_0_wmma_prefill_bf16_f32_out(
            1, 2, 3, rows=0, in_features=64, out_features=16
        )

    # Bad out_features
    with pytest.raises(ValueError, match="out_features"):
        gguf_q8_0_wmma_prefill_bf16_f32_out(
            1, 2, 3, rows=4, in_features=64, out_features=0
        )

    # Unsupported tile size
    with pytest.raises(ValueError, match="tile"):
        gguf_q8_0_wmma_prefill_bf16_f32_out(
            1,
            2,
            3,
            rows=4,
            in_features=64,
            out_features=16,
            tile_m=24,
            tile_n=32,
        )

    # Unsupported tile combo (64, 64) is not in _ALLOWED_TILES
    with pytest.raises(ValueError, match="tile"):
        gguf_q8_0_wmma_prefill_bf16_f32_out(
            1,
            2,
            3,
            rows=4,
            in_features=64,
            out_features=16,
            tile_m=64,
            tile_n=64,
        )


# ---------------------------------------------------------------------------
# 2. GPU correctness: synthetic Q8_0 vs CPU reference, multiple rows / shapes
#    / dtype combinations, including non-multiple-of-tile cases.
# ---------------------------------------------------------------------------


_OUT_DTYPE_TO_NUMPY = {
    "bf16": np.uint16,  # bf16 bits stored as uint16
    "fp16": np.float16,
    "f32": np.float32,
}

_IN_DTYPE_TO_NUMPY = {
    "bf16": np.uint16,
    "fp16": np.float16,
    "f32": np.float32,
}


# Tolerance budget. Calibrated from the inline smoke run during P8.1
# development (see WORKLOG.md). The atol/rtol pair has to absorb (a) the
# WMMA reorder of the K reduction vs the CPU reference's straight-line
# summation, plus (b) one ULP of the output type. Output magnitudes in the
# synthetic fixture sit around 10..100 (in_features=512 with small Q8_0
# quants and bf16 activations), so a few-ULP relative tolerance dominates.
_TOLERANCES: dict[tuple[str, str], dict[str, float]] = {
    ("bf16", "bf16"): {"rtol": 5e-3, "atol": 5e-2},
    ("bf16", "fp16"): {"rtol": 5e-3, "atol": 1e-2},
    ("bf16", "f32"): {"rtol": 1e-4, "atol": 1e-5},
    ("fp16", "bf16"): {"rtol": 5e-3, "atol": 5e-2},
    ("fp16", "fp16"): {"rtol": 5e-3, "atol": 1e-2},
    ("fp16", "f32"): {"rtol": 1e-4, "atol": 1e-5},
    ("f32", "bf16"): {"rtol": 5e-3, "atol": 5e-2},
    ("f32", "fp16"): {"rtol": 5e-3, "atol": 1e-2},
    ("f32", "f32"): {"rtol": 1e-5, "atol": 1e-5},
}


def _float_array_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16, returned as uint16 bits."""

    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """uint16 bf16 bits -> float32."""

    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_activation(rows: int, in_features: int, seed: int = 0) -> np.ndarray:
    """Deterministic bounded fp32 activation matrix.

    The arange-mod-13 pattern keeps values in roughly ``[-6/8, 6/8]`` which
    fits cleanly in fp16/bf16/f32 and avoids accidental overflow under
    accumulation.
    """

    rng = (
        np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features)
        + seed
    )
    return ((rng % 13) - 6) / 8.0


def _prepare_input(activation_f32: np.ndarray, in_dtype: str) -> np.ndarray:
    if in_dtype == "bf16":
        return _float_array_to_bf16_bits(activation_f32)
    if in_dtype == "fp16":
        return activation_f32.astype(np.float16)
    if in_dtype == "f32":
        return activation_f32.astype(np.float32)
    raise ValueError(f"unsupported in_dtype: {in_dtype}")


def _decode_output(out_arr: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return _bf16_bits_to_float32(out_arr)
    if out_dtype == "fp16":
        return out_arr.astype(np.float32)
    if out_dtype == "f32":
        return out_arr
    raise ValueError(f"unsupported out_dtype: {out_dtype}")


def _run_q8_0_wmma_prefill_gpu(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    in_dtype: str,
    out_dtype: str,
    tile: tuple[int, int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Launch the WMMA prefill kernel on the GPU; return (actual_f32, ref_f32).

    Caller is responsible for ``@pytest.mark.skipif(not _hip_available(), ...)``.
    """

    activation = _make_activation(rows, in_features, seed=seed)
    qweight = make_q8_0_weight(out_features, in_features)
    reference_f32 = gguf_q8_0_gemv(activation, qweight)

    host_in = _prepare_input(activation, in_dtype)
    host_out = np.zeros((rows, out_features), dtype=_OUT_DTYPE_TO_NUMPY[out_dtype])

    library = build_gguf_q8_0_prefill(load=True)
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    wrapper = _ALL_DTYPE_WRAPPERS[(in_dtype, out_dtype)]

    bufs = []
    try:
        x_dev = malloc(host_in.nbytes, runtime=runtime)
        qw_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, qw_dev, out_dev))
        copy_host_to_device(
            x_dev, host_array_ptr(np.ascontiguousarray(host_in)), runtime=runtime
        )
        copy_host_to_device(
            qw_dev, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime
        )
        wrapper_kwargs: dict[str, Any] = {"library": library, "runtime": runtime}
        if tile is not None:
            wrapper_kwargs["tile_m"], wrapper_kwargs["tile_n"] = tile
        wrapper(
            x_dev.ptr,
            qw_dev.ptr,
            out_dev.ptr,
            rows,
            in_features,
            out_features,
            **wrapper_kwargs,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    actual_f32 = _decode_output(host_out, out_dtype)
    return actual_f32, reference_f32


# Rows from the task description plus a few extras to lock in tile
# boundaries.
_ROWS_GRID: list[int] = [4, 16, 17, 32, 33, 48, 64]

# (in_features, out_features) shapes. Each row sweeps a different tile
# alignment scenario. in_features is always a multiple of 32 (the Q8_0
# block size is non-negotiable for this kernel). out_features deliberately
# straddles tile sizes 16 and 32.
_SHAPE_GRID: list[tuple[int, int]] = [
    (64, 16),       # smallest aligned
    (64, 24),       # out_features not multiple of 16
    (96, 32),
    (128, 48),
    (256, 64),
    (256, 80),      # out_features not multiple of 32
    (512, 128),     # larger K dimension
]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS_GRID)
@pytest.mark.parametrize(("in_features", "out_features"), _SHAPE_GRID)
def test_gguf_q8_0_wmma_prefill_bf16_to_f32_matches_cpu_reference(
    rows: int, in_features: int, out_features: int
) -> None:
    """Primary correctness sweep: bf16 in / f32 out vs gguf_q8_0_gemv.

    bf16 -> f32 is the most numerically informative comparison: the input
    representation matches the runner's hidden state, and the f32 output
    preserves the WMMA accumulator without an additional lossy cast.
    """

    actual, reference = _run_q8_0_wmma_prefill_gpu(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="f32",
    )
    tol = _TOLERANCES[("bf16", "f32")]
    np.testing.assert_allclose(actual, reference, **tol)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [
        ("bf16", "bf16"),
        ("bf16", "fp16"),
        ("bf16", "f32"),
        ("fp16", "bf16"),
        ("fp16", "fp16"),
        ("fp16", "f32"),
        ("f32", "bf16"),
        ("f32", "fp16"),
        ("f32", "f32"),
    ],
)
def test_gguf_q8_0_wmma_prefill_full_dtype_matrix_matches_cpu_reference(
    in_dtype: str, out_dtype: str
) -> None:
    """All 9 dtype combinations on one representative shape (rows=32, 256x128)."""

    actual, reference = _run_q8_0_wmma_prefill_gpu(
        rows=32,
        in_features=256,
        out_features=128,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    tol = _TOLERANCES[(in_dtype, out_dtype)]
    np.testing.assert_allclose(actual, reference, **tol)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("tile", sorted(_ALLOWED_TILES))
def test_gguf_q8_0_wmma_prefill_explicit_tile_selection_matches_cpu_reference(
    tile: tuple[int, int],
) -> None:
    """Every allowed (tile_m, tile_n) selection produces correct output."""

    # Pick a shape that's a clean multiple of every allowed tile so we
    # isolate tile correctness from boundary masking. 64 rows, 256 in,
    # 64 out is a multiple of all of {16, 32, 64}.
    actual, reference = _run_q8_0_wmma_prefill_gpu(
        rows=64,
        in_features=256,
        out_features=64,
        in_dtype="bf16",
        out_dtype="f32",
        tile=tile,
    )
    tol = _TOLERANCES[("bf16", "f32")]
    np.testing.assert_allclose(actual, reference, **tol)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q8_0_wmma_prefill_handles_unaligned_rows_and_out_features() -> None:
    """Non-multiple-of-tile rows AND out_features in the same call.

    rows=17, out_features=24 forces every active tile to be a boundary
    tile under the default (32, 32) heuristic — exercises the safe_token
    / safe_out masking AND the boundary store masks at the same time.
    """

    actual, reference = _run_q8_0_wmma_prefill_gpu(
        rows=17,
        in_features=64,
        out_features=24,
        in_dtype="bf16",
        out_dtype="f32",
    )
    tol = _TOLERANCES[("bf16", "f32")]
    np.testing.assert_allclose(actual, reference, **tol)


# ---------------------------------------------------------------------------
# 3. Dispatch integration: launch_gguf_linear + use_wmma_prefill=True
#    actually fires the WMMA path end-to-end and produces correct bf16
#    output. Complements the test_gguf_linear_dispatch.py unit tests with
#    a real GPU exec.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_launch_gguf_linear_wmma_prefill_matches_cpu_reference() -> None:
    """End-to-end: launch_gguf_linear(use_wmma_prefill=True) bf16->bf16.

    Confirms the new dispatch ABI ('wmma_raw'), the kernel build, and the
    bf16 output cast all work together. Tolerance is one bf16 ULP relative
    + a small absolute floor; that matches what the existing decode path
    already produces (see WORKLOG task #3 smoke).
    """

    from hipengine.core.hip import get_hip_runtime

    rows, in_features, out_features = 64, 256, 128

    activation = _make_activation(rows, in_features)
    qweight = make_q8_0_weight(out_features, in_features)
    reference_f32 = gguf_q8_0_gemv(activation, qweight)
    host_in = _float_array_to_bf16_bits(activation)
    host_out = np.zeros((rows, out_features), dtype=np.uint16)

    runtime = get_hip_runtime()
    bufs = []
    try:
        x_dev = malloc(host_in.nbytes, runtime=runtime)
        qw_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, qw_dev, out_dev))
        copy_host_to_device(
            x_dev, host_array_ptr(np.ascontiguousarray(host_in)), runtime=runtime
        )
        copy_host_to_device(
            qw_dev,
            host_array_ptr(np.ascontiguousarray(qweight)),
            runtime=runtime,
        )

        # Minimal weight-shape fake matching the runtime's expectations.
        weight = SimpleNamespace(
            spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0"),
            allocation=lambda name="raw": SimpleNamespace(
                tensor=SimpleNamespace(ptr=qw_dev.ptr)
            ),
        )
        launch_gguf_linear(
            weight,
            x_dev.ptr,
            out_dev.ptr,
            rows,
            in_features,
            out_features,
            output_dtype="bf16",
            runtime=runtime,
            use_wmma_prefill=True,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    actual_f32 = _bf16_bits_to_float32(host_out)
    np.testing.assert_allclose(actual_f32, reference_f32, **_TOLERANCES[("bf16", "bf16")])
