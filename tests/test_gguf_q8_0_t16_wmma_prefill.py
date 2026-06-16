"""Correctness tests for P10.B4 dense Q8_0 T16 WMMA prefill.

The Q8T16 dense WMMA prefill kernel consumes the resident T16 tile layout
defined in ``hipengine/quant/gguf_t16.py``:

    tiles[out_tiles16, blocks_per_row, 544]

Each tile slab packs 16 fp16 d values plus 32 K-rows x 16 cols of int8
quant data, so one tile covers a 16-col output slab over a single Q8_0
32-K block. The kernel mirrors ``gguf_q8_0_prefill_wmma_kernel`` (raw
GGUF) but keeps a T16-specific shape-aware (tile_m, tile_n) default policy.
"""

from __future__ import annotations

import ctypes
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    _default_tiles,
    build_gguf_q8_0_t16_prefill,
    gguf_q8_0_t16_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_bf16_f32_out,
    gguf_q8_0_t16_wmma_prefill_bf16_fp16_out,
    gguf_q8_0_t16_wmma_prefill_f32_bf16_out,
    gguf_q8_0_t16_wmma_prefill_f32_f32_out,
    gguf_q8_0_t16_wmma_prefill_f32_fp16_out,
    gguf_q8_0_t16_wmma_prefill_fp16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_fp16_f32_out,
    gguf_q8_0_t16_wmma_prefill_fp16_fp16_out,
    plan_gguf_q8_0_t16_prefill_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf_t16 import (
    GGUF_Q8_0_T16_BLOCK_BYTES,
    repack_gguf_q8_0_tile16,
)
from tests.test_gguf_k_gemv import make_q8_0_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


_ALL_DTYPE_WRAPPERS: dict[tuple[str, str], Any] = {
    ("bf16", "bf16"): gguf_q8_0_t16_wmma_prefill_bf16_bf16_out,
    ("bf16", "fp16"): gguf_q8_0_t16_wmma_prefill_bf16_fp16_out,
    ("bf16", "f32"): gguf_q8_0_t16_wmma_prefill_bf16_f32_out,
    ("fp16", "bf16"): gguf_q8_0_t16_wmma_prefill_fp16_bf16_out,
    ("fp16", "fp16"): gguf_q8_0_t16_wmma_prefill_fp16_fp16_out,
    ("fp16", "f32"): gguf_q8_0_t16_wmma_prefill_fp16_f32_out,
    ("f32", "bf16"): gguf_q8_0_t16_wmma_prefill_f32_bf16_out,
    ("f32", "fp16"): gguf_q8_0_t16_wmma_prefill_f32_fp16_out,
    ("f32", "f32"): gguf_q8_0_t16_wmma_prefill_f32_f32_out,
}

_ALLOWED_TILES = [(16, 16), (32, 16), (16, 32), (32, 32), (64, 16), (64, 32)]

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


# ---------------------------------------------------------------------------
# No-GPU surface checks.
# ---------------------------------------------------------------------------


def test_gguf_q8_0_t16_prefill_default_tiles_match_t16_policy() -> None:
    assert _default_tiles(rows=512, in_features=2048, out_features=4096) == (64, 32)
    assert _default_tiles(rows=768, in_features=2048, out_features=4096) == (64, 32)
    assert _default_tiles(rows=1024, in_features=2048, out_features=4096) == (32, 32)
    assert _default_tiles(rows=512, in_features=2048, out_features=8192) == (64, 32)
    assert _default_tiles(rows=768, in_features=2048, out_features=8192) == (32, 32)
    assert _default_tiles(rows=1024, in_features=2048, out_features=8192) == (32, 32)
    assert _default_tiles(rows=512, in_features=4096, out_features=2048) == (32, 32)
    assert _default_tiles(rows=512, in_features=2048, out_features=2048) == (32, 32)
    assert _default_tiles(rows=1024, in_features=2048, out_features=2048) == (32, 16)
    assert _default_tiles(rows=512, in_features=2048, out_features=512) == (16, 32)
    assert _default_tiles(rows=768, in_features=2048, out_features=512) == (32, 16)
    assert _default_tiles(rows=1024, in_features=2048, out_features=512) == (32, 16)
    assert _default_tiles(rows=512, in_features=512, out_features=2048) == (32, 32)
    assert _default_tiles(rows=1024, in_features=512, out_features=2048) == (32, 16)
    assert _default_tiles(rows=31, in_features=2048, out_features=4096) == (64, 16)
    assert _default_tiles(rows=31, in_features=2048, out_features=512) == (16, 16)
    assert _default_tiles(rows=31, in_features=4096, out_features=2048) == (32, 16)


def test_gguf_q8_0_t16_prefill_registry_and_build_plan() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_bf16_bf16_out
    )
    # Aliased ``t16_wmma_prefill_*`` rewrite spelling is also resolvable.
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_bf16_bf16_out
    )

    artifact = plan_gguf_q8_0_t16_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q8_0_t16_prefill.so"
    assert any(path.name == "gguf_q8_0_t16_prefill.hip" for path in artifact.sources)


@pytest.mark.parametrize("wrapper", list(_ALL_DTYPE_WRAPPERS.values()))
def test_gguf_q8_0_t16_prefill_wrapper_validates_common_contract(wrapper: Any) -> None:
    kwargs = dict(
        x_ptr=1,
        tiles_ptr=2,
        out_ptr=3,
        rows=17,
        in_features=64,
        out_features=32,
    )
    with pytest.raises(ValueError, match="rows"):
        wrapper(**{**kwargs, "rows": 0})
    with pytest.raises(ValueError, match="in_features"):
        wrapper(**{**kwargs, "in_features": 0})
    with pytest.raises(ValueError, match="Q8_0 block size"):
        wrapper(**{**kwargs, "in_features": 30})
    with pytest.raises(ValueError, match="out_features.*multiple of 16"):
        wrapper(**{**kwargs, "out_features": 24})
    with pytest.raises(ValueError, match="tile"):
        wrapper(**{**kwargs, "tile_m": 8, "tile_n": 16})


# ---------------------------------------------------------------------------
# GPU correctness fixtures.
# ---------------------------------------------------------------------------


def _float_array_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_activation(rows: int, in_features: int, seed: int = 0) -> np.ndarray:
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
    raise ValueError(in_dtype)


def _decode_output(out_arr: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return _bf16_bits_to_float32(out_arr)
    if out_dtype == "fp16":
        return out_arr.astype(np.float32)
    if out_dtype == "f32":
        return out_arr
    raise ValueError(out_dtype)


_OUT_DTYPE_TO_NUMPY = {"bf16": np.uint16, "fp16": np.float16, "f32": np.float32}


def _run_q8_0_t16_wmma_prefill_gpu(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    in_dtype: str,
    out_dtype: str,
    tile: tuple[int, int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    activation = _make_activation(rows, in_features, seed=seed)
    qweight = make_q8_0_weight(out_features, in_features)
    reference_f32 = gguf_q8_0_gemv(activation, qweight)
    tiles = repack_gguf_q8_0_tile16(qweight).tiles
    assert tiles.shape[-1] == GGUF_Q8_0_T16_BLOCK_BYTES

    host_in = _prepare_input(activation, in_dtype)
    host_out = np.zeros((rows, out_features), dtype=_OUT_DTYPE_TO_NUMPY[out_dtype])

    library = build_gguf_q8_0_t16_prefill(load=True)
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    wrapper = _ALL_DTYPE_WRAPPERS[(in_dtype, out_dtype)]

    bufs = []
    try:
        x_dev = malloc(host_in.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, tiles_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(np.ascontiguousarray(host_in)), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(np.ascontiguousarray(tiles)), runtime=runtime)
        wrapper_kwargs: dict[str, Any] = {"library": library, "runtime": runtime}
        if tile is not None:
            wrapper_kwargs["tile_m"], wrapper_kwargs["tile_n"] = tile
        wrapper(
            x_dev.ptr,
            tiles_dev.ptr,
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

    return _decode_output(host_out, out_dtype), reference_f32


_ROWS_GRID: list[int] = [4, 16, 17, 32, 33, 48, 64]
_SHAPE_GRID: list[tuple[int, int]] = [
    (64, 16),
    (64, 32),
    (96, 32),
    (128, 48),
    (256, 64),
    (256, 80),
    (512, 128),
]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS_GRID)
@pytest.mark.parametrize(("in_features", "out_features"), _SHAPE_GRID)
def test_p10_b4_q8_0_t16_wmma_prefill_bf16_to_f32_matches_cpu_reference(
    rows: int, in_features: int, out_features: int
) -> None:
    actual, reference = _run_q8_0_t16_wmma_prefill_gpu(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="f32",
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[("bf16", "f32")])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    sorted(_ALL_DTYPE_WRAPPERS),
)
def test_p10_b4_q8_0_t16_wmma_prefill_full_dtype_matrix_matches_cpu_reference(
    in_dtype: str, out_dtype: str
) -> None:
    actual, reference = _run_q8_0_t16_wmma_prefill_gpu(
        rows=32,
        in_features=256,
        out_features=128,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[(in_dtype, out_dtype)])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("tile", _ALLOWED_TILES)
def test_p10_b4_q8_0_t16_wmma_prefill_explicit_tile_selection_matches_cpu_reference(
    tile: tuple[int, int],
) -> None:
    actual, reference = _run_q8_0_t16_wmma_prefill_gpu(
        rows=64,
        in_features=256,
        out_features=64,
        in_dtype="bf16",
        out_dtype="f32",
        tile=tile,
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[("bf16", "f32")])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_p10_b4_q8_0_t16_wmma_prefill_runs_exported_bf16_symbol_on_w7900() -> None:
    """Tiny launch confirms the BF16 symbol runs through to f32 output."""

    actual, reference = _run_q8_0_t16_wmma_prefill_gpu(
        rows=8,
        in_features=64,
        out_features=16,
        in_dtype="bf16",
        out_dtype="f32",
        seed=9,
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[("bf16", "f32")])
