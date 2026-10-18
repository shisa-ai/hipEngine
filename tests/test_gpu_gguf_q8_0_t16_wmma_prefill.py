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
    _four_wave_prefill_applies,
    _two_wave_prefill_applies,
    build_gguf_q8_0_t16_prefill,
    gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
    gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_bf16_f32_out,
    gguf_q8_0_t16_wmma_prefill_bf16_fp16_out,
    gguf_q8_0_t16_wmma_prefill_f32_bf16_out,
    gguf_q8_0_t16_wmma_prefill_f32_f32_out,
    gguf_q8_0_t16_wmma_prefill_f32_fp16_out,
    gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_fp16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_fp16_f32_out,
    gguf_q8_0_t16_wmma_prefill_fp16_fp16_out,
    plan_gguf_q8_0_t16_prefill_build,
    q8_t16_two_wave_prefill_session,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf_t16 import (
    GGUF_Q8_0_T16_BLOCK_BYTES,
    repack_gguf_q8_0_tile16,
)
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight


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


def test_lcp3_four_wave_policy_is_defaultable_explicit_and_shape_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE", raising=False)
    assert not _four_wave_prefill_applies(tile_m=32, tile_n=32, out_features=8192)
    assert _four_wave_prefill_applies(
        tile_m=32,
        tile_n=32,
        out_features=8192,
        default=True,
    )
    # The Gemma 4 expert gate/up width (2 x 704) is the one measured sub-2048
    # width the schedule covers: 1408 is a whole multiple of the 128-column
    # tile, and the 2026-10-07 A/B on the 8060S measured +13.2% over nwave.
    assert _four_wave_prefill_applies(
        tile_m=32,
        tile_n=32,
        out_features=1408,
        default=True,
    )
    # No other sub-2048 width gains eligibility: the policy change is pinned
    # to the measured shape.
    assert not _four_wave_prefill_applies(
        tile_m=32,
        tile_n=32,
        out_features=1536,
        default=True,
    )
    assert not _four_wave_prefill_applies(
        tile_m=32,
        tile_n=32,
        out_features=512,
        default=True,
    )
    with q8_t16_two_wave_prefill_session(False):
        assert not _four_wave_prefill_applies(
            tile_m=32,
            tile_n=32,
            out_features=8192,
            default=True,
        )

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE", "0")
    assert not _four_wave_prefill_applies(
        tile_m=32,
        tile_n=32,
        out_features=8192,
        default=True,
    )
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE", "1")
    assert _four_wave_prefill_applies(tile_m=32, tile_n=32, out_features=8192)
    assert _four_wave_prefill_applies(tile_m=64, tile_n=32, out_features=2048)
    assert not _four_wave_prefill_applies(tile_m=16, tile_n=32, out_features=8192)
    assert not _four_wave_prefill_applies(tile_m=32, tile_n=16, out_features=8192)
    assert not _four_wave_prefill_applies(tile_m=32, tile_n=32, out_features=512)


def test_lcp3_auto_wrapper_is_request_scoped_with_two_wave_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int | None, ...]]] = []

    class FakeKernel:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, *args: Any) -> int:
            calls.append((self.name, tuple(getattr(arg, "value", arg) for arg in args)))
            return 0

    class FakeLibrary:
        hipengine_gguf_q8_0_t16_wmma_prefill_bf16_bf16_out = FakeKernel("production")
        hipengine_gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out = FakeKernel("two_wave")
        hipengine_gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out = FakeKernel("four_wave")

    kwargs = {
        "x_ptr": 1,
        "tiles_ptr": 2,
        "out_ptr": 3,
        "rows": 512,
        "in_features": 2048,
        "out_features": 8192,
        "library": FakeLibrary(),
        "runtime": object(),
    }
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE", raising=False)
    with q8_t16_two_wave_prefill_session(True):
        gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out(**kwargs)
    assert calls[-1][0] == "four_wave"
    assert calls[-1][1][6:8] == (128, 32)

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE", "0")
    with q8_t16_two_wave_prefill_session(True):
        gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out(**kwargs)
    assert calls[-1][0] == "two_wave"
    assert calls[-1][1][6:8] == (64, 32)

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE")
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", "1")
    with q8_t16_two_wave_prefill_session(False):
        gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out(**kwargs)
    assert calls[-1][0] == "two_wave"
    assert calls[-1][1][6:8] == (64, 32)

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", "0")
    with q8_t16_two_wave_prefill_session(True):
        gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out(**kwargs)
    assert calls[-1][0] == "production"
    assert calls[-1][1][6:8] == (64, 32)


def test_gpf5a_two_wave_policy_is_explicit_and_shape_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", raising=False)
    assert _two_wave_prefill_applies(tile_m=32, tile_n=32, out_features=8192) is False
    assert (
        _two_wave_prefill_applies(
            tile_m=32,
            tile_n=32,
            out_features=8192,
            default=True,
        )
        is True
    )

    with q8_t16_two_wave_prefill_session(False):
        assert (
            _two_wave_prefill_applies(
                tile_m=32,
                tile_n=32,
                out_features=8192,
                default=True,
            )
            is False
        )
    assert (
        _two_wave_prefill_applies(
            tile_m=32,
            tile_n=32,
            out_features=8192,
            default=True,
        )
        is True
    )

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", "0")
    assert (
        _two_wave_prefill_applies(
            tile_m=32,
            tile_n=32,
            out_features=8192,
            default=True,
        )
        is False
    )

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", "1")
    assert _two_wave_prefill_applies(tile_m=32, tile_n=32, out_features=8192) is True
    assert _two_wave_prefill_applies(tile_m=64, tile_n=32, out_features=4096) is True
    assert _two_wave_prefill_applies(tile_m=16, tile_n=32, out_features=512) is False
    assert _two_wave_prefill_applies(tile_m=32, tile_n=16, out_features=8192) is False


def test_gguf_q8_0_t16_prefill_registry_and_build_plan() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out
    )
    # Aliased ``t16_wmma_prefill_*`` rewrite spelling is also resolvable.
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="wmma_prefill_2wave_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="wmma_prefill_4wave_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out
    )

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_pair",
            quant="gguf_q8_0_t16_v1",
            variant="t16_dual_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out
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
    wrapper_override: Any | None = None,
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
    wrapper = wrapper_override or _ALL_DTYPE_WRAPPERS[(in_dtype, out_dtype)]

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
@pytest.mark.parametrize(("rows", "out_features"), [(17, 64), (32, 80), (33, 128)])
def test_gpf5a_two_wave_q8_t16_prefill_is_bit_exact_to_32x32(
    rows: int,
    out_features: int,
) -> None:
    baseline, _ = _run_q8_0_t16_wmma_prefill_gpu(
        rows=rows,
        in_features=256,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="bf16",
        tile=(32, 32),
    )
    candidate, _ = _run_q8_0_t16_wmma_prefill_gpu(
        rows=rows,
        in_features=256,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="bf16",
        tile=(64, 32),
        wrapper_override=gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out,
    )
    np.testing.assert_array_equal(candidate, baseline)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(("rows", "out_features"), [(17, 128), (32, 144), (33, 256)])
def test_lcp3_four_wave_q8_t16_prefill_is_bit_exact_to_32x32(
    rows: int,
    out_features: int,
) -> None:
    baseline, _ = _run_q8_0_t16_wmma_prefill_gpu(
        rows=rows,
        in_features=256,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="bf16",
        tile=(32, 32),
    )
    candidate, _ = _run_q8_0_t16_wmma_prefill_gpu(
        rows=rows,
        in_features=256,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="bf16",
        tile=(128, 32),
        wrapper_override=gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out,
    )
    np.testing.assert_array_equal(candidate, baseline)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_t16_dual_wmma_prefill_is_bit_exact_to_singletons() -> None:
    rows, in_features, out_features = 37, 256, 16
    host_in = _prepare_input(_make_activation(rows, in_features, seed=71), "bf16")
    raw_a = make_q8_0_weight(out_features, in_features)
    raw_b = np.roll(raw_a, 1, axis=0).copy()
    tiles_a = repack_gguf_q8_0_tile16(raw_a)
    tiles_b = repack_gguf_q8_0_tile16(raw_b)
    expected_a = np.empty((rows, out_features), dtype=np.uint16)
    expected_b = np.empty_like(expected_a)
    actual_a = np.empty_like(expected_a)
    actual_b = np.empty_like(expected_a)

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q8_0_t16_prefill(load=True)
    buffers = []

    def upload(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        copy_host_to_device(
            buffer, host_array_ptr(np.ascontiguousarray(array)), runtime=runtime
        )
        buffers.append(buffer)
        return buffer

    try:
        x_dev = upload(host_in)
        a_dev = upload(tiles_a.tiles)
        b_dev = upload(tiles_b.tiles)
        out_devs = [malloc(expected_a.nbytes, runtime=runtime) for _ in range(4)]
        buffers.extend(out_devs)
        for tiles_dev, out_dev in zip((a_dev, b_dev), out_devs[:2], strict=True):
            gguf_q8_0_t16_wmma_prefill_bf16_bf16_out(
                x_dev.ptr, tiles_dev.ptr, out_dev.ptr, rows, in_features,
                out_features, tile_m=16, tile_n=32, library=library,
                runtime=runtime,
            )
        gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out(
            x_dev.ptr, a_dev.ptr, b_dev.ptr, out_devs[2].ptr, out_devs[3].ptr,
            rows, in_features, out_features, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in zip(
            (expected_a, expected_b, actual_a, actual_b), out_devs, strict=True
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


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


# --- pre-converted f16 weight tiles ----------------------------------------


def _run_two_wave_and_f16w(*, rows: int, in_features: int, out_features: int, seed: int = 0):
    """Raw BF16 outputs of the two-wave launch, the f16w launch, the registered
    auto wrapper, and the f16w launch over zeroed weights."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_prefill as q8_t16
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        gguf_q8_0_t16_f16w_bytes,
        gguf_q8_0_t16_f16w_for,
        gguf_q8_0_t16_tiles_to_f16w,
        gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out,
        register_gguf_q8_0_t16_f16w,
        unregister_gguf_q8_0_t16_f16w,
    )

    activation = _make_activation(rows, in_features, seed=seed)
    qweight = make_q8_0_weight(out_features, in_features)
    tiles = np.ascontiguousarray(repack_gguf_q8_0_tile16(qweight).tiles)
    host_in = np.ascontiguousarray(_prepare_input(activation, "bf16"))
    library = build_gguf_q8_0_t16_prefill(load=True)
    runtime = get_hip_runtime()
    f16w_bytes = gguf_q8_0_t16_f16w_bytes(in_features, out_features)
    assert f16w_bytes == in_features * out_features * 2
    outputs: dict[str, np.ndarray] = {}
    bufs = []

    def read(out_dev) -> np.ndarray:
        runtime.device_synchronize()
        host = np.zeros((rows, out_features), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), out_dev, runtime=runtime)
        return host

    try:
        x_dev = malloc(host_in.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        out_dev = malloc(rows * out_features * 2, runtime=runtime)
        f16w_dev = malloc(f16w_bytes, runtime=runtime)
        zero_dev = malloc(f16w_bytes, runtime=runtime)
        bufs.extend((x_dev, tiles_dev, out_dev, f16w_dev, zero_dev))
        copy_host_to_device(x_dev, host_array_ptr(host_in), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        zeros = np.zeros(f16w_bytes // 2, dtype=np.uint16)
        copy_host_to_device(zero_dev, host_array_ptr(zeros), runtime=runtime)
        kwargs = {"library": library, "runtime": runtime}

        gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out(
            x_dev.ptr, tiles_dev.ptr, out_dev.ptr, rows, in_features, out_features,
            tile_m=64, tile_n=32, **kwargs,
        )
        outputs["two_wave"] = read(out_dev)

        gguf_q8_0_t16_tiles_to_f16w(
            tiles_dev.ptr, f16w_dev.ptr, in_features, out_features, **kwargs
        )
        gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out(
            x_dev.ptr, tiles_dev.ptr, f16w_dev.ptr, out_dev.ptr, rows, in_features,
            out_features, **kwargs,
        )
        outputs["f16w"] = read(out_dev)

        gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out(
            x_dev.ptr, tiles_dev.ptr, zero_dev.ptr, out_dev.ptr, rows, in_features,
            out_features, **kwargs,
        )
        outputs["zero_weights"] = read(out_dev)

        # The four-wave schedule, which the gfx1151 registry binds, has the same
        # pair of launches and reads the same f16w tiles.
        q8_t16.gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out(
            x_dev.ptr, tiles_dev.ptr, out_dev.ptr, rows, in_features, out_features,
            tile_m=128, tile_n=32, **kwargs,
        )
        outputs["four_wave"] = read(out_dev)
        q8_t16.gguf_q8_0_t16_wmma_prefill_4wave_f16w_bf16_bf16_out(
            x_dev.ptr, tiles_dev.ptr, f16w_dev.ptr, out_dev.ptr, rows, in_features,
            out_features, **kwargs,
        )
        outputs["four_wave_f16w"] = read(out_dev)
        register_gguf_q8_0_t16_f16w(tiles_dev.ptr, zero_dev.ptr, in_features, out_features)
        try:
            q8_t16.gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out(
                x_dev.ptr, tiles_dev.ptr, out_dev.ptr, rows, in_features, out_features,
                tile_m=32, tile_n=32, **kwargs,
            )
            outputs["auto_4wave_registered_zero"] = read(out_dev)
        finally:
            unregister_gguf_q8_0_t16_f16w(tiles_dev.ptr)

        # The registered auto wrapper under its own default, which is the
        # two-wave schedule the gfx1100 registry binds: registered against the
        # zeroed tiles so the route it took is visible, then unregistered.
        register_gguf_q8_0_t16_f16w(tiles_dev.ptr, zero_dev.ptr, in_features, out_features)
        try:
            gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out(
                x_dev.ptr, tiles_dev.ptr, out_dev.ptr, rows, in_features, out_features,
                tile_m=32, tile_n=32, **kwargs,
            )
            outputs["auto_registered_zero"] = read(out_dev)
            # A different shape for the same pointer is not a match.
            assert gguf_q8_0_t16_f16w_for(tiles_dev.ptr, in_features, out_features + 16) is None
        finally:
            unregister_gguf_q8_0_t16_f16w(tiles_dev.ptr)
        assert gguf_q8_0_t16_f16w_for(tiles_dev.ptr, in_features, out_features) is None
        gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out(
            x_dev.ptr, tiles_dev.ptr, out_dev.ptr, rows, in_features, out_features,
            tile_m=32, tile_n=32, **kwargs,
        )
        outputs["auto_unregistered"] = read(out_dev)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    return outputs


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        (17, 256, 2048),    # one partial row tile
        (33, 256, 2064),    # a column count that is not a whole 64-column block
        (64, 96, 2112),     # three K blocks
        (1024, 2816, 4224),  # Gemma 4 26B-A4B's fused dense gate|up
        (512, 2112, 2816),   # ... and its dense down
        (7, 32, 2048),      # one K block, fewer rows than a tile
    ],
)
def test_f16w_wide_wave_prefill_is_bit_exact_to_the_decoding_launches(
    rows: int, in_features: int, out_features: int,
) -> None:
    """Loading the pre-converted weights changes no output bit.

    The same launch over zeroed weights writes all zeros, so the candidate
    arm did read the f16w operand; and the registered auto wrapper takes that
    route only while the weight's entry is registered for this shape.
    """

    outputs = _run_two_wave_and_f16w(rows=rows, in_features=in_features, out_features=out_features)

    np.testing.assert_array_equal(outputs["f16w"], outputs["two_wave"])
    assert np.any(outputs["two_wave"] & 0x7FFF)
    assert not np.any(outputs["zero_weights"] & 0x7FFF)
    assert not np.any(outputs["auto_registered_zero"] & 0x7FFF)
    np.testing.assert_array_equal(outputs["auto_unregistered"], outputs["two_wave"])
    np.testing.assert_array_equal(outputs["four_wave_f16w"], outputs["four_wave"])
    assert np.any(outputs["four_wave"] & 0x7FFF)
    # Every shape here is at least 2048 wide, where the four-wave default applies.
    assert not np.any(outputs["auto_4wave_registered_zero"] & 0x7FFF)
