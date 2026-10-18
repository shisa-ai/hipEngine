"""Dual-resident Q8 projections retain the repaired T16 WMMA prefill owner."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey, _KERNELS
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _q8_t16_tiles_prefill_dispatch,
    _q8_t16_tiles_small_rows_dispatch,
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear,
)


@pytest.mark.parametrize("rows", [2, 4, 17, 128])
@pytest.mark.parametrize("tiles", [False, True])
def test_dual_resident_wmma_uses_t16_when_available(monkeypatch, rows, tiles):
    target = KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_wmma_prefill_bf16_bf16_out")
    monkeypatch.setitem(_KERNELS, target, lambda *a, **kw: None)
    base = GGUFLinearDispatch(KernelKey("hip_gfx1100", "linear", "gguf_q8_0",
                                       "wmma_prefill_bf16_bf16_out"), "wmma_raw")
    weight = SimpleNamespace(has_allocation=lambda name: tiles and name == "tiles")
    got = _q8_t16_tiles_prefill_dispatch(base, weight=weight, rows=rows,
                                        in_features=256, out_features=256)
    if tiles:
        assert got.key == target
        assert got.abi == "t16"
    else:
        assert got is base


@pytest.mark.parametrize("rows", [9, 17, 128])
def test_launch_dual_resident_prefill_reads_tiles_after_wmma_selection(monkeypatch, rows):
    from tests.test_unit_gguf_q8_t16_tiles_decode_dispatch import _Weight

    target = KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1",
                       "t16_wmma_prefill_bf16_bf16_out")
    calls = []
    monkeypatch.setitem(_KERNELS, target, lambda *a, **kw: calls.append(a))
    clear_gguf_linear_dispatch_cache()
    launch_gguf_linear(
        _Weight(tiles=True), x_ptr=100, out_ptr=200, rows=rows,
        in_features=256, out_features=256, use_wmma_prefill=True,
        runtime="fake-runtime", stream=7,
    )
    assert calls == [(100, 14, 200, rows, 256, 256)]


_GEMV = KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out")
_RAW_PREFILL = GGUFLinearDispatch(
    KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out"), "raw")
_T16_WMMA = GGUFLinearDispatch(
    KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_wmma_prefill_bf16_bf16_out"), "t16")


def _small_rows(dispatch, *, rows, out_features=2816, tiles=True, quant_key="gguf_q8_0"):
    from tests.test_unit_gguf_q8_t16_tiles_decode_dispatch import _Weight

    weight = _Weight(tiles=tiles)
    weight.spec.quant_key = quant_key
    return _q8_t16_tiles_small_rows_dispatch(
        dispatch, weight=weight, rows=rows, in_features=4096, out_features=out_features)


@pytest.mark.parametrize("rows", [2, 3, 5, 8])
@pytest.mark.parametrize("source", [_RAW_PREFILL, _T16_WMMA])
def test_small_blocks_take_the_one_row_gemv(rows, source):
    got = _small_rows(source, rows=rows)
    assert got.key == _GEMV
    assert got.abi == "t16"


@pytest.mark.parametrize("rows", [1, 9, 64])
def test_rows_outside_two_to_eight_keep_their_route(rows):
    assert _small_rows(_RAW_PREFILL, rows=rows) is _RAW_PREFILL
    assert _small_rows(_T16_WMMA, rows=rows) is _T16_WMMA


@pytest.mark.parametrize("rows, expected", [(2, _GEMV), (4, _GEMV), (5, None), (8, None)])
def test_a_wide_output_keeps_wmma_above_four_rows(rows, expected):
    got = _small_rows(_T16_WMMA, rows=rows, out_features=8192)
    if expected is None:
        assert got is _T16_WMMA
    else:
        assert got.key == expected
    # Without WMMA (the exact verify) the GEMV serves every width.
    assert _small_rows(_RAW_PREFILL, rows=rows, out_features=8192).key == _GEMV


def test_only_dual_resident_raw_q8_0_is_routed():
    assert _small_rows(_RAW_PREFILL, rows=5, tiles=False) is _RAW_PREFILL
    assert _small_rows(_RAW_PREFILL, rows=5, quant_key="gguf_q8_0_t16_v1") is _RAW_PREFILL
    other = GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out"), "raw")
    assert _small_rows(other, rows=5) is other


@pytest.mark.parametrize("use_wmma", [False, True])
def test_launch_runs_a_small_block_on_the_gemv(monkeypatch, use_wmma):
    from tests.test_unit_gguf_q8_t16_tiles_decode_dispatch import _Weight

    calls = []
    monkeypatch.setitem(_KERNELS, _GEMV, lambda *a, **kw: calls.append(a))
    clear_gguf_linear_dispatch_cache()
    launch_gguf_linear(
        _Weight(tiles=True), x_ptr=100, out_ptr=200, rows=5,
        in_features=256, out_features=256, use_wmma_prefill=use_wmma,
        runtime="fake-runtime", stream=7,
    )
    assert calls == [(100, 14, 200, 5, 256, 256)]
