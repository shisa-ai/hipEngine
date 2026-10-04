"""Dual-resident Q8 projections retain the repaired T16 WMMA prefill owner."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey, _KERNELS
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _q8_t16_tiles_prefill_dispatch,
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


@pytest.mark.parametrize("rows", [2, 17, 128])
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
