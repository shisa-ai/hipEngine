"""Routing tests for the D9 rows==1 Q8_0 -> Q8T16 tiles decode rewrite.

A Gemma-4 Q8_0 dense projection ships a byte-neutral ``tiles`` allocation
beside its raw blocks (planned in ``gemma4_gguf_device``). These tests pin
the rewrite in :func:`hipengine.runtime.gguf_linear._q8_t16_tiles_decode_dispatch`
to that signal alone: rows==1 + Q8_0 + tiles present + the legacy ``pack8_gemv``
alias -> the registered ``t16_gemv_decode`` sibling under the tiles quant key.
Everything else -- no tiles, rows many, another quant, an explicit
``pack8_gemv_decode`` opt-in, a non-tile shape -- keeps the dispatch it had.
"""

from __future__ import annotations

from types import SimpleNamespace

# Real kernel module imports keep the registry populated across tests.
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, resolve, is_registered
from hipengine.kernels.registry import _KERNELS
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime import gguf_linear
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    GGUF_OUTPUT_BF16,
    _q8_t16_tiles_decode_dispatch,
    launch_gguf_linear,
)

_Q8_T16_DECODE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"
)


def _base_dispatch(variant: str = "pack8_gemv_bf16_bf16_out", quant: str = "gguf_q8_0"):
    return GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", quant, variant), "raw"
    )


class _Weight:
    def __init__(self, *, tiles: bool) -> None:
        self.spec = SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
        self._tiles = tiles

    def has_allocation(self, name: str) -> bool:
        return name == "tiles" and self._tiles

    def allocation(self, name: str | None = "raw"):
        if name == "tiles" and self._tiles:
            return SimpleNamespace(tensor=SimpleNamespace(ptr=14))
        if name in (None, "raw"):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=10))
        raise KeyError(name)


def _rewrite(dispatch, *, tiles: bool, rows: int = 1, in_f: int = 2816, out_f: int = 4096):
    return _q8_t16_tiles_decode_dispatch(
        dispatch,
        weight=_Weight(tiles=tiles),
        rows=rows,
        in_features=in_f,
        out_features=out_f,
    )


def test_t16_target_is_registered() -> None:
    """The rewrite's target sibling exists for the fixture's default route."""

    assert is_registered(_Q8_T16_DECODE)


def test_rows1_q8_with_tiles_routes_to_the_t16_sibling() -> None:
    out = _rewrite(_base_dispatch(), tiles=True)
    assert out.key == _Q8_T16_DECODE
    assert out.abi == "t16"


def test_no_registered_sibling_for_the_output_dtype_declines() -> None:
    """Capability, not blanket rewrite: ``bf16_f32_out`` has no t16 sibling
    registered, so the dispatch stays on the pack8 alias."""

    base = _base_dispatch("pack8_gemv_bf16_f32_out")
    out = _rewrite(base, tiles=True, out_f=2048)
    assert out is base


def test_without_tiles_keeps_the_pack8_alias() -> None:
    base = _base_dispatch()
    out = _rewrite(base, tiles=False)
    assert out is base


def test_rows_many_keeps_prefill_dispatch() -> None:
    base = _base_dispatch()
    out = _rewrite(base, tiles=True, rows=8)
    assert out is base


def test_other_quant_keeps_its_dispatch_even_with_tiles() -> None:
    base = _base_dispatch("pack8_gemv_bf16_bf16_out", quant="gguf_q5_k")
    out = _rewrite(base, tiles=True)
    assert out is base


def test_explicit_gemv_decode_opt_in_is_not_overridden() -> None:
    """The P9.B3 ``pack8_gemv_decode`` opt-in is a user choice, not the alias."""

    base = _base_dispatch("pack8_gemv_decode_bf16_bf16_out")
    out = _rewrite(base, tiles=True)
    assert out is base


def test_pre_pack8_base_variant_is_left_alone() -> None:
    """The rewrite sees the chain after the pack8 alias, so a raw ``gemv_*``
    variant reaching it directly (registration missing) must not match."""

    base = _base_dispatch("gemv_bf16_bf16_out")
    out = _rewrite(base, tiles=True)
    assert out is base


def test_shape_that_is_not_tile16_keeps_the_pack8_alias() -> None:
    base = _base_dispatch()
    assert _rewrite(base, tiles=True, out_f=1000) is base
    assert _rewrite(base, tiles=True, in_f=1000) is base


def _capture_launch(*, tiles: bool, rows: int, in_f: int = 2816, out_f: int = 4096):
    weight = _Weight(tiles=tiles)
    captured: dict[str, object] = {"key": None, "args": None, "kwargs": None}
    keys = (
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out"),
        _Q8_T16_DECODE,
    )
    originals = {
        k: resolve(
            backend=k.backend,
            layer=k.layer,
            quant=k.quant,
            variant=k.variant,
            missing="none",
        )
        for k in keys
    }

    def make_fake(key: KernelKey):
        def fake(*args, **kwargs):
            captured["key"] = key
            captured["args"] = args
            captured["kwargs"] = kwargs

        return fake

    try:
        for k in keys:
            register(k, make_fake(k), replace=True)
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=in_f,
            out_features=out_f,
            output_dtype=GGUF_OUTPUT_BF16,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        for k, fn in originals.items():
            if fn is None:
                _KERNELS.pop(k, None)
            else:
                register(k, fn, replace=True)
    return captured


def test_launch_rows1_with_tiles_reads_the_tiles_allocation() -> None:
    """End to end: the launch key is the t16 sibling and the weight operand
    is the tiles pointer, not the raw one."""

    gguf_linear.clear_gguf_linear_dispatch_cache()
    captured = _capture_launch(tiles=True, rows=1)
    assert captured["key"] == _Q8_T16_DECODE
    # t16 ABI: (x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features)
    assert captured["args"] == (100, 14, 200, 1, 2816, 4096)


def test_launch_rows1_without_tiles_reads_the_raw_allocation() -> None:
    gguf_linear.clear_gguf_linear_dispatch_cache()
    captured = _capture_launch(tiles=False, rows=1)
    assert captured["key"] == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out"
    )
    assert captured["args"] == (100, 10, 200, 1, 2816, 4096)