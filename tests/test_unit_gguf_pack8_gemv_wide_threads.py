"""RED/GREEN for the DE-3 narrow-projection block-size lever.

``_pack8_gemv_decode_threads`` decides whether a decode-shaped pack8 GEMV
launches with a wider block than the launcher default.  The rule exists because
a narrow single-token projection launches ``out_features / 8`` workgroups and
cannot fill the device regardless of kernel quality; measured on gfx1100 it is
worth -32.4% at out=320 and -19.4% at out=640, and is neutral (within 2%, mixed
sign) at 2560 or more output features.

The predicate is pure Python, so these cases run without ROCm.  They pin the
gate boundaries: which shapes are eligible, that a caller-supplied block size
always wins, that prefill is untouched, and that the decode family rewrite is
exempt because it brings its own shape policy.
"""

from __future__ import annotations

import pytest

from hipengine.runtime import gguf_linear
from hipengine.runtime.gguf_linear import (
    _PACK8_GEMV_WIDE_ENV,
    _PACK8_GEMV_WIDE_MAX_OUT,
    _PACK8_GEMV_WIDE_THREADS,
    _pack8_gemv_decode_threads,
)

_DECODE_VARIANT = "pack8_gemv_f32_f32_out"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_PACK8_GEMV_WIDE_ENV, raising=False)
    yield
    monkeypatch.delenv(_PACK8_GEMV_WIDE_ENV, raising=False)


def _call(*, rows: int = 1, out_features: int, variant: str = _DECODE_VARIANT, threads: int = 0) -> int:
    return _pack8_gemv_decode_threads(
        rows=rows,
        out_features=out_features,
        variant=variant,
        threads=threads,
    )


def test_flag_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env opt-in means no behaviour change anywhere."""
    assert not gguf_linear._pack8_gemv_wide_threads_enabled()
    for out in (320, 512, 640):
        assert _call(out_features=out) == 0


def test_enabled_at_or_below_the_knee(monkeypatch: pytest.MonkeyPatch) -> None:
    """The three measured starved shapes get the wider block."""
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, "1")
    for out in (320, 512, 640):
        assert _call(out_features=out) == _PACK8_GEMV_WIDE_THREADS


def test_boundary_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knee is inclusive at the bound and exclusive one step above it."""
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, "1")
    assert _PACK8_GEMV_WIDE_MAX_OUT == 640
    assert _call(out_features=_PACK8_GEMV_WIDE_MAX_OUT) == _PACK8_GEMV_WIDE_THREADS
    assert _call(out_features=_PACK8_GEMV_WIDE_MAX_OUT + 8) == 0


def test_saturated_shapes_are_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every shape already at 90%+ of roof keeps the launcher default."""
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, "1")
    for out in (2560, 6144, 10240, 12288, 248320):
        assert _call(out_features=out) == 0


def test_caller_threads_always_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit block size from the caller is never overridden."""
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, "1")
    for threads in (64, 128, 256):
        assert _call(out_features=320, threads=threads) == 0


def test_prefill_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """rows > 1 is the prefill path and has its own column-block policy."""
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, "1")
    for rows in (2, 8, 64, 512):
        assert _call(rows=rows, out_features=320) == 0


def test_only_the_decode_shaped_pack8_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other variants are out of scope, including the decode rewrite family."""
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, "1")
    for variant in (
        "gemv_f32_f32_out",
        "gemv_bf16_bf16_out",
        "prefill_f32_f32_out",
        "wmma_prefill_f32_f32_out",
        "rowtile_f32_f32_out",
        "pack8_gemv_decode_bf16_bf16_out",
        "pack8_gemv_decode_fp16_fp16_out",
        "selected_pack8_gemv_bf16_bf16_out",
    ):
        assert _call(out_features=320, variant=variant) == 0, variant


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On"])
def test_truthy_env_spellings(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, value)
    assert _call(out_features=320) == _PACK8_GEMV_WIDE_THREADS


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2", "maybe"])
def test_falsy_env_spellings(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_PACK8_GEMV_WIDE_ENV, value)
    assert _call(out_features=320) == 0
