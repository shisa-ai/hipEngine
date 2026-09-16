"""Dispatch contract for the wide-row Q8_0 dense prefill route.

The kernel has been registered and measured since 2026-09-16 at 6.74x the
production dispatch on the replay packet, but nothing in ``hipengine/runtime``
or ``hipengine/dispatch`` referenced it, so no dispatch path could select it and
neither the production gate nor any end-to-end measurement could reach it.
These tests pin the selector that makes it reachable, and the conditions under
which it must decline to the registered exact parent.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.registry import KernelKey, register
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _q8_dense_wide_dispatch,
)

BACKEND = "hip_gfx1151"
CANDIDATE = KernelKey(BACKEND, "linear", "gguf_q8_0", "dense_wide256_f32_f32_out")
# layers.8.attn_qkv on Qwen3.8-Flash-Next UD-Q4_K_XL: the shape the candidate's
# 6.74x replay measurement was taken on.
PACKET_K = 2560
PACKET_N = 10240


def _parent(variant: str = "coltile8_rowbatch4_wave_scale_f32_f32_out"):
    return GGUFLinearDispatch(KernelKey(BACKEND, "linear", "gguf_q8_0", variant), "raw")


@pytest.fixture(autouse=True)
def _registered_candidate():
    register(CANDIDATE, lambda *args, **kwargs: None, replace=True)


@pytest.mark.parametrize(
    "variant",
    [
        "prefill_f32_f32_out",
        "coltile8_rowbatch4_f32_f32_out",
        "coltile8_rowbatch4_wave_scale_f32_f32_out",
    ],
)
def test_selects_the_wide_kernel_from_every_exact_f32_parent(variant, monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")

    actual = _q8_dense_wide_dispatch(
        _parent(variant), rows=1024, in_features=PACKET_K, out_features=PACKET_N
    )

    assert actual == GGUFLinearDispatch(CANDIDATE, "raw")


def test_declines_when_the_route_is_not_enabled():
    """Default-off: the exact coltile parent stays the default path."""

    parent = _parent()
    assert (
        _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N
        )
        == parent
    )


@pytest.mark.parametrize("rows", [1, 4, 64, 255, 256])
def test_declines_at_and_below_the_sub_256_row_path(rows, monkeypatch):
    """The kernel stages 256 rows per block; smaller shapes stay exact."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    parent = _parent()

    assert (
        _q8_dense_wide_dispatch(
            parent, rows=rows, in_features=PACKET_K, out_features=PACKET_N
        )
        == parent
    )


@pytest.mark.parametrize("in_features", [2528, 2560 - 32, 96 + 1, 0, -64])
def test_declines_when_k_is_not_a_positive_multiple_of_the_k_tile(
    in_features, monkeypatch
):
    """``_launch`` raises on in_features % 64; the selector must not reach it."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    parent = _parent()

    assert (
        _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=in_features, out_features=PACKET_N
        )
        == parent
    )


def test_declines_for_a_non_q8_0_quant(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    parent = GGUFLinearDispatch(
        KernelKey(BACKEND, "linear", "gguf_q4_k", "coltile8_rowbatch4_f32_f32_out"),
        "raw",
    )

    assert (
        _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N
        )
        == parent
    )


def test_declines_for_a_non_raw_abi(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    parent = GGUFLinearDispatch(
        KernelKey(BACKEND, "linear", "gguf_q8_0", "coltile8_rowbatch4_f32_f32_out"),
        "t16",
    )

    assert (
        _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N
        )
        == parent
    )


def test_declines_when_the_candidate_is_not_registered(monkeypatch):
    """Fail closed to the registered exact parent rather than to a missing key."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setattr(
        "hipengine.runtime.gguf_linear.is_registered", lambda key: False
    )
    monkeypatch.setattr(
        "hipengine.runtime.gguf_linear._ensure_linear_kernel_registered",
        lambda key: None,
    )
    parent = _parent()

    assert (
        _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N
        )
        == parent
    )


def test_variant_scoped_library_beats_the_quant_default():
    """dense_wide has its own shared object; the quant default is the wrong one.

    ``dense_wide256`` is exported from ``gguf_q8_0_dense_wide.so`` while the
    coltile family lives in the q8_0 gemv library, so a caller-supplied map
    keyed only by quant would hand the wide kernel a library that does not
    export its symbol.
    """

    from hipengine.runtime.gguf_linear import _variant_scoped_library

    wide_library = object()
    quant_default = object()
    libraries = {
        "gguf_q8_0": quant_default,
        "gguf_q8_0:dense_wide256_f32_f32_out": wide_library,
    }

    assert _variant_scoped_library(libraries, CANDIDATE) is wide_library
    # The coltile parent has no variant-scoped entry and keeps the quant default.
    assert (
        _variant_scoped_library(
            libraries,
            KernelKey(BACKEND, "linear", "gguf_q8_0", "coltile8_rowbatch4_f32_f32_out"),
        )
        is quant_default
    )
    assert _variant_scoped_library(None, CANDIDATE) is None


def test_the_wide_selector_is_wired_into_the_launch_chain():
    """A selector nothing calls is what left this kernel unreachable before."""

    import inspect

    from hipengine.runtime import gguf_linear

    chain = inspect.getsource(gguf_linear.launch_gguf_linear)
    assert "_q8_dense_wide_dispatch(" in chain, (
        "launch_gguf_linear must run the wide-row selector; Qwen4Exp prefill "
        "reaches its Q8_0 linears through this entry point"
    )
