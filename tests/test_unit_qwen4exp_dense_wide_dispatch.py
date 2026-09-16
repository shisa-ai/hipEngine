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


def _wmma_claimed_parent(rows: int = 512, in_features: int = PACKET_K):
    """Run the rewrite the launch chain runs before the wide selector.

    ``_wmma_prefill_dispatch`` sits earlier in ``launch_gguf_linear`` than
    ``_q8_dense_wide_dispatch``, so this is the dispatch the wide selector
    actually receives on any weight the WMMA route claims.
    """

    from hipengine.runtime.gguf_linear import _wmma_prefill_dispatch

    claimed = _wmma_prefill_dispatch(
        _parent("prefill_f32_f32_out"),
        rows=rows,
        in_features=in_features,
        use_wmma=True,
    )
    assert claimed.key.variant == "wmma_prefill_f32_f32_out"
    assert claimed.abi == "wmma_raw"
    return claimed


def test_claims_the_weight_the_wmma_rewrite_already_claimed(monkeypatch):
    """The WMMA rewrite runs first and changes the ABI out from under this route.

    ``HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS`` is bound to layers 16-47 for this
    quant by the production default, which is the same window this route is
    scoped to. Every Q8_0 dense linear in that window therefore reaches the
    wide selector as ``wmma_prefill_f32_f32_out`` with ABI ``wmma_raw``.
    Declining that parent is what left the route unreachable while its own
    gates passed, its layer scope was correct and its family was registered:
    the selector returned before any of that was consulted.
    """

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv(
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS",
        ",".join(str(n) for n in range(16, 48)),
    )

    actual = _q8_dense_wide_dispatch(
        _wmma_claimed_parent(),
        rows=512,
        in_features=PACKET_K,
        out_features=PACKET_N,
        weight=_weight("layers.20.attn_qkv"),
    )

    assert actual == GGUFLinearDispatch(CANDIDATE, "raw")


def test_an_unclaimable_parent_is_named_rather_than_ignored(monkeypatch):
    """The silent decline is what hid this route; the diagnostic must name it."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    parent = GGUFLinearDispatch(
        KernelKey(
            BACKEND,
            "linear",
            "gguf_q8_0",
            "iu8_wmma_prefill_f32_f32_out",
        ),
        "raw",
    )

    with pytest.warns(RuntimeWarning, match="iu8_wmma_prefill_f32_f32_out"):
        assert (
            _q8_dense_wide_dispatch(
                parent,
                rows=512,
                in_features=PACKET_K,
                out_features=PACKET_N,
                weight=_weight("layers.20.attn_qkv"),
            )
            == parent
        )


def test_a_wmma_claim_does_not_widen_the_layer_scope(monkeypatch):
    """Accepting the rewritten parent must not hand the route an unscoped window."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", "16,17,18")
    parent = _wmma_claimed_parent()

    for layer in (0, 8, 15, 47):
        assert (
            _q8_dense_wide_dispatch(
                parent,
                rows=512,
                in_features=PACKET_K,
                out_features=PACKET_N,
                weight=_weight(f"layers.{layer}.attn_qkv"),
            )
            == parent
        ), layer


def test_the_wide_selector_is_wired_into_the_launch_chain():
    """A selector nothing calls is what left this kernel unreachable before."""

    import inspect

    from hipengine.runtime import gguf_linear

    chain = inspect.getsource(gguf_linear.launch_gguf_linear)
    assert "_q8_dense_wide_dispatch(" in chain, (
        "launch_gguf_linear must run the wide-row selector; Qwen4Exp prefill "
        "reaches its Q8_0 linears through this entry point"
    )


def test_the_wide_selector_claims_a_parent_the_wmma_rewrite_produced():
    """The rewrite runs earlier in the chain, so the parent set must cover it.

    A route that only accepts the pre-rewrite parent is unreachable wherever
    the WMMA route is enabled, which is the entire certified layer window.
    """

    import inspect

    from hipengine.runtime import gguf_linear

    chain = inspect.getsource(gguf_linear.launch_gguf_linear)
    assert chain.index("_wmma_prefill_dispatch(") < chain.index(
        "_q8_dense_wide_dispatch("
    ), "the WMMA rewrite must precede the wide selector for this contract to hold"
    assert gguf_linear._q8_dense_wide_claims_parent(
        GGUFLinearDispatch(
            KernelKey(BACKEND, "linear", "gguf_q8_0", "wmma_prefill_f32_f32_out"),
            "wmma_raw",
        )
    )


def _weight(slot_path: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        spec=SimpleNamespace(slot_path=slot_path, quant_key="gguf_q8_0")
    )


def test_unscoped_route_covers_every_layer(monkeypatch):
    """No scope means all 48 layers, which is the scope the f16 sibling fails."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", raising=False)

    for layer in (0, 15, 16, 47):
        actual = _q8_dense_wide_dispatch(
            _parent(),
            rows=1024,
            in_features=PACKET_K,
            out_features=PACKET_N,
            weight=_weight(f"layers.{layer}.attn_qkv"),
        )
        assert actual == GGUFLinearDispatch(CANDIDATE, "raw"), layer


def test_layer_scope_narrows_the_route_to_the_certified_window(monkeypatch):
    """A gate must be able to ask for 16-47 rather than 0-47.

    The sibling f16 route fails the calibrated envelope at 0-47 and passes at
    16-47, so a route with no scope can only express the failing configuration.
    """

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv(
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS",
        ",".join(str(n) for n in range(16, 48)),
    )

    inside = _q8_dense_wide_dispatch(
        _parent(), rows=1024, in_features=PACKET_K, out_features=PACKET_N,
        weight=_weight("layers.16.attn_qkv"),
    )
    assert inside == GGUFLinearDispatch(CANDIDATE, "raw")

    parent = _parent()
    for layer in (0, 8, 15):
        assert (
            _q8_dense_wide_dispatch(
                parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N,
                weight=_weight(f"layers.{layer}.attn_qkv"),
            )
            == parent
        ), layer


@pytest.mark.parametrize(
    "slot_path", ["token_embd", "output", "output_hc_down", "not.a.layer"]
)
def test_a_scoped_route_declines_a_weight_it_cannot_place(slot_path, monkeypatch):
    """Fail closed: a weight with no layer index is outside any layer scope.

    token_embd and output are Q8_0 in this model but sit outside the blocks, so
    a scoped route must not silently claim them.
    """

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", "16,17,18")
    parent = _parent()

    assert (
        _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N,
            weight=_weight(slot_path),
        )
        == parent
    )
    # Unscoped, the same weight is claimed: the decline is the scope, not the path.
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS")
    assert _q8_dense_wide_dispatch(
        parent, rows=1024, in_features=PACKET_K, out_features=PACKET_N,
        weight=_weight(slot_path),
    ) == GGUFLinearDispatch(CANDIDATE, "raw")
