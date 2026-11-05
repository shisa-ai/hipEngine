"""Reachability matrix for the staged-F32 dense Q8_0 evaluation interface.

The interface is opt-in and layer-scoped, but its selector only replaces an
*exact* Q8/F32 dense parent. Two earlier selectors own those shapes by default:

- the 2-8 row rowtile rewrite, which replaces ``prefill_f32_f32_out`` with
  ``rowtile_f32_f32_out``;
- the DPP pipelined leaf, which replaces the wave-scale pipelined parent at
  rows above the rowtile band.

This module pins that reachability at the dispatch, without a GPU. It is the
CPU half of ``tests/test_live_qwen4exp_staged_f32_interface.py``: the live test
stands both preemptors down explicitly and proves the public path engages.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import gguf_linear

STAGED = gguf_linear._Q8_STAGED_F32_VARIANT
DPP_ENV = "HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION"
EXACT_PARENTS = gguf_linear._Q8_DENSE_WIDE_EXACT_PARENTS | {
    "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out",
}


class _Spec:
    layout = "raw_gguf"
    quant_key = "gguf_q8_0"

    def __init__(self, layer: int) -> None:
        self.slot_path = f"layers.{layer}.attn_qkv.weight"


class _Allocation:
    class tensor:  # noqa: N801 - mirrors the resident-weight attribute path
        ptr = 4096


class _Weight:
    backend = "hip_gfx1151"

    def __init__(self, layer: int) -> None:
        self.spec = _Spec(layer)

    @staticmethod
    def allocation(name: str) -> _Allocation:
        assert name == "raw"
        return _Allocation()


class _Recorded(Exception):
    """Stop the launch at the final-dispatch hook, before any device call."""


@pytest.fixture
def dispatch_cache():
    saved = dict(gguf_linear._DISPATCH_RESOLVE_CACHE)
    gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
    try:
        yield gguf_linear._DISPATCH_RESOLVE_CACHE
    finally:
        gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
        gguf_linear._DISPATCH_RESOLVE_CACHE.update(saved)


def _final_variant(
    monkeypatch: pytest.MonkeyPatch,
    cache: dict,
    *,
    rows: int,
    layer: int = 0,
    staged: str | None = "0",
    rowtile: bool = True,
    dpp: bool = False,
) -> str:
    if staged is None:
        monkeypatch.delenv(gguf_linear._Q8_STAGED_F32_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(gguf_linear._Q8_STAGED_F32_LAYERS_ENV, staged)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", "")
    # Opposing retired input must not override the explicit execution session.
    monkeypatch.setenv(DPP_ENV, "0" if dpp else "1")
    cache.clear()
    recorded: list[str] = []

    def stop(**kwargs):
        recorded.append(kwargs["variant"])
        raise _Recorded

    monkeypatch.setattr(gguf_linear, "_record_gguf_linear_dispatch", stop)
    with (
        gguf_linear.raw_k_prefill_rowbatch_session(32),
        gguf_linear.raw_k_prefill_variant_session("coltile8_wave_scale"),
        gguf_linear.q4k_rowtile_session(rowtile),
        gguf_linear.q8_dpp_reduction_session(dpp),
    ):
        try:
            gguf_linear.launch_gguf_linear(
                _Weight(layer),
                0,
                0,
                rows,
                2560,
                10240,
                activation_dtype=gguf_linear.GGUF_ACTIVATION_F32,
                output_dtype=gguf_linear.GGUF_OUTPUT_F32,
                libraries={},
            )
        except _Recorded:
            pass
    assert recorded, "the dispatch never reached the final-dispatch hook"
    return recorded[0]


@pytest.mark.parametrize(
    "rows,rowtile,dpp,staged,layer,expected",
    [
        # A one-row prefill is decode-shaped: neither the staged selector nor
        # either preemptor's prefill path is reachable.
        (1, True, True, "0", 0, "decode"),
        (1, False, False, "0", 0, "decode"),
        # The 2-8 row band is owned by the rowtile rewrite. Standing it down
        # exposes the exact parent the staged selector accepts.
        (2, True, True, "0", 0, "rowtile"),
        (2, False, True, "0", 0, STAGED),
        (8, True, True, "0", 0, "rowtile"),
        (8, False, True, "0", 0, STAGED),
        # Above the rowtile band the DPP pipelined leaf owns the exact chain.
        # Standing DPP down exposes the non-DPP pipelined parent.
        (9, True, True, "0", 0, "dpp"),
        (9, True, False, "0", 0, STAGED),
        (73, True, True, "0", 0, "dpp"),
        (73, True, False, "0", 0, STAGED),
        (512, True, True, "0", 0, "dpp"),
        (512, True, False, "0", 0, STAGED),
        # Empty/unset is off, and ``0`` is layer zero rather than every layer.
        (73, True, False, "", 0, "exact"),
        (73, True, False, None, 0, "exact"),
        (73, True, False, "0", 1, "exact"),
    ],
)
def test_staged_selector_reachability(
    monkeypatch: pytest.MonkeyPatch,
    dispatch_cache: dict,
    rows: int,
    rowtile: bool,
    dpp: bool,
    staged: str | None,
    layer: int,
    expected: str,
) -> None:
    register_gfx1151_kernels(replace=True)
    assert is_registered(
        KernelKey("hip_gfx1151", "linear", "gguf_q8_0", STAGED)
    )

    variant = _final_variant(
        monkeypatch,
        dispatch_cache,
        rows=rows,
        layer=layer,
        staged=staged,
        rowtile=rowtile,
        dpp=dpp,
    )

    assert is_registered(
        KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant)
    ), variant
    if expected == STAGED:
        assert variant == STAGED
    elif expected == "rowtile":
        assert variant == "rowtile_f32_f32_out"
    elif expected == "dpp":
        assert "pipelined_dpp" in variant, variant
    elif expected == "decode":
        assert variant.startswith("pack8_gemv"), variant
    else:
        assert variant != STAGED
        assert variant in EXACT_PARENTS, variant
