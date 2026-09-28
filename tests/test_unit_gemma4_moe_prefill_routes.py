"""Prefill expert-route selection for Gemma 4 MoE (unit tier).

``HIPENGINE_GEMMA4_MOE_PREFILL`` pins which prefill owner the MoE projections
use. ``auto`` is the production default and selects the grouped int8 MMQ
route, which ``gemma4_experts`` documents as the production path measured at
the campaign recipe. ``grouped`` and ``selected`` pin the exact routes instead --
both measured bit-identical to the strict teacher-forced reference, which is
what the campaign logits gate requires -- and stay reachable as the rollback
levers.

The WMMA owners are 2.7x faster on a 1024-token prefill but breach the
campaign's absolute ``kl_max`` bar on 1 of 1023 rows, so they stay off the
default path until that bar's applicability to a reordering-class change is
ruled on. ``wmma`` selects the compensated form and ``wmma_plain`` the
uncompensated form that isolates the fp16 weight-rounding term.

The registration half of this file is what keeps those variants resolvable: the
mode selector can only pick a compensated owner if the compensated symbol is
actually registered under the variant name the selector builds.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    _PREFILL_MODE_ENV,
    _PREFILL_MODES,
    _WMMA_DUAL_PREFILL_COMP_VARIANT,
    _WMMA_DUAL_PREFILL_VARIANT,
    _WMMA_PREFILL_COMP_VARIANT,
    _WMMA_PREFILL_VARIANT,
    _prefill_mode,
    _prefill_route_flags,
)


def test_auto_selects_the_production_route(monkeypatch: pytest.MonkeyPatch) -> None:
    # `auto` has been the grouped int8 MMQ route since that route became the
    # production default. The exact routes are selected by pinning them, which
    # the next test covers.
    monkeypatch.delenv(_PREFILL_MODE_ENV, raising=False)
    assert _prefill_mode() == "auto"
    assert _prefill_route_flags("auto") == (False, False, True)


def test_pinned_exact_modes_never_probe_wmma() -> None:
    assert _prefill_route_flags("grouped") == (False, False, False)
    assert _prefill_route_flags("selected") == (False, False, False)


def test_wmma_modes_select_compensated_and_plain() -> None:
    assert _prefill_route_flags("wmma") == (True, True, False)
    assert _prefill_route_flags("wmma_plain") == (True, False, False)


def test_mmq_mode_selects_only_the_grouped_int8_owner() -> None:
    # The MMQ leaf replaces the BF16 gate_up owners rather than supplementing
    # them, so it must not also ask for the WMMA route.
    assert _prefill_route_flags("mmq") == (False, False, True)
    assert "mmq" in _PREFILL_MODES


def test_unknown_selector_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PREFILL_MODE_ENV, "not-a-route")
    assert _prefill_mode() == "auto"


def test_selector_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PREFILL_MODE_ENV, "WMMA")
    assert _prefill_mode() == "wmma"


def test_compensated_variants_are_distinct_from_the_plain_ones() -> None:
    # A selector that built the plain variant name while claiming to compensate
    # would silently measure the uncompensated arithmetic.
    assert _WMMA_DUAL_PREFILL_COMP_VARIANT != _WMMA_DUAL_PREFILL_VARIANT
    assert _WMMA_PREFILL_COMP_VARIANT != _WMMA_PREFILL_VARIANT
    assert _WMMA_DUAL_PREFILL_COMP_VARIANT.endswith("_comp_bf16_bf16_out")
    assert _WMMA_PREFILL_COMP_VARIANT.endswith("_comp_bf16_bf16_out")


def test_compensated_q4_k_dual_owner_is_registered() -> None:
    from hipengine.kernels.registry import KernelKey, is_registered

    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        register_gguf_q4_k_selected_prefill_kernels,
    )

    register_gguf_q4_k_selected_prefill_kernels()
    for variant in (_WMMA_DUAL_PREFILL_VARIANT, _WMMA_DUAL_PREFILL_COMP_VARIANT):
        assert is_registered(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k", variant)
        ), variant


def test_compensated_q5_1_grouped_owner_is_registered() -> None:
    from hipengine.kernels.registry import KernelKey, is_registered

    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        register_qwen4_exp_q5_1_kernels,
    )

    register_qwen4_exp_q5_1_kernels()
    for variant in (_WMMA_PREFILL_VARIANT, _WMMA_PREFILL_COMP_VARIANT):
        assert is_registered(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", variant)
        ), variant
