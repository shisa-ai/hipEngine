"""Prefill expert-route selection for Gemma 4 MoE (unit tier).

``HIPENGINE_GEMMA4_MOE_PREFILL`` pins which prefill owner the MoE projections
use. ``auto`` is the production default and pins no owner, letting the forward
pass resolve the default-on levers (fused T16 WMMA gate/up + down). ``grouped``
and ``selected`` pin the exact routes -- both measured bit-identical to the
strict teacher-forced reference -- and stay reachable as rollback levers.
``mmq`` pins the grouped int8 family.

The compensated WMMA candidate (former ``wmma`` / ``wmma_plain`` modes) was
retired: on the T16-converted layout its pin disabled the faster fused default
and fell back to selected_gemv, a ~4x prefill regression on gfx1151 and gfx1100,
and the compensated owner never engaged. Those values are no longer recognized
modes and resolve to ``auto``. The plain WMMA down owner survives only as the
fallback behind the T16 down owner, so its variant stays registered.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    _PREFILL_MODE_ENV,
    _PREFILL_MODES,
    _WMMA_PREFILL_VARIANT,
    _prefill_mode,
    _prefill_route_flags,
)


def test_auto_selects_the_production_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_PREFILL_MODE_ENV, raising=False)
    assert _prefill_mode() == "auto"
    assert _prefill_route_flags("auto") == (False, False, False)


def test_pinned_exact_modes_select_no_wmma_or_mmq() -> None:
    assert _prefill_route_flags("grouped") == (False, False, False)
    assert _prefill_route_flags("selected") == (False, False, False)


def test_retired_wmma_modes_are_not_selectable() -> None:
    # The compensated candidate is retired: its former pins are not modes, and
    # no route flag combination asks for a WMMA gate_up or a compensated owner.
    assert "wmma" not in _PREFILL_MODES
    assert "wmma_plain" not in _PREFILL_MODES
    for mode in _PREFILL_MODES:
        use_wmma, compensated, _ = _prefill_route_flags(mode)
        assert not use_wmma and not compensated


def test_retired_wmma_env_values_resolve_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("wmma", "wmma_plain", "WMMA"):
        monkeypatch.setenv(_PREFILL_MODE_ENV, value)
        assert _prefill_mode() == "auto"


def test_mmq_mode_selects_only_the_grouped_int8_owner() -> None:
    # The MMQ leaf replaces the BF16 gate_up owners rather than supplementing
    # them, so it must not also ask for the WMMA route.
    assert _prefill_route_flags("mmq") == (False, False, True)
    assert "mmq" in _PREFILL_MODES


def test_unknown_selector_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PREFILL_MODE_ENV, "not-a-route")
    assert _prefill_mode() == "auto"


def test_selector_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PREFILL_MODE_ENV, "MMQ")
    assert _prefill_mode() == "mmq"


def test_plain_wmma_down_owner_variant_is_registered() -> None:
    # The plain WMMA down owner remains the fallback behind the T16 down owner on
    # the unpinned/mmq routes, so its variant must stay resolvable.
    from hipengine.kernels.registry import KernelKey, is_registered

    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        register_qwen4_exp_q5_1_kernels,
    )

    register_qwen4_exp_q5_1_kernels()
    assert is_registered(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", _WMMA_PREFILL_VARIANT)
    )