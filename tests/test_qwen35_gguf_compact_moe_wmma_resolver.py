"""Unit coverage for the P10 compact selected-MoE WMMA prefill resolver.

P10.B1 wires the existing Q4T16 selected dual WMMA prefill kernel into
``_COMPACT_MOE_Q4_DUAL_KEYS`` so the bulk prefill path can dispatch a real
WMMA kernel when ``HIPENGINE_GGUF_DECODE_REPACK=1`` materializes the gate /
up experts as Q4T16 tiles. The resolver mirrors the GEMV decode resolver
and now also carries the per-weight allocation name (``"raw"`` vs
``"tiles"``) so the caller does not need a quant branch.

These tests cover the resolver only: end-to-end dispatch into the runner
is exercised in ``tests/test_qwen35_gguf_compact_moe_wmma_routing.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen35_gguf_runner as qgr


def _fake_weight(quant_key: str, *, backend: str = "hip_gfx1100") -> object:
    return SimpleNamespace(spec=SimpleNamespace(quant_key=quant_key), backend=backend)


def test_raw_q4_q5_resolves_returns_plan_with_raw_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _fake_weight("gguf_q4_k")
    up = _fake_weight("gguf_q4_k")
    down = _fake_weight("gguf_q5_k")
    monkeypatch.setattr(
        qgr,
        "resolve",
        lambda *, backend, layer, quant, variant, missing="error": lambda: f"{quant}.{variant}",
    )

    plan = qgr._resolve_compact_moe_wmma_kernels(gate, up, down)

    assert plan is not None
    assert plan.gate_allocation == "raw"
    assert plan.up_allocation == "raw"
    assert plan.down_allocation == "raw"
    assert callable(plan.gate_up_fn)
    assert callable(plan.down_fn)


@pytest.mark.parametrize("down_quant", ["gguf_q5_k_t16_v1", "gguf_q6_k_t16_v1"])
def test_t16_q4_q5_q6_resolves_returns_plan_with_tiles_allocations(
    monkeypatch: pytest.MonkeyPatch, down_quant: str
) -> None:
    """P10.B1+B2+B3 wired: Q4T16/Q4T16 gate+up plus Q5T16 or Q6T16 down resolves.

    Once all three keys are present in their respective dispatch tables,
    the resolver returns a plan whose allocation names are ``"tiles"`` so
    the caller picks up the resident T16 byte-lossless layout.
    """

    gate = _fake_weight("gguf_q4_k_t16_v1")
    up = _fake_weight("gguf_q4_k_t16_v1")
    down = _fake_weight(down_quant)
    monkeypatch.setattr(
        qgr,
        "resolve",
        lambda *, backend, layer, quant, variant, missing="error": lambda: f"{quant}.{variant}",
    )

    plan = qgr._resolve_compact_moe_wmma_kernels(gate, up, down)

    assert plan is not None
    assert plan.gate_allocation == "tiles"
    assert plan.up_allocation == "tiles"
    assert plan.down_allocation == "tiles"
    assert callable(plan.gate_up_fn)
    assert callable(plan.down_fn)


def test_t16_q4_dispatch_key_points_to_registered_t16_alias() -> None:
    """P10.B1: the T16 entry resolves to the Q4T16 prefill alias spelling.

    ``register_gguf_q4_k_t16_selected_prefill_kernels`` registers the
    Q4T16 dual WMMA kernel under both ``selected_dual_wmma_prefill_compact32_*``
    (its native spelling) and the shorter ``selected_dual_wmma_prefill_compact_*``
    alias that the raw-Q4 dispatch already uses, so the new
    ``_COMPACT_MOE_Q4_DUAL_KEYS`` entry can route on quant key alone.
    """

    key = qgr._COMPACT_MOE_Q4_DUAL_KEYS[("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1")]
    assert key.backend == "hip_gfx1100"
    assert key.layer == "moe_linear"
    assert key.quant == "gguf_q4_k_t16_v1"
    assert key.variant == "selected_dual_wmma_prefill_compact_bf16_bf16_out"


def test_t16_q4_explicit_shared_x_mode_resolves_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", "shared_x")
    resolved_variants: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, missing="error"):
        resolved_variants.append(variant)
        return lambda: None

    monkeypatch.setattr(qgr, "resolve", fake_resolve)
    plan = qgr._resolve_compact_moe_wmma_kernels(
        _fake_weight("gguf_q4_k_t16_v1", backend="hip_gfx1151"),
        _fake_weight("gguf_q4_k_t16_v1", backend="hip_gfx1151"),
        _fake_weight("gguf_q5_k_t16_v1", backend="hip_gfx1151"),
    )

    assert plan is not None
    assert "selected_dual_wmma_prefill_compact32_shared_x_bf16_bf16_out" in resolved_variants


def test_t16_q4_gfx1151_auto_resolves_baseline_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", raising=False)
    gate = _fake_weight("gguf_q4_k_t16_v1", backend="hip_gfx1151")
    up = _fake_weight("gguf_q4_k_t16_v1", backend="hip_gfx1151")
    down = _fake_weight("gguf_q5_k_t16_v1", backend="hip_gfx1151")
    resolved_variants: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, missing="error"):
        del backend, layer, quant, missing
        resolved_variants.append(variant)
        return lambda: None

    monkeypatch.setattr(qgr, "resolve", fake_resolve)

    plan = qgr._resolve_compact_moe_wmma_kernels(gate, up, down)

    assert plan is not None
    assert "selected_dual_wmma_prefill_compact_bf16_bf16_out" in resolved_variants
    assert not any("shared_x" in variant for variant in resolved_variants)


def test_t16_q4_invalid_explicit_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", "unknown")
    with pytest.raises(ValueError, match="Q4 T16 selected prefill mode"):
        qgr._resolve_compact_moe_wmma_kernels(
            _fake_weight("gguf_q4_k_t16_v1"),
            _fake_weight("gguf_q4_k_t16_v1"),
            _fake_weight("gguf_q5_k_t16_v1"),
        )


def test_t16_q4_missing_explicit_shared_x_kernel_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", "shared_x")
    monkeypatch.setattr(qgr, "_ensure_compact_moe_wmma_registered", lambda: None)
    monkeypatch.setattr(qgr, "load_backend_kernel_package", lambda backend: None)

    def fake_resolve(*, backend, layer, quant, variant, missing="error"):
        if "shared_x" in variant:
            return None
        return lambda: None

    monkeypatch.setattr(qgr, "resolve", fake_resolve)
    with pytest.raises(RuntimeError, match="explicit Q4 T16 selected prefill mode"):
        qgr._resolve_compact_moe_wmma_kernels(
            _fake_weight("gguf_q4_k_t16_v1"),
            _fake_weight("gguf_q4_k_t16_v1"),
            _fake_weight("gguf_q5_k_t16_v1"),
        )


def test_raw_q4_resolver_ignores_t16_only_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", "shared_x")
    variants: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, missing="error"):
        variants.append(variant)
        return lambda: None

    monkeypatch.setattr(qgr, "resolve", fake_resolve)
    plan = qgr._resolve_compact_moe_wmma_kernels(
        _fake_weight("gguf_q4_k"),
        _fake_weight("gguf_q4_k"),
        _fake_weight("gguf_q5_k"),
    )

    assert plan is not None
    assert "selected_dual_wmma_prefill_compact_bf16_bf16_out" in variants
    assert not any("shared_x" in variant for variant in variants)


def test_missing_down_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown down quant key declines the WMMA dispatch.

    The resolver falls back to ``None`` so the runner can take its slower
    per-row selected GEMV path instead of crashing. This documents that
    only the four wired quants (raw + T16 for Q5 / Q6) are valid.
    """

    gate = _fake_weight("gguf_q4_k_t16_v1")
    up = _fake_weight("gguf_q4_k_t16_v1")
    down = _fake_weight("gguf_q8_0_t16_v1")  # Q8 is dense, not selected
    monkeypatch.setattr(
        qgr,
        "resolve",
        lambda *, backend, layer, quant, variant, missing="error": lambda: f"{quant}.{variant}",
    )

    plan = qgr._resolve_compact_moe_wmma_kernels(gate, up, down)

    assert plan is None


def test_t16_down_keys_point_at_registered_aliases() -> None:
    """P10.B2 / P10.B3: T16 down keys resolve to the new compact aliases."""

    q5 = qgr._COMPACT_MOE_DOWN_KEYS["gguf_q5_k_t16_v1"]
    assert q5.backend == "hip_gfx1100"
    assert q5.layer == "moe_linear"
    assert q5.quant == "gguf_q5_k_t16_v1"
    assert q5.variant == "selected_wmma_prefill_compact_bf16_bf16_out"

    q6 = qgr._COMPACT_MOE_DOWN_KEYS["gguf_q6_k_t16_v1"]
    assert q6.backend == "hip_gfx1100"
    assert q6.layer == "moe_linear"
    assert q6.quant == "gguf_q6_k_t16_v1"
    assert q6.variant == "selected_wmma_prefill_compact_bf16_bf16_out"


def test_allocation_name_helper() -> None:
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q4_k")) == "raw"
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q5_k")) == "raw"
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q6_k")) == "raw"
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q4_k_t16_v1")) == "tiles"
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q5_k_t16_v1")) == "tiles"
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q6_k_t16_v1")) == "tiles"
    assert qgr._selected_wmma_allocation_name(_fake_weight("gguf_q8_0_t16_v1")) == "tiles"
