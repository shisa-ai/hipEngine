"""Session-frozen GDN prefill mode resolution (design review 2026-08-15).

Covers the converged Option C contract: one quant-shape-aware resolver with
availability fallback, the session-scratch freeze (environment mutations
cannot redirect a production session), the verify-gate diagnostic
alternation path, and the dense gfx1151 0.8B liveness-is-None contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import hipengine.runtime.qwen35_gguf_runner as qgr
import pytest


def _weights(file_type: str, *, k_heads=16, v_heads=16, state=128, inner=2048):
    return SimpleNamespace(
        file_type_name=file_type,
        config=SimpleNamespace(
            ssm_group_count=k_heads,
            ssm_time_step_rank=v_heads,
            ssm_state_size=state,
            ssm_inner_size=inner,
        ),
    )


def _with_env(monkeypatch, mode=None, **extra):
    if mode is None:
        monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", mode)
    for key, value in extra.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_resolver_matrix_gfx1151(monkeypatch):
    _with_env(monkeypatch)
    q4 = _weights("MOSTLY_Q4_K_M")
    q8 = _weights("MOSTLY_Q8_0")
    moe_unkeyed = _weights("MOSTLY_Q4_K_M", k_heads=32, v_heads=16, inner=1024)

    assert (
        qgr._gguf_gdn_prefill_session_mode("hip_gfx1151", weights=q4, cfg=q4.config)
        == "chain_peer_cluster8"
    )
    assert (
        qgr._gguf_gdn_prefill_session_mode("hip_gfx1151", weights=q8, cfg=q8.config)
        == "chain_peer_cluster8"
    )
    # Unkeyed shapes keep the architecture default exact-direct route.
    assert (
        qgr._gguf_gdn_prefill_session_mode(
            "hip_gfx1151", weights=moe_unkeyed, cfg=moe_unkeyed.config
        )
        == "chain_lds32_direct_nonvolatile"
    )
    # Backend-only call (no model context) also keeps the default.
    assert (
        qgr._gguf_gdn_prefill_session_mode("hip_gfx1151")
        == "chain_lds32_direct_nonvolatile"
    )


def test_resolver_explicit_and_exact(monkeypatch):
    _with_env(monkeypatch, mode="chain_peer_wave32")
    assert qgr._gguf_gdn_prefill_session_mode("hip_gfx1151") == "chain_peer_wave32"
    _with_env(monkeypatch, mode="exact")
    assert (
        qgr._gguf_gdn_prefill_session_mode("hip_gfx1151")
        == "chain_lds32_direct_nonvolatile"
    )


def test_resolver_availability_fallback(monkeypatch):
    _with_env(monkeypatch)
    real = qgr._resolve_gguf_gdn_prefill_plan

    def plan_missing_cluster8(backend):
        plan = real(backend)
        view = SimpleNamespace(
            **{
                name: getattr(plan, name)
                for name in dir(plan)
                if not name.startswith("_")
            }
        )
        view.auto_mode = "chain_peer_cluster8"
        view.has_chain_peer_cluster8 = False
        view.has_fused = True
        return view

    monkeypatch.setattr(qgr, "_resolve_gguf_gdn_prefill_plan", plan_missing_cluster8)
    weights = _weights("MOSTLY_Q8_0")
    assert (
        qgr._gguf_gdn_prefill_session_mode(
            "hip_gfx1151", weights=weights, cfg=weights.config
        )
        == "fused"
    )


def test_dense_gfx1151_08b_liveness_remains_dedicated(monkeypatch):
    """Today's contract: dense 0.8B gfx1151 has no liveness alias admission."""

    _with_env(monkeypatch)
    for file_type in ("MOSTLY_Q4_K_M", "MOSTLY_Q8_0"):
        weights = _weights(file_type)
        runner = SimpleNamespace(weights=weights, backend="hip_gfx1151")
        for rows in (768, 4096):
            fields = qgr._gguf_prefill_scratch_liveness_disabled_fields(
                runner, rows=rows
            )
            assert fields is None, (file_type, rows, fields)


def test_dispatch_consumes_frozen_mode(monkeypatch):
    """A production scratch freezes the route; later env flips are no-ops."""

    from tests.test_qwen35_gguf_gdn_prefill_routing import (
        _make_layer,
        _make_scratch,
        _new_runner,
        _recorder,
    )

    runner = _new_runner()
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(ssm_inner_size=2048, ssm_time_step_rank=16),
        file_type_name="MOSTLY_Q4_K_M",
    )
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_peer_normalized=_recorder(calls, "peer_prepare"),
        recurrent_peer_cluster8=_recorder(calls, "peer_cluster8"),
        recurrent_segments_peer_cluster8=_recorder(calls, "segments_peer_cluster8"),
        prepare_compact_peer_normalized=None,
        recurrent_compact_peer_wave32=None,
        auto_mode="chain_lds32_direct_nonvolatile",
        auto_modes_by_quant_shape={},
    )
    cfg = SimpleNamespace(
        ssm_group_count=16,
        ssm_time_step_rank=16,
        ssm_state_size=128,
        rms_norm_eps=1.0e-6,
    )
    scratch = _make_scratch()
    # Freeze an env-incompatible route on the session scratch; a later env
    # flip to the exact-direct default must NOT redirect dispatch.
    scratch.gdn_effective_mode = "chain_peer_cluster8"
    scratch.gdn_mode_diagnostic = False
    _with_env(monkeypatch, mode="chain_lds32_direct_nonvolatile")

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=scratch,
        cfg=cfg,
        rows=512,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )
    assert [name for name, _ in calls] == [
        "peer_prepare",
        "segments_peer_cluster8",
        "rmsnorm_gate",
    ]


def test_diagnostic_scratch_still_follows_env(monkeypatch):
    """Verify-gate superset scratch keeps in-session mode alternation."""

    weights = _weights("MOSTLY_Q4_K_M")
    frozen = qgr._gguf_gdn_prefill_session_mode(
        "hip_gfx1151", weights=weights, cfg=weights.config
    )
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE", "1")
    _with_env(monkeypatch, mode="chain_lds32_direct_nonvolatile")
    assert qgr._gguf_gdn_prefill_scratch_diagnostic() is True
    alternate = qgr._gguf_gdn_prefill_session_mode(
        "hip_gfx1151", weights=weights, cfg=weights.config
    )
    assert alternate != frozen
    assert alternate == "chain_lds32_direct_nonvolatile"


def test_diagnostic_scratch_still_follows_env(monkeypatch):
    """Verify-gate superset scratch keeps in-session mode alternation."""

    weights = _weights("MOSTLY_Q4_K_M")
    frozen = qgr._gguf_gdn_prefill_session_mode(
        "hip_gfx1151", weights=weights, cfg=weights.config
    )
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE", "1")
    _with_env(monkeypatch, mode="chain_lds32_direct_nonvolatile")
    assert qgr._gguf_gdn_prefill_scratch_diagnostic() is True
    alternate = qgr._gguf_gdn_prefill_session_mode(
        "hip_gfx1151", weights=weights, cfg=weights.config
    )
    assert alternate != frozen
    assert alternate == "chain_lds32_direct_nonvolatile"


def test_scratch_freeze_fields_exist():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(qgr._GGUFFullAttentionPrefillScratch)}
    assert {"gdn_effective_mode", "gdn_mode_diagnostic"} <= fields
    # Runners are shared across sessions; the freeze must not live there.
    assert not hasattr(qgr.Qwen35GGUFFullStackRunner, "gdn_effective_mode")
