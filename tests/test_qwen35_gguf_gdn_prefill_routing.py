"""Routing tests for the qwen35 GGUF GDN prefill plan (task P9.A1).

These tests cover the registry-only dispatch added in task #17:

* ``_resolve_gguf_gdn_prefill_plan()`` returns a complete chain (prepare +
  recurrent + rmsnorm_gate) when the new ``gguf_qwen35`` registry aliases are
  registered.
* ``Qwen35GGUFFullStackRunner._run_gdn_prefill(...)`` calls the chain in the
  correct order at single-segment prefill row counts.
* The same helper falls back to the legacy fused ``decode_order_bf16`` kernel
  when the chain is incomplete (so behaviour matches the pre-P9 path until the
  k2 chain is registered).
* The ``HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD`` env var controls whether
  the segments_k2 kernel is dispatched, when one is registered.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.runtime import qwen35_gguf_runner as qgr


@pytest.fixture(autouse=True)
def _reset_segment_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", raising=False)


def test_resolve_gguf_gdn_prefill_plan_returns_complete_chain() -> None:
    register_qwen35_linear_attn_gdn_kernels()
    plan = qgr._resolve_gguf_gdn_prefill_plan()
    assert plan.has_chain
    assert plan.has_fused
    assert plan.prepare is qwen35_linear_attn_prefill_prepare_f32_bf16
    assert plan.recurrent is qwen35_gdn_prefill_recurrent_k2_f32
    assert plan.recurrent_segments is qwen35_gdn_prefill_recurrent_segments_k2_f32
    assert plan.rmsnorm_gate is qwen35_gdn_prefill_rmsnorm_gate_bf16
    assert plan.fused_decode_order is qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order


def test_run_gdn_prefill_uses_chain_under_threshold() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
    )
    layer = _make_layer()
    scratch = _make_scratch()
    cfg = _make_cfg()

    runner._run_gdn_prefill(
        layer=layer,
        scratch=scratch,
        cfg=cfg,
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == ["prepare", "recurrent_k2", "rmsnorm_gate"]
    prepare_args = next(args for name, args in calls if name == "prepare")
    assert prepare_args[0] == scratch.conv_out.ptr
    assert prepare_args[5:10] == (
        scratch.prefill_query.ptr,
        scratch.prefill_key.ptr,
        scratch.prefill_value.ptr,
        scratch.prefill_beta.ptr,
        scratch.prefill_decay.ptr,
    )
    recurrent_args = next(args for name, args in calls if name == "recurrent_k2")
    assert recurrent_args[5:7] == (0xDEAD0001, scratch.recurrent_out.ptr)
    assert recurrent_args[7] == 64
    assert recurrent_args[8:11] == (
        cfg.ssm_time_step_rank,
        cfg.ssm_state_size,
        runner.ssm_value_dim,
    )
    rmsnorm_args = next(args for name, args in calls if name == "rmsnorm_gate")
    assert rmsnorm_args[0] == scratch.recurrent_out.ptr
    assert rmsnorm_args[3] == scratch.recurrent_bf16.ptr


def test_run_gdn_prefill_uses_segments_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "128")
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=256,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0002),
        stream=0,
        runtime="runtime-sentinel",
    )

    names = [name for name, _ in calls]
    assert names == ["prepare", "recurrent_segments_k2", "rmsnorm_gate"]
    segments_args = next(args for name, args in calls if name == "recurrent_segments_k2")
    # cu_seqlens and state_indices pointers + total_tokens=256 + segments=1
    assert segments_args[7:11] == (
        _make_scratch().gdn_cu_seqlens.ptr,
        _make_scratch().gdn_state_indices.ptr,
        256,
        1,
    )


def test_run_gdn_prefill_skips_segments_when_scratch_missing() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
    )
    scratch = _make_scratch(include_gdn_segment_fields=False)

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=scratch,
        cfg=_make_cfg(),
        rows=4096,
        recurrent_state=SimpleNamespace(ptr=0x77),
        stream=0,
        runtime="runtime-sentinel",
    )

    names = [name for name, _ in calls]
    assert names == ["prepare", "recurrent_k2", "rmsnorm_gate"]


def test_run_gdn_prefill_falls_back_to_fused_when_chain_incomplete() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused"),
    )
    layer = _make_layer()
    scratch = _make_scratch()
    cfg = _make_cfg()

    runner._run_gdn_prefill(
        layer=layer,
        scratch=scratch,
        cfg=cfg,
        rows=128,
        recurrent_state=SimpleNamespace(ptr=0xBEEF),
        stream=0,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == ["fused"]
    fused_args = next(args for name, args in calls if name == "fused")
    # Spot-check fused signature mirrors the legacy decode_order kernel:
    # (conv_out, gate, alpha, beta, dt_bias, a_log, norm_weight, state, out, eps, tokens, ...)
    assert fused_args[0] == scratch.conv_out.ptr
    assert fused_args[1] == scratch.linear_z.ptr
    assert fused_args[8] == scratch.recurrent_bf16.ptr
    assert fused_args[10] == 128


def test_run_gdn_prefill_raises_when_no_kernels_registered() -> None:
    runner = _new_runner()
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=None,
        fused_decode_order=None,
    )
    with pytest.raises(RuntimeError, match="no qwen35 GGUF GDN prefill kernels"):
        runner._run_gdn_prefill(
            layer=_make_layer(),
            scratch=_make_scratch(),
            cfg=_make_cfg(),
            rows=4,
            recurrent_state=SimpleNamespace(ptr=0x12),
            stream=0,
            runtime="runtime-sentinel",
        )


def test_segment_threshold_env_override_invalid_values_fall_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "not-a-number")
    assert qgr._gguf_gdn_prefill_segment_threshold() == 1025
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "0")
    assert qgr._gguf_gdn_prefill_segment_threshold() == 1
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "128")
    assert qgr._gguf_gdn_prefill_segment_threshold() == 128
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", raising=False)
    assert qgr._gguf_gdn_prefill_segment_threshold() == 1025


def _new_runner() -> qgr.Qwen35GGUFFullStackRunner:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    # ssm_value_dim is a derived property; feed a fake weights/config so the
    # property resolves to 128 (= 4096 / 32).
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(ssm_inner_size=4096, ssm_time_step_rank=32),
    )
    return runner


def _make_layer():
    weights = {
        "ssm_dt_bias": _Weight(0xA001),
        "ssm_a": _Weight(0xA002),
        "ssm_norm": _Weight(0xA003),
    }

    def weight(name: str) -> object:
        return weights[name]

    return SimpleNamespace(weight=weight)


def _make_scratch(*, include_gdn_segment_fields: bool = True) -> SimpleNamespace:
    fields = {
        "conv_out": SimpleNamespace(ptr=0xC0),
        "linear_alpha": SimpleNamespace(ptr=0xC1),
        "linear_beta": SimpleNamespace(ptr=0xC2),
        "linear_z": SimpleNamespace(ptr=0xC3),
        "prefill_query": SimpleNamespace(ptr=0xD0),
        "prefill_key": SimpleNamespace(ptr=0xD1),
        "prefill_value": SimpleNamespace(ptr=0xD2),
        "prefill_beta": SimpleNamespace(ptr=0xD3),
        "prefill_decay": SimpleNamespace(ptr=0xD4),
        "recurrent_out": SimpleNamespace(ptr=0xE0),
        "recurrent_bf16": SimpleNamespace(ptr=0xE1),
    }
    if include_gdn_segment_fields:
        fields["gdn_cu_seqlens"] = SimpleNamespace(ptr=0xF0)
        fields["gdn_state_indices"] = SimpleNamespace(ptr=0xF1)
    return SimpleNamespace(**fields)


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        ssm_group_count=4,
        ssm_time_step_rank=32,
        ssm_state_size=128,
        rms_norm_eps=1.0e-6,
    )


class _Weight:
    def __init__(self, tensor_ptr: int) -> None:
        self._allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=tensor_ptr))

    def allocation(self, name: str = "main") -> object:
        return self._allocation


def _recorder(sink: list[tuple[str, object]], name: str):
    def fake(*args, **kwargs):
        sink.append((name, args))

    return fake
