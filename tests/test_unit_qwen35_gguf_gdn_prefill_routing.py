"""Routing tests for the qwen35 GGUF GDN prefill plan (task P9.A1).

These tests cover the registry-only dispatch added in task #17:

* ``_resolve_gguf_gdn_prefill_plan()`` returns a complete chain (prepare +
  recurrent + rmsnorm_gate) when the new ``gguf_qwen35`` registry aliases are
  registered.
* ``Qwen35GGUFFullStackRunner._run_gdn_prefill(...)`` prefers the legacy fused
  ``decode_order_bf16`` kernel while the split k2 chain is parity-blocked on the
  real GGUF target trace.
* The helper still calls the chain in the correct order when the fused fallback
  is unavailable, preserving registry-only fallback coverage.
* The ``HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD`` env var controls whether
  the segments_k2 kernel is dispatched, when the chain fallback is used.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q8_0_T16,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_f32_bf16_out,
    qwen35_gdn_prefill_recurrent_decode_order_exact_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32,
    qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state,
    qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state,
    qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32,
    qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
    qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32,
    qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32,
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16,
    qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16,
    qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16,
    qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.runtime import qwen35_gguf_runner as qgr


@pytest.fixture(autouse=True)
def _reset_segment_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)


def test_default_q4_prefill_native_resolver_skips_cpu_reference_fallback() -> None:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1100"
    runner._gguf_prefill_quant = "gguf_q4_k_m"

    assert (
        runner._full_attn_prefill_native_fn()
        is qgr.qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans
    )


def test_gdn_output_boundary_is_selected_by_ssm_out_weight_plugin() -> None:
    register_qwen35_linear_attn_gdn_kernels()
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1100"
    runner._gguf_prefill_quant = "gguf_q4_k_m"
    dense_bf16 = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q5_k"),
    )
    q5_t16 = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(
            layout=LAYOUT_GGUF_Q5_K_T16,
            quant_key="gguf_q5_k_t16_v1",
        ),
    )
    q8_t16 = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(
            layout=LAYOUT_GGUF_Q8_0_T16,
            quant_key="gguf_q8_0_t16_v1",
        ),
    )

    assert runner._gdn_decode_output_cast_for_weight(dense_bf16) is f32_to_bf16
    assert runner._gdn_decode_output_cast_for_weight(q5_t16) is f32_to_bf16
    assert runner._gdn_decode_output_cast_for_weight(q8_t16) is None
    assert (
        runner._gdn_decode_output_fusion_for_weight(q5_t16)
        is qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out
    )
    assert (
        runner._gdn_chain_output_fusion_for_weight(q5_t16)
        is qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_f32_bf16_out
    )
    runner.fp16_recurrent_state = True
    assert (
        runner._gdn_decode_output_fusion_for_weight(q5_t16)
        is qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state
    )
    runner.fp16_recurrent_state = False
    assert runner._gdn_decode_output_fusion_for_weight(dense_bf16) is None
    assert runner._gdn_chain_output_fusion_for_weight(dense_bf16) is None


def test_fp16_chain_journal_plan_selects_typed_state_and_snapshot_writers() -> None:
    register_qwen35_linear_attn_gdn_kernels()

    strict = qgr._resolve_gguf_linear_attention_chain_journal_plan(
        "hip_gfx1151",
        use_fp16_state=False,
    )
    production = qgr._resolve_gguf_linear_attention_chain_journal_plan(
        "hip_gfx1151",
        use_fp16_state=True,
    )

    assert strict.available and not strict.snapshot_available
    assert production.available and not production.snapshot_available
    assert production.conv is strict.conv
    assert strict.conv_snapshot is None
    assert production.conv_snapshot is None
    assert production.gdn.__name__.endswith("_fp16state")
    assert production.gdn_snapshot is None


def test_q3_decode_output_width_policy_is_registry_selected() -> None:
    register_qwen35_linear_attn_gdn_kernels()
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1100"
    runner._gguf_prefill_quant = "gguf_ud_q3_k_m"

    assert runner._gdn_decode_output_cast_fn() is f32_to_bf16

    runner._gguf_prefill_quant = "gguf_qwen35"
    runner.__dict__.pop("_gguf_gdn_decode_output_cast_fn_cache", None)
    assert runner._gdn_decode_output_cast_fn() is None


def test_resolve_gguf_gdn_prefill_plan_preserves_registered_overrides() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "gdn_chain_recurrent_rmsnorm_gate+cast+snapshot",
        "gguf_q5_k_t16_v1",
        "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )

    def counted_override(*_args, **_kwargs):
        return None

    register(key, counted_override, replace=True)
    try:
        qgr._resolve_gguf_gdn_prefill_plan()
        assert (
            resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
            is counted_override
        )
    finally:
        register(key, original, replace=True)


def test_resolve_gguf_gdn_prefill_plan_returns_complete_chain() -> None:
    register_qwen35_linear_attn_gdn_kernels()
    plan = qgr._resolve_gguf_gdn_prefill_plan()
    assert plan.has_chain
    assert plan.has_fused
    assert plan.prepare is qwen35_linear_attn_prefill_prepare_f32_bf16
    assert (
        plan.prepare_peer_normalized
        is qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16
    )
    assert plan.recurrent is qwen35_gdn_prefill_recurrent_k2_f32
    assert plan.recurrent_segments is qwen35_gdn_prefill_recurrent_segments_k2_f32
    assert plan.rmsnorm_gate is qwen35_gdn_prefill_rmsnorm_gate_bf16
    assert plan.fused_decode_order is qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order
    assert plan.exact_prepare is qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16
    assert (
        plan.exact_prepare_compact
        is qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16
    )
    assert plan.exact_recurrent is qwen35_gdn_prefill_recurrent_decode_order_exact_f32
    assert (
        plan.exact_recurrent_segments
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32
    )
    assert plan.has_exact_chain
    assert (
        plan.exact_recurrent_tile64
        is qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32
    )
    assert (
        plan.exact_recurrent_segments_tile64
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32
    )
    assert (
        plan.exact_recurrent_tile32
        is qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32
    )
    assert (
        plan.exact_recurrent_segments_tile32
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32
    )
    assert plan.has_exact_chain_tile64
    assert plan.has_exact_chain_tile32
    assert (
        plan.exact_recurrent_lds64
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32
    )
    assert (
        plan.exact_recurrent_segments_lds64
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32
    )
    assert (
        plan.exact_recurrent_lds32
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32
    )
    assert (
        plan.exact_recurrent_segments_lds32
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32
    )
    assert plan.has_exact_chain_lds64
    assert plan.has_exact_chain_lds32
    assert (
        plan.exact_recurrent_lds32_direct
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32
    )
    assert (
        plan.exact_recurrent_segments_lds32_direct
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32
    )
    assert plan.has_exact_chain_lds32_direct
    assert (
        plan.exact_recurrent_lds32_direct_nonvolatile
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32
    )
    assert (
        plan.exact_recurrent_segments_lds32_direct_nonvolatile
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32
    )
    assert plan.has_exact_chain_lds32_direct_nonvolatile
    assert (
        plan.exact_recurrent_wave32
        is qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32
    )
    assert (
        plan.exact_recurrent_segments_wave32
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32
    )
    assert plan.has_exact_chain_wave32
    assert (
        plan.recurrent_wave32_tree
        is qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32
    )
    assert (
        plan.recurrent_segments_wave32_tree
        is qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32
    )
    assert plan.has_chain_wave32_tree
    assert (
        plan.recurrent_peer_wave32
        is qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32
    )
    assert (
        plan.recurrent_segments_peer_wave32
        is qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32
    )
    assert plan.has_chain_peer_wave32
    assert (
        plan.prepare_compact_peer_normalized
        is qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16
    )
    assert (
        plan.recurrent_compact_peer_wave32
        is qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32
    )
    assert (
        plan.recurrent_compact_peer_wave32_fp16state
        is qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state
    )
    assert (
        plan.recurrent_compact_segments_peer_wave32
        is qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32
    )
    assert (
        plan.recurrent_compact_segments_peer_wave32_fp16state
        is qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state
    )
    assert plan.has_chain_compact_peer_wave32
    assert plan.auto_mode == "chain_compact_peer_wave32"
    assert (
        plan.recurrent_peer_cluster8
        is qwen35_gdn_prefill_recurrent_normalized_cluster8_f32
    )
    assert (
        plan.recurrent_segments_peer_cluster8
        is qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32
    )
    assert plan.has_chain_peer_cluster8


def test_resolve_gguf_gdn_prefill_plan_uses_gfx1151_package_default() -> None:
    plan = qgr._resolve_gguf_gdn_prefill_plan("hip_gfx1151")

    assert plan.auto_mode == "chain_lds32_direct_nonvolatile"
    assert plan.auto_modes_by_quant_shape == {
        ("mostly_q4_k_m", 16, 16, 128, 128): "chain_peer_cluster8",
        ("mostly_q4_k_m", 16, 48, 128, 128): "chain_compact_peer_wave32",
        ("mostly_q4_k_s", 16, 48, 128, 128): "chain_compact_peer_wave32",
        # D08-X2-K2: fresh five-block gate admitted Q8_0 on this geometry.
        ("mostly_q8_0", 16, 16, 128, 128): "chain_peer_cluster8",
    }
    assert plan.compact_peer_chunk_rows == 1024
    assert plan.has_exact_chain_lds32
    assert plan.has_exact_chain_lds32_direct
    assert plan.has_exact_chain_lds32_direct_nonvolatile


@pytest.mark.parametrize("file_type_name", ("MOSTLY_Q4_K_M", "MOSTLY_Q4_K_S"))
def test_gfx1151_qwen38_auto_uses_compact_qk_capacity_and_peer_liveness(
    file_type_name: str,
) -> None:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1151"
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(
            ssm_group_count=16,
            ssm_time_step_rank=48,
            ssm_state_size=128,
            ssm_inner_size=6144,
            is_moe=False,
        ),
        file_type_name=file_type_name,
        model_id="Qwen3.8-27B",
        quant="gguf_q4_k_m",
    )

    assert (
        qgr._gguf_gdn_prefill_session_mode(
            runner.backend,
            weights=runner.weights,
            cfg=runner.weights.config,
        )
        == "chain_compact_peer_wave32"
    )
    assert qgr._gguf_prefill_normalized_qk_heads(runner) == 16


def test_run_gdn_prefill_auto_uses_quant_shape_scoped_cluster8_override() -> None:
    runner = _new_runner()
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(ssm_inner_size=2048, ssm_time_step_rank=16),
        file_type_name="MOSTLY_Q4_K_M",
    )
    runner._gguf_prefill_quant = "gguf_q4_k_m"
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
        auto_mode="chain_lds32_direct_nonvolatile",
        auto_modes_by_quant_shape={
            ("mostly_q4_k_m", 16, 16, 128, 128): "chain_peer_cluster8",
        },
    )
    cfg = SimpleNamespace(
        ssm_group_count=16,
        ssm_time_step_rank=16,
        ssm_state_size=128,
        rms_norm_eps=1.0e-6,
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
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


def test_run_gdn_prefill_auto_keeps_q8_on_exact_default() -> None:
    runner = _new_runner()
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(ssm_inner_size=2048, ssm_time_step_rank=16),
        file_type_name="MOSTLY_Q8_0",
    )
    # A stale caller-selected quant label must not override the actual file type.
    runner._gguf_prefill_quant = "gguf_q4_k_m"
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_peer_normalized=_recorder(calls, "peer_prepare"),
        exact_prepare_compact=_recorder(calls, "exact_prepare_compact"),
        exact_recurrent_lds32_direct_nonvolatile=_recorder(calls, "exact_direct"),
        exact_recurrent_segments_lds32_direct_nonvolatile=_recorder(
            calls, "exact_segments_direct"
        ),
        recurrent_peer_cluster8=_recorder(calls, "peer_cluster8"),
        recurrent_segments_peer_cluster8=_recorder(calls, "segments_peer_cluster8"),
        auto_mode="chain_lds32_direct_nonvolatile",
        auto_modes_by_quant_shape={
            ("mostly_q4_k_m", 16, 16, 128, 128): "chain_peer_cluster8",
        },
    )
    cfg = SimpleNamespace(
        ssm_group_count=16,
        ssm_time_step_rank=16,
        ssm_state_size=128,
        rms_norm_eps=1.0e-6,
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=cfg,
        rows=512,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare_compact",
        "exact_segments_direct",
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_prefers_fused_decode_order_when_available() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
    )
    scratch = _make_scratch()

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=scratch,
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == ["fused_decode_order"]
    fused_args = calls[0][1]
    assert fused_args[0] == scratch.conv_out.ptr
    assert fused_args[1] == scratch.linear_z.ptr
    assert fused_args[8] == scratch.recurrent_bf16.ptr
    assert fused_args[10] == 64


def test_run_gdn_prefill_auto_uses_arch_scoped_lds32_default() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        exact_recurrent_lds32=_recorder(calls, "exact_lds32"),
        exact_recurrent_segments_lds32=_recorder(calls, "exact_segments_lds32"),
        auto_mode="chain_lds32",
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare",
        "exact_lds32",
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_auto_falls_back_to_fused_when_preferred_mode_missing() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        auto_mode="chain_lds32",
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == ["fused_decode_order"]


@pytest.mark.parametrize(
    ("rows", "expected_recurrent"),
    [(64, "exact_lds32_direct"), (1025, "exact_segments_lds32_direct")],
)
def test_run_gdn_prefill_explicit_direct_lds32_uses_compact_abi(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    expected_recurrent: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_lds32_direct")
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare_compact=_recorder(calls, "exact_prepare_compact"),
        exact_recurrent_lds32_direct=_recorder(calls, "exact_lds32_direct"),
        exact_recurrent_segments_lds32_direct=_recorder(
            calls, "exact_segments_lds32_direct"
        ),
    )
    scratch = _make_scratch()

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=scratch,
        cfg=_make_cfg(),
        rows=rows,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0003),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare_compact",
        expected_recurrent,
        "rmsnorm_gate",
    ]
    prepare_args = calls[0][1]
    assert prepare_args[:9] == (
        scratch.conv_out.ptr,
        scratch.linear_alpha.ptr,
        scratch.linear_beta.ptr,
        0xA001,
        0xA002,
        scratch.prefill_beta.ptr,
        scratch.prefill_decay.ptr,
        scratch.prefill_query_scale.ptr,
        scratch.prefill_key_scale.ptr,
    )
    recurrent_args = calls[1][1]
    assert recurrent_args[:7] == (
        scratch.conv_out.ptr,
        scratch.prefill_beta.ptr,
        scratch.prefill_decay.ptr,
        scratch.prefill_query_scale.ptr,
        scratch.prefill_key_scale.ptr,
        0xDEAD0003,
        scratch.recurrent_out.ptr,
    )
    if rows == 64:
        assert recurrent_args[7:12] == (64, 4, 32, 128, 128)
    else:
        assert recurrent_args[7:11] == (
            scratch.gdn_cu_seqlens.ptr,
            scratch.gdn_state_indices.ptr,
            1025,
            1,
        )
        assert recurrent_args[11:15] == (4, 32, 128, 128)


@pytest.mark.parametrize(
    ("rows", "expected_recurrent"),
    [(64, "exact_lds32_nonvolatile"), (1025, "exact_segments_lds32_nonvolatile")],
)
@pytest.mark.parametrize(
    "requested_mode",
    ("chain_lds32_direct_nonvolatile", "exact"),
)
def test_run_gdn_prefill_explicit_nonvolatile_direct_uses_compact_abi(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    expected_recurrent: str,
    requested_mode: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", requested_mode)
    runner = _new_runner()
    runner.backend = "hip_gfx1100"
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare_compact=_recorder(calls, "exact_prepare_compact"),
        exact_recurrent_lds32_direct_nonvolatile=_recorder(
            calls, "exact_lds32_nonvolatile"
        ),
        exact_recurrent_segments_lds32_direct_nonvolatile=_recorder(
            calls, "exact_segments_lds32_nonvolatile"
        ),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=rows,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0005),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare_compact",
        expected_recurrent,
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_gfx1151_auto_uses_compact_direct_lds32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare_compact=_recorder(calls, "exact_prepare_compact"),
        exact_recurrent_lds32_direct=_recorder(calls, "exact_lds32_direct"),
        exact_recurrent_segments_lds32_direct=_recorder(
            calls, "exact_segments_lds32_direct"
        ),
        auto_mode="chain_lds32_direct",
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0004),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare_compact",
        "exact_lds32_direct",
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_explicit_chain_overrides_available_fused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain")
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
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == ["prepare", "recurrent_k2", "rmsnorm_gate"]


def test_run_gdn_prefill_explicit_chain_prefers_exact_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain")
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "legacy_prepare"),
        recurrent=_recorder(calls, "legacy_recurrent"),
        recurrent_segments=_recorder(calls, "legacy_segments"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        exact_recurrent=_recorder(calls, "exact_recurrent"),
        exact_recurrent_segments=_recorder(calls, "exact_segments"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare",
        "exact_recurrent",
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_explicit_chain_k2_bypasses_exact_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_k2")
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        exact_recurrent=_recorder(calls, "exact_recurrent"),
        exact_recurrent_segments=_recorder(calls, "exact_segments"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "prepare",
        "recurrent_k2",
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_explicit_compact_peer_route_uses_compact_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_compact_peer_wave32"
    )
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_peer_normalized=_recorder(calls, "peer_prepare"),
        prepare_compact_peer_normalized=_recorder(calls, "compact_peer_prepare"),
        recurrent_peer_wave32=_recorder(calls, "peer_wave32"),
        recurrent_compact_peer_wave32=_recorder(calls, "compact_peer_wave32"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "compact_peer_prepare",
        "compact_peer_wave32",
        "rmsnorm_gate",
    ]
    prepare_args = calls[0][1]
    recurrent_args = calls[1][1]
    assert prepare_args[5:10] == (0xD0, 0xD1, 0xD2, 0xD3, 0xD4)
    assert recurrent_args[7:12] == (64, 4, 32, 128, 128)


def test_run_gdn_prefill_compact_peer_segments_use_indexed_state_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_compact_peer_wave32"
    )
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_compact_peer_normalized=_recorder(calls, "compact_peer_prepare"),
        recurrent_compact_peer_wave32=_recorder(calls, "compact_peer_wave32"),
        recurrent_compact_segments_peer_wave32=_recorder(
            calls, "compact_peer_segments"
        ),
    )
    scratch = _make_scratch()
    scratch.gdn_active_segments = 2

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=scratch,
        cfg=_make_cfg(),
        rows=7,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "compact_peer_prepare",
        "compact_peer_segments",
        "rmsnorm_gate",
    ]
    recurrent_args = calls[1][1]
    assert recurrent_args[7:15] == (
        0xF0,
        0xF1,
        7,
        2,
        4,
        32,
        128,
        128,
    )


def test_run_gdn_prefill_chunks_compact_peer_recurrence_with_state_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_compact_peer_wave32"
    )
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_compact_peer_normalized=_recorder(calls, "compact_peer_prepare"),
        recurrent_compact_peer_wave32=_recorder(calls, "compact_peer_wave32"),
        compact_peer_chunk_rows=1024,
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=4096,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "compact_peer_prepare",
        "compact_peer_wave32",
        "compact_peer_wave32",
        "compact_peer_wave32",
        "compact_peer_wave32",
        "rmsnorm_gate",
    ]
    recurrent = [args for name, args in calls if name == "compact_peer_wave32"]
    assert all(args[5] == 0xDEAD0001 for args in recurrent)
    assert [args[7] for args in recurrent] == [1024] * 4
    assert [args[0] for args in recurrent] == [
        0xD0,
        0xD0 + 0x200000,
        0xD0 + 0x400000,
        0xD0 + 0x600000,
    ]
    assert [args[2] for args in recurrent] == [
        0xD2,
        0xD2 + 0x1000000,
        0xD2 + 0x2000000,
        0xD2 + 0x3000000,
    ]
    assert [args[3] for args in recurrent] == [
        0xD3,
        0xD3 + 0x20000,
        0xD3 + 0x40000,
        0xD3 + 0x60000,
    ]
    assert [args[6] for args in recurrent] == [
        0xE0,
        0xE0 + 0x1000000,
        0xE0 + 0x2000000,
        0xE0 + 0x3000000,
    ]


def test_run_gdn_prefill_rejects_peer_route_after_compact_qk_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_peer_wave32")
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_peer_normalized=_recorder(calls, "peer_prepare"),
        recurrent_peer_wave32=_recorder(calls, "peer_wave32"),
    )
    scratch = _make_scratch()
    scratch.rows = 64
    compact_qk_bytes = 64 * 4 * 128 * 4
    scratch.prefill_query.nbytes = compact_qk_bytes
    scratch.prefill_key.nbytes = compact_qk_bytes

    with pytest.raises(RuntimeError, match="mode changed after session allocation"):
        runner._run_gdn_prefill(
            layer=_make_layer(),
            scratch=scratch,
            cfg=_make_cfg(),
            rows=64,
            recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
            stream=7,
            runtime="runtime-sentinel",
        )

    assert calls == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("chain_peer_wave32", "peer_wave32"),
        ("chain_peer_cluster8", "peer_cluster8"),
    ],
)
def test_run_gdn_prefill_explicit_peer_route_uses_normalized_prepare(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", mode)
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        prepare_peer_normalized=_recorder(calls, "peer_prepare"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        recurrent_peer_wave32=_recorder(calls, "peer_wave32"),
        recurrent_segments_peer_wave32=_recorder(calls, "segments_peer_wave32"),
        recurrent_peer_cluster8=_recorder(calls, "peer_cluster8"),
        recurrent_segments_peer_cluster8=_recorder(calls, "segments_peer_cluster8"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "peer_prepare",
        expected,
        "rmsnorm_gate",
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("chain_tile64", "exact_tile64"),
        ("chain_tile32", "exact_tile32"),
        ("chain_lds64", "exact_lds64"),
        ("chain_lds32", "exact_lds32"),
        ("chain_wave32", "exact_wave32"),
        ("chain_wave32_tree", "wave32_tree"),
    ],
)
def test_run_gdn_prefill_explicit_tiled_chain_selects_registered_variant(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", mode)
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "legacy_prepare"),
        recurrent=_recorder(calls, "legacy_recurrent"),
        recurrent_segments=_recorder(calls, "legacy_segments"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        exact_recurrent=_recorder(calls, "exact_recurrent"),
        exact_recurrent_segments=_recorder(calls, "exact_segments"),
        exact_recurrent_tile64=_recorder(calls, "exact_tile64"),
        exact_recurrent_segments_tile64=_recorder(calls, "exact_segments_tile64"),
        exact_recurrent_tile32=_recorder(calls, "exact_tile32"),
        exact_recurrent_segments_tile32=_recorder(calls, "exact_segments_tile32"),
        exact_recurrent_lds64=_recorder(calls, "exact_lds64"),
        exact_recurrent_segments_lds64=_recorder(calls, "exact_segments_lds64"),
        exact_recurrent_lds32=_recorder(calls, "exact_lds32"),
        exact_recurrent_segments_lds32=_recorder(calls, "exact_segments_lds32"),
        exact_recurrent_wave32=_recorder(calls, "exact_wave32"),
        exact_recurrent_segments_wave32=_recorder(calls, "exact_segments_wave32"),
        recurrent_wave32_tree=_recorder(calls, "wave32_tree"),
        recurrent_segments_wave32_tree=_recorder(calls, "segments_wave32_tree"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare",
        expected,
        "rmsnorm_gate",
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("chain_tile64", "exact_segments_tile64"),
        ("chain_lds64", "exact_segments_lds64"),
        ("chain_lds32", "exact_segments_lds32"),
        ("chain_wave32", "exact_segments_wave32"),
        ("chain_wave32_tree", "segments_wave32_tree"),
    ],
)
def test_run_gdn_prefill_candidate_chain_selects_matching_segment_variant(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", mode)
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=None,
        recurrent=None,
        recurrent_segments=None,
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        exact_recurrent_tile64=_recorder(calls, "exact_tile64"),
        exact_recurrent_segments_tile64=_recorder(calls, "exact_segments_tile64"),
        exact_recurrent_lds64=_recorder(calls, "exact_lds64"),
        exact_recurrent_segments_lds64=_recorder(calls, "exact_segments_lds64"),
        exact_recurrent_lds32=_recorder(calls, "exact_lds32"),
        exact_recurrent_segments_lds32=_recorder(calls, "exact_segments_lds32"),
        exact_recurrent_wave32=_recorder(calls, "exact_wave32"),
        exact_recurrent_segments_wave32=_recorder(calls, "exact_segments_wave32"),
        recurrent_wave32_tree=_recorder(calls, "wave32_tree"),
        recurrent_segments_wave32_tree=_recorder(calls, "segments_wave32_tree"),
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=1025,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0002),
        stream=0,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == [
        "exact_prepare",
        expected,
        "rmsnorm_gate",
    ]


def test_run_gdn_prefill_explicit_fused_overrides_available_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "fused")
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=_recorder(calls, "fused_decode_order"),
        exact_prepare=_recorder(calls, "exact_prepare"),
        exact_recurrent_lds32=_recorder(calls, "exact_lds32"),
        auto_mode="chain_lds32",
    )

    runner._run_gdn_prefill(
        layer=_make_layer(),
        scratch=_make_scratch(),
        cfg=_make_cfg(),
        rows=64,
        recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
        stream=7,
        runtime="runtime-sentinel",
    )

    assert [name for name, _ in calls] == ["fused_decode_order"]


@pytest.mark.parametrize(
    ("mode", "plan", "message"),
    [
        (
            "chain",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain' is unavailable",
        ),
        (
            "fused",
            qgr._GGUFGDNPrefillPlan(
                lambda *args, **kwargs: None,
                lambda *args, **kwargs: None,
                None,
                lambda *args, **kwargs: None,
                None,
            ),
            "explicit GGUF GDN prefill mode 'fused' is unavailable",
        ),
        (
            "chain_peer_wave32",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_peer_wave32' is unavailable",
        ),
        (
            "chain_peer_cluster8",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_peer_cluster8' is unavailable",
        ),
        (
            "chain_tile64",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_tile64' is unavailable",
        ),
        (
            "chain_tile32",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_tile32' is unavailable",
        ),
        (
            "chain_lds64",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_lds64' is unavailable",
        ),
        (
            "chain_lds32",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_lds32' is unavailable",
        ),
        (
            "chain_lds32_direct",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_lds32_direct' is unavailable",
        ),
        (
            "chain_wave32",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_wave32' is unavailable",
        ),
        (
            "chain_wave32_tree",
            qgr._GGUFGDNPrefillPlan(
                None, None, None, lambda *args, **kwargs: None, lambda *args, **kwargs: None
            ),
            "explicit GGUF GDN prefill mode 'chain_wave32_tree' is unavailable",
        ),
    ],
)
def test_run_gdn_prefill_explicit_unavailable_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    plan: qgr._GGUFGDNPrefillPlan,
    message: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", mode)
    runner = _new_runner()
    runner._gguf_gdn_prefill_plan_cache = plan

    with pytest.raises(RuntimeError, match=message):
        runner._run_gdn_prefill(
            layer=_make_layer(),
            scratch=_make_scratch(),
            cfg=_make_cfg(),
            rows=64,
            recurrent_state=SimpleNamespace(ptr=0xDEAD0001),
            stream=7,
            runtime="runtime-sentinel",
        )


def test_gdn_prefill_mode_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "maybe")
    with pytest.raises(ValueError, match="HIPENGINE_GGUF_GDN_PREFILL_MODE"):
        qgr._gguf_gdn_prefill_mode()


def test_gdn_prefill_exact_mode_rejects_quality_admitted_backend_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qgr,
        "backend_package_capability",
        lambda *_args, **_kwargs: "chain_peer_wave32",
    )
    with pytest.raises(RuntimeError, match="exact mode must be one of"):
        qgr._gguf_gdn_prefill_backend_exact_mode("hip_gfx1100")


def test_run_gdn_prefill_uses_chain_under_threshold_when_fused_missing() -> None:
    runner = _new_runner()
    calls: list[tuple[str, object]] = []
    runner._gguf_gdn_prefill_plan_cache = qgr._GGUFGDNPrefillPlan(
        prepare=_recorder(calls, "prepare"),
        recurrent=_recorder(calls, "recurrent_k2"),
        recurrent_segments=_recorder(calls, "recurrent_segments_k2"),
        rmsnorm_gate=_recorder(calls, "rmsnorm_gate"),
        fused_decode_order=None,
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
        fused_decode_order=None,
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
        fused_decode_order=None,
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
    assert qgr._gguf_gdn_prefill_segment_threshold() == 256
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "0")
    assert qgr._gguf_gdn_prefill_segment_threshold() == 1
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "128")
    assert qgr._gguf_gdn_prefill_segment_threshold() == 128
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", raising=False)
    assert qgr._gguf_gdn_prefill_segment_threshold() == 256


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
        "prefill_query_scale": SimpleNamespace(ptr=0xD5),
        "prefill_key_scale": SimpleNamespace(ptr=0xD6),
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


def test_fp16_recurrent_state_defaults_only_for_validated_gfx1151_q4ks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime.qwen35_gguf_runner import (
        _gguf_fp16_recurrent_state_enabled,
    )

    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    assert _gguf_fp16_recurrent_state_enabled(
        backend="hip_gfx1151",
        file_type_name="mostly_q4_k_s",
    )
    assert not _gguf_fp16_recurrent_state_enabled(
        backend="hip_gfx1151",
        file_type_name="mostly_q4_k_m",
    )
    assert not _gguf_fp16_recurrent_state_enabled(
        backend="hip_gfx1100",
        file_type_name="mostly_q4_k_s",
    )

    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "0")
    assert not _gguf_fp16_recurrent_state_enabled(
        backend="hip_gfx1151",
        file_type_name="mostly_q4_k_s",
    )
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    assert _gguf_fp16_recurrent_state_enabled(
        backend="hip_gfx1100",
        file_type_name="mostly_q4_k_m",
    )


def test_gdn_decode_order_state_rows_kernel_selects_fp16_under_flag(monkeypatch) -> None:
    """Verify-capture decode-order row-state writers route to fp16 under the flag.

    The fp16 flag must select the fp16-state writers (which operate on the
    half-sized recurrent-state buffers); with the flag off the strict FP32
    wrappers are the identity fallback.
    """
    import hipengine.runtime.qwen35_gguf_runner as runner_mod
    from hipengine.runtime.qwen35_gguf_runner import (
        _gdn_decode_order_segments_state_rows_kernel,
        _gdn_decode_order_state_rows_kernel,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state,
    )

    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    assert _gdn_decode_order_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy
    )
    assert _gdn_decode_order_segments_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy
    )

    monkeypatch.setenv("HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE", "1")
    assert _gdn_decode_order_segments_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy
    )
    monkeypatch.setattr(runner_mod, "physical_exact_rowtiles_enabled", lambda: True)
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE")
    assert _gdn_decode_order_segments_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce
    )
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE", "0")
    assert _gdn_decode_order_segments_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy
    )
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE", "1")
    assert _gdn_decode_order_segments_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce
    )

    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    assert _gdn_decode_order_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state
    )
    assert _gdn_decode_order_segments_state_rows_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state
    )


def test_gdn_state_rows_wave_reduce_wrapper_is_c5c8_physical_shape_scoped(
    monkeypatch,
) -> None:
    from hipengine.kernels.hip_gfx1100.linear_attn import gdn as gdn_mod

    calls = []
    monkeypatch.setattr(
        gdn_mod,
        "qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy",
        lambda *args, **kwargs: calls.append(kwargs.get("_symbol")),
    )
    base_args = list(range(19))
    for segments, total_tokens, expect_candidate in (
        (4, 12, False),
        (5, 15, True),
        (6, 24, True),
        (8, 32, True),
        (9, 27, False),
        (6, 30, False),
    ):
        args = base_args.copy()
        args[13] = total_tokens
        args[14] = segments
        gdn_mod.qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce(
            *args
        )
        assert bool(calls[-1]) is expect_candidate


def test_gdn_decode_order_segments_inplace_kernel_selects_fp16_under_flag(monkeypatch) -> None:
    """In-place segmented decode-order prefill writer routes to fp16 under the flag.

    The packed AR multi-slot prefill uses ``_gdn_decode_order_segments_inplace_kernel``
    (per-slot packed state mutated in place).  Under the fp16 flag it must select
    the fp16-state writer (half-sized per-slot state); with the flag off the
    strict FP32 wrapper is the identity fallback.
    """
    from hipengine.runtime.qwen35_gguf_runner import (
        _gdn_decode_order_segments_inplace_kernel,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments,
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state,
    )

    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    assert _gdn_decode_order_segments_inplace_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments
    )

    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    assert _gdn_decode_order_segments_inplace_kernel() is (
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state
    )
