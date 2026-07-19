"""Unit coverage for the P9.E3 GGUF decode graph bucket policy."""

from __future__ import annotations

import csv as _csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFDecodeGraphWeightRole,
    _resolve_gguf_attention_context_limit,
    build_qwen35_gguf_decode_graph_bucket_key,
    qwen35_gguf_decode_graph_active_symbol_groups,
)


def _load_graph_smoke_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "qwen35_gguf_decode_graph_smoke.py"
    module_name = "_qwen35_gguf_decode_graph_smoke_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SMOKE = _load_graph_smoke_module()


def _roles() -> tuple[Qwen35GGUFDecodeGraphWeightRole, ...]:
    return (
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_gate_exps", "gguf_q4_k_t16_v1", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_up_exps", "gguf_q4_k_t16_v1", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_down_exps", "gguf_q5_k_t16_v1", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.1.ffn_down_exps", "gguf_q6_k_t16_v1", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.attn_q", "gguf_q8_0_t16_v1", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_gate_shexp", "gguf_q8_0_t16_v1", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_up_shexp", "gguf_q8_0_t16_v1", 2),
        Qwen35GGUFDecodeGraphWeightRole("root.lm_head", "gguf_q6_k_t16_v1", 2),
    )


def _iq_roles() -> tuple[Qwen35GGUFDecodeGraphWeightRole, ...]:
    return (
        Qwen35GGUFDecodeGraphWeightRole("root.lm_head", "gguf_q6_k", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.attn_qkv", "gguf_q8_0", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_gate_inp", "bf16", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_gate_shexp", "gguf_q8_0", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_up_shexp", "gguf_q8_0", 2),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_gate_exps", "gguf_iq3_xxs", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_up_exps", "gguf_iq3_xxs", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.0.ffn_down_exps", "gguf_iq4_xs", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.39.ffn_gate_exps", "gguf_iq4_xs", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.39.ffn_up_exps", "gguf_iq4_xs", 3),
        Qwen35GGUFDecodeGraphWeightRole("layers.39.ffn_down_exps", "gguf_q6_k", 3),
    )


def test_decode_graph_bucket_key_tracks_replay_budget_and_active_p9_groups() -> None:
    key = build_qwen35_gguf_decode_graph_bucket_key(
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
        block_size=256,
        max_positions=768,
        is_moe=True,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        weight_roles=_roles(),
        use_gemv_decode=True,
    )

    assert key.active_c == 1
    assert key.replay_context_limit == 640
    assert key.context_bucket == 768
    assert key.replay_steps == 1
    assert key.max_replay_steps == 128
    assert key.decode_repack is True
    assert set(key.active_symbol_groups) == {
        "gdn_decode",
        "paged_kv_write",
        "paged_full_attention_decode",
        "moe_q4_k_selected_dual",
        "moe_q5_k_selected",
        "moe_q6_k_selected",
        "dense_q8_0_single",
        "dense_q8_0_dual",
        "dense_q6_k_lm_head",
    }


@pytest.mark.parametrize("use_gemv_decode", [False, True])
def test_decode_graph_iq_roles_require_fused_single_and_weighted_groups(use_gemv_decode: bool) -> None:
    groups = qwen35_gguf_decode_graph_active_symbol_groups(
        is_moe=True,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        weight_roles=_iq_roles(),
        use_gemv_decode=use_gemv_decode,
    )

    assert set(groups) == {
        "gdn_decode",
        "paged_kv_write",
        "paged_full_attention_decode",
        "moe_iq3_xxs_selected",
        "moe_iq4_xs_selected",
        "moe_iq4_xs_weighted_down",
        "moe_q6_k_selected",
        "dense_q8_0_single",
        "dense_q8_0_dual",
        "dense_q6_k_lm_head",
    }


def test_decode_graph_bucket_requires_dense_q4_only_when_active() -> None:
    roles = _roles() + (Qwen35GGUFDecodeGraphWeightRole("layers.0.attn_output", "gguf_q4_k", 2),)

    without_opt_in = qwen35_gguf_decode_graph_active_symbol_groups(
        is_moe=True,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        weight_roles=roles,
        use_gemv_decode=False,
    )
    with_opt_in = qwen35_gguf_decode_graph_active_symbol_groups(
        is_moe=True,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        weight_roles=roles,
        use_gemv_decode=True,
    )

    assert "dense_q4_k" not in without_opt_in
    assert "dense_q4_k" in with_opt_in


def test_decode_graph_attention_scan_bound_covers_the_full_replay_span() -> None:
    assert _resolve_gguf_attention_context_limit(
        position=512,
        max_context_len=None,
        max_positions=768,
    ) == (513, 513)
    assert _resolve_gguf_attention_context_limit(
        position=512,
        max_context_len=640,
        max_positions=768,
    ) == (513, 640)
    with pytest.raises(ValueError, match="smaller than the active context"):
        _resolve_gguf_attention_context_limit(
            position=512,
            max_context_len=512,
            max_positions=768,
        )
    with pytest.raises(ValueError, match="exceeds the resident cache"):
        _resolve_gguf_attention_context_limit(
            position=512,
            max_context_len=769,
            max_positions=768,
        )


def test_decode_graph_bucket_rejects_replay_budget_beyond_cache() -> None:
    with pytest.raises(ValueError, match="bucket exceeds resident cache"):
        build_qwen35_gguf_decode_graph_bucket_key(
            position=512,
            steps_per_replay=1,
            max_replay_steps=257,
            block_size=256,
            max_positions=768,
            is_moe=True,
            layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
            weight_roles=_roles(),
            use_gemv_decode=True,
        )


def test_decode_graph_symbol_coverage_accepts_all_active_groups() -> None:
    expected = (
        "moe_q4_k_selected_dual",
        "moe_q5_k_selected",
        "moe_q6_k_selected",
        "dense_q8_0_single",
        "dense_q8_0_dual",
        "dense_q6_k_lm_head",
        "gdn_decode",
        "paged_kv_write",
        "paged_full_attention_decode",
    )
    kernels = [
        "void (anonymous namespace)::q4_k_t16_selected_dual_silu_direct_gemv_kernel<unsigned short>(...)",
        "void (anonymous namespace)::qk_t16_selected_direct_gemv_kernel<unsigned short, 5>(...)",
        "void (anonymous namespace)::qk_t16_selected_direct_gemv_kernel<unsigned short, 6>(...)",
        "void (anonymous namespace)::q8_0_t16_gemv_kernel<unsigned short, unsigned short>(...)",
        "void (anonymous namespace)::q8_0_t16_dual_gemv_kernel<unsigned short, unsigned short>(...)",
        "void (anonymous namespace)::q6_k_t16_gemv_kernel<unsigned short, float>(...)",
        "void (anonymous namespace)::qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<unsigned short>(...)",
        "void (anonymous namespace)::qwen35_write_paged_kv_mixed_value_position_tensor_kernel<unsigned short>(...)",
        "(anonymous namespace)::qwen35_paged_full_attn_decode_context_tensor_kernel(...)",
    ]

    coverage = SMOKE.validate_decode_graph_symbol_coverage(kernels, expected_groups=expected)

    assert coverage["passed"] is True
    assert coverage["missing_symbol_groups"] == []
    assert set(coverage["observed_symbol_groups"]) == set(expected)


def test_decode_graph_symbol_coverage_accepts_iq_groups() -> None:
    expected = (
        "moe_iq3_xxs_selected",
        "moe_iq4_xs_selected",
        "moe_iq4_xs_weighted_down",
    )
    kernels = [
        "(anonymous namespace)::gguf_iq3_xxs_selected_dual_silu_gemv_kernel(...) ",
        "(anonymous namespace)::gguf_iq4_xs_selected_gemv_kernel(...) ",
        "(anonymous namespace)::gguf_iq4_xs_weighted_selected_down_kernel(...) ",
    ]

    coverage = SMOKE.validate_decode_graph_symbol_coverage(kernels, expected_groups=expected)

    assert coverage["passed"] is True
    assert coverage["missing_symbol_groups"] == []
    assert set(coverage["observed_symbol_groups"]) == set(expected)


def test_decode_graph_symbol_coverage_accepts_raw_legacy_fallback_groups() -> None:
    expected = (
        "moe_q6_k_selected",
        "dense_q8_0_single",
        "dense_q8_0_dual",
        "dense_q6_k_lm_head",
    )
    kernels = [
        "void (anonymous namespace)::gguf_k_selected_pack8_prefill_out_kernel<unsigned short, unsigned short, 6>(...)",
        "void (anonymous namespace)::gguf_k_pack8_prefill_out_kernel<unsigned short, unsigned short, 8>(...)",
        "void (anonymous namespace)::gguf_k_dual_prefill_out_kernel<unsigned short, unsigned short, 8>(...)",
        "void (anonymous namespace)::gguf_k_pack8_prefill_out_kernel<unsigned short, float, 6>(...)",
    ]

    coverage = SMOKE.validate_decode_graph_symbol_coverage(kernels, expected_groups=expected)

    assert coverage["passed"] is True
    assert coverage["missing_symbol_groups"] == []
    assert set(coverage["observed_symbol_groups"]) == set(expected)


def test_decode_graph_symbol_coverage_cli_fails_on_missing_group(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    with csv_path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=["Kernel_Name", "Start_Timestamp", "End_Timestamp"])
        writer.writeheader()
        writer.writerow(
            {
                "Kernel_Name": "void (anonymous namespace)::q4_k_t16_selected_dual_silu_direct_gemv_kernel<unsigned short>(...)",
                "Start_Timestamp": 0,
                "End_Timestamp": 1,
            }
        )

    rc = SMOKE.main(
        [
            "--coverage-only",
            "--coverage-csv",
            str(csv_path),
            "--expected-symbol-groups",
            "moe_q4_k_selected_dual,gdn_decode",
            "--json",
            str(tmp_path / "coverage.json"),
        ]
    )

    assert rc == 1
    payload = json.loads((tmp_path / "coverage.json").read_text())
    assert payload["decode_graph_symbol_coverage"]["missing_symbol_groups"] == ["gdn_decode"]
