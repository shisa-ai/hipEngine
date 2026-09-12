from __future__ import annotations

import json
from pathlib import Path

from scripts.mtp_acceptance_status_report import build_report, render_markdown


def test_build_report_aligns_acceptance_denominators(tmp_path: Path) -> None:
    compare = tmp_path / "compare.json"
    diagnostics = tmp_path / "diagnostics.json"
    b2_sweep = tmp_path / "b2_sweep.json"
    b2_count = tmp_path / "b2_count.json"
    b2_minimal = tmp_path / "b2_minimal.json"
    b1_visible = tmp_path / "b1_visible.json"
    greeting = tmp_path / "greeting.json"
    llama_hip = tmp_path / "llama_hip.json"
    llama_vulkan = tmp_path / "llama_vulkan.json"

    compare.write_text(json.dumps(_compare_summary()))
    diagnostics.write_text(json.dumps(_diagnostics_summary()))
    b2_sweep.write_text(json.dumps(_b2_sweep()))
    b2_count.write_text(json.dumps(_metrics_artifact(0.2, 0.2857, 1.24, 0.91)))
    b2_minimal.write_text(json.dumps(_metrics_artifact(0.1, 0.1667, 1.03, 0.78)))
    b1_visible.write_text(json.dumps(_metrics_artifact(0.15, 0.1304, 1.02, 0.94)))
    greeting.write_text(json.dumps(_greeting_comparison()))
    llama_hip.write_text(json.dumps(_llama_summary(91.11, 0.7430555556)))
    llama_vulkan.write_text(json.dumps(_llama_summary(108.96, 0.7465277778)))

    report = build_report(
        compare_summary_path=compare,
        diagnostics_summary_path=diagnostics,
        b2_sweep_path=b2_sweep,
        b2_count_path=b2_count,
        b2_minimal_path=b2_minimal,
        b1_visible_path=b1_visible,
        greeting_comparison_path=greeting,
        llamacpp_hip_path=llama_hip,
        llamacpp_vulkan_path=llama_vulkan,
    )

    assert report["status"] == "diagnostic_recorded"
    assert report["headline"]["current_status"] == (
        "behind_llamacpp_and_candidate_set_parity_incomplete"
    )
    assert report["headline"]["hipengine_b1_accepted_per_output"] == 0.3600970227
    assert report["headline"]["llamacpp_hip_b4_accepted_per_output"] == 0.7430555556
    assert report["current_broad_comparison"]["gaps_vs_llamacpp_b4"]["hip"][
        "classification"
    ] == "behind_llamacpp_b4"
    sweep = report["native_gguf_diagnostics"]["b2_three_prompt_sweep"]
    assert sweep["total_accepted_draft_tokens"] == 3
    assert sweep["total_generated_draft_tokens"] == 30
    assert sweep["accept_per_draft"] == 0.1
    assert sweep["accepted_per_output"] == 3 / 18
    assert sweep["best_warm_prompt"]["slug"] == "count"
    assert report["denominator_notes"]["accepted_per_output"].startswith(
        "accepted draft tokens"
    )

    markdown = render_markdown(report)
    assert "MTP acceptance status" in markdown
    assert "llama.cpp HIP UD-Q4_K_M" in markdown
    assert "accepted/output" in markdown


def _compare_summary() -> dict:
    return {
        "results": {
            "hipengine_paro_mtp_b1": {
                "artifact": "hipengine.json",
                "budget": 1,
                "prompts": 9,
                "all_exact_ar_match": True,
                "actual_decode_speedup_vs_ar_prompt_mean": 0.9115820665,
                "actual_decode_speedup_vs_ar_total_time": 0.9040705392,
                "ar_decode_tok_s_prompt_mean": 65.3747955174,
                "mtp_decode_tok_s_prompt_mean": 59.5599940076,
                "cycle_wall_ms_per_cycle": 25.9593199532,
                "verify_ms_per_cycle": 22.1565800753,
                "proposal_update_ms_per_cycle": 3.778407603,
                "visible_tokens_per_cycle": 1.5627369076,
                "accepted_tokens_per_cycle": 0.5627369076,
                "accepted_per_output": 0.3600970227,
                "decode_tokens": 32,
            }
        }
    }


def _llama_summary(mean_tps: float, accepted_per_output: float) -> dict:
    return {
        "rows": [
            {
                "mode": "base",
                "total_predicted": 288,
                "wall_tps": 42.0,
                "mean_tps": 62.0,
                "median_tps": 62.0,
                "total_draft": 0,
                "total_accepted": 0,
                "accept_rate": None,
                "accepted_per_output": 0.0,
                "draft_per_output": 0.0,
                "speedup_mean_vs_base": 1.0,
                "speedup_wall_vs_base": 1.0,
            },
            {
                "mode": "b4",
                "total_predicted": 288,
                "wall_tps": 52.0,
                "mean_tps": mean_tps,
                "median_tps": mean_tps,
                "total_draft": 234,
                "total_accepted": 214,
                "accept_rate": 0.9145,
                "accepted_per_output": accepted_per_output,
                "draft_per_output": 0.8125,
                "speedup_mean_vs_base": 1.79,
                "speedup_wall_vs_base": 1.41,
            },
        ]
    }


def _b2_sweep() -> dict:
    return {
        "cycles_per_prompt": 5,
        "aggregate": {
            "mean_warm_speedup_vs_ar_visible": 1.0191,
            "mean_cold_speedup_vs_ar_visible": 0.775,
            "warm_positive_prompts": 2,
            "depth2_accept_prompt_count": 1,
            "total_accepted_draft_tokens": 3,
            "total_generated_draft_tokens": 30,
        },
        "prompts": [
            {
                "slug": "capital",
                "warm_speedup_vs_ar_visible": 1.0357,
                "warm_accept_per_draft": 0.125,
                "accepted_per_output": 0.1667,
            },
            {
                "slug": "count",
                "warm_speedup_vs_ar_visible": 1.2347,
                "warm_accept_per_draft": 0.25,
                "accepted_per_output": 0.2857,
            },
            {
                "slug": "greeting",
                "warm_speedup_vs_ar_visible": 0.7869,
                "warm_accept_per_draft": 0.0,
                "accepted_per_output": 0.0,
            },
        ],
    }


def _metrics_artifact(
    accept_per_draft: float,
    accepted_per_output: float,
    warm_speedup: float,
    cold_speedup: float,
) -> dict:
    return {
        "metrics": {
            "accept_per_draft": accept_per_draft,
            "accepted_per_output": accepted_per_output,
            "tokens_per_sec": 14.0,
            "speedup_vs_ar_visible": cold_speedup,
            "total_accepted": 2,
            "total_drafts": 10,
            "warm_excluding_cycle0": {
                "accept_per_draft": accept_per_draft + 0.05,
                "accepted_per_output": accepted_per_output + 0.05,
                "tokens_per_sec": 19.0,
                "speedup_vs_ar_visible": warm_speedup,
                "total_accepted": 2,
                "total_drafts": 8,
            },
        }
    }


def _greeting_comparison() -> dict:
    return {
        "finding": "target absent from native top-k",
        "native_artifact": "native-topk.json",
        "llamacpp_trace": "llamacpp-trace.json",
        "llamacpp_summary": {
            "draft_acceptance": 1.0,
            "draft_n_accepted": 3,
        },
        "llamacpp_response_reasoning_content": "Thinking Process:\n\n1.",
    }


def _diagnostics_summary() -> dict:
    item = {
        "mtp_decode_tok_s_prompt_mean": 59.7,
        "ar_decode_tok_s_prompt_mean": 65.2,
        "speedup_prompt_mean": 0.916,
        "accepted_total": 99,
        "accepted_per_output_aggregate": 0.34375,
        "accept_per_active_budget_aggregate": 0.5439560439,
        "accepted_tokens_per_cycle_aggregate": 0.5439560439,
    }
    return {
        "hipengine": {
            "exact_b1_decode_batched": item,
            "exact_b3_decode_batched": dict(
                item,
                mtp_decode_tok_s_prompt_mean=47.6,
                accepted_total=131,
                accepted_per_output_aggregate=0.4548611111,
            ),
            "exact_b1_c1_loop": dict(item, mtp_decode_tok_s_prompt_mean=57.6),
        }
    }
