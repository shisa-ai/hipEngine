from __future__ import annotations

from types import SimpleNamespace

from scripts.gguf_mtp_category_bench import build_summary


def _row(prompt_id: str, category: str, *, output: int, accepted: int, drafts: int, ar_ms: float, draft_ms: float) -> dict:
    return {
        "prompt_id": prompt_id,
        "category": category,
        "suite_category": category,
        "metrics": {
            "total_output_tokens": output,
            "total_accepted": accepted,
            "total_drafts": drafts,
            "total_cycle_ms": ar_ms + draft_ms,
        },
        "cycles": [
            {
                "ar_decode_ms": ar_ms,
                "mtp_draft_ms": draft_ms,
            }
        ],
    }


def test_category_summary_marks_b1_verifier_off_as_non_promotable() -> None:
    """A verifier-derived ``off`` row is not a true AR/no-MTP baseline.

    This prevents the native diagnostic category wrapper from being reused as a
    retained "MTP beats AR" speed table until the harness measures a separate
    autoregressive generation path over the same prompt suite.
    """
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [
        {"id": "code_1", "category": "code", "prompt": "write code"},
        {"id": "general_1", "category": "general_en", "prompt": "explain"},
    ]
    raw = {
        1: [
            _row("code_1", "code", output=1, accepted=0, drafts=1, ar_ms=10.0, draft_ms=2.0),
            _row("general_1", "general_en", output=2, accepted=1, drafts=1, ar_ms=20.0, draft_ms=3.0),
        ],
        5: [
            _row("code_1", "code", output=1, accepted=0, drafts=5, ar_ms=10.0, draft_ms=10.0),
            _row("general_1", "general_en", output=2, accepted=1, drafts=5, ar_ms=20.0, draft_ms=12.0),
        ],
    }

    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])

    assert summary["status"] == "diagnostic_retained"
    assert summary["performance_claim"] is False
    assert summary["speed_claim_eligible"] is False
    assert "true AR baseline" in summary["promotion_blocker"]
    assert summary["ar_baseline_contract"] == {
        "required_for_speed_claims": "true_no_mtp_autoregressive_generation",
        "current_off_kind": "verifier_derived_from_b1_target_ar",
        "current_off_true_autoregressive_path": False,
    }
    assert summary["totals"]["off"]["baseline_kind"] == "verifier_derived_from_b1_target_ar"
    assert summary["totals"]["off"]["true_autoregressive_path"] is False
    assert summary["categories"]["code"]["off"]["true_autoregressive_path"] is False
