from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.laguna_dflash_category_bench import (
    RETAINED_CHUNK_SIZE,
    _aggregate_scope,
    _correctness,
    _fixed_horizon_state_aligned,
    _pair_rows,
    _parse_args,
    _promotion_gate,
    _resolved_target_prefill_variants,
)


def _mode_row(
    *,
    prompt_id: str,
    category: str,
    repetition: int,
    mode: str,
    generated_ids: tuple[int, ...] = (1, 2, 3, 4),
    decode_seconds: float = 1.0,
) -> dict[str, object]:
    common: dict[str, object] = {
        "mode": mode,
        "prompt_id": prompt_id,
        "category": category,
        "prompt_tokens": 8,
        "prompt_token_ids_sha256": f"prompt-{prompt_id}",
        "generated_ids": list(generated_ids),
        "generated_ids_sha256": f"ids-{generated_ids}",
        "output_tokens": 4,
        "decode_output_tokens": 3,
        "ttft_seconds": 2.0 if mode == "ar" else 4.0,
        "prefill_tok_s": 4.0 if mode == "ar" else 2.0,
        "decode_seconds": decode_seconds,
        "decode_tok_s": 3 / decode_seconds,
        "total_seconds": (2.0 if mode == "ar" else 4.0) + decode_seconds,
        "e2e_output_tok_s": 4 / ((2.0 if mode == "ar" else 4.0) + decode_seconds),
        "repetition": repetition,
        "target_position": 10,
        "expected_target_position": 10,
        "state_aligned": True,
    }
    if mode == "ar":
        common["finite_logits"] = True
        return common
    common.update(
        {
            "finite_draft_logits": True,
            "finite_verify_logits": True,
            "cycles": [],
            "cycle_count": 2,
            "accepted_lengths": [2, 1],
            "accepted_draft_tokens": 3,
            "draft_tokens_proposed": 8,
            "target_verify_rows": 10,
            "target_verify_rows_per_output_token": 10 / 3,
            "draft_acceptance": 3 / 8,
            "accepted_per_output": 1.0,
            "proposal_seconds": 0.2,
            "target_verify_seconds": 0.4,
            "draft_commit_enqueue_seconds": 0.1,
            "cycle_host_seconds": 0.8,
            "cycle_wall_seconds": decode_seconds,
            "post_verify_residual_seconds": max(0.0, decode_seconds - 0.6),
            "all_verifier_addresses_stable": True,
            "drafter_context_tokens": 11,
        }
    )
    return common


def _pairs(*, spec_decode_seconds: float = 0.8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prompts = (
        ("code_merge_intervals", "code"),
        ("code_markdown_table", "code"),
    )
    for repetition in (0, 1):
        for prompt_id, category in prompts:
            rows.append(
                _mode_row(
                    prompt_id=prompt_id,
                    category=category,
                    repetition=repetition,
                    mode="ar",
                )
            )
            rows.append(
                _mode_row(
                    prompt_id=prompt_id,
                    category=category,
                    repetition=repetition,
                    mode="dflash",
                    decode_seconds=spec_decode_seconds,
                )
            )
    return _pair_rows(
        rows,
        candidate_budget=4,
        peak_allocated_bytes=100,
        allocated_after_load_bytes=90,
        backend="hip_gfx1151",
    )


def test_laguna_dflash_accepts_direct_target_residency_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "laguna_dflash_category_bench.py",
            "/models/target.gguf",
            "/models/drafter",
            "--direct-gguf",
            "--safety-reserve-gib",
            "4",
            "--quant-label",
            "UD-Q2_K_XL",
            "--iq3-selected-down-tile",
            "4",
        ],
    )

    args = _parse_args()

    assert args.direct_gguf is True
    assert args.safety_reserve_gib == pytest.approx(4.0)
    assert args.quant_label == "UD-Q2_K_XL"
    assert args.iq3_selected_down_tile == 4


def test_laguna_dflash_refresh_uses_promoted_chunk_policy() -> None:
    assert RETAINED_CHUNK_SIZE == 128
    target = SimpleNamespace(
        kv_cache=SimpleNamespace(
            layers=(
                SimpleNamespace(
                    attention_type="full_attention",
                    attention_prefill_variant="global_context_rows_spans",
                ),
            )
        ),
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    assert _resolved_target_prefill_variants(target) == (
        "global_context_rows_spans",
        "swa_context_rows_wave32_exact_spans",
    )


def test_laguna_dflash_fixed_horizon_accepts_prediction_or_committed_boundary() -> None:
    assert _fixed_horizon_state_aligned(
        target_position=30,
        drafter_context_tokens=31,
        expected_prediction_position=30,
    )
    assert _fixed_horizon_state_aligned(
        target_position=31,
        drafter_context_tokens=32,
        expected_prediction_position=30,
    )
    assert not _fixed_horizon_state_aligned(
        target_position=32,
        drafter_context_tokens=33,
        expected_prediction_position=30,
    )
    assert not _fixed_horizon_state_aligned(
        target_position=31,
        drafter_context_tokens=31,
        expected_prediction_position=30,
    )


def test_laguna_dflash_scope_uses_weighted_true_ar_and_acceptance_counts() -> None:
    metrics = _aggregate_scope(_pairs(spec_decode_seconds=0.8))

    assert metrics["ar"]["decode_tok_s_weighted"] == pytest.approx(3.0)
    assert metrics["dflash"]["decode_tok_s_weighted"] == pytest.approx(3.75)
    assert metrics["comparison"]["decode_speedup_vs_true_ar"] == pytest.approx(1.25)
    assert metrics["comparison"]["ttft_speedup_vs_true_ar"] == pytest.approx(0.5)
    assert metrics["dflash"]["draft_acceptance"] == pytest.approx(3 / 8)
    assert metrics["dflash"]["target_verify_rows_per_output"] == pytest.approx(10 / 3)
    assert metrics["unique_prompts"] == 2


def test_laguna_dflash_correctness_fails_closed_on_pair_or_repeat_drift() -> None:
    pairs = _pairs()
    assert _correctness(pairs)["pass"] is True

    pairs[-1]["raw"]["dflash"]["generated_ids"][-1] = 99
    pairs[-1]["raw"]["dflash"]["generated_ids_sha256"] = "changed"
    failed = _correctness(pairs)
    assert failed["pass"] is False
    assert failed["same_mode_repeat_deterministic"] is False
    assert failed["pair_rows"][-1]["exact_match_ar"] is False


def test_laguna_dflash_promotion_requires_exact_protocol_and_gt_1p10() -> None:
    fast_pairs = _pairs(spec_decode_seconds=0.8)
    fast = _aggregate_scope(fast_pairs)
    fast_heldout = _aggregate_scope(
        [row for row in fast_pairs if row["split"] == "heldout"]
    )
    fast_categories = {
        "code": _aggregate_scope(
            [row for row in fast_pairs if row["prompt"]["category"] == "code"]
        )
    }
    accepted = _promotion_gate(
        correctness_passed=True,
        protocol_eligible=True,
        full_metrics=fast,
        heldout_metrics=fast_heldout,
        category_metrics=fast_categories,
    )
    assert accepted["pass"] is True

    slow_pairs = _pairs(spec_decode_seconds=1.0)
    slow = _aggregate_scope(slow_pairs)
    rejected = _promotion_gate(
        correctness_passed=True,
        protocol_eligible=True,
        full_metrics=slow,
        heldout_metrics=_aggregate_scope(
            [row for row in slow_pairs if row["split"] == "heldout"]
        ),
        category_metrics={"code": slow},
    )
    assert rejected["pass"] is False
    assert "full_suite_decode_speedup_gt_1p10" in rejected["failed_checks"]

    incorrect = _promotion_gate(
        correctness_passed=False,
        protocol_eligible=True,
        full_metrics=fast,
        heldout_metrics=fast_heldout,
        category_metrics=fast_categories,
    )
    assert incorrect["pass"] is False
    assert "exact_finite_state_correctness" in incorrect["failed_checks"]


def test_laguna_dflash_promotion_rejects_heldout_or_category_regression() -> None:
    fast_pairs = _pairs(spec_decode_seconds=0.8)
    fast = _aggregate_scope(fast_pairs)
    regressive = _aggregate_scope(_pairs(spec_decode_seconds=1.1))

    rejected = _promotion_gate(
        correctness_passed=True,
        protocol_eligible=True,
        full_metrics=fast,
        heldout_metrics=regressive,
        category_metrics={"code": fast, "general_en": regressive},
    )

    assert rejected["pass"] is False
    assert "heldout_decode_non_regression" in rejected["failed_checks"]
    assert "category_general_en_decode_non_regression" in rejected["failed_checks"]
    assert rejected["category_decode_speedups_vs_true_ar"]["code"] == pytest.approx(1.25)
