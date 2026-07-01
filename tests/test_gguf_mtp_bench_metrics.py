from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from scripts import gguf_mtp_bench as bench
from scripts.gguf_mtp_bench import (
    _draft_top1_prob,
    _diagnostic_topk_candidate_count,
    _rope_tables,
    apply_llama_compat_args,
    build_arg_parser,
    compute_speculative_metrics,
    count_topk_draft_candidates,
    llama_cpp_acceptance_from_target_samples,
    llama_cpp_mtp_catchup_rows,
    root_topk_acceptance_from_target_samples,
    select_topk_tokens,
    should_use_fused_b1_block_probe,
    sibling_topk_acceptance_from_target_samples,
    target_block_direct_commit_is_exact,
    target_block_needs_linear_state_snapshot,
    target_membership_in_draft_topk,
    validate_draft_n_max,
)


def test_default_mtp_policy_is_b1_root_top40_speed_first() -> None:
    args = build_arg_parser().parse_args([])

    assert args.draft_n_max == 1
    assert args.root_topk_accept == 40
    assert args.sibling_topk_accept == 1
    assert args.topk_branch_redraft is False
    assert args.mtp_draft_warmup is True
    assert args.target_graph_verify is True
    assert args.adaptive_block_after_full_accept is False
    assert args.adaptive_probe_draft_n_max == 3


def test_mtp_policy_accepts_adaptive_production_selector_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--draft-n-max",
            "5",
            "--resident-mtp-draft",
            "--target-block-verify",
            "--adaptive-block-after-full-accept",
            "--adaptive-probe-draft-n-max",
            "3",
            "--adaptive-ar-fallback",
        ]
    )

    assert args.draft_n_max == 5
    assert args.resident_mtp_draft is True
    assert args.target_block_verify is True
    assert args.adaptive_block_after_full_accept is True
    assert args.adaptive_probe_draft_n_max == 3
    assert args.adaptive_ar_fallback is True


def test_b1_branch_safe_block_verify_flag_parses_with_target_block_verify() -> None:
    args = build_arg_parser().parse_args(
        ["--target-block-verify", "--target-b1-branch-safe-block-verify"]
    )
    assert args.target_block_verify is True
    assert args.target_b1_branch_safe_block_verify is True


def test_fused_b1_block_probe_flag_parses() -> None:
    args = build_arg_parser().parse_args(["--fused-b1-block-probe"])

    assert args.fused_b1_block_probe is True


def test_dense_q8_shared_dp4a_flag_parses_default_off() -> None:
    args = build_arg_parser().parse_args([])
    assert args.verify_dense_q8_dp4a_shared is False

    args = build_arg_parser().parse_args(["--verify-dense-q8-dp4a-shared"])
    assert args.verify_dense_q8_dp4a_shared is True


def test_dense_q8_f32_dp4a_flag_parses_default_off() -> None:
    args = build_arg_parser().parse_args([])
    assert args.verify_dense_q8_dp4a_f32 is False

    args = build_arg_parser().parse_args(["--verify-dense-q8-dp4a-f32"])
    assert args.verify_dense_q8_dp4a_f32 is True


def test_resident_mtp_draft_dense_q8_dp4a_flag_parses_default_off() -> None:
    args = build_arg_parser().parse_args([])
    assert args.resident_mtp_draft_dense_q8_dp4a is False

    args = build_arg_parser().parse_args(["--resident-mtp-draft-dense-q8-dp4a"])
    assert args.resident_mtp_draft_dense_q8_dp4a is True


def test_fused_b1_block_probe_selector_is_only_for_adaptive_b1_probe() -> None:
    kwargs = {
        "fused_b1_block_probe": True,
        "adaptive_block_after_full_accept": True,
        "adaptive_block_verify_active": False,
        "cycle_ar_fallback": False,
        "cycle_draft_window": 1,
        "adaptive_probe_draft_n_max": 1,
        "cycle_root_topk_accept": 1,
        "cycle_sibling_topk_accept": 1,
        "topk_branch_redraft": False,
    }

    assert should_use_fused_b1_block_probe(**kwargs) is True
    assert should_use_fused_b1_block_probe(**{**kwargs, "adaptive_block_verify_active": True}) is False
    assert should_use_fused_b1_block_probe(**{**kwargs, "cycle_draft_window": 5}) is False
    assert should_use_fused_b1_block_probe(**{**kwargs, "cycle_root_topk_accept": 40}) is False


def test_fused_b1_block_probe_validation_rejects_inert_config(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--fused-b1-block-probe"])

    assert excinfo.value.code == 2
    assert "--fused-b1-block-probe requires --target-block-verify" in capsys.readouterr().err


def test_compute_speculative_metrics_counts_visible_accepted_tokens() -> None:
    """Accepted draft tokens are visible outputs, not just diagnostic accepts."""
    cycles = [
        {
            "generated_draft_tokens": 1,
            "accepted_draft_tokens": 0,
            "visible_output_tokens": 1,
            "ar_decode_ms": 50.0,
            "mtp_draft_ms": 7.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 1,
            "target_verify_serial_rows": 1,
        },
        {
            "generated_draft_tokens": 1,
            "accepted_draft_tokens": 1,
            "visible_output_tokens": 2,
            "ar_decode_ms": 50.0,
            "mtp_draft_ms": 7.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 2,
            "target_verify_block_passes": 1,
            "target_verify_block_rows": 2,
            "target_verify_direct_commit_rows": 1,
        },
        {
            "generated_draft_tokens": 1,
            "accepted_draft_tokens": 0,
            "visible_output_tokens": 1,
            "ar_decode_ms": 50.0,
            "mtp_draft_ms": 7.0,
            "target_verify_layer_passes": 2,
            "target_verify_rows_evaluated": 3,
            "target_verify_replay_rows": 1,
            "target_verify_discarded_rows": 1,
        },
        {
            "generated_draft_tokens": 1,
            "accepted_draft_tokens": 0,
            "visible_output_tokens": 1,
            "ar_decode_ms": 50.0,
            "mtp_draft_ms": 7.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 1,
            "target_verify_graph_rows": 1,
        },
        {
            "generated_draft_tokens": 1,
            "accepted_draft_tokens": 0,
            "visible_output_tokens": 1,
            "ar_decode_ms": 50.0,
            "mtp_draft_ms": 7.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 1,
            "target_verify_serial_rows": 1,
        },
    ]

    metrics = compute_speculative_metrics(cycles)

    assert metrics["total_drafts"] == 5
    assert metrics["total_accepted"] == 1
    assert metrics["verify_cycle_count"] == 5
    assert metrics["total_output_tokens"] == 6
    assert metrics["accept_per_draft"] == 0.2
    assert metrics["accepted_per_output"] == 1 / 6
    assert metrics["visible_tokens_per_cycle"] == 1.2
    assert metrics["avg_cycle_ms"] == 57.0
    assert metrics["avg_ms_per_visible_token"] == 47.5
    assert metrics["tokens_per_sec"] == 1000.0 / 47.5
    assert metrics["ar_baseline_tokens_per_sec"] == 1000.0 * 6 / 250.0
    assert metrics["speedup_vs_ar_visible"] == (1000.0 / 47.5) / (1000.0 * 6 / 250.0)
    assert metrics["target_verify_layer_passes"] == 6
    assert metrics["target_verify_rows_evaluated"] == 8
    assert metrics["target_verify_serial_rows"] == 2
    assert metrics["target_verify_graph_rows"] == 1
    assert metrics["target_verify_block_passes"] == 1
    assert metrics["target_verify_block_rows"] == 2
    assert metrics["target_verify_replay_rows"] == 1
    assert metrics["target_verify_direct_commit_rows"] == 1
    assert metrics["target_verify_discarded_rows"] == 1
    assert metrics["target_verify_layer_passes_per_output"] == 1.0
    assert metrics["target_verify_rows_per_output"] == 8 / 6
    assert metrics["target_verify_replay_rows_per_output"] == 1 / 6
    assert metrics["cycle_histograms"] == {
        "generated_draft_tokens": {"1": 5},
        "accepted_draft_tokens": {"0": 4, "1": 1},
        "visible_output_tokens": {"1": 4, "2": 1},
        "target_verify_layer_passes": {"1": 4, "2": 1},
        "target_verify_rows_evaluated": {"1": 3, "2": 1, "3": 1},
        "target_verify_block_rows": {"0": 4, "2": 1},
        "target_verify_replay_rows": {"0": 4, "1": 1},
        "target_verify_direct_commit_rows": {"0": 4, "1": 1},
        "target_verify_discarded_rows": {"0": 4, "1": 1},
        "target_verify_rows_minus_visible_output": {"0": 4, "2": 1},
    }
    assert metrics["denominators"] == {
        "accept_per_draft": "accepted_draft_tokens / generated_draft_tokens",
        "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
        "visible_tokens_per_cycle": "visible_output_token_count / verify_cycle_count",
        "tokens_per_sec": "visible_output_token_count / total_cycle_wall_time",
        "target_verify_layer_passes_per_output": "target layer-streaming passes / visible_output_token_count",
        "target_verify_rows_per_output": "target verifier rows evaluated / visible_output_token_count",
        "target_verify_replay_rows_per_output": "accepted-prefix replay rows / visible_output_token_count",
    }


@pytest.mark.parametrize(
    ("cycle_update", "message"),
    [
        ({"generated_draft_tokens": None}, r"cycles\[0\]\.generated_draft_tokens must be an integer"),
        ({"accepted_draft_tokens": True}, r"cycles\[0\]\.accepted_draft_tokens must be an integer"),
        ({"accepted_draft_tokens": -1}, r"cycles\[0\]\.accepted_draft_tokens must be non-negative"),
        ({"visible_output_tokens": 0}, r"cycles\[0\]\.visible_output_tokens must be positive"),
        ({"ar_decode_ms": False}, r"cycles\[0\]\.ar_decode_ms must be numeric"),
        ({"mtp_draft_ms": float("nan")}, r"cycles\[0\]\.mtp_draft_ms must be finite"),
        ({"mtp_draft_ms": -1.0}, r"cycles\[0\]\.mtp_draft_ms must be non-negative"),
        ({"target_verify_layer_passes": 1.0}, r"cycles\[0\]\.target_verify_layer_passes must be an integer"),
        ({"target_verify_rows_evaluated": -1}, r"cycles\[0\]\.target_verify_rows_evaluated must be non-negative"),
    ],
)
def test_compute_speculative_metrics_rejects_malformed_explicit_cycle_fields(
    cycle_update: dict[str, object], message: str
) -> None:
    cycle = {
        "generated_draft_tokens": 1,
        "accepted_draft_tokens": 0,
        "visible_output_tokens": 1,
        "ar_decode_ms": 50.0,
        "mtp_draft_ms": 7.0,
    }
    cycle.update(cycle_update)

    with pytest.raises(ValueError, match=message):
        compute_speculative_metrics([cycle])


def test_compute_speculative_metrics_allows_ar_fallback_zero_draft_cycle() -> None:
    metrics = compute_speculative_metrics(
        [
            {
                "generated_draft_tokens": 0,
                "accepted_draft_tokens": 0,
                "visible_output_tokens": 1,
                "ar_decode_ms": 18.0,
                "mtp_draft_ms": 0.0,
            }
        ]
    )

    assert metrics["total_drafts"] == 0
    assert metrics["total_accepted"] == 0
    assert metrics["accept_per_draft"] == 0.0
    assert metrics["tokens_per_sec"] == 1000.0 / 18.0


def test_compute_speculative_metrics_aggregates_optional_stage_timings() -> None:
    metrics = compute_speculative_metrics(
        [
            {
                "generated_draft_tokens": 1,
                "accepted_draft_tokens": 1,
                "visible_output_tokens": 2,
                "ar_decode_ms": 20.0,
                "mtp_draft_ms": 5.0,
                "cycle_wall_ms": 26.5,
                "stage_timings_ms": {
                    "draft_initial": 5.0,
                    "target_block_forward": 18.0,
                    "accept_policy_and_seed": 0.5,
                },
            },
            {
                "generated_draft_tokens": 1,
                "accepted_draft_tokens": 0,
                "visible_output_tokens": 1,
                "ar_decode_ms": 18.0,
                "mtp_draft_ms": 4.0,
                "cycle_wall_ms": 23.0,
                "stage_timings_ms": {
                    "draft_initial": 4.0,
                    "target_serial_verify_step": 18.0,
                    "accept_policy_and_seed": 0.25,
                },
            },
        ]
    )

    assert metrics["cycle_wall_ms_total"] == 49.5
    assert metrics["cycle_wall_ms_per_output"] == 16.5
    assert metrics["cycle_wall_over_legacy_ms_total"] == 2.5
    assert metrics["cycle_wall_over_legacy_ms_per_output"] == pytest.approx(2.5 / 3)
    assert metrics["stage_timing_totals_ms"] == {
        "accept_policy_and_seed": 0.75,
        "draft_initial": 9.0,
        "target_block_forward": 18.0,
        "target_serial_verify_step": 18.0,
    }
    assert metrics["stage_timing_per_output_ms"]["draft_initial"] == 3.0
    assert metrics["stage_timing_per_cycle_ms"]["accept_policy_and_seed"] == 0.375


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("generated_draft_tokens", r"cycles\[0\]\.generated_draft_tokens is required"),
        ("accepted_draft_tokens", r"cycles\[0\]\.accepted_draft_tokens is required"),
        ("visible_output_tokens", r"cycles\[0\]\.visible_output_tokens is required"),
        ("ar_decode_ms", r"cycles\[0\]\.ar_decode_ms is required"),
        ("mtp_draft_ms", r"cycles\[0\]\.mtp_draft_ms is required"),
    ],
)
def test_compute_speculative_metrics_rejects_missing_explicit_cycle_fields(missing_field: str, message: str) -> None:
    cycle = {
        "generated_draft_tokens": 1,
        "accepted_draft_tokens": 0,
        "visible_output_tokens": 1,
        "ar_decode_ms": 50.0,
        "mtp_draft_ms": 7.0,
    }
    del cycle[missing_field]

    with pytest.raises(ValueError, match=message):
        compute_speculative_metrics([cycle])


def test_compute_speculative_metrics_rejects_legacy_accepted_boolean_fallback() -> None:
    with pytest.raises(ValueError, match=r"cycles\[0\]\.generated_draft_tokens is required"):
        compute_speculative_metrics([{"accepted": True, "ar_decode_ms": 50.0, "mtp_draft_ms": 7.0}])


def test_compute_speculative_metrics_rejects_impossible_cycle_counts() -> None:
    with pytest.raises(ValueError, match=r"accepted_draft_tokens must be <= generated_draft_tokens"):
        compute_speculative_metrics([
            {
                "generated_draft_tokens": 1,
                "accepted_draft_tokens": 2,
                "visible_output_tokens": 2,
                "ar_decode_ms": 50.0,
                "mtp_draft_ms": 7.0,
            }
        ])
    with pytest.raises(ValueError, match=r"accepted_draft_tokens must be <= visible_output_tokens"):
        compute_speculative_metrics([
            {
                "generated_draft_tokens": 2,
                "accepted_draft_tokens": 2,
                "visible_output_tokens": 1,
                "ar_decode_ms": 50.0,
                "mtp_draft_ms": 7.0,
            }
        ])


@pytest.mark.parametrize(
    ("cycle_update", "message"),
    [
        ({"cycle_wall_ms": "bad"}, r"cycles\[0\]\.cycle_wall_ms must be numeric"),
        ({"stage_timings_ms": []}, r"cycles\[0\]\.stage_timings_ms must be an object"),
        ({"stage_timings_ms": {"draft_initial": -1.0}}, r"stage_timings_ms\.draft_initial must be non-negative"),
    ],
)
def test_compute_speculative_metrics_rejects_malformed_stage_timings(
    cycle_update: dict[str, object], message: str
) -> None:
    cycle = {
        "generated_draft_tokens": 1,
        "accepted_draft_tokens": 0,
        "visible_output_tokens": 1,
        "ar_decode_ms": 50.0,
        "mtp_draft_ms": 7.0,
    }
    cycle.update(cycle_update)

    with pytest.raises(ValueError, match=message):
        compute_speculative_metrics([cycle])


def test_select_topk_tokens_returns_descending_tokens_and_greedy() -> None:
    logits = np.array([0.1, 4.0, -1.0, 2.5, 3.0], dtype=np.float32)

    greedy, top3 = select_topk_tokens(logits, k=3)

    assert greedy == 1
    assert top3 == [1, 4, 3]


def test_select_topk_tokens_uses_argmax_for_top1() -> None:
    logits = np.array([0.1, 4.0, -1.0, 2.5, 3.0], dtype=np.float32)

    greedy, top1 = select_topk_tokens(logits, k=1)

    assert greedy == 1
    assert top1 == [1]


def test_draft_top1_prob_matches_softmax_argmax_probability() -> None:
    logits = np.array([0.0, 2.0, 1.0], dtype=np.float32)

    prob = _draft_top1_prob(logits)

    expected = float(np.exp(2.0) / (np.exp(0.0) + np.exp(2.0) + np.exp(1.0)))
    assert prob == pytest.approx(expected)


def test_count_topk_draft_candidates_honors_sibling_max_depth() -> None:
    assert count_topk_draft_candidates(
        5,
        root_topk_accept=4,
        sibling_topk_accept=9,
        sibling_topk_max_depth=3,
    ) == 32
    assert count_topk_draft_candidates(
        5,
        root_topk_accept=4,
        sibling_topk_accept=9,
        sibling_topk_max_depth=4,
    ) == 40


def test_target_membership_in_draft_topk_handles_short_draft_trace() -> None:
    found, ranks = target_membership_in_draft_topk(
        [11, 22, 33],
        [[10, 11], [20, 21]],
    )

    assert found == [True, False, False]
    assert ranks == [2, None, None]


def test_rope_tables_use_split_half_layout() -> None:
    cos, sin = _rope_tables(max_positions=3, rotary_dim=4, base=10_000.0)

    assert cos.shape == (3, 4)
    assert sin.shape == (3, 4)
    np.testing.assert_allclose(cos[:, :2], cos[:, 2:])
    np.testing.assert_allclose(sin[:, :2], sin[:, 2:])
    np.testing.assert_allclose(cos[0], np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(sin[0], np.zeros(4, dtype=np.float32))


def test_select_topk_tokens_has_no_input_conditioned_rerank_branches() -> None:
    """Keep the selector structurally simple: validate shape, then argmax/top-k.

    This catches future attempts to reintroduce prompt/candidate/depth-specific
    acceptance gaming with a new set of token IDs not covered by the explicit
    historical regression cases below.
    """
    source = inspect.getsource(select_topk_tokens)
    tree = ast.parse(source)
    if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert len(if_nodes) == 1
    assert "logits_row.ndim" in ast.unparse(if_nodes[0].test)
    assert "draft_depth" not in ast.unparse(if_nodes[0].test)
    assert "candidate_pool" not in ast.unparse(if_nodes[0].test)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            text = ast.unparse(node)
            assert "draft_depth" not in text
            assert "candidate_pool" not in text


def _logits_with_top3(vocab: int, top3: list[int]) -> np.ndarray:
    """Build a logits row whose strict argsort prefix is exactly ``top3``."""
    logits = np.full((vocab,), -1.0, dtype=np.float32)
    for rank, token in enumerate(top3):
        logits[token] = float(len(top3) - rank)
    return logits


@pytest.mark.parametrize(
    "top3",
    [
        [96288, 1510, 96035],  # was force-selected to 1510 at depth 4
        [220, 1510, 96035],    # was force-selected to 1510 at depth 3
        [16, 23, 1510],        # was force-selected to 1510 at depth 0
        [220, 16, 1510],       # was force-selected to 421 at depth 1
        [25, 314, 248046],     # was force-selected to 248045 at depth 2
    ],
)
def test_select_topk_tokens_is_pure_argmax_no_prompt_specific_rerank(top3: list[int]) -> None:
    """select_topk_tokens must be pure greedy top-k.

    Hardcoding token-id reranks to lift benchmark acceptance on a fixed prompt is
    an INVALID benchmark (it overfits one prompt and does not generalize). This
    guard fails if any prompt-specific override is reintroduced: the selected
    token must always be the argmax, for every draft depth.
    """
    vocab = 248_047
    logits = _logits_with_top3(vocab, top3)
    expected_argmax = top3[0]
    for draft_depth in range(0, 6):
        greedy, returned_top3 = select_topk_tokens(logits, k=3, draft_depth=draft_depth)
        assert greedy == expected_argmax, (
            f"draft_depth={draft_depth} top3={top3} returned {greedy}, expected argmax {expected_argmax}"
        )
        assert returned_top3 == top3


def test_arg_parser_allows_b3_device_kv_context_replay() -> None:
    args = build_arg_parser().parse_args(
        ["--draft-n-max", "3", "--mtp-device-kv-cache", "--mtp-context-replay"]
    )

    assert args.draft_n_max == 3
    assert args.mtp_device_kv_cache is True
    assert args.mtp_context_replay is True


def test_arg_parser_exposes_batched_target_graph_verify_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--target-graph-batched-verify"])

    assert args.target_graph_batched_verify is True


def test_arg_parser_exposes_target_block_verify_diagnostic() -> None:
    args = build_arg_parser().parse_args(
        ["--target-block-verify", "--target-block-verify-mode", "native", "--target-block-wmma-prefill"]
    )

    assert args.target_block_verify is True
    assert args.target_block_verify_mode == "native"
    assert args.target_block_wmma_prefill is True


def test_target_block_direct_commit_exactness_policy() -> None:
    assert target_block_direct_commit_is_exact("native", start_position=4096, rows=5) is True
    assert target_block_direct_commit_is_exact("serial-exact", start_position=4096, rows=5) is True
    assert target_block_direct_commit_is_exact("bulk", start_position=8, rows=4) is True
    assert target_block_direct_commit_is_exact("bulk", start_position=1020, rows=4) is False
    assert target_block_direct_commit_is_exact("unknown", start_position=8, rows=2) is False


def test_target_block_snapshot_policy_skips_exact_direct_commit() -> None:
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=True,
        direct_state_commit_exact_mode=True,
    ) is False
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=True,
        direct_state_commit_exact_mode=False,
    ) is True
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=False,
        direct_state_commit_exact_mode=True,
    ) is True


def test_arg_parser_defaults_target_block_wmma_prefill_off() -> None:
    args = build_arg_parser().parse_args(["--target-block-verify"])

    assert args.target_block_wmma_prefill is False


def test_arg_parser_exposes_draft_vocab_cap_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--mtp-draft-vocab-cap", "65536"])

    assert args.mtp_draft_vocab_cap == 65536


def test_arg_parser_exposes_adaptive_full_vocab_recovery() -> None:
    args = build_arg_parser().parse_args(
        [
            "--resident-mtp-draft",
            "--mtp-draft-vocab-cap",
            "32768",
            "--adaptive-full-vocab-after-cap-miss",
        ]
    )

    assert args.adaptive_full_vocab_after_cap_miss is True


def test_arg_parser_exposes_resident_mtp_device_seed() -> None:
    args = build_arg_parser().parse_args(["--resident-mtp-draft", "--resident-mtp-device-seed"])

    assert args.resident_mtp_draft is True
    assert args.resident_mtp_device_seed is True


def test_arg_parser_exposes_resident_mtp_device_chain() -> None:
    args = build_arg_parser().parse_args(["--resident-mtp-draft", "--resident-mtp-device-chain"])

    assert args.resident_mtp_draft is True
    assert args.resident_mtp_device_chain is True


def test_arg_parser_exposes_resident_mtp_draft_sync_stage_timings() -> None:
    args = build_arg_parser().parse_args(
        [
            "--resident-mtp-draft",
            "--record-cycle-stage-timings",
            "--resident-mtp-draft-sync-stage-timings",
        ]
    )

    assert args.resident_mtp_draft is True
    assert args.record_cycle_stage_timings is True
    assert args.resident_mtp_draft_sync_stage_timings is True


def test_arg_parser_exposes_target_block_sync_stage_timings() -> None:
    args = build_arg_parser().parse_args(
        [
            "--target-block-verify",
            "--record-cycle-stage-timings",
            "--target-block-sync-stage-timings",
        ]
    )

    assert args.target_block_verify is True
    assert args.record_cycle_stage_timings is True
    assert args.target_block_sync_stage_timings is True


def test_arg_parser_exposes_record_draft_confidence_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-draft-confidence"])

    assert args.record_draft_confidence is True


def test_arg_parser_exposes_cycle_stage_timing_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-cycle-stage-timings"])

    assert args.record_cycle_stage_timings is True


def test_arg_parser_exposes_llama_compat_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--llama-compat"])

    assert args.llama_compat is True


def test_arg_parser_exposes_verify_lm_head_q6_top1_dp4a_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--verify-lm-head-q6-top1-dp4a"])

    assert args.verify_lm_head_q6_top1_dp4a is True


def test_arg_parser_exposes_selected_down_x8_repack_mode() -> None:
    args = build_arg_parser().parse_args(["--selected-down-x8-repack", "q6"])

    assert args.selected_down_x8_repack == "q6"


def test_arg_parser_exposes_selected_gate_up_x8_mode() -> None:
    args = build_arg_parser().parse_args(["--selected-gate-up-x8"])

    assert args.selected_gate_up_x8 is True


def test_arg_parser_exposes_selected_gate_up_raw_mode() -> None:
    args = build_arg_parser().parse_args(["--selected-gate-up-raw"])

    assert args.selected_gate_up_raw is True


def test_arg_parser_exposes_q6_top1_stage1_thread_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--resident-mtp-draft-q6-top1-stage1-threads", "64"])

    assert args.resident_mtp_draft_q6_top1_stage1_threads == 64


def test_arg_parser_exposes_q6_top1_stage1_shape_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--resident-mtp-draft-q6-top1-stage1-shape", "row"])

    assert args.resident_mtp_draft_q6_top1_stage1_shape == "row"

    scalehoist_args = build_arg_parser().parse_args(
        ["--resident-mtp-draft-q6-top1-stage1-shape", "pack8_scalehoist"]
    )

    assert scalehoist_args.resident_mtp_draft_q6_top1_stage1_shape == "pack8_scalehoist"

    pack16_args = build_arg_parser().parse_args(["--resident-mtp-draft-q6-top1-stage1-shape", "pack16"])

    assert pack16_args.resident_mtp_draft_q6_top1_stage1_shape == "pack16"

    x8_args = build_arg_parser().parse_args(["--resident-mtp-draft-q6-top1-stage1-shape", "x8"])

    assert x8_args.resident_mtp_draft_q6_top1_stage1_shape == "x8"


def test_apply_llama_compat_args_forces_b2_no_probe_context_route() -> None:
    args = build_arg_parser().parse_args(
        [
            "--llama-compat",
            "--draft-n-max",
            "5",
            "--root-topk-accept",
            "40",
            "--sibling-topk-accept",
            "8",
            "--topk-branch-redraft",
            "--draft-p-min",
            "0.5",
            "--mtp-draft-vocab-cap",
            "32768",
            "--no-resident-mtp-draft",
            "--no-mtp-device-kv-cache",
            "--target-graph-batched-verify",
            "--no-target-block-verify",
            "--target-block-verify-mode",
            "native",
            "--target-block-min-rows",
            "4",
            "--no-target-block-direct-state-commit",
            "--target-b1-branch-safe-block-verify",
            "--adaptive-draft-window",
            "--adaptive-ar-fallback",
            "--adaptive-ar-fallback-cooldown",
            "4",
            "--adaptive-full-vocab-after-cap-miss",
            "--adaptive-block-after-full-accept",
            "--fused-b1-block-probe",
            "--adaptive-probe-draft-n-max",
            "3",
            "--adaptive-strict-block-probe",
        ]
    )

    apply_llama_compat_args(args)

    assert args.draft_n_max == 2
    assert args.root_topk_accept == 1
    assert args.sibling_topk_accept == 1
    assert args.topk_branch_redraft is False
    assert args.draft_p_min == 0.0
    assert args.mtp_draft_vocab_cap == 0
    assert args.resident_mtp_draft is True
    assert args.resident_mtp_device_seed is False
    assert args.resident_mtp_device_chain is False
    assert args.mtp_context_replay is True
    assert args.mtp_device_kv_cache is True
    assert args.target_graph_verify is False
    assert args.target_graph_batched_verify is False
    assert args.target_block_verify is True
    assert args.target_block_verify_mode == "bulk"
    assert args.target_block_min_rows == 2
    assert args.target_block_direct_state_commit is True
    assert args.target_b1_branch_safe_block_verify is False
    assert args.adaptive_draft_window is False
    assert args.adaptive_ar_fallback is False
    assert args.adaptive_ar_fallback_max_accepted == 0
    assert args.adaptive_ar_fallback_cooldown == 0
    assert args.adaptive_full_vocab_after_cap_miss is False
    assert args.adaptive_block_after_full_accept is False
    assert args.fused_b1_block_probe is False
    assert args.adaptive_probe_draft_n_max == 1
    assert args.adaptive_strict_block_probe is False


def test_apply_llama_compat_args_preserves_explicit_device_lifecycle_route_flags() -> None:
    args = build_arg_parser().parse_args(
        ["--llama-compat", "--resident-mtp-device-seed", "--resident-mtp-device-chain"]
    )

    apply_llama_compat_args(args)

    assert args.resident_mtp_device_seed is True
    assert args.resident_mtp_device_chain is True


def test_llama_compat_diagnostic_topk_does_not_force_host_top10() -> None:
    args = build_arg_parser().parse_args(["--llama-compat"])
    apply_llama_compat_args(args)

    assert _diagnostic_topk_candidate_count(args, 1) == 1


def test_default_diagnostic_topk_keeps_rank_histogram_width() -> None:
    args = build_arg_parser().parse_args([])

    assert _diagnostic_topk_candidate_count(args, 1) == 10


def test_main_allows_resident_device_seed_with_context_replay(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(
            [
                "--resident-mtp-draft",
                "--resident-mtp-device-seed",
                "--mtp-context-replay",
                "--mtp-device-kv-cache",
            ]
        )

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "ROCm/HIP not available" in stderr
    assert "not yet compatible" not in stderr


def test_main_rejects_resident_device_chain_without_resident_draft(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--resident-mtp-device-chain"])

    assert excinfo.value.code == 2
    assert "--resident-mtp-device-chain requires --resident-mtp-draft" in capsys.readouterr().err


def test_main_rejects_resident_draft_sync_stage_timings_without_cycle_timings(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--resident-mtp-draft", "--resident-mtp-draft-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert (
        "--resident-mtp-draft-sync-stage-timings requires --record-cycle-stage-timings"
        in capsys.readouterr().err
    )


def test_main_rejects_resident_draft_sync_stage_timings_without_resident_draft(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--record-cycle-stage-timings", "--resident-mtp-draft-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert "--resident-mtp-draft-sync-stage-timings requires --resident-mtp-draft" in capsys.readouterr().err


def test_main_rejects_target_block_sync_stage_timings_without_cycle_timings(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--target-block-verify", "--target-block-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert "--target-block-sync-stage-timings requires --record-cycle-stage-timings" in capsys.readouterr().err


def test_main_rejects_target_block_sync_stage_timings_without_block_verify(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--record-cycle-stage-timings", "--target-block-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert "--target-block-sync-stage-timings requires --target-block-verify" in capsys.readouterr().err


def test_arg_parser_exposes_adaptive_strict_block_probe() -> None:
    args = build_arg_parser().parse_args(
        [
            "--adaptive-strict-block-probe",
            "--adaptive-strict-probe-cycles",
            "2",
            "--adaptive-strict-probe-min-accepted",
            "2",
            "--adaptive-strict-fallback-draft-n-max",
            "1",
            "--adaptive-strict-fallback-root-topk",
            "40",
        ]
    )

    assert args.adaptive_strict_block_probe is True
    assert args.adaptive_strict_probe_cycles == 2
    assert args.adaptive_strict_probe_min_accepted == 2
    assert args.adaptive_strict_fallback_draft_n_max == 1
    assert args.adaptive_strict_fallback_root_topk == 40


def test_validate_draft_n_max_accepts_b1_through_b8() -> None:
    for budget in range(1, 9):
        assert validate_draft_n_max(budget) == budget
    with pytest.raises(ValueError, match="1..8"):
        validate_draft_n_max(0)
    with pytest.raises(ValueError, match="1..8"):
        validate_draft_n_max(9)


def test_root_topk_acceptance_accepts_non_argmax_root_branch() -> None:
    acceptance = root_topk_acceptance_from_target_samples(
        draft_tokens=[10, 11, 12],
        draft_topk_tokens=[[10, 20, 30], [11, 21, 31]],
        target_samples=[20, 99],
        root_topk_accept=2,
    )

    assert acceptance == {
        "accepted_draft_tokens": 1,
        "visible_output_tokens": 2,
        "output_tokens": [20, 99],
        "comparison_target_tokens": [20, 99],
        "pending_hidden_row_index": 1,
    }


def test_root_topk_acceptance_defers_to_linear_argmax_path() -> None:
    assert root_topk_acceptance_from_target_samples(
        draft_tokens=[10, 11],
        draft_topk_tokens=[[10, 20]],
        target_samples=[10, 11, 99],
        root_topk_accept=2,
    ) is None


def test_root_topk_acceptance_rejects_out_of_set_target() -> None:
    assert root_topk_acceptance_from_target_samples(
        draft_tokens=[10, 11],
        draft_topk_tokens=[[10, 20]],
        target_samples=[30, 99],
        root_topk_accept=2,
    ) is None


def test_sibling_topk_acceptance_accepts_deeper_non_argmax_sibling() -> None:
    acceptance = sibling_topk_acceptance_from_target_samples(
        draft_tokens=[10, 11, 12],
        draft_topk_tokens=[[10, 20], [11, 21], [12, 22]],
        target_samples=[10, 21, 99],
        root_topk_accept=2,
        sibling_topk_accept=2,
    )

    assert acceptance == {
        "accepted_draft_tokens": 2,
        "visible_output_tokens": 3,
        "output_tokens": [10, 21, 99],
        "comparison_target_tokens": [10, 21, 99],
        "pending_hidden_row_index": 2,
        "topk_branch_depth": 1,
    }


def test_sibling_topk_acceptance_rejects_deeper_branch_when_disabled() -> None:
    assert sibling_topk_acceptance_from_target_samples(
        draft_tokens=[10, 11, 12],
        draft_topk_tokens=[[10, 20], [11, 21], [12, 22]],
        target_samples=[10, 21, 99],
        root_topk_accept=2,
        sibling_topk_accept=1,
    ) is None


def test_llama_cpp_mtp_catchup_rows_shift_target_hidden_right() -> None:
    tokens, hidden = llama_cpp_mtp_catchup_rows(
        [101, 102, 103],
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )

    assert tokens == [101, 102, 103]
    np.testing.assert_allclose(
        hidden,
        np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_llama_cpp_mtp_catchup_rows_validates_prompt_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        llama_cpp_mtp_catchup_rows([101, 102], np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        llama_cpp_mtp_catchup_rows([], np.zeros((0, 2), dtype=np.float32))


def test_llama_cpp_acceptance_counts_corrective_target_after_reject() -> None:
    summary = llama_cpp_acceptance_from_target_samples([10, 11], [20])

    assert summary["accepted_draft_tokens"] == 0
    assert summary["visible_output_tokens"] == 1
    assert summary["output_tokens"] == [20]
    assert summary["comparison_target_tokens"] == [20]
    assert summary["pending_hidden_row_index"] == 0


def test_llama_cpp_acceptance_counts_partial_prefix_plus_corrective() -> None:
    summary = llama_cpp_acceptance_from_target_samples([10, 11], [10, 20])

    assert summary["accepted_draft_tokens"] == 1
    assert summary["visible_output_tokens"] == 2
    assert summary["output_tokens"] == [10, 20]
    assert summary["comparison_target_tokens"] == [10, 20]
    assert summary["pending_hidden_row_index"] == 1


def test_llama_cpp_acceptance_requires_corrective_after_full_accept() -> None:
    summary = llama_cpp_acceptance_from_target_samples([10, 11], [10, 11, 20])

    assert summary["accepted_draft_tokens"] == 2
    assert summary["visible_output_tokens"] == 3
    assert summary["output_tokens"] == [10, 11, 20]
    assert summary["comparison_target_tokens"] == [10, 11]
    assert summary["pending_hidden_row_index"] == 2

    with pytest.raises(ValueError, match="corrective target"):
        llama_cpp_acceptance_from_target_samples([10], [10])
