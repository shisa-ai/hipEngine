from __future__ import annotations

import ast
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import gguf_mtp_bench as bench
from scripts.gguf_mtp_bench import (
    draft_topk_margins_from_scores,
    draft_topk_scores_from_logits,
    hidden_state_summary,
    _draft_top1_prob,
    _diagnostic_topk_candidate_count,
    _rope_tables,
    _target_hidden_seed_summary_rows,
    _target_lm_head_score_rows,
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
    target_block_direct_partial_commit_is_exact,
    target_block_direct_partial_commit_is_llama_style,
    target_block_needs_linear_state_snapshot,
    target_block_state_replay_uses_serial_exact,
    target_membership_in_draft_topk,
    validate_draft_n_max,
)


def test_default_mtp_policy_is_b1_root_top40_speed_first() -> None:
    args = build_arg_parser().parse_args([])

    assert args.draft_n_max == 1
    assert args.root_topk_accept == 40
    assert args.sibling_topk_accept == 1
    assert args.topk_branch_redraft is False
    assert not hasattr(args, "mtp_draft_warmup")
    assert not hasattr(args, "target_graph_verify")
    assert args.adaptive_block_after_full_accept is False
    assert args.adaptive_probe_draft_n_max == 3


@pytest.mark.parametrize(
    "option",
    [
        "--mtp-draft-warmup",
        "--no-mtp-draft-warmup",
        "--target-graph-verify",
        "--no-target-graph-verify",
        "--target-graph-batched-verify",
        "--no-target-graph-batched-verify",
        "--target-block-replay-state-commit",
        "--no-target-block-replay-state-commit",
    ],
)
def test_retired_warmup_and_target_graph_selectors_are_rejected(option, capsys, monkeypatch) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: pytest.fail("retired CLI reached HIP"))
    with pytest.raises(SystemExit) as exc:
        bench.main([option])

    assert exc.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err
    assert option not in build_arg_parser().format_help()


@pytest.mark.parametrize("shape", ["row", "pack8_scalehoist", "pack8_llama", "pack16", "x8_dscale"])
@pytest.mark.parametrize("sync", [False, True])
def test_explicit_head_cli_survives_suite_preset_retirement(monkeypatch, shape, sync):
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE", "x8")
    options = ["--llama-compat", "--resident-mtp-draft-q6-top1-dp4a",
               "--resident-mtp-draft-q6-top1-stage1-shape", shape]
    if sync:
        options.append("--resident-mtp-draft-sync-stage-timings")
    args = build_arg_parser().parse_args(options)
    apply_llama_compat_args(args)
    assert args.resident_mtp_draft_q6_top1_stage1_shape == shape
    assert args.resident_mtp_draft_q6_top1_dp4a is True
    assert args.resident_mtp_draft_sync_stage_timings is sync
    assert args.draft_n_max == 2


def test_prompt_catchup_inputs_are_not_retained_in_cycle_history() -> None:
    tree = ast.parse(inspect.getsource(bench.main))
    cycles = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "cycle"
    ]
    assert len(cycles) == 1
    prompt_inputs = {"mtp_context_tokens", "mtp_context_positions", "mtp_context_hidden_rows"}
    assert not [
        node.id for node in ast.walk(cycles[0]) if isinstance(node, ast.Name) and node.id in prompt_inputs
    ], "prompt priming rows must not grow as an unused host cycle history"


def test_context_replay_help_describes_all_row_prompt_capture() -> None:
    help_text = " ".join(build_arg_parser().format_help().split())
    assert "bulk target hidden rows" in help_text
    assert "does not expose all hidden rows" not in help_text


def test_device_kv_help_is_not_limited_to_b1() -> None:
    help_text = " ".join(build_arg_parser().format_help().split())
    assert "device-resident MTP dense KV cache for drafting" in help_text
    assert "dense KV cache for B1 drafting" not in help_text


def test_target_topk_score_flags_parse_default_off() -> None:
    args = build_arg_parser().parse_args([])

    assert args.record_target_topk_scores is False
    assert args.target_topk_score_count == 10
    assert args.target_score_candidate_tokens == ()

    args = build_arg_parser().parse_args(
        [
            "--record-target-topk-scores",
            "--target-topk-score-count",
            "3",
            "--target-score-candidate-tokens",
            "7,5,7",
        ]
    )

    assert args.record_target_topk_scores is True
    assert args.target_topk_score_count == 3
    assert args.target_score_candidate_tokens == (7, 5)


def test_target_lm_head_score_rows_are_compact_and_candidate_driven() -> None:
    logits = np.asarray(
        [
            [0.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 1.0, 2.0, 4.0, 3.0],
        ],
        dtype=np.float32,
    )

    rows = _target_lm_head_score_rows(
        logits,
        input_tokens=[100, 101],
        target_tokens=[1, 0],
        draft_tokens=[2, 3],
        extra_candidate_tokens=(4, 2),
        top_k=2,
    )

    assert [row["input_token"] for row in rows] == [100, 101]
    assert [row["target_token"] for row in rows] == [1, 0]
    assert rows[0]["top_k"] == [
        {"rank": 1, "token": 1, "score": 4.0, "delta_vs_top1": 0.0},
        {"rank": 2, "token": 2, "score": 3.0, "delta_vs_top1": -1.0},
    ]
    assert rows[0]["candidate_scores"] == [
        {"token": 2, "score": 3.0, "delta_vs_top1": -1.0},
        {"token": 3, "score": 2.0, "delta_vs_top1": -2.0},
        {"token": 1, "score": 4.0, "delta_vs_top1": 0.0},
        {"token": 0, "score": 0.0, "delta_vs_top1": -4.0},
        {"token": 4, "score": 1.0, "delta_vs_top1": -3.0},
    ]


@pytest.mark.parametrize("logits", [None, np.empty((0, 4), dtype=np.float32)])
def test_target_score_rows_without_available_logits_are_empty(logits) -> None:
    assert _target_lm_head_score_rows(
        logits, input_tokens=[3], target_tokens=[1], draft_tokens=[2],
    ) == []


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3)])
def test_target_score_rows_require_rank_two_logits(shape) -> None:
    with pytest.raises(ValueError, match="rank-2"):
        _target_lm_head_score_rows(
            np.zeros(shape, dtype=np.float32), input_tokens=[3], target_tokens=[1], draft_tokens=[2],
        )


@pytest.mark.parametrize("input_tokens,target_tokens", [([8], [1, 2]), ([8, 9], [1])])
def test_target_score_rows_bound_metadata_and_preserve_near_tie_candidates(input_tokens, target_tokens) -> None:
    second = np.nextafter(np.float32(1), np.float32(0))
    logits = np.asarray([[0, 1, second], [2, 0, 1]], dtype=np.float32)
    rows = _target_lm_head_score_rows(
        logits, input_tokens=input_tokens, target_tokens=target_tokens, draft_tokens=[2, 2],
        extra_candidate_tokens=(-1, 99, 2), top_k=99,
    )
    assert len(rows) == 1
    assert rows[0]["row"] == rows[0]["position_offset"] == 0
    assert rows[0]["input_token"] == 8 and rows[0]["target_token"] == 1
    assert [item["token"] for item in rows[0]["top_k"]] == [1, 2, 0]
    candidates = rows[0]["candidate_scores"]
    assert [item["token"] for item in candidates] == list(dict.fromkeys([2, *target_tokens, -1, 99]))
    assert candidates[0] == {"token": 2, "score": float(second), "delta_vs_top1": float(second) - 1.0}
    assert candidates[-2:] == [
        {"token": -1, "score": None, "delta_vs_top1": None},
        {"token": 99, "score": None, "delta_vs_top1": None},
    ]


@pytest.mark.parametrize("argv,error", [
    (["--record-target-topk-scores"], "--record-target-topk-scores requires --target-block-verify"),
    (["--target-block-verify", "--native-spec-target-cycle", "--record-target-topk-scores"],
     "--native-spec-target-cycle does not support target logits readback"),
    (["--target-topk-score-count", "0"], "--target-topk-score-count must be positive"),
])
def test_target_score_cli_dependencies_fail_before_loading(monkeypatch, capsys, argv, error) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    with pytest.raises(SystemExit) as excinfo:
        bench.main(argv)
    assert excinfo.value.code == 2
    assert error in capsys.readouterr().err


def test_target_hidden_seed_summary_rows_align_to_score_rows() -> None:
    hidden_rows = [
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
    ]
    score_rows = [
        {"row": 0, "input_token": 100, "target_token": 200},
        {"row": 2, "input_token": 102, "target_token": 202},
    ]

    rows = _target_hidden_seed_summary_rows(
        hidden_rows,
        score_rows,
        cycle_start_seq_position=64,
    )

    assert rows[0]["row"] == 0
    assert rows[0]["input_token"] == 100
    assert rows[0]["target_token"] == 200
    assert rows[0]["hidden_available"] is True
    assert rows[0]["summary"]["label"] == "target_hidden_seed"
    assert rows[0]["summary"]["position"] == 64
    assert rows[0]["summary"]["first8"] == [1.0, 2.0, 3.0, 4.0]
    assert rows[1] == {
        "row": 2,
        "input_token": 102,
        "target_token": 202,
        "hidden_available": False,
    }


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


@pytest.mark.parametrize("option", ["--target-b1-branch-safe-block-verify", "--no-target-b1-branch-safe-block-verify"])
def test_retired_b1_branch_safe_option_is_rejected(option, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_arg_parser().parse_args([option])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_b1_branch_safe_retirement_keeps_false_artifacts_without_runtime_selector() -> None:
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(bench))
    assert not hasattr(build_arg_parser().parse_args([]), "target_b1_branch_safe_block_verify")
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "target_b1_branch_safe_block_verify"
        or isinstance(node, ast.Name) and node.id in {"can_b1_branch_safe_block_verify", "b1_branch_safe_block_verify_used"}
        for node in ast.walk(tree)
    )
    artifacts = [
        value for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "target_b1_branch_safe_block_verify"
    ]
    assert len(artifacts) == 2
    assert all(isinstance(value, ast.Constant) and value.value is False for value in artifacts)


def test_native_complete_cycle_flag_parses_default_off() -> None:
    assert build_arg_parser().parse_args([]).native_spec_complete_cycle is False
    assert build_arg_parser().parse_args(
        ["--native-spec-complete-cycle"]
    ).native_spec_complete_cycle is True
    assert build_arg_parser().parse_args([]).native_spec_proposal_graph is False
    assert build_arg_parser().parse_args(
        ["--native-spec-proposal-graph"]
    ).native_spec_proposal_graph is True


@pytest.mark.parametrize("option,field", [
    ("native-spec-target-cycle", "native_spec_target_cycle"),
    ("native-spec-device-accept-commit", "native_spec_device_accept_commit"),
    ("native-spec-complete-cycle", "native_spec_complete_cycle"),
    ("native-spec-proposal-graph", "native_spec_proposal_graph"),
])
def test_native_cycle_diagnostic_flags_preserve_default_and_explicit_off(option, field) -> None:
    parser = build_arg_parser()
    assert getattr(parser.parse_args([]), field) is False
    assert getattr(parser.parse_args([f"--{option}"]), field) is True
    assert getattr(parser.parse_args([f"--{option}", f"--no-{option}"]), field) is False


@pytest.mark.parametrize("argv,expected_error", [
    (["--native-spec-target-cycle"], "--native-spec-target-cycle requires --target-block-verify"),
    (["--native-spec-device-accept-commit"], "--native-spec-device-accept-commit requires --native-spec-target-cycle"),
    (["--native-spec-complete-cycle"], "--native-spec-complete-cycle requires --native-spec-device-accept-commit"),
    (["--native-spec-proposal-graph"], "--native-spec-proposal-graph requires --native-spec-complete-cycle"),
    (["--target-block-verify", "--native-spec-target-cycle", "--target-block-wmma-prefill"],
     "--native-spec-target-cycle requires --no-target-block-wmma-prefill"),
    (["--target-block-verify", "--native-spec-target-cycle", "--target-block-verify-mode", "serial-exact"],
     "--native-spec-target-cycle requires --target-block-verify-mode bulk"),
])
def test_native_cycle_diagnostic_stage_dependencies_fail_before_loading(
    monkeypatch, capsys, argv, expected_error,
) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    # main writes diagnostic environment before validating stage dependencies.
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--root-topk-accept", "1", *argv])
    assert excinfo.value.code == 2
    assert expected_error in capsys.readouterr().err


def test_fused_b1_block_probe_flag_parses() -> None:
    args = build_arg_parser().parse_args(["--fused-b1-block-probe"])

    assert args.fused_b1_block_probe is True


@pytest.mark.parametrize("option", ["--verify-dense-q8-dp4a-shared", "--no-verify-dense-q8-dp4a-shared"])
def test_retired_shared_q8_option_is_rejected(option: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_arg_parser().parse_args([option])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("option", [None, "--no-verify-dp4a", "--verify-dp4a"])
@pytest.mark.parametrize("inherited", [None, "0", "1"])
def test_verify_dp4a_only_enables_selected_adapters_and_preserves_ambient_values(
    monkeypatch, option, inherited,
) -> None:
    names = (
        "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A",
        "HIPENGINE_GGUF_T16_SELECTED_DP4A",
        "HIPENGINE_GGUF_RAW_SELECTED_DP4A",
    )
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    for name in names:
        if inherited is None:
            bench.os.environ.pop(name, None)
        else:
            bench.os.environ[name] = inherited

    class ReachedHIPBoundary(Exception):
        pass

    def stop_before_runtime():
        raise ReachedHIPBoundary

    monkeypatch.setattr(bench, "_hip_available", stop_before_runtime)
    argv = [] if option is None else [option]
    assert build_arg_parser().parse_args(argv).verify_dp4a is (option == "--verify-dp4a")
    with pytest.raises(ReachedHIPBoundary):
        bench.main(argv)
    expected = "1" if option == "--verify-dp4a" else inherited
    assert [bench.os.environ.get(name) for name in names] == [expected] * len(names)


def test_verify_dp4a_help_names_the_arithmetic_change_without_historical_rates() -> None:
    parser = build_arg_parser()
    help_text = next(action.help for action in parser._actions if action.dest == "verify_dp4a")
    assert "q8_1" in help_text
    assert "accuracy-traded" in help_text
    assert "61.6" not in help_text
    assert "67.3" not in help_text


def test_dense_q8_f32_dp4a_flag_parses_default_off() -> None:
    args = build_arg_parser().parse_args([])
    assert args.verify_dense_q8_dp4a_f32 is False

    args = build_arg_parser().parse_args(["--verify-dense-q8-dp4a-f32"])
    assert args.verify_dense_q8_dp4a_f32 is True


def test_resident_mtp_draft_dense_q8_dp4a_flag_parses_default_off() -> None:
    args = build_arg_parser().parse_args([])
    assert args.resident_mtp_draft_dense_q8_dp4a is False
    assert args.resident_mtp_draft_dense_q8_dp4a_stages == ""

    args = build_arg_parser().parse_args(
        ["--resident-mtp-draft-dense-q8-dp4a", "--resident-mtp-draft-dense-q8-dp4a-stages", "draft"]
    )
    assert args.resident_mtp_draft_dense_q8_dp4a is True
    assert args.resident_mtp_draft_dense_q8_dp4a_stages == "draft"


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


def test_fused_b1_block_probe_validation_rejects_inert_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--fused-b1-block-probe"])

    assert excinfo.value.code == 2
    assert "--fused-b1-block-probe requires --target-block-verify" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("fused_b1_block_probe", False), ("adaptive_block_after_full_accept", False),
    ("adaptive_block_verify_active", True), ("cycle_ar_fallback", True),
    ("topk_branch_redraft", True),
    ("cycle_draft_window", 0), ("cycle_draft_window", 2), ("cycle_draft_window", 5),
    ("adaptive_probe_draft_n_max", 0), ("adaptive_probe_draft_n_max", 2),
    ("cycle_root_topk_accept", 2), ("cycle_root_topk_accept", 40),
    ("cycle_sibling_topk_accept", 2), ("cycle_sibling_topk_accept", 40),
])
def test_fused_b1_probe_rejects_each_non_probe_dimension(field, value):
    options = dict(fused_b1_block_probe=True, adaptive_block_after_full_accept=True,
                   adaptive_block_verify_active=False, cycle_ar_fallback=False,
                   cycle_draft_window=1, adaptive_probe_draft_n_max=1,
                   cycle_root_topk_accept=1, cycle_sibling_topk_accept=1,
                   topk_branch_redraft=False)
    assert should_use_fused_b1_block_probe(**options) is True
    assert should_use_fused_b1_block_probe(**{**options, field: value}) is False
    assert should_use_fused_b1_block_probe(**options) is True


@pytest.mark.parametrize("override,message", [
    (["--no-target-block-verify"], "requires --target-block-verify"),
    (["--no-adaptive-block-after-full-accept"], "requires --adaptive-block-after-full-accept"),
    (["--adaptive-probe-draft-n-max", "2"], "requires --adaptive-probe-draft-n-max 1"),
    (["--target-block-min-rows", "1"], "requires --target-block-min-rows 2"),
    (["--target-block-min-rows", "3"], "requires --target-block-min-rows 2"),
    (["--root-topk-accept", "2"], "requires strict top-1 linear drafting"),
    (["--sibling-topk-accept", "2"], "requires strict top-1 linear drafting"),
    (["--topk-branch-redraft"], "requires strict top-1 linear drafting"),
])
def test_fused_b1_probe_cli_preconditions_fail_before_model_load(monkeypatch, tmp_path, capsys, override, message):
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE", "pack8")
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS", "128")
    options = ["--model", str(tmp_path / "absent.gguf"), "--draft-n-max", "5",
               "--fused-b1-block-probe", "--target-block-verify",
               "--adaptive-block-after-full-accept", "--adaptive-probe-draft-n-max", "1",
               "--target-block-min-rows", "2", "--root-topk-accept", "1",
               "--sibling-topk-accept", "1"]
    with pytest.raises(SystemExit) as excinfo:
        bench.main([*options, *override])
    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    assert f"--fused-b1-block-probe {message}" in error
    assert "Model file not found" not in error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("budget", [1, 2, 5, 7])
def test_fused_b1_probe_is_valid_under_larger_configured_budgets(monkeypatch, tmp_path, capsys, budget):
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    for name, value in [("HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE", "pack8"),
                        ("HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS", "128"),
                        ("HIPENGINE_GGUF_DECODE_REPACK", "1")]:
        monkeypatch.setenv(name, value)
    options = ["--model", str(tmp_path / "absent.gguf"), "--draft-n-max", str(budget),
               "--fused-b1-block-probe", "--target-block-verify",
               "--adaptive-block-after-full-accept", "--adaptive-probe-draft-n-max", "1",
               "--target-block-min-rows", "2", "--root-topk-accept", "1",
               "--sibling-topk-accept", "1"]
    with pytest.raises(SystemExit) as excinfo:
        bench.main(options)
    assert excinfo.value.code == 1
    assert "Model file not found" in capsys.readouterr().err


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


def test_draft_topk_scores_and_margins_follow_selected_token_order() -> None:
    logits = np.array([0.1, 4.0, -1.0, 2.5, 3.0], dtype=np.float32)

    scores = draft_topk_scores_from_logits(logits, [1, 4, 3])

    assert scores == pytest.approx([4.0, 3.0, 2.5])
    assert draft_topk_margins_from_scores(scores) == pytest.approx([0.0, 1.0, 1.5])
    with pytest.raises(ValueError, match="outside logits row"):
        draft_topk_scores_from_logits(logits, [99])


def test_hidden_state_summary_is_stable_and_compact() -> None:
    row = np.asarray([[1.0, -2.0, 3.5, 4.0]], dtype=np.float32)

    summary = hidden_state_summary(
        row,
        label="seed",
        depth=2,
        token_id=42,
        position=9,
    )

    assert summary["label"] == "seed"
    assert summary["depth"] == 2
    assert summary["token_id"] == 42
    assert summary["position"] == 9
    assert summary["size"] == 4
    assert summary["mean"] == pytest.approx(1.625)
    assert summary["rms"] == pytest.approx(float(np.sqrt((1.0 + 4.0 + 12.25 + 16.0) / 4.0)))
    assert summary["first8"] == pytest.approx([1.0, -2.0, 3.5, 4.0])
    assert summary["last8"] == pytest.approx([1.0, -2.0, 3.5, 4.0])
    assert isinstance(summary["sha256_16"], str)
    assert len(summary["sha256_16"]) == 16


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


def test_benchmark_uses_one_step_graph_without_generated_history() -> None:
    tree = ast.parse(inspect.getsource(bench.main))
    captures = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "capture_decode_graph"
    ]
    assert len(captures) == 1
    keywords = {arg.arg: arg.value for arg in captures[0].keywords}
    assert isinstance(keywords["steps_per_replay"], ast.Constant)
    assert keywords["steps_per_replay"].value == 1
    assert isinstance(keywords["record_steps"], ast.Constant)
    assert keywords["record_steps"].value == 0
    assert isinstance(keywords["record_hidden_seeds"], ast.Constant)
    assert keywords["record_hidden_seeds"].value is False
    assert not [
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        and node.attr in {"read_generated_token_ids", "read_generated_hidden_seeds"}
    ]


def test_target_block_commit_invalidates_preexisting_target_graph() -> None:
    class FakeGraph:
        closed = False

        def close(self) -> None:
            self.closed = True

    graph = FakeGraph()
    target_graph, enabled, reason = bench.invalidate_target_graph_after_block_verify(graph)

    assert graph.closed is True
    assert target_graph is None
    assert enabled is False
    assert reason == "target block verifier advanced resident state; fell back to eager target verification"


def test_arg_parser_exposes_target_block_verify_diagnostic() -> None:
    args = build_arg_parser().parse_args(
        ["--target-block-verify", "--target-block-verify-mode", "native", "--target-block-wmma-prefill"]
    )

    assert args.target_block_verify is True
    assert args.target_block_verify_mode == "native"
    assert args.target_block_wmma_prefill is True


def test_benchmark_has_no_replay_state_selector_or_branch() -> None:
    args = build_arg_parser().parse_args(["--target-block-verify"])
    assert not hasattr(args, "target_block_replay_state_commit")
    tree = ast.parse(inspect.getsource(bench.main))
    assert not [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "replay_state_commit"
        or isinstance(node, ast.Attribute) and node.attr == "target_block_replay_state_commit"
    ]


def test_arg_parser_exposes_target_block_direct_partial_replay_mode() -> None:
    args = build_arg_parser().parse_args(
        [
            "--target-block-verify",
            "--target-block-direct-state-commit",
            "--target-block-direct-partial-replay-mode",
            "serial-state-only",
        ]
    )

    assert args.target_block_verify is True
    assert args.target_block_direct_state_commit is True
    assert args.target_block_direct_partial_replay_mode == "serial-state-only"


def test_target_block_direct_commit_exactness_policy() -> None:
    assert target_block_direct_commit_is_exact("native", start_position=4096, rows=5) is True
    assert target_block_direct_commit_is_exact("serial-exact", start_position=4096, rows=5) is True
    assert target_block_direct_commit_is_exact("bulk", start_position=8, rows=4) is True
    assert target_block_direct_commit_is_exact("bulk", start_position=1020, rows=4) is False
    assert target_block_direct_commit_is_exact("unknown", start_position=8, rows=2) is False
    assert target_block_direct_partial_commit_is_exact("native") is True
    assert target_block_direct_partial_commit_is_exact("serial-exact") is True
    assert target_block_direct_partial_commit_is_exact("bulk") is False
    assert target_block_direct_partial_commit_is_llama_style("bulk", "direct-commit") is True
    assert target_block_direct_partial_commit_is_llama_style("native", "direct-commit") is False
    assert target_block_direct_partial_commit_is_llama_style("bulk", "serial-state-only") is False


def test_target_block_snapshot_policy_skips_exact_direct_commit() -> None:
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=True,
        direct_state_commit_exact_mode=True,
        direct_partial_commit_exact_mode=True,
    ) is False
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=True,
        direct_state_commit_exact_mode=False,
        direct_partial_commit_exact_mode=True,
    ) is True
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=True,
        direct_state_commit_exact_mode=True,
        direct_partial_commit_exact_mode=False,
    ) is True
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=True,
        direct_state_commit_exact_mode=True,
        direct_partial_commit_exact_mode=False,
        direct_partial_commit_llama_style=True,
    ) is False
    assert target_block_needs_linear_state_snapshot(
        direct_state_commit=False,
        direct_state_commit_exact_mode=True,
        direct_partial_commit_exact_mode=True,
    ) is True


@pytest.mark.parametrize(
    "direct,mode,partial_mode,expected",
    [
        (True, "bulk", "serial-exact", True),
        (True, "bulk", "serial-state-only", True),
        (True, "bulk", "bulk-state-only", False),
        (True, "bulk", "native-state-only", False),
        (True, "bulk", "direct-commit", False),
        (True, "serial-exact", "bulk-state-only", True),
        (False, "serial-exact", "serial-exact", True),
        (False, "bulk", "serial-exact", False),
    ],
)
def test_target_block_partial_replay_policy(direct, mode, partial_mode, expected) -> None:
    assert target_block_state_replay_uses_serial_exact(
        direct_state_commit=direct,
        verify_mode=mode,
        direct_partial_replay_mode=partial_mode,
    ) is expected


def test_arg_parser_defaults_target_block_wmma_prefill_off() -> None:
    args = build_arg_parser().parse_args(["--target-block-verify"])

    assert args.target_block_wmma_prefill is False


@pytest.mark.parametrize("cap", [0, 17, 32767, 32768, 32769, 65536, 999999])
def test_arg_parser_exposes_explicit_draft_vocab_cap(cap) -> None:
    args = build_arg_parser().parse_args(["--mtp-draft-vocab-cap", str(cap)])
    assert args.mtp_draft_vocab_cap == cap


def test_arg_parser_defaults_to_full_draft_vocabulary() -> None:
    assert build_arg_parser().parse_args([]).mtp_draft_vocab_cap == 0


def test_draft_vocab_cap_help_states_scope_without_qualification_gate() -> None:
    action = next(
        action for action in build_arg_parser()._actions
        if action.dest == "mtp_draft_vocab_cap"
    )
    assert "0 = full vocabulary" in action.help
    assert "target verification uses the full vocabulary" in action.help
    assert "validated" not in action.help
    assert "retained" not in action.help


def test_negative_draft_vocab_cap_fails_before_hip(monkeypatch, capsys) -> None:
    for name in ("HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS", "HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE"):
        monkeypatch.setenv(name, "test-original")
    monkeypatch.setattr(bench, "_hip_available", lambda: pytest.fail("negative cap reached HIP"))
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--mtp-draft-vocab-cap", "-1"])
    assert excinfo.value.code == 2
    assert "--mtp-draft-vocab-cap must be non-negative" in capsys.readouterr().err


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


@pytest.mark.parametrize(
    "enabled,drafts,accepted,maximum,cap,has_recovery,recovering,ar_enabled,expected",
    [
        (False, [7], 0, 0, 32, True, False, True, (False, True, False)),
        (True, [], 0, 0, 32, True, False, True, (False, False, False)),
        (True, [7], 1, 0, 32, True, False, True, (False, False, False)),
        (True, [7], 0, 0, 32, True, False, True, (True, False, True)),
        (True, [7, 8], 1, 0, 32, True, False, True, (False, False, False)),
        (True, [7, 8], 1, 1, 32, True, False, True, (True, False, True)),
        (True, [7], 0, 0, 100, True, False, True, (False, True, False)),
        (True, [7], 0, 0, 32, False, False, True, (False, True, False)),
        (True, [7, 8], 1, 1, 100, True, True, True, (True, True, False)),
        (True, [7, 8], 2, 0, 100, True, True, True, (False, False, False)),
        (True, [7], 0, 0, 32, True, False, False, (True, False, True)),
    ],
)
def test_full_vocab_recovery_and_ar_fallback_transitions(
    enabled, drafts, accepted, maximum, cap, has_recovery, recovering, ar_enabled, expected,
) -> None:
    from types import SimpleNamespace

    tree = ast.parse(inspect.getsource(bench.main))
    cycle = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "cycle"
    )
    start = next(
        index for index, node in enumerate(cycle.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "low_accept_miss" for target in node.targets)
    )
    # Execute the actual adjacent recovery/fallback statements, not a second
    # implementation of the policy in the test.
    policy = ast.Module(body=cycle.body[start:start + 4], type_ignores=[])
    namespace = {
        "args": SimpleNamespace(
            adaptive_full_vocab_after_cap_miss=enabled,
            adaptive_ar_fallback_max_accepted=maximum,
            adaptive_ar_fallback=ar_enabled,
            adaptive_ar_fallback_cooldown=3,
        ),
        "draft_tokens": drafts,
        "accepted_draft_tokens": accepted,
        "cycle_full_vocab_recovery": recovering,
        "resident_draft_full_vocab": object() if has_recovery else None,
        "cycle_draft_vocab_cap": cap,
        "sh_raw": SimpleNamespace(shape=(100,)),
        "adaptive_full_vocab_recovery_active": recovering,
        "adaptive_ar_fallback_active": False,
        "adaptive_ar_fallback_cooldown_left": 0,
    }
    exec(compile(policy, "<bench-recovery-policy>", "exec"), namespace)
    assert (
        namespace["adaptive_full_vocab_recovery_active"],
        namespace["adaptive_ar_fallback_active"],
        namespace["suppress_ar_fallback_for_full_vocab_recovery"],
    ) == expected
    assert namespace["adaptive_ar_fallback_cooldown_left"] == (3 if expected[1] else 0)


def test_arg_parser_exposes_resident_mtp_device_seed() -> None:
    args = build_arg_parser().parse_args(["--resident-mtp-draft", "--resident-mtp-device-seed"])

    assert args.resident_mtp_draft is True
    assert args.resident_mtp_device_seed is True


def test_device_only_draft_cache_remains_selectable_without_prompt_replay() -> None:
    args = build_arg_parser().parse_args([
        "--resident-mtp-draft", "--resident-mtp-device-seed", "--mtp-device-kv-cache",
        "--mtp-draft-vocab-cap", "32768",
    ])
    assert args.resident_mtp_draft is True
    assert args.resident_mtp_device_seed is True
    assert args.mtp_device_kv_cache is True
    assert args.mtp_context_replay is False


@pytest.mark.parametrize("block_verify", [False, True])
def test_context_device_seed_flags_remain_selectable_with_either_verifier(block_verify) -> None:
    args = build_arg_parser().parse_args([
        "--resident-mtp-draft", "--resident-mtp-device-seed", "--mtp-context-replay",
        "--mtp-device-kv-cache", "--mtp-draft-vocab-cap", "32768",
        "--adaptive-full-vocab-after-cap-miss",
        "--target-block-verify" if block_verify else "--no-target-block-verify",
    ])
    assert args.resident_mtp_device_seed is True
    assert args.mtp_context_replay is True
    assert args.mtp_device_kv_cache is True
    assert args.adaptive_full_vocab_after_cap_miss is True
    assert args.target_block_verify is block_verify


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


def test_arg_parser_exposes_native_spec_target_cycle_diagnostic() -> None:
    args = build_arg_parser().parse_args(
        ["--target-block-verify", "--native-spec-target-cycle"]
    )

    assert args.target_block_verify is True
    assert args.native_spec_target_cycle is True


def test_arg_parser_exposes_native_spec_n2_device_accept_commit() -> None:
    args = build_arg_parser().parse_args(
        [
            "--target-block-verify",
            "--native-spec-target-cycle",
            "--native-spec-device-accept-commit",
            "--resident-mtp-draft",
            "--resident-mtp-device-seed",
        ]
    )

    assert args.native_spec_target_cycle is True
    assert args.native_spec_device_accept_commit is True
    assert args.resident_mtp_device_seed is True


def test_retired_record_draft_confidence_is_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_arg_parser().parse_args(["--record-draft-confidence"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_confidence_retirement_preserves_empty_artifacts_without_selector() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(bench))
    assert not hasattr(build_arg_parser().parse_args([]), "record_draft_confidence")
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "record_draft_confidence"
        or isinstance(node, ast.Name) and node.id == "cycle_draft_top1_probs"
        for node in ast.walk(tree)
    )
    artifacts = {
        key.value: value
        for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value in {"record_draft_confidence", "draft_top1_probs"}
    }
    assert isinstance(artifacts["record_draft_confidence"], ast.Constant)
    assert artifacts["record_draft_confidence"].value is False
    assert isinstance(artifacts["draft_top1_probs"], ast.List)
    assert artifacts["draft_top1_probs"].elts == []


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.5, 0.8, 0.99])
def test_draft_probability_threshold_remains_explicitly_selectable(threshold) -> None:
    args = build_arg_parser().parse_args(["--draft-p-min", str(threshold)])
    assert args.draft_p_min == threshold


def test_arg_parser_exposes_record_draft_topk_scores_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-draft-topk-scores"])

    assert args.record_draft_topk_scores is True


def test_arg_parser_exposes_record_draft_hidden_stats_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-draft-hidden-stats"])

    assert args.record_draft_hidden_stats is True


def test_arg_parser_exposes_record_draft_cache_rows_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-draft-cache-rows", "0,1,2,48,49,48"])

    assert args.record_draft_cache_rows == (0, 1, 2, 48, 49)


def test_arg_parser_exposes_record_draft_attention_debug_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-draft-attention-debug"])

    assert args.record_draft_attention_debug is True


def test_arg_parser_exposes_cycle_stage_timing_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--record-cycle-stage-timings"])

    assert args.record_cycle_stage_timings is True


def test_arg_parser_exposes_llama_compat_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--llama-compat"])

    assert args.llama_compat is True


def test_arg_parser_exposes_verify_lm_head_q6_top1_dp4a_diagnostic() -> None:
    args = build_arg_parser().parse_args(["--verify-lm-head-q6-top1-dp4a"])

    assert args.verify_lm_head_q6_top1_dp4a is True


@pytest.mark.parametrize("option", [None, "--verify-lm-head-q6-top1-dp4a", "--no-verify-lm-head-q6-top1-dp4a"])
@pytest.mark.parametrize("ambient", [None, "0", "1"])
def test_verifier_lm_head_cli_publishes_load_and_runtime_controls(monkeypatch, option, ambient) -> None:
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    names = ("HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR", "HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A")
    for name in names:
        if ambient is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, ambient)
    # Argument validation runs after publishing controls, before materialization/HIP.
    with pytest.raises(SystemExit) as excinfo:
        bench.main(([option] if option else []) + ["--root-topk-accept", "0"])
    assert excinfo.value.code == 2
    expected = "1" if option == "--verify-lm-head-q6-top1-dp4a" else ambient
    assert [bench.os.environ.get(name) for name in names] == [expected, expected]


@pytest.mark.parametrize("option", [None, "--resident-mtp-draft-router-row-parallel", "--no-resident-mtp-draft-router-row-parallel"])
@pytest.mark.parametrize("ambient", [None, "0", "1"])
def test_draft_router_cli_preserves_inherited_construction_policy(monkeypatch, option, ambient) -> None:
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    name = "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"
    if ambient is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, ambient)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(([option] if option else []) + ["--root-topk-accept", "0"])
    assert excinfo.value.code == 2
    assert bench.os.environ.get(name) == ("1" if option == "--resident-mtp-draft-router-row-parallel" else ambient)


@pytest.mark.parametrize("option", [None, "--resident-mtp-draft-selected-silu-down-fused", "--no-resident-mtp-draft-selected-silu-down-fused"])
@pytest.mark.parametrize("ambient", [None, "0", "1"])
def test_benchmark_preserves_explicit_silu_down_comparison(monkeypatch, option, ambient):
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    name = "HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED"
    if ambient is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, ambient)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(([option] if option else []) + ["--root-topk-accept", "0"])
    assert excinfo.value.code == 2
    assert bench.os.environ.get(name) == ("1" if option == "--resident-mtp-draft-selected-silu-down-fused" else ambient)


@pytest.mark.parametrize(
    "option,control_env",
    [
        ("--resident-mtp-draft-selected-silu-down-fused", "HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED"),
        ("--resident-mtp-draft-router-row-parallel", "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"),
    ],
)
def test_benchmark_draft_runner_cache_key_separates_comparisons(monkeypatch, tmp_path, option, control_env):
    from hipengine.loading import gguf as gguf_module
    from hipengine.quant import gguf as quant_module
    from hipengine.runtime import qwen35_gguf_runner as runner_module
    from hipengine.speculative import mtp_resident_draft as draft_module
    from hipengine.tokenization import gguf as tokenizer_module

    name = control_env
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HIPENGINE_MTP_BENCH_CACHE_SESSION", "1")
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    monkeypatch.setattr(
        bench,
        "_rope_tables",
        lambda **kwargs: (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(bench, "select_prompt_tokens", lambda *args, **kwargs: [1])
    monkeypatch.setattr(bench, "_DRAFT_RUNNER_CACHE", {})
    monkeypatch.setattr(bench, "_SESSION_CACHE", {})

    model = tmp_path / "model.gguf"
    model.write_bytes(b"benchmark-cache-key-model")
    tensor_names = (
        "blk.40.nextn.eh_proj.weight", "blk.40.nextn.hnorm.weight",
        "blk.40.nextn.enorm.weight", "blk.40.attn_norm.weight",
        "blk.40.attn_q.weight", "blk.40.attn_k.weight", "blk.40.attn_v.weight",
        "blk.40.attn_output.weight", "blk.40.attn_q_norm.weight",
        "blk.40.attn_k_norm.weight", "blk.40.post_attention_norm.weight",
        "blk.40.ffn_gate_inp.weight", "blk.40.ffn_gate_exps.weight",
        "blk.40.ffn_up_exps.weight", "blk.40.ffn_down_exps.weight",
        "blk.40.ffn_gate_inp_shexp.weight", "blk.40.ffn_gate_shexp.weight",
        "blk.40.ffn_up_shexp.weight", "blk.40.ffn_down_shexp.weight",
        "blk.40.nextn.shared_head_norm.weight", "output.weight", "token_embd.weight",
    )

    class Reader:
        info = SimpleNamespace(
            metadata={},
            tensors=[
                SimpleNamespace(
                    name=tensor_name,
                    ggml_type=quant_module.GGMLQuantizationType.Q8_0,
                    shape=(1, 1),
                )
                for tensor_name in tensor_names
            ],
        )

        def __init__(self, path):
            pass

        def tensor_data(self, tensor_name):
            if tensor_name == "output.weight":
                return np.zeros((8, 1), dtype=np.uint8)
            if tensor_name == "token_embd.weight":
                return np.zeros((8, 8), dtype=np.float32)
            return np.zeros((1, 1), dtype=np.float32)

    class Tokenizer:
        @classmethod
        def from_gguf_info(cls, info):
            return cls()

        def decode(self, tokens):
            return "prompt"

    class Session:
        runtime = object()
        use_gemv_decode = True
        use_wmma_prefill = True

        def __init__(self, *args, **kwargs):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def close(self):
            pass

    class ReachedRunnerCache(Exception):
        pass

    constructed = []

    class Runner:
        def __init__(self, *args, **kwargs):
            constructed.append(os.environ.get(name))
            self.closed = False

        @property
        def _device_chain_enabled(self):
            raise ReachedRunnerCache

        def close(self):
            self.closed = True

    monkeypatch.setattr(gguf_module, "GGUFReader", Reader)
    monkeypatch.setattr(
        quant_module,
        "dequantize_gguf_data",
        lambda data, qtype: np.asarray(data, dtype=np.float32),
    )
    monkeypatch.setattr(tokenizer_module, "Qwen35GGUFTokenizer", Tokenizer)
    monkeypatch.setattr(runner_module, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setattr(
        draft_module, "Qwen35GGUFResidentMTPDraftRunner", Runner
    )

    def run(fused):
        argv = [
            "--model", str(model),
            "--resident-mtp-draft",
            "--root-topk-accept", "1",
        ]
        if fused:
            argv.append(option)
        with pytest.raises(ReachedRunnerCache):
            bench.main(argv)

    run(True)
    monkeypatch.delenv(name, raising=False)
    run(False)
    run(True)

    # The first and third runs share one cache key; the unfused run cannot reuse
    # the fused runner because the captured environment is part of that key.
    assert constructed == ["1", None]


def test_arg_parser_exposes_selected_down_x8_repack_mode() -> None:
    args = build_arg_parser().parse_args(["--selected-down-x8-repack", "q6"])

    assert args.selected_down_x8_repack == "q6"


@pytest.mark.parametrize("option", ["--selected-gate-up-x8", "--no-selected-gate-up-x8"])
def test_retired_selected_gate_up_x8_option_is_rejected(option: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_arg_parser().parse_args([option])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("false", False), ("0", False)])
def test_bench_keeps_ambient_x8_layout_for_parity_callers(monkeypatch, value, expected) -> None:
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_GATE_UP_X8", value)
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    assert not hasattr(build_arg_parser().parse_args([]), "selected_gate_up_x8")
    # Stop at argument validation, after route environment is applied and before loading.
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--root-topk-accept", "0"])
    assert excinfo.value.code == 2
    assert bench.gguf_selected_gate_up_x8_enabled() is expected
    assert bench.os.environ["HIPENGINE_GGUF_SELECTED_GATE_UP_X8"] == value


@pytest.mark.parametrize("option", ["--selected-gate-up-raw", "--no-selected-gate-up-raw"])
def test_retired_selected_gate_up_raw_option_is_rejected(option: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_arg_parser().parse_args([option])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("false", False), ("0", False)])
def test_bench_keeps_ambient_raw_layout_for_explicit_callers(monkeypatch, value, expected) -> None:
    monkeypatch.setattr(bench.os, "environ", dict(bench.os.environ))
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_GATE_UP_RAW", value)
    monkeypatch.setattr(bench, "_hip_available", lambda: True)
    assert not hasattr(build_arg_parser().parse_args([]), "selected_gate_up_raw")
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--root-topk-accept", "0"])
    assert excinfo.value.code == 2
    assert bench.gguf_selected_gate_up_raw_enabled() is expected
    assert bench.os.environ["HIPENGINE_GGUF_SELECTED_GATE_UP_RAW"] == value


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
            "--no-target-block-verify",
            "--target-block-verify-mode",
            "native",
            "--target-block-min-rows",
            "4",
            "--no-target-block-direct-state-commit",
            "--adaptive-draft-window",
            "--adaptive-ar-fallback",
            "--adaptive-ar-fallback-cooldown",
            "4",
            "--adaptive-full-vocab-after-cap-miss",
            "--adaptive-block-after-full-accept",
            "--fused-b1-block-probe",
            "--adaptive-probe-draft-n-max",
            "3",
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
    assert not hasattr(args, "target_graph_verify")
    assert not hasattr(args, "target_graph_batched_verify")
    assert args.target_block_verify is True
    assert args.target_block_verify_mode == "bulk"
    assert args.target_block_min_rows == 2
    assert args.target_block_direct_state_commit is True
    assert args.target_block_direct_partial_replay_mode == "direct-commit"
    assert not hasattr(args, "target_b1_branch_safe_block_verify")
    assert args.adaptive_draft_window is False
    assert args.adaptive_ar_fallback is False
    assert args.adaptive_ar_fallback_max_accepted == 0
    assert args.adaptive_ar_fallback_cooldown == 0
    assert args.adaptive_full_vocab_after_cap_miss is False
    assert args.adaptive_block_after_full_accept is False
    assert args.fused_b1_block_probe is False
    assert args.adaptive_probe_draft_n_max == 1
    assert not hasattr(args, "adaptive_strict_block_probe")


def test_apply_llama_compat_args_preserves_explicit_device_lifecycle_route_flags() -> None:
    args = build_arg_parser().parse_args(
        ["--llama-compat", "--resident-mtp-device-seed", "--resident-mtp-device-chain"]
    )

    apply_llama_compat_args(args)

    assert args.resident_mtp_device_seed is True
    assert args.resident_mtp_device_chain is True


def test_apply_llama_compat_args_preserves_explicit_partial_replay_diagnostic() -> None:
    args = build_arg_parser().parse_args(
        [
            "--llama-compat",
            "--target-block-direct-partial-replay-mode",
            "bulk-state-only",
        ]
    )

    apply_llama_compat_args(args)

    assert args.target_block_direct_partial_replay_mode == "bulk-state-only"


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


def test_main_rejects_draft_attention_debug_without_stage_stats(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--record-draft-attention-debug", "--mtp-device-kv-cache"])

    assert excinfo.value.code == 2
    assert "--record-draft-attention-debug requires --record-draft-stage-stats" in capsys.readouterr().err


def test_main_rejects_draft_attention_debug_without_device_kv(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--record-draft-attention-debug", "--record-draft-stage-stats"])

    assert excinfo.value.code == 2
    assert "--record-draft-attention-debug requires --mtp-device-kv-cache" in capsys.readouterr().err


def test_main_rejects_resident_draft_sync_stage_timings_without_cycle_timings(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--resident-mtp-draft", "--resident-mtp-draft-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert (
        "--resident-mtp-draft-sync-stage-timings requires --record-cycle-stage-timings"
        in capsys.readouterr().err
    )


def test_main_rejects_resident_draft_sync_stage_timings_without_resident_draft(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--record-cycle-stage-timings", "--resident-mtp-draft-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert "--resident-mtp-draft-sync-stage-timings requires --resident-mtp-draft" in capsys.readouterr().err


def test_main_rejects_target_block_sync_stage_timings_without_cycle_timings(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--target-block-verify", "--target-block-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert "--target-block-sync-stage-timings requires --record-cycle-stage-timings" in capsys.readouterr().err


def test_main_rejects_target_block_sync_stage_timings_without_block_verify(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--record-cycle-stage-timings", "--target-block-sync-stage-timings"])

    assert excinfo.value.code == 2
    assert "--target-block-sync-stage-timings requires --target-block-verify" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [
    ("--adaptive-strict-block-probe", None),
    ("--no-adaptive-strict-block-probe", None),
    ("--adaptive-strict-probe-cycles", "2"),
    ("--adaptive-strict-probe-min-accepted", "2"),
    ("--adaptive-strict-fallback-draft-n-max", "1"),
    ("--adaptive-strict-fallback-root-topk", "40"),
])
def test_retired_strict_probe_options_fail_before_hip(option, value, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "_hip_available", lambda: pytest.fail("retired strict probe reached HIP"))
    with pytest.raises(SystemExit) as excinfo:
        bench.main([option, *([value] if value is not None else [])])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_retired_strict_probe_artifacts_keep_inactive_defaults_without_runtime_policy() -> None:
    tree = ast.parse(inspect.getsource(bench))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr.startswith("adaptive_strict_")
        or isinstance(node, ast.Name) and node.id.startswith("adaptive_strict_")
        for node in ast.walk(tree)
    )
    expected = {
        "adaptive_strict_block_probe": False,
        "adaptive_strict_probe_cycles": 2,
        "adaptive_strict_probe_min_accepted": 2,
        "adaptive_strict_fallback_draft_n_max": 1,
        "adaptive_strict_fallback_root_topk": 40,
        "adaptive_strict_probe_decision": None,
        "adaptive_strict_probe_history": [],
    }
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in expected:
                    assert ast.literal_eval(value) == expected[key.value]
                    seen.add(key.value)
    assert seen == expected.keys()


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
