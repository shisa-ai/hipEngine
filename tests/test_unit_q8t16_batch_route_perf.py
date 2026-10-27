from __future__ import annotations

import pytest

from scripts.q8t16_batch_route_perf import (
    KIND,
    PerformanceQualityError,
    _execution_affecting_dirty,
    revalidate_generated_id_policy,
    validate_quality_artifact,
    summarize_by_configuration,
)


def test_summarize_by_configuration_keeps_matched_pair_ratios() -> None:
    runs = [
        {"configuration": "c2", "label": "strict", "pair": 1, "decode_tok_s": 75.0},
        {"configuration": "c2", "label": "candidate", "pair": 1, "decode_tok_s": 76.5},
        {"configuration": "c2", "label": "candidate", "pair": 2, "decode_tok_s": 73.5},
        {"configuration": "c2", "label": "strict", "pair": 2, "decode_tok_s": 75.0},
        {"configuration": "c4", "label": "strict", "pair": 1, "decode_tok_s": 100.0},
        {"configuration": "c4", "label": "candidate", "pair": 1, "decode_tok_s": 101.0},
        {"configuration": "c4", "label": "candidate", "pair": 2, "decode_tok_s": 98.0},
        {"configuration": "c4", "label": "strict", "pair": 2, "decode_tok_s": 100.0},
        {"configuration": "c8", "label": "strict", "pair": 1, "decode_tok_s": 150.0},
        {"configuration": "c8", "label": "candidate", "pair": 1, "decode_tok_s": 153.0},
        {"configuration": "c8", "label": "candidate", "pair": 2, "decode_tok_s": 147.0},
        {"configuration": "c8", "label": "strict", "pair": 2, "decode_tok_s": 150.0},
    ]

    summary = summarize_by_configuration(runs)

    assert summary["c2"]["candidate_over_strict"] == pytest.approx(1.0)
    assert summary["c2"]["paired_ratios"] == pytest.approx([1.02, 0.98])
    assert summary["c4"]["candidate_over_strict"] == pytest.approx(0.995)
    assert summary["c4"]["paired_ratios"] == pytest.approx([1.01, 0.98])
    assert summary["c8"]["candidate_over_strict"] == pytest.approx(1.0)
    assert summary["c8"]["paired_ratios"] == pytest.approx([1.02, 0.98])


def test_validate_quality_artifact_requires_bundled_router_when_requested() -> None:
    artifact = {
        "kind": "hipengine_execution_profile_gguf_batch_route_requalification_capture",
        "status": "complete",
        "measurement_valid": True,
        "protocol": {"complete_prompt_and_heldout_suite": True},
        "quality": {"hard_gates_passed": True},
        "repeat_determinism": {"passed": True},
        "route": {"router_candidate_included": True},
    }

    validate_quality_artifact(artifact, include_router_candidate=True)
    artifact["route"]["router_candidate_included"] = False
    with pytest.raises(PerformanceQualityError, match="router candidate"):
        validate_quality_artifact(artifact, include_router_candidate=True)


def _complete_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for pair in range(1, 8):
        runs.extend(
            [
                {
                    "configuration": "c2",
                    "label": "strict",
                    "pair": pair,
                    "decode_tok_s": 75.0,
                    "trajectory_sha256": [f"strict-{pair}"],
                },
                {
                    "configuration": "c2",
                    "label": "candidate",
                    "pair": pair,
                    "decode_tok_s": 76.0,
                    "trajectory_sha256": [f"candidate-{pair}"],
                },
            ]
        )
    return runs


def _revalidatable_source(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "kind": KIND,
        "status": "invalid_or_screen_only",
        "measurement_valid": False,
        "performance_claim": False,
        "protocol": {
            "counterbalanced_pairs": 7,
            "prompt_tokens": 512,
            "decode_steps": 128,
            "configurations": ["c2"],
        },
        "provenance": provenance,
        "memory": {"teardown_exact": True},
        "runs": _complete_runs(),
    }


def test_revalidate_accepts_a_prose_only_dirty_worktree() -> None:
    """Docs, worklog entries, and other agents' scratch prose cannot change a run."""

    source = _revalidatable_source(
        {
            "dirty": True,
            "staged_dirty": False,
            "unstaged_dirty": True,
            "untracked_dirty": True,
            "dirty_paths": ["docs/REFACTOR.md"],
            "untracked_paths": ["worklog/entries/20261027T000000.000000Z-note.md"],
            "execution_affecting_dirty": False,
            "execution_affecting_paths": [],
        }
    )

    result = revalidate_generated_id_policy(
        source,
        source_sha256="abc",
        command=["python", "revalidate"],
        revalidator_commit="deadbeef",
    )

    assert result["measurement_valid"] is True
    assert result["performance_claim"] is True
    assert result["policy_revalidation"]["mechanical_checks"]["clean_source_provenance"] is True


def test_revalidate_refuses_a_kernel_dirty_worktree() -> None:
    """An execution-affecting change still refuses, however small the diff."""

    source = _revalidatable_source(
        {
            "dirty": True,
            "staged_dirty": True,
            "unstaged_dirty": False,
            "untracked_dirty": False,
            "dirty_paths": ["hipengine/runtime/qwen35_gguf_runner.py"],
            "untracked_paths": [],
            "execution_affecting_dirty": True,
            "execution_affecting_paths": ["hipengine/runtime/qwen35_gguf_runner.py"],
        }
    )

    with pytest.raises(ValueError, match="timing protocol"):
        revalidate_generated_id_policy(
            source,
            source_sha256="abc",
            command=["python", "revalidate"],
            revalidator_commit="deadbeef",
        )


def test_revalidate_refuses_an_untracked_execution_affecting_file() -> None:
    """An untracked non-prose path is execution-affecting, not scratch."""

    source = _revalidatable_source(
        {
            "dirty": True,
            "staged_dirty": False,
            "unstaged_dirty": False,
            "untracked_dirty": True,
            "dirty_paths": [],
            "untracked_paths": ["scratch_probe.py"],
            "execution_affecting_dirty": True,
            "execution_affecting_paths": ["scratch_probe.py"],
        }
    )

    with pytest.raises(ValueError, match="timing protocol"):
        revalidate_generated_id_policy(
            source,
            source_sha256="abc",
            command=["python", "revalidate"],
            revalidator_commit="deadbeef",
        )


def test_revalidate_keeps_the_strict_reading_for_pre_schema3_provenance() -> None:
    """Artifacts without the classified flag cannot separate prose from kernels."""

    source = _revalidatable_source({"dirty": True})

    with pytest.raises(ValueError, match="timing protocol"):
        revalidate_generated_id_policy(
            source,
            source_sha256="abc",
            command=["python", "revalidate"],
            revalidator_commit="deadbeef",
        )


def test_execution_affecting_dirty_prefers_the_classified_flag() -> None:
    assert _execution_affecting_dirty({"dirty": True, "execution_affecting_dirty": False}) is False
    assert _execution_affecting_dirty({"dirty": False, "execution_affecting_dirty": True}) is True
    assert _execution_affecting_dirty({"dirty": True}) is True
    assert _execution_affecting_dirty({"dirty": False}) is False
    # A present but malformed flag is never read as clean.
    assert _execution_affecting_dirty({"dirty": True, "execution_affecting_dirty": None}) is True


def test_revalidate_generated_id_policy_keeps_mismatch_diagnostic() -> None:
    runs = []
    for pair in range(1, 8):
        runs.extend(
            [
                {
                    "configuration": "c2",
                    "label": "strict",
                    "pair": pair,
                    "decode_tok_s": 75.0,
                    "trajectory_sha256": [f"strict-{pair}"],
                },
                {
                    "configuration": "c2",
                    "label": "candidate",
                    "pair": pair,
                    "decode_tok_s": 76.0,
                    "trajectory_sha256": [f"candidate-{pair}"],
                },
            ]
        )
    source = {
        "kind": KIND,
        "status": "invalid_or_screen_only",
        "measurement_valid": False,
        "performance_claim": False,
        "protocol": {
            "counterbalanced_pairs": 7,
            "prompt_tokens": 512,
            "decode_steps": 128,
            "configurations": ["c2"],
        },
        "provenance": {"dirty": False},
        "memory": {"teardown_exact": True},
        "runs": runs,
    }

    result = revalidate_generated_id_policy(
        source,
        source_sha256="abc",
        command=["python", "revalidate"],
        revalidator_commit="deadbeef",
    )

    assert result["measurement_valid"] is True
    assert result["performance_claim"] is True
    assert result["generated_id_equality"] == {
        "binding": False,
        "reason": (
            "production-profile free-running equality is diagnostic; "
            "strict-teacher full logits are the binding quality comparison"
        ),
        "all_trajectories_exact": False,
    }
    assert result["policy_revalidation"]["source_measurement_valid"] is False
