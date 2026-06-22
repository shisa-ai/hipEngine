from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.gguf_mtp_category_bench import (
    DEFAULT_FULL_PROMPT_IDS,
    DEFAULT_HELDOUT_PROMPT_IDS,
    DEFAULT_PROMPTS,
    BenchError,
    build_split_contract,
    build_summary,
    compare_objective_metrics,
    load_prompt_rows,
    objective_metrics_for_budget,
    prompt_sha256,
    validate_speed_claim_contract,
    write_markdown,
)
from scripts.gguf_true_ar_category_bench import build_true_ar_artifact


TEST_REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TEST_PROMPTS = "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
TEST_REPO_PROVENANCE = {
    "repo_root": str(TEST_REPO_ROOT),
    "git_commit": "test-commit",
    "git_branch": "test-branch",
    "git_tracked_dirty": False,
    "git_untracked_count": 0,
}
TEST_SUMMARY_COMMANDS = ["python3 scripts/gguf_mtp_category_bench.py --budgets 1 --output summary.json"]
TEST_TRUE_AR_COMMANDS = ["python3 scripts/gguf_true_ar_category_bench.py --output true-ar.json"]
TEST_TRUE_AR_PROTOCOL = {
    "model": TEST_MODEL,
    "model_normalized": str(Path(TEST_MODEL).resolve(strict=False)),
    "quant": "UD-Q4_K_M GGUF",
    "prompt_file": TEST_PROMPTS,
    "prompt_file_normalized": str((TEST_REPO_ROOT / TEST_PROMPTS).resolve(strict=False)),
    "decode_tokens": 10,
    "warmup_decode_tokens": 1,
    "prompt_count": 10,
}


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
    assert summary["repo"]["repo_root"]
    assert set(summary["repo"]) == {"repo_root", "git_commit", "git_branch", "git_tracked_dirty", "git_untracked_count"}
    assert summary["performance_claim"] is False
    assert summary["speed_claim_eligible"] is False
    assert summary["true_ar_comparison_available"] is False
    assert "true AR baseline" in summary["promotion_blocker"]
    assert summary["ar_baseline_contract"] == {
        "required_for_speed_claims": "true_no_mtp_autoregressive_generation",
        "current_off_kind": "verifier_derived_from_b1_target_ar",
        "current_off_true_autoregressive_path": False,
    }
    assert summary["true_ar_baseline"] == {
        "available": False,
        "true_autoregressive_path": False,
        "same_prompt_suite": False,
        "same_timing_protocol": False,
        "source": None,
    }
    assert summary["prompts"][0]["prompt_sha256"] == prompt_sha256("write code")
    assert summary["totals"]["off"]["baseline_kind"] == "verifier_derived_from_b1_target_ar"
    assert summary["totals"]["off"]["true_autoregressive_path"] is False
    assert summary["categories"]["code"]["off"]["true_autoregressive_path"] is False


def test_category_summary_rejects_impossible_acceptance_metrics() -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=2, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="accepted draft tokens exceed proposed drafts"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_non_positive_total_cycle_ms() -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    row = _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)
    row["metrics"]["total_cycle_ms"] = 0.0
    raw = {1: [row]}

    with pytest.raises(BenchError, match="non-positive total_cycle_ms"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ar_decode_ms", -1.0, "negative timing"),
        ("mtp_draft_ms", float("nan"), "non-finite timing"),
    ],
)
def test_category_summary_rejects_invalid_cycle_timings(field: str, value: float, message: str) -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    row = _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)
    row["cycles"][0][field] = value
    raw = {1: [row]}

    with pytest.raises(BenchError, match=message):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


@pytest.mark.parametrize(
    ("raw_rows", "message"),
    [
        (
            [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)],
            "missing prompt rows",
        ),
        (
            [
                _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
                _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
                _row("general_1", "general_en", output=20, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
            ],
            "duplicate prompt rows",
        ),
        (
            [
                _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
                _row("general_1", "code", output=20, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
            ],
            "category mismatch",
        ),
    ],
)
def test_category_summary_rejects_bad_budget_prompt_coverage(raw_rows: list[dict], message: str) -> None:
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

    with pytest.raises(BenchError, match=message):
        build_summary(args=args, prompts=prompts, raw={1: raw_rows}, commands=[])


def test_category_summary_reports_train_heldout_and_full_suite_metrics() -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [
        {"id": "code_merge_intervals", "category": "code", "prompt": "p"},
        {"id": "code_topological_sort", "category": "code", "prompt": "p"},
        {"id": "code_lru_cache", "category": "code", "prompt": "p"},
        {"id": "code_markdown_table", "category": "code", "prompt": "p"},
        {"id": "general_en_plan", "category": "general_en", "prompt": "p"},
        {"id": "general_en_explain", "category": "general_en", "prompt": "p"},
        {"id": "general_ja_plan", "category": "general_ja", "prompt": "p"},
        {"id": "general_ja_explain", "category": "general_ja", "prompt": "p"},
        {"id": "mixed_ja_en_translate", "category": "mixed_ja_en", "prompt": "p"},
        {"id": "mixed_ja_en_review", "category": "mixed_ja_en", "prompt": "p"},
    ]

    def rows_for_budget(budget: int) -> list[dict]:
        rows = []
        for prompt in prompts:
            is_heldout = prompt["id"] in DEFAULT_HELDOUT_PROMPT_IDS
            accepted = min(2 if is_heldout else 1, budget)
            rows.append(
                _row(
                    prompt["id"],
                    prompt["category"],
                    output=10,
                    accepted=accepted,
                    drafts=budget,
                    ar_ms=10.0,
                    draft_ms=float(budget),
                )
            )
        return rows

    summary = build_summary(args=args, prompts=prompts, raw={1: rows_for_budget(1), 5: rows_for_budget(5)}, commands=[])
    split_contract = summary["splits"]["contract"]

    assert split_contract["heldout_ids"] == [
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    ]
    assert split_contract["train_ids"] == [
        "code_merge_intervals",
        "code_topological_sort",
        "code_lru_cache",
        "general_en_plan",
        "general_ja_plan",
        "mixed_ja_en_translate",
    ]
    assert split_contract["heldout_has_all_present_categories"] is True
    assert split_contract["missing_default_heldout_ids"] == []

    assert summary["splits"]["full"]["metrics"]["b5"]["prompts"] == 10
    assert summary["splits"]["train"]["metrics"]["b5"]["prompts"] == 6
    assert summary["splits"]["heldout"]["metrics"]["b5"]["prompts"] == 4
    assert summary["splits"]["train"]["metrics"]["b5"]["accepted_per_output"] == 0.1
    assert summary["splits"]["heldout"]["metrics"]["b5"]["accepted_per_output"] == 0.2
    assert summary["splits"]["full"]["metrics"]["b5"]["accepted_per_output"] == 0.14


def test_default_prompt_fixture_keeps_one_heldout_per_category() -> None:
    prompts = load_prompt_rows(DEFAULT_PROMPTS)
    contract = build_split_contract(prompts)

    assert len(prompts) == 10
    assert contract["default_full_ids"] == list(DEFAULT_FULL_PROMPT_IDS)
    assert contract["full_ids"] == list(DEFAULT_FULL_PROMPT_IDS)
    assert contract["full_suite_matches_default"] is True
    assert set(contract["heldout_ids"]) == DEFAULT_HELDOUT_PROMPT_IDS
    assert contract["heldout_ids"] == [
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    ]
    assert len(contract["train_ids"]) == 6
    assert contract["heldout_has_all_present_categories"] is True
    assert contract["missing_default_heldout_ids"] == []
    assert contract["missing_default_full_ids"] == []
    assert contract["extra_vs_default_full_ids"] == []
    assert contract["heldout_categories"] == ["code", "general_en", "general_ja", "mixed_ja_en"]


def test_speed_claim_contract_rejects_verifier_derived_ar_baseline() -> None:
    summary = {
        "speed_claim_eligible": True,
        "repo": dict(TEST_REPO_PROVENANCE),
        "commands": list(TEST_SUMMARY_COMMANDS),
        "true_ar_baseline": {
            "available": False,
            "true_autoregressive_path": False,
            "same_prompt_suite": False,
            "same_timing_protocol": False,
            "source": None,
            "repo": dict(TEST_REPO_PROVENANCE),
            "commands": list(TEST_TRUE_AR_COMMANDS),
        },
    }

    with pytest.raises(BenchError, match="true no-MTP autoregressive baseline"):
        validate_speed_claim_contract(summary)


def test_speed_claim_contract_accepts_same_protocol_true_ar_baseline() -> None:
    summary = {
        "speed_claim_eligible": True,
        "repo": dict(TEST_REPO_PROVENANCE),
        "commands": list(TEST_SUMMARY_COMMANDS),
        "true_ar_baseline": {
            "available": True,
            "true_autoregressive_path": True,
            "same_prompt_suite": True,
            "same_timing_protocol": True,
            "source": "future true AR harness artifact",
            "repo": dict(TEST_REPO_PROVENANCE),
            "commands": list(TEST_TRUE_AR_COMMANDS),
            "protocol": dict(TEST_TRUE_AR_PROTOCOL),
        },
    }

    assert validate_speed_claim_contract(summary) is summary


def test_speed_claim_contract_rejects_missing_command_provenance() -> None:
    summary = {
        "speed_claim_eligible": True,
        "repo": dict(TEST_REPO_PROVENANCE),
        "true_ar_baseline": {
            "available": True,
            "true_autoregressive_path": True,
            "same_prompt_suite": True,
            "same_timing_protocol": True,
            "source": "future true AR harness artifact",
            "repo": dict(TEST_REPO_PROVENANCE),
            "commands": list(TEST_TRUE_AR_COMMANDS),
        },
    }

    with pytest.raises(BenchError, match="speed-claim summary requires non-empty commands provenance"):
        validate_speed_claim_contract(summary)


def test_speed_claim_contract_rejects_missing_true_ar_command_provenance() -> None:
    summary = {
        "speed_claim_eligible": True,
        "repo": dict(TEST_REPO_PROVENANCE),
        "commands": list(TEST_SUMMARY_COMMANDS),
        "true_ar_baseline": {
            "available": True,
            "true_autoregressive_path": True,
            "same_prompt_suite": True,
            "same_timing_protocol": True,
            "source": "future true AR harness artifact",
            "repo": dict(TEST_REPO_PROVENANCE),
        },
    }

    with pytest.raises(BenchError, match="speed-claim true_ar_baseline requires non-empty commands provenance"):
        validate_speed_claim_contract(summary)


def test_speed_claim_contract_rejects_missing_true_ar_protocol() -> None:
    summary = {
        "speed_claim_eligible": True,
        "repo": dict(TEST_REPO_PROVENANCE),
        "commands": list(TEST_SUMMARY_COMMANDS),
        "true_ar_baseline": {
            "available": True,
            "true_autoregressive_path": True,
            "same_prompt_suite": True,
            "same_timing_protocol": True,
            "source": "future true AR harness artifact",
            "repo": dict(TEST_REPO_PROVENANCE),
            "commands": list(TEST_TRUE_AR_COMMANDS),
        },
    }

    with pytest.raises(BenchError, match="speed-claim true_ar_baseline requires true AR protocol metadata"):
        validate_speed_claim_contract(summary)


def _write_true_ar_baseline(
    path: Path,
    rows: list[dict],
    *,
    prompt_text_by_id: dict[str, str],
    same_prompt_suite: bool | None = True,
    include_repo: bool = True,
    include_prompt_hashes: bool = True,
    include_row_hashes: bool = True,
    include_commands: bool = True,
    model: str = TEST_MODEL,
    prompt_file: str = TEST_PROMPTS,
    prompt_count: int | None = None,
) -> None:
    prompt_hashes = {str(prompt_id): prompt_sha256(text) for prompt_id, text in prompt_text_by_id.items()}
    metric_rows = []
    for row in rows:
        row_copy = dict(row)
        prompt_id = str(row_copy.get("id") or row_copy.get("prompt_id") or "")
        if include_row_hashes and prompt_id in prompt_hashes:
            row_copy.setdefault("prompt_sha256", prompt_hashes[prompt_id])
        metric_rows.append(row_copy)
    payload = {
        "kind": "hipengine_gguf_true_ar_category_baseline",
        "true_autoregressive_path": True,
        "same_timing_protocol": True,
        "model": model,
        "quant": "UD-Q4_K_M GGUF",
        "prompt_file": prompt_file,
        "decode_tokens": 10,
        "warmup_decode_tokens": 1,
        "prompt_count": len(prompt_text_by_id) if prompt_count is None else prompt_count,
        "prompt_metrics": metric_rows,
    }
    if include_prompt_hashes:
        payload["prompt_hashes"] = prompt_hashes
    if include_commands:
        payload["commands"] = list(TEST_TRUE_AR_COMMANDS)
    if include_repo:
        payload["repo"] = dict(TEST_REPO_PROVENANCE)
    if same_prompt_suite is not None:
        payload["same_prompt_suite"] = same_prompt_suite
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_category_summary_attaches_valid_true_ar_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    prompts = [
        {"id": "code_1", "category": "code", "prompt": "write code"},
        {"id": "general_1", "category": "general_en", "prompt": "explain"},
    ]
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_1", "category": "general_en", "output_tokens": 20, "decode_ms": 200.0},
        ],
        prompt_text_by_id={row["id"]: row["prompt"] for row in prompts},
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    raw = {
        1: [
            _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
            _row("general_1", "general_en", output=20, accepted=2, drafts=2, ar_ms=200.0, draft_ms=20.0),
        ]
    }

    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=list(TEST_SUMMARY_COMMANDS))

    assert summary["speed_claim_eligible"] is False
    assert summary["true_ar_comparison_available"] is True
    assert "not a retained speed claim" in summary["promotion_blocker"]
    assert summary["true_ar_baseline"]["available"] is True
    assert summary["true_ar_baseline"]["same_prompt_suite"] is True
    assert summary["true_ar_baseline"]["same_timing_protocol"] is True
    assert summary["true_ar_baseline"]["repo"] == TEST_REPO_PROVENANCE
    assert summary["true_ar_baseline"]["protocol"]["model"] == TEST_MODEL
    assert summary["true_ar_baseline"]["protocol"]["prompt_file"] == TEST_PROMPTS
    assert summary["true_ar_baseline"]["protocol"]["prompt_count"] == 2
    assert summary["true_ar_baseline"]["totals"]["decode_tok_s_weighted"] == 100.0
    assert summary["objective_metrics_available"] is False
    assert "heldout coverage" in summary["objective_metrics_blocker"]
    assert summary["objectives"] == {}
    assert summary["totals"]["b1"]["true_ar_decode_tok_s_weighted"] == 100.0
    assert summary["totals"]["b1"]["mtp_vs_true_ar_decode_ratio"] == pytest.approx((30.0 / 330.0 * 1000.0) / 100.0)
    assert summary["categories"]["code"]["b1"]["mtp_vs_true_ar_decode_ratio"] == pytest.approx((10.0 / 110.0 * 1000.0) / 100.0)


def _default_objective_summary(tmp_path: Path, name: str, *, accepted: list[int], draft_ms: float, true_ar_name: str = "shared") -> dict:
    prompts = load_prompt_rows(DEFAULT_PROMPTS)
    assert [row["id"] for row in prompts] == list(DEFAULT_FULL_PROMPT_IDS)
    assert len(accepted) == len(prompts)
    baseline_path = tmp_path / f"{true_ar_name}-true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": row["id"], "category": row["category"], "output_tokens": 10, "decode_ms": 100.0}
            for row in prompts
        ],
        prompt_text_by_id={row["id"]: row["prompt"] for row in prompts},
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    raw = {
        1: [
            _row(row["id"], row["category"], output=10, accepted=acc, drafts=max(acc, 1), ar_ms=100.0, draft_ms=draft_ms)
            for row, acc in zip(prompts, accepted, strict=True)
        ]
    }
    return build_summary(args=args, prompts=prompts, raw=raw, commands=list(TEST_SUMMARY_COMMANDS))


def test_objective_metrics_for_budget_requires_full_default_suite(tmp_path: Path) -> None:
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], draft_ms=10.0)

    metrics = objective_metrics_for_budget(summary, 1)

    assert summary["objective_metrics_available"] is True
    assert summary["objective_metrics_blocker"] is None
    assert summary["objectives"]["b1"] == metrics
    assert metrics["budget"] == "b1"
    assert metrics["true_ar_comparison_available"] is True
    assert metrics["speed_claim_eligible"] is False
    assert metrics["performance_claim"] is False
    assert metrics["summary_commands"] == TEST_SUMMARY_COMMANDS
    assert metrics["true_ar_commands"] == TEST_TRUE_AR_COMMANDS
    assert metrics["true_ar_protocol"]["model"] == TEST_MODEL
    assert metrics["true_ar_protocol"]["prompt_file"] == TEST_PROMPTS
    assert metrics["true_ar_protocol"]["prompt_count"] == 10
    assert metrics["full"]["accepted_per_output"] == pytest.approx(55 / 100)
    assert metrics["full"]["draft_acceptance"] == pytest.approx(1.0)
    assert metrics["full"]["mtp_vs_true_ar_decode_ratio"] == pytest.approx((100.0 / 1100.0 * 1000.0) / 100.0)
    assert metrics["train"]["prompts"] == 6
    assert metrics["train"]["accepted_per_output"] == pytest.approx(27 / 60)
    assert metrics["heldout"]["prompts"] == 4
    assert metrics["heldout"]["accepted_per_output"] == pytest.approx(28 / 40)
    assert metrics["heldout_ids"] == [
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    ]


def test_objective_metrics_for_budget_rejects_missing_summary_repo_provenance(tmp_path: Path) -> None:
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1] * 10, draft_ms=10.0)
    del summary["repo"]

    with pytest.raises(BenchError, match="objective summary requires repo provenance metadata"):
        objective_metrics_for_budget(summary, "b1")


def test_objective_metrics_for_budget_rejects_missing_attached_true_ar_repo_provenance(tmp_path: Path) -> None:
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1] * 10, draft_ms=10.0)
    del summary["true_ar_baseline"]["repo"]

    with pytest.raises(BenchError, match="attached true_ar_baseline requires repo provenance metadata"):
        objective_metrics_for_budget(summary, "b1")


def test_objective_metrics_for_budget_rejects_missing_attached_true_ar_protocol(tmp_path: Path) -> None:
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1] * 10, draft_ms=10.0)
    del summary["true_ar_baseline"]["protocol"]
    summary["objectives"] = {}

    with pytest.raises(BenchError, match="attached true_ar_baseline requires true AR protocol metadata"):
        objective_metrics_for_budget(summary, "b1")


def test_objective_metrics_for_budget_rejects_missing_summary_command_provenance(tmp_path: Path) -> None:
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1] * 10, draft_ms=10.0)
    summary["commands"] = []
    summary["objectives"] = {}

    with pytest.raises(BenchError, match="objective summary requires non-empty commands provenance"):
        objective_metrics_for_budget(summary, "b1")


def test_objective_metrics_for_budget_rejects_missing_attached_true_ar_command_provenance(tmp_path: Path) -> None:
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1] * 10, draft_ms=10.0)
    del summary["true_ar_baseline"]["commands"]
    summary["objectives"] = {}

    with pytest.raises(BenchError, match="attached true_ar_baseline requires non-empty commands provenance"):
        objective_metrics_for_budget(summary, "b1")


def test_objective_metrics_for_budget_rejects_partial_suite_even_with_present_category_heldouts(tmp_path: Path) -> None:
    baseline_path = tmp_path / "partial-true-ar.json"
    prompts = [
        {"id": "code_merge_intervals", "category": "code", "prompt": "write code"},
        {"id": "code_markdown_table", "category": "code", "prompt": "write table"},
        {"id": "general_en_plan", "category": "general_en", "prompt": "plan"},
        {"id": "general_en_explain", "category": "general_en", "prompt": "explain"},
    ]
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_merge_intervals", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "code_markdown_table", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_en_plan", "category": "general_en", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_en_explain", "category": "general_en", "output_tokens": 10, "decode_ms": 100.0},
        ],
        prompt_text_by_id={row["id"]: row["prompt"] for row in prompts},
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    raw = {
        1: [
            _row(row["id"], row["category"], output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)
            for row in prompts
        ]
    }
    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=list(TEST_SUMMARY_COMMANDS))

    assert summary["splits"]["contract"]["heldout_has_all_present_categories"] is True
    assert summary["splits"]["contract"]["full_suite_matches_default"] is False
    assert summary["objective_metrics_available"] is False
    assert "full default" in summary["objective_metrics_blocker"]
    with pytest.raises(BenchError, match="full default"):
        objective_metrics_for_budget(summary, "b1")


def test_compare_objective_metrics_passes_when_full_and_heldout_do_not_regress(tmp_path: Path) -> None:
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=20.0)
    candidate = _default_objective_summary(tmp_path, "candidate", accepted=[2] * 10, draft_ms=10.0)

    comparison = compare_objective_metrics(baseline, candidate, "b1")

    assert comparison["passed"] is True
    assert comparison["regressions"] == []
    assert comparison["deltas"]["full"]["accepted_per_output"] > 0
    assert comparison["deltas"]["heldout"]["accepted_per_output"] > 0
    assert comparison["deltas"]["full"]["mtp_vs_true_ar_decode_ratio"] > 0
    assert "train deltas are report-only" in comparison["decision_rule"]


def test_compare_objective_metrics_rejects_changed_true_ar_baseline(tmp_path: Path) -> None:
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=10.0, true_ar_name="baseline-ar")
    candidate = _default_objective_summary(tmp_path, "candidate", accepted=[2] * 10, draft_ms=10.0, true_ar_name="candidate-ar")

    with pytest.raises(BenchError, match="identical attached true_ar_baseline"):
        compare_objective_metrics(baseline, candidate, "b1")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary.update({"cycles": 2}),
        lambda summary: summary["prompts"][0].update({"prompt_chars": summary["prompts"][0]["prompt_chars"] + 1}),
        lambda summary: summary["prompts"][0].update({"prompt_sha256": "0" * 64}),
    ],
)
def test_compare_objective_metrics_rejects_changed_protocol_metadata(tmp_path: Path, mutate) -> None:
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=10.0)
    candidate = copy.deepcopy(baseline)
    mutate(candidate)

    with pytest.raises(BenchError, match="identical benchmark protocol metadata"):
        compare_objective_metrics(baseline, candidate, "b1")


def test_compare_objective_metrics_rejects_heldout_acceptance_regression(tmp_path: Path) -> None:
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=10.0)
    candidate = _default_objective_summary(tmp_path, "candidate", accepted=[2, 2, 2, 0, 2, 0, 2, 0, 2, 0], draft_ms=10.0)

    comparison = compare_objective_metrics(baseline, candidate, "b1")

    assert comparison["passed"] is False
    assert {
        "split": "heldout",
        "field": "accepted_per_output",
        "baseline": comparison["baseline"]["heldout"]["accepted_per_output"],
        "candidate": comparison["candidate"]["heldout"]["accepted_per_output"],
        "delta": comparison["deltas"]["heldout"]["accepted_per_output"],
    } in comparison["regressions"]
    assert comparison["deltas"]["train"]["accepted_per_output"] > 0


def test_compare_objective_metrics_rejects_true_ar_ratio_regression(tmp_path: Path) -> None:
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=10.0)
    candidate = _default_objective_summary(tmp_path, "candidate", accepted=[1] * 10, draft_ms=200.0)

    comparison = compare_objective_metrics(baseline, candidate, "b1")

    assert comparison["passed"] is False
    assert any(
        regression["field"] == "mtp_vs_true_ar_decode_ratio" and regression["split"] in {"full", "heldout"}
        for regression in comparison["regressions"]
    )


def test_objective_metrics_cli_prints_guarded_metrics(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], draft_ms=10.0)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_mtp_category_bench.py"),
            "--objective-summary-json",
            str(summary_path),
            "--objective-budget",
            "b1",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    metrics = json.loads(completed.stdout)
    assert metrics["budget"] == "b1"
    assert metrics["full"]["accepted_per_output"] == pytest.approx(55 / 100)
    assert metrics["heldout"]["prompts"] == 4
    assert metrics["speed_claim_eligible"] is False


def test_objective_metrics_cli_can_print_scalar_metric(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], draft_ms=10.0)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_mtp_category_bench.py"),
            "--objective-summary-json",
            str(summary_path),
            "--objective-budget",
            "b1",
            "--objective-split",
            "heldout",
            "--objective-field",
            "accepted_per_output",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert float(completed.stdout.strip()) == pytest.approx(28 / 40)


def test_objective_metrics_cli_requires_scalar_split_and_field_together(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = _default_objective_summary(tmp_path, "summary", accepted=[1] * 10, draft_ms=10.0)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_mtp_category_bench.py"),
            "--objective-summary-json",
            str(summary_path),
            "--objective-budget",
            "b1",
            "--objective-split",
            "full",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "--objective-split and --objective-field must be provided together" in completed.stderr


def test_compare_objective_metrics_cli_prints_comparison(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=20.0)
    candidate = _default_objective_summary(tmp_path, "candidate", accepted=[2] * 10, draft_ms=10.0)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_mtp_category_bench.py"),
            "--compare-baseline-summary-json",
            str(baseline_path),
            "--compare-candidate-summary-json",
            str(candidate_path),
            "--compare-budget",
            "b1",
            "--compare-require-pass",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    comparison = json.loads(completed.stdout)
    assert comparison["passed"] is True
    assert comparison["regressions"] == []
    assert comparison["deltas"]["full"]["accepted_per_output"] > 0


def test_compare_objective_metrics_cli_can_fail_on_regression(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline = _default_objective_summary(tmp_path, "baseline", accepted=[1] * 10, draft_ms=10.0)
    candidate = _default_objective_summary(tmp_path, "candidate", accepted=[2, 2, 2, 0, 2, 0, 2, 0, 2, 0], draft_ms=10.0)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_mtp_category_bench.py"),
            "--compare-baseline-summary-json",
            str(baseline_path),
            "--compare-candidate-summary-json",
            str(candidate_path),
            "--compare-budget",
            "b1",
            "--compare-require-pass",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    comparison = json.loads(completed.stdout)
    assert comparison["passed"] is False
    assert any(regression["split"] == "heldout" for regression in comparison["regressions"])


def test_objective_metrics_cli_rejects_verifier_only_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}
    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])
    summary_path = tmp_path / "verifier-only-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_mtp_category_bench.py"),
            "--objective-summary-json",
            str(summary_path),
            "--objective-budget",
            "b1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "true_ar_comparison_available=true" in completed.stderr


def test_objective_metrics_for_budget_rejects_verifier_only_summary() -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}
    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])

    assert summary["objective_metrics_available"] is False
    assert summary["objective_metrics_blocker"] == "true AR comparison is not attached"
    assert summary["objectives"] == {}
    with pytest.raises(BenchError, match="true_ar_comparison_available=true"):
        objective_metrics_for_budget(summary, "b1")


@pytest.mark.parametrize(
    ("row_patch", "message"),
    [
        ({"decode_ms": float("nan")}, "non-finite timing"),
        ({"output_tokens": 0}, "positive output_tokens"),
    ],
)
def test_category_summary_rejects_invalid_true_ar_prompt_metrics(tmp_path: Path, row_patch: dict, message: str) -> None:
    baseline_path = tmp_path / "true-ar.json"
    row = {"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}
    row.update(row_patch)
    _write_true_ar_baseline(baseline_path, [row], prompt_text_by_id={"code_1": "write code"})
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match=message):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_model_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        model="/models/gguf/other-model.gguf",
    )
    args = SimpleNamespace(
        model=TEST_MODEL,
        prompts=TEST_PROMPTS,
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="true AR model path mismatch"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_prompt_file_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        prompt_file="benchmarks/prompts/other.jsonl",
    )
    args = SimpleNamespace(
        model=TEST_MODEL,
        prompts=TEST_PROMPTS,
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="true AR prompt_file mismatch"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_prompt_count_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        prompt_count=2,
    )
    args = SimpleNamespace(
        model=TEST_MODEL,
        prompts=TEST_PROMPTS,
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="true AR prompt_count mismatch"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_without_command_provenance(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        include_commands=False,
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="true AR baseline artifact requires non-empty commands provenance"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_without_repo_provenance(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        include_repo=False,
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="true AR baseline artifact requires repo provenance metadata"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_prompt_hash_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0, "prompt_sha256": "0" * 64}],
        prompt_text_by_id={"code_1": "write code"},
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="true AR prompt hash mismatch"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_without_prompt_hashes(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        include_prompt_hashes=False,
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="requires prompt_hashes metadata"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_without_row_prompt_hash(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
        include_row_hashes=False,
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}

    with pytest.raises(BenchError, match="requires prompt_sha256"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_without_same_prompt_suite_flag(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_1", "category": "general_en", "output_tokens": 20, "decode_ms": 200.0},
        ],
        prompt_text_by_id={"code_1": "write code", "general_1": "explain"},
        same_prompt_suite=None,
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [
        {"id": "code_1", "category": "code", "prompt": "write code"},
        {"id": "general_1", "category": "general_en", "prompt": "explain"},
    ]
    raw = {
        1: [
            _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
            _row("general_1", "general_en", output=20, accepted=2, drafts=2, ar_ms=200.0, draft_ms=20.0),
        ]
    }

    with pytest.raises(BenchError, match="same_prompt_suite=true"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_category_summary_rejects_true_ar_baseline_with_missing_prompt(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
        ],
        prompt_text_by_id={"code_1": "write code", "general_1": "explain"},
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [
        {"id": "code_1", "category": "code", "prompt": "write code"},
        {"id": "general_1", "category": "general_en", "prompt": "explain"},
    ]
    raw = {
        1: [
            _row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
            _row("general_1", "general_en", output=20, accepted=2, drafts=2, ar_ms=200.0, draft_ms=20.0),
        ]
    }

    with pytest.raises(BenchError, match="must exactly match selected prompts"):
        build_summary(args=args, prompts=prompts, raw=raw, commands=[])


def test_true_ar_category_artifact_schema_matches_attachment_contract() -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        decode_tokens=32,
        warmup_decode_tokens=1,
    )
    prompts = [
        {"id": "code_1", "category": "code", "prompt": "write code"},
        {"id": "general_1", "category": "general_en", "prompt": "explain"},
    ]
    prompt_metrics = [
        {"id": "code_1", "category": "code", "output_tokens": 32, "decode_ms": 640.0},
        {"id": "general_1", "category": "general_en", "output_tokens": 32, "decode_ms": 320.0},
    ]

    artifact = build_true_ar_artifact(args=args, prompts=prompts, prompt_metrics=prompt_metrics, commands=["cmd"])

    assert artifact["kind"] == "hipengine_gguf_true_ar_category_baseline"
    assert artifact["repo"]["repo_root"]
    assert set(artifact["repo"]) == {"repo_root", "git_commit", "git_branch", "git_tracked_dirty", "git_untracked_count"}
    assert artifact["performance_claim"] is False
    assert artifact["true_autoregressive_path"] is True
    assert artifact["same_timing_protocol"] is True
    assert artifact["same_prompt_suite"] is True
    assert artifact["prompt_ids"] == ["code_1", "general_1"]
    assert artifact["prompt_hashes"] == {"code_1": prompt_sha256("write code"), "general_1": prompt_sha256("explain")}
    assert artifact["prompt_metrics"][0]["prompt_sha256"] == prompt_sha256("write code")
    assert artifact["prompt_metrics"][1]["prompt_sha256"] == prompt_sha256("explain")
    assert artifact["totals"]["decode_tok_s_weighted"] == pytest.approx(64 / 0.960)
    assert artifact["categories"]["code"]["decode_tok_s_weighted"] == pytest.approx(50.0)
    assert artifact["categories"]["general_en"]["decode_tok_s_weighted"] == pytest.approx(100.0)
    assert artifact["prompt_metrics"] is prompt_metrics


def test_true_ar_category_artifact_rejects_prompt_order_mismatch() -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        decode_tokens=32,
        warmup_decode_tokens=1,
    )
    prompts = [
        {"id": "code_1", "category": "code", "prompt": "write code"},
        {"id": "general_1", "category": "general_en", "prompt": "explain"},
    ]
    prompt_metrics = [
        {"id": "general_1", "category": "general_en", "output_tokens": 32, "decode_ms": 320.0},
        {"id": "code_1", "category": "code", "output_tokens": 32, "decode_ms": 640.0},
    ]

    with pytest.raises(BenchError, match="order/ids must match"):
        build_true_ar_artifact(args=args, prompts=prompts, prompt_metrics=prompt_metrics, commands=[])


def test_markdown_labels_verifier_off_as_diagnostic_not_plain_ar(tmp_path: Path) -> None:
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}
    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])

    markdown_path = tmp_path / "summary.md"
    write_markdown(summary, markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "Diagnostic only" in markdown
    assert "vs verifier off" in markdown
    assert "vs true AR" not in markdown
    assert "| vs AR |" not in markdown


def test_markdown_separates_true_ar_from_verifier_off(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [{"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0}],
        prompt_text_by_id={"code_1": "write code"},
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [{"id": "code_1", "category": "code", "prompt": "write code"}]
    raw = {1: [_row("code_1", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0)]}
    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])

    markdown_path = tmp_path / "summary.md"
    write_markdown(summary, markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "true no-MTP AR baseline attached" in markdown
    assert "not a retained speed claim" in markdown
    assert "vs verifier off | vs true AR" in markdown
    assert "| vs AR |" not in markdown


def test_true_ar_category_cli_dry_run_emits_attachable_schema(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"dry-run-placeholder")
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        json.dumps({"id": "code_1", "category": "code", "prompt": "write code"}) + "\n"
        + json.dumps({"id": "general_1", "category": "general_en", "prompt": "explain"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "true-ar-baseline.json"
    raw_root = tmp_path / "raw"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gguf_true_ar_category_bench.py"),
            "--model",
            str(model_path),
            "--prompts",
            str(prompts_path),
            "--decode-tokens",
            "4",
            "--raw-root",
            str(raw_root),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["kind"] == "hipengine_gguf_true_ar_category_baseline"
    assert artifact["status"] == "dry_run"
    assert artifact["performance_claim"] is False
    assert artifact["true_autoregressive_path"] is True
    assert artifact["same_timing_protocol"] is True
    assert artifact["same_prompt_suite"] is True
    assert artifact["prompt_ids"] == ["code_1", "general_1"]
    assert artifact["prompt_hashes"] == {"code_1": prompt_sha256("write code"), "general_1": prompt_sha256("explain")}
    assert [row["id"] for row in artifact["prompt_metrics"]] == ["code_1", "general_1"]
    assert all(row["output_tokens"] == 4 for row in artifact["prompt_metrics"])
    assert artifact["totals"]["total_output_tokens"] == 8
