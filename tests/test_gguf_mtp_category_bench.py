from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.gguf_mtp_category_bench import (
    DEFAULT_HELDOUT_PROMPT_IDS,
    DEFAULT_PROMPTS,
    BenchError,
    build_split_contract,
    build_summary,
    load_prompt_rows,
    objective_metrics_for_budget,
    validate_speed_claim_contract,
    write_markdown,
)
from scripts.gguf_true_ar_category_bench import build_true_ar_artifact


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
    assert summary["totals"]["off"]["baseline_kind"] == "verifier_derived_from_b1_target_ar"
    assert summary["totals"]["off"]["true_autoregressive_path"] is False
    assert summary["categories"]["code"]["off"]["true_autoregressive_path"] is False


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
            accepted = 2 if is_heldout else 1
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
    assert contract["heldout_categories"] == ["code", "general_en", "general_ja", "mixed_ja_en"]


def test_speed_claim_contract_rejects_verifier_derived_ar_baseline() -> None:
    summary = {
        "speed_claim_eligible": True,
        "true_ar_baseline": {
            "available": False,
            "true_autoregressive_path": False,
            "same_prompt_suite": False,
            "same_timing_protocol": False,
            "source": None,
        },
    }

    with pytest.raises(BenchError, match="true no-MTP autoregressive baseline"):
        validate_speed_claim_contract(summary)


def test_speed_claim_contract_accepts_same_protocol_true_ar_baseline() -> None:
    summary = {
        "speed_claim_eligible": True,
        "true_ar_baseline": {
            "available": True,
            "true_autoregressive_path": True,
            "same_prompt_suite": True,
            "same_timing_protocol": True,
            "source": "future true AR harness artifact",
        },
    }

    assert validate_speed_claim_contract(summary) is summary


def _write_true_ar_baseline(path: Path, rows: list[dict], *, same_prompt_suite: bool | None = True) -> None:
    payload = {
        "kind": "hipengine_gguf_true_ar_category_baseline",
        "true_autoregressive_path": True,
        "same_timing_protocol": True,
        "prompt_metrics": rows,
    }
    if same_prompt_suite is not None:
        payload["same_prompt_suite"] = same_prompt_suite
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_category_summary_attaches_valid_true_ar_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_1", "category": "general_en", "output_tokens": 20, "decode_ms": 200.0},
        ],
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
            _row("general_1", "general_en", output=20, accepted=2, drafts=1, ar_ms=200.0, draft_ms=20.0),
        ]
    }

    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])

    assert summary["speed_claim_eligible"] is False
    assert summary["true_ar_comparison_available"] is True
    assert "not a retained speed claim" in summary["promotion_blocker"]
    assert summary["true_ar_baseline"]["available"] is True
    assert summary["true_ar_baseline"]["same_prompt_suite"] is True
    assert summary["true_ar_baseline"]["same_timing_protocol"] is True
    assert summary["true_ar_baseline"]["totals"]["decode_tok_s_weighted"] == 100.0
    assert summary["objective_metrics_available"] is False
    assert "heldout coverage" in summary["objective_metrics_blocker"]
    assert summary["objectives"] == {}
    assert summary["totals"]["b1"]["true_ar_decode_tok_s_weighted"] == 100.0
    assert summary["totals"]["b1"]["mtp_vs_true_ar_decode_ratio"] == pytest.approx((30.0 / 330.0 * 1000.0) / 100.0)
    assert summary["categories"]["code"]["b1"]["mtp_vs_true_ar_decode_ratio"] == pytest.approx((10.0 / 110.0 * 1000.0) / 100.0)


def test_objective_metrics_for_budget_requires_true_ar_and_reports_splits(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_merge_intervals", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "code_markdown_table", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_en_plan", "category": "general_en", "output_tokens": 20, "decode_ms": 200.0},
            {"id": "general_en_explain", "category": "general_en", "output_tokens": 20, "decode_ms": 200.0},
        ],
    )
    args = SimpleNamespace(
        model="/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        prompts="benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        cycles=1,
        raw_root="/tmp/raw",
        true_ar_baseline_json=baseline_path,
    )
    prompts = [
        {"id": "code_merge_intervals", "category": "code", "prompt": "write code"},
        {"id": "code_markdown_table", "category": "code", "prompt": "write table"},
        {"id": "general_en_plan", "category": "general_en", "prompt": "plan"},
        {"id": "general_en_explain", "category": "general_en", "prompt": "explain"},
    ]
    raw = {
        1: [
            _row("code_merge_intervals", "code", output=10, accepted=1, drafts=1, ar_ms=100.0, draft_ms=10.0),
            _row("code_markdown_table", "code", output=10, accepted=2, drafts=1, ar_ms=100.0, draft_ms=10.0),
            _row("general_en_plan", "general_en", output=20, accepted=3, drafts=1, ar_ms=200.0, draft_ms=20.0),
            _row("general_en_explain", "general_en", output=20, accepted=4, drafts=1, ar_ms=200.0, draft_ms=20.0),
        ]
    }
    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=[])

    metrics = objective_metrics_for_budget(summary, 1)

    assert summary["objective_metrics_available"] is True
    assert summary["objective_metrics_blocker"] is None
    assert summary["objectives"]["b1"] == metrics
    assert metrics["budget"] == "b1"
    assert metrics["true_ar_comparison_available"] is True
    assert metrics["speed_claim_eligible"] is False
    assert metrics["performance_claim"] is False
    assert metrics["full"]["accepted_per_output"] == pytest.approx(10 / 60)
    assert metrics["full"]["draft_acceptance"] == pytest.approx(10 / 4)
    assert metrics["full"]["mtp_vs_true_ar_decode_ratio"] == pytest.approx((60.0 / 660.0 * 1000.0) / 100.0)
    assert metrics["train"]["prompts"] == 2
    assert metrics["train"]["accepted_per_output"] == pytest.approx(4 / 30)
    assert metrics["heldout"]["prompts"] == 2
    assert metrics["heldout"]["accepted_per_output"] == pytest.approx(6 / 30)
    assert metrics["heldout_ids"] == ["code_markdown_table", "general_en_explain"]


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


def test_category_summary_rejects_true_ar_baseline_without_same_prompt_suite_flag(tmp_path: Path) -> None:
    baseline_path = tmp_path / "true-ar.json"
    _write_true_ar_baseline(
        baseline_path,
        [
            {"id": "code_1", "category": "code", "output_tokens": 10, "decode_ms": 100.0},
            {"id": "general_1", "category": "general_en", "output_tokens": 20, "decode_ms": 200.0},
        ],
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
            _row("general_1", "general_en", output=20, accepted=2, drafts=1, ar_ms=200.0, draft_ms=20.0),
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
            _row("general_1", "general_en", output=20, accepted=2, drafts=1, ar_ms=200.0, draft_ms=20.0),
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
    assert artifact["performance_claim"] is False
    assert artifact["true_autoregressive_path"] is True
    assert artifact["same_timing_protocol"] is True
    assert artifact["same_prompt_suite"] is True
    assert artifact["prompt_ids"] == ["code_1", "general_1"]
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
    assert [row["id"] for row in artifact["prompt_metrics"]] == ["code_1", "general_1"]
    assert all(row["output_tokens"] == 4 for row in artifact["prompt_metrics"])
    assert artifact["totals"]["total_output_tokens"] == 8
