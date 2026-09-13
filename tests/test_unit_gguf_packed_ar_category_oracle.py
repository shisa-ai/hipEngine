from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gguf_mtp_category_bench import load_prompt_rows
from scripts.gguf_packed_ar_category_oracle import (
    DEFAULT_HELDOUTS,
    _group_prompt_rows,
    _repeat_determinism,
    _session_build_policy,
    _validate_prompt_contract,
    build_parser,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"


def test_category_oracle_groups_complete_suites_without_filling_tail() -> None:
    primary = load_prompt_rows(PRIMARY)
    heldouts = load_prompt_rows(DEFAULT_HELDOUTS)

    primary_groups = _group_prompt_rows(primary, max_group_size=4)
    heldout_groups = _group_prompt_rows(heldouts, max_group_size=4)

    assert [len(group) for group in primary_groups] == [4, 4, 2]
    assert [len(group) for group in heldout_groups] == [4, 4]
    assert [row["id"] for group in primary_groups for row in group] == [
        row["id"] for row in primary
    ]
    assert [row["id"] for group in heldout_groups for row in group] == [
        row["id"] for row in heldouts
    ]


def test_category_oracle_prompt_contract_covers_primary_and_heldout_categories() -> None:
    primary = load_prompt_rows(PRIMARY)
    heldouts = load_prompt_rows(DEFAULT_HELDOUTS)

    _validate_prompt_contract(primary, heldouts)

    broken = [dict(row) for row in heldouts]
    broken[0]["category"] = "code"
    broken = [row for row in broken if row["category"] != "mixed_ja_en"]
    with pytest.raises(ValueError, match="heldouts must cover"):
        _validate_prompt_contract(primary, broken)


def test_category_oracle_repeat_determinism_reports_prompt_ids() -> None:
    exact, mismatches = _repeat_determinism(
        {
            "a": [{"trajectory_sha256": "x"}, {"trajectory_sha256": "x"}],
            "b": [{"trajectory_sha256": "y"}, {"trajectory_sha256": "z"}],
        }
    )

    assert exact is False
    assert mismatches == ["b"]


def test_category_oracle_cached_build_policy_reads_compiler_file(tmp_path) -> None:
    compiler = tmp_path / "hipcc-version.txt"
    compiler.write_text("HIP version: test\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--compiler-version-file",
            str(compiler),
            "--require-cached-build",
            "--json",
            str(tmp_path / "out.json"),
        ]
    )

    assert _session_build_policy(args) == {
        "compiler_version": "HIP version: test",
        "require_cached_build": True,
    }


def test_category_oracle_defaults_to_complete_b4_contract(tmp_path) -> None:
    args = build_parser().parse_args(["--json", str(tmp_path / "out.json")])

    assert args.prompts == PRIMARY
    assert args.heldouts == DEFAULT_HELDOUTS
    assert args.backend == "hip_gfx1100"
    assert args.group_size == 4
    assert args.decode_steps == 24
    assert args.repeats == 3
