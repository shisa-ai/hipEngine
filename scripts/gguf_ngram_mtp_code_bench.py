#!/usr/bin/env python3
"""Run the SPECDEC2 bridge on an independent repetition-heavy code suite.

This wrapper deliberately reuses ``specdec2_perf_bridge`` without changing its
true-AR, timing-ownership, lifecycle, provenance, or correctness contracts.  It
only binds a separate four-prompt code-copy fixture with a fixed 2/2
train/heldout split.  Results from this suite diagnose the workload class where
n-gram speculation is expected to help; they never replace the mandatory
four-category canonical and category-heldout gate.
"""

from __future__ import annotations

from pathlib import Path

from scripts import specdec2_perf_bridge as bridge

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_REPETITION_PROMPTS = (
    REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-repetition.jsonl"
)
CODE_REPETITION_PROMPT_IDS = (
    "code_copy_python",
    "code_copy_typescript",
    "code_copy_rust",
    "code_copy_sql",
)
CODE_REPETITION_HELDOUT_IDS = frozenset(
    {"code_copy_rust", "code_copy_sql"}
)


def configure_code_repetition_contract() -> None:
    """Bind the independent fixture before the shared parser/validator runs."""

    bridge.DEFAULT_PROMPTS = DEFAULT_CODE_REPETITION_PROMPTS
    bridge.FULL_PROMPT_IDS = CODE_REPETITION_PROMPT_IDS
    bridge._REQUIRED_CATEGORIES = frozenset({"code"})
    bridge._HELDOUT_IDS = CODE_REPETITION_HELDOUT_IDS


def main() -> int:
    configure_code_repetition_contract()
    return bridge.main()


if __name__ == "__main__":
    raise SystemExit(main())
