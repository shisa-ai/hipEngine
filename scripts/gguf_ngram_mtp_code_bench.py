#!/usr/bin/env python3
"""Run the SPECDEC2 bridge on an independent repetition-heavy code suite.

This wrapper deliberately reuses ``specdec2_perf_bridge`` without changing its
true-AR, timing-ownership, lifecycle, provenance, or correctness contracts.  It
only binds a separate four-prompt code-copy fixture with a fixed 2/2
train/heldout split.  Results from this suite diagnose the workload class where
n-gram speculation is expected to help; they never replace the mandatory
four-category canonical and category-heldout gate.

The four shared bridge globals are rebound through a scoped context manager so
the binding cannot outlive one run.  An unscoped rebinding leaks the
four-prompt suite into any later in-process consumer of the canonical contract
(worklog 20260830T193959 recorded eight order-dependent bridge failures from
exactly that leak inside one pytest process).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
_CONTRACT_GLOBALS = (
    "DEFAULT_PROMPTS",
    "FULL_PROMPT_IDS",
    "_REQUIRED_CATEGORIES",
    "_HELDOUT_IDS",
)


def configure_code_repetition_contract() -> None:
    """Bind the independent fixture before the shared parser/validator runs.

    This rebinding is process-wide and permanent. Callers that do not own the
    process must use :func:`code_repetition_contract` instead.
    """

    bridge.DEFAULT_PROMPTS = DEFAULT_CODE_REPETITION_PROMPTS
    bridge.FULL_PROMPT_IDS = CODE_REPETITION_PROMPT_IDS
    bridge._REQUIRED_CATEGORIES = frozenset({"code"})
    bridge._HELDOUT_IDS = CODE_REPETITION_HELDOUT_IDS


@contextmanager
def code_repetition_contract() -> Iterator[None]:
    """Bind the independent fixture and restore the canonical contract after."""

    bound = {name: getattr(bridge, name) for name in _CONTRACT_GLOBALS}
    configure_code_repetition_contract()
    try:
        yield
    finally:
        for name, value in bound.items():
            setattr(bridge, name, value)


def main() -> int:
    with code_repetition_contract():
        return bridge.main()


if __name__ == "__main__":
    raise SystemExit(main())
