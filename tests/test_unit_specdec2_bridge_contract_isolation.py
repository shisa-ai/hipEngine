"""Contract-binding isolation tests for the SPECDEC2 bridge wrappers.

`scripts/gguf_ngram_mtp_code_bench.py` rebinds four shared
``scripts.specdec2_perf_bridge`` module globals to point at its own four-prompt
code-repetition fixture. That is correct inside its own single-purpose process,
but an unscoped rebinding leaks into every later test in the same pytest
process: `validate_bridge_artifact` then compares canonical artifacts against
the four-prompt suite and fails with "full bridge has an incomplete canonical
prompt suite". Worklog 20260830T193959 recorded eight such failures under broad
ordering that did not reproduce when the bridge file ran alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import gguf_ngram_mtp_code_bench as code_bench
from scripts import specdec2_perf_bridge as bridge

_CONTRACT_GLOBALS = (
    "DEFAULT_PROMPTS",
    "FULL_PROMPT_IDS",
    "_REQUIRED_CATEGORIES",
    "_HELDOUT_IDS",
)
_CANONICAL_REQUIRED_CATEGORIES = frozenset(
    {"code", "general_en", "general_ja", "mixed_ja_en"}
)


def _bound_contract() -> dict[str, object]:
    return {name: getattr(bridge, name) for name in _CONTRACT_GLOBALS}


def _code_repetition_contract() -> dict[str, object]:
    return {
        "DEFAULT_PROMPTS": code_bench.DEFAULT_CODE_REPETITION_PROMPTS,
        "FULL_PROMPT_IDS": code_bench.CODE_REPETITION_PROMPT_IDS,
        "_REQUIRED_CATEGORIES": frozenset({"code"}),
        "_HELDOUT_IDS": code_bench.CODE_REPETITION_HELDOUT_IDS,
    }


def test_canonical_bridge_contract_is_the_canonical_suite() -> None:
    bound = _bound_contract()

    assert Path(str(bound["DEFAULT_PROMPTS"])).name == "mtpbench-code-general-ja.jsonl"
    assert len(tuple(bound["FULL_PROMPT_IDS"])) == 10
    assert bound["_REQUIRED_CATEGORIES"] == _CANONICAL_REQUIRED_CATEGORIES
    assert len(bound["_HELDOUT_IDS"]) == 4


def test_code_repetition_contract_restores_the_shared_bridge_globals() -> None:
    before = _bound_contract()

    with code_bench.code_repetition_contract():
        assert _bound_contract() == _code_repetition_contract()
        rows = bridge.load_prompt_suite(code_bench.DEFAULT_CODE_REPETITION_PROMPTS)
        assert tuple(row["id"] for row in rows) == code_bench.CODE_REPETITION_PROMPT_IDS

    assert _bound_contract() == before


def test_exception_inside_the_contract_still_restores() -> None:
    before = _bound_contract()

    with pytest.raises(RuntimeError, match="boom"):
        with code_bench.code_repetition_contract():
            raise RuntimeError("boom")

    assert _bound_contract() == before


def test_nested_code_repetition_contracts_restore_last_in_first_out() -> None:
    before = _bound_contract()

    with code_bench.code_repetition_contract():
        with code_bench.code_repetition_contract():
            assert _bound_contract() == _code_repetition_contract()
        assert _bound_contract() == _code_repetition_contract()

    assert _bound_contract() == before


def test_wrapper_main_binds_the_fixture_through_the_scoped_contract() -> None:
    source = Path(code_bench.__file__).read_text(encoding="utf-8")

    assert "with code_repetition_contract():" in source
