"""Host contracts for the Surya protocol oracle case table and its capture CLI.

R0493 needs ``a4`` in the protocol case table, and its documented clearing
command must parse and select only that page on a host without torch or
transformers. The capture itself needs the ``datalab-to/surya-ocr-2``
checkpoint, so these tests fake the model and the greedy decode: what they pin
is the case table, the argument surface, and that a single-page capture cannot
erase the committed rows.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT
from scripts.surya_bench_pages import expected_lines

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "surya_oracle_greedy", _ROOT / "scripts" / "surya_oracle_greedy.py"
)
assert _SPEC is not None and _SPEC.loader is not None
G = importlib.util.module_from_spec(_SPEC)
# register before exec so the module is resolvable from sys.modules
sys.modules[_SPEC.name] = G
_SPEC.loader.exec_module(G)

FIXTURES = _ROOT / "tests" / "fixtures" / "surya"
PROTOCOL_ORACLE = FIXTURES / "oracle_fullpage_protocol.json"

# The command docs/REFACTOR.md R0493 names as the clearing command.
CLEARING_COMMAND = (
    "--case", "protocol",
    "--only", "a4",
    "--device", "cpu",
    "--page-dir", "tests/fixtures/surya",
    "--out-dir", "/tmp/surya-a4-oracle",
)

# Pages whose ids are driven by drawn text, so they measure an ids-per-character
# rate. ``table`` is markup-dominated (448 ids for 18 drawn characters) and
# ``blank`` has no text at all; both enter the estimate as fixed costs instead.
TEXT_PAGES = ("ja", "mixed", "dense", "scan", "long")
TABLE_PAGE = "table"
# A derived floor is an estimate, not a measurement, so the budget must clear it
# with room for the layout the page actually produces.
MIN_HEADROOM = 1.2


def _captured() -> dict:
    return json.loads(PROTOCOL_ORACLE.read_text())


def _drawn_chars(page: str) -> int:
    return sum(len(unit) for unit in expected_lines(page))


# --- case table -------------------------------------------------------------


def test_the_protocol_table_has_the_page_scale_case() -> None:
    """A4 is the page-scale transcription case and had no table entry."""

    assert "a4" in G.PROTOCOL_MAX_TOKENS
    assert G.PROTOCOL_MAX_TOKENS["a4"] > 0
    assert G.page_filename("a4") == "page_a4.png"


def test_a4_budget_covers_the_worst_observed_cost_per_drawn_character() -> None:
    """The A4 budget must come from the page's content, not from truncation.

    A budget that cuts the transcription gates only a prefix, which is the
    failure the protocol budgets exist to avoid. A4 has no capture yet, so its
    floor is arithmetic on what the committed rows measured: the worst
    ids-per-drawn-character rate among the text pages, applied to the
    characters drawn on A4, plus the captured cost of the same ruled 7x4 table
    the A4 page draws.
    """

    captured = _captured()
    per_char = max(
        len(captured[page]["ids"]) / _drawn_chars(page) for page in TEXT_PAGES
    )
    table_ids = len(captured[TABLE_PAGE]["ids"])
    floor = _drawn_chars("a4") * per_char + table_ids

    budget = G.PROTOCOL_MAX_TOKENS["a4"]
    assert budget >= MIN_HEADROOM * floor, (
        f"a4 budget {budget} leaves no headroom over the derived floor "
        f"{floor:.0f} ({per_char:.2f} ids/char x {_drawn_chars('a4')} chars "
        f"+ {table_ids} table ids)"
    )


# --- argument surface -------------------------------------------------------


def test_the_r0493_clearing_command_selects_only_a4() -> None:
    args, only = G.parse_args(list(CLEARING_COMMAND))

    assert args.case == "protocol"
    assert only == ["a4"]
    assert args.device == "cpu"
    assert args.page_dir == Path("tests/fixtures/surya")
    assert args.out_dir == Path("/tmp/surya-a4-oracle")


def test_parsing_the_clearing_command_needs_neither_torch_nor_transformers() -> None:
    """The parser is a host contract; only the capture needs the checkpoint.

    Run in a subprocess because another test in the same session may already
    have imported torch: the question is what *this* import pulls in.
    """

    code = (
        "import sys;"
        f"sys.path.insert(0, {str(_ROOT)!r});"
        "import scripts.surya_oracle_greedy as g;"
        f"args, only = g.parse_args({list(CLEARING_COMMAND)!r});"
        "assert only == ['a4'] and args.case == 'protocol';"
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False False", done.stdout


def test_documented_script_help_works_without_pythonpath(monkeypatch) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [sys.executable, "scripts/surya_oracle_greedy.py", "--help"],
        cwd=_ROOT, env=env, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "protocol" in done.stdout


def test_an_unknown_protocol_page_is_rejected_before_the_model_loads() -> None:
    with pytest.raises(SystemExit, match="unknown protocol page"):
        G.parse_args(["--case", "protocol", "--only", "a9"])


def test_only_is_rejected_for_a_case_that_captures_every_page() -> None:
    """A selector that would be ignored must not be accepted silently."""

    with pytest.raises(SystemExit, match="--only applies to"):
        G.parse_args(["--case", "image", "--only", "a4"])


@pytest.mark.parametrize("selection", ["", " ", ",", " , "])
def test_empty_explicit_selection_cannot_expand_to_all_pages(selection) -> None:
    with pytest.raises(SystemExit, match="--only must name at least one page"):
        G.parse_args(["--case", "protocol", "--only", selection])


def test_the_default_selection_is_unchanged() -> None:
    args, only = G.parse_args([])

    assert args.case == "all"
    assert args.device == "cpu"
    assert only is None


# --- capture writes ---------------------------------------------------------


def _fake_greedy(monkeypatch, ids: tuple[int, ...] = (11, 22, 33)) -> None:
    """Stand in for the torch prefill+decode; the page files are real."""

    def fake(model, processor, page_path, *, device, max_tokens, prompt=None):
        assert prompt == FULL_PAGE_HTML_PROMPT
        assert Path(page_path).exists(), page_path
        caps = {"image_grid_thw": np.asarray([[1, 220, 156]])}
        return caps, {}, list(ids), "<div>page</div>"

    monkeypatch.setattr(G, "_greedy_from_page", fake)


def test_a_single_page_capture_merges_into_an_existing_oracle(
    tmp_path: Path, monkeypatch
) -> None:
    """``--only a4`` must not replace a complete oracle with one row.

    This is the R0493 hazard: the clearing command captures one page, and
    writing that subset on its own would delete every committed row.
    """

    original = PROTOCOL_ORACLE.read_text()
    oracle = tmp_path / "oracle_fullpage_protocol.json"
    oracle.write_text(original)
    _fake_greedy(monkeypatch)

    G._run_protocol(None, None, tmp_path, "cpu", ["a4"], FIXTURES)

    merged = json.loads(oracle.read_text())
    before = json.loads(original)
    assert list(merged) == [*before, "a4"]
    for name, row in before.items():
        assert merged[name] == row, f"capturing a4 changed the {name} row"
    assert merged["a4"]["page"] == "page_a4.png"
    assert merged["a4"]["prompt"] == FULL_PAGE_HTML_PROMPT
    assert merged["a4"]["max_tokens"] == G.PROTOCOL_MAX_TOKENS["a4"]
    assert merged["a4"]["reached_limit"] is False
    assert merged["a4"]["finish_reason"] == "eos"
    # Rows that were not captured keep their exact serialization, so the
    # committed fixture's bytes are a prefix of the merged file.
    assert oracle.read_text().startswith(original[:-2]), (
        "merging reformatted rows it did not capture"
    )


def test_a_scratch_capture_writes_only_the_requested_page(
    tmp_path: Path, monkeypatch
) -> None:
    """The documented ``--out-dir /tmp/...`` capture is a one-row table."""

    committed = PROTOCOL_ORACLE.read_bytes()
    _fake_greedy(monkeypatch)

    G._run_protocol(None, None, tmp_path, "cpu", ["a4"], FIXTURES)

    written = json.loads((tmp_path / "oracle_fullpage_protocol.json").read_text())
    assert list(written) == ["a4"]
    assert PROTOCOL_ORACLE.read_bytes() == committed, (
        "the capture wrote through --page-dir into the committed fixture"
    )


def test_a_full_capture_writes_exactly_the_case_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Merging is for a subset; a full run replaces a stale table."""

    oracle = tmp_path / "oracle_fullpage_protocol.json"
    oracle.write_text(json.dumps({"ghost": {"ids": [1]}}, indent=1))
    _fake_greedy(monkeypatch)

    G._run_protocol(None, None, tmp_path, "cpu", None, FIXTURES)

    written = json.loads(oracle.read_text())
    assert list(written) == list(G.PROTOCOL_MAX_TOKENS)
    assert "ghost" not in written
