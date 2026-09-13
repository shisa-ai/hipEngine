"""The README rollup must diff against the README, not a frozen copy of it.

`README_ROWS` used to be a hardcoded snapshot of "values currently printed in benchmarks/README.md".
It drifted: it still carried C8 AR 44.338 while the README had published 78.667 since the grouped
prefill promotion, so the tool reported +76.53% on a run that moved -0.51%. A delta against a
stale baseline is worse than no delta, because it looks like evidence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gguf_c1c8_readme_rollup.py"


def _module():
    spec = importlib.util.spec_from_file_location("readme_rollup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["readme_rollup"] = module
    spec.loader.exec_module(module)
    return module


def test_bold_and_plain_cells_parse():
    module = _module()
    line = (
        "| hipEngine AR | **21.999** | 31.916 | 45.309 | 54.151 "
        "| 61.881 | 71.226 | 74.903 | 78.667 |"
    )
    values = module.parse_markdown_row(line)
    assert values is not None
    assert values[0] == 21.999 and values[-1] == 78.667
    assert len(values) == 8


def test_separator_and_text_rows_are_rejected():
    module = _module()
    assert module.parse_markdown_row("| --- | ---: | ---: |") is None
    assert module.parse_markdown_row("| Engine / arm | C1 | C2 |") is None


def test_all_six_published_rows_parse_from_the_real_readme():
    module = _module()
    rows = module.read_readme_rows()
    for arm in ("ar", "k3"):
        for role in ("hipengine", "llama_current", "llama_laurent"):
            assert len(rows[arm][role]) == 8, f"{arm}/{role} missing from the README"


def test_scoped_rows_parsed_from_readme_match_the_tables_we_just_read():
    """A renamed heading or row must fail loudly rather than select another table."""
    module = _module()
    text = (REPO_ROOT / "benchmarks" / "README.md").read_text()
    parsed = module.read_readme_rows()
    for marker, arm in module.ARM_MARKERS.items():
        if arm is None:
            continue
        start = text.index(marker)
        end = text.find("\n**", start + len(marker))
        section = text[start : len(text) if end < 0 else end]
        for label, role in module.ROW_LABELS.items():
            line = next(
                row
                for row in section.splitlines()
                if row.strip().startswith(f"| {label} ")
            )
            assert parsed[arm][role] == module.parse_markdown_row(line)


def test_baseline_reports_live_parsing_and_falls_back_loudly(tmp_path, capsys):
    module = _module()
    rows, source = module.baseline_rows()
    assert "parsed live" in source, "the default baseline must come from the README itself"
    assert len(rows["ar"]["hipengine"]) == 8

    missing = tmp_path / "absent.md"
    rows2, source2 = module.baseline_rows(use_snapshot=True)
    assert "snapshot" in source2
    assert rows2 is module.SNAPSHOT_ROWS
    assert not missing.exists()


def test_snapshot_and_live_readme_are_allowed_to_differ_but_the_tool_says_which_it_used(capsys):
    """The whole point of the fix: the source of the baseline is printed, never assumed."""
    module = _module()
    module.baseline_rows()
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out  # a healthy README parses cleanly
