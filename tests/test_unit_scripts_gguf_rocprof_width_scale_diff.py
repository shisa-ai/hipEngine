"""RED test for the wave-width kernel-scale diff.

The question the tool answers is #14's open one: decode per-step cost rises from 33.2 ms at
width 1 to 50.4 ms at width 2, where a weight-bound batched step should stay near 33 ms, so
some decode work is per-row. Naming the culprit needs two kernel traces compared per kernel,
because the two signatures look different in the data:

  * per-row launches - the launch count scales with rows (one launch per row);
  * per-row work inside one launch - the launch count is flat but each launch gets longer.

Synthetic traces make both signatures and the refusals checkable without a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gguf_rocprof_width_scale_diff.py"

HEADER = "Kind,Kernel_Name,Grid/X,Grid/Y,Grid/Z,Start_Timestamp,End_Timestamp\n"


def _write_trace(root: pathlib.Path, kernels: dict[str, tuple[int, int]]) -> None:
    """One CSV per trace: `count` launches of `duration_ns` each, grid 16x1."""
    root.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    clock = 1_000_000
    for name, (count, duration_ns) in kernels.items():
        for _ in range(count):
            lines.append(f"5,{name},16,1,1,{clock},{clock + duration_ns}\n")
            clock += duration_ns + 500
    (root / "results.csv").write_text("".join(lines))


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("gguf_rocprof_width_scale_diff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def traces(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    base = tmp_path / "c1"
    cand = tmp_path / "c2"
    _write_trace(
        base,
        {
            "row_gemv_like": (10, 100_000),   # doubles launches at width 2
            "tile_like": (10, 100_000),       # same launches, doubles duration at width 2
            "flat_like": (10, 50_000),        # unchanged
            "base_only_like": (10, 20_000),   # absent in candidate
        },
    )
    _write_trace(
        cand,
        {
            "row_gemv_like": (20, 100_000),
            "tile_like": (10, 200_000),
            "flat_like": (10, 50_000),
            "cand_only_like": (5, 10_000),
        },
    )
    return base, cand


def _report(tool, base: pathlib.Path, cand: pathlib.Path, **kw) -> dict:
    return tool.diff_dirs(base, cand, rows_base=1, rows_candidate=2, **kw)


def test_per_row_launches_are_flagged_by_launch_count(tool, traces) -> None:
    base, cand = traces
    entry = _report(tool, base, cand)["kernels"]["row_gemv_like"]
    assert entry["classification"] == "per_row_launches", entry
    assert entry["launches_base"] == 10 and entry["launches_candidate"] == 20


def test_per_row_work_inside_a_launch_is_flagged_by_mean_duration(tool, traces) -> None:
    base, cand = traces
    entry = _report(tool, base, cand)["kernels"]["tile_like"]
    assert entry["classification"] == "per_row_inside_launch", entry
    assert entry["launches_candidate"] == entry["launches_base"]
    assert entry["mean_ratio"] == pytest.approx(2.0, rel=0.01)


def test_width_independent_kernels_classify_flat(tool, traces) -> None:
    base, cand = traces
    entry = _report(tool, base, cand)["kernels"]["flat_like"]
    assert entry["classification"] == "flat", entry


def test_launches_present_in_only_one_arm_are_reported_not_silently_dropped(
    tool, traces
) -> None:
    base, cand = traces
    report = _report(tool, base, cand)
    assert report["only_in_base"] == ["base_only_like"], report["only_in_base"]
    assert report["only_in_candidate"] == ["cand_only_like"], report["only_in_candidate"]


def test_expected_scale_is_recorded_so_the_verdict_is_auditable(tool, traces) -> None:
    base, cand = traces
    report = _report(tool, base, cand)
    assert report["rows_base"] == 1 and report["rows_candidate"] == 2
    assert report["expected_scale"] == pytest.approx(2.0)


def test_widths_that_cannot_produce_a_ratio_are_refused(tool, traces) -> None:
    base, cand = traces
    with pytest.raises(ValueError):
        tool.diff_dirs(base, cand, rows_base=0, rows_candidate=2)
    with pytest.raises(ValueError):
        tool.diff_dirs(base, cand, rows_base=2, rows_candidate=1)


def test_missing_or_empty_trace_dirs_refuse_instead_of_reporting_nothing(
    tool, traces, tmp_path
) -> None:
    base, _cand = traces
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        tool.diff_dirs(base, empty, rows_base=1, rows_candidate=2)
    with pytest.raises(FileNotFoundError):
        tool.diff_dirs(tmp_path / "nope", base, rows_base=1, rows_candidate=2)


def test_json_output_round_trips(tmp_path: pathlib.Path, tool, traces) -> None:
    base, cand = traces
    out = tmp_path / "diff.json"
    rc = tool.main(["--base-dir", str(base), "--candidate-dir", str(cand), "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["kernels"]["row_gemv_like"]["classification"] == "per_row_launches"
