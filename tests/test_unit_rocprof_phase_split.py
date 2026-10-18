"""Unit tests for scripts/rocprof_phase_split.py.

The splitter exists because two worklog entries on this branch reported a
"prefill" counter figure that had summed the decode phase, and the mistake was
available because nothing in the toolchain separated the phases. These tests pin
the behaviour that makes it unavailable: the boundary is the first marker
dispatch, dispatches are assigned by id rather than by file order, and both
CSV layouts rocprofv3 writes are found.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "rocprof_phase_split", REPO_ROOT / "scripts" / "rocprof_phase_split.py"
)
assert _SPEC is not None and _SPEC.loader is not None
phase_split = importlib.util.module_from_spec(_SPEC)
sys.modules["rocprof_phase_split"] = phase_split
_SPEC.loader.exec_module(phase_split)

COLUMNS = [
    "Correlation_Id", "Dispatch_Id", "Agent_Id", "Queue_Id", "Process_Id",
    "Thread_Id", "Grid_Size", "Kernel_Id", "Kernel_Name", "Workgroup_Size",
    "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count",
    "SGPR_Count", "Counter_Name", "Counter_Value", "Start_Timestamp",
    "End_Timestamp",
]

PREFILL = "void (anonymous namespace)::gemma4_attention_prefill_kernel<unsigned short>(...)"
DECODE = "void (anonymous namespace)::gemma4_attention_decode_class_kernel<unsigned short>(...)"


def _write(path: Path, rows: list[tuple[int, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for dispatch, kernel, value in rows:
            writer.writerow([dispatch, dispatch, "Agent 1", 2, 1, 1, 1024, 1,
                             kernel, 128, 0, 0, 40, 0, 128, "FETCH_SIZE", value,
                             0, 0])


def test_kernel_name_strips_namespace_and_template() -> None:
    assert phase_split.kernel_name(PREFILL) == "gemma4_attention_prefill_kernel"
    assert phase_split.kernel_name(DECODE) == "gemma4_attention_decode_class_kernel"


def test_kernel_name_falls_back_without_a_kernel_suffix() -> None:
    assert phase_split.kernel_name("__amd_rocclr_copyBuffer") == "__amd_rocclr_copyBuffer"


def test_load_rows_finds_the_flat_layout(tmp_path: Path) -> None:
    _write(tmp_path / "f_counter_collection.csv", [(1, PREFILL, 10.0)])
    assert phase_split.load_rows(tmp_path, "f") == [(1, "gemma4_attention_prefill_kernel", 10.0)]


def test_load_rows_finds_the_pass_layout(tmp_path: Path) -> None:
    _write(tmp_path / "pass_1" / "f_counter_collection.csv", [(1, PREFILL, 10.0)])
    _write(tmp_path / "pass_2" / "f_counter_collection.csv", [(2, PREFILL, 5.0)])
    rows = phase_split.load_rows(tmp_path, "f")
    assert sorted(rows) == [
        (1, "gemma4_attention_prefill_kernel", 10.0),
        (2, "gemma4_attention_prefill_kernel", 5.0),
    ]


def test_load_rows_rejects_a_directory_with_no_counters(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        phase_split.load_rows(tmp_path, "f")


def test_split_puts_the_marker_dispatch_in_phase_two(tmp_path: Path, capsys) -> None:
    """The boundary is the marker's own first dispatch, not the one before it."""
    _write(tmp_path / "f_counter_collection.csv", [
        (1, PREFILL, 100.0),
        (2, PREFILL, 200.0),
        (3, DECODE, 1_000_000.0),
        (4, DECODE, 2_000_000.0),
    ])
    argv = sys.argv
    sys.argv = ["rocprof_phase_split.py", str(tmp_path), "-o", "f",
                "--label-1", "PREFILL", "--label-2", "DECODE", "--top", "5"]
    try:
        assert phase_split.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "PREFILL: 300" in out, out
    assert "DECODE: 3,000,000" in out, out
    assert "DECODE begins at dispatch 3" in out, out


def test_missing_marker_reports_the_kernels_present(tmp_path: Path, capsys) -> None:
    """An unmarked run must say why rather than silently returning one phase."""
    _write(tmp_path / "f_counter_collection.csv", [(1, PREFILL, 100.0)])
    argv = sys.argv
    sys.argv = ["rocprof_phase_split.py", str(tmp_path), "-o", "f"]
    try:
        assert phase_split.main() == 2
    finally:
        sys.argv = argv
    err = capsys.readouterr().err
    assert "never dispatched" in err
    assert "gemma4_attention_prefill_kernel" in err
