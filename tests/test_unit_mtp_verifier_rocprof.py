from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from scripts.mtp_verifier_rocprof import (
    _filter_trace_rows_by_windows,
    _rocprof_command,
    _smoke_command,
    _summarize_api_rows,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model=Path("/models/test"),
        prompt_tokens="1,2,3",
        decode_tokens=8,
        candidate_budget=1,
        backend="hip_gfx1100",
        chain_attn_mode="c1_loop",
        graph_mode="auto",
        rocprof_warmup_cycles=0,
        rocprof_verify_cycles=0,
        rocprofv3="rocprofv3",
        raw_root=tmp_path,
        region="cycle",
    )


def test_profiler_requires_cached_child_and_collects_hip_runtime_trace(tmp_path: Path) -> None:
    args = _args(tmp_path)
    smoke = _smoke_command(args, tmp_path / "smoke.json")
    command = _rocprof_command(args, smoke)

    assert "--require-cached-build" in smoke
    assert "--kernel-trace" in command
    assert "--hip-runtime-trace" in command
    assert "--marker-trace" in command


def test_hip_api_summary_uses_only_fully_contained_marker_rows() -> None:
    rows = [
        {"function": "hipGraphLaunch", "start_ns": 110, "end_ns": 120, "duration_ns": 10},
        {"function": "hipStreamSynchronize", "start_ns": 130, "end_ns": 150, "duration_ns": 20},
        {"function": "hipMemcpy", "start_ns": 90, "end_ns": 115, "duration_ns": 25},
        {"function": "hipGraphLaunch", "start_ns": 210, "end_ns": 220, "duration_ns": 10},
    ]

    selected = _filter_trace_rows_by_windows(rows, [(100, 200)])
    summary = _summarize_api_rows(selected, verifier_passes=2, top=10)

    assert [row["function"] for row in selected] == ["hipGraphLaunch", "hipStreamSynchronize"]
    assert summary["calls"] == 2
    assert summary["total_ns"] == 30
    assert summary["calls_per_pass"] == 1.0
    by_name = {row["function"]: row for row in summary["functions"]}
    assert by_name["hipGraphLaunch"]["calls_per_pass"] == 0.5
    assert by_name["hipStreamSynchronize"]["ms_per_pass"] == 0.00001
