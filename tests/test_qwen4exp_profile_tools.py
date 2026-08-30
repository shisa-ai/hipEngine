from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_trace(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "sample_marker_api_trace.csv",
        ["Function", "Start_Timestamp", "End_Timestamp", "Thread_Id"],
        [
            {"Function": "qwen4exp_prefill_p508_0", "Start_Timestamp": 100, "End_Timestamp": 500, "Thread_Id": 1},
            {"Function": "qwen4exp_role:moe:layers.2.expert_gate", "Start_Timestamp": 150, "End_Timestamp": 300, "Thread_Id": 1},
        ],
    )
    _write_csv(
        tmp_path / "sample_hip_api_trace.csv",
        ["Function", "Start_Timestamp", "End_Timestamp", "Thread_Id", "Correlation_Id"],
        [
            {
                "Function": "hipLaunchKernel",
                "Start_Timestamp": 160,
                "End_Timestamp": 170,
                "Thread_Id": 1,
                "Correlation_Id": 7,
            }
        ],
    )
    _write_csv(
        tmp_path / "sample_kernel_trace.csv",
        ["Kernel_Name", "Start_Timestamp", "End_Timestamp", "Correlation_Id"],
        [
            {
                "Kernel_Name": "q4_k_selected_test_kernel",
                "Start_Timestamp": 180,
                "End_Timestamp": 260,
                "Correlation_Id": 7,
            },
            {
                "Kernel_Name": "__amd_rocclr_copyBuffer",
                "Start_Timestamp": 320,
                "End_Timestamp": 340,
                "Correlation_Id": 8,
            },
        ],
    )


def test_qwen4exp_trace_analyze_synthetic_window(tmp_path: Path) -> None:
    module = _load_script("qwen4exp_trace_analyze.py")
    _synthetic_trace(tmp_path)
    result = module.summarize(
        argparse.Namespace(
            trace_dir=tmp_path,
            engine="hipengine",
            marker_prefix="qwen4exp_prefill_p508_",
            output=tmp_path / "summary.json",
        )
    )
    assert result["kernel"]["rows"] == 2
    assert result["kernel"]["sum_ms"] == 0.0001
    assert result["hip_api"]["direct_launch_correlations"] == 1
    families = {row["name"]: row for row in result["kernel"]["families"]}
    assert families["moe_gate_up_q4"]["rows"] == 1
    assert families["copy_fill_kernel"]["rows"] == 1


def test_qwen4exp_role_analyze_synthetic_window(tmp_path: Path) -> None:
    module = _load_script("qwen4exp_role_analyze.py")
    _synthetic_trace(tmp_path)
    result = module.analyze(tmp_path, "qwen4exp_prefill_p508_")
    assert result["kernel_rows"] == 2
    assert result["attributed_rows"] == 1
    roles = {row["name"]: row for row in result["roles"]}
    assert roles["moe:layers.*.expert_gate"]["rows"] == 1
    assert roles["unattributed"]["rows"] == 1


def test_qwen4exp_p508_prompt_fixture_is_pinned() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "prompts"
        / "qwen4exp-p508.txt"
    )
    data = fixture.read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "9cf9d353b81b6ce1df61405b590f037b0502b52c7f6c0c19a543c33cbcb6dbb4"
    )
    assert data.startswith(b"<|im_start|>user\n")
    assert len(data) == 2613


def test_qwen4exp_perf_gap_report_renders_committed_artifact() -> None:
    module = _load_script("qwen4exp_perf_gap_report.py")
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "results"
        / "2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json"
    )
    artifact = json.loads(artifact_path.read_text())
    report = module.render_report(artifact)
    assert "| p508 prefill tok/s | 84.83 | 272.83 | 331.03 | 3.22x | 3.90x |" in report
    assert "| Layer-2 Q5_K gate/up | 301.46 | 15.38 | 19.61x |" in report
    assert "1195 direct launches" in report
