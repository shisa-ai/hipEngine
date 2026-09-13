from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_role_analyze.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_role_analyze", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_reports_kernel_and_api_breakdown_for_exact_role(tmp_path: Path) -> None:
    module = _load_script()
    role = "moe:layers.2.expert_gate"
    _write_csv(
        tmp_path / "trace_marker_api_trace.csv",
        ["Function", "Start_Timestamp", "End_Timestamp", "Thread_Id"],
        [
            {
                "Function": "qwen4exp_prefill_p508_0",
                "Start_Timestamp": 0,
                "End_Timestamp": 1000,
                "Thread_Id": 1,
            },
            {
                "Function": f"qwen4exp_role:{role}",
                "Start_Timestamp": 100,
                "End_Timestamp": 900,
                "Thread_Id": 1,
            },
        ],
    )
    _write_csv(
        tmp_path / "trace_hip_api_trace.csv",
        ["Function", "Start_Timestamp", "End_Timestamp", "Correlation_Id", "Thread_Id"],
        [
            {
                "Function": "hipLaunchKernel",
                "Start_Timestamp": 200,
                "End_Timestamp": 210,
                "Correlation_Id": 7,
                "Thread_Id": 1,
            },
            {
                "Function": "hipMemcpy",
                "Start_Timestamp": 300,
                "End_Timestamp": 310,
                "Correlation_Id": 8,
                "Thread_Id": 1,
            },
        ],
    )
    _write_csv(
        tmp_path / "trace_kernel_trace.csv",
        ["Kernel_Name", "Start_Timestamp", "End_Timestamp", "Correlation_Id"],
        [
            {
                "Kernel_Name": "qwen35_moe_group_count_kernel",
                "Start_Timestamp": 220,
                "End_Timestamp": 260,
                "Correlation_Id": 7,
            },
            {
                "Kernel_Name": "copyBuffer",
                "Start_Timestamp": 320,
                "End_Timestamp": 350,
                "Correlation_Id": 8,
            },
        ],
    )

    report = module.analyze(tmp_path, "qwen4exp_prefill_p508_")

    assert report["exact_role_kernels"] == [
        {
            "role": role,
            "kernel": "qwen35_moe_group_count_kernel",
            "api": "hipLaunchKernel",
            "ms": 0.00004,
            "rows": 1,
        },
        {
            "role": role,
            "kernel": "copyBuffer",
            "api": "hipMemcpy",
            "ms": 0.00003,
            "rows": 1,
        },
    ]
