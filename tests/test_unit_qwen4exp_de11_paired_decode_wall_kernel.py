from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_de11_paired_decode_wall_kernel.py"

TRACE_HEADER = (
    "Kind,Agent_Id,Queue_Id,Stream_Id,Thread_Id,Dispatch_Id,Kernel_Id,"
    "Kernel_Name,Correlation_Id,Start_Timestamp,End_Timestamp,LDS_Block_Size,"
    "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Workgroup_Size_X,"
    "Workgroup_Size_Y,Workgroup_Size_Z,Grid_Size_X,Grid_Size_Y,Grid_Size_Z"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_de11_paired_decode_wall_kernel", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(**overrides):
    payload = {
        "protocol": {"case_ids": ["code-p512"], "decode_transitions": 2},
        "samples": [
            {
                "repetition": 0,
                "decode_ms": 20.0,
                "prefill_tok_s": 100.0,
                "phase_windows_ns": {"decode": [1000, 2000]},
            },
            {
                "repetition": 1,
                "decode_ms": 22.0,
                "prefill_tok_s": 101.0,
                "phase_windows_ns": {"decode": [3000, 4000]},
            },
        ],
        "source": {"head": "deadbeef"},
        "profile": {"manifest_sha256": "manifest"},
        "host": {"machine_id": "host"},
    }
    payload.update(overrides)
    return payload


def _write_trace(tmp_path: Path, rows) -> Path:
    path = tmp_path / "kernel_trace.csv"
    lines = [TRACE_HEADER]
    for name, start, end in rows:
        lines.append(
            f'"KERNEL_DISPATCH","Agent 1",2,0,1,1,4,"{name}",31,{start},{end},'
            "0,0,16,0,128,512,1,1,1024,1,1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_summarize_wall_reports_median_per_step_and_spread() -> None:
    module = _load_script()
    summary = module.summarize_wall(_artifact())
    # 20 ms / 2 steps and 22 ms / 2 steps -> median 10.5 ms/step.
    assert summary["wall_ms_per_step_median"] == pytest.approx(10.5)
    assert summary["wall_ms_per_step_samples"] == pytest.approx([10.0, 11.0])
    assert summary["within_arm_spread"] == pytest.approx(1.0 / 10.5)
    assert summary["case_id"] == "code-p512"
    assert summary["decode_transitions"] == 2


def test_summarize_kernel_slices_last_measured_decode_window(tmp_path) -> None:
    module = _load_script()
    trace = _write_trace(
        tmp_path,
        [
            ("early_prefill_kernel", 100, 200),  # before the last decode window
            ("first_rep_kernel", 1100, 1300),  # inside the first (unused) window
            ("decode_a", 3100, 3400),  # inside the last decode window (300 ns)
            ("decode_b", 3500, 4000),  # inside the last decode window (500 ns)
            ("after_kernel", 4100, 4200),  # after the window
        ],
    )
    summary = module.summarize_kernel(_artifact(), trace)
    # Only the two kernels inside [3000, 4000] count: 800 ns over 2 steps.
    assert summary["launches"] == 2
    assert summary["launches_per_step"] == pytest.approx(1.0)
    assert summary["kernel_ms"] == pytest.approx(800.0 / 1_000_000.0)
    assert summary["kernel_ms_per_step"] == pytest.approx(400.0 / 1_000_000.0)
    assert summary["span_s"] == pytest.approx(1000.0 / 1_000_000_000.0)
    assert set(summary["families"]) == {"decode_a", "decode_b"}


def test_summarize_kernel_requires_a_kernel_inside_the_window(tmp_path) -> None:
    module = _load_script()
    trace = _write_trace(tmp_path, [("outside", 100, 200)])
    with pytest.raises(ValueError, match="no kernel dispatches fall inside"):
        module.summarize_kernel(_artifact(), trace)


def test_pair_row_computes_remainder() -> None:
    module = _load_script()
    wall = module.summarize_wall(_artifact())
    kernel = {
        "case_id": "code-p512",
        "kernel_ms_per_step": 6.5,
        "launches_per_step": 1234.0,
        "source_head": "deadbeef",
        "profile_manifest_sha256": "manifest",
        "host": "host",
    }
    row = module.pair_row(wall, kernel)
    assert row["wall_ms_per_step"] == pytest.approx(10.5)
    assert row["remainder_ms_per_step"] == pytest.approx(4.0)
    assert row["kernel_share"] == pytest.approx(6.5 / 10.5)


def test_pair_row_rejects_case_mismatch() -> None:
    module = _load_script()
    wall = module.summarize_wall(_artifact())
    with pytest.raises(ValueError, match="case mismatch"):
        module.pair_row(wall, {"case_id": "code-p4096", "kernel_ms_per_step": 1.0})


def test_session_summary_joins_wall_and_kernel_runs(tmp_path) -> None:
    module = _load_script()
    (tmp_path / "wall").mkdir()
    (tmp_path / "kernel").mkdir()
    for tag, value in (("A", 10.0), ("B", 12.0)):
        (tmp_path / "wall" / f"code-p512-{tag}.json").write_text(
            json.dumps({"case_id": "code-p512", "wall_ms_per_step_median": value})
        )
    for tag, value in (("A", 6.0), ("B", 8.0)):
        (tmp_path / "kernel" / f"code-p512-{tag}.json").write_text(
            json.dumps(
                {"case_id": "code-p512", "kernel_ms_per_step": value, "launches_per_step": 1700.0}
            )
        )
    summary = module.session_summary(tmp_path)
    case = summary["cases"]["code-p512"]
    assert case["wall_ms_per_step_median"] == pytest.approx(11.0)
    assert case["kernel_ms_per_step_median"] == pytest.approx(7.0)
    assert case["remainder_ms_per_step"] == pytest.approx(4.0)


def test_session_summary_requires_both_arms(tmp_path) -> None:
    module = _load_script()
    (tmp_path / "wall").mkdir()
    (tmp_path / "kernel").mkdir()
    (tmp_path / "wall" / "code-p512-A.json").write_text(
        json.dumps({"case_id": "code-p512", "wall_ms_per_step_median": 10.0})
    )
    with pytest.raises(ValueError, match="need at least one wall and one kernel"):
        module.session_summary(tmp_path)


def test_decode_steps_falls_back_to_default() -> None:
    module = _load_script()
    assert module.decode_steps({}) == module.DEFAULT_STEPS
    assert module.decode_steps({"protocol": {"decode_transitions": 7}}) == 7


def test_cli_writes_json(tmp_path, capsys) -> None:
    module = _load_script()
    artifact_path = tmp_path / "bench.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    out = tmp_path / "wall.json"
    assert module.main.__module__ == module.__name__
    import sys

    argv = sys.argv
    try:
        sys.argv = ["prog", "wall", str(artifact_path), "--output", str(out)]
        assert module.main() == 0
    finally:
        sys.argv = argv
    assert json.loads(out.read_text())["case_id"] == "code-p512"
