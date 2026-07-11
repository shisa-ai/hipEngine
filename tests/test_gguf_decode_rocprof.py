from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.gguf_decode_rocprof import (
    MARKER_PREFIX,
    _amdahl_speedup,
    _build_child_command,
    _filter_kernels_by_windows,
    _read_kernels,
    _read_marker_windows,
    _session_kwargs,
    _summarize_rows,
    _summarize_wall_runs,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_marker_windows_filter_exact_decode_region(tmp_path: Path) -> None:
    markers = tmp_path / "trace_marker_api_trace.csv"
    kernels = tmp_path / "trace_kernel_trace.csv"
    _write_csv(
        markers,
        ["Function", "Start_Timestamp", "End_Timestamp"],
        [
            {"Function": "unrelated", "Start_Timestamp": 1, "End_Timestamp": 99},
            {"Function": f"{MARKER_PREFIX}0", "Start_Timestamp": 100, "End_Timestamp": 200},
            {"Function": f"{MARKER_PREFIX}1", "Start_Timestamp": 300, "End_Timestamp": 410},
        ],
    )
    _write_csv(
        kernels,
        ["Kernel_Name", "Start_Timestamp", "End_Timestamp", "VGPR_Count", "Scratch_Size"],
        [
            {"Kernel_Name": "prefill_kernel", "Start_Timestamp": 50, "End_Timestamp": 90, "VGPR_Count": 8, "Scratch_Size": 0},
            {"Kernel_Name": "q8_0_t16_gemv_kernel", "Start_Timestamp": 120, "End_Timestamp": 170, "VGPR_Count": 32, "Scratch_Size": 0},
            {"Kernel_Name": "q6_k_t16_gemv_kernel", "Start_Timestamp": 320, "End_Timestamp": 390, "VGPR_Count": 40, "Scratch_Size": 0},
            {"Kernel_Name": "after_kernel", "Start_Timestamp": 420, "End_Timestamp": 430, "VGPR_Count": 4, "Scratch_Size": 0},
        ],
    )

    windows = _read_marker_windows(markers, MARKER_PREFIX)
    rows = _read_kernels(kernels)
    selected = _filter_kernels_by_windows(rows, [(start, end) for _, start, end in windows])

    assert windows == [(0, 100, 200), (1, 300, 410)]
    assert [row["family"] for row in selected] == ["q8_0_t16_gemv", "q6_k_t16_gemv"]


def test_layer_family_summary_emits_per_token_and_amdahl_ceiling() -> None:
    rows = [
        {
            "kernel": "q8_0_t16_gemv_kernel",
            "family": "q8_0_t16_gemv",
            "bucket": "dense_q8_0_gemv",
            "start_ns": 0,
            "end_ns": 60_000,
            "duration_ns": 60_000,
            "vgpr": 32,
            "scratch": 0,
        },
        {
            "kernel": "q6_k_t16_gemv_kernel",
            "family": "q6_k_t16_gemv",
            "bucket": "lm_head",
            "start_ns": 70_000,
            "end_ns": 110_000,
            "duration_ns": 40_000,
            "vgpr": 40,
            "scratch": 0,
        },
    ]

    summary = _summarize_rows(rows, steps=2, top=10)

    assert summary["total_gpu_us"] == pytest.approx(100.0)
    assert summary["gpu_us_per_token"] == pytest.approx(50.0)
    dense = summary["buckets"][0]
    assert dense["name"] == "dense_q8_0_gemv"
    assert dense["share_pct"] == pytest.approx(60.0)
    assert dense["calls_per_token"] == pytest.approx(0.5)
    assert dense["amdahl_speedup_if_2x"] == pytest.approx(1.428571, rel=1e-5)
    assert dense["amdahl_speedup_if_infinite"] == pytest.approx(2.5)


def test_amdahl_speedup_handles_zero_and_full_share() -> None:
    assert _amdahl_speedup(0.0, 2.0) == 1.0
    assert _amdahl_speedup(1.0, 2.0) == 2.0
    assert _amdahl_speedup(1.0, float("inf")) is None


def test_wall_summary_requires_exact_tokens() -> None:
    runs = [
        {"wall_ms_per_token": 20.0, "generated_token_ids": [9707, 9707], "expected_token_id": 9707},
        {"wall_ms_per_token": 18.0, "generated_token_ids": [9707, 9707], "expected_token_id": 9707},
        {"wall_ms_per_token": 19.0, "generated_token_ids": [9707, 9707], "expected_token_id": 9707},
    ]
    summary = _summarize_wall_runs(runs, expected_token_id=9707)
    assert summary["all_tokens_exact"] is True
    assert summary["median_ms_per_token"] == pytest.approx(19.0)
    assert summary["median_tok_s"] == pytest.approx(1000.0 / 19.0)

    runs[1]["generated_token_ids"] = [9707, 9]
    with pytest.raises(ValueError, match="unexpected token"):
        _summarize_wall_runs(runs, expected_token_id=9707)


def test_child_command_pins_route_and_cached_build(tmp_path: Path) -> None:
    command = _build_child_command(
        python="/venv/bin/python",
        script=Path("scripts/gguf_decode_rocprof.py"),
        child_mode="profile",
        source_root=tmp_path,
        model=Path("/models/model.gguf"),
        backend="hip_gfx1151",
        prompt_token_id=9707,
        prompt_length=512,
        expected_token_id=9707,
        steps=24,
        warmup_steps=4,
        benchmark_warmups=0,
        repetitions=1,
        compiler_version_file=Path("/tmp/hipcc.txt"),
        child_json=Path("/tmp/child.json"),
        require_cached=True,
    )

    assert command[:3] == ["/venv/bin/python", "scripts/gguf_decode_rocprof.py", "--child-mode"]
    assert "profile" in command
    assert command[command.index("--backend") + 1] == "hip_gfx1151"
    assert command[command.index("--prompt-length") + 1] == "512"
    assert "--require-cached" in command


def test_historical_session_kwargs_omit_unsupported_backend() -> None:
    class HistoricalSession:
        def __init__(
            self,
            model_path: Path,
            *,
            max_sequence_length: int,
            compiler_version: str | None,
            require_cached_build: bool,
            use_wmma_prefill: bool,
            use_gemv_decode: bool,
        ) -> None:
            del model_path, max_sequence_length, compiler_version
            del require_cached_build, use_wmma_prefill, use_gemv_decode

    kwargs, backend_supported = _session_kwargs(
        HistoricalSession,
        max_sequence_length=256,
        compiler_version="hipcc",
        require_cached_build=True,
        backend="hip_gfx1151",
    )

    assert backend_supported is False
    assert "backend" not in kwargs
    assert kwargs["use_wmma_prefill"] is True
    assert kwargs["use_gemv_decode"] is True
