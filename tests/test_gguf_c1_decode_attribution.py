"""Deterministic checks for the C1 decode attribution reduction.

The GPU run is expensive and the reduction is where a silent mistake would be
worst: an attribution table that does not add up reads as authoritative while
being wrong. These tests therefore pin the classification, the CSV readers, the
interval union, and the additive decomposition identity on synthetic traces, so
the profiled run only has to supply real numbers.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from scripts.gguf_c1_decode_attribution import (
    COMPILER_VERSION_FILE_ENV,
    REQUIRE_CACHED_BUILD_ENV,
    STEP_MARKER_PREFIX,
    _union_ns,
    classify_kernel,
    compare_api_to_copies,
    read_hip_api,
    read_kernels,
    read_marker_windows,
    read_memory_copies,
    split_step_phases,
    summarize,
    trace_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_MARKER_COLUMNS = ("Function", "Start_Timestamp", "End_Timestamp")
_KERNEL_COLUMNS = ("Kernel_Name", "Start_Timestamp", "End_Timestamp")
_API_COLUMNS = ("Function", "Start_Timestamp", "End_Timestamp")
_COPY_COLUMNS = ("Direction", "Start_Timestamp", "End_Timestamp")


def _write(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _window(index: int, start: int, end: int) -> dict[str, object]:
    return {
        "Function": f"{STEP_MARKER_PREFIX}{index}",
        "Start_Timestamp": str(start),
        "End_Timestamp": str(end),
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("__amd_rocclr_copyBuffer", "rocclr_copy"),
        (
            "void qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<hip_bfloat16>",
            "paged_full_attn_decode",
        ),
        ("void qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel<float>", "paged_full_attn_decode"),
        ("void qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel", "paged_attn_prefill"),
        ("void gdn_recurrent_scan_kernel", "gdn_linear_attention"),
        ("void top1_sampler_kernel", "sampler_top1"),
        ("void gguf_q4_k_q8_1_mmq_prefill_kernel", "gguf_q4_k_q8_1_mmq"),
        ("void gguf_q6_k_t16_gemv_rowtile_col8_kernel<float, 8, true>", "gguf_q6_k_t16_gemv_rowtile"),
        ("void gguf_q4_k_t16_gemv_rowtile_kernel<float, 4>", "gguf_q4_k_t16_gemv_rowtile"),
        ("void gguf_q4_t16_dense_wmma_prefill_silu_bf16_kernel", "gguf_q4_k_t16_dense_wmma"),
        ("void something_unrecognized", "other"),
    ),
)
def test_classify_kernel_buckets(name: str, expected: str) -> None:
    assert classify_kernel(name) == expected


def test_split_step_phases_separates_readback_from_host_work() -> None:
    split = split_step_phases([0.035, 0.030], [20.0, 12.5])

    assert split["step_walls_ms"] == [35.0, 30.0]
    assert split["readback_ms"] == [20.0, 12.5]
    assert split["host_outside_readback_ms"] == [15.0, 17.5]
    # The split must reconstruct the wall, or the phases do not partition it.
    for wall, readback, host in zip(
        split["step_walls_ms"],
        split["readback_ms"],
        split["host_outside_readback_ms"],
        strict=True,
    ):
        assert readback + host == pytest.approx(wall)


def test_split_step_phases_rejects_a_readback_longer_than_its_step() -> None:
    # A mis-attributed copy would otherwise surface as negative host time, which
    # reads as an improvement rather than an error.
    with pytest.raises(ValueError, match="cannot exceed its own wall"):
        split_step_phases([0.020], [25.0])


def test_split_step_phases_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="per step"):
        split_step_phases([0.035, 0.030], [20.0])


def test_read_marker_windows_sorts_and_filters(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "marker_api_trace.csv",
        _MARKER_COLUMNS,
        [
            _window(1, 200, 300),
            {"Function": "unrelated_range", "Start_Timestamp": "0", "End_Timestamp": "1"},
            _window(0, 100, 150),
            {"Function": "another", "Start_Timestamp": "0", "End_Timestamp": "1"},
        ],
    )

    windows = read_marker_windows(path)

    assert [w["step"] for w in windows] == [0, 1]
    assert windows[0] == {"step": 0, "start_ns": 100, "end_ns": 150}


def test_read_marker_windows_skips_inverted_ranges(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "marker_api_trace.csv",
        _MARKER_COLUMNS,
        [_window(0, 300, 200), _window(1, 100, 150)],
    )

    assert [w["step"] for w in read_marker_windows(path)] == [1]


def test_read_kernels_derives_duration_and_family(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "kernel_trace.csv",
        _KERNEL_COLUMNS,
        [
            {
                "Kernel_Name": "void gguf_q4_k_t16_gemv_rowtile_kernel<float>",
                "Start_Timestamp": "1000",
                "End_Timestamp": "1500",
            }
        ],
    )

    rows = read_kernels(path)

    assert rows[0]["duration_ns"] == 500
    assert rows[0]["family"] == "gguf_q4_k_t16_gemv_rowtile"


def test_read_hip_api_strips_arguments(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "hip_api_trace.csv",
        _API_COLUMNS,
        [{"Function": "hipLaunchKernel(void*, dim3)", "Start_Timestamp": "10", "End_Timestamp": "20"}],
    )

    rows = read_hip_api(path)

    assert rows[0]["function"] == "hipLaunchKernel"
    assert rows[0]["duration_ns"] == 10


def test_read_memory_copies_reads_direction(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "memory_copy_trace.csv",
        _COPY_COLUMNS,
        [
            {"Direction": "MEMORY_COPY_HOST_TO_DEVICE", "Start_Timestamp": "10", "End_Timestamp": "40"},
            {"Direction": "MEMORY_COPY_DEVICE_TO_HOST", "Start_Timestamp": "50", "End_Timestamp": "70"},
        ],
    )

    rows = read_memory_copies(path)

    assert [r["direction"] for r in rows] == [
        "MEMORY_COPY_HOST_TO_DEVICE",
        "MEMORY_COPY_DEVICE_TO_HOST",
    ]
    assert [r["duration_ns"] for r in rows] == [30, 20]


def test_union_merges_overlapping_intervals() -> None:
    assert _union_ns([(0, 10), (5, 20), (30, 40)]) == 30
    assert _union_ns([(0, 10), (10, 20)]) == 20
    assert _union_ns([]) == 0
    assert _union_ns([(50, 60), (0, 10)]) == 20


def test_summarize_attributes_only_in_window_intervals() -> None:
    windows = [{"step": 0, "start_ns": 1000, "end_ns": 2000}]

    summary = summarize(
        windows=windows,
        kernels=[
            # inside: 400 ns of device work
            {"family": "gguf_q4_k_t16_gemv_rowtile", "start_ns": 1100, "end_ns": 1400, "duration_ns": 300},
            {"family": "paged_full_attn_decode", "start_ns": 1500, "end_ns": 1600, "duration_ns": 100},
            # outside the window and must not be counted
            {"family": "gguf_q4_k_t16_dense_wmma", "start_ns": 5000, "end_ns": 9000, "duration_ns": 4000},
        ],
        hip_api=[
            {"function": "hipLaunchKernel", "start_ns": 1050, "end_ns": 1150, "duration_ns": 100},
            {"function": "hipGetLastError", "start_ns": 5000, "end_ns": 5001, "duration_ns": 1},
        ],
        copies=[
            {
                "direction": "MEMORY_COPY_DEVICE_TO_HOST",
                "start_ns": 1700,
                "end_ns": 1750,
                "duration_ns": 50,
            },
            {
                "direction": "MEMORY_COPY_HOST_TO_DEVICE",
                "start_ns": 9000,
                "end_ns": 9100,
                "duration_ns": 100,
            },
        ],
        step_walls_ms=[12.0],
    )

    assert summary["windows"]["window_total_ns"] == 1000
    assert summary["windows"]["measured_steps"] == 1
    assert summary["device"]["kernel_launches"] == 2
    assert summary["device"]["device_union_ns"] == 400
    assert summary["device"]["gpu_idle_ns"] == 600
    assert summary["hip_api"]["calls"] == 1
    assert summary["hip_api"]["api_union_ns"] == 100
    assert summary["memory_copies"]["copies"] == 1
    assert summary["memory_copies"]["bytes_available"] is False
    assert summary["memory_copies"]["by_direction"][0]["direction"] == (
        "MEMORY_COPY_DEVICE_TO_HOST"
    )
    assert {row["family"] for row in summary["kernel_families"]} == {
        "gguf_q4_k_t16_gemv_rowtile",
        "paged_full_attn_decode",
    }


def test_summarize_clips_intervals_that_straddle_a_window_edge() -> None:
    """A dispatch that starts before a window must contribute only its overlap."""

    summary = summarize(
        windows=[{"step": 0, "start_ns": 1000, "end_ns": 2000}],
        kernels=[
            # starts 500 ns before the window and ends 100 ns inside it
            {"family": "other", "start_ns": 500, "end_ns": 1100, "duration_ns": 600},
        ],
        hip_api=[],
        copies=[],
        step_walls_ms=[1.0],
    )

    assert summary["device"]["kernel_time_sum_ns"] == 100
    assert summary["device"]["device_union_ns"] == 100
    assert summary["decomposition"]["residual_ns"] == 900


def test_family_shares_sum_to_one_hundred() -> None:
    summary = summarize(
        windows=[{"step": 0, "start_ns": 0, "end_ns": 1000}],
        kernels=[
            {"family": "a", "start_ns": 0, "end_ns": 300, "duration_ns": 300},
            {"family": "b", "start_ns": 300, "end_ns": 500, "duration_ns": 200},
            {"family": "c", "start_ns": 500, "end_ns": 600, "duration_ns": 100},
        ],
        hip_api=[],
        copies=[],
        step_walls_ms=[1.0],
    )

    total = sum(row["share_pct_of_kernel_time"] for row in summary["kernel_families"])
    assert round(total, 6) == 100.0
    assert all(
        row["share_pct_of_kernel_time"] <= 100.0 for row in summary["kernel_families"]
    )


def test_dispatch_overlap_ratio_exposes_overlapping_intervals() -> None:
    """Overlapping dispatch intervals must be visible, not silently over-shared."""

    summary = summarize(
        windows=[{"step": 0, "start_ns": 0, "end_ns": 1000}],
        kernels=[
            {"family": "a", "start_ns": 0, "end_ns": 600, "duration_ns": 600},
            {"family": "b", "start_ns": 100, "end_ns": 700, "duration_ns": 600},
        ],
        hip_api=[],
        copies=[],
        step_walls_ms=[1.0],
    )

    assert summary["device"]["device_union_ns"] == 700
    assert summary["device"]["kernel_time_sum_ns"] == 1200
    assert summary["device"]["dispatch_overlap_ratio"] == pytest.approx(1.714, abs=1e-3)


def test_summarize_decomposition_balances() -> None:
    summary = summarize(
        windows=[{"step": 0, "start_ns": 0, "end_ns": 700}, {"step": 1, "start_ns": 1000, "end_ns": 1500}],
        kernels=[
            {"family": "other", "start_ns": 100, "end_ns": 400, "duration_ns": 300},
            {"family": "other", "start_ns": 1100, "end_ns": 1200, "duration_ns": 100},
        ],
        hip_api=[{"function": "hipLaunchKernel", "start_ns": 50, "end_ns": 150, "duration_ns": 100}],
        copies=[],
        step_walls_ms=[10.0, 11.0],
    )

    decomposition = summary["decomposition"]
    assert decomposition["balances"] is True
    assert decomposition["residual_ns"] >= 0
    # 1200 ns of measured span, 450 ns covered by device or API work
    assert decomposition["window_ns"] == 1200
    assert decomposition["device_union_ns"] == 400
    assert decomposition["hip_api_union_ns"] == 100
    assert decomposition["combined_union_ns"] == 450
    assert decomposition["residual_ns"] == 750
    assert decomposition["overlap_device_api_ns"] == 50
    # the GPU-idle split must be a real identity too
    assert decomposition["gpu_idle_ns"] == 800
    assert (
        decomposition["api_while_gpu_idle_ns"] + decomposition["residual_ns"]
        == decomposition["gpu_idle_ns"]
    )


def test_summarize_records_sync_memcpy_callsites() -> None:
    summary = summarize(
        windows=[{"step": 0, "start_ns": 0, "end_ns": 1000}],
        kernels=[],
        hip_api=[],
        copies=[],
        step_walls_ms=[1.0],
        sync_callsites=[
            {"site": "runner.py:100 step", "calls": 2, "total_ns": 4_000_000}
        ],
    )

    stall = summary["sync_memcpy_stall"]
    assert stall["calls"] == 2
    assert stall["per_step_calls"] == 2.0
    assert stall["per_step_ms"] == 4.0
    assert stall["callsites"][0]["site"] == "runner.py:100 step"


def test_compare_api_to_copies_measures_queue_drain() -> None:
    """A synchronous copy's API duration must be compared with its transfer."""

    result = compare_api_to_copies(
        hip_api=[
            {
                "function": "hipMemcpy",
                "correlation_id": 7,
                "start_ns": 0,
                "end_ns": 27_480_000,
                "duration_ns": 27_480_000,
            },
            {
                "function": "hipLaunchKernel",
                "correlation_id": 8,
                "start_ns": 0,
                "end_ns": 100,
                "duration_ns": 100,
            },
        ],
        copies=[
            {
                "direction": "MEMORY_COPY_HOST_TO_DEVICE",
                "correlation_id": 7,
                "start_ns": 0,
                "end_ns": 10_000,
                "duration_ns": 10_000,
            }
        ],
    )

    assert result["matched_calls"] == 1
    assert result["median_api_ms"] == pytest.approx(27.48, abs=1e-3)
    assert result["median_transfer_ms"] == pytest.approx(0.01, abs=1e-6)
    assert result["api_over_transfer_ratio"] > 1000
    assert result["directions"] == {"MEMORY_COPY_HOST_TO_DEVICE": 1}


def test_compare_api_to_copies_handles_no_matches() -> None:
    result = compare_api_to_copies(hip_api=[], copies=[])

    assert result["matched_calls"] == 0
    assert "note" in result


def test_summarize_rejects_a_trace_without_step_windows() -> None:
    with pytest.raises(ValueError, match="no decode-step marker windows"):
        summarize(windows=[], kernels=[], hip_api=[], copies=[], step_walls_ms=[])


def test_summarize_reports_per_step_rates() -> None:
    summary = summarize(
        windows=[{"step": 0, "start_ns": 0, "end_ns": 100}, {"step": 1, "start_ns": 200, "end_ns": 300}],
        kernels=[
            {"family": "other", "start_ns": 0, "end_ns": 10, "duration_ns": 10},
            {"family": "other", "start_ns": 200, "end_ns": 210, "duration_ns": 10},
        ],
        hip_api=[],
        copies=[],
        step_walls_ms=[1.0, 2.0],
    )

    assert summary["device"]["launches_per_step"] == 1.0
    assert summary["hip_api"]["calls_per_step"] == 0.0
    assert summary["windows"]["child_step_wall_median_ms"] == 1.5


def test_trace_environment_uses_the_names_the_build_module_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text("AMD clang version 22.0.0git\n", encoding="utf-8")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")

    environment = trace_environment(version_file, "hip_gfx1100")

    assert environment[REQUIRE_CACHED_BUILD_ENV] == "1"
    assert environment[COMPILER_VERSION_FILE_ENV] == str(version_file)
    assert "HIPENGINE_HIP_REQUIRE_CACHED_BUILD" not in environment
    assert "ROCR_VISIBLE_DEVICES" not in environment


def test_trace_environment_rejects_an_empty_version_file(tmp_path: Path) -> None:
    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="compiler version file is empty"):
        trace_environment(version_file, "hip_gfx1100")


def test_no_script_uses_a_lookalike_cache_only_switch() -> None:
    pattern = re.compile(r"""["']HIPENGINE_HIP_REQUIRE_CACHED_BUILD["']""")
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "scripts").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )

    assert offenders == []
