"""TP2 decode kernel inventory rollup: synthetic traces, no GPU or profiler."""
import csv
import json
from types import SimpleNamespace

import pytest

from scripts.tp2_decode_kernel_inventory import (
    DECODE_MARKER,
    _agent_rank_map,
    _read_kernel_rows,
    _read_region_window,
    _rollup,
    _summarize,
)


def _write_kernel_csv(path, rows):
    """rows: (kernel, agent, grid_x, start_ns, duration_ns)."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Kind",
                "Agent_Id",
                "Kernel_Name",
                "Start_Timestamp",
                "End_Timestamp",
                "Grid_Size_X",
                "VGPR_Count",
                "Scratch_Size",
            ]
        )
        for kernel, agent, grid, start, duration in rows:
            writer.writerow(
                ["KERNEL_DISPATCH", agent, kernel, start, start + duration, grid, 64, 0]
            )


def _write_marker_csv(path, name, start, end):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Domain",
                "Function",
                "Process_Id",
                "Thread_Id",
                "Correlation_Id",
                "Start_Timestamp",
                "End_Timestamp",
            ]
        )
        writer.writerow(["MARKER_CORE_RANGE_API", name, 1, 1, 7, start, end])


def _child(steps=2, layers=4, decode_steps=2):
    return {
        "kind": "tp2_decode_kernel_inventory_child",
        "host": "test",
        "model": "fake.gguf",
        "model_sha256": "0" * 64,
        "route": {"mode": "tp2", "reduce_mode": "device"},
        "capacity": 256,
        "prompt_tokens": 8,
        "decode_steps": steps,
        "layer_count": layers,
        "devices": {"0": {"name": "GPU0"}, "1": {"name": "GPU1"}},
        "marker_prefix": DECODE_MARKER,
        "probe": {
            "kernel_fragment": "f32_to_bf16_kernel",
            "launches": [
                {"rank": 0, "device_index": 0, "elements": 1024},
                {"rank": 1, "device_index": 1, "elements": 65536},
            ],
        },
        "region_wall_ms_per_step": 10.0,
        "command": "child --cmd",
    }


def _trace(tmp_path, *, steps=2, layers=4, probe=True, spin_rank1_stalls=True):
    """A two-rank trace: one probe launch per rank, then per-layer kernels.

    Per step and per rank: one 100 us projection, one 5 us norm, one 2 us add,
    and one exchange add whose rank-1 duration carries a stall.
    """
    region_start = 1_000_000
    rows = []
    if probe:
        rows.append(("f32_to_bf16_kernel", "Agent 1", 1024, region_start - 500, 400))
        rows.append(("f32_to_bf16_kernel", "Agent 2", 65536, region_start - 400, 700))
    t = region_start + 100
    for _step in range(steps):
        for layer in range(layers):
            for agent, offset in (("Agent 1", 0), ("Agent 2", 30)):
                rows.append(("gemv_kernel", agent, 1, t + offset, 100_000))
                rows.append(("norm_kernel", agent, 1, t + offset + 1, 5_000))
                rows.append(("add_kernel", agent, 1, t + offset + 2, 2_000))
                stall = 60_000 if (agent == "Agent 2" and spin_rank1_stalls and layer == 0) else 0
                rows.append(("spin_add", agent, 1, t + offset + 3, 4_000 + stall))
            t += 200_000
    region_end = t + 1_000
    _write_kernel_csv(tmp_path / "trace_kernel_trace.csv", rows)
    _write_marker_csv(
        tmp_path / "trace_marker_api_trace.csv", f"{DECODE_MARKER}start", region_start, region_end
    )
    return tmp_path


def test_read_kernel_rows_keeps_agent_as_the_trace_labels_it(tmp_path):
    """rocprofv3 emits Agent_Id as 'Agent N', not a number."""
    _write_kernel_csv(tmp_path / "k_kernel_trace.csv", [("k", "Agent 2", 64, 10, 5)])
    rows = _read_kernel_rows(tmp_path / "k_kernel_trace.csv")
    assert rows[0]["agent_id"] == "Agent 2"
    assert rows[0]["grid_x"] == 64.0
    assert rows[0]["duration_ns"] == 5


def test_read_region_window_requires_exactly_one_region(tmp_path):
    path = tmp_path / "m_marker_api_trace.csv"
    _write_marker_csv(path, "other:start", 1, 2)
    with pytest.raises(ValueError, match="no tp2decode: marker region"):
        _read_region_window(path, DECODE_MARKER)
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow(
            ["MARKER_CORE_RANGE_API", f"{DECODE_MARKER}start", 1, 1, 8, 10, 20]
        )
        csv.writer(handle).writerow(
            ["MARKER_CORE_RANGE_API", f"{DECODE_MARKER}again", 1, 1, 9, 30, 40]
        )
    with pytest.raises(ValueError, match="expected one tp2decode: region"):
        _read_region_window(path, DECODE_MARKER)


def test_agent_rank_map_uses_the_distinct_probe_grid_per_rank():
    probe = _child()["probe"]
    kernels = [
        {"kernel": "f32_to_bf16_kernel", "agent_id": "Agent 1", "grid_x": 1024.0},
        {"kernel": "f32_to_bf16_kernel", "agent_id": "Agent 2", "grid_x": 65536.0},
    ]
    mapping, note = _agent_rank_map(kernels, probe)
    assert mapping == {"Agent 1": 0, "Agent 2": 1}
    assert "labeled probe" in note


def test_agent_rank_map_refuses_an_unverifiable_split():
    probe = _child()["probe"]
    # Same grid on both agents: the trace cannot say which is which.
    ambiguous = [
        {"kernel": "f32_to_bf16_kernel", "agent_id": "Agent 1", "grid_x": 5120.0},
        {"kernel": "f32_to_bf16_kernel", "agent_id": "Agent 2", "grid_x": 5120.0},
    ]
    mapping, note = _agent_rank_map(ambiguous, probe)
    assert mapping == {} and "not reported" in note
    # A missing agent is refused too, rather than guessed.
    missing, note = _agent_rank_map(ambiguous[:1], probe)
    assert missing == {} and "expected 2" in note


def test_summarize_separates_copies_and_reports_a_duration_spread():
    rows = [
        {"kernel": "spin_add", "duration_ns": 4_000},
        {"kernel": "spin_add", "duration_ns": 64_000},
        {"kernel": "spin_add", "duration_ns": 4_000},
        {"kernel": "__amd_rocclr_copyBuffer", "duration_ns": 2_000},
    ]
    summary = _summarize(rows, steps=1, layers=1)
    assert summary["kernel_calls"] == 3
    assert summary["copy_calls_per_step"] == 1.0
    spin = summary["kernels"][0]
    # The stall is visible as the gap between min and p90, not in the mean alone.
    assert spin["min_us"] == 4.0
    assert spin["p50_us"] == 4.0
    assert spin["max_us"] == 64.0
    assert spin["calls_per_step"] == 3.0
    assert spin["calls_per_layer"] == 3.0


def test_rollup_slices_to_the_region_and_splits_by_rank(tmp_path):
    root = _trace(tmp_path)
    report = _rollup(root, _child(), top=10)
    assert report["agent_rank_map"] == {"Agent 1": 0, "Agent 2": 1}
    assert report["decode_steps"] == 2 and report["layer_count"] == 4
    # 4 kernels per layer x 4 layers = 16 per step per rank; the probe is outside
    # the region and must not be counted.
    for rank in ("0", "1"):
        assert report["per_rank"][rank]["kernel_calls_per_step"] == 16.0
    assert report["per_rank"]["0"]["name"] == "GPU0"
    assert report["combined"]["kernel_calls_per_step"] == 32.0
    # The exchange stall lands on rank 1 only.
    rank0 = {e["kernel"]: e for e in report["per_rank"]["0"]["kernels"]}
    rank1 = {e["kernel"]: e for e in report["per_rank"]["1"]["kernels"]}
    assert rank0["spin_add"]["p90_us"] == 4.0
    assert rank1["spin_add"]["p90_us"] == 64.0
    assert report["top"][0]["kernel"] == "gemv_kernel"


def test_rollup_omits_the_per_rank_split_when_the_probe_cannot_separate(tmp_path):
    root = _trace(tmp_path, probe=False)
    report = _rollup(root, _child(), top=5)
    assert report["per_rank"] == {}
    assert "not reported" in report["agent_rank_source"]
    # The combined inventory is still reported.
    assert report["combined"]["kernel_calls_per_step"] == 32.0


def test_rollup_rejects_a_trace_with_no_decode_region(tmp_path):
    root = _trace(tmp_path)
    _write_marker_csv(root / "trace_marker_api_trace.csv", "prefill:start", 1, 2)
    with pytest.raises(ValueError, match="no tp2decode: marker region"):
        _rollup(root, _child(), top=5)


def test_rollup_ignores_kernels_outside_the_region(tmp_path):
    root = _trace(tmp_path)
    with (root / "trace_kernel_trace.csv").open("a", newline="") as handle:
        # Long after the region ends: warmup or teardown work.
        csv.writer(handle).writerow(
            ["KERNEL_DISPATCH", "Agent 1", "late_kernel", 99_000_000, 99_100_000, 1, 64, 0]
        )
    report = _rollup(root, _child(), top=20)
    assert all(e["kernel"] != "late_kernel" for e in report["combined"]["kernels"])
