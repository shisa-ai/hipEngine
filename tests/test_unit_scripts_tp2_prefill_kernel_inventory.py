"""Unit tier: the TP2 bulk-prefill kernel inventory's rollup.

The inventory's value is its ranking, so what needs pinning is that the ranking
counts the right rows: kernels are attributed by start timestamp inside the ROCTX
region, a kernel that starts before the region opens belongs to the warmup and
must not appear, and the per-prefill columns divide by the prefill count rather
than by the step count the decode inventory uses.

No device contact and no rocprofv3: the rollup reads two CSVs, so this drives it
with a synthetic trace whose durations are chosen so the expected ranking is
obvious by inspection.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tp2_prefill_kernel_inventory.py"

KERNEL_COLUMNS = [
    "Kind",
    "Agent_Id",
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
    "Grid_Size_X",
]


def _load_inventory():
    spec = importlib.util.spec_from_file_location("tp2_prefill_kernel_inventory", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = _load_inventory()


def _write_trace(root: Path, kernels, *, region=(1_000, 10_000_000)) -> None:
    """One kernel-trace CSV plus one marker CSV holding a single region.

    Timestamps are nanoseconds, as rocprofv3 writes them, and the default region
    is wide enough for millisecond-scale kernels to sit inside it.
    """

    root.mkdir(parents=True, exist_ok=True)
    with (root / "tp2-prefill-inventory_kernel_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KERNEL_COLUMNS)
        writer.writeheader()
        for index, (name, agent, start, end, grid) in enumerate(kernels):
            writer.writerow(
                {
                    "Kind": "KERNEL_DISPATCH",
                    "Agent_Id": agent,
                    "Kernel_Name": name,
                    "Start_Timestamp": start,
                    "End_Timestamp": end,
                    "Grid_Size_X": grid,
                }
            )
    with (root / "tp2-prefill-inventory_marker_api_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Function", "Start_Timestamp", "End_Timestamp"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "Function": "tp2prefill:start",
                "Start_Timestamp": region[0],
                "End_Timestamp": region[1],
            }
        )


def _child(**overrides):
    record = {
        "prefills": 2,
        "layer_count": 4,
        "prompt_tokens": 128,
        "bulk_prefill_workspace_rows": 128,
        "probe": {},
        "logits_sha256": ["aa", "aa"],
    }
    record.update(overrides)
    return record


def test_rollup_attributes_by_start_timestamp_inside_the_region(tmp_path: Path) -> None:
    """A kernel that starts before the region is warmup, not measured work."""

    _write_trace(
        tmp_path,
        [
            # Warmup: starts before the region opens, so it is excluded even
            # though it is the longest kernel in the file.
            ("warmup_kernel", "Agent 1", 10, 900_000, 64),
            ("inside_kernel", "Agent 1", 2_000_000, 2_500_000, 64),
            ("after_kernel", "Agent 1", 20_000_000, 20_400_000, 64),
        ],
    )
    report = inventory._rollup(tmp_path, _child(), top=10)
    names = [entry["kernel"] for entry in report["combined"]["kernels"]]
    assert names == ["inside_kernel"]


def test_rollup_divides_by_prefills_not_by_steps(tmp_path: Path) -> None:
    """Per-prefill columns use the prefill count; calls halve when it doubles."""

    # Timestamps are nanoseconds: 1 ms per call.
    _write_trace(
        tmp_path,
        [
            ("k", "Agent 1", 2_000_000, 3_000_000, 64),
            ("k", "Agent 1", 3_100_000, 4_100_000, 64),
        ],
    )
    two = inventory._rollup(tmp_path, _child(prefills=2), top=10)["combined"]
    assert two["kernel_calls_per_prefill"] == 1.0
    assert two["kernels"][0]["ms_per_prefill"] == 1.0

    one = inventory._rollup(tmp_path, _child(prefills=1), top=10)["combined"]
    assert one["kernel_calls_per_prefill"] == 2.0
    assert one["kernels"][0]["ms_per_prefill"] == 2.0


def test_rollup_ranks_by_cost_per_prefill(tmp_path: Path) -> None:
    _write_trace(
        tmp_path,
        [
            ("cheap", "Agent 1", 2_000_000, 2_100_000, 64),
            ("expensive", "Agent 1", 3_000_000, 6_000_000, 64),
            ("middling", "Agent 1", 6_100_000, 6_600_000, 64),
        ],
    )
    report = inventory._rollup(tmp_path, _child(), top=10)
    names = [entry["kernel"] for entry in report["combined"]["kernels"]]
    assert names == ["expensive", "middling", "cheap"]


def test_rollup_reports_per_layer_and_per_token_columns(tmp_path: Path) -> None:
    """The per-layer column is what a fusion decision reads."""

    # 800 us, once, over 2 prefills of 4 layers and 128 prompt tokens.
    _write_trace(tmp_path, [("k", "Agent 1", 2_000_000, 2_800_000, 64)])
    entry = inventory._rollup(tmp_path, _child(), top=10)["combined"]["kernels"][0]
    assert entry["us_per_layer"] == 100.0
    assert entry["us_per_prompt_token"] == 800_000 / 1000 / 2 / 128
    assert entry["ms_per_prefill"] == 0.4


def test_rollup_flags_repeatability_from_the_logits_digests(tmp_path: Path) -> None:
    _write_trace(tmp_path, [("k", "Agent 1", 2_000_000, 2_800_000, 64)])
    assert inventory._rollup(tmp_path, _child(), top=10)["logits_repeatable"] is True
    assert (
        inventory._rollup(tmp_path, _child(logits_sha256=["aa", "bb"]), top=10)[
            "logits_repeatable"
        ]
        is False
    )


def test_rollup_rejects_a_region_with_no_kernels(tmp_path: Path) -> None:
    _write_trace(tmp_path, [("warmup_kernel", "Agent 1", 10, 900, 64)])
    try:
        inventory._rollup(tmp_path, _child(), top=10)
    except ValueError as error:
        assert "contains no kernels" in str(error)
    else:  # pragma: no cover - the assertion above is the contract
        raise AssertionError("a region with no kernels must not roll up")


def test_summarize_separates_small_kernels_from_the_ranked_list() -> None:
    """Small-kernel accounting is the fusion question, so it is pinned."""

    rows = [
        {"kernel": "big", "start_ns": 0, "end_ns": 1_000_000, "duration_ns": 1_000_000,
         "agent_id": "Agent 1", "grid_x": 1.0, "vgpr": None, "scratch": None},
        {"kernel": "tiny", "start_ns": 0, "end_ns": 2_000, "duration_ns": 2_000,
         "agent_id": "Agent 1", "grid_x": 1.0, "vgpr": None, "scratch": None},
    ]
    summary = inventory._summarize(rows, prefills=1, layers=4, prompt_tokens=128)
    assert summary["kernel_ms_per_prefill"] == 1.002
    assert summary["small_kernel_count"] == 1
    assert summary["small_kernel_ms_per_prefill"] == 0.002
    assert [entry["kernel"] for entry in summary["kernels"]] == ["big", "tiny"]


def test_summarize_keeps_device_copies_out_of_the_kernel_ranking() -> None:
    rows = [
        {"kernel": "__amd_rocclr_copy_buffer", "start_ns": 0, "end_ns": 5_000,
         "duration_ns": 5_000, "agent_id": "Agent 1", "grid_x": None,
         "vgpr": None, "scratch": None},
        {"kernel": "real", "start_ns": 0, "end_ns": 1_000, "duration_ns": 1_000,
         "agent_id": "Agent 1", "grid_x": 1.0, "vgpr": None, "scratch": None},
    ]
    summary = inventory._summarize(rows, prefills=1, layers=4, prompt_tokens=128)
    assert [entry["kernel"] for entry in summary["kernels"]] == ["real"]
    assert summary["copy_calls_per_prefill"] == 1.0
    assert summary["copy_ms_per_prefill"] == 0.005


def test_child_records_the_head_rows_it_projects() -> None:
    """The inventory must profile the shipped path, which projects one row."""

    import inspect

    source = inspect.getsource(inventory._child)
    assert "logits_rows" in source
    assert "args.logits_rows" in source
