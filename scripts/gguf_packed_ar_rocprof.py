#!/usr/bin/env python3
"""Profile the observable c1/packed GGUF execution boundary.

The parent warm-builds both leaf workloads outside rocprofv3, then profiles one
synchronized steady decode transition for c1 and a declared packed c2, c4, or
c8 lane in separate cached-only children. The packed runtime manifest counts host
row loops, metadata/state movement, synchronizations, scalar fallbacks, and
graph replay. This script checks its route-dependent row-local launch count
against the trace and buckets all packed GPU work as ``exact_row_local`` or
``packed_native``.

Eager mode retains the C1/C2/C3 route census. Graph mode proves one captured
packed replay window, never throughput. Model load, prefill, packed-state import,
graph capture, diagnostic readback, and flush are excluded by one ROCTX marker
window.
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_NATIVE_C8_PHASE_BY_TARGET = {
    "gfx1100": "e2",
    "gfx1151": "e1",
}
_CORRECTNESS_GATE_BY_BACKEND = {
    "hip_gfx1100": (
        "benchmarks/results/"
        "2026-07-16-gfx1100-gguf-concurrency-b4-category-lifecycle.json"
    ),
    "hip_gfx1151": (
        "benchmarks/results/"
        "2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json"
    ),
}
SCHEMA = 1
MARKER_PREFIX = "hipengine_gguf_packed_c1_profile_c"
_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
_GDN_PREFILL_MODE_ENV = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
_PROFILER_CONCURRENCIES = frozenset(range(1, 9))


def _profiler_concurrency_supported(concurrency: int | None) -> bool:
    return concurrency is not None and int(concurrency) in _PROFILER_CONCURRENCIES


def _artifact_kind(
    target_arch: str,
    *,
    closure_level: str,
    packed_concurrency: int,
) -> str:
    target = str(target_arch)
    if packed_concurrency == 8 and closure_level == "c4":
        try:
            phase = _NATIVE_C8_PHASE_BY_TARGET[target]
        except KeyError as exc:
            raise ValueError(f"unsupported gfx11 profiler target: {target!r}") from exc
        suffix = f"{phase}_native_c8_graph_profiler_census"
    elif closure_level == "c4":
        suffix = f"c{int(packed_concurrency)}_graph_replay_census"
    elif closure_level == "c3":
        suffix = "c3_model_boundaries_census"
    elif closure_level == "c2":
        suffix = "c2_recurrent_census"
    else:
        suffix = "c1_hybrid_census"
    return f"{target}_gguf_concurrency_{suffix}"


def _correctness_gate(backend: str) -> str:
    try:
        return _CORRECTNESS_GATE_BY_BACKEND[str(backend)]
    except KeyError as exc:
        raise ValueError(f"unsupported gfx11 profiler backend: {backend!r}") from exc


@dataclass(frozen=True)
class KernelTraceRow:
    kernel: str
    duration_ns: int
    grid_y: int | None = None
    grid: tuple[int, int, int] | None = None
    workgroup: tuple[int, int, int] | None = None
    vgpr: int | None = None
    scratch_bytes: int | None = None
    lds_bytes: int | None = None
    start_ns: int | None = None
    end_ns: int | None = None


class _Roctx:
    def __init__(self) -> None:
        try:
            library = ctypes.CDLL("libroctx64.so")
        except OSError as exc:  # pragma: no cover - requires ROCm SDK
            raise RuntimeError(f"libroctx64.so is required for C1 marker windows: {exc}") from exc
        self._push = library.roctxRangePushA
        self._pop = library.roctxRangePop
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        self._push(name.encode("utf-8"))

    def pop(self) -> None:
        self._pop()


def _int_or_none(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _triple(row: Mapping[str, str], prefix: str) -> tuple[int, int, int] | None:
    values = tuple(_int_or_none(row.get(f"{prefix}_{axis}")) for axis in ("X", "Y", "Z"))
    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values)  # type: ignore[arg-type,return-value]


def _normalise_kernel(name: str) -> str:
    return re.sub(r"<[^>]*>", "<...>", str(name)).strip()


def classify_packed_execution_bucket(row: KernelTraceRow) -> str:
    """Classify current packed kernels by the known exact row boundary."""

    name = row.kernel.lower()
    scalar_only_substrings = (
        "linear_attn_conv_decode_lowp_kernel",
        "gdn_recurrent_rmsnorm_gate_lowp_kernel",
    )
    if any(part in name for part in scalar_only_substrings):
        return "exact_row_local"

    # These projection bodies serve both scalar and row-shaped decode. rocprof
    # exposes the independent row count in grid-Y, so kernel name alone is not
    # sufficient after C2 removes the host row loop.
    row_extent = (
        int(row.grid[1])
        if row.grid is not None
        else None if row.grid_y is None else int(row.grid_y)
    )
    row_shaped_substrings = (
        "q8_0_t16_dual_split_gemv_kernel",
        "dense_gemv_bf16_f32w_bf16_out_kernel",
    )
    if any(part in name for part in row_shaped_substrings) and row_extent in {None, 1}:
        return "exact_row_local"
    if (
        "q8_0_t16_gemv_kernel" in name
        and "float const*" in name
        and row_extent in {None, 1}
    ):
        return "exact_row_local"
    if "gguf_rmsnorm_bf16_f32_weight_kernel" in name:
        if row.grid is not None and row.workgroup is not None:
            if int(row.grid[0]) == int(row.workgroup[0]):
                return "exact_row_local"
        elif int(row.grid_y or 0) == 1:
            # Synthetic/unit rows may omit full launch geometry.
            return "exact_row_local"
    return "packed_native"


def classify_decode_kernel_family(kernel: str) -> str:
    """Bucket one steady GGUF decode kernel by its optimization boundary."""

    name = str(kernel).lower()
    if any(
        part in name
        for part in (
            "prepare_packed_decode_metadata",
            "commit_packed_decode_graph_step",
            "set_decode_position",
            "set_i64_scalar",
            "record_i64_scalar",
            "token_feedback",
            "copybuffer",
            "fillbuffer",
            "memcpy",
            "rocclr",
        )
    ):
        return "metadata_lifecycle"
    if "q6_k_t16_gemv" in name or "lm_head" in name or "argmax" in name:
        return "lm_head_sampler"
    if "embedding" in name:
        return "embedding"
    if "router" in name:
        return "moe_router"
    if any(part in name for part in ("linear_attn", "gdn", "ssm", "_conv_")):
        return "linear_attention_state"
    if any(
        part in name
        for part in (
            "paged_full_attn",
            "write_paged_kv",
            "head_rmsnorm_partial_rotary",
            "split_qgate",
            "full_attn_gate",
            "bf16_to_f32",
        )
    ):
        return "full_attention_core"
    if any(
        part in name
        for part in (
            "selected",
            "silu_mul",
            "weighted_sum",
            "shared_gate_combine",
        )
    ):
        return "moe_selected_combine"
    if any(part in name for part in ("rmsnorm", "residual", "bf16_add")):
        return "norm_residual"
    if any(
        part in name
        for part in (
            "gemv",
            "wmma_prefill",
            "f32_to_bf16",
        )
    ):
        return "dense_projection"
    return "other"


def summarize_decode_kernel_families(
    rows: Sequence[KernelTraceRow],
) -> dict[str, Any]:
    """Attribute dispatch count and diagnostic GPU time without hiding unknowns."""

    total_ns = sum(int(row.duration_ns) for row in rows)
    by_family: dict[str, list[KernelTraceRow]] = collections.defaultdict(list)
    for row in rows:
        by_family[classify_decode_kernel_family(row.kernel)].append(row)

    families: list[dict[str, Any]] = []
    for family, family_rows in sorted(
        by_family.items(),
        key=lambda item: (-sum(int(row.duration_ns) for row in item[1]), item[0]),
    ):
        duration_ns = sum(int(row.duration_ns) for row in family_rows)
        by_kernel: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for row in family_rows:
            kernel = _normalise_kernel(row.kernel)
            by_kernel[kernel][0] += 1
            by_kernel[kernel][1] += int(row.duration_ns)
        families.append(
            {
                "family": family,
                "dispatches": len(family_rows),
                "total_duration_ns": duration_ns,
                "total_duration_ms": duration_ns / 1.0e6,
                "share_of_gpu_duration": duration_ns / total_ns if total_ns else None,
                "kernels": [
                    {
                        "kernel": kernel,
                        "dispatches": values[0],
                        "total_duration_ns": values[1],
                    }
                    for kernel, values in sorted(
                        by_kernel.items(), key=lambda item: (-item[1][1], item[0])
                    )
                ],
            }
        )

    unclassified = by_family.get("other", [])
    return {
        "total_dispatches": len(rows),
        "total_duration_ns": total_ns,
        "families": families,
        "unclassified": {
            "dispatches": len(unclassified),
            "total_duration_ns": sum(int(row.duration_ns) for row in unclassified),
            "kernel_names": sorted({_normalise_kernel(row.kernel) for row in unclassified}),
        },
    }


def _summary(rows: Sequence[KernelTraceRow]) -> dict[str, Any]:
    dispatches = len(rows)
    total_ns = sum(int(row.duration_ns) for row in rows)
    by_kernel: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    shapes: dict[str, collections.Counter[tuple[Any, ...]]] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        key = _normalise_kernel(row.kernel)
        by_kernel[key][0] += 1
        by_kernel[key][1] += int(row.duration_ns)
        shapes[key][
            (
                row.grid,
                row.workgroup,
                row.vgpr,
                row.scratch_bytes,
                row.lds_bytes,
            )
        ] += 1
    top = [
        {
            "kernel": kernel,
            "dispatches": values[0],
            "total_duration_ns": values[1],
            "launch_shapes": [
                {
                    "count": count,
                    "grid": None if shape[0] is None else list(shape[0]),
                    "workgroup": None if shape[1] is None else list(shape[1]),
                    "vgpr": shape[2],
                    "scratch_bytes": shape[3],
                    "lds_bytes": shape[4],
                }
                for shape, count in sorted(
                    shapes[kernel].items(),
                    key=lambda item: (-item[1], repr(item[0])),
                )
            ],
        }
        for kernel, values in sorted(
            by_kernel.items(), key=lambda item: (-item[1][1], item[0])
        )
    ]
    return {
        "dispatches": dispatches,
        "total_duration_ns": total_ns,
        "total_duration_ms": total_ns / 1.0e6,
        "top_kernels": top,
    }


def _rows_matching(
    rows: Sequence[KernelTraceRow],
    substring: str,
) -> list[KernelTraceRow]:
    needle = str(substring).lower()
    return [row for row in rows if needle in row.kernel.lower()]


def _row_extent(row: KernelTraceRow) -> int | None:
    if row.grid is not None:
        return int(row.grid[1])
    return None if row.grid_y is None else int(row.grid_y)


def _all_row_extent(rows: Sequence[KernelTraceRow], expected: int) -> bool:
    return bool(rows) and all(_row_extent(row) == int(expected) for row in rows)


def _q6_rowtile_dispatch_count(rows: int, *, max_chunk: int = 6) -> int:
    """Mirror the exact LM-head row-tile lowering without hiding c1 tails."""

    remaining = int(rows)
    if remaining <= 0:
        raise ValueError("rows must be positive")
    if int(max_chunk) < 2:
        raise ValueError("max_chunk must be at least two")
    dispatches = 0
    while remaining > 0:
        take = min(remaining, int(max_chunk))
        if remaining - take == 1:
            take -= 1
        if take < 2:
            raise ValueError("Q6 row-tile lowering would create a c1 tail")
        remaining -= take
        dispatches += 1
    return dispatches


def build_c3_family_census(
    c4_rows: Sequence[KernelTraceRow],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the C3 c-aware families and steady movement from one c4 trace."""

    row_count = int(manifest["rows"])
    layers = manifest["layers"]
    full_layers = int(layers["full_attention"])  # type: ignore[index]
    total_layers = int(layers["total"])  # type: ignore[index]
    families = manifest["layer_families"]
    selected_lanes = int(families["moe_ffn"]["selected_lanes"])  # type: ignore[index]

    full_context = _rows_matching(
        c4_rows,
        "qwen35_paged_full_attn_decode_context_tensor_batch_kernel",
    )
    full_kv_write = _rows_matching(
        c4_rows,
        "qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel",
    )
    full_passed = (
        len(full_context) == full_layers
        and len(full_kv_write) == full_layers
        and (full_layers == 0 or _all_row_extent(full_context, row_count))
        and (full_layers == 0 or _all_row_extent(full_kv_write, row_count))
    )

    selected_gate_up = _rows_matching(
        c4_rows,
        "selected_dual_direct_gemv_kernel",
    )
    selected_down = _rows_matching(
        c4_rows,
        "qk_t16_selected_direct_gemv_kernel",
    )
    moe_combine = _rows_matching(
        c4_rows,
        "weighted_sum_shared_gate_combine_residual_batch_out_kernel",
    )
    moe_passed = (
        len(selected_gate_up) == total_layers
        and len(selected_down) == total_layers
        and len(moe_combine) == total_layers
        and _all_row_extent(selected_gate_up, selected_lanes)
        and _all_row_extent(selected_down, selected_lanes)
        and _all_row_extent(moe_combine, row_count)
    )

    lm_head_path = str(manifest["lm_head_decode_path"])
    if lm_head_path == "q6_rowtile_f32_logits":
        lm_head_rows = _rows_matching(c4_rows, "q6_k_t16_gemv_rowtile")
        expected_lm_head_dispatches = _q6_rowtile_dispatch_count(row_count)
    elif lm_head_path == "direct_top1_rows":
        lm_head_rows = _rows_matching(c4_rows, "top1_gather")
        expected_lm_head_dispatches = 1
    else:
        lm_head_rows = [
            row
            for row in c4_rows
            if "gemv_kernel" in row.kernel.lower()
            and _row_extent(row) == row_count
        ]
        expected_lm_head_dispatches = 1
    sampler_path = str(manifest["sampler_decode_path"])
    argmax_stage1 = _rows_matching(c4_rows, "argmax_rows_stage1_i32_kernel")
    argmax_stage2 = _rows_matching(c4_rows, "argmax_rows_stage2_i32_kernel")
    sampler_passed = (
        len(lm_head_rows) == expected_lm_head_dispatches
        and (
            sampler_path == "fused_top1_i32_rows"
            or (
                len(argmax_stage1) == 1
                and len(argmax_stage2) == 1
                and _all_row_extent(argmax_stage1, row_count)
            )
        )
        and manifest["layer_families"]["lm_head"]["full_vocab_host_readback"] is False  # type: ignore[index]
    )

    movement = manifest["host_device_movement"]
    expected_copies = (
        int(movement["host_to_device_total_copies"])  # type: ignore[index]
        + int(movement["device_to_device_state_import_copies"])  # type: ignore[index]
        + int(movement["device_to_device_state_scatter_copies"])  # type: ignore[index]
        + int(movement["device_to_host_vector_copies"])  # type: ignore[index]
    )
    copy_rows = _rows_matching(c4_rows, "__amd_rocclr_copybuffer")
    metadata_prepare_rows = _rows_matching(
        c4_rows,
        "prepare_packed_decode_metadata",
    )
    expected_metadata_launches = int(movement["device_metadata_prepare_launches"])  # type: ignore[index]
    movement_passed = (
        len(copy_rows) == expected_copies
        and len(metadata_prepare_rows) == expected_metadata_launches
    )

    return {
        "route_check_passed": bool(
            full_passed and moe_passed and sampler_passed and movement_passed
        ),
        "full_attention": {
            "passed": full_passed,
            "path": manifest["full_attention_decode_path"],
            "context_dispatches": len(full_context),
            "kv_write_dispatches": len(full_kv_write),
            "row_extent": row_count,
        },
        "moe_ffn": {
            "passed": moe_passed,
            "path": manifest["moe_decode_path"],
            "selected_lanes": selected_lanes,
            "selected_gate_up_dispatches": len(selected_gate_up),
            "selected_down_dispatches": len(selected_down),
            "combine_dispatches": len(moe_combine),
        },
        "lm_head_sampler": {
            "passed": sampler_passed,
            "lm_head_path": lm_head_path,
            "sampler_path": sampler_path,
            "expected_lm_head_dispatches": expected_lm_head_dispatches,
            "lm_head_dispatches": len(lm_head_rows),
            "argmax_stage1_dispatches": len(argmax_stage1),
            "argmax_stage2_dispatches": len(argmax_stage2),
            "full_vocab_host_readback": False,
        },
        "host_device_movement": {
            "passed": movement_passed,
            "expected_copy_dispatches": expected_copies,
            "observed_copy_dispatches": len(copy_rows),
            "expected_metadata_prepare_dispatches": expected_metadata_launches,
            "observed_metadata_prepare_dispatches": len(metadata_prepare_rows),
        },
    }


def build_execution_census(
    c1_rows: Sequence[KernelTraceRow],
    packed_rows: Sequence[KernelTraceRow],
    *,
    manifest: Mapping[str, Any],
    packed_concurrency: int = 4,
) -> dict[str, Any]:
    expected = int(
        manifest["model_step"]["expected_exact_row_local_kernel_launches"]  # type: ignore[index]
    )
    bucket_rows: dict[str, list[KernelTraceRow]] = {
        "exact_row_local": [],
        "packed_native": [],
    }
    for row in packed_rows:
        bucket_rows[classify_packed_execution_bucket(row)].append(row)
    buckets = {name: _summary(rows) for name, rows in bucket_rows.items()}
    packed_total_ns = sum(int(bucket["total_duration_ns"]) for bucket in buckets.values())
    share_key = f"share_of_c{int(packed_concurrency)}_gpu_duration"
    for bucket in buckets.values():
        bucket[share_key] = (
            float(bucket["total_duration_ns"]) / packed_total_ns
            if packed_total_ns > 0
            else None
        )
    observed = int(buckets["exact_row_local"]["dispatches"])
    route_check_passed = (
        observed == expected
        and int(buckets["packed_native"]["dispatches"]) > 0
        and int(manifest.get("scalar_fallbacks", -1)) == 0
        and int(manifest.get("model_step", {}).get("complete_c1_session_replays", -1)) == 0
    )
    packed_label = f"c{int(packed_concurrency)}"
    return {
        "route_check_passed": route_check_passed,
        "packed_concurrency": int(packed_concurrency),
        "c3_family_census": build_c3_family_census(packed_rows, manifest=manifest),
        "family_attribution": {
            "c1": summarize_decode_kernel_families(c1_rows),
            packed_label: summarize_decode_kernel_families(packed_rows),
        },
        "c1_reference": _summary(c1_rows),
        packed_label: {
            "all": _summary(packed_rows),
            "buckets": buckets,
            "expected_exact_row_local_dispatches": expected,
            "observed_exact_row_local_dispatches": observed,
        },
    }


def execution_census_closure_level(
    manifest: Mapping[str, Any],
    census: Mapping[str, Any],
) -> str:
    """Return the highest concurrency roadmap boundary proven by the census."""

    c2_recurrent_closed = (
        manifest.get("linear_attention_decode_path") == "indexed_batch"
        and manifest.get("model_step", {}).get("host_model_row_loop_sites") == 0
        and manifest.get("model_step", {}).get("expected_exact_row_local_kernel_launches") == 0
    )
    movement = manifest.get("host_device_movement", {})
    c3_model_boundaries_closed = (
        c2_recurrent_closed
        and census.get("c3_family_census", {}).get("route_check_passed") is True
        and manifest.get("metadata_prepare_path")
        in {"device_positions_persistent", "device_prepare_persistent"}
        and movement.get("host_to_device_metadata_copies") == 0
        and movement.get("device_metadata_prepare_launches") == 1
    )
    graph = manifest.get("graph", {})
    c4_graph_replay_closed = (
        c3_model_boundaries_closed
        and manifest.get("mode") == "decode_graph_replay"
        and graph.get("captured") is True
        and int(graph.get("replay_count", 0)) > 0
        and int(graph.get("replayed_steps", 0)) > 0
        and movement.get("host_to_device_total_copies") == 0
        and movement.get("device_to_host_vector_copies") == 0
        and manifest.get("synchronizations") == 0
    )
    if c4_graph_replay_closed:
        return "c4"
    if c3_model_boundaries_closed:
        return "c3"
    if c2_recurrent_closed:
        return "c2"
    return "c1"


def _read_kernel_csv(path: Path) -> list[KernelTraceRow]:
    result: list[KernelTraceRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            start = _int_or_none(raw.get("Start_Timestamp"))
            end = _int_or_none(raw.get("End_Timestamp"))
            if start is None or end is None or end < start:
                continue
            kernel = str(
                raw.get("Kernel_Name")
                or raw.get("KernelName")
                or raw.get("Name")
                or ""
            ).strip()
            if not kernel:
                continue
            grid = _triple(raw, "Grid_Size")
            result.append(
                KernelTraceRow(
                    kernel=kernel,
                    duration_ns=end - start,
                    grid_y=None if grid is None else int(grid[1]),
                    grid=grid,
                    workgroup=_triple(raw, "Workgroup_Size"),
                    vgpr=_int_or_none(raw.get("VGPR_Count")),
                    scratch_bytes=_int_or_none(raw.get("Scratch_Size")),
                    lds_bytes=_int_or_none(raw.get("LDS_Block_Size")),
                    start_ns=start,
                    end_ns=end,
                )
            )
    return result


def _read_marker_window(path: Path, marker_name: str) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(
                row.get("Function")
                or row.get("Name")
                or row.get("Message")
                or ""
            )
            if marker_name not in name:
                continue
            start = _int_or_none(row.get("Start_Timestamp"))
            end = _int_or_none(row.get("End_Timestamp"))
            if start is not None and end is not None and end >= start:
                matches.append((start, end))
    if len(matches) != 1:
        raise ValueError(f"expected one marker window {marker_name!r}, observed {len(matches)}")
    return matches[0]


def _filter_window(rows: Sequence[KernelTraceRow], window: tuple[int, int]) -> list[KernelTraceRow]:
    start, end = window
    return [
        row
        for row in rows
        if row.start_ns is not None
        and row.end_ns is not None
        and int(row.start_ns) >= start
        and int(row.end_ns) <= end
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {root}, observed {matches}")
    return matches[0]


def _optional_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    if len(matches) > 1:
        raise ValueError(f"expected at most one {pattern} under {root}, observed {matches}")
    return matches[0] if matches else None


def _sum_trace_time(path: Path | None, window: tuple[int, int]) -> tuple[int, int]:
    """Count and sum on-device durations of traced rows inside the marker window."""
    if path is None or not path.exists():
        return (0, 0)
    ws, we = window
    count = 0
    total = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = _int_or_none(row.get("Start_Timestamp"))
            end = _int_or_none(row.get("End_Timestamp"))
            if start is None or end is None or end < start:
                continue
            if start >= ws and end <= we:
                count += 1
                total += end - start
    return (count, total)


def _timeline_accounting(
    rows: Sequence[KernelTraceRow],
    window: tuple[int, int],
    hip_api_csv: Path | None,
    memory_copy_csv: Path | None,
) -> dict[str, int]:
    """Per-lane wall accounting that does not double-count overlapping kernels.

    kernel_interval_union is the distinct marker-window time covered by at least
    one kernel (a true busy-time lower bound under multi-stream overlap), and
    uncovered_ns is marker wall minus that union. Summed kernel duration is
    reported separately and must NOT be treated as wall coverage.
    """
    ws, we = window
    wall = we - ws
    kern_sum = 0
    union = 0
    if rows:
        kern_sum = sum(int(r.duration_ns) for r in rows)
        ivs = sorted((int(r.start_ns), int(r.end_ns)) for r in rows)
        cur_s, cur_e = None, None
        for s, e in ivs:
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                union += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            union += cur_e - cur_s
    hip_count, hip_time = _sum_trace_time(hip_api_csv, window)
    mem_count, mem_time = _sum_trace_time(memory_copy_csv, window)
    return {
        "marker_wall_ns": wall,
        "kernel_sum_ns": kern_sum,
        "kernel_interval_union_ns": union,
        "uncovered_ns": wall - union,
        "kernel_count": len(rows),
        "hip_api_calls": hip_count,
        "hip_api_time_ns": hip_time,
        "memory_copy_ops": mem_count,
        "memory_copy_time_ns": mem_time,
    }


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.expanduser().read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty compiler-version file: {path}")
    return text


def _marker_name(concurrency: int) -> str:
    return f"{MARKER_PREFIX}{int(concurrency)}_steady_decode_step"


def _run_c1_child(session: Any, args: argparse.Namespace, marker: _Roctx | None) -> dict[str, Any]:
    prompt = [int(args.prompt_token_id)] * int(args.prompt_length)
    first = session.prefill(
        prompt,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    current = int(first.token_id)
    warm = session.step(current, return_logits=False)
    current = int(warm.token_id)
    session.runtime.device_synchronize()
    if marker is not None:
        marker.push(_marker_name(1))
    try:
        result = session.step(current, return_logits=False)
        session.runtime.device_synchronize()
    finally:
        if marker is not None:
            marker.pop()
    observed = [int(first.token_id), int(warm.token_id), int(result.token_id)]
    return {
        "concurrency": 1,
        "generated_token_ids": observed,
        "all_tokens_exact": all(token == int(args.expected_token_id) for token in observed),
        "execution_manifest": None,
    }


def _run_packed_child(owner: Any, sessions: Sequence[Any], args: argparse.Namespace, marker: _Roctx | None) -> dict[str, Any]:
    prompts = tuple(
        tuple([int(args.prompt_token_id)] * int(args.prompt_length))
        for _ in sessions
    )
    session_tuple = tuple(sessions)
    first = owner.prefill_batch_native(prompts, sessions=session_tuple, return_logits=False)
    current = [int(result.token_id) for result in first]
    warm = owner.step_batch_native(
        current,
        sessions=session_tuple,
        positions=[int(session.position) for session in sessions],
        return_logits=False,
        scatter_state=False,
    )
    current = [int(result.token_id) for result in warm]
    owner.runtime.device_synchronize()

    if str(args.decode_mode) == "graph":
        graph = owner.capture_packed_decode_graph(
            current,
            sessions=session_tuple,
            steps_per_replay=1,
            max_replay_steps=1,
            record_steps=1,
            submission_transport=str(args.submission_transport),
        )
        try:
            if marker is not None:
                marker.push(_marker_name(len(session_tuple)))
            try:
                graph.replay(1)
            finally:
                if marker is not None:
                    marker.pop()
            final = graph.read_generated_token_ids(1)[0]
            manifest = dict(graph.execution_manifest)
            flushed = bool(graph.flush_packed_state())
        finally:
            graph.close()
    else:
        if marker is not None:
            marker.push(_marker_name(len(session_tuple)))
        try:
            result = owner.step_batch_native(
                current,
                sessions=session_tuple,
                positions=[int(session.position) for session in sessions],
                return_logits=False,
                scatter_state=False,
            )
            owner.runtime.device_synchronize()
        finally:
            if marker is not None:
                marker.pop()
        final = [int(item.token_id) for item in result]
        manifest = dict(owner.last_packed_execution_manifest)
        flushed = bool(owner.flush_packed_decode_state())

    observed = [*[int(item.token_id) for item in first], *current, *final]
    return {
        "concurrency": len(session_tuple),
        "decode_mode": str(args.decode_mode),
        "submission_transport": str(args.submission_transport),
        "prefill_plan": dict(owner.last_packed_prefill_plan),
        "warmup_token_ids": current,
        "profile_token_ids": final,
        "generated_token_ids": observed,
        "all_tokens_exact": all(token == int(args.expected_token_id) for token in observed),
        "execution_manifest": manifest,
        "flush_executed": flushed,
    }


def _run_child(args: argparse.Namespace) -> int:
    os.environ.setdefault(
        "HIPENGINE_HIP_ARCH",
        "gfx1151" if str(args.backend) == "hip_gfx1151" else "gfx1100",
    )
    os.environ[_CAPTURE_PREFILL_GDN_ENV] = "1"
    os.environ[_GDN_PREFILL_MODE_ENV] = "exact"
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    compiler_version = _read_compiler_version(args.compiler_version_file)
    max_sequence_length = int(args.prompt_length) + 8
    marker = _Roctx() if str(args.child_mode) == "profile" else None
    kwargs = {
        "backend": str(args.backend),
        "max_sequence_length": max_sequence_length,
        "compiler_version": compiler_version,
        "require_cached_build": bool(args.require_cached_build),
        "use_wmma_prefill": True,
        "use_gemv_decode": True,
    }
    with ExitStack() as stack:
        owner = stack.enter_context(Qwen35GGUFResidentSession(args.model, **kwargs))
        if int(args.concurrency) == 1 and str(args.decode_mode) == "eager":
            payload = _run_c1_child(owner, args, marker)
        elif _profiler_concurrency_supported(int(args.concurrency)):
            if owner.runner is None:
                raise RuntimeError("GGUF owner did not materialize a shared runner")
            sessions = [owner]
            for _ in range(int(args.concurrency) - 1):
                sessions.append(
                    stack.enter_context(
                        Qwen35GGUFResidentSession(
                            args.model,
                            runtime=owner.runtime,
                            shared_runner=owner.runner,
                            **kwargs,
                        )
                    )
                )
            payload = _run_packed_child(owner, sessions, args, marker)
        else:
            raise ValueError("GGUF profiler child supports only c1-c8 (graph c1 routes through packed graph)")
        payload.update(
            {
                "schema": 1,
                "kind": "gguf_packed_ar_rocprof_child",
                "child_mode": str(args.child_mode),
                "backend": str(owner.backend),
                "target_arch": str(owner.runner.target_arch),
                "decode_mode": str(args.decode_mode),
                "submission_transport": str(args.submission_transport),
                "require_cached_build": bool(args.require_cached_build),
            }
        )
    args.child_json.parent.mkdir(parents=True, exist_ok=True)
    args.child_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["all_tokens_exact"]:
        raise RuntimeError("profile child token correctness failed")
    return 0


def _roctx_candidates_for_prefix(prefix: Path) -> list[Path]:
    """SDK ROCTX libraries installed under a python prefix, whatever python3.x it uses.

    Marker tracing needs the real SDK shim, not legacy libroctx64. The matching copy usually lives in
    the environment that provides the profiler, which on multi-env hosts is not the venv running the
    traced process (docs/KERNELS.md "MTP verify profiling traps", trap 1). Globbed so versioned
    sonames (`.so.1.3.2`) are found too.
    """
    found: list[Path] = []
    for package in ("_rocm_sdk_core", "_rocm_sdk_devel"):
        found.extend(
            sorted(
                (prefix / "lib").glob(
                    f"python3*/site-packages/{package}/lib/librocprofiler-sdk-roctx.so*"
                ),
                reverse=True,  # newest python3.x first when an env carries several
            )
        )
    return found


def _profiler_prefix() -> Path | None:
    """Prefix of the rocprofv3 that will actually run, or None when it is not on PATH."""
    executable = shutil.which("rocprofv3")
    if not executable:
        return None
    resolved = Path(executable).resolve()
    # bin/rocprofv3 -> <prefix>/bin/rocprofv3; keep the parent only when the layout looks like one.
    return resolved.parents[1] if resolved.parent.name == "bin" else None


def _default_roctx_sdk() -> Path:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    names = ("librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so")
    preferred = [
        Path(root) / "lib" / python_dir / "site-packages" / pkg / "lib" / name
        for root in dict.fromkeys((sys.prefix, sys.base_prefix))
        for pkg in ("_rocm_sdk_core", "_rocm_sdk_devel")
        for name in names
    ]
    profiler_prefix = _profiler_prefix()
    candidates = (
        _roctx_candidates_for_prefix(profiler_prefix)
        if profiler_prefix is not None
        else []
    )
    candidates += preferred
    candidates += [Path("/opt/rocm/lib") / name for name in names]
    candidates += [
        Path("/opt/rocm/lib/libroctx64.so.4"),
        Path("/opt/rocm/lib/libroctx64.so"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred[0]


def _prepare_roctx_override(sdk_path: Path, raw_root: Path) -> tuple[Path, tuple[Path, ...]]:
    if not sdk_path.exists():
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override = raw_root / "roctx-override"
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    dependencies = tuple(path for path in (sdk_path.parent, Path(sys.prefix) / "lib") if path.exists())
    return override, dependencies


def _child_command(
    args: argparse.Namespace,
    *,
    mode: str,
    concurrency: int,
    child_json: Path,
    require_cached: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-mode",
        str(mode),
        "--concurrency",
        str(concurrency),
        "--model",
        str(args.model),
        "--backend",
        str(args.backend),
        "--decode-mode",
        str(args.decode_mode),
        "--submission-transport",
        str(args.submission_transport),
        "--prompt-length",
        str(args.prompt_length),
        "--prompt-token-id",
        str(args.prompt_token_id),
        "--expected-token-id",
        str(args.expected_token_id),
        "--child-json",
        str(child_json),
    ]
    if args.compiler_version_file is not None:
        command.extend(["--compiler-version-file", str(args.compiler_version_file)])
    if require_cached:
        command.append("--require-cached-build")
    return command


def _run_checked(command: Sequence[str], *, env: Mapping[str, str], stdout: Path, stderr: Path) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out_handle, stderr.open("w", encoding="utf-8") as err_handle:
        result = subprocess.run(list(command), cwd=REPO_ROOT, env=dict(env), stdout=out_handle, stderr=err_handle)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}; stderr={stderr}"
        )


def _profile_one(
    args: argparse.Namespace,
    *,
    concurrency: int,
    raw_root: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    lane = raw_root / f"c{concurrency}"
    child_json = lane / "child.json"
    trace_dir = lane / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    child = _child_command(
        args,
        mode="profile",
        concurrency=concurrency,
        child_json=child_json,
        require_cached=True,
    )
    command = [
        str(args.rocprofv3),
        "--kernel-trace",
        "--marker-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--output-format",
        "csv",
        "-d",
        str(trace_dir),
        "--",
        *child,
    ]
    _run_checked(command, env=env, stdout=lane / "stdout.log", stderr=lane / "stderr.log")
    payload = json.loads(child_json.read_text(encoding="utf-8"))
    if payload.get("all_tokens_exact") is not True:
        raise ValueError(f"c{concurrency} child token correctness failed")
    if payload.get("require_cached_build") is not True:
        raise ValueError(f"c{concurrency} profiled child did not require cached builds")
    kernel_csv = _single_file(trace_dir, "*_kernel_trace.csv")
    marker_csv = _single_file(trace_dir, "*_marker_api_trace.csv")
    hip_api_csv = _optional_file(trace_dir, "*_hip_api_trace.csv")
    memory_copy_csv = _optional_file(trace_dir, "*_memory_copy_trace.csv")
    window = _read_marker_window(marker_csv, _marker_name(concurrency))
    all_rows = _read_kernel_csv(kernel_csv)
    selected = _filter_window(all_rows, window)
    if not selected:
        raise ValueError(f"c{concurrency} marker window contains no kernels")
    accounting = _timeline_accounting(selected, window, hip_api_csv, memory_copy_csv)
    return {
        "command": command,
        "child": payload,
        "rows": selected,
        "timeline_accounting": accounting,
        "raw_trace": {
            "kernel_csv": str(kernel_csv),
            "kernel_csv_sha256": _sha256(kernel_csv),
            "marker_csv": str(marker_csv),
            "marker_csv_sha256": _sha256(marker_csv),
            "hip_api_csv": str(hip_api_csv) if hip_api_csv else None,
            "memory_copy_csv": str(memory_copy_csv) if memory_copy_csv else None,
            "whole_trace_dispatches": len(all_rows),
            "selected_dispatches": len(selected),
            "marker_start_ns": window[0],
            "marker_end_ns": window[1],
        },
    }


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    raw_root = Path(args.raw_root).expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("HIPENGINE_HIP_ARCH", "gfx1151" if args.backend == "hip_gfx1151" else "gfx1100")
    env[_CAPTURE_PREFILL_GDN_ENV] = "1"
    env[_GDN_PREFILL_MODE_ENV] = "exact"
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    packed_concurrency = int(args.packed_concurrency)
    packed_label = f"c{packed_concurrency}"
    warmbuild_commands: list[list[str]] = []
    if not args.skip_warmbuild:
        for concurrency in (1, packed_concurrency):
            lane = raw_root / f"warmbuild-c{concurrency}"
            command = _child_command(
                args,
                mode="warmbuild",
                concurrency=concurrency,
                child_json=lane / "child.json",
                require_cached=False,
            )
            warmbuild_commands.append(command)
            _run_checked(command, env=env, stdout=lane / "stdout.log", stderr=lane / "stderr.log")

    override, dependencies = _prepare_roctx_override(Path(args.roctx_sdk), raw_root)
    profile_env = env.copy()
    prefix = os.pathsep.join([str(override), *(str(path) for path in dependencies)])
    profile_env["LD_LIBRARY_PATH"] = f"{prefix}:{profile_env.get('LD_LIBRARY_PATH', '')}"

    c1 = _profile_one(args, concurrency=1, raw_root=raw_root, env=profile_env)
    packed = _profile_one(
        args,
        concurrency=packed_concurrency,
        raw_root=raw_root,
        env=profile_env,
    )
    manifest = packed["child"].get("execution_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{packed_label} child did not emit an execution manifest")
    census = build_execution_census(
        c1["rows"],
        packed["rows"],
        manifest=manifest,
        packed_concurrency=packed_concurrency,
    )
    closure_level = execution_census_closure_level(manifest, census)
    physical_shape_exact = bool(
        manifest.get("rows") == packed_concurrency
        and manifest.get("physical_rows") == packed_concurrency
        and manifest.get("active_rows") == packed_concurrency
        and manifest.get("active_mask") == [True] * packed_concurrency
    )
    passed = (
        census["route_check_passed"] is True
        and c1["child"]["all_tokens_exact"] is True
        and packed["child"]["all_tokens_exact"] is True
        and physical_shape_exact
        and manifest.get("steady_packed_state_reused") is True
        and manifest.get("host_device_movement", {}).get("device_to_device_state_import_copies") == 0
        and manifest.get("host_device_movement", {}).get("device_to_device_state_scatter_copies") == 0
        and (str(args.decode_mode) != "graph" or closure_level == "c4")
    )

    target_arch = str(packed["child"]["target_arch"])
    command = [sys.executable, "scripts/gguf_packed_ar_rocprof.py", *sys.argv[1:]]
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(packed["child"]["backend"]),
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": profile_env.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_COMPILER_VERSION_FILE": profile_env.get("HIPENGINE_COMPILER_VERSION_FILE"),
            "HIP_VISIBLE_DEVICES": profile_env.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": profile_env.get("ROCR_VISIBLE_DEVICES"),
            "GPU_MAX_HW_QUEUES": profile_env.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
        },
        build_profile=f"cached_only_paired_c1_{packed_label}_leaf_rocprof",
        timing_protocol=(
            f"single_synchronized_{args.decode_mode}_steady_decode_marker_window_nonperformance"
        ),
        warmups=1,
        repetitions=1,
        profiler={
            "kind": "rocprofv3_kernel_marker_hipapi_memcopy_trace",
            "c1_command": c1["command"],
            f"{packed_label}_command": packed["command"],
        },
        hipcc_version=_read_compiler_version(args.compiler_version_file),
    )
    return {
        "schema": SCHEMA,
        "kind": _artifact_kind(
            target_arch,
            closure_level=closure_level,
            packed_concurrency=packed_concurrency,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "native_c8_graph_profiler_census_complete"
            if passed and packed_concurrency == 8 and closure_level == "c4"
            else f"c{packed_concurrency}_graph_replay_census_complete"
            if passed and closure_level == "c4"
            else "c3_model_boundaries_census_complete"
            if passed and closure_level == "c3"
            else "c2_recurrent_census_complete"
            if passed and closure_level == "c2"
            else "c1_census_complete"
            if passed
            else "failed"
        ),
        "passed": passed,
        "performance_claim": False,
        "claim_level": (
            "native_c8_graph_replay_profiler_census"
            if packed_concurrency == 8 and closure_level == "c4"
            else f"native_c{packed_concurrency}_graph_replay_profiler_census"
            if closure_level == "c4"
            else "exact_hybrid_model_boundaries_closed_profiler_census"
            if closure_level == "c3"
            else "exact_hybrid_recurrent_closed_profiler_census"
            if closure_level == "c2"
            else "exact_hybrid_profiler_census"
        ),
        "workload": {
            "model": str(model),
            "quant": "Q4_K_M",
            "kv_dtype": "bf16",
            "backend": str(args.backend),
            "prompt_tokens_per_row": int(args.prompt_length),
            "profiled_decode_transitions": 1,
            "c1_rows": 1,
            f"{packed_label}_rows": packed_concurrency,
            "physical_bucket_width": packed_concurrency,
            "active_mask": [True] * packed_concurrency,
            "c1_decode_mode": str(args.decode_mode),
            f"{packed_label}_decode_mode": str(args.decode_mode),
            "submission_transport": str(args.submission_transport),
            "sampling": "greedy_top1",
            "speculative_decode": False,
        },
        "correctness": {
            "c1_all_tokens_exact": c1["child"]["all_tokens_exact"],
            f"{packed_label}_all_tokens_exact": packed["child"]["all_tokens_exact"],
            "physical_shape_exact": physical_shape_exact,
            "expected_token_id": int(args.expected_token_id),
            "c1_generated_token_ids": c1["child"]["generated_token_ids"],
            f"{packed_label}_warmup_token_ids": packed["child"]["warmup_token_ids"],
            f"{packed_label}_profile_token_ids": packed["child"]["profile_token_ids"],
            "phase_b_gate": _correctness_gate(str(args.backend)),
        },
        "execution_manifest": dict(manifest),
        "profiler_census": census,
        "raw_traces": {
            "c1": c1["raw_trace"],
            packed_label: packed["raw_trace"],
        },
        "timeline_accounting": {
            "c1": c1["timeline_accounting"],
            packed_label: packed["timeline_accounting"],
        },
        "commands": {
            "parent": command,
            "warmbuild": warmbuild_commands,
            "c1_profiler": c1["command"],
            f"{packed_label}_profiler": packed["command"],
        },
        "provenance": provenance,
        "limitations": [
            "One synchronized steady decode transition is a route census, not a throughput sample.",
            (
                (
                    f"{_NATIVE_C8_PHASE_BY_TARGET[target_arch].upper()} proves one real native-c8 graph replay with zero undeclared c1 work or "
                    "steady transfer; retained scaling and d128 equality are joined separately."
                    if packed_concurrency == 8
                    else f"Physical c{packed_concurrency} proves one real graph replay with zero "
                    "undeclared c1 work or steady transfer; native/scaling promotion still requires "
                    "the d128 and direct performance packet."
                )
                if closure_level == "c4"
                else "C3 closes the declared per-row model and metadata boundaries, but the route remains "
                "exact_hybrid until the C4 replay, equality, and scaling gates complete."
                if closure_level == "c3"
                else "C2 closes the recurrent linear-attention row loop, but the route remains "
                "exact_hybrid until the later C3/C4 correctness, replay, and scaling gates complete."
                if closure_level == "c2"
                else f"The c{packed_concurrency} route remains exact_hybrid because recurrent linear "
                "attention replays a c1-exact row subgraph."
            ),
            "Profiler instrumentation and marker synchronization make all durations diagnostic only.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--decode-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--submission-transport",
        choices=("hipgraph", "aql"),
        default="hipgraph",
        help="Graph replay transport; direct AQL exposes inner dispatches to rocprofv3.",
    )
    parser.add_argument(
        "--packed-concurrency",
        type=int,
        choices=tuple(range(2, 9)),
        default=4,
        help="Physical packed lane c2-c8 to compare against c1 (default: c4).",
    )
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-gfx11-c1-census"))
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--skip-warmbuild", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child-mode", choices=("warmbuild", "profile"), help=argparse.SUPPRESS)
    parser.add_argument("--concurrency", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--require-cached-build", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.prompt_length) <= 0 or int(args.prompt_length) + 2 >= 1024:
        raise ValueError("prompt-length must be positive and leave two decode positions below 1024")
    if args.submission_transport != "hipgraph" and (
        args.backend != "hip_gfx1100" or args.decode_mode != "graph"
    ):
        raise ValueError("direct AQL profiling requires hip_gfx1100 graph decode")
    if args.child_mode is not None:
        if not _profiler_concurrency_supported(args.concurrency) or args.child_json is None:
            raise ValueError(
                "child mode requires --concurrency in 1..8 and --child-json"
            )
        return _run_child(args)
    payload = _run_parent(args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
