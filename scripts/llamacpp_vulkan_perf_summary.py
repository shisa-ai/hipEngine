#!/usr/bin/env python3
"""Summarize llama.cpp Vulkan per-operation timestamp logs.

llama.cpp's Vulkan backend emits these logs when GGML_VK_PERF_LOGGER=1.
The logger prints one section per graph_compute call by default, so decode logs
can contain one warmup section followed by one section per generated token.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "hipengine.llamacpp_vulkan_perf_summary.v1"
CLASSIFIER_VERSION = "2-selected-matmul-first"

_OPERATION_RE = re.compile(
    r"^(?P<name>.+):\s+(?P<count>\d+)\s+x\s+"
    r"(?P<average>[0-9.eE+-]+)\s+us\s+=\s+"
    r"(?P<total>[0-9.eE+-]+)\s+us(?:\s+\([^)]*\))?$"
)
_TOTAL_RE = re.compile(r"^Total time:\s+(?P<total>[0-9.eE+-]+)\s+us\.$")


@dataclass(frozen=True)
class OperationTiming:
    name: str
    dispatches: int
    average_us: float
    total_us: float


@dataclass(frozen=True)
class TimingSection:
    operations: tuple[OperationTiming, ...]
    total_us: float

    @property
    def dispatches(self) -> int:
        return sum(operation.dispatches for operation in self.operations)

    @property
    def operation_total_us(self) -> float:
        return sum(operation.total_us for operation in self.operations)


def classify_operation(name: str) -> str:
    upper = name.upper()
    if "MUL_MAT_ID" in upper and "Q4_K" in upper:
        return "llama_selected_q4"
    if "MUL_MAT_ID" in upper and "Q5_K" in upper:
        return "llama_selected_q5"
    if "MUL_MAT_ID" in upper and "Q6_K" in upper:
        return "llama_selected_q6"
    if "MUL_MAT_ID" in upper and "Q5_1" in upper:
        return "llama_selected_q51"
    if "MUL_MAT_ID" in upper and "Q8_0" in upper:
        return "llama_selected_q8"
    if "MUL_MAT_ID" in upper:
        return "llama_selected_other"
    if "MUL_MAT_VEC Q6_K" in upper:
        return "llama_lm_head"
    if "MUL_MAT" in upper and "Q8_0" in upper:
        return "llama_dense_q8"
    if "MUL_MAT" in upper and "F32" in upper:
        return "llama_f32_matmul"
    if "GATED_DELTA_NET" in upper:
        return "llama_gdn"
    if "SSM_CONV" in upper:
        return "llama_linear_attn_conv"
    if "FLASH_ATTN" in upper:
        return "llama_flash_attn"
    if "RMS_NORM" in upper or upper.startswith("L2_NORM"):
        return "llama_norm"
    if "TOPK" in upper or "ARGSORT" in upper or upper.startswith("GET_ROWS"):
        return "llama_router_topk"
    if upper.startswith(("CONCAT", "CONT", "CPY", "SET_ROWS")):
        return "llama_copy_layout"
    if "SOFT_MAX" in upper or upper.startswith("SOFTMAX"):
        return "llama_softmax"
    if upper.startswith(
        (
            "ADD",
            "CLAMP",
            "DIV",
            "GLU",
            "MUL",
            "MULTI_ADD",
            "SCALE",
            "SIGMOID",
            "SILU",
            "SOFTPLUS",
            "SUM_ROWS",
        )
    ):
        return "llama_elementwise"
    if upper.startswith("ROPE"):
        return "llama_rope"
    return "other"


def parse_perf_log(path: Path) -> list[TimingSection]:
    sections: list[TimingSection] = []
    in_section = False
    operations: list[OperationTiming] = []

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line == "Vulkan Timings:":
            if in_section:
                raise ValueError("incomplete Vulkan timing section before next header")
            in_section = True
            operations = []
            continue
        if not in_section:
            continue

        total_match = _TOTAL_RE.match(line)
        if total_match:
            if not operations:
                raise ValueError("Vulkan timing section contains no operations")
            sections.append(TimingSection(tuple(operations), float(total_match.group("total"))))
            in_section = False
            operations = []
            continue

        operation_match = _OPERATION_RE.match(line)
        if operation_match:
            operations.append(
                OperationTiming(
                    name=operation_match.group("name"),
                    dispatches=int(operation_match.group("count")),
                    average_us=float(operation_match.group("average")),
                    total_us=float(operation_match.group("total")),
                )
            )

    if in_section:
        raise ValueError("incomplete Vulkan timing section at end of log")
    if not sections:
        raise ValueError(f"no Vulkan timing sections found in {path}")
    return sections


def _aggregate_rows(
    selected: Sequence[TimingSection],
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
    operations: dict[str, dict[str, float | int]] = defaultdict(lambda: {"dispatches": 0, "total_us": 0.0})
    families: dict[str, dict[str, float | int]] = defaultdict(lambda: {"dispatches": 0, "total_us": 0.0})
    for section in selected:
        for operation in section.operations:
            operation_row = operations[operation.name]
            operation_row["dispatches"] += operation.dispatches
            operation_row["total_us"] += operation.total_us

            family_row = families[classify_operation(operation.name)]
            family_row["dispatches"] += operation.dispatches
            family_row["total_us"] += operation.total_us
    return operations, families


def _format_rows(
    rows: dict[str, dict[str, float | int]],
    *,
    key_name: str,
    total_us: float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    output = []
    sorted_rows = sorted(rows.items(), key=lambda item: float(item[1]["total_us"]), reverse=True)
    if limit is not None:
        sorted_rows = sorted_rows[:limit]
    for name, row in sorted_rows:
        row_total_us = float(row["total_us"])
        dispatches = int(row["dispatches"])
        output.append(
            {
                key_name: name,
                "total_ms": row_total_us / 1000.0,
                "dispatches": dispatches,
                "avg_dispatch_us": row_total_us / dispatches if dispatches else 0.0,
                "share_of_total": row_total_us / total_us if total_us else 0.0,
            }
        )
    return output


def build_summary(
    log_path: Path,
    *,
    label: str,
    command: str | None,
    discard_first_sections: int,
    top: int,
    select_last_sections: int | None = None,
) -> dict[str, Any]:
    if discard_first_sections < 0:
        raise ValueError("discard_first_sections must be non-negative")
    if select_last_sections is not None and select_last_sections <= 0:
        raise ValueError("select_last_sections must be positive")
    if top <= 0:
        raise ValueError("top must be positive")

    sections = parse_perf_log(log_path)
    selected = sections[discard_first_sections:]
    if select_last_sections is not None:
        if len(selected) < select_last_sections:
            raise ValueError(
                f"requested last {select_last_sections} Vulkan timing sections, "
                f"but only {len(selected)} remain after discard"
            )
        selected = selected[-select_last_sections:]
    if not selected:
        raise ValueError("no selected Vulkan timing sections remain after discard")

    total_us = sum(section.total_us for section in selected)
    operation_line_total_us = sum(section.operation_total_us for section in selected)
    operations, families = _aggregate_rows(selected)
    return {
        "schema": SCHEMA,
        "classifier_version": CLASSIFIER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "command": command,
        "source_log": str(log_path),
        "sections_found": len(sections),
        "discard_first_sections": discard_first_sections,
        "select_last_sections": select_last_sections,
        "selected_section_count": len(selected),
        "total_gpu_ms": total_us / 1000.0,
        "avg_section_gpu_ms": total_us / 1000.0 / len(selected),
        "total_dispatches": sum(section.dispatches for section in selected),
        "section_gpu_ms": [section.total_us / 1000.0 for section in selected],
        "operation_line_total_ms": operation_line_total_us / 1000.0,
        "operation_vs_reported_total_delta_ms": (operation_line_total_us - total_us) / 1000.0,
        "families": _format_rows(families, key_name="family", total_us=total_us),
        "top_operations": _format_rows(operations, key_name="operation", total_us=total_us, limit=top),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="llama.cpp stderr containing Vulkan Timings sections")
    parser.add_argument("--json", required=True, type=Path, help="output summary JSON")
    parser.add_argument("--label", required=True)
    parser.add_argument("--command")
    parser.add_argument("--discard-first-sections", type=int, default=0)
    parser.add_argument(
        "--select-last-sections",
        type=int,
        help="after discarding, keep only the final N sections (for timed decode tokens after depth setup)",
    )
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_summary(
        args.log,
        label=args.label,
        command=args.command,
        discard_first_sections=args.discard_first_sections,
        top=args.top,
        select_last_sections=args.select_last_sections,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
