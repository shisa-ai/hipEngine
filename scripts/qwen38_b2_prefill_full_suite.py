#!/usr/bin/env python3
"""Summarize the B2 input-F16 prefill full-suite screen and kernel trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("passed") is not True:
        raise ValueError(f"incomplete benchmark arm: {path}")
    return payload


def _ids(payload: dict[str, Any]) -> dict[tuple[int, str, str], list[list[int]]]:
    result: dict[tuple[int, str, str], list[list[int]]] = {}
    for cell in payload["cells"]:
        for arm in ("ar", "mtp"):
            result[(int(cell["width"]), str(cell["prompt_id"]), arm)] = [
                [int(token) for token in row["generated_ids"]]
                for row in cell[arm]["rows"]
            ]
    return result


def _id_comparison(
    left: dict[str, Any], right: dict[str, Any], *, max_details: int = 4
) -> dict[str, Any]:
    left_ids = _ids(left)
    right_ids = _ids(right)
    if left_ids.keys() != right_ids.keys():
        raise ValueError("benchmark arms do not contain the same ID cells")
    mismatches = []
    for (width, prompt_id, arm), expected in left_ids.items():
        observed = right_ids[(width, prompt_id, arm)]
        if expected != observed:
            mismatches.append(
                {
                    "width": width,
                    "prompt_id": prompt_id,
                    "arm": arm,
                    "left": expected,
                    "right": observed,
                }
            )
    return {
        "compared_arm_cells": len(left_ids),
        "equal_arm_cells": len(left_ids) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:max_details],
        "mismatch_details_truncated": len(mismatches) > max_details,
    }


def _wall(payload: dict[str, Any], width: str) -> dict[str, Any]:
    summary = payload["summary"][width]
    ar_wall = float(summary["ar"]["wall_seconds"])
    mtp_wall = float(summary["mtp"]["wall_seconds"])
    generated = int(summary["ar"]["generated_tokens"]) + int(
        summary["mtp"]["generated_tokens"]
    )
    combined = ar_wall + mtp_wall
    return {
        "ar_wall_seconds": ar_wall,
        "second_arm_wall_seconds": mtp_wall,
        "combined_wall_seconds": combined,
        "combined_generated_tokens": generated,
        "combined_generated_tok_s": generated / combined,
    }


def _trace_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty kernel trace: {path}")
    fields = tuple(rows[0])
    name_key = next(
        key
        for key in fields
        if key.lower() in {"kernel_name", "name"}
        or "kernel_name" in key.lower()
    )
    duration_key = next(
        (
            key
            for key in fields
            if key.lower() == "durationns" or "duration" in key.lower()
        ),
        None,
    )
    start_key = next(
        (key for key in fields if key.lower() == "start_timestamp"), None
    )
    end_key = next(
        (key for key in fields if key.lower() == "end_timestamp"), None
    )
    if duration_key is None and (start_key is None or end_key is None):
        raise ValueError("kernel trace has no duration or timestamp pair")
    selected: dict[str, list[float]] = {"cast": [], "q4_f16": [], "q5_f16": []}
    names: dict[str, set[str]] = {key: set() for key in selected}
    for row in rows:
        name = str(row[name_key])
        lowered = name.lower()
        family = None
        if "cast_bf16_to_f16" in lowered:
            family = "cast"
        elif "gguf_q4_t16_dense_wmma_prefill" in lowered and (
            "half" in lowered or "float16" in lowered
        ):
            family = "q4_f16"
        elif "gguf_q5_t16_dense_wmma_prefill" in lowered and (
            "half" in lowered or "float16" in lowered
        ):
            family = "q5_f16"
        if family is None:
            continue
        names[family].add(name)
        if duration_key is not None:
            duration_ns = float(row[duration_key])
        else:
            assert start_key is not None and end_key is not None
            duration_ns = float(row[end_key]) - float(row[start_key])
        selected[family].append(duration_ns)
    if any(not names[family] for family in selected):
        missing = [family for family in selected if not names[family]]
        raise ValueError(f"expected B2 trace families missing: {missing}")
    families = {}
    for family, durations in selected.items():
        families[family] = {
            "calls": len(durations),
            "duration_ms_sum": sum(durations) / 1e6,
            "duration_ms_median": statistics.median(durations) / 1e6,
            "kernel_names": sorted(names[family]),
        }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "families": families,
        "expected_families_observed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-repeat", type=Path, required=True)
    parser.add_argument("--kernel-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "control": args.control,
        "candidate": args.candidate,
        "candidate_repeat": args.candidate_repeat,
    }
    raw = {name: _load(path) for name, path in paths.items()}
    commits = {str(payload["source"]["commit"]) for payload in raw.values()}
    if len(commits) != 1 or any(payload["source"]["dirty"] for payload in raw.values()):
        raise ValueError("B2 arms must use one clean source commit")
    protocols = {json.dumps(payload["protocol"], sort_keys=True) for payload in raw.values()}
    if len(protocols) != 1:
        raise ValueError("B2 arms do not use one protocol")

    per_width: dict[str, Any] = {}
    for width in ("2", "8"):
        control = _wall(raw["control"], width)
        candidate = _wall(raw["candidate"], width)
        repeat = _wall(raw["candidate_repeat"], width)
        per_width[width] = {
            "control": control,
            "candidate": candidate,
            "candidate_repeat": repeat,
            "candidate_wall_reduction_pct": 100.0
            * (1.0 - candidate["combined_wall_seconds"] / control["combined_wall_seconds"]),
            "candidate_speedup_pct": 100.0
            * (control["combined_wall_seconds"] / candidate["combined_wall_seconds"] - 1.0),
            "repeat_wall_delta_pct": 100.0
            * (repeat["combined_wall_seconds"] / candidate["combined_wall_seconds"] - 1.0),
        }

    all_cells = [cell for payload in raw.values() for cell in payload["cells"]]
    candidate_repeat_ids = _id_comparison(
        raw["candidate"], raw["candidate_repeat"]
    )
    control_candidate_ids = _id_comparison(raw["control"], raw["candidate"])
    performance_screen_passed = all(
        row["candidate_wall_reduction_pct"] > 0.0 for row in per_width.values()
    )
    output = {
        "schema": 1,
        "kind": "gfx1151_qwen38_b2_f16_prefill_screen",
        "date": "2026-09-02",
        "status": "screen_passed_pending_lifecycle_and_production_gates"
        if performance_screen_passed
        else "screen_rejected",
        "performance_claim": False,
        "source_commit": commits.pop(),
        "physical_host": {
            "hostname": "gfx1151",
            "gpu": "AMD Strix Halo Radeon Graphics / 1002:1586",
            "architecture": "gfx1151",
        },
        "model": raw["control"]["model"],
        "prompt_suite": raw["control"]["protocol"]["prompts"],
        "runtime_profile": raw["control"]["runtime_profile"],
        "protocol": raw["control"]["protocol"],
        "arm_definition": {
            "control": "HIPENGINE_GGUF_PREFILL_F16_STAGING=0",
            "candidate": "HIPENGINE_GGUF_PREFILL_F16_STAGING=1",
            "candidate_repeat": "independent candidate process with the same environment and schedule",
            "combined_wall": "sum of the independently measured AR-labeled and speculative-request-labeled D1 walls; expected_mtp_widths=none makes both typed no-admission prefill arms",
        },
        "per_width": per_width,
        "correctness": {
            "self_exact_cells": sum(bool(cell["correctness"]["passed"]) for cell in all_cells),
            "cells": len(all_cells),
            "route_expectation_cells": sum(
                not bool(cell["mtp_engaged"]) for cell in all_cells
            ),
            "candidate_repeat_generated_ids": candidate_repeat_ids,
            "control_candidate_generated_ids_diagnostic": control_candidate_ids,
            "generated_id_equality_policy": raw["control"]["protocol"][
                "generated_id_equality"
            ],
        },
        "kernel_trace": _trace_summary(args.kernel_csv),
        "screen_verdict": {
            "performance_screen_passed": performance_screen_passed,
            "retained": False,
            "blockers": [
                "F16 staging workspace is module-global rather than request/session lifecycle-owned",
                "candidate repeat differs in one free-running C8 arm cell; the binding strict-teacher and same-schedule determinism gates remain pending",
                "production numerical, isolation, lifecycle, BF16-relative, task, and full retention gates remain pending",
            ],
        },
        "raw_sources": [
            {
                "role": name,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        ],
        "commands": {
            "suite_template": "HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_COMPILER_VERSION_FILE=/tmp/q38-z0-run-20260901-201730/hipcc-version.txt HIPENGINE_REQUIRE_CACHED_BUILD=1 GPU_MAX_HW_QUEUES=2 HIPENGINE_GGUF_PREFILL_F16_STAGING=<0|1> .venv/bin/python scripts/gguf_mtp_c1c8_server_bench.py --model /home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend hip_gfx1151 --execution-profile production --widths 2,8 --expected-mtp-widths none --max-tokens 1 --batch-window-ms 50 --correctness-contract mtp_self_exact --output <arm.json>",
            "trace": "HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_COMPILER_VERSION_FILE=/tmp/q38-z0-run-20260901-201730/hipcc-version.txt HIPENGINE_REQUIRE_CACHED_BUILD=1 HIPENGINE_GGUF_PREFILL_F16_STAGING=1 GPU_MAX_HW_QUEUES=2 rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/q38-b2-run/pf3-rocprof -- .venv/bin/python scripts/qwen38_prefill_sweep_trace.py --model /home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend hip_gfx1151 --rows 72,288 --max-sequence-length 1152 --compiler-version-file /tmp/q38-z0-run-20260901-201730/hipcc-version.txt --require-cached-build --output /tmp/q38-b2-run/pf3-trace-raw.json",
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "per_width": per_width}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
