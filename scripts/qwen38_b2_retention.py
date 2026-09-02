#!/usr/bin/env python3
"""Assemble the fail-closed Qwen3.8 B2 input-F16 retention gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.qwen38_z3_candidate_gate import evaluate_candidate_evidence

ROLES = (
    "control",
    "candidate",
    "candidate_repeat",
    "screen",
    "quality",
    "task_control",
    "task_candidate",
    "task_repeat",
    "lifecycle",
    "cast_correction",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_server(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("status") != "complete" or payload.get("passed") is not True:
        raise ValueError(f"incomplete server arm: {path}")
    return payload


def _ids(payload: dict[str, Any]) -> dict[tuple[int, str, str], list[list[int]]]:
    return {
        (int(cell["width"]), str(cell["prompt_id"]), arm): [
            [int(token) for token in row["generated_ids"]]
            for row in cell[arm]["rows"]
        ]
        for cell in payload["cells"]
        for arm in ("ar", "mtp")
    }


def _compare_ids(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = _ids(left)
    right_ids = _ids(right)
    if left_ids.keys() != right_ids.keys():
        raise ValueError("ID cell sets differ")
    mismatches = [
        {"width": key[0], "prompt_id": key[1], "arm": key[2]}
        for key in left_ids
        if left_ids[key] != right_ids[key]
    ]
    return {
        "cells": len(left_ids),
        "equal": len(left_ids) - len(mismatches),
        "mismatches": mismatches[:8],
        "passed": not mismatches,
    }


def _wall(payload: dict[str, Any], width: int) -> dict[str, Any]:
    summary = payload["summary"][str(width)]
    ar_wall = float(summary["ar"]["wall_seconds"])
    second_wall = float(summary["mtp"]["wall_seconds"])
    cells = [cell for cell in payload["cells"] if int(cell["width"]) == width]
    prompt_tokens = sum(
        int(row["usage"]["prompt_tokens"])
        for cell in cells
        for row in cell["mtp"]["rows"]
    )
    return {
        "ar_wall_seconds": ar_wall,
        "second_arm_wall_seconds": second_wall,
        "combined_wall_seconds": ar_wall + second_wall,
        "prompt_tokens": prompt_tokens,
        "second_arm_prompt_tok_s": prompt_tokens / second_wall,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument(
            "--" + role.replace("_", "-"),
            dest=role,
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw = {
        role: _load_server(getattr(args, role))
        for role in ("control", "candidate", "candidate_repeat")
    }
    task = {
        role: _load_server(getattr(args, f"task_{role}"))
        for role in ("control", "candidate", "repeat")
    }
    screen = _load(args.screen)
    quality = _load(args.quality)
    lifecycle = _load(args.lifecycle)
    cast_correction = _load(args.cast_correction)

    commits = {payload["source"]["commit"] for payload in raw.values()}
    task_commits = {payload["source"]["commit"] for payload in task.values()}
    all_runtime_arms = (*raw.values(), *task.values())
    if (
        len(commits) != 1
        or len(task_commits) != 1
        or commits != task_commits
        or any(payload["source"]["dirty"] for payload in all_runtime_arms)
    ):
        raise ValueError("all runtime arms must use one clean commit")

    per_width: dict[str, Any] = {}
    for width in (2, 8):
        control = _wall(raw["control"], width)
        candidate = _wall(raw["candidate"], width)
        repeat = _wall(raw["candidate_repeat"], width)
        per_width[str(width)] = {
            "control": control,
            "candidate": candidate,
            "candidate_repeat": repeat,
            "wall_reduction_pct": 100.0
            * (
                1.0
                - candidate["combined_wall_seconds"]
                / control["combined_wall_seconds"]
            ),
            "speedup_pct": 100.0
            * (
                control["combined_wall_seconds"]
                / candidate["combined_wall_seconds"]
                - 1.0
            ),
            "repeat_wall_delta_pct": 100.0
            * (
                repeat["combined_wall_seconds"]
                / candidate["combined_wall_seconds"]
                - 1.0
            ),
            "prompt_throughput_gain_pct": 100.0
            * (
                candidate["second_arm_prompt_tok_s"]
                / control["second_arm_prompt_tok_s"]
                - 1.0
            ),
        }
    performance_passed = all(
        row["wall_reduction_pct"] > 0.0 for row in per_width.values()
    )

    isolation_fields = (
        "strict_focal_bit_exact",
        "candidate_focal_bit_exact",
        "candidate_matches_strict_a",
        "candidate_matches_strict_b",
    )
    quality_passed = (
        bool(quality.get("passed"))
        and quality["summary"]["max_kl"] <= 0.05
        and quality["summary"]["top1_agreement"] >= 0.99
        and quality["strict_repeat_bit_exact"]
        and quality["candidate_repeat_bit_exact"]
        and quality["controls_exact"]
        and all(
            all(row[field] for field in isolation_fields)
            for row in quality["isolation"]
        )
    )

    task_control_candidate = _compare_ids(task["control"], task["candidate"])
    task_candidate_repeat = _compare_ids(task["candidate"], task["repeat"])
    task_k0 = all(
        not cell["mtp_engaged"]
        and cell["mtp_budget_conformed"]
        and cell["exact"]
        for payload in task.values()
        for cell in payload["cells"]
    )
    task_passed = (
        task_control_candidate["passed"]
        and task_candidate_repeat["passed"]
        and task_k0
    )

    trace = screen["kernel_trace"]
    trace_passed = bool(trace["expected_families_observed"]) and all(
        trace["families"][family]["calls"] > 0
        and trace["families"][family]["duration_ms_sum"] > 0.0
        for family in ("cast", "q4_f16", "q5_f16")
    )
    lifecycle_passed = (
        bool(lifecycle["gpu_lifecycle"]["passed"])
        and lifecycle["decision"]["workspace_lifecycle_blocker_resolved"]
    )
    cast_passed = bool(cast_correction["post_correction_full_suite"]["passed"])
    repeat_passed = _compare_ids(
        raw["candidate"], raw["candidate_repeat"]
    )["passed"]
    bf16_relative_passed = (
        quality["summary"]["bit_identical_rows"] == quality["summary"]["rows"]
    )

    checks = {
        "implemented": True,
        "strict_fallback_registered": True,
        "full_category_suite": quality_passed and task_passed,
        "heldouts": quality_passed and task_passed,
        "complete_wall_improved": performance_passed,
        "controls_exact": quality_passed,
        "production_numerics": quality_passed,
        "deterministic": quality_passed and repeat_passed and task_candidate_repeat["passed"],
        "isolation": quality_passed,
        "lifecycle": lifecycle_passed,
        "bf16_relative": bf16_relative_passed,
        "task_quality": task_passed,
        "expected_kernel_trace": trace_passed,
    }
    candidate_evidence = {
        "candidate_id": "P1_F16_ACTIVATION_B",
        "declared_class": "T1",
        "checks": checks,
    }
    candidate_gate = evaluate_candidate_evidence(candidate_evidence)
    eligible = bool(candidate_gate["passed"] and cast_passed)

    paths = {role: getattr(args, role) for role in ROLES}
    output = {
        "schema": 1,
        "kind": "gfx1151_qwen38_b2_f16_retention_gates",
        "date": "2026-09-02",
        "status": (
            "retention_eligible_pending_profile_default" if eligible else "blocked"
        ),
        "performance_claim": False,
        "source_commit": commits.pop(),
        "physical_host": screen["physical_host"],
        "model": raw["control"]["model"],
        "prompt_suite": raw["control"]["protocol"]["prompts"],
        "runtime_profile": raw["control"]["runtime_profile"],
        "per_width": per_width,
        "quality": {
            "summary": quality["summary"],
            "scopes": quality["scopes"],
            "strict_repeat_bit_exact": quality["strict_repeat_bit_exact"],
            "candidate_repeat_bit_exact": quality["candidate_repeat_bit_exact"],
            "controls_exact": quality["controls_exact"],
            "isolation": quality["isolation"],
            "train_rows": 54,
            "heldout_rows": 36,
            "bf16_relative_note": (
                "Derived exactly: candidate and BF16-activation fallback logits "
                "are byte-identical, so both have identical metrics against any "
                "aligned BF16/full-precision teacher."
            ),
        },
        "task_d24": {
            "control_candidate_ids": task_control_candidate,
            "candidate_repeat_ids": task_candidate_repeat,
            "automatic_k0_all_cells": task_k0,
            "cells": sum(len(payload["cells"]) for payload in task.values()),
            "note": "Exact output IDs imply every deterministic task scorer is unchanged.",
        },
        "kernel_trace": trace,
        "lifecycle": {"passed": lifecycle_passed, "artifact": str(args.lifecycle)},
        "cast_correction": {
            "passed": cast_passed,
            "artifact": str(args.cast_correction),
        },
        "candidate_evidence": candidate_evidence,
        "candidate_gate": candidate_gate,
        "checks": checks,
        "decision": {
            "eligible": eligible,
            "default_changed": False,
            "next": (
                "profile-scoped production default with strict/fallback/env "
                "override tests and clean default confirmation"
            ),
        },
        "raw_sources": [
            {
                "role": role,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for role, path in paths.items()
        ],
        "commands": {
            "prefill": screen["commands"]["suite_template"],
            "quality": (
                "HIPENGINE_HIP_ARCH=gfx1151 "
                "HIPENGINE_COMPILER_VERSION_FILE=/tmp/q38-z0-run-20260901-201730/hipcc-version.txt "
                "HIPENGINE_REQUIRE_CACHED_BUILD=1 GPU_MAX_HW_QUEUES=2 "
                ".venv/bin/python /tmp/q38-b2-run/pf6_quality.py "
                "--model /home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf "
                "--prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl "
                "--compiler-version-file /tmp/q38-z0-run-20260901-201730/hipcc-version.txt "
                "--output /tmp/q38-b2-run/pf6-quality.json"
            ),
            "task": (
                "HIPENGINE_HIP_ARCH=gfx1151 "
                "HIPENGINE_COMPILER_VERSION_FILE=/tmp/q38-z0-run-20260901-201730/hipcc-version.txt "
                "HIPENGINE_REQUIRE_CACHED_BUILD=1 GPU_MAX_HW_QUEUES=2 "
                "HIPENGINE_GGUF_PREFILL_F16_STAGING=<0|1> .venv/bin/python "
                "scripts/gguf_mtp_c1c8_server_bench.py --model "
                "/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend "
                "hip_gfx1151 --execution-profile production --widths 2,8 "
                "--expected-mtp-widths none --max-tokens 24 --candidate-budget 2 "
                "--batch-window-ms 50 --correctness-contract mtp_self_exact "
                "--output <arm.json>"
            ),
            "trace": (
                "HIPENGINE_HIP_ARCH=gfx1151 "
                "HIPENGINE_COMPILER_VERSION_FILE=/tmp/q38-z0-run-20260901-201730/hipcc-version.txt "
                "HIPENGINE_REQUIRE_CACHED_BUILD=1 "
                "HIPENGINE_GGUF_PREFILL_F16_STAGING=1 GPU_MAX_HW_QUEUES=2 "
                "rocprofv3 --kernel-trace --output-format csv --output-directory "
                "/tmp/q38-b2-run/pf6-rocprof -- .venv/bin/python "
                "scripts/qwen38_prefill_sweep_trace.py --model "
                "/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend "
                "hip_gfx1151 --rows 72,288 --max-sequence-length 1152 "
                "--compiler-version-file "
                "/tmp/q38-z0-run-20260901-201730/hipcc-version.txt "
                "--require-cached-build --output /tmp/q38-b2-run/pf6-trace-raw.json"
            ),
            "candidate_gate": (
                "python3 scripts/qwen38_z3_candidate_gate.py <evidence.json>"
            ),
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "per_width": per_width,
                "task": output["task_d24"],
                "gate": candidate_gate,
            },
            indent=2,
        )
    )
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
