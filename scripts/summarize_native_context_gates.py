#!/usr/bin/env python3
"""Compact native-context diagnostics without retaining large prompt/state dumps."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_performance_pair(before: dict, after: dict) -> dict:
    from scripts.qwen36_dense_gguf_suite import FULL_PROMPT_IDS

    identity_keys = (
        "host_name", "hipengine_commit", "resolved_backend", "target_arch",
        "device_name", "model_fingerprint", "quant", "kv_dtype",
        "rocm_version", "hipcc_version",
    )
    for data in (before, after):
        gate, workload = data["context_gate"], data["workload"]
        if (data["status"] != "complete_exact"
                or data["provenance"]["dirty"] or gate["identity"]["dirty"]
                or gate["foreign_gpu_allocations"]
                or gate["fixed_prompt_length"] is not None
                or gate["native_eager"] or gate["bulk_prefill"]
                or tuple(workload["prompt_ids"]) != FULL_PROMPT_IDS
                or workload["runs"] < 3 or not workload["warmup"]
                or workload["candidate_budgets"] != [3]
                or workload["max_new_tokens_visible"] not in (25, 129)
                or not data["correctness"]["all_exact_greedy"]
                or not data["correctness"]["all_gpu_accept_match_cpu"]
                or data["memory_after_close"]["active_allocations"] != 0):
            raise ValueError("performance pair requires clean, exact, full-suite repeated natural runs")
        for summary in (data["summary"]["true_ar"]["full"], data["summary"]["mtp"]["3"]["full"]):
            if summary["request_count"] != 10 * workload["runs"]:
                raise ValueError("incomplete repetitions")
    for key in identity_keys:
        if before["provenance"][key] != after["provenance"][key]:
            raise ValueError(f"performance pair identity differs: {key}")
    if before["workload"] != after["workload"]:
        raise ValueError("performance pair workloads differ")
    for key in ("pci", "kfd_gpu_id", "environment"):
        if before["context_gate"].get(key) != after["context_gate"].get(key):
            raise ValueError(f"performance pair device/environment differs: {key}")
    if (before["context_gate"]["native_context_limit"] != 95
            or after["context_gate"]["native_context_limit"] is not None
            or ("serial_exact" not in before["context_gate"]["actual_verify_modes"]
                and before["workload"]["max_new_tokens_visible"] == 129)
            or set(after["context_gate"]["actual_verify_modes"]) != {"native"}
            or after["context_gate"]["graph_submissions"] <= 0):
        raise ValueError("performance pair did not exercise the intended policies")
    old = before["summary"]["mtp"]["3"]["full"]["decode_tok_s_weighted"]
    new = after["summary"]["mtp"]["3"]["full"]["decode_tok_s_weighted"]
    return {"before_mtp_tok_s": old, "after_mtp_tok_s": new,
            "mtp_change_percent": 100 * (new / old - 1)}


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    result = {
        "source_name": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "status": data["status"],
        "performance_claim": False,
        "command": data.get("command", data.get("context_gate", {}).get("command")),
        "memory_after_close": data.get("memory_after_close"),
    }
    for key in ("provenance", "model", "workload", "correctness", "timing_contract"):
        if key in data:
            result[key] = data[key]
    rows = data.get("rows", [])
    if isinstance(rows, list):
        counts = Counter(
            (r["context"], r["budget"], r["transport"], r["graph"], r["mode"])
            for r in rows
        )
        result.update(
            cases=len(rows),
            passed=sum(r["passed"] is True for r in rows),
            following_step_cases=sum(bool(r.get("following_step")) for r in rows),
            coverage=[
                dict(context=c, budget=b, transport=t, graph=g, mode=m, cases=n)
                for (c, b, t, g, m), n in sorted(counts.items())
            ],
        )
    else:
        gate = dict(data["context_gate"])
        prompts = gate.pop("prompt_ids", {})
        gate["prompt_fingerprints"] = {
            k: {"tokens": len(v), "sha256": hashlib.sha256(
                json.dumps(v, separators=(",", ":")).encode()
            ).hexdigest()}
            for k, v in prompts.items()
        }
        result["context_gate"] = gate
        result["summary"] = data.get("summary")
    if "error" in data:
        result["error"] = data["error"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--performance-pair", action="store_true")
    args = parser.parse_args()
    payload = {"schema": 1, "performance_claim": False,
               "attempts": [summarize(path) for path in args.inputs]}
    if args.performance_pair:
        if len(args.inputs) != 2:
            parser.error("performance pair requires old-limit95 then new-capacity inputs")
        payload["comparison"] = validate_performance_pair(
            *(json.loads(path.read_text()) for path in args.inputs)
        )
        payload["performance_claim"] = True
    with args.output.open("x") as out:
        out.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
