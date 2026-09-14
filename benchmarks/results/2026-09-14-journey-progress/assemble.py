"""Checkpoint completed experiments without converting partial work into wins."""

import argparse
import hashlib
import json
from pathlib import Path

FILES = (
    "production-baseline.json",
    "production-repair-q8gdn.json",
    "gdn-multi-profile.json",
    "family-localization.json",
    "family-localization-combined.json",
    "family-localization-exact-flags.json",
    "family-localization-q8down.json",
    "owner-refresh-code4k.json",
    "owner-refresh-other4k.json",
    "graph-code4k.json",
    "graph-other4k.json",
    "ple-pread-warm-random.json",
    "ple-pread-workers-warm-random.json",
    "ple-pread-workers-cold-random.json",
    "gdn-multi-screen.json",
    "chunk2048-allocation.json",
    "chunk4096-allocation.json",
    "q8-residual-owner.json",
    "q8-residual-profile.json",
    "ple-copy-elision-model-screen.json",
)


def compact(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("status") in ("running", None) and not path.name.startswith("graph-"):
        raise ValueError(f"unfinished observation: {path}")
    kind = data.get("kind")
    if kind is None and "chunk_size" in data and "resident_capacity" in data:
        kind = "qwen4exp_chunk_memory_probe"
    result = {
        "raw_path": str(path), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": kind, "status": data.get("status", "summarized"),
        "source": data.get("source"), "host": data.get("host"),
        "command": data.get("command"), "performance_claim": False,
    }
    if kind == "qwen4exp_execution_profile_candidate_gate":
        result.update(
            candidate=data["candidate"], protocol=data["protocol"],
            measurement_valid=data["measurement_valid"],
            quality=data["quality"]["quality"],
            deterministic=data["quality"]["repeat_determinism"]["passed"],
            state_repeat=data["state_repeat_gate"]["passed"],
            lifecycle=data["lifecycle"]["passed"],
            task_status=data["task_gate"]["status"],
            task_exact_count=data["task_gate"]["strict_exact_count"],
        )
    elif kind == "journey_incumbent_family_localization":
        result["protocol"] = data["protocol"]
        result["arms"] = {
            name: {"overrides": arm["overrides"],
                   "summary": arm["quality"]["summary"],
                   "by_scope": arm["quality"]["by_scope"],
                   "hard_gates_passed": arm["quality"]["hard_gates_passed"]}
            for name, arm in data["arms"].items()
        }
        result["limits"] = "50 diagnostic rows, one repeat; not a full production gate."
    elif kind == "qwen4exp_hip_semantic_capture":
        result["model_identity"] = data["model_identity"]
        result["cases"] = [{key: case[key] for key in (
            "id", "phase", "command", "profile", "raw_path", "raw_sha256",
        )} for case in data["cases"]]
    elif kind == "qwen4exp_graph_census":
        result["cases"] = [{key: case[key] for key in (
            "id", "phase", "windows", "memory_after_close", "trace_sources",
        )} for case in data["cases"]]
        result["limits"] = data["limitations"]
    elif kind == "qwen4exp_ple_sorted_gather_screen":
        result["method"] = data["method"]
        result["parent_mapping"] = data.get("parent_mapping")
        result["scope"] = data["scope"]
        result["cases"] = [{key: case[key] for key in (
            "id", "category", "rows", "unique_rows", "cache_mode",
            "median_ns", "parent_over_candidate", "bit_exact",
        )} for case in data["cases"]]
    elif kind == "qwen4exp_gdn_dpp_screen":
        result["cases"] = data["cases"]
        result["libraries"] = data["libraries"]
        result["scope"] = data["boundary"]
    elif kind == "qwen4exp_chunk_memory_probe":
        result.update(
            chunk_size=data["chunk_size"],
            requested_context_length=data["requested_context_length"],
            resident_capacity=data["resident_capacity"],
            allocation_margins=data.get("allocation_margins"),
            prepared_memory=data.get("prepared_memory"),
            memory_after_close=data["memory_after_close"],
            error=data.get("error"),
            limits="Allocation succeeded but modeled scratch allowance failed; no inference or throughput claim.",
        )
    elif kind == "qwen4exp_q8_weight_plane_screen":
        result.update(
            shape=data["shape"], tensor=data["tensor"],
            tensor_sha256=data["tensor_sha256"], routing=data["routing"],
            cases=[{key: case[key] for key in ("tokens", "routing", "median_ms", "mse_vs_strict")}
                   for case in data["cases"]],
        )
    elif kind == "qwen4exp_ple_complete_model_ab":
        result.update(
            protocol=data["protocol"],
            summary={key: value for key, value in data["summary"].items()
                     if key in ("correctness", "by_shape", "by_category")},
            exact_final_logits_and_state=data["exact_final_logits_and_state"],
            after_close=data["after_close"],
        )
    else:
        raise ValueError(f"unsupported observation kind: {data['kind']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("artifact.json"))
    args = parser.parse_args()
    payload = {
        "schema": 1, "kind": "journey_execution_checkpoint",
        "status": "incomplete_campaign", "performance_claim": False,
        "adopted": [
            {"commit": "47e9f0b9e", "mechanism": "mapping-only random PLE advice"},
            {"commit": "04653266c", "mechanism": "single-column DPP GDN suffix"},
        ],
        "observations": {name: compact(args.raw_root / name) for name in FILES},
        "remaining": [
            "Resolve incumbent full numerical/task qualification",
            "Complete PLE dedup/cache-aware pread/asynchronous staging qualification",
            "Qualify or reject multi-column GDN and serial-prefix changes",
            "Larger-chunk allocation, real-routing matrix/repair/compaction experiments",
            "HC/BF16-stream/conv/gather fusion experiments",
            "Ordered BF16 QSA scoring/top-k/packing/attention depth gates",
            "Exact runtime-pin review and three-arm graph/PM4 qualification if justified",
            "Batched target MTP verifier and full-suite economics",
            "External compatibility gaps and final counterbalanced closure",
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
