"""Assemble the exact host-cache comparison and source-default smoke."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_conservative_cost import summarize_cost
from scripts.qwen4exp_canonical_ar_bench import token_ids_sha256

CANDIDATE = "production_moe_backend_cache"


def assemble(root):
    hashes = {}

    def load(name):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    ab = load("resume-backend-cache-ab.json")
    smoke = load("resume-backend-cache-default-smoke.json")
    cases = {f"{category}-p{shape}" for category in (
        "code", "general_en", "general_ja", "mixed_ja_en")
        for shape in (512, 1024, 4096)}
    arms = ("before", CANDIDATE)
    expected = {(case, arm, repeat) for case in cases for arm in arms for repeat in range(3)}
    samples = ab["samples"]
    observed = {(row["case_id"], row["arm"], row["repetition"]) for row in samples}
    if (ab["status"] != "completed" or not ab["source"]["tracked_clean"]
            or ab["candidates"] != [CANDIDATE] or len(samples) != 72
            or observed != expected or not ab["protocol"]["shared_decode_graphs"]
            or ab["protocol"]["pairs"] != 3 or ab["after_close"]["current_allocated_bytes"]):
        raise ValueError("incomplete or invalid A/B")
    flag = "HIPENGINE_QWEN4_EXP_MOE_BACKEND_CACHE"
    if (ab["arm_overrides"][CANDIDATE] != {flag: "1"}
            or ab["arm_overrides"]["before"].get(flag) not in (None, "0")):
        raise ValueError("unexpected intervention")
    case_costs = {}
    for case in sorted(cases):
        rows = [row for row in samples if row["case_id"] == case]
        for row in rows:
            if (not row["finite"] or row["decode_transitions"] != 128
                    or row["output_token_count"] != 129
                    or len(row["output_token_ids"]) != 129
                    or token_ids_sha256(row["output_token_ids"]) != row["output_token_ids_sha256"]
                    or any(not math.isfinite(row[key]) or row[key] <= 0
                           for key in ("prefill_ms", "decode_ms", "client_wall_s"))):
                raise ValueError("invalid work or finite-output evidence")
        for key in ("output_token_ids_sha256", "logits_sha256", "state_sha256"):
            if len({row[key] for row in rows}) != 1:
                raise ValueError(f"cross-arm or repeat mismatch: {case}/{key}")
        walls = {arm: [row["client_wall_s"] for row in sorted(
            rows, key=lambda item: item["repetition"]) if row["arm"] == arm] for arm in arms}
        ratio = sum(walls["before"]) / sum(walls[CANDIDATE])
        if ratio <= 1:
            raise ValueError(f"complete-request improvement not established: {case}")
        paired = [before / after for before, after in zip(
            walls["before"], walls[CANDIDATE], strict=True)]
        case_costs[case] = dict(
            complete_request_speedup=ratio, paired_speedups=paired,
            wall_cv={arm: statistics.stdev(values) / statistics.mean(values)
                     for arm, values in walls.items()})
    summary = summarize_cost(samples, 3)
    if summary != ab["comparisons"][CANDIDATE]:
        raise ValueError("reported rates do not reproduce from samples")
    if (smoke["status"] != "completed" or smoke["overrides"]
            or not smoke["quality"]["hard_gates_passed"]
            or smoke["quality"]["summary"]["max_abs_logit_delta"] != 0
            or smoke["quality"]["summary"]["rows"] != 9
            or not smoke["deterministic"] or not smoke["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in smoke["lifecycle"].values())):
        raise ValueError("fresh source-default smoke failed")
    sample_keys = (
        "case_id", "category", "prompt_tokens", "prompt_token_ids_sha256", "repetition",
        "arm", "sequence_slot", "prefill_ms", "decode_ms", "decode_transitions",
        "client_wall_s", "output_token_count", "output_token_ids_sha256",
        "logits_sha256", "state_sha256", "finite",
    )
    return dict(
        schema=1, status="retained_exact_prefill_and_request_win",
        arithmetic_class="T0", performance_claim=True, raw_sha256=hashes,
        source=ab["source"], host=ab["host"], model=ab["model"],
        execution_profile="production", quant="UD-Q4_K_XL", kv="BF16",
        command=ab["command"], protocol=ab["protocol"], base_manifest=ab["base_manifest"],
        environment={
            "HIPENGINE_HIP_ARCH": "gfx1151", "HIPENGINE_REQUIRE_CACHED_BUILD": "1",
            "GPU_MAX_HW_QUEUES": "2", "PYTHONPATH": ".",
            "PATH": "/home/lhl/miniforge3/envs/therock/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": ":".join(
                "/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/"
                "_rocm_sdk_devel/" + suffix for suffix in ("lib", "lib64", "lib/llvm/lib")),
        },
        arm_overrides=ab["arm_overrides"], summary=summary, case_costs=case_costs,
        samples=[{key: row[key] for key in sample_keys} for row in samples],
        graph_stats=ab["graph_stats"], after_close=ab["after_close"],
        runtime_libraries=json.loads(Path(__file__).with_name("runtime-libraries.json").read_bytes()),
        source_default_smoke=smoke,
        quality_basis=[
            "No GPU kernel, arithmetic, graph-node or allocation change.",
            "Generation mutation, failed refresh and wrapper-preservation CPU tests pass.",
            "Four MoE GPU CPU-reference/fallback fixtures pass.",
            "All 72 samples match IDs/final full logits/state across arms and repeats.",
            "Prior production arithmetic qualification: benchmarks/results/2026-09-15-q8-blockscale-restoration/artifact.json",
        ],
        limitations=[
            "Three pairs are this optimization comparison, not the five-pair final external closure.",
            "TG is lower by 0.16/0.17/1.36 percent at 512/1K/4K; this is not a decode win.",
            "Complete-request gains are measured at 128 decode transitions, not arbitrary output lengths.",
            "No measured causal attribution for the decode difference; no clock/affinity workaround.",
            "The worktree source-default smoke supplements the clean pinned A/B, not a new full qualification.",
            "No MTP, multimodal, other-host or new long-form factual-quality claim.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root), indent=2) + "\n")
