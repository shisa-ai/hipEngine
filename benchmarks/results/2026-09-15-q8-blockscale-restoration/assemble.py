"""Assemble bounded evidence for guarded Q8 restoration."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_conservative_cost import summarize_cost


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_gate(packet):
    return {
        key: packet[key] for key in (
            "source", "host", "model", "command", "candidate",
            "candidate_dispatch_calls", "measurement_valid", "decision",
        )
    } | {
        "quality": packet["quality"]["quality"],
        "repeatability_passed": packet["quality"]["repeat_determinism"]["passed"],
        "state_gate_passed": packet["state_repeat_gate"]["passed"],
        "task_gate": {key: packet["task_gate"][key] for key in (
            "status", "candidate_repeat_exact", "strict_exact_count", "total", "divergences")},
        "lifecycle": packet["lifecycle"],
    }


def assemble(root, trace, candidate, depth_name, ab_name):
    hashes = {}

    def load(name):
        path = root / name
        hashes[name] = digest(path)
        return json.loads(path.read_bytes())

    guarded = load("blockscale-guarded-profile.json")
    combined = load("blockscale-guarded-mmq-profile.json")
    chosen = guarded if candidate != "q8_blockscale_guarded_mmq" else combined
    if not chosen["measurement_valid"] or not chosen["decision"]["passed"]:
        raise ValueError("selected short gate has not passed")
    depth = load(depth_name)
    if (depth["status"] != "completed" or not depth["quality"]["hard_gates_passed"]
            or not depth["deterministic"] or not depth["state_gate"]["passed"]
            or depth["candidate_dispatch_calls"] <= 0):
        raise ValueError("selected depth gate has not passed")
    exact_base = load("blockscale-guarded-quad-depth.json")
    if (exact_base["quality"]["summary"]["rows"] != 780
            or not exact_base["quality"]["hard_gates_passed"]
            or not exact_base["state_gate"]["passed"] or not exact_base["deterministic"]):
        raise ValueError("exact canonical base is incomplete")
    if candidate == "q8_blockscale_guarded_ordered":
        expected = {f"{category}-p4096" for category in (
            "code", "general_en", "general_ja", "mixed_ja_en")}
        if (set(depth["protocol"]["case_ids"]) != expected
                or depth["quality"]["summary"]["rows"] != 516):
            raise ValueError("all four sparse-decode categories required")
    ab = load(ab_name)
    expected_samples = 12 * 3 * (1 + len(ab["candidates"]))
    if (ab["status"] != "completed" or len(ab["samples"]) != expected_samples
            or not ab["protocol"]["shared_decode_graphs"]
            or ab["after_close"]["current_allocated_bytes"]):
        raise ValueError("full closed A/B required")
    expected_cases = {f"{category}-p{rows}" for category in (
        "code", "general_en", "general_ja", "mixed_ja_en")
        for rows in (512, 1024, 4096)}
    expected_cells = {(case, arm, repeat) for case in expected_cases
                      for arm in ("before", *ab["candidates"]) for repeat in range(3)}
    cells = {(row["case_id"], row["arm"], row["repetition"]) for row in ab["samples"]}
    if cells != expected_cells or any(not row["finite"] for row in ab["samples"]):
        raise ValueError("missing, duplicated or nonfinite A/B cells")
    for case in expected_cases:
        rows = [row for row in ab["samples"] if row["case_id"] == case]
        for key in ("output_token_ids_sha256", "logits_sha256", "state_sha256"):
            if len({row[key] for row in rows}) != 1:
                raise ValueError(f"A/B repeat or cross-arm mismatch: {case}/{key}")
    smoke = load("blockscale-default-smoke.json")
    if (smoke["status"] != "completed" or not smoke["quality"]["hard_gates_passed"]
            or not smoke["deterministic"] or not smoke["state_gate"]["passed"]
            or smoke["overrides"]
            or any(row["current_allocated_bytes"] for row in smoke["lifecycle"].values())):
        raise ValueError("fresh default smoke failed")
    selected_env = ab["arm_overrides"][candidate]
    if (candidate not in ab["candidates"]
            or any(selected_env.get(key) != value for key, value in depth["overrides"].items())
            or any(value != ab["arm_overrides"]["before"][key] for key, value in selected_env.items()
                   if key not in depth["overrides"])):
        raise ValueError("candidate/depth/A-B override mismatch")
    request_speedups = {}
    for case in sorted(expected_cases):
        walls = {arm: sum(row["client_wall_s"] for row in ab["samples"]
                          if row["case_id"] == case and row["arm"] == arm)
                 for arm in ("before", candidate)}
        request_speedups[case] = walls["before"] / walls[candidate]
    direct_decode_comparison = summarize_cost([
        {**row, "mode": "before" if row["arm"] == "q8_blockscale_guarded_quad" else "after"}
        for row in ab["samples"] if row["arm"] in (
            "q8_blockscale_guarded_quad", "q8_blockscale_guarded_ordered")
    ], 3)
    mmq_depth = load("blockscale-guarded-mmq-depth.json")
    dense_depth = load("blockscale-guarded-restore-depth.json")
    ordered_ab = load("blockscale-ordered-ab.json")
    partial_quad_ab = load("blockscale-quad-ab.json")
    ordered_depth = load("blockscale-guarded-ordered-depth.json")
    owner = load("blockscale-final-owner.json")
    for case in owner["cases"]:
        if case["errors"][-1]["changed"]:
            raise ValueError("owner exact repair failed")
    con = sqlite3.connect(f"file:{trace}?mode=ro", uri=True)
    trace_rows = [
        dict(name=row[0], calls=row[1], min_ns=row[2], max_ns=row[3],
             vgpr=row[4], lds=row[5], scratch=row[6])
        for row in con.execute(
            "select s.display_name,count(*),min(d.end-d.start),max(d.end-d.start),"
            "s.arch_vgpr_count,s.group_segment_size,s.private_segment_size "
            "from rocpd_kernel_dispatch d join rocpd_info_kernel_symbol s "
            "on s.id=d.kernel_id and s.guid=d.guid "
            "where s.display_name like '%q8_0_selected%' group by s.id")
    ]
    con.close()
    if not any("sparse_repair" in row["name"] and row["calls"] for row in trace_rows):
        raise ValueError("missing sparse repair trace")
    return dict(
        schema=1, candidate=candidate, raw_sha256=hashes,
        source=ab["source"], host=ab["host"], model=ab["model"],
        quant="UD-Q4_K_XL", kv="BF16", performance_claim=True,
        short_gates={"guarded": compact_gate(guarded), "guarded_mmq": compact_gate(combined)},
        depth=depth,
        exact_canonical_base=exact_base,
        default_smoke=smoke,
        short_gate_basis="Guarded Q8 short gate; added QSA paths are inactive at natural-prompt depth.",
        blocked_mmq_depth={
            "source": mmq_depth["source"], "command": mmq_depth["command"],
            "quality": mmq_depth["quality"],
            "candidate_dispatch_calls": mmq_depth["candidate_dispatch_calls"],
        },
        blocked_dense_restore_depth={
            "source": dense_depth["source"], "command": dense_depth["command"],
            "quality": dense_depth["quality"],
            "candidate_dispatch_calls": dense_depth["candidate_dispatch_calls"],
        },
        independent_cache_diagnostics={
            "reason": "Separate graph instances are an uncontrolled nuisance for unchanged captured decode units.",
            "source": ordered_ab["source"], "command": ordered_ab["command"],
            "summary": ordered_ab["summary"],
            "depth_quality": ordered_depth["quality"],
            "partial_quad_run": {
                "status": partial_quad_ab["status"],
                "samples": len(partial_quad_ab["samples"]),
                "after_close": partial_quad_ab["after_close"],
                "reason_stopped": "Replace with shared-graph three-arm protocol",
            },
        },
        ab={key: ab[key] for key in (
            "source", "host", "model", "command", "protocol", "candidates",
            "arm_overrides", "comparisons", "graph_stats", "after_close", "base_manifest")},
        complete_request_speedups=request_speedups,
        ordered_vs_prefill_only=direct_decode_comparison,
        owner=owner, trace=dict(path=str(trace), sha256=digest(trace), kernels=trace_rows),
        limitations=[
            "Risk screening is empirical; no universal bit-identity proof is claimed.",
            "Owner uses real weights but synthetic activations and seeded skewed routing.",
            "Full model gates use unchanged model/quant/BF16 KV and fixed category/heldout suites.",
            "MTP and multimodal performance are outside this text-AR qualification.",
            "Short free trajectories and depth parity are not a complete-EOS factual-quality comparison.",
            "The 780-row base and 516-row sparse-decode gate overlap; counts are not additive.",
            "Disabled GDN/MoE families are not individually proven faulty; dense/GR restoration failed as a combination.",
            "Shared graphs control a nuisance variable but do not establish the cause of category decode regressions.",
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--candidate", required=True,
                        choices=("q8_blockscale_guarded", "q8_blockscale_guarded_mmq",
                                 "q8_blockscale_guarded_quad", "q8_blockscale_guarded_ordered"))
    parser.add_argument("--depth", required=True)
    parser.add_argument("--ab", required=True)
    args = parser.parse_args()
    packet = assemble(args.raw_root, args.trace, args.candidate, args.depth, args.ab)
    Path(__file__).with_name("artifact.json").write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
