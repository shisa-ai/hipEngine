"""Preserve current-profile constructor and lazy-queue allocation probes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.qwen4exp_chunk_memory_probe import allocation_margins
from scripts.qwen4exp_conservative_cost import summarize_cost


def assemble(root):
    packets, hashes = {}, {}
    for name in ("resume-chunk2048-allocation.json", "resume-chunk2048-lazy-allocation.json",
                 "resume-chunk4096-lazy-allocation.json", "resume-chunk2048-depth.json",
                 "resume-chunk-workspace-check.json", "resume-chunk2048-workspace-ab.json"):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        packets[name] = json.loads(raw)
    for size in (2048, 4096):
        packet = packets[f"resume-chunk{size}-lazy-allocation.json"]
        if (packet["schema"] != 2 or not packet["source"]["tracked_clean"]
                or packet["prepared_runners"] != 1 or packet["prepared_context"] != 4352
                or packet["chunk_size"] != size
                or packet["memory_after_close"]["current_allocated_bytes"]):
            raise ValueError("invalid prepared allocation packet")
        queues = packet["lazy_group_risk"][0]["queues"]
        if (len(queues) != 2
                or {row["owner"] for row in queues} != {
                    "gdn_prefill_scratch", "qsa_prefill_scratch"}
                or any(row["rows"] != size for row in queues)):
            raise ValueError("missing lazy owner")
        prepared = packet["prepared_memory"]["current_allocated_bytes"]
        delta = prepared - packet["before_lazy_memory"]["current_allocated_bytes"]
        if delta != sum(row["nbytes"] for row in queues):
            raise ValueError("lazy bytes do not reconcile")
        margins = allocation_margins(packet["admission"]["plan"], prepared)
        if margins != packet["allocation_margins"]:
            raise ValueError("allocation margins do not reproduce")
        expected = "passed" if margins["scratch_margin_bytes"] >= 0 else "failed"
        if packet["status"] != expected:
            raise ValueError("allocation verdict mismatch")
    left = packets["resume-chunk2048-lazy-allocation.json"]
    right = packets["resume-chunk4096-lazy-allocation.json"]
    if (left["source"] != right["source"] or left["model_identity"] != right["model_identity"]
            or left["host"]["machine_id"] != right["host"]["machine_id"]
            or left["manifest_sha256"] != right["manifest_sha256"]
            or any(packet["admission"]["plan"]["scratch_bytes"] != 4 * 1024**3
                   for packet in (left, right))):
        raise ValueError("allocation probes use different lanes or allowances")
    depth = packets["resume-chunk2048-depth.json"]
    if (depth["status"] != "completed" or not depth["source"]["tracked_clean"]
            or not depth["quality"]["hard_gates_passed"]
            or depth["quality"]["summary"]["rows"] != 780
            or depth["quality"]["summary"]["max_abs_logit_delta"] != 0
            or not depth["deterministic"] or not depth["state_gate"]["passed"]
            or not all(row["strict_candidate_state_exact"] for row in depth["state_gate"]["prompts"])
            or depth["protocol"]["chunk"] != 2048 or depth["protocol"]["strict_chunk"] != 1024
            or depth["protocol"]["repeats"] != 3 or depth["protocol"]["decode_steps"] != 64
            or depth["model"] != left["model_identity"]
            or depth["host"]["machine_id"] != left["host"]["machine_id"]
            or depth["production_base_manifest"] != left["manifest_sha256"]
            or depth["allocation_evidence"] != left
            or any(row["current_allocated_bytes"] for row in depth["lifecycle"].values())):
        raise ValueError("invalid 2048 numerical gate")
    case_tokens = {f"{category}-p{size}": size for category in (
        "code", "general_en", "general_ja", "mixed_ja_en") for size in (512, 1024, 4096)}
    expected_cases = set(case_tokens)
    expected = {(case, "strict", 0) for case in expected_cases} | {
        (case, "candidate", repeat) for case in expected_cases for repeat in range(3)}
    traces = depth["chunk_dispatches"]
    if len(traces) != 48 or {(r["case_id"], r["arm"], r["repeat"]) for r in traces} != expected:
        raise ValueError("incomplete chunk dispatch evidence")
    for trace in traces:
        tokens = case_tokens[trace["case_id"]]
        size = 1024 if trace["arm"] == "strict" else 2048
        if trace["chunks"] != [min(size, tokens - start) for start in range(0, tokens, size)]:
            raise ValueError("wrong executed chunk split")
    workspace = packets["resume-chunk-workspace-check.json"]
    check = workspace["workspace_check"]
    if (workspace["status"] != "workspace_check_passed" or not workspace["source"]["tracked_clean"]
            or not check["logits_exact"] or not check["state_exact"]
            or check["rows"] != 5 or workspace["after_close"]["current_allocated_bytes"]
            or check["native_chunks"] != [1024] * 4
            or check["borrowed_chunks"] != [1024] * 4):
        raise ValueError("workspace borrowing equivalence failed")
    for arm, size in (("before", 1024), ("production_baseline", 2048)):
        description = workspace["workspaces"][arm]
        if any(description[key] != size for key in ("chunk_size", "token_capacity", "metadata_rows")):
            raise ValueError("workspace is not correctly sized")
    ab = packets["resume-chunk2048-workspace-ab.json"]
    samples = ab["samples"]
    expected_samples = {(case, arm, repeat) for case in expected_cases
                        for arm in ("before", "production_baseline") for repeat in range(3)}
    if (ab["status"] != "completed" or not ab["source"]["tracked_clean"]
            or len(samples) != 72
            or {(r["case_id"], r["arm"], r["repetition"]) for r in samples} != expected_samples
            or not ab["protocol"]["shared_decode_graphs"]
            or ab["protocol"]["chunk_by_arm"] != {"before": 1024, "production_baseline": 2048}
            or ab["arm_overrides"] != {"before": {}, "production_baseline": {}}
            or ab["after_close"]["current_allocated_bytes"]
            or any(value for group in ab["donor_graph_stats"].values() for value in group.values())):
        raise ValueError("invalid chunk performance comparison")
    request_speedups = {}
    for case in sorted(expected_cases):
        rows = [row for row in samples if row["case_id"] == case]
        for key in ("output_token_ids_sha256", "logits_sha256", "state_sha256"):
            if len({row[key] for row in rows}) != 1:
                raise ValueError("chunk comparison output or state mismatch")
        for row in rows:
            size = 1024 if row["arm"] == "before" else 2048
            tokens = case_tokens[case]
            if (row["active_chunk_size"] != size or row["prompt_tokens"] != tokens
                    or row["prefill_chunks"] != [min(size, tokens - start)
                                                for start in range(0, tokens, size)]
                    or row["decode_transitions"] != 128 or not row["finite"]
                    or any(not math.isfinite(row[key]) or row[key] <= 0
                           for key in ("prefill_ms", "decode_ms", "client_wall_s"))):
                raise ValueError("wrong timed chunk workload")
        walls = {arm: sum(row["client_wall_s"] for row in rows if row["arm"] == arm)
                 for arm in ("before", "production_baseline")}
        request_speedups[case] = walls["before"] / walls["production_baseline"]
    summary = summarize_cost(samples, 3)
    if summary != ab["comparisons"]["production_baseline"]:
        raise ValueError("timing summary does not reproduce")
    compact_keys = (
        "case_id", "category", "prompt_tokens", "repetition", "arm", "mode",
        "sequence_slot", "prefill_ms", "decode_ms", "client_wall_s", "active_chunk_size",
        "prefill_chunks", "decode_transitions", "output_token_count",
        "output_token_ids_sha256", "logits_sha256", "state_sha256", "finite", "memory_delta",
    )
    packets["resume-chunk2048-workspace-ab.json"] = {
        **ab, "samples": [{key: row[key] for key in compact_keys} for row in samples]}
    return dict(
        schema=1, performance_claim=True, promotion_claim=False,
        source=ab["source"], host=ab["host"], model=ab["model"], command=ab["command"],
        execution_profile="production", quant="UD-Q4_K_XL", kv="BF16",
        environment={
            "HIPENGINE_HIP_ARCH": "gfx1151", "HIPENGINE_REQUIRE_CACHED_BUILD": "1",
            "GPU_MAX_HW_QUEUES": "2", "PYTHONPATH": ".",
            "PATH": "/home/lhl/miniforge3/envs/therock/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": ":".join(
                "/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/"
                "_rocm_sdk_devel/" + suffix for suffix in ("lib", "lib64", "lib/llvm/lib")),
        },
        inference_scope="Canonical c1 512/1K/4K:64 teacher-forced steps/three repeats; "
                        "performance128 AR transitions/three pairs",
        status="2048_measured_tradeoff_further_admission_pending_4096_accounting_blocker",
        performance_summary=summary, complete_request_speedups=request_speedups,
        raw_sha256=hashes, captures=packets,
        limits=[
            "Constructor-only 2048 pass omits lazy queues and is diagnostic.",
            "2048 improves measured 4K performance but has small short-prompt costs; default remains1024.",
            "Active-shape task/isolation and wider admission remain; no universal speedup or new default claim.",
            "4096 allocates physically; its failure is under-accounted scratch, not device OOM.",
            "No hidden-seed export, graph capture, driver scratch, native-depth or c2 inference claim.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root), indent=2) + "\n")
