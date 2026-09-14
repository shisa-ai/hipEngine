"""Preserve current-profile constructor and lazy-queue allocation probes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.qwen4exp_chunk_memory_probe import allocation_margins


def assemble(root):
    packets, hashes = {}, {}
    for name in ("resume-chunk2048-allocation.json", "resume-chunk2048-lazy-allocation.json",
                 "resume-chunk4096-lazy-allocation.json", "resume-chunk2048-depth.json"):
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
    return dict(
        schema=1, performance_claim=False, promotion_claim=False,
        inference_scope="Canonical 512/1K/4K,64 teacher-forced decode steps,three repeats only",
        status="2048_numerical_pass_task_performance_pending_4096_accounting_blocker",
        raw_sha256=hashes, captures=packets,
        limits=[
            "Constructor-only 2048 pass omits lazy queues and is diagnostic.",
            "2048 passes bounded numerics; performance/task/isolation and wider admission remain.",
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
