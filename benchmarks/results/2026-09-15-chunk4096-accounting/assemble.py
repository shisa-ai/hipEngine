"""Reconcile allocator census, historical device totals and repaired admission."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(root):
    names = (
        "resume-scratch-census.json", "resume-scratch-census-native.json",
        "resume-chunk2048-lazy-allocation.json", "resume-chunk4096-lazy-allocation.json",
        "resume-chunk4096-native-c2-accounted.json",
        "resume-chunk4096-depth.json",
    )
    packets, hashes = {}, {}
    for name in names:
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        packets[name] = json.loads(raw)
    cpu = packets[names[0]]
    native_cpu = packets[names[1]]
    native = packets["resume-chunk4096-native-c2-accounted.json"]
    reconciled = []
    for packet in (cpu, native_cpu):
        if not packet["source"]["tracked_clean"]:
            raise ValueError("census source must be clean")
        for record in packet["records"]:
            if (sum(record["owner_bytes"].values()) != record["prepared_bytes"]
                    or sum(row["nbytes"] for row in record["allocations"]) != record["prepared_bytes"]
                    or record["teardown_bytes"]):
                raise ValueError("CPU census does not reconcile")
    for chunk in (2048, 4096):
        record = next(row for row in cpu["records"] if row["chunk"] == chunk)
        old = packets[f"resume-chunk{chunk}-lazy-allocation.json"]
        plan = old["admission"]["plan"]
        if (record["context"] != 4352 or old["prepared_context"] != 4352
                or plan["device_weight_bytes"] + record["prepared_bytes"]
                != old["prepared_memory"]["current_allocated_bytes"]
                or old["memory_after_close"]["current_allocated_bytes"]):
            raise ValueError("historical device allocation does not match real recipes")
        scratch = record["prepared_bytes"] - sum(plan[key] for key in (
            "kv_bytes", "index_bytes", "runtime_state_bytes"))
        reconciled.append(dict(
            chunk=chunk, context=4352, runner_bytes=record["prepared_bytes"],
            device_scratch_bytes=scratch, prior_scratch_bytes=plan["scratch_bytes"],
            prior_device_margin_bytes=plan["scratch_bytes"] - scratch,
            prior_combined_margin_bytes=old["allocation_margins"]["scratch_margin_bytes"]))
    plan = native["admission"]["plan"]
    record = native_cpu["records"][0]
    if (native["schema"] != 3 or native["status"] != "passed"
            or not native["source"]["tracked_clean"]
            or native["prepared_context"] != 262144 or native["prepared_runners"] != 2
            or native["chunk_size"] != 4096
            or record["context"] != 262144 or record["chunk"] != 4096
            or plan["reserve_bytes"] != 4 * 1024**3
            or native["device_allocation_margins"]["device_scratch_margin_bytes"] != 0
            or plan["device_weight_bytes"] + 2 * record["prepared_bytes"]
            != native["prepared_memory"]["current_allocated_bytes"]
            or native["memory_after_close"]["current_allocated_bytes"]
            or native["model_identity"] != packets[names[2]]["model_identity"]):
        raise ValueError("repaired native-c2 allocation does not reconcile")
    expected_scratch = 2 * record["prepared_bytes"] - sum(plan[key] for key in (
        "kv_bytes", "index_bytes", "runtime_state_bytes"))
    if expected_scratch != plan["scratch_bytes"]:
        raise ValueError("mandatory scratch differs from census")
    if (len(native["lazy_group_risk"]) != 2
            or any(group["queues"] != record["repair_queues"] for group in native["lazy_group_risk"])):
        raise ValueError("native queue preparation differs from census")
    depth = packets["resume-chunk4096-depth.json"]
    if (depth["status"] != "completed" or not depth["source"]["tracked_clean"]
            or depth["candidate"] != "production_baseline" or depth["overrides"]
            or depth["protocol"]["chunk"] != 4096 or depth["protocol"]["strict_chunk"] != 1024
            or not depth["protocol"]["complete_fixture"] or depth["protocol"]["decode_steps"] != 64
            or depth["protocol"]["repeats"] != 3 or depth["quality"]["summary"]["rows"] != 780
            or not depth["quality"]["hard_gates_passed"]
            or depth["quality"]["summary"]["max_abs_logit_delta"] != 0
            or not depth["deterministic"] or not depth["state_gate"]["passed"]
            or not all(row["strict_candidate_state_exact"] for row in depth["state_gate"]["prompts"])
            or depth["host"]["machine_id"] != native["host"]["machine_id"]
            or depth["model"] != native["model_identity"]
            or any(row["current_allocated_bytes"] for row in depth["lifecycle"].values())):
        raise ValueError("invalid chunk4096 canonical gate")
    if len(depth["chunk_dispatches"]) != 48:
        raise ValueError("missing canonical chunk traces")
    for trace in depth["chunk_dispatches"]:
        prompt = int(trace["case_id"].rsplit("p", 1)[1])
        size = 1024 if trace["arm"] == "strict" else 4096
        whole, tail = divmod(prompt, size)
        if trace["chunks"] != [size] * whole + ([tail] if tail else []):
            raise ValueError("canonical chunk trace mismatch")
    return dict(
        schema=2, status="accounting_and_canonical_numerics_passed_more_gates_pending",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashes, bounded_reconciliation=reconciled,
        captures=packets,
        limits=[
            "Canonical numerics pass; boundary/full-payload, active-task, c2 inference and performance gates remain.",
            "Native-c2 allocation is not native-depth generation qualification.",
            "The mandatory footprint excludes optional MMQ, graph, verification and transaction resources.",
            "The4GiB scratch floor and separate4GiB reserve remain; larger mandatory buffers raise accounting.",
            "Host staging is reserved separately and is not credited against device scratch.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root), indent=2) + "\n")
