"""Reconcile allocator census, historical device totals and repaired admission."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.qwen4exp_chunk_c2_gate import validate_traces
from scripts.qwen4exp_conservative_cost import summarize_cost


def assemble(root):
    names = (
        "resume-scratch-census.json", "resume-scratch-census-native.json",
        "resume-chunk2048-lazy-allocation.json", "resume-chunk4096-lazy-allocation.json",
        "resume-chunk4096-native-c2-accounted.json",
        "resume-chunk4096-depth.json",
        "resume-chunk4096-boundaries.json",
        "resume-chunk4096-active-tasks.json",
        "resume-chunk4096-c2.json", "resume-chunk4096-c2-deferred.json",
        "resume-chunk4096-workspace-check.json", "resume-chunk4096-workspace-ab.json",
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
    boundary = packets["resume-chunk4096-boundaries.json"]
    lengths = (2047, 2048, 2049, 2051, 2052, 4095, 4097)
    ids = [f"{category}-boundary{length}" for category in ("code", "general_ja") for length in lengths]
    if (boundary["status"] != "completed" or not boundary["source"]["tracked_clean"]
            or boundary["protocol"] != {
                "steps": 64, "candidate_repeats": 3, "capacity": 4352,
                "strict_chunk": 1024, "candidate_chunk": 4096, "complete_matrix": True}
            or boundary["model"] != depth["model"]
            or boundary["host"]["machine_id"] != native["host"]["machine_id"]
            or boundary["quality"]["summary"]["rows"] != 910
            or boundary["quality"]["summary"]["max_abs_logit_delta"] != 0
            or not boundary["quality"]["hard_gates_passed"] or not boundary["deterministic"]
            or not boundary["state_gate"]["passed"] or not boundary["payload_exact_vs_strict"]
            or [row["id"] for row in boundary["cases"]] != ids
            or any(row["current_allocated_bytes"] for row in boundary["lifecycle"].values())):
        raise ValueError("invalid chunk4096 boundary gate")
    for case in boundary["cases"]:
        if (len(case["candidate_payloads"]) != 3
                or case["intervening_prefill_chunks"] != [[257], [257]]
                or not all(case[key] for key in (
                    "deterministic", "control_exact", "state_finite", "payload_exact_vs_strict"))):
            raise ValueError("missing boundary reuse evidence")
        for phase in ("prefill", "final"):
            expected = case["strict_payload"][phase]
            if (expected["full_kv_bytes"] <= 0 or expected["live_index_bytes"] <= 0
                    or not expected["full_kv_finite"] or not expected["live_index_finite"]
                    or not expected["recurrent"]["finite"]
                    or any(row[phase] != expected for row in case["candidate_payloads"])):
                raise ValueError("boundary payloads do not match")
        for row, size in [(case["strict_payload"], 1024)] + [
                (row, 4096) for row in case["candidate_payloads"]]:
            count, tail = divmod(case["prompt_tokens"], size)
            if row["chunks"] != [size] * count + ([tail] if tail else []):
                raise ValueError("boundary chunk trace mismatch")
    tasks = packets["resume-chunk4096-active-tasks.json"]
    if (tasks["status"] != "passed_supplemental" or not tasks["source"]["tracked_clean"]
            or tasks["candidate"] != "production_baseline" or tasks["overrides"]
            or tasks["model"] != depth["model"]
            or tasks["protocol"] != {
                "context": 4096, "repeats": 3, "max_tokens": 64,
                "strict_chunk": 1024, "candidate_chunk": 4096}
            or len(tasks["comparisons"]) != 6
            or any(not all(row[key] for key in (
                "valid", "strict_correct", "candidate_correct", "output_ids_exact"))
                   for row in tasks["comparisons"])
            or any(row["current_allocated_bytes"] for row in tasks["lifecycle"].values())):
        raise ValueError("invalid chunk4096 active tasks")
    for arm, chunks in (("strict", [1024] * 4), ("candidate", [4096])):
        cases = tasks["cases"][arm]
        if [case["id"] for case in cases] != [row["id"] for row in tasks["comparisons"]]:
            raise ValueError("active task coverage mismatch")
        for case in cases:
            if (not case["correct"] or not case["repeated"] or len(case["runs"]) != 3
                    or case["prompt"]["prompt_format"] != "qwen4exp_embedded"
                    or case["prompt"]["enable_thinking"]
                    or any(run != case["runs"][0] or not run["finite"]
                           or run["finish"] != "eos" or run["prefill_chunks"] != chunks
                           for run in case["runs"])):
                raise ValueError("invalid active task repetition or chunk")
    for before, after in zip(tasks["cases"]["strict"], tasks["cases"]["candidate"], strict=True):
        if (before["prompt"] != after["prompt"]
                or before["runs"][0]["ids"] != after["runs"][0]["ids"]
                or before["runs"][0]["state_sha256"] != after["runs"][0]["state_sha256"]):
            raise ValueError("active task parity does not reproduce")
    for name, mode, checkpoints in (
        ("resume-chunk4096-c2.json", "each_checkpoint", 28),
        ("resume-chunk4096-c2-deferred.json", "deferred", 2),
    ):
        c2 = packets[name]
        if (c2["status"] != "passed" or not c2["source"]["tracked_clean"]
                or c2["model"] != depth["model"] or c2["manifest"] != native["manifest_sha256"]
                or c2["host"]["machine_id"] != native["host"]["machine_id"]
                or c2["protocol"] != {
                    "capacity": 4352, "chunk": 4096, "resident_runners": 2,
                    "compact_outputs": True, "repeats": 3, "decode_steps": 8,
                    "inspection_mode": mode}
                or not c2["disjoint_owner_ranges"] or len(c2["owner_range_counts"]) != 2
                or min(c2["owner_range_counts"]) <= 0
                or c2["repair_queues"] != [row["queues"] for row in native["lazy_group_risk"]]
                or len(c2["repeats"]) != 3
                or c2["memory_after_close"]["current_allocated_bytes"]):
            raise ValueError("invalid chunk4096 c2 gate")
        if validate_traces(c2["chunk_traces"], {
                "a": range(2052), "b": range(4097), "c": range(2049)}, chunk=4096) != c2["trace_gate"]:
            raise ValueError("c2 chunk traces do not reproduce")
        if {row["owners"]["a"] for row in c2["repeats"]} != {0, 1}:
            raise ValueError("c2 physical owner coverage missing")
        for index, row in enumerate(c2["repeats"]):
            if (row["repeat"] != index or row["checkpoints"] != checkpoints
                    or set(row["owners"].values()) != {0, 1}
                    or row["peer_cancel_tokens"] != 2 or row["partial_cancel_tokens"] != 0
                    or row["active_cancel_tokens"] != 8 or row["replacement_cancel_tokens"] != 8):
                raise ValueError("c2 lifecycle comparison incomplete")
    if packets["resume-chunk4096-c2.json"]["references"] != packets[
            "resume-chunk4096-c2-deferred.json"]["references"]:
        raise ValueError("c2 isolated references differ across inspection modes")
    workspace = packets["resume-chunk4096-workspace-check.json"]
    if (workspace["status"] != "workspace_check_passed" or not workspace["source"]["tracked_clean"]
            or workspace["workspace_check"] != {
                "case_id": "code-p4096", "rows": 5, "logits_exact": True,
                "reference_profile": "production", "state_exact": True,
                "native_chunks": [1024] * 4, "borrowed_chunks": [1024] * 4}
            or workspace["after_close"]["current_allocated_bytes"]):
        raise ValueError("invalid workspace control")
    ab = packets["resume-chunk4096-workspace-ab.json"]
    samples = ab["samples"]
    cases = depth["protocol"]["case_ids"]
    expected = {(case, arm, repeat) for case in cases
                for arm in ("before", "production_baseline") for repeat in range(3)}
    if (ab["status"] != "completed" or not ab["source"]["tracked_clean"]
            or ab["model"] != depth["model"] or ab["host"]["machine_id"] != native["host"]["machine_id"]
            or ab["base_manifest"] != native["manifest_sha256"]
            or len(samples) != 72
            or {(r["case_id"], r["arm"], r["repetition"]) for r in samples} != expected
            or ab["arm_overrides"] != {"before": {}, "production_baseline": {}}
            or not ab["protocol"]["shared_decode_graphs"]
            or ab["protocol"]["chunk_by_arm"] != {"before": 1024, "production_baseline": 4096}
            or ab["after_close"]["current_allocated_bytes"]
            or any(value for stats in ab["donor_graph_stats"].values() for value in stats.values())):
        raise ValueError("invalid matched chunk timing")
    if (ab["graph_stats"]["before"] != ab["graph_stats"]["production_baseline"]
            or ab["graph_stats"]["before"]["moe_graph_cache"]["capture"] != 48
            or ab["graph_stats"]["before"]["moe_graph_cache"]["replay"] <= 0):
        raise ValueError("shared decode graph evidence differs")
    for arm, size in (("before", 1024), ("production_baseline", 4096)):
        if any(ab["workspaces"][arm][key] != size
               for key in ("chunk_size", "token_capacity", "metadata_rows")):
            raise ValueError("timed workspace dimensions differ")
    guard = ab["workspace_memory_guard"]
    if (guard["reserve_bytes"] != 4 * 1024**3
            or guard["free_before_donor"] < guard["extra_runner_bound"] + guard["reserve_bytes"]
            or guard["free_after_setup"] < guard["reserve_bytes"]):
        raise ValueError("timed workspace consumed reserve")
    request_speedups = {}
    for case in cases:
        rows = [row for row in samples if row["case_id"] == case]
        if any(len({row[key] for row in rows}) != 1
               for key in ("output_token_ids_sha256", "logits_sha256", "state_sha256")):
            raise ValueError("timed output/state parity failed")
        for row in rows:
            chunk = 1024 if row["arm"] == "before" else 4096
            tokens = int(case.rsplit("p", 1)[1])
            count, tail = divmod(tokens, chunk)
            if (row["active_chunk_size"] != chunk or row["prompt_tokens"] != tokens
                    or row["prefill_chunks"] != [chunk] * count + ([tail] if tail else [])
                    or row["decode_transitions"] != 128 or not row["finite"]
                    or any(not math.isfinite(row[key]) or row[key] <= 0
                           for key in ("prefill_ms", "decode_ms", "client_wall_s"))):
                raise ValueError("timed workload differs")
        walls = {arm: sum(row["client_wall_s"] for row in rows if row["arm"] == arm)
                 for arm in ("before", "production_baseline")}
        request_speedups[case] = walls["before"] / walls["production_baseline"]
    summary = summarize_cost(samples, 3)
    if summary != ab["comparisons"]["production_baseline"]:
        raise ValueError("timing summary does not reproduce")
    compact_keys = (
        "case_id", "category", "prompt_tokens", "repetition", "arm", "mode", "sequence_slot",
        "prefill_ms", "decode_ms", "client_wall_s", "active_chunk_size", "prefill_chunks",
        "decode_transitions", "output_token_count", "output_token_ids_sha256",
        "logits_sha256", "state_sha256", "finite", "memory_delta")
    packets["resume-chunk4096-workspace-ab.json"] = {
        **ab, "samples": [{key: row[key] for key in compact_keys} for row in samples]}
    return dict(
        schema=3, status="qualified_explicit4096_tradeoff_default1024",
        performance_claim=True, promotion_claim=False,
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
        performance_summary=summary, complete_request_speedups=request_speedups,
        default_decision="Keep1024 globally because short-request costs remain;4096 is a qualified explicit option.",
        raw_sha256=hashes, bounded_reconciliation=reconciled,
        captures=packets,
        limits=[
            "4096 improves4K in its matched comparison; short-request costs prevent an unscoped default change.",
            "This is not a direct2048-versus4096 experiment; do not compare independent session rates as a paired gain.",
            "Shared decode graphs control graph-instance differences, not CPU/GPU frequency; no cause is assigned to TG movement.",
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
