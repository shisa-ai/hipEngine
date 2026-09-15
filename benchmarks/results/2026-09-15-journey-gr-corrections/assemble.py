"""Assemble operand-screen evidence for default-off GR corrections."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def assemble(root, trace):
    captures, hashes = {}, {}
    for variant, filename in (
        ("original", "resume-gr-operands.json"),
        ("p4", "resume-gr-p4-operands.json"),
        ("compensated", "resume-gr-compensated-operands.json"),
    ):
        raw = (root / filename).read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        packet = json.loads(raw)
        if (packet["status"] != "completed_diagnostic" or not packet["source"]["tracked_clean"]
                or not packet["control_logits_state_exact"]
                or packet["after_close"]["current_allocated_bytes"] or len(packet["records"]) != 12):
            raise ValueError("invalid operand replay")
        captures[variant] = packet
    ratios = {}
    original = {row["weight"]: row for row in captures["original"]["records"]}
    for variant in ("p4", "compensated"):
        packet = captures[variant]
        if packet["model"] != captures["original"]["model"] or packet["projection_variant"] != variant:
            raise ValueError("mismatched model or variant")
        if packet["host"]["machine_id"] != captures["original"]["host"]["machine_id"]:
            raise ValueError("different physical hosts")
        ratios[variant] = []
        for row in packet["records"]:
            before = original[row["weight"]]
            for key in ("input_sha256", "sampled_weight_sha256", "shape", "sampled_rows", "sampled_columns"):
                if row[key] != before[key]:
                    raise ValueError("operand comparisons are not aligned")
            ratios[variant].append(dict(
                weight=row["weight"], leg=row["leg"],
                original_over_candidate_mse=before["fp64_vs_candidate"]["mse"] / row["fp64_vs_candidate"]["mse"],
                candidate_over_strict_mse=row["fp64_vs_candidate"]["mse"] / row["fp64_vs_parent"]["mse"]))
    trace_rows = list(csv.DictReader(trace.open()))
    kernels = {}
    for row in trace_rows:
        name = row["Kernel_Name"]
        if "q8_0_iu8_wmma_prefill_f32_f32_kernel<" not in name:
            continue
        record = kernels.setdefault(name, dict(durations_ns=[], vgpr=int(row["VGPR_Count"]),
                                               lds=int(row["LDS_Block_Size"]), scratch=int(row["Scratch_Size"])))
        duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
        if duration <= 0:
            raise ValueError("invalid trace duration")
        record["durations_ns"].append(duration)
    for suffix in ("<4, false>", "<3, true>"):
        if not any(suffix in name for name in kernels):
            raise ValueError("correction specialization not traced")
    model_path = root / "resume-gr-down-compensated-depth.json"
    model_raw = model_path.read_bytes()
    model_gate = json.loads(model_raw)
    hashes[model_path.name] = hashlib.sha256(model_raw).hexdigest()
    expected_shapes = [{"arguments": [512, 10240, 320], "calls": 1152},
                       {"arguments": [1024, 10240, 320], "calls": 5760}]
    if (model_gate["status"] != "completed" or not model_gate["source"]["tracked_clean"]
            or model_gate["candidate"] != "production_gr_down_compensated"
            or model_gate["candidate_dispatch_mode"] != "registry"
            or model_gate["candidate_dispatch_calls"] != 6912
            or model_gate["candidate_dispatch_shapes"] != expected_shapes
            or model_gate["quality"]["summary"]["rows"] != 780
            or model_gate["quality"]["hard_gates_passed"]
            or not model_gate["deterministic"] or not model_gate["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in model_gate["lifecycle"].values())):
        raise ValueError("invalid compensated model rejection evidence")
    p4_path = root / "resume-gr-up-p4-depth.json"
    p4_raw = p4_path.read_bytes()
    p4_gate = json.loads(p4_raw)
    hashes[p4_path.name] = hashlib.sha256(p4_raw).hexdigest()
    if (p4_gate["status"] != "completed" or not p4_gate["source"]["tracked_clean"]
            or p4_gate["candidate"] != "production_gr_up_p4"
            or p4_gate["candidate_dispatch_mode"] != "registry"
            or p4_gate["candidate_dispatch_calls"] != 6912
            or p4_gate["candidate_dispatch_shapes"] != [
                {"arguments": [512, 320, 10240], "calls": 1152},
                {"arguments": [1024, 320, 10240], "calls": 5760}]
            or p4_gate["quality"]["summary"]["rows"] != 780
            or p4_gate["quality"]["hard_gates_passed"]
            or not p4_gate["deterministic"] or not p4_gate["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in p4_gate["lifecycle"].values())):
        raise ValueError("invalid P4 model rejection evidence")
    for packet in captures.values():
        packet["records"] = [{key: value for key, value in row.items()
                              if key not in ("sampled_fp64", "sampled_parent", "sampled_candidate")}
                             for row in packet["records"]]
    return dict(
        schema=1, status="both_corrections_fail_model_gates_removed",
        performance_claim=False, promotion_claim=False, raw_sha256=hashes,
        captures=captures, ratios=ratios, compensated_model_gate=model_gate, p4_model_gate=p4_gate,
        trace=dict(sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), kernels=kernels),
        limits=[
            "Sampled FP64 MSE is not the production model admission criterion.",
            "P4 carries higher VGPR/LDS use; throughput is not measured here.",
            "Compensated down fails full model numerics despite better projection MSE; no cost/task run follows.",
            "P4-up also fails full model numerics; both correction paths and selectors are removed.",
            "Historical replay and gate commands require their pinned source revisions.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root, args.trace), indent=2) + "\n")
