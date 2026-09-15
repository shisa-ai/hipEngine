"""Preserve the counted multi-column GDN short gate on corrected production."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(path, depth_path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    if (not packet["measurement_valid"] or not packet["source"]["tracked_clean"]
            or packet["candidate"]["candidate_id"] != "production_gdn_multi_restore"
            or packet["candidate_dispatch_calls"] <= 0
            or packet["quality"]["quality"]["summary"]["rows"] != 594
            or not packet["decision"]["numerical_passed"]
            or not packet["decision"]["state_passed"]
            or not packet["decision"]["lifecycle_passed"]):
        raise ValueError("incomplete or invalid multi-column short qualification")
    depth_raw = depth_path.read_bytes()
    depth = json.loads(depth_raw)
    if (depth["status"] != "completed" or not depth["source"]["tracked_clean"]
            or depth["candidate"] != "production_gdn_multi_restore"
            or depth["overrides"] != packet["candidate"]["environment"]
            or depth["host"]["machine_id"] != packet["host"]["machine_id"]
            or depth["arithmetic_class"] != "T2"
            or depth["candidate_dispatch_calls"] != 1080
            or depth["candidate_dispatch_mode"] != "registry"
            or depth["candidate_registered_key"][-1] != "qwen4exp_gdn_tiled16_multi_prefill"
            or not depth["protocol"]["complete_fixture"]
            or len(depth["protocol"]["case_ids"]) != 12
            or depth["protocol"]["chunk"] != 1024
            or depth["protocol"]["decode_steps"] != 64
            or depth["protocol"]["repeats"] != 3
            or depth["quality"]["summary"]["rows"] != 780
            or not depth["quality"]["hard_gates_passed"]
            or not depth["deterministic"] or not depth["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in depth["lifecycle"].values())):
        raise ValueError("incomplete or invalid multi-column depth qualification")
    directory = Path(__file__).parent
    task_review = json.loads((directory / "task-review.json").read_bytes())
    for name, expected in (
        ("remaining-task-capture.json", task_review["evidence"]["remaining_capture_sha256"]),
        ("rate-limiter-review.json", task_review["evidence"]["rate_limiter_review_sha256"]),
    ):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("task review capture hash mismatch")
    if (task_review["candidate"] != depth["candidate"]
            or task_review["verdict"] != "not_admitted_material_native_support_misstatement"
            or task_review["full_task_suite_passed"]
            or task_review["promotion_claim"]):
        raise ValueError("inconsistent task decision")
    return dict(
        schema=3, status="numerical_pass_not_admitted_task_finding",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        capture=packet,
        depth_raw_sha256=hashlib.sha256(depth_raw).hexdigest(), depth_capture=depth,
        task_review=task_review,
        limits=[
            "Complete rate-limiter requested API review finds no new defect; shared omitted tests remain.",
            "The single-column GDN task failure does not automatically reject this different variant.",
            "Multi-column has its own native-support prose task finding; no isolation or timing promotion follows.",
            "One Japanese-plan pair is truncated; complete-EOS suite certification is not claimed.",
            "Historical short capture retains T1 metadata; current review and depth arm classify this as T2.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--depth-capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.capture, args.depth_capture), indent=2) + "\n")
