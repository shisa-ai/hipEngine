"""Preserve short/depth GDN restoration evidence without asserting task admission."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def assemble(root):
    packets, hashes = {}, {}
    for name in ("resume-gdn-profile.json", "resume-gdn-depth.json", "resume-gdn-ja-task.json"):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        packets[name] = json.loads(raw)
    short = packets["resume-gdn-profile.json"]
    depth = packets["resume-gdn-depth.json"]
    task = packets["resume-gdn-ja-task.json"]
    review = json.loads(Path(__file__).with_name("task-review.json").read_bytes())
    if (review["capture_sha256"] != hashes["resume-gdn-ja-task.json"]
            or task["status"] != "captured_requires_manual_review"
            or task["candidate"] != review["candidate"]
            or not task["source"]["tracked_clean"]
            or set(task["arms"]) != {"strict", review["candidate"]}
            or task["arms"][review["candidate"]]["overrides"] != short["candidate"]["environment"]):
        raise ValueError("task review does not match capture")
    task_cases = {}
    for name, arm in task["arms"].items():
        if len(arm["cases"]) != 1:
            raise ValueError("targeted task packet must contain one prompt per arm")
        case = arm["cases"][0]
        if (case["id"] != review["prompt_id"] or not case["deterministic"]
                or len(case["repeats"]) != 2
                or case["repeats"][0] != case["repeats"][1]
                or case["repeats"][0]["finish_reason"] != "eos"):
            raise ValueError("incomplete or nonrepeatable targeted task")
        task_cases[name] = {**case, "repeats": [case["repeats"][0]]}
        expected_tokens = review["strict_tokens" if name == "strict" else "candidate_tokens"]
        if len(case["repeats"][0]["ids"]) != expected_tokens:
            raise ValueError("review token count mismatch")
    if any(row["current_allocated_bytes"] for row in task["lifecycle"].values()):
        raise ValueError("task ownership did not close")
    subprocess.run(
        ["git", "diff", "--exit-code", short["source"]["head"], task["source"]["head"],
         "--", "hipengine"],
        cwd=Path(__file__).resolve().parents[3], check=True, capture_output=True)
    if (not short["measurement_valid"] or not short["decision"]["numerical_passed"]
            or not short["decision"]["state_passed"] or not short["lifecycle"]["passed"]
            or short["candidate_dispatch_calls"] <= 0):
        raise ValueError("invalid short qualification")
    if (depth["status"] != "completed" or not depth["quality"]["hard_gates_passed"]
            or not depth["deterministic"] or not depth["state_gate"]["passed"]
            or depth["candidate_dispatch_calls"] <= 0
            or any(row["current_allocated_bytes"] for row in depth["lifecycle"].values())):
        raise ValueError("invalid depth qualification")
    if (short["quality"]["quality"]["summary"]["rows"] != 594
            or depth["quality"]["summary"]["rows"] != 780
            or short["source"] != depth["source"]
            or short["host"]["machine_id"] != depth["host"]["machine_id"]
            or short["candidate"]["environment"] != depth["overrides"]):
        raise ValueError("incomplete or mismatched qualification")
    return dict(
        schema=1, status="numerical_pass_task_rejected",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashes,
        short={key: short[key] for key in (
            "source", "host", "model", "command", "candidate", "candidate_dispatch_calls",
            "measurement_valid", "decision", "quality", "task_gate", "lifecycle",
            "state_repeat_gate")},
        depth=depth,
        task={key: task[key] for key in (
            "source", "host", "model", "command", "candidate", "complete_suite", "lifecycle")},
        task_cases=task_cases, task_review=review,
        limits=[
            "One complete Japanese response fails the predeclared task criterion; expected quality is not established.",
            "No performance run or default promotion in this packet.",
            "The previously failed full production stack is not retroactively admitted.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root), indent=2) + "\n")
