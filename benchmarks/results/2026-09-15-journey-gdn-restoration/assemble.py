"""Preserve short/depth GDN restoration evidence without asserting task admission."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(root):
    packets, hashes = {}, {}
    for name in ("resume-gdn-profile.json", "resume-gdn-depth.json"):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        packets[name] = json.loads(raw)
    short = packets["resume-gdn-profile.json"]
    depth = packets["resume-gdn-depth.json"]
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
        schema=1, status="numerical_pass_task_and_performance_pending",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashes,
        short={key: short[key] for key in (
            "source", "host", "model", "command", "candidate", "candidate_dispatch_calls",
            "measurement_valid", "decision", "quality", "task_gate", "lifecycle",
            "state_repeat_gate")},
        depth=depth,
        limits=[
            "One short Japanese output differs; incomplete prefixes are not a task verdict.",
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
