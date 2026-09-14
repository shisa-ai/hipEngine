"""Preserve the independently counted GR-up numerical rejection."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    expected_flags = {
        "HIPENGINE_QWEN4_EXP_" + flag: "1" if flag == "GR_IU8" else "0"
        for flag in ("GR_IU8", "GR_IU8_DOWN", "Q8_IU8_WMM", "Q8_MMQ_PREFILL")}
    expected_shapes = [
        {"arguments": [512, 320, 10240], "calls": 1152},
        {"arguments": [1024, 320, 10240], "calls": 5760}]
    if (packet["status"] != "completed" or not packet["source"]["tracked_clean"]
            or packet["candidate"] != "production_gr_up_restore"
            or packet["overrides"] != expected_flags
            or packet["candidate_dispatch_calls"] != 6912
            or packet["candidate_dispatch_shapes"] != expected_shapes
            or packet["candidate_dispatch_mode"] != "direct_alias"
            or packet["quality"]["summary"]["rows"] != 780
            or packet["quality"]["hard_gates_passed"]
            or not packet["deterministic"] or not packet["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in packet["lifecycle"].values())):
        raise ValueError("capture does not establish the declared isolated failure")
    return dict(
        schema=1, status="isolated_gr_up_numerical_failure",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(), capture=packet,
        limitations=[
            "This rejects the unchanged GR-up IU8 arithmetic, not all GR optimizations.",
            "Passing leaf tolerance and maximum KL cannot compensate for failed mean/p95/top1.",
            "No task or performance run spent after this binding numerical failure.",
            "Weight/activation/accumulation boundary localization remains open before a correction.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.capture), indent=2) + "\n")
