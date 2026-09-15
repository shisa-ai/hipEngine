"""Preserve independently counted GR-down numerical evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    flags = {"HIPENGINE_QWEN4_EXP_" + name: "1" if name == "GR_IU8_DOWN" else "0"
             for name in ("GR_IU8", "GR_IU8_DOWN", "Q8_IU8_WMM", "Q8_MMQ_PREFILL")}
    shapes = [{"arguments": [512, 10240, 320], "calls": 1152},
              {"arguments": [1024, 10240, 320], "calls": 5760}]
    if (packet["status"] != "completed" or not packet["source"]["tracked_clean"]
            or packet["candidate"] != "production_gr_down_restore"
            or packet["overrides"] != flags
            or packet["candidate_dispatch_shapes"] != shapes
            or packet["candidate_dispatch_calls"] != 6912
            or packet["candidate_dispatch_mode"] != "direct_alias"
            or packet["quality"]["summary"]["rows"] != 780
            or packet["quality"]["hard_gates_passed"]
            or not packet["deterministic"] or not packet["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in packet["lifecycle"].values())):
        raise ValueError("invalid or insufficient isolated GR-down failure evidence")
    return dict(
        schema=1, status="isolated_gr_down_numerical_failure",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(), capture=packet,
        limitations=[
            "The unchanged GR-down IU8 path fails independently; GR-up and generic dense Q8 were off.",
            "No task or performance measurement follows this binding numerical failure.",
            "Passing max KL cannot compensate for failed mean/p95/top1.",
            "This does not establish which operand or accumulation boundary needs correction.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.capture), indent=2) + "\n")
