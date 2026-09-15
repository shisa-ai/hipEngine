"""Preserve the generic Q8 IU8 numerical gate, including its GR-down overlap."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    flags = {
        "HIPENGINE_QWEN4_EXP_" + flag: "1" if flag == "Q8_IU8_WMM" else "0"
        for flag in ("GR_IU8", "GR_IU8_DOWN", "Q8_IU8_WMM", "Q8_MMQ_PREFILL")}
    base_shapes = [
        (640, 2560, 576), (2560, 512, 288), (2560, 640, 1152),
        (2560, 2560, 12), (2560, 6144, 432), (2560, 10240, 444),
        (2560, 12288, 144), (6144, 2560, 576), (10240, 320, 1152)]
    shapes = [
        {"arguments": [rows, k, n], "calls": calls * factor}
        for rows, factor in ((512, 1), (1024, 5))
        for k, n, calls in base_shapes]
    protocol = packet["protocol"]
    if (packet["status"] != "completed" or not packet["source"]["tracked_clean"]
            or packet["candidate"] != "production_dense_q8_restore"
            or packet["overrides"] != flags
            or packet["candidate_dispatch_mode"] != "registry"
            or packet["candidate_dispatch_calls"] != 28656
            or packet["candidate_dispatch_shapes"] != shapes
            or not protocol["complete_fixture"] or len(protocol["case_ids"]) != 12
            or protocol["decode_steps"] != 64 or protocol["repeats"] != 3
            or protocol["chunk"] != 1024 or protocol["strict_chunk"] != 1024
            or packet["quality"]["summary"]["rows"] != 780
            or packet["quality"]["hard_gates_passed"]
            or not packet["deterministic"] or not packet["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in packet["lifecycle"].values())):
        raise ValueError("capture does not establish the declared generic Q8 failure")
    return dict(
        schema=1, status="generic_q8_iu8_numerical_failure",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(), capture=packet,
        limitations=[
            "Generic coverage includes 6912 GR-down calls; this is not a non-GR-only ablation.",
            "This rejects the generic switch as tested, not each covered projection independently.",
            "No task or performance run follows the binding numerical failure.",
            "Sampled state checks do not establish complete dynamic-serving isolation.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.capture), indent=2) + "\n")
