"""Preserve the counted multi-column GDN short gate on corrected production."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(path):
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
    return dict(
        schema=1, status="short_numerical_pass_depth_task_performance_pending",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        capture=packet,
        limits=[
            "One incomplete rate-limiter code response differs; this is not a semantic rejection.",
            "The single-column GDN task failure does not automatically reject this different variant.",
            "Canonical depth, applicable task, and current-model performance are not qualified here.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.capture), indent=2) + "\n")
