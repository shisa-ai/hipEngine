"""Compact the complete named-production baseline without changing its verdict."""

import argparse
import hashlib
import json
from pathlib import Path


def build(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["candidate"]["candidate_id"] != "production_baseline":
        raise ValueError("not a named production baseline")
    if data["candidate"]["environment"] or not data["profiles"]["candidate_named_profile_intact"]:
        raise ValueError("baseline has overrides")
    if data["protocol"]["teacher_forced_rows"] != 594 or data["protocol"]["prompt_count"] != 18:
        raise ValueError("incomplete baseline")
    return {
        "schema": 1,
        "kind": "journey_named_production_baseline",
        "measurement_created_at": data["created_at"],
        "performance_claim": False,
        "status": data["status"],
        "measurement_valid": data["measurement_valid"],
        "raw_path": str(path),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source": data["source"],
        "host": data["host"],
        "model": data["model"],
        "model_identity": {
            "algorithm": "sha256-directory-manifest-v1",
            "fingerprint": "fb1f2fbf73d588c9ac27f24bade5663bd3da8ac1862f62ee5bf457578a88ec53",
            "file_count": 4,
            "size_bytes": 111334654784,
            "sampled_bytes": 12582912,
            "verification": "independently recomputed before recording this result",
        },
        "command": data["command"],
        "environment": {
            "GPU_MAX_HW_QUEUES": "2",
            "HIPENGINE_HIP_ARCH": "gfx1151",
            "runtime_root": "/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel",
            "compiler_version_file": "/tmp/hipengine-journey-hipcc-version-20260913.txt",
        },
        "protocol": data["protocol"],
        "manifest_sha256": data["profiles"]["candidate_manifest_sha256"],
        "strict_manifest_sha256": data["profiles"]["strict_manifest_sha256"],
        "quality": data["quality"]["quality"],
        "repeat_determinism_passed": data["quality"]["repeat_determinism"]["passed"],
        "state_repeat_passed": data["state_repeat_gate"]["passed"],
        "lifecycle": data["lifecycle"],
        "task": {
            "status": data["task_gate"]["status"],
            "repeat_exact": data["task_gate"]["candidate_repeat_exact"],
            "strict_exact_count": data["task_gate"]["strict_exact_count"],
            "total": data["task_gate"]["total"],
            "divergent_prompt_ids": [row["id"] for row in data["task_gate"]["divergences"]],
            "note": "Deterministic text differences require task review, not automatic semantic failure.",
        },
        "decision": {
            "new_arithmetic_promotion": "blocked_by_incumbent_numerical_failure",
            "thresholds_changed": False,
            "runtime_defaults_changed": False,
            "next": "Localize shape/category/tail failures; exact-preserving screens may continue.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("artifact.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.raw), indent=2) + "\n")


if __name__ == "__main__":
    main()
