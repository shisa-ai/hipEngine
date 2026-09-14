"""Preserve GDN isolation diagnostics without raw logits or GPU traces."""

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def assemble(raw_root):
    screen, screen_hash = load(raw_root / "gdn-isolation.json")
    full, full_hash = load(raw_root / "gdn-isolated-profile-clean.json")
    exploratory, exploratory_hash = load(raw_root / "gdn-isolated-profile.json")
    if screen["status"] != "completed" or not full["measurement_valid"]:
        raise ValueError("incomplete or invalid diagnostic")
    if screen["arms"]["all_flags_strict"]["quality"]["summary"]["kl_max"] != 0:
        raise ValueError("strict control did not reproduce")
    if full["quality"]["quality"]["summary"]["rows"] != 594:
        raise ValueError("full 594-row gate required")
    return {
        "kind": "gdn_numerical_isolation",
        "performance_claim": False,
        "promotion_claim": False,
        "raw_sha256": {"screen": screen_hash, "full": full_hash,
                       "exploratory": exploratory_hash},
        "exploratory": {
            "measurement_valid": exploratory["measurement_valid"],
            "source": exploratory["source"],
            "quality": exploratory["quality"]["quality"]["summary"],
            "reason_not_qualification": "uncommitted tracked harness",
        },
        "screen": screen,
        "full": {key: full[key] for key in (
            "candidate", "command", "host", "model", "source", "protocol",
            "profiles", "quality", "state_repeat_gate", "task_gate",
            "lifecycle", "decision", "measurement_valid", "status",
        )},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).with_name("artifact.json")
    output.write_text(json.dumps(assemble(args.raw_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
