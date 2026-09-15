"""Preserve prefill-only owner regression and allocation evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(root):
    path = root / "resume-owned-workspace-check.json"
    raw = path.read_bytes()
    packet = json.loads(raw)
    ref_path = root / "resume-chunk4096-c2-deferred.json"
    ref_raw = ref_path.read_bytes()
    reference = json.loads(ref_raw)
    census_path = root / "resume-scratch-census.json"
    census_raw = census_path.read_bytes()
    census = json.loads(census_raw)
    small = next(row for row in census["records"] if row["chunk"] == 1024)
    prefill_keys = ("gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
                    "qsa_prefill_metadata", "_prefill_buffers")
    expected_extra = sum(small["owner_bytes"][key] for key in prefill_keys)
    if (packet["status"] != "passed" or not packet["source"]["tracked_clean"]
            or packet["model"] != reference["model"]
            or packet["host"]["machine_id"] != reference["host"]["machine_id"]
            or packet["manifest"] != reference["manifest"]
            or packet["reference_sha256"] != hashlib.sha256(ref_raw).hexdigest()
            or packet["extra_workspace_bytes"] != expected_extra or packet["owner_count"] != 2
            or packet["reserve_bytes"] != 4 << 30
            or packet["free_before_extra"] < expected_extra + packet["reserve_bytes"]
            or packet["free_after_extra"] < packet["reserve_bytes"]
            or packet["memory_after_close"]["current_allocated_bytes"]
            or len(packet["cases"]) != 9):
        raise ValueError("invalid prefill-only owner capture")
    expected = {(role, repeat) for role in ("a", "b", "c") for repeat in range(3)}
    if {(row["role"], row["repeat"]) for row in packet["cases"]} != expected:
        raise ValueError("incomplete owner repeat coverage")
    lengths = {"a": 2052, "b": 4097, "c": 2049}
    for row in packet["cases"]:
        if (not row["exact"] or row["rows"] != 9
                or row["final_state"] != reference["references"][row["role"]][8]["state"]):
            raise ValueError("owned workspace state differs from frozen reference")
        for key, size in (("native_chunks", 4096), ("borrowed_chunks", 1024)):
            count, tail = divmod(lengths[row["role"]], size)
            if row[key] != [size] * count + ([tail] if tail else []):
                raise ValueError("owned workspace chunk coverage differs")
    return dict(
        schema=1, status="owned_prefill_storage_verified_automatic_selection_pending",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        reference_sha256=hashlib.sha256(ref_raw).hexdigest(),
        census_sha256=hashlib.sha256(census_raw).hexdigest(), capture=packet,
        allocation_comparison=dict(
            context=4352, extra_chunk=1024, measured_prefill_owner_bytes=expected_extra,
            full_runner_recipe_bytes=small["prepared_bytes"],
            derived_duplicate_state_bytes_avoided=small["prepared_bytes"] - expected_extra),
        limits=[
            "No automatic dispatch or new throughput claim.",
            "27 unique logit rows repeated three times; this is an ownership regression check, not a new statistical numerical gate.",
            "Avoided duplicate-state bytes are derived from the matched-capacity validated allocation census.",
            "Primary allocation order and sizes match the frozen CPU census; no kernel arithmetic changed.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root), indent=2) + "\n")
