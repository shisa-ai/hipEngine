"""Compact a completed canonical A/B into a same-source retention packet."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.qwen4exp_halo_box_campaign_ab import summarize_campaign_ab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if raw["status"] != "completed" or raw["diagnostic_subset"] or raw.get("screen_only",False):
        raise ValueError("requires completed full-suite A/B")
    summary = summarize_campaign_ab(raw["samples"], repetitions_per_mode=3)
    if len(raw["samples"]) != 72 or len(summary["by_case"]) != 12:
        raise ValueError("requires canonical72 samples/12 cases")
    if any(raw["memory_after_close"][k] != 0
           for k in ("active_allocations", "current_allocated_bytes")):
        raise ValueError("ownership did not close")
    provenance = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", device_name="AMD Radeon 8060S", model_path=raw["model_root"],
        model_revision="8bdc666649440e9bdc97e16f3f75782c98478ff5", quant="UD-Q4_K_XL",
        kv_dtype="BF16", command=raw["commands"]["argv"],
        timing_protocol=raw["protocol"]["timing_boundary"], warmups=1, repetitions=3,
        hipcc_version=raw["host"]["hipcc_version"], rocm_version=raw["host"]["rocm_platform"])
    if (provenance["hipengine_commit"] != raw["source"]["head"]
            or provenance["staged_dirty"] or provenance["unstaged_dirty"]):
        raise ValueError("package on the same clean runtime source before promotion")
    totals = defaultdict(lambda: defaultdict(float))
    for row in raw["samples"]:
        totals[row["case_id"]][row["mode"]] += row["prefill_ms"] + row["decode_ms"]
    walls = {k: v["before"] / v["after"] for k, v in totals.items()}
    if not all(v > 1 for v in walls.values()) or not all(
            v["after_over_before_prefill"] > 1 for v in summary["by_case"].values()):
        raise ValueError("this retention helper requires every prefill and request-wall case to improve")
    packet = {k: raw[k] for k in (
        "host", "source", "commands", "model_root", "model_identity", "fixture_sha256",
        "profile", "arms", "protocol", "route_package", "memory_after_close")}
    packet.update(
        schema=1, status="retained_exact_prefill", performance_claim=True,
        raw_path=str(args.input), raw_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        hipengine_artifact_provenance=provenance,
        correctness=summary["correctness"], by_shape=summary["by_shape"],
        by_case=summary["by_case"], by_category=summary["by_category"],
        request_wall_speedups=walls,
        total_request_wall_speedup=sum(v["before"] for v in totals.values()) /
                                   sum(v["after"] for v in totals.values()),
        max_prefill_cv=max(c["prefill_tok_s"]["coefficient_of_variation"]
                          for mode in ("before", "after") for c in summary[mode]["cases"].values()),
        max_decode_cv=max(c["decode_tok_s"]["coefficient_of_variation"]
                         for mode in ("before", "after") for c in summary[mode]["cases"].values()),
        measured_first_to_last_span_seconds=(
            raw["samples"][-1]["phase_windows_ns"]["decode"][1]
            - raw["samples"][0]["phase_windows_ns"]["prefill"][0]) / 1e9,
        samples=[{k: r[k] for k in (
            "case_id", "mode", "repetition", "sequence_slot", "prefill_ms", "decode_ms",
            "candidate_calls", "output_token_ids_sha256")} for r in raw["samples"]],
        limits="First-to-last measured span excludes initial warmup/loading. "
               "Decode rate changes are incidental, not a decode-kernel win. "
               "This is not an external parity or native-capacity qualification.")
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
