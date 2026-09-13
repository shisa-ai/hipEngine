"""Derive the DPP retention packet from complete-model and owner samples."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qwen4exp_route_pair import _mean_ci95
from scripts.qwen4exp_halo_box_campaign_ab import summarize_campaign_ab


def build(model_path, owner_path):
    model = json.loads(model_path.read_text())
    owner = json.loads(owner_path.read_text())
    if model["status"] != "completed" or owner["status"] != "completed":
        raise ValueError("incomplete experiment")
    if model["method"] != "gdn_dpp" or len(model["samples"]) != 72:
        raise ValueError("wrong model experiment")
    if model["host"]["machine_id"] != owner["host"]["machine_id"]:
        raise ValueError("host mismatch")
    samples = model["samples"]
    by_case = {}
    all_ratios = []
    for case in sorted({r["case_id"] for r in samples}):
        rows = [r for r in samples if r["case_id"] == case]
        if sorted(r["sequence_slot"] for r in rows) != list(range(6)):
            raise ValueError("bad case coverage")
        for key in ("output_token_ids_sha256", "final_logits_sha256", "final_state_sha256"):
            if len({r[key] for r in rows}) != 1:
                raise ValueError("output/state mismatch")
        for row in rows:
            expected = (60 if row["prompt_tokens"] == 4096 else 15) if row["mode"] == "after" else 0
            if row["dpp_calls"] != expected:
                raise ValueError("wrong DPP engagement")
        ratios = []
        for slot in (0, 2, 4):
            pair = {r["mode"]: r for r in rows if r["sequence_slot"] in (slot, slot + 1)}
            ratios.append(pair["before"]["prefill_ms"] / pair["after"]["prefill_ms"])
        all_ratios.extend(ratios)
        by_case[case] = {"ratios": ratios, "mean": statistics.mean(ratios), "mean_95ci": _mean_ci95(ratios)}
    if len(by_case) != 12:
        raise ValueError("incomplete suite")
    if any(c["bit_exact_outputs_and_state"] is not True or c["parent_over_dpp"] <= 1 for c in owner["cases"]):
        raise ValueError("owner gate failed")
    if model["after_close"]["active_allocations"] or model["after_close"]["current_allocated_bytes"]:
        raise ValueError("teardown failed")
    summary = summarize_campaign_ab(samples, repetitions_per_mode=3)
    return {
        "schema": 1, "kind": "journey_gdn_dpp_retained",
        "status": "retained", "performance_claim": True,
        "source": model["source"], "host": model["host"],
        "model_identity": model["model_identity"], "protocol": model["protocol"],
        "command": model["command"], "candidate_library_sha256": model["candidate_library_sha256"],
        "measured_manifest_sha256": model["manifest_sha256"],
        "scope": "existing admitted GDN tiled16 suffix only; no early-layer widening",
        "raw": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_path, owner_path)},
        "paired_prefill": {"mean": statistics.mean(all_ratios), "mean_95ci": _mean_ci95(all_ratios), "by_case": by_case},
        "by_shape": summary["by_shape"], "by_category": summary["by_category"],
        "correctness": {"exact_generated_ids_final_logits_state": True,
                        "cpu_reference": "rows16 complete recurrence/norm/gate and carried state; rtol2e-4/atol2e-5",
                        "incumbent_numerical_failure": "unchanged; not a new strict-teacher certificate"},
        "after_close": model["after_close"],
        "owner": owner,
        "samples": [{k: r[k] for k in (
            "case_id", "mode", "sequence_slot", "prefill_ms", "decode_ms", "dpp_calls",
            "output_token_ids_sha256", "final_logits_sha256", "final_state_sha256",
        )} for r in samples],
        "limits": [
            "Owner synthetic inputs do not estimate whole-model speedup.",
            "Late decode drift is retained; no decode-speedup claim.",
            "Numerical baseline remains failed independently of this exact change.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("artifact.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.model, args.owner), indent=2) + "\n")


if __name__ == "__main__":
    main()
