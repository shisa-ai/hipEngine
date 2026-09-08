"""Consolidate the 2026-09-08 hidden-alias adoption evidence into the
committed benchmark artifact.

Covers the A/B verification pair at 73,728 tokens (alias off/on, both on
the clean tree at 6c6d1ad4a with the env pinned per arm) and the alias-era
capacity ladder on the adopted default (307e7633f). Fails closed on
identity, commits, wrapper hashes, statuses and the A/B equality checks.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MON = Path("/tmp/xtx-capacity")
POINT_WRAPPER_SHA = "e0eac0bce37084cf41354b9efedb2f0049f07e1dfda0c13cad154f8d1a56d103"
AB_WRAPPER_SHA = "b6a85482bd583ac70ee3172f1ab1a70c49cf40f76c39d84ff433219e46df0be5"
ADOPTION_COMMIT = "307e7633f300453f440e26ef0d6ce99c6b26490e"
DOCS_ONLY_COMMIT = "4f5e594f3636dc95db0d08e20d16eac1bc10c562"  # docs-only descendant
GATE_COMMIT = "6c6d1ad4a5d34fe70485facbda0b96eef974944d"


def load(rel: str) -> dict:
    return json.loads((MON / rel).read_text())


def main() -> int:
    ab_off = load("dms-int8-alias-off-73728.monitor.json")
    ab_on = load("dms-int8-alias-on-73728.monitor.json")
    for m, mode, commit in ((ab_off, "off", GATE_COMMIT), (ab_on, "on", GATE_COMMIT)):
        assert m["alias_mode"] == mode and m["source_commit"] == commit, mode
        assert m["status"] == "execution_fit" and m["returns_to_baseline"], mode
        assert m["wrapper_sha256"] == AB_WRAPPER_SHA, mode
        assert m["unique_id"] == "cc4d02090dc9c3ff", mode
    # The A/B equality bars: byte-identical logits, identical greedy tokens.
    assert ab_on["decode_logits_sha256"] == ab_off["decode_logits_sha256"]
    assert ab_on["decode_output_tokens"] == ab_off["decode_output_tokens"]
    peak_saved = ab_off["whole_card_vram"]["peak_bytes"] - ab_on["whole_card_vram"]["peak_bytes"]

    ladder_spec = [
        (200704, "execution_fit"), (216064, "execution_fit"),
        (220672, "execution_fit"), (224256, "execution_fit"),
        (232448, "execution_fit"), (234496, "timeout"),  # wrapper 2700s bound
    ]
    ladder = []
    for tokens, expected in ladder_spec:
        m = load(f"dms-int8-target4lo256k-{tokens}.monitor.json")
        assert m["source_commit"] in (ADOPTION_COMMIT, DOCS_ONLY_COMMIT), tokens
        docs_only = m["source_commit"] == DOCS_ONLY_COMMIT
        assert m["wrapper_sha256"] == POINT_WRAPPER_SHA, tokens
        assert m["unique_id"] == "cc4d02090dc9c3ff", tokens
        assert m["returns_to_baseline"] is True, tokens
        # The 234,496 point ran on the docs-only descendant with the
        # untracked publisher script present; porcelain sees a dirty tree.
        assert m["working_tree_clean"] is (not docs_only), tokens
        if expected is not None:
            assert m["status"] == expected, (tokens, m["status"])
        if m["status"] == "execution_fit":
            raw = json.loads(Path(m["raw_output"]).read_text())
            snap = raw["cycles"][0]["dms_snapshots"][0]["capacity"]
            store_gib = snap["device_resident_bytes"] / 2**30
            compression = snap["actual_compression_ratio"]
        else:
            # Timeout/failed points have no raw probe output; store stats
            # unknown, peak/headroom still recorded from the sampler.
            store_gib = None
            compression = None
        ladder.append({
            "prompt_tokens": tokens,
            "status": m["status"],
            "source_commit": m["source_commit"],
            "source_note": ("docs-only descendant of the adoption commit; "
                "runner code identical, porcelain-dirty from the untracked "
                "publisher script" if docs_only else None),
            "whole_card_peak_gib": m["whole_card_vram"]["peak_gib"],
            "sampled_headroom_mib": m["sampled_headroom_bytes"] / 2**20,
            "tracked_peak_bytes": m["tracked_peak_bytes"],
            "store_device_resident_gib": store_gib,
            "store_compression_ratio": compression,
            "elapsed_seconds": m["elapsed_seconds"],
            "raw_sha256": m["raw_sha256"],
        })
    passes = [p for p in ladder if p["status"] == "execution_fit"]
    ooms = [p for p in ladder if p["status"] == "oom"]
    assert passes, "no passing ladder point"
    highest_pass = max(p["prompt_tokens"] for p in passes)

    artifact = {
        "date": "2026-09-08",
        "date_timezone": "UTC",
        "kind": "xtx_dms_int8_hidden_alias_adoption_and_ladder",
        "hardware": {
            "backend": "hip_gfx1100",
            "cpu": "AMD Ryzen 9 5950X",
            "gpu": "RX 7900 XTX",
            "host": "epyc",
            "other_workloads": "GPU0 not used; GPU1 idle baseline verified per run",
            "pci": "0000:10:00.0",
            "physical_gpu_index": 1,
            "unique_id": "cc4d02090dc9c3ff",
            "vram_total_bytes": 25753026560,
        },
        "route": {
            "codec": "int8_evaluation",
            "dms_prefill_mode": "layer_outer",
            "hidden_plane_alias": "single plane (adopted default, "
                "HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS=0 rolls back)",
            "sessions": 1,
            "decode_steps": 8,
            "refill_cycles": 1,
            "model": "Qwen3.8-27B-Q4_K_M.gguf",
        },
        "adoption_ab": {
            "gate_commit": GATE_COMMIT,
            "adoption_commit": ADOPTION_COMMIT,
            "prompt_tokens": 73728,
            "control": {
                "peak_gib": ab_off["whole_card_vram"]["peak_gib"],
                "prefill_seconds": ab_off["prefill_seconds"],
                "decode_seconds": ab_off["decode_seconds"],
                "tracked_peak_bytes": ab_off["tracked_peak_bytes"],
            },
            "treatment": {
                "peak_gib": ab_on["whole_card_vram"]["peak_gib"],
                "prefill_seconds": ab_on["prefill_seconds"],
                "decode_seconds": ab_on["decode_seconds"],
                "tracked_peak_bytes": ab_on["tracked_peak_bytes"],
            },
            "whole_card_peak_saved_bytes": peak_saved,
            "logits_byte_identical": True,
            "greedy_tokens_identical": True,
            "returns_to_baseline_both_arms": True,
            "note": "aliasing is an allocation change; arithmetic must be "
                "bit-exact, and was: all eight decode logits_sha256 matched "
                "the two-plane control",
        },
        "ladder": ladder,
        "boundary": {
            "highest_observed_passing_prompt_tokens": highest_pass,
            "prior_highest_observed_passing_prompt_tokens": 172288,
            "capacity_increase_percent_over_prior": round(
                (highest_pass / 172288 - 1) * 100, 2),
            "capacity_increase_percent_over_campaign_baseline": round(
                (highest_pass / 73728 - 1) * 100, 2),
            "smallest_observed_oom_prompt_tokens": (
                min(p["prompt_tokens"] for p in ooms) if ooms else None),
            "no_alias_era_oom_note": "the alias-era ladder recorded no OOM; "
                "the 176,128 OOM was measured on the pre-alias two-plane "
                "route (commit eb3242a7a) and does not bound the alias route; "
                "234,496 hit the wrapper's 2700-second bound with 116.1 MiB "
                "headroom remaining (status timeout, unqualified, clean "
                "teardown) — it may fit; no smaller alias-era OOM is known",
            "intermediate_sizes_unqualified": True,
            "model_context_bound": 262144,
            "model_context_bound_note": "the model file's declared context "
                "bounds prompt+decode at 262,144 positions; the earlier "
                "262,144-token probe failed validation (not OOM) against "
                "this bound",
            "peak_variance_note": "whole-card peaks carry roughly +/-0.2 GiB "
                "content/allocator variance at these scales (the 220,672 "
                "point peaked 216 MiB above the later 224,256 point); the "
                "store itself is monotone (~16.8 KiB/token, kept fraction "
                "~0.5185)",
            "proven_maximum": False,
        },
        "scope": "capacity/finiteness/ownership only; no numerical, "
            "throughput, quality, long-output or serving qualification; DMS "
            "discards history, so fitting a longer prompt does not "
            "establish full-context quality; decode horizons remain eight "
            "steps; performance_claim false",
    }
    out = ROOT / "benchmarks/results/2026-09-08-rx7900xtx-dms-int8-hidden-alias-ladder.json"
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"highest pass: {highest_pass}, smallest OOM: "
          f"{artifact['boundary']['smallest_observed_oom_prompt_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
