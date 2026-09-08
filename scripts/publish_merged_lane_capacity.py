"""Consolidate the 2026-09-08 merged-lane layer_outer capacity monitors into
the committed benchmark artifact.

Reads the per-point monitor JSONs written by the host-local point-check
wrapper (scripts/xtx_dms_capacity_point_check.py, byte-identical copy of the
campaign's /tmp wrapper; per-point monitor files embed its sha256), folds in
the pre/post-merge decode-step timing from the raw probe outputs, and emits
benchmarks/results/2026-09-08-rx7900xtx-dms-int8-merged-lane-capacity.json.

Fails closed: every monitor must carry the expected card identity, source
commit and wrapper hash, and the point statuses must match the campaign
record (three passes then one OOM on the merged lane).
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MON = Path("/tmp/xtx-capacity")
WRAPPER_SHA = "e0eac0bce37084cf41354b9efedb2f0049f07e1dfda0c13cad154f8d1a56d103"
EXPECTED = [
    # (path, tokens, status, source_commit)
    ("archive-pre-merge/dms-int8-target4lo256k-139264.monitor.json",
     139264, "execution_fit", "f097c19140ade7fcf46cb077ddc90351c761dac1"),
    ("dms-int8-target4lo256k-139264.monitor.json",
     139264, "execution_fit", "eb3242a7ab7d9d96e212332bbd4a24b364bd734c"),
    ("dms-int8-target4lo256k-168704.monitor.json",
     168704, "execution_fit", "eb3242a7ab7d9d96e212332bbd4a24b364bd734c"),
    ("dms-int8-target4lo256k-172288.monitor.json",
     172288, "execution_fit", "eb3242a7ab7d9d96e212332bbd4a24b364bd734c"),
    ("dms-int8-target4lo256k-176128.monitor.json",
     176128, "oom", "eb3242a7ab7d9d96e212332bbd4a24b364bd734c"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    points = []
    for rel, tokens, status, commit in EXPECTED:
        path = MON / rel
        m = json.loads(path.read_text())
        assert m["prompt_tokens"] == tokens, (rel, m["prompt_tokens"])
        assert m["status"] == status, (rel, m["status"])
        assert m["source_commit"] == commit, (rel, m["source_commit"])
        assert m["wrapper_sha256"] == WRAPPER_SHA, rel
        assert m["unique_id"] == "cc4d02090dc9c3ff", rel
        assert m["returns_to_baseline"] is True, rel
        w = m["whole_card_vram"]
        points.append({
            "role": ("pre_merge_reference" if "archive" in rel
                     else "merged_lane"),
            "prompt_tokens": tokens,
            "decode_steps": m["decode_steps"],
            "status": status,
            "source_commit": commit,
            "working_tree_clean": m["working_tree_clean"],
            "whole_card_peak_bytes": w["peak_bytes"],
            "whole_card_peak_gib": w["peak_gib"],
            "sampled_headroom_bytes": m["sampled_headroom_bytes"],
            "sampled_headroom_mib": m["sampled_headroom_bytes"] / 2**20,
            "tracked_peak_bytes": m["tracked_peak_bytes"],
            "elapsed_seconds": m["elapsed_seconds"],
            "returns_to_baseline": True,
            "failure_tail": m["failure_tail"],
            "raw_output": m["raw_output"],
            "raw_sha256": m["raw_sha256"],
            "log_sha256": m["log_sha256"],
            "wrapper_command": m["wrapper_command"],
        })

    # Merge-neutrality and decode-timing evidence from the 139,264 pair.
    pre = json.loads((MON / "archive-pre-merge"
                      / "dms-int8-target4lo256k-139264.json").read_text())
    post = json.loads((MON / "dms-int8-target4lo256k-139264.json").read_text())

    def decode_ms(doc):
        rows = doc["cycles"][0]["decode"]
        secs = [r["seconds"] for r in rows]
        return [s * 1000.0 for s in secs], [r["output_token"] for r in rows]

    pre_ms, pre_tok = decode_ms(pre)
    post_ms, post_tok = decode_ms(post)
    assert pre_tok == post_tok, "greedy tokens diverged across the merge"
    pre_peak = points[0]["whole_card_peak_bytes"]
    post_peak = points[1]["whole_card_peak_bytes"]
    snap = post["cycles"][0]["dms_snapshots"][0]["capacity"]

    artifact = {
        "date": "2026-09-08",
        "date_timezone": "UTC",
        "kind": "xtx_dms_int8_merged_lane_capacity_ladder",
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
            "sessions": 1,
            "decode_steps": 8,
            "refill_cycles": 1,
            "model": "Qwen3.8-27B-Q4_K_M.gguf",
        },
        "merge_validation": {
            "merged_upstream": "90b9aa510 (DMS speed campaign: staging-upload "
                "hoist, in-place finalize, wave6 gfx1100 producer, chunked "
                "keep-scan INT8 append, wave-grouped INT8 producer)",
            "merge_commit": "eb3242a7ab7d9d96e212332bbd4a24b364bd734c",
            "pre_merge_reference_commit": "f097c19140ade7fcf46cb077ddc90351c761dac1 "
                "(dirty tree: the layer_outer prefill change, later committed as 22ded5522)",
            "whole_card_peak_delta_bytes_at_139264": post_peak - pre_peak,
            "memory_neutral": abs(post_peak - pre_peak) < 2**20,
            "greedy_tokens_identical_across_merge": True,
            "logits_sha256_differ_across_merge": True,
            "logits_note": "Kernels changed floating-point evaluation order "
                "(wave-grouped GQA producer, in-register dequant); greedy "
                "tokens matched on all eight steps.",
            "decode_step_ms_pre_merge_diagnostic": pre_ms,
            "decode_step_ms_post_merge_diagnostic": post_ms,
            "decode_step_mean_ms_pre_merge": sum(pre_ms) / len(pre_ms),
            "decode_step_mean_ms_post_merge": sum(post_ms) / len(post_ms),
            "decode_timing_scope": "single-session diagnostic timing from the "
                "capacity probe; performance_claim is false and no throughput "
                "claim is made",
        },
        "memory_ledger_at_139264_bytes": {
            "weights_packed_small_arena": 16401463296,
            "compact_store_device_resident": 2501602448,
            "oracle_pair_bf16_one_layer": 571473920,
            "hidden_stream_bf16_two_planes": 2857369600,
            "prefill_token_buffer": 1116160,
            "subtotal": 22333025424,
            "residual_unattributed": 1660667912,
            "residual_note": "bulk prefill scratch, split-K partials, "
                "decision/collector planes; itemization pending",
            "tracked_peak": 23993693336,
            "whole_card_minus_tracked": 423340904,
            "whole_card_minus_tracked_note": "whole-card (24,417,034,240 B) "
                "minus tracked peak at the 139,264 post-merge point; "
                "attribution not established",
        },
        "points": points,
        "boundary": {
            "highest_observed_passing_prompt_tokens": 172288,
            "prior_highest_observed_passing_prompt_tokens": 114688,
            "capacity_increase_percent": round(
                (172288 / 114688 - 1) * 100, 2),
            "capacity_increase_percent_over_campaign_baseline": round(
                (172288 / 73728 - 1) * 100, 2),
            "smallest_observed_oom_prompt_tokens": 176128,
            "intermediate_sizes_unqualified": True,
            "cancelled_probe": "172,800 was started and cancelled by operator "
                "choice before completion; it is unqualified and not evidence "
                "of either outcome",
            "oom_failure_mode": "HIP error 2 during a compact layer payload "
                "allocation (v_slot in _ensure_layer) at 23.969 GiB "
                "whole-card; full rollback, GPU1 returned to baseline",
            "passing_point_sampled_headroom_mib": 48.16,
            "proven_maximum": False,
        },
        "slope_observations_bytes_per_token": {
            "note": "the marginal slope is not constant; the compact store "
                "grows sublinearly because the DMS compression ratio "
                "improves with context length",
            "band_73728_to_139264_whole_card": 45346,
            "band_139264_to_168704_whole_card": 33899,
            "band_139264_to_168704_tracked": 35082,
            "band_168704_to_172288_whole_card": 80191,
            "store_compression_ratio_at_139264": snap["actual_compression_ratio"],
            "store_device_resident_bytes_at_139264": snap["device_resident_bytes"],
        },
        "identified_not_validated": {
            "hidden_plane_aliasing": {
                "finding": "the geometry's liveness policy requires >=4096 "
                    "scratch rows for single-plane hidden aliasing while the "
                    "row-cap policy clamps scratch rows to 1024, so the "
                    "layer_outer route always allocates two full-capacity "
                    "BF16 hidden planes",
                "single_plane_bytes_at_139520_positions": 1428684800,
                "single_plane_gib": 1.3306,
                "scope_caution": "the threshold is geometry-wide, not "
                    "DMS-scoped; any change must be route-scoped to "
                    "layer_outer and verified with GPU-side evidence "
                    "(greedy-token equality, per-layer hidden/logit equality, "
                    "allocation ownership, partial/tail chunks, teardown), "
                    "not CPU-codec equality alone",
                "linear_model_ceiling_band_tokens": [215000, 240000],
                "linear_model_disclaimer": "estimate band from measured "
                    "slopes only; the slope is non-constant and the ledger "
                    "residual is not yet itemized; not a capacity claim",
            },
        },
        "scope": "capacity/finiteness/ownership only; no numerical, "
            "throughput, quality, long-output or serving qualification; DMS "
            "discards history, so fitting a longer prompt does not "
            "establish full-context quality; decode horizons remain eight "
            "steps; performance_claim false",
    }
    out = ROOT / "benchmarks/results/2026-09-08-rx7900xtx-dms-int8-merged-lane-capacity.json"
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
