#!/usr/bin/env python3
"""Assemble the Flash-Next same-file engine comparison artifact.

Reads the raw per-engine comparator outputs and the kernel attribution, and
writes the compact artifact. Every number here comes from a raw file whose
path and sha256 are recorded; nothing is retyped.

The profiled rows (``*-bf16``/``*-f16`` with a ``code-p4096`` only shape) are
kernel attribution runs, not rate measurements, and are kept out of the rate
table on purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import comparison  # noqa: E402  (sibling module, resolved by the insert above)

RATE_SOURCES = {
    "pwilkin-40a9f4d01": "pwilkin-40a9f4d01-f16.json",
    "halobox-69946438a-f16": "halobox-69946438a-f16.json",
    # Re-measured unprofiled on 2026-09-16. The earlier capture of this same
    # source put mixed_ja_en 73-86% low and is not used as a rate row.
    "halobox-69946438a-bf16": "halobox-69946438a-bf16-clean.json",
    "halobox-pr63-c4aa30229-bf16": "halobox-pr63-c4aa30229-bf16.json",
    "halobox-5f85164": "halobox-5f85164.json",
    "upstream-6011c34ce-f16": "upstream-6011c34ce-f16.json",
    "upstream-6011c34ce-bf16": "upstream-6011c34ce.json",
}
PROFILE_SOURCES = {
    "pwilkin-40a9f4d01": "prof-pwilkin-40a9f4d01-f16.json",
    "halobox-69946438a-bf16": "prof-halobox-69946438a-bf16.json",
    "upstream-6011c34ce-bf16": "prof-upstream-6011c34ce-bf16.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--hipengine", type=Path, required=True)
    parser.add_argument(
        "--census", type=Path, required=True,
        help=(
            "The 2026-09-14 journey-progress artifact. Its "
            "owner-refresh-code4k case is the role-marked census at 53512b509, "
            "which is the only same-case owner total available for the "
            "arithmetic-restored configuration."
        ),
    )
    parser.add_argument(
        "--hipengine-profile", type=Path, required=True,
        help=(
            "The marker-delimited owner capture for the same case and HEAD as "
            "--hipengine. Its owner totals are the cross-check on the "
            "name-based bucketing, because a kernel shared between owners is "
            "attributed by marker there and by name here."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.raw_root
    raw_hashes: dict[str, str] = {}

    # Every engine is loaded through the same validated path and reduced with
    # the same estimator. The llama.cpp-family rows used to be an equal-weight
    # mean of per-case medians while hipEngine's was total tokens over total
    # prefill time; those are different quantities, so their ratio was not a
    # speedup. comparison.py now rejects a run whose case set, prompt token
    # hashes, repetition count, or timing scope do not line up, and
    # weighted_tok_s gives one estimator to every row.
    measurements = []
    for label, name in RATE_SOURCES.items():
        path = raw / name
        measurements.append(comparison.load_comparator(path, label))
        raw_hashes[str(path)] = _sha256(path)
    hipengine_measurement = comparison.load_hipengine(
        args.hipengine, "hipengine-current"
    )
    measurements.append(hipengine_measurement)

    validation_notes = comparison.validate(measurements)
    rates = comparison.comparison_table(measurements)
    rate_eligible = {row["label"]: row for row in rates if not row["excluded"]}

    profiles = []
    for label, name in PROFILE_SOURCES.items():
        path = raw / name
        payload = json.loads(path.read_text())
        profiles.append({
            "label": label,
            "kv_dtype": payload["kv_dtype"],
            "source": payload.get("source") or {},
            "case_id": payload["cases"][0]["id"],
            "profiled_prompt_ms": [
                row["prompt_ms"] for row in payload["cases"][0]["repetitions"]
            ],
            "measurement_class": payload["measurement_class"],
        })
        raw_hashes[str(path)] = _sha256(path)

    hip = json.loads(args.hipengine.read_text())
    raw_hashes[str(args.hipengine)] = _sha256(args.hipengine)
    hipengine_row = {
        "label": "hipengine-current",
        "kv_dtype": "bf16",
        "harness": "scripts/qwen4exp_canonical_ar_bench.py hipengine",
        "source": hip["source"],
        "protocol": hip["protocol"],
        "prefill_tok_s_by_shape": {
            shape: round(payload["prefill_tok_s_weighted"], 2)
            for shape, payload in sorted(hip["summary"]["shapes"].items())
        },
        "prefill_tok_s_median_by_shape": {
            shape: round(payload["prefill_tok_s"]["median"], 2)
            for shape, payload in sorted(hip["summary"]["shapes"].items())
        },
        "prefill_coefficient_of_variation": {
            shape: round(payload["prefill_tok_s"]["coefficient_of_variation"], 5)
            for shape, payload in sorted(hip["summary"]["shapes"].items())
        },
        "decode_tok_s_weighted_by_shape": {
            shape: round(payload["decode_tok_s_weighted"], 2)
            for shape, payload in sorted(hip["summary"]["shapes"].items())
        },
        "all_cases_deterministic": hip["summary"]["all_cases_deterministic"],
    }

    attribution = json.loads(args.attribution.read_text())
    raw_hashes[str(args.attribution)] = _sha256(args.attribution)

    hip_profile = json.loads(args.hipengine_profile.read_text())
    raw_hashes[str(args.hipengine_profile)] = _sha256(args.hipengine_profile)
    hip_owners = hip_profile["cases"][0]["profile"]
    hipengine_row["owner_ms"] = hip_owners["owner_ms"]
    hipengine_row["owner_total_device_ms"] = hip_owners["total_device_ms"]
    hipengine_row["owner_profiled_window_ms"] = hip_owners["profiled_window_ms"]

    # The bucket table is built from kernel names and the owner table from
    # markers, so a kernel that runs under more than one owner lands in one
    # bucket. Reconcile the two rather than present either as exact.
    hip_buckets = next(
        row["per_prefill_ms"] for row in attribution["rows"]
        if row["label"] == "hipengine-current"
    )
    owner_pairs = (
        ("MoE (gate/up + down)", ("moe_gate_up", "moe_down"), ("moe",)),
        ("hyper-connection / GR", ("gr_read",), ("gr_read",)),
        ("attention", ("attention",), ("qsa",)),
        ("Gated DeltaNet", ("gdn",), ("gdn",)),
        ("dense matmul", ("dense_matmul",), ("linear",)),
    )
    reconciliation = []
    for work, buckets, owners in owner_pairs:
        owner_ms = round(sum(hip_owners["owner_ms"].get(o, 0.0) for o in owners), 1)
        bucket_ms = round(sum(hip_buckets.get(b, 0.0) for b in buckets), 1)
        reconciliation.append({
            "work": work,
            "buckets_compared": list(buckets),
            "owners_compared": list(owners),
            "owner_ms": owner_ms,
            "bucket_ms": bucket_ms,
            "delta_ms": round(bucket_ms - owner_ms, 1),
        })
    reconciliation.append({
        "work": "TOTAL",
        "buckets_compared": ["all buckets"],
        "owners_compared": ["all owners"],
        "owner_ms": round(hip_owners["total_device_ms"], 1),
        "bucket_ms": round(sum(hip_buckets.values()), 1),
        "delta_ms": round(sum(hip_buckets.values()) - hip_owners["total_device_ms"], 1),
    })

    journey = json.loads(args.census.read_text())
    raw_hashes[str(args.census)] = _sha256(args.census)
    census_case = journey["observations"]["owner-refresh-code4k.json"]["cases"][0]
    census_profile = census_case["profile"]
    census_owners = census_profile["owner_ms"]
    census_row = {
        "label": "arithmetic-restored-census",
        "source": journey["observations"]["owner-refresh-code4k.json"]["source"],
        "case_id": census_case["id"],
        "command": census_case["command"],
        "owner_ms": census_owners,
        "total_device_ms": census_profile["total_device_ms"],
        "profiled_window_ms": census_profile["profiled_window_ms"],
        "coverage_fraction": census_profile["coverage_fraction"],
        "derived_window_tok_s": round(
            4096 / (census_profile["profiled_window_ms"] / 1000.0), 2
        ),
        "derived_device_tok_s": round(
            4096 / (census_profile["total_device_ms"] / 1000.0), 2
        ),
        "performance_claim": False,
        "limits": (
            "Profiling capture, not a harness rate: the two derived rates divide "
            "4096 tokens by the marker-delimited window and by device time. It "
            "is usable as a same-case device-time divisor and not as a "
            "throughput result."
        ),
    }
    current_device = next(
        row["per_prefill_device_ms"] for row in attribution["rows"]
        if row["label"] == "hipengine-current"
    )
    pwilkin_device = next(
        row["per_prefill_device_ms"] for row in attribution["rows"]
        if row["label"] == "pwilkin-40a9f4d01"
    )
    gap_decomposition = {
        "basis": "device milliseconds per 4096-token prefill, code-p4096",
        "current_default_ms": current_device,
        "arithmetic_restored_census_ms": round(census_profile["total_device_ms"], 1),
        "pwilkin_ms": pwilkin_device,
        "conservative_arithmetic_factor": round(
            current_device / census_profile["total_device_ms"], 3
        ),
        "implementation_factor": round(
            census_profile["total_device_ms"] / pwilkin_device, 3
        ),
        "total_factor": round(current_device / pwilkin_device, 3),
        "note": (
            "The census restored six of the fifteen recovery selectors, so the "
            "arithmetic factor is a lower bound and the implementation factor "
            "is the matching upper bound."
        ),
    }

    reference = rate_eligible["pwilkin-40a9f4d01"]["estimate"]
    hipengine_rates = rate_eligible["hipengine-current"]["estimate"]
    ratios = {
        label: {
            shape: round(value / hipengine_rates["weighted_tok_s_by_shape"][shape], 3)
            for shape, value in row["estimate"]["weighted_tok_s_by_shape"].items()
            if shape in hipengine_rates["weighted_tok_s_by_shape"]
        }
        for label, row in rate_eligible.items()
    }
    ratios_weighted = {
        label: round(
            row["estimate"]["weighted_tok_s"]
            / hipengine_rates["weighted_tok_s"],
            3,
        )
        for label, row in rate_eligible.items()
    }

    artifact = {
        "schema": 1,
        "kind": "flashnext_same_file_engine_comparison",
        "performance_claim": True,
        "numerics_evaluated": False,
        "question": (
            "On one host and one GGUF, how fast is prefill in hipEngine's "
            "current production default against every llama.cpp-family engine "
            "that can open the same file?"
        ),
        "model": (
            "unsloth Qwen3.8-Flash-Next UD-Q4_K_XL, 4 shards, "
            "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
        ),
        "host": "Framework gfx1151, Radeon 8060S, machine 55ea6c509d0b49eea8de7094a1023668",
        "protocol": {
            "phase": "prefill only, n_predict=1",
            "prompts": "exact token ids from the committed canonical fixture",
            "cases": "12 (4 categories x 512/1024/4096)",
            "llamacpp_server_args": (
                "-ngl 999 -fa on -c 4352 -b 8192 -ub 2048 -t 4, --parallel 1"
            ),
            "warmups_per_case": 1,
            "measured_repetitions_per_case": 3,
            "aggregation": (
                "one estimator for every engine: total prompt tokens over total "
                "prompt time (weighted_tok_s), with per-case medians, "
                "per-category and per-shape weighted rates reported beside it"
            ),
            "validation": validation_notes,
            "hipengine_protocol": hip["protocol"],
        },
        "rates": rates,
        "rates_by_shape": {
            label: row["estimate"]["weighted_tok_s_by_shape"]
            for label, row in rate_eligible.items()
        },
        "rates_by_category": {
            label: row["estimate"]["weighted_tok_s_by_category"]
            for label, row in rate_eligible.items()
        },
        "case_medians_by_engine": {
            label: row["estimate"]["case_medians_tok_s"]
            for label, row in rate_eligible.items()
        },
        "rate_ratios_weighted": ratios_weighted,
        "hipengine_current": hipengine_row,
        "ratio_vs_hipengine": ratios,
        "kernel_profiles": profiles,
        "per_prefill_kernel_attribution": attribution["rows"],
        "attribution_vs_owner_reconciliation": {
            "why": (
                "The bucket table attributes by kernel name and the owner table "
                "by marker span. Kernels shared between owners (the generic "
                "K-quant dense kernel runs under both linear and the MoE down "
                "projection) are split by marker but not by name, so the two "
                "views differ on those rows while their totals agree."
            ),
            "rows": reconciliation,
        },
        "arithmetic_restored_census": census_row,
        "gap_decomposition": gap_decomposition,
        "raw_sha256": raw_hashes,
        "findings": [
            (
                "Upstream llama.cpp supports this model: LLM_ARCH_QWEN4EXP is in "
                "origin/master, so the upstream row is a same-file comparator."
            ),
            (
                "pwilkin's default branch (master, e8e6c7af2, 2026-07-22) has no "
                "qwen4exp support at all. The live Flash-Next work is on branch "
                "strix-halo, head 40a9f4d01 (2026-09-15)."
            ),
            (
                "pwilkin 40a9f4d01 asserts at model load with BF16 KV: "
                "src/models/qwen4exp.cpp:1360 requires k/v F16 on the sparse-QSA "
                "direct-indices path. Its arm therefore runs F16 KV."
            ),
            (
                "KV dtype is worth about 1% at 4K (halobox 661 f16 vs 653 bf16; "
                "upstream 459 f16 vs 453 bf16), so it does not explain any row."
            ),
            (
                "halogen refuses UD-Q4_K_XL by name at startup (its README lists "
                "Q4_K/Q5_K/Q5_1/Q4_1 as refused); its published 1246-1424 tok/s "
                "GGUF figures are on UD-IQ4_XS. It is out of scope for a same-file "
                "comparison."
            ),
            (
                "halo-box 69946438a runs code/general_en/general_ja at 735-757 "
                "tok/s at 4K but mixed_ja_en at 374-429, stable across two runs; "
                "pwilkin runs the same prompt at 1053. A single-category or "
                "single-prompt comparator number is not a rate."
            ),
        ],
    }
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")
    print(json.dumps({
        "weighted_tok_s": {
            label: round(row["estimate"]["weighted_tok_s"], 2)
            for label, row in rate_eligible.items()
        },
        "ratios_weighted_vs_hipengine": ratios_weighted,
        "excluded": [
            {"label": row["label"], "reason": row["exclusion_reason"]}
            for row in rates if row["excluded"]
        ],
    }, indent=1))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
