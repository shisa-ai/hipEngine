"""Assemble Q8 localization evidence; retain qualification blockers."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_conservative_cost import summarize_cost


def compact_diagnostic(packet):
    result = {key: value for key, value in packet.items() if key != "arms"}
    result["arms"] = {}
    for name, arm in packet["arms"].items():
        quality = arm["quality"]
        compact = {key: quality[key] for key in (
            "summary", "scope_failures", "hard_gates_passed",
            "rows_over_review_boundary", "top1_mismatch_rows",
        )}
        compact["scopes"] = {
            dimension: {
                value: {key: summary[key] for key in (
                    "rows", "kl_mean", "kl_p95", "kl_p99", "kl_max",
                    "top1_agreement", "passed",
                )}
                for value, summary in values.items()
            }
            for dimension, values in quality["by_scope"].items()
        }
        if "all_rows" in quality:
            compact["all_rows"] = [
                {key: row[key] for key in (
                    "request_id", "category", "shape", "teacher_step",
                    "kl_mean", "top1_agreement", "strict_margin_min",
                    "max_abs_logit_delta", "topk_overlap_mean",
                )}
                for row in quality["all_rows"]
            ]
        result["arms"][name] = dict(overrides=arm["overrides"], quality=compact)
    return result


def assemble(root):
    hashes = {}

    def load(name):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    diagnostics = {
        name: load(name + ".json") for name in (
            "prefill-decode-isolation", "prefill-family-isolation",
            "prefill-dense-isolation", "q8-shape-isolation-cleared",
            "q8-boundary-replay",
        )
    }
    for packet in diagnostics.values():
        if packet["status"] != "completed" or packet["after_close"]["current_allocated_bytes"]:
            raise ValueError("incomplete/leaking diagnostic")
    original = load("q8-layer-shape-isolation.json")
    original_counts = load("q8-layer-shape-isolation.counts.json")
    counts = load("q8-shape-isolation-cleared.counts.json")
    for k, n in ((2560, 10240), (2560, 12288), (6144, 2560), (10240, 320),
                 (2560, 2560), (2560, 640), (2560, 512)):
        shape = f"{k}x{n}"
        if counts.get(f"mmq:{shape}:{shape}:disabled=True", 0) <= 0:
            raise ValueError("missing intended shape intervention")
    baseline = diagnostics["prefill-decode-isolation"]["arms"]["bound"]["quality"]["summary"]
    for name in ("q8-shape-isolation-cleared", "q8-boundary-replay"):
        if diagnostics[name]["arms"]["bound"]["quality"]["summary"] != baseline:
            raise ValueError("instrumented control changed")
    operands = load("q8-boundary-replay.operands.json")
    if len(operands) != 90:
        raise ValueError("expected 18 prompts times five Q8-down calls")
    samples = [s for r in operands for s in r["samples"]]
    reference = np.array([s["fp64_raw"] for s in samples])
    error = {}
    for key in ("fp64_fp16_weights", "fp64_fp16_activations",
                "fp64_fp16_both", "kernel_bf16"):
        delta = np.array([s[key] for s in samples]) - reference
        error[key] = dict(mse=float(np.mean(delta ** 2)),
                          max_abs=float(np.max(np.abs(delta))))
    def bf16(values):
        bits = np.asarray(values, dtype=np.float32).view(np.uint32)
        return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    raw_bits = bf16(reference)
    half_bits = bf16([s["fp64_fp16_both"] for s in samples])
    kernel_bits = bf16([s["kernel_bf16"] for s in samples])
    full = load("production-q8-fallback-profile.json")
    if not full["measurement_valid"] or full["quality"]["quality"]["summary"]["rows"] != 594:
        raise ValueError("valid full numerical packet required")
    tasks = load("q8-repair-complete-tasks.json")
    review = json.loads(Path(__file__).with_name("task-review.json").read_text())
    if review["capture_complete"]:
        if tasks["status"] != "captured_requires_manual_review":
            raise ValueError("task capture incomplete")
    elif review["status"] != "rejected_for_default_promotion":
        raise ValueError("incomplete capture needs explicit rejection record")
    for arm in tasks["arms"].values():
        for case in arm["cases"]:
            repeats = case.pop("repeats")
            case.update(
                text_sha256=hashlib.sha256(repeats[0]["text"].encode()).hexdigest(),
                token_count=len(repeats[0]["ids"]),
                finish_reason=repeats[0]["finish_reason"],
                repeated_ids_sha256=[
                    hashlib.sha256(json.dumps(r["ids"]).encode()).hexdigest()
                    for r in repeats],
            )
    recovery = load("production-conservative-profile.json")
    depth = load("conservative-depth.json")
    default_smoke = load("recovery-default-smoke.json")
    if (default_smoke["status"] != "completed"
            or default_smoke["arms"]["bound"]["quality"]["summary"]["kl_max"] != 0
            or default_smoke["after_close"]["current_allocated_bytes"] != 0):
        raise ValueError("fresh default-bound smoke must reproduce strict and close")
    if not recovery["measurement_valid"] or not recovery["decision"]["passed"]:
        raise ValueError("conservative short gate must pass")
    if (depth["status"] != "completed" or not depth["quality"]["hard_gates_passed"]
            or not depth["deterministic"] or not depth["state_gate"]["passed"]
            or any(row["current_allocated_bytes"] for row in depth["lifecycle"].values())):
        raise ValueError("conservative depth gate must pass")
    cost = load("conservative-cost.json")
    if (len(cost["samples"]) != 72 or cost["after_close"]["current_allocated_bytes"]
            or not all(row["finite"] for row in cost["samples"])):
        raise ValueError("complete finite cost matrix and clean teardown required")
    summary = summarize_cost(cost["samples"], 3)
    if len(summary["by_case"]) != 12:
        raise ValueError("all canonical cases required")
    cost["reassembled_summary"] = summary
    cost["summary_repair"] = (
        "Allow cross-arm ID differences between old failed arithmetic and recovery; "
        "within-arm determinism remains mandatory. Raw samples are unchanged."
    )
    for sample in cost["samples"]:
        sample.pop("output_token_ids", None)
    return dict(
        schema=1, kind="q8_prefill_numerical_localization",
        performance_claim=True, default_changed=True,
        claim_scope="Measured cost of conservative arithmetic recovery; not a speedup",
        raw_sha256=hashes,
        diagnostics={name: compact_diagnostic(packet)
                     for name, packet in diagnostics.items()},
        invalid_shape_sequence=dict(
            source=original["source"], command=original["command"],
            reason="custom shape intervention absent from host dispatch cache key",
            used_for_shape_attribution=False, counts=original_counts),
        layer_interventions={k: v for k, v in compact_diagnostic(original)["arms"].items()
                             if k.startswith("down_off_")},
        cleared_shape_counts=counts,
        operand_replay=dict(
            records=len(operands),
            activation_elements=sum(r["activation_elements"] for r in operands),
            fp16_changed=sum(r["activation_fp16_changed"] for r in operands),
            fp16_zeroed=sum(r["activation_fp16_zeroed"] for r in operands),
            fp16_nonfinite=sum(r["activation_fp16_nonfinite"] for r in operands),
            max_activation=max(r["activation_max_abs"] for r in operands),
            sampled_output_errors=error,
            sampled_outputs=int(raw_bits.size),
            fp16_operand_bf16_changes=int(np.count_nonzero(raw_bits != half_bits)),
            kernel_vs_fp16_ideal_changes=int(np.count_nonzero(kernel_bits != half_bits)),
            kernel_vs_raw_fp64_changes=int(np.count_nonzero(kernel_bits != raw_bits)),
            limitations="fixed geometry sample, not an all-output proof"),
        numerical_gate=full,
        tasks=tasks,
        task_review=review,
        conservative_recovery=recovery,
        canonical_depth=depth,
        default_binding_smoke=compact_diagnostic(default_smoke),
        binding_source_sha256=hashlib.sha256(
            (ROOT / "hipengine/generation/qwen4_exp_profiles.py").read_bytes()).hexdigest(),
        cost=cost,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.raw_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
