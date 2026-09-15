"""Summarize identical-input GR replays without claiming model qualification."""

import argparse
import hashlib
import json
from pathlib import Path


def assemble(path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    expected = {f"layers.{layer}.hc_{role}_{leg}" for layer in (0, 23, 47)
                for role in ("attn", "ffn") for leg in ("down", "up")}
    records = packet["records"]
    if (packet["status"] != "completed_diagnostic"
            or not packet["source"]["tracked_clean"]
            or not packet["control_logits_state_exact"]
            or packet["after_close"]["current_allocated_bytes"]
            or len(records) != 12 or {row["weight"] for row in records} != expected):
        raise ValueError("incomplete or mutating GR replay")
    ratios = {}
    for row in records:
        expected_shape = [512, 320, 10240] if row["leg"] == "up" else [512, 10240, 320]
        if row["shape"] != expected_shape or row["sampled_rows"] != [0, 255, 511]:
            raise ValueError("unexpected operand sampling")
        if row["leg"] == "up" and row["parent_epilogue_vs_fused"]["changed"] != 0:
            raise ValueError("cannot exclude epilogue split for this capture")
        ratios[row["weight"]] = {
            "candidate_over_parent_sample_mse":
                row["fp64_vs_candidate"]["mse"] / row["fp64_vs_parent"]["mse"],
            "candidate_over_simulated_reconstruction_sample_mse":
                row["fp64_vs_candidate"]["mse"] / row["fp64_vs_reconstructed_activation"]["mse"],
        }
    return dict(
        schema=1, status="localized_projection_error_correction_pending",
        performance_claim=False, promotion_claim=False,
        raw_sha256=hashlib.sha256(raw).hexdigest(), capture=packet, mse_ratios=ratios,
        findings=[
            "Strict projection plus separate sigmoid/mean exactly matches fused GR-up output in six captured roles.",
            "GR-down sampled IU8 MSE is 7.84-32.07x the strict projection MSE versus FP64.",
            "GR-down sampled IU8 MSE is 24.66-198.17x the simulated reconstruction-only MSE.",
            "GR-up sampled IU8 MSE is 1.15-1.75x the simulated reconstruction-only MSE.",
        ],
        hypotheses=[
            "Compensated block accumulation is a targeted GR-down candidate.",
            "A fourth residual activation plane is a targeted GR-up candidate.",
            "Neither hypothesis is measured as a kernel improvement or qualified by these samples.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("artifact.json").write_text(
        json.dumps(assemble(args.capture), indent=2) + "\n")
