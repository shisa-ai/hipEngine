#!/usr/bin/env python3
"""Publish selector-unset Laguna production prefill evidence.

The matrix-chunk screen predates Laguna's quality-admitted approximate prefill
arithmetic.  Its cross-policy byte-equality gate must still reject M512 versus
M128, but that rejection does not invalidate a selector-unset M512 timing run.
This postprocessor binds that timing evidence to the complete category quality
gate and the current gfx1151 package defaults without rerunning the GPU workload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from hipengine.kernels.backends import backend_package_capability


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUALITY_ARTIFACT = ROOT / (
    "benchmarks/results/"
    "2026-07-25-gfx1151-laguna-prefill-350-d8-category.json"
)
DEFAULT_MATRIX_ARTIFACT = ROOT / (
    "benchmarks/results/"
    "2026-07-25-gfx1151-laguna-prefill-350-production-default.json"
)
DEFAULT_OUTPUT = ROOT / (
    "benchmarks/results/"
    "2026-07-25-gfx1151-laguna-prefill-350-production.json"
)
EXPECTED_BACKEND = "hip_gfx1151"
EXPECTED_LENGTHS = (512, 1024, 4096)
EXPECTED_MATRIX_ROWS = 512
EXPECTED_ATTENTION_ROWS = 128
EXPECTED_MATRIX_FAILURES = frozenset(
    {
        "matrix_policy_outputs_or_state_not_exact",
        "no_larger_policy_improves_every_length",
    }
)
CAPABILITY_DEFAULTS = {
    "f16_prefill_strategy": ("LAGUNA_F16_PREFILL_STRATEGY", "wmma_comp_swa"),
    "matrix_rows": ("LAGUNA_PREFILL_MATRIX_ROWS", EXPECTED_MATRIX_ROWS),
    "global_prefill_variant": (
        "LAGUNA_GLOBAL_PREFILL_VARIANT",
        "global_context_rows_qrow4_m128_online_spans",
    ),
    "swa_prefill_variant": (
        "LAGUNA_SWA_PREFILL_VARIANT",
        "swa_context_rows_qrow4_m128_online_spans",
    ),
    "selected_gate_up_mode": (
        "LAGUNA_SELECTED_GATE_UP_MODE",
        "mmq128x32_d8_f32_wavecols_direct",
    ),
    "selected_down_mode": (
        "LAGUNA_SELECTED_DOWN_MODE",
        "mmq64x32_d4_f32_wavecols_q4",
    ),
    "dense_q4_prefill_mode": (
        "LAGUNA_DENSE_Q4_PREFILL_MODE",
        "wmma_pack8",
    ),
    "f16_projection_mode": (
        "LAGUNA_F16_PREFILL_MODE",
        "hipblaslt_scaled",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-artifact", type=Path, default=DEFAULT_QUALITY_ARTIFACT)
    parser.add_argument("--matrix-artifact", type=Path, default=DEFAULT_MATRIX_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-pp512-tok-s", type=float, default=350.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_zero(stats: Mapping[str, Any]) -> bool:
    return (
        int(stats.get("active_allocations", -1)) == 0
        and int(stats.get("current_allocated_bytes", -1)) == 0
    )


def _candidate_modes(quality: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        quality.get("protocol", {})
        .get("prefill_lane_configurations", {})
        .get("prefill_350_candidate", {})
    )


def _expected_candidate_modes(defaults: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dense_q4_prefill_mode": defaults["dense_q4_prefill_mode"],
        "f16_prefill_mode": defaults["f16_prefill_strategy"],
        "f16_projection_mode": defaults["f16_projection_mode"],
        # Qrow4 is F32 byte-identical to the admitted online-qrow2 arithmetic
        # on complete tiles and preserves qrow2 for residual tiles.
        "global_prefill_variant": (
            "global_context_rows_qrow2_online_spans"
            if defaults["global_prefill_variant"]
            == "global_context_rows_qrow4_m128_online_spans"
            else defaults["global_prefill_variant"]
        ),
        # Row-vector down staging is BF16 byte-identical to the D4 arithmetic
        # admitted by this historical 320-step quality artifact.
        "selected_down_mode": (
            "mmq64x32_d4_f32"
            if defaults["selected_down_mode"]
            == "mmq64x32_d4_f32_wavecols_q4"
            else defaults["selected_down_mode"]
        ),
        # The current wave-column consumer is BF16 byte-identical to the D8
        # arithmetic admitted by this historical 320-step quality artifact.
        "selected_gate_up_mode": (
            "mmq128x32_d8_f32"
            if defaults["selected_gate_up_mode"]
            in (
                "mmq128x32_d8_f32_rowvec",
                "mmq128x32_d8_f32_wavecols",
                "mmq128x32_d8_f32_wavecols_direct",
            )
            else defaults["selected_gate_up_mode"]
        ),
        # Source qualification skips only the current/cache source that the
        # admitted online-qrow2/qrow4 arithmetic would not consume.
        "swa_prefill_variant": (
            "swa_context_rows_qrow2_online_spans"
            if defaults["swa_prefill_variant"]
            == "swa_context_rows_qrow4_m128_online_spans"
            else defaults["swa_prefill_variant"]
        ),
    }


def summarize_publication(
    quality: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    minimum_pp512_tok_s: float = 350.0,
) -> dict[str, Any]:
    """Build a fail-closed publication summary from existing evidence."""

    failed: list[str] = []

    if quality.get("kind") != "hipengine_laguna_prefill_prefill_350_category":
        failed.append("unexpected_quality_artifact_kind")
    if quality.get("status") != "retained_category_gate":
        failed.append("quality_artifact_not_retained")
    if quality.get("pass") is not True or quality.get("performance_claim") is not True:
        failed.append("quality_artifact_failed")
    if quality.get("repo", {}).get("tracked_clean") is not True:
        failed.append("quality_repo_not_clean")
    if quality.get("promotion", {}).get("pass") is not True:
        failed.append("quality_promotion_failed")
    if quality.get("promotion", {}).get("failed_checks") != []:
        failed.append("quality_promotion_has_failed_checks")

    teacher = quality.get("quality", {}).get("teacher_forced", {})
    categories = teacher.get("categories", {})
    if teacher.get("pass") is not True:
        failed.append("teacher_forced_quality_failed")
    if int(teacher.get("steps", -1)) != 320:
        failed.append("teacher_forced_suite_incomplete")
    if float(teacher.get("max_kl_divergence", float("inf"))) > 0.05:
        failed.append("teacher_forced_kl_exceeded")
    if float(teacher.get("top1_agreement", -1.0)) < 0.90:
        failed.append("teacher_forced_top1_below_threshold")
    if set(categories) != {"code", "general_en", "general_ja", "mixed_ja_en"}:
        failed.append("teacher_forced_categories_incomplete")
    elif any(
        not value.get("finite", False)
        or float(value.get("top1_agreement", -1.0)) < 0.90
        or float(value.get("max_kl_divergence", float("inf"))) > 0.05
        for value in categories.values()
    ):
        failed.append("teacher_forced_category_threshold_failed")

    oracle = quality.get("quality", {}).get("poolside_oracle", {})
    if oracle.get("pass") is not True:
        failed.append("poolside_oracle_failed")
    if quality.get("quality", {}).get("free_running", {}).get(
        "same_mode_repeat_deterministic"
    ) is not True:
        failed.append("quality_free_running_not_deterministic")
    quality_memory = quality.get("memory", {})
    if not _clean_zero(quality_memory.get("tracked_before", {})) or not _clean_zero(
        quality_memory.get("tracked_after", {})
    ):
        failed.append("quality_lifecycle_not_recovered")

    expected_modes = _expected_candidate_modes(defaults)
    if dict(_candidate_modes(quality)) != expected_modes:
        failed.append("quality_candidate_modes_do_not_match_defaults")
    for name, (_, expected) in CAPABILITY_DEFAULTS.items():
        if defaults.get(name) != expected:
            failed.append(f"production_default_mismatch:{name}")

    if matrix.get("kind") != "hipengine_laguna_ar_o3_matrix_chunk_screen":
        failed.append("unexpected_matrix_artifact_kind")
    if matrix.get("repo", {}).get("tracked_clean") is not True:
        failed.append("matrix_repo_not_clean")
    platform = matrix.get("platform", {})
    if (
        platform.get("backend") != EXPECTED_BACKEND
        or platform.get("target_arch") != "gfx1151"
    ):
        failed.append("matrix_platform_mismatch")
    protocol = matrix.get("protocol", {})
    if tuple(int(value) for value in protocol.get("lengths", ())) != EXPECTED_LENGTHS:
        failed.append("matrix_lengths_mismatch")
    if set(int(value) for value in protocol.get("matrix_rows", ())) != {128, 256, 512}:
        failed.append("matrix_policy_set_mismatch")
    if int(protocol.get("attention_rows", -1)) != EXPECTED_ATTENTION_ROWS:
        failed.append("matrix_attention_rows_mismatch")
    repetitions = int(protocol.get("repetitions", 0))
    if repetitions < 3:
        failed.append("matrix_repetitions_below_three")

    actual_matrix_failures = frozenset(
        matrix.get("decision", {}).get("failed_checks", ())
    )
    if (
        matrix.get("status") != "measured_rejected"
        or matrix.get("pass") is not False
        or matrix.get("performance_claim") is not False
        or actual_matrix_failures != EXPECTED_MATRIX_FAILURES
    ):
        failed.append("unexpected_historical_matrix_gate_outcome")

    correctness = matrix.get("correctness", {})
    if correctness.get("same_mode_repeat_deterministic") is not True:
        failed.append("matrix_same_mode_not_deterministic")
    if correctness.get("tracked_returned_to_baseline") is not True:
        failed.append("matrix_lifecycle_not_recovered")
    matrix_memory = matrix.get("memory", {})
    if not _clean_zero(matrix_memory.get("tracked_before", {})) or not _clean_zero(
        matrix_memory.get("tracked_after", {})
    ):
        failed.append("matrix_global_lifecycle_not_recovered")

    selected_rows = [
        row
        for row in matrix.get("rows", ())
        if int(row.get("matrix_rows", -1)) == EXPECTED_MATRIX_ROWS
    ]
    expected_row_count = repetitions * len(EXPECTED_LENGTHS)
    if len(selected_rows) != expected_row_count:
        failed.append("matrix_selected_rows_incomplete")
    if any(
        row.get("session_tracked_returned_to_baseline") is not True
        for row in matrix.get("rows", ())
    ):
        failed.append("matrix_session_lifecycle_not_recovered")

    deterministic_fields = (
        "next_token_id",
        "next_token_logit_hex",
        "logits_sha256",
        "final_hidden_sha256",
        "post_layer_hidden_sha256",
        "kv_sha256",
        "final_position",
    )
    milestones: dict[str, Any] = {}
    for length in EXPECTED_LENGTHS:
        rows = sorted(
            (row for row in selected_rows if int(row.get("length", -1)) == length),
            key=lambda row: int(row.get("repetition", -1)),
        )
        if len(rows) != repetitions:
            failed.append(f"matrix_length_incomplete:{length}")
            continue
        if {int(row.get("repetition", -1)) for row in rows} != set(range(repetitions)):
            failed.append(f"matrix_repetitions_incomplete:{length}")
        if any(
            any(row.get(field) != rows[0].get(field) for field in deterministic_fields)
            for row in rows[1:]
        ):
            failed.append(f"matrix_length_not_deterministic:{length}")
        aggregate = matrix.get("aggregate", {}).get(str(EXPECTED_MATRIX_ROWS), {}).get(
            "lengths", {}
        ).get(str(length), {})
        milestones[str(length)] = {
            "median_seconds": float(aggregate.get("median_seconds", float("inf"))),
            "median_tok_s": float(aggregate.get("median_tok_s", -1.0)),
            "samples_seconds": [
                float(row.get("prefill_seconds", float("inf"))) for row in rows
            ],
            "samples_tok_s": [
                float(row.get("prefill_tok_s", -1.0)) for row in rows
            ],
            "next_token_id": rows[0].get("next_token_id"),
        }

    pp512 = milestones.get("512", {})
    if float(pp512.get("median_tok_s", -1.0)) < minimum_pp512_tok_s:
        failed.append("pp512_median_below_target")
    if any(
        value < minimum_pp512_tok_s for value in pp512.get("samples_tok_s", ())
    ):
        failed.append("pp512_sample_below_target")

    quality_model = quality.get("provenance", {}).get("model_fingerprint", {}).get(
        "value"
    )
    matrix_model = matrix.get("provenance", {}).get("model_fingerprint", {}).get(
        "value"
    )
    if not quality_model or quality_model != matrix_model:
        failed.append("model_fingerprint_mismatch")

    passed = not failed
    return {
        "schema": 1,
        "kind": "hipengine_laguna_prefill_production_publication",
        "status": "retained_production_default" if passed else "rejected",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "selector-unset production Laguna-S 2.1 Q4_K_M prefill on gfx1151"
        ),
        "decision": {
            "pass": passed,
            "failed_checks": failed,
            "minimum_pp512_tok_s": float(minimum_pp512_tok_s),
            "policy": (
                "complete retained category quality plus selector-unset M512 "
                "three-repeat speed, deterministic same-policy state, exact "
                "tracked lifecycle, clean provenance, and package-default binding"
            ),
        },
        "production_defaults": {
            **dict(defaults),
            "attention_rows": EXPECTED_ATTENTION_ROWS,
        },
        "headline": {
            "prompt_tokens": 512,
            "matrix_rows": EXPECTED_MATRIX_ROWS,
            "attention_rows": EXPECTED_ATTENTION_ROWS,
            **pp512,
        },
        "milestones": milestones,
        "quality": {
            "teacher_forced_steps": teacher.get("steps"),
            "teacher_forced_max_kl_divergence": teacher.get("max_kl_divergence"),
            "teacher_forced_top1_agreement": teacher.get("top1_agreement"),
            "teacher_forced_top1_matches": teacher.get("top1_matches"),
            "categories": categories,
            "poolside_oracle": oracle,
            "free_running_same_mode_repeat_deterministic": quality.get(
                "quality", {}
            )
            .get("free_running", {})
            .get("same_mode_repeat_deterministic"),
        },
        "lifecycle": {
            "quality_tracked_before": quality_memory.get("tracked_before"),
            "quality_tracked_after": quality_memory.get("tracked_after"),
            "matrix_tracked_before": matrix_memory.get("tracked_before"),
            "matrix_tracked_after": matrix_memory.get("tracked_after"),
            "matrix_all_sessions_returned_to_baseline": all(
                row.get("session_tracked_returned_to_baseline") is True
                for row in matrix.get("rows", ())
            ),
        },
        "limitations": [
            (
                "The historical matrix-policy gate remains rejected because "
                "quality-admitted approximate arithmetic is not byte-identical "
                "across M128/M256/M512."
            ),
            (
                "This publication does not admit a new matrix chunk policy; it "
                "measures the already-selected M512 package default."
            ),
            (
                "The 512/1024/4096 speed screen uses one deterministic "
                "canonical-prompt-derived token stream."
            ),
        ],
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _current_defaults() -> dict[str, Any]:
    return {
        name: backend_package_capability(EXPECTED_BACKEND, capability)
        for name, (capability, _) in CAPABILITY_DEFAULTS.items()
    }


def main() -> int:
    args = _parse_args()
    quality_path = args.quality_artifact.resolve()
    matrix_path = args.matrix_artifact.resolve()
    quality = json.loads(quality_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    artifact = summarize_publication(
        quality,
        matrix,
        defaults=_current_defaults(),
        minimum_pp512_tok_s=args.minimum_pp512_tok_s,
    )

    quality_revision = str(quality.get("repo", {}).get("revision", ""))
    matrix_revision = str(matrix.get("repo", {}).get("revision", ""))
    current_revision = _git("rev-parse", "HEAD")
    ancestry_failed = []
    for label, revision in (
        ("quality", quality_revision),
        ("matrix", matrix_revision),
    ):
        if not revision or subprocess.run(
            ("git", "merge-base", "--is-ancestor", revision, current_revision),
            cwd=ROOT,
            check=False,
        ).returncode != 0:
            ancestry_failed.append(f"{label}_revision_not_current_ancestor")
    if _git("status", "--porcelain", "--untracked-files=no"):
        ancestry_failed.append("current_repo_not_tracked_clean")
    if ancestry_failed:
        artifact["decision"]["failed_checks"].extend(ancestry_failed)
        artifact["decision"]["pass"] = False
        artifact["pass"] = False
        artifact["performance_claim"] = False
        artifact["status"] = "rejected"

    artifact.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repo": {
                "revision": current_revision,
                "tracked_clean": "current_repo_not_tracked_clean"
                not in ancestry_failed,
                "quality_revision": quality_revision,
                "matrix_revision": matrix_revision,
                "source_revisions_are_ancestors": not any(
                    item.endswith("_revision_not_current_ancestor")
                    for item in ancestry_failed
                ),
            },
            "sources": {
                "quality": {
                    "path": str(quality_path.relative_to(ROOT)),
                    "sha256": _sha256(quality_path),
                },
                "matrix": {
                    "path": str(matrix_path.relative_to(ROOT)),
                    "sha256": _sha256(matrix_path),
                    "historical_failed_checks": sorted(EXPECTED_MATRIX_FAILURES),
                },
            },
            "command": [
                "env",
                "PYTHONPATH=.",
                "python3",
                "scripts/laguna_prefill_production_publication.py",
                "--quality-artifact",
                str(quality_path.relative_to(ROOT)),
                "--matrix-artifact",
                str(matrix_path.relative_to(ROOT)),
                "--output",
                str(args.output),
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
