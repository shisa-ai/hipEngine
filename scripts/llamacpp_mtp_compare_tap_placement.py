#!/usr/bin/env python3
"""Compare a llama.cpp hidden-in row against hipEngine checkpoint arrays."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    pack_float32,
    sha256_bytes,
    top_abs_diff_entries,
    unpack_float32,
)

DEFAULT_CAPTURE = Path("benchmarks/results/mtp-gguf-iter294-llamacpp-hidden-in-allrows-scan.json")
DEFAULT_HIPENGINE_ARRAYS = Path(
    "benchmarks/results/mtp-gguf-iter280-layer3-full-attn-actual-routing-full-arrays.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter296-llamacpp-tap-placement-compare.json")
TARGET_KEY = "hidden_in_f32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--hipengine-arrays", type=Path, default=DEFAULT_HIPENGINE_ARRAYS)
    parser.add_argument("--target-key", default=TARGET_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=296)
    args = parser.parse_args()

    artifact = build_tap_placement_artifact(
        capture_path=args.capture,
        hipengine_arrays_path=args.hipengine_arrays,
        target_key=args.target_key,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "best_key": artifact["ranking"]["best_same_width"]["key"],
                "target_rank": artifact["ranking"]["target_rank"],
                "conclusion": artifact["conclusion"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_tap_placement_artifact(
    *,
    capture_path: Path,
    hipengine_arrays_path: Path,
    target_key: str = TARGET_KEY,
    iteration: int = 296,
) -> dict[str, Any]:
    capture_doc = json.loads(capture_path.read_text())
    hip_doc = json.loads(hipengine_arrays_path.read_text())
    llama = read_capture_row(capture_doc)
    comparisons, skipped = compare_against_arrays(
        llama["values"],
        arrays=hip_doc.get("arrays") or {},
        target_key=target_key,
    )
    ranking = rank_comparisons(comparisons, target_key=target_key)
    conclusion = conclude(ranking, target_key=target_key)
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_tap_placement_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "matched" if ranking["exact_matches"] else "mismatched",
        "capture_path": str(capture_path),
        "hipengine_arrays_path": str(hipengine_arrays_path),
        "target_key": target_key,
        "llamacpp_capture": llama["summary"],
        "hipengine_context": summarize_hipengine_context(hip_doc),
        "comparisons": comparisons,
        "skipped_arrays": skipped,
        "ranking": ranking,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion),
    }


def read_capture_row(capture_doc: dict[str, Any]) -> dict[str, Any]:
    capture = capture_doc.get("capture") or {}
    binary_path = Path(capture["binary_path"])
    values = unpack_float32(binary_path.read_bytes())
    return {
        "values": values,
        "summary": {
            "binary_path": str(binary_path),
            "sha256": sha256_bytes(binary_path.read_bytes()),
            "count": len(values),
            "layer": capture_doc.get("layer"),
            "position": capture_doc.get("position"),
            "status": capture_doc.get("status"),
            "source_artifact_sha256": capture.get("sha256"),
        },
    }


def compare_against_arrays(
    llama_values: list[float], *, arrays: dict[str, Any], target_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for key, raw in arrays.items():
        if not isinstance(raw, list) or not all(is_number(value) for value in raw):
            skipped.append({"key": key, "reason": "non_numeric_or_not_list"})
            continue
        if len(raw) != len(llama_values):
            skipped.append(
                {
                    "key": key,
                    "reason": "width_mismatch",
                    "count": len(raw),
                    "expected_count": len(llama_values),
                }
            )
            continue
        reference = [float(value) for value in raw]
        comparisons.append(compare_one(key, llama_values, reference, is_target=key == target_key))
    return comparisons, skipped


def compare_one(
    key: str, llama_values: list[float], reference: list[float], *, is_target: bool
) -> dict[str, Any]:
    diffs = [actual - expected for actual, expected in zip(llama_values, reference)]
    abs_diffs = [abs(value) for value in diffs]
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    return {
        "key": key,
        "is_target": is_target,
        "count": len(reference),
        "reference_sha256": sha256_bytes(pack_float32(reference)),
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else 0.0,
        "actual_l2": math.sqrt(sum(value * value for value in llama_values)),
        "reference_l2": math.sqrt(sum(value * value for value in reference)),
        "samples": [round(value, 8) for value in reference[:8]],
        "diff_samples": [round(value, 8) for value in diffs[:8]],
        "top_abs_diff": top_abs_diff_entries(llama_values, reference, limit=8),
        "exact_match": max_abs == 0.0,
    }


def rank_comparisons(comparisons: list[dict[str, Any]], *, target_key: str) -> dict[str, Any]:
    by_rmse = sorted(comparisons, key=lambda item: item["rmse"])
    exact_matches = [item for item in comparisons if item["exact_match"]]
    target_index = next(
        (index for index, item in enumerate(by_rmse, start=1) if item["key"] == target_key),
        None,
    )
    return {
        "best_same_width": compact_rank_row(by_rmse[0]) if by_rmse else None,
        "target_rank": target_index,
        "by_rmse": [compact_rank_row(item) for item in by_rmse],
        "exact_matches": [compact_rank_row(item) for item in exact_matches],
    }


def compact_rank_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": item["key"],
        "is_target": item["is_target"],
        "rmse": item["rmse"],
        "max_abs_diff": item["max_abs_diff"],
        "mean_abs_diff": item["mean_abs_diff"],
        "reference_sha256": item["reference_sha256"],
    }


def conclude(ranking: dict[str, Any], *, target_key: str) -> str:
    if ranking["exact_matches"]:
        return "found_exact_tap_match"
    best = ranking.get("best_same_width")
    if best is None:
        return "no_same_width_arrays"
    if best["key"] == target_key:
        return "target_hidden_in_is_closest_but_mismatched"
    return "non_hidden_array_is_closest"


def next_action(conclusion: str) -> str:
    if conclusion == "found_exact_tap_match":
        return "switch_to_matching_tap_key_and_continue_layer_checkpoint_compare"
    if conclusion == "non_hidden_array_is_closest":
        return "inspect_llamacpp_tap_placement_against_closest_hipengine_array"
    if conclusion == "target_hidden_in_is_closest_but_mismatched":
        return "inspect_graph_path_or_materialization_difference_for_hidden_in"
    return "add_more_same_width_reference_arrays_or_capture_taps"


def summarize_hipengine_context(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": doc.get("kind"),
        "iteration": doc.get("iteration"),
        "model": doc.get("model"),
        "layer_id": doc.get("layer_id"),
        "position": doc.get("position"),
        "token_id": doc.get("token_id"),
        "run_preceding_layers": doc.get("run_preceding_layers"),
        "array_keys": doc.get("array_keys"),
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    main()
