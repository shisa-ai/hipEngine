#!/usr/bin/env python3
"""Sweep llama.cpp hidden-in capture layer IDs and rank numeric closeness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_EXPECTED_SHA256,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    DEFAULT_REFERENCE_ARRAYS,
    DEFAULT_REFERENCE_KEY,
    run_hidden_in_capture,
)

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter294-llamacpp-hidden-in-capture-harness-allrows-compile.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter295-llamacpp-hidden-in-layer-sweep.json")
DEFAULT_OUTPUT_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter295-hidden-in-layer-sweep")
DEFAULT_LAYERS = "0-6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--expected-sha256", default=DEFAULT_EXPECTED_SHA256)
    parser.add_argument("--reference-arrays", type=Path, default=DEFAULT_REFERENCE_ARRAYS)
    parser.add_argument("--reference-key", default=DEFAULT_REFERENCE_KEY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--iteration", type=int, default=295)
    args = parser.parse_args()

    artifact = build_layer_sweep_artifact(
        compile_artifact_path=args.compile_artifact,
        model_path=args.model,
        prompt_tokens=args.prompt_tokens,
        layers=parse_layers(args.layers),
        position=args.position,
        expected_sha256=args.expected_sha256,
        reference_arrays_path=args.reference_arrays,
        reference_key=args.reference_key,
        output_dir=args.output_dir,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "layers": artifact["layers"],
                "best_selected": artifact["ranking"].get("best_selected"),
                "best_any_row": artifact["ranking"].get("best_any_row"),
                "conclusion": artifact["conclusion"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_layer_sweep_artifact(
    *,
    compile_artifact_path: Path,
    model_path: Path,
    prompt_tokens: str,
    layers: Iterable[int],
    position: int,
    expected_sha256: str,
    reference_arrays_path: Path,
    reference_key: str,
    output_dir: Path,
    n_gpu_layers: int = 999,
    threads: int = 8,
    timeout_seconds: int = 2400,
    env: Mapping[str, str] | None = None,
    iteration: int = 295,
) -> dict[str, Any]:
    layer_ids = list(layers)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_results = []
    for layer in layer_ids:
        prefix = output_dir / f"layer{layer}-pos{position}"
        capture = run_hidden_in_capture(
            compile_artifact_path=compile_artifact_path,
            model_path=model_path,
            prompt_tokens=prompt_tokens,
            layer=layer,
            position=position,
            expected_sha256=expected_sha256,
            output_prefix=prefix,
            reference_arrays_path=reference_arrays_path,
            reference_key=reference_key,
            n_gpu_layers=n_gpu_layers,
            threads=threads,
            all_rows=True,
            timeout_seconds=timeout_seconds,
            env=env,
            iteration=iteration,
        )
        layer_results.append(summarize_layer_capture(capture))
    ranking = rank_layers(layer_results)
    conclusion = conclude(ranking=ranking, target_layer=3)
    return {
        "schema": 1,
        "kind": "llamacpp_hidden_in_layer_sweep",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_results(layer_results, ranking),
        "compile_artifact_path": str(compile_artifact_path),
        "model": str(model_path),
        "layers": layer_ids,
        "position": int(position),
        "expected_sha256": expected_sha256,
        "reference_arrays_path": str(reference_arrays_path),
        "reference_key": reference_key,
        "output_dir": str(output_dir),
        "layer_results": layer_results,
        "ranking": ranking,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion),
    }


def summarize_layer_capture(capture: dict[str, Any]) -> dict[str, Any]:
    delta = capture.get("numeric_delta") or {}
    scan = delta.get("all_rows_scan") or {}
    best_any = scan.get("best_by_rmse")
    result = {
        "layer": capture["layer"],
        "status": capture["status"],
        "returncode": capture["run"].get("returncode"),
        "elapsed_seconds": capture["run"].get("elapsed_seconds"),
        "capture_sha256": capture.get("capture", {}).get("sha256"),
        "matches_expected": capture.get("comparison", {}).get("matches_expected"),
        "selected_row": capture["position"],
        "selected_max_abs_diff": delta.get("max_abs_diff"),
        "selected_mean_abs_diff": delta.get("mean_abs_diff"),
        "selected_rmse": delta.get("rmse"),
        "all_rows_available": scan.get("available") is True,
        "all_rows_count": scan.get("rows"),
        "best_any_row": best_any,
        "exact_rows": scan.get("matches", []),
        "binary_path": capture.get("capture", {}).get("binary_path"),
        "all_rows_path": capture.get("capture", {}).get("all_rows", {}).get("binary_path"),
    }
    if capture["status"] != "mismatched":
        result["stdout_tail"] = capture["run"].get("stdout_tail", "")[-1000:]
        result["stderr_tail"] = capture["run"].get("stderr_tail", "")[-1000:]
    return result


def rank_layers(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_candidates = [r for r in results if r.get("selected_rmse") is not None]
    any_row_candidates = [
        r for r in results if (r.get("best_any_row") or {}).get("rmse") is not None
    ]
    exact_matches = [r for r in results if r.get("matches_expected")]
    exact_any_rows = [
        {"layer": r["layer"], **row}
        for r in results
        for row in r.get("exact_rows", [])
    ]
    selected_sorted = sorted(selected_candidates, key=lambda r: r["selected_rmse"])
    any_sorted = sorted(
        any_row_candidates,
        key=lambda r: r["best_any_row"]["rmse"],
    )
    return {
        "best_selected": selected_summary(selected_sorted[0]) if selected_sorted else None,
        "best_any_row": any_row_summary(any_sorted[0]) if any_sorted else None,
        "selected_by_rmse": [selected_summary(r) for r in selected_sorted],
        "any_row_by_rmse": [any_row_summary(r) for r in any_sorted],
        "exact_selected_matches": exact_matches,
        "exact_any_row_matches": exact_any_rows,
    }


def selected_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": row["layer"],
        "row": row["selected_row"],
        "rmse": row["selected_rmse"],
        "max_abs_diff": row["selected_max_abs_diff"],
        "mean_abs_diff": row["selected_mean_abs_diff"],
        "sha256": row["capture_sha256"],
    }


def any_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    best = row["best_any_row"]
    return {
        "layer": row["layer"],
        "row": best.get("row"),
        "rmse": best.get("rmse"),
        "max_abs_diff": best.get("max_abs_diff"),
        "mean_abs_diff": best.get("mean_abs_diff"),
        "sha256": best.get("sha256"),
    }


def conclude(*, ranking: dict[str, Any], target_layer: int) -> str:
    if ranking["exact_selected_matches"] or ranking["exact_any_row_matches"]:
        return "found_exact_layer_match"
    best_any = ranking.get("best_any_row")
    if best_any and best_any["layer"] != target_layer:
        return "different_layer_is_closest"
    best_selected = ranking.get("best_selected")
    if best_selected and best_selected["layer"] != target_layer:
        return "different_selected_layer_is_closest"
    return "target_layer_best_but_mismatched"


def status_from_results(results: list[dict[str, Any]], ranking: dict[str, Any]) -> str:
    if any(result["returncode"] != 0 for result in results):
        return "capture_failed"
    if ranking["exact_selected_matches"] or ranking["exact_any_row_matches"]:
        return "matched"
    return "mismatched"


def next_action(conclusion: str) -> str:
    if conclusion == "found_exact_layer_match":
        return "switch_llamacpp_oracle_layer_to_matching_layer_and_compare_next_tensor"
    if conclusion in {"different_layer_is_closest", "different_selected_layer_is_closest"}:
        return "inspect_layer_numbering_between_hipengine_and_llamacpp"
    return "inspect_tap_placement_or_graph_path_for_layer3_hidden_in"


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(item))
    return layers


if __name__ == "__main__":
    main()
