#!/usr/bin/env python3
"""Find the earliest layer where llama.cpp and hipEngine hidden_in diverge."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_capture_path_audit import (  # noqa: E402
    audit_precision_contractions_for_layers,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    pack_float32,
    sha256_bytes,
    top_abs_diff_entries,
    unpack_float32,
)
from scripts.llamacpp_mtp_sweep_hidden_in_layers import parse_layers  # noqa: E402

DEFAULT_LAYER_SWEEP = Path(
    "benchmarks/results/mtp-gguf-iter295-llamacpp-hidden-in-layer-sweep.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter298-hidden-in-earliest-divergence.json"
)
DEFAULT_LAYERS = "0-3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer-sweep", type=Path, default=DEFAULT_LAYER_SWEEP)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--compiler-version")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--exact-atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=298)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    artifact = build_earliest_divergence_artifact(
        model_path=args.model,
        layer_sweep_path=args.layer_sweep,
        layers=parse_layers(args.layers),
        prompt_tokens=parse_prompt_tokens(args.prompt_tokens),
        position=args.position,
        compiler_version=args.compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=args.max_sequence_length,
        exact_atol=args.exact_atol,
        iteration=args.iteration,
        dry_run=bool(args.dry_run),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "first_mismatch_layer": artifact["ranking"].get("first_mismatch_layer"),
                "last_exact_prefix_layer": artifact["ranking"].get("last_exact_prefix_layer"),
                "conclusion": artifact["conclusion"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_earliest_divergence_artifact(
    *,
    model_path: Path,
    layer_sweep_path: Path,
    layers: Sequence[int],
    prompt_tokens: tuple[int, ...],
    position: int,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
    max_sequence_length: int | None = None,
    exact_atol: float = 0.0,
    iteration: int = 298,
    dry_run: bool = False,
    hipengine_captures: Mapping[int, Sequence[float]] | None = None,
    precision_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layer_ids = list(layers)
    layer_sweep = read_json(layer_sweep_path)
    precision = precision_audit or audit_precision_contractions_for_layers(
        model_path,
        layers=tuple(layer_ids),
    )
    if dry_run:
        status = "dry_run"
        layer_results: list[dict[str, Any]] = []
    elif hipengine_captures is None and not hip_available():
        status = "skipped_no_hip_runtime"
        layer_results = []
    else:
        hip_captures = dict(hipengine_captures or capture_hipengine_hidden_ins(
            model_path=model_path,
            prompt_tokens=prompt_tokens,
            position=position,
            layers=layer_ids,
            compiler_version=compiler_version,
            require_cached_build=bool(require_cached_build),
            max_sequence_length=max_sequence_length,
        ))
        layer_results = compare_layers(
            layer_sweep=layer_sweep,
            hipengine_captures=hip_captures,
            layers=layer_ids,
            precision_records=precision.get("records") or [],
            exact_atol=exact_atol,
        )
        status = status_from_layer_results(layer_results)
    ranking = rank_layer_results(layer_results)
    conclusion = conclude(ranking=ranking, status=status)
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_hidden_in_earliest_divergence",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "model": str(model_path),
        "layer_sweep_path": str(layer_sweep_path),
        "layers": layer_ids,
        "position": int(position),
        "token_id": int(prompt_tokens[position]),
        "prompt_tokens": list(prompt_tokens),
        "compiler_version": compiler_version,
        "require_cached_build": bool(require_cached_build),
        "max_sequence_length": max_sequence_length,
        "exact_atol": float(exact_atol),
        "precision_audit": precision,
        "layer_results": layer_results,
        "ranking": ranking,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion, ranking=ranking),
    }


def capture_hipengine_hidden_ins(
    *,
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layers: Sequence[int],
    compiler_version: str | None,
    require_cached_build: bool,
    max_sequence_length: int | None,
) -> dict[int, list[float]]:
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    captures: dict[int, list[float]] = {}
    with Qwen35GGUFResidentSession(
        model_path,
        compiler_version=compiler_version,
        require_cached_build=bool(require_cached_build),
        max_sequence_length=max_seq,
    ) as session:
        for layer in layers:
            session.reset()
            for pos, token_id in enumerate(prompt_tokens[:position]):
                session._run_token_to_final_hidden(int(token_id), position=pos)  # noqa: SLF001
            capture = session.capture_attention_layer(
                int(prompt_tokens[position]),
                position=position,
                layer_id=int(layer),
                run_preceding_layers=True,
            )
            captures[int(layer)] = [float(x) for x in capture.hidden_in_f32.reshape(-1)]
    return captures


def compare_layers(
    *,
    layer_sweep: dict[str, Any],
    hipengine_captures: Mapping[int, Sequence[float]],
    layers: Sequence[int],
    precision_records: Sequence[dict[str, Any]],
    exact_atol: float,
) -> list[dict[str, Any]]:
    results = []
    for layer in layers:
        llama = load_llamacpp_layer_values(layer_sweep, layer=int(layer))
        hip = [float(x) for x in hipengine_captures[int(layer)]]
        comparison = compare_vectors(llama["values"], hip, exact_atol=exact_atol)
        results.append(
            {
                "layer": int(layer),
                "status": "matched" if comparison["exact_match"] else "mismatched",
                "llamacpp": llama["summary"],
                "hipengine": {
                    "count": len(hip),
                    "sha256": sha256_bytes(pack_float32(hip)),
                    "samples": [round(value, 8) for value in hip[:8]],
                },
                "numeric_delta": comparison,
                "preceding_precision_contractions": precision_records_before_layer(
                    precision_records,
                    target_layer=int(layer),
                ),
                "current_layer_precision_contractions": precision_records_for_layer(
                    precision_records,
                    layer=int(layer),
                ),
            }
        )
    return results


def load_llamacpp_layer_values(layer_sweep: dict[str, Any], *, layer: int) -> dict[str, Any]:
    layer_rows = [row for row in layer_sweep.get("layer_results", []) if row.get("layer") == layer]
    if not layer_rows:
        raise KeyError(f"llama.cpp layer {layer} missing from layer sweep")
    binary_path = Path(layer_rows[0]["binary_path"])
    data = binary_path.read_bytes()
    values = unpack_float32(data)
    return {
        "values": values,
        "summary": {
            "binary_path": str(binary_path),
            "sha256": sha256_bytes(data),
            "count": len(values),
            "source_status": layer_rows[0].get("status"),
            "source_selected_row": layer_rows[0].get("selected_row"),
        },
    }


def compare_vectors(
    actual: Sequence[float], reference: Sequence[float], *, exact_atol: float
) -> dict[str, Any]:
    if len(actual) != len(reference):
        return {
            "shape_match": False,
            "actual_count": len(actual),
            "reference_count": len(reference),
            "exact_match": False,
        }
    diffs = [float(a) - float(b) for a, b in zip(actual, reference)]
    abs_diffs = [abs(value) for value in diffs]
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    mean_abs = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
    rmse = math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else 0.0
    return {
        "shape_match": True,
        "count": len(actual),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rmse": rmse,
        "actual_l2": math.sqrt(sum(float(value) * float(value) for value in actual)),
        "reference_l2": math.sqrt(sum(float(value) * float(value) for value in reference)),
        "diff_samples": [round(value, 8) for value in diffs[:8]],
        "top_abs_diff": top_abs_diff_entries(list(actual), list(reference), limit=8),
        "exact_match": max_abs <= exact_atol,
    }


def rank_layer_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first_mismatch = next((row for row in results if row["status"] != "matched"), None)
    exact_prefix = []
    for row in results:
        if row["status"] != "matched":
            break
        exact_prefix.append(row["layer"])
    comparable = [row for row in results if row["numeric_delta"].get("shape_match")]
    by_rmse = sorted(
        comparable,
        key=lambda row: row["numeric_delta"].get("rmse", float("inf")),
    )
    return {
        "first_mismatch_layer": first_mismatch.get("layer") if first_mismatch else None,
        "last_exact_prefix_layer": exact_prefix[-1] if exact_prefix else None,
        "exact_prefix_layers": exact_prefix,
        "by_rmse": [compact_layer_row(row) for row in by_rmse],
        "mismatched_layers": [
            compact_layer_row(row) for row in results if row["status"] != "matched"
        ],
    }


def compact_layer_row(row: dict[str, Any]) -> dict[str, Any]:
    delta = row["numeric_delta"]
    return {
        "layer": row["layer"],
        "status": row["status"],
        "rmse": delta.get("rmse"),
        "max_abs_diff": delta.get("max_abs_diff"),
        "mean_abs_diff": delta.get("mean_abs_diff"),
        "preceding_precision_contraction_count": row[
            "preceding_precision_contractions"
        ]["count"],
    }


def conclude(*, ranking: dict[str, Any], status: str) -> str:
    if status in {"dry_run", "skipped_no_hip_runtime", "capture_failed"}:
        return status
    first = ranking.get("first_mismatch_layer")
    if first is None:
        return "hidden_in_matches_through_requested_layers"
    if first == 0:
        return "layer0_hidden_in_mismatch_embedding_or_capture"
    return f"first_hidden_in_divergence_after_layer_{int(first) - 1}"


def next_action(conclusion: str, *, ranking: dict[str, Any]) -> str:
    if conclusion == "hidden_in_matches_through_requested_layers":
        return "continue_checkpoint_compare_after_last_requested_layer"
    if conclusion == "layer0_hidden_in_mismatch_embedding_or_capture":
        return "compare_token_embedding_weight_materialization_and_embedding_kernel"
    if conclusion.startswith("first_hidden_in_divergence_after_layer_"):
        layer = ranking.get("first_mismatch_layer")
        previous = int(layer) - 1 if layer is not None else None
        return f"compare_layer_{previous}_internal_taps_and_precision_contractors"
    if conclusion == "skipped_no_hip_runtime":
        return "rerun_on_rocm_host"
    if conclusion == "dry_run":
        return "run_without_dry_run_on_rocm_host"
    return "inspect_capture_failure"


def status_from_layer_results(results: Sequence[dict[str, Any]]) -> str:
    if not results:
        return "capture_failed"
    if any(not row["numeric_delta"].get("shape_match") for row in results):
        return "shape_mismatch"
    if all(row["status"] == "matched" for row in results):
        return "matched"
    return "mismatched"


def precision_records_before_layer(
    records: Sequence[dict[str, Any]], *, target_layer: int
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if 0 <= layer_from_slot(record.get("slot_path")) < target_layer
    ]
    return precision_record_summary(selected)


def precision_records_for_layer(records: Sequence[dict[str, Any]], *, layer: int) -> dict[str, Any]:
    selected = [record for record in records if layer_from_slot(record.get("slot_path")) == layer]
    return precision_record_summary(selected)


def precision_record_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "slot_paths": [str(record.get("slot_path")) for record in records],
    }


def layer_from_slot(slot_path: object) -> int:
    if not isinstance(slot_path, str):
        return -1
    parts = slot_path.split(".")
    if len(parts) < 2 or parts[0] != "layers":
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def parse_prompt_tokens(text: str) -> tuple[int, ...]:
    tokens = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not tokens:
        raise ValueError("prompt token list is empty")
    return tokens


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
