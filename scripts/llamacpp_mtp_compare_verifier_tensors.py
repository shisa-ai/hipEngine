#!/usr/bin/env python3
"""Compact llama.cpp-vs-hipEngine MTP verifier tensor comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "llamacpp_mtp_verifier_tensor_compare.v1"


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_boundary_pairs(value: str | None) -> list[tuple[str, str, str]]:
    if value is None or not value.strip():
        return []
    pairs: list[tuple[str, str, str]] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                "--boundary-pairs entries must be hip_key=llamacpp_label_pattern"
            )
        hip_key, llama_pattern = [piece.strip() for piece in part.split("=", 1)]
        if not hip_key or not llama_pattern:
            raise SystemExit(
                "--boundary-pairs entries must be hip_key=llamacpp_label_pattern"
            )
        pairs.append((hip_key, hip_key, llama_pattern))
    return pairs


def _find_llama_record(
    path: Path,
    *,
    cycle: int,
    task_id: int | None,
    draft_tokens: list[int] | None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("cycle", -1)) != int(cycle):
                continue
            if task_id is not None and int(record.get("task_id", -1)) != int(task_id):
                continue
            if draft_tokens is not None and [int(x) for x in record.get("draft_token_ids", [])] != draft_tokens:
                continue
            matches.append(record)
    if not matches:
        details = f"cycle={cycle}"
        if task_id is not None:
            details += f", task_id={task_id}"
        if draft_tokens is not None:
            details += f", draft_tokens={draft_tokens}"
        raise SystemExit(f"no llama.cpp JSONL record matched {details}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous llama.cpp JSONL selection: {len(matches)} records matched")
    return matches[0]


def _walk_value_records(node: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "values" in node and "label" in node:
            records.append(node)
        for value in node.values():
            records.extend(_walk_value_records(value))
    elif isinstance(node, list):
        for value in node:
            records.extend(_walk_value_records(value))
    return records


def _llama_values_by_label(
    record: dict[str, Any],
    *,
    token_id: int,
    position: int,
) -> dict[str, list[float]]:
    labels: dict[str, list[float]] = {}
    h_nextn_count = 0
    for value_record in _walk_value_records(record):
        if int(value_record.get("token_id", -1)) != int(token_id):
            continue
        if int(value_record.get("position", -1)) != int(position):
            continue
        label = str(value_record["label"])
        if label == "llama_stage_h_nextn":
            h_nextn_count += 1
            label = f"llama_stage_h_nextn_{h_nextn_count}"
        labels[label] = [float(value) for value in value_record["values"]]
    return labels


def _hip_row(artifact: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise SystemExit("hipEngine artifact is missing result object")
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("hipEngine artifact result is missing rows list")
    for row in rows:
        if int(row.get("row", -1)) == int(row_index):
            return row
    raise SystemExit(f"hipEngine artifact has no row {row_index}")


def _hip_boundary_capture(
    artifact: dict[str, Any],
    *,
    layer: int,
    row_index: int,
    source: str,
) -> dict[str, Any]:
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise SystemExit("hipEngine artifact is missing result object")
    if source == "scored":
        source_keys = ["scored_layer_boundary_captures"]
    elif source == "isolated":
        source_keys = ["layer_boundary_captures"]
    elif source == "auto":
        source_keys = ["scored_layer_boundary_captures", "layer_boundary_captures"]
    else:
        raise SystemExit("--boundary-source must be auto, scored, or isolated")
    for source_key in source_keys:
        captures = result.get(source_key, [])
        if not isinstance(captures, list):
            continue
        for capture in captures:
            if int(capture.get("layer", -1)) == int(layer) and int(capture.get("row", -1)) == int(row_index):
                values = capture.get("values")
                if not isinstance(values, dict):
                    raise SystemExit(
                        f"hipEngine {source_key} layer={layer} row={row_index} has no values object"
                    )
                capture = dict(capture)
                capture["_source_key"] = source_key
                return capture
    searched = ", ".join(source_keys)
    raise SystemExit(
        f"hipEngine artifact has no {searched} capture for layer={layer} row={row_index}"
    )


def _as_f32(values: list[float], *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise SystemExit(f"{label} has no values")
    return np.ascontiguousarray(array)


def _sha256_16(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()[:16]


def _delta(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    ref = _as_f32(reference, label="reference")
    cand = _as_f32(candidate, label="candidate")
    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch: reference {ref.shape}, candidate {cand.shape}")
    diff = ref.astype(np.float64) - cand.astype(np.float64)
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)
    ref_norm = float(np.linalg.norm(ref64))
    cand_norm = float(np.linalg.norm(cand64))
    return {
        "size": int(ref.size),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "cosine": float(np.dot(ref64, cand64) / (ref_norm * cand_norm)) if ref_norm and cand_norm else None,
        "llamacpp_sha256_16": _sha256_16(ref),
        "hipengine_sha256_16": _sha256_16(cand),
        "first8_diff": [float(value) for value in diff[:8]],
    }


def _candidate_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["token_id"]): row for row in rows}


def _token_margin(sample_trace: dict[str, Any], hip_row: dict[str, Any], *, token_a: int, token_b: int) -> dict[str, Any]:
    llama_candidates = _candidate_map(list(sample_trace.get("candidate_scores", [])))
    hip_candidates = _candidate_map(list(hip_row.get("candidate_scores", [])))

    def engine_row(candidates: dict[int, dict[str, Any]], sampled: int | None) -> dict[str, Any]:
        if token_a not in candidates or token_b not in candidates:
            raise SystemExit(f"candidate scores must include {token_a} and {token_b}")
        logit_a = float(candidates[token_a]["logit"])
        logit_b = float(candidates[token_b]["logit"])
        return {
            "sampled_token": sampled,
            "logits": {str(token_a): logit_a, str(token_b): logit_b},
            f"{token_a}_minus_{token_b}": float(logit_a - logit_b),
            "ranks": {str(token_a): candidates[token_a].get("rank"), str(token_b): candidates[token_b].get("rank")},
        }

    return {
        "token_a": int(token_a),
        "token_b": int(token_b),
        "llamacpp": engine_row(llama_candidates, sample_trace.get("sampled_token")),
        "hipengine": engine_row(hip_candidates, hip_row.get("sampled_token")),
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    draft_tokens = _parse_int_list(args.draft_tokens) if args.draft_tokens is not None else None
    candidate_tokens = _parse_int_list(args.candidate_tokens)
    if len(candidate_tokens) != 2:
        raise SystemExit("--candidate-tokens must contain exactly two comma-separated token IDs")
    layers = _parse_int_list(args.layers)
    boundary_layers = _parse_int_list(args.boundary_layers or "")
    boundary_pairs = _parse_boundary_pairs(args.boundary_pairs)
    if not layers and not boundary_layers:
        raise SystemExit("--layers or --boundary-layers must contain at least one layer ID")
    if boundary_layers and not boundary_pairs:
        raise SystemExit("--boundary-pairs is required when --boundary-layers is set")

    llama_record = _find_llama_record(
        args.llamacpp_jsonl,
        cycle=int(args.cycle),
        task_id=args.task_id,
        draft_tokens=draft_tokens,
    )
    hip_artifact = _json_load(args.hipengine_json)
    hip_row = _hip_row(hip_artifact, row_index=int(args.row))
    token_id = int(args.token_id) if args.token_id is not None else int(hip_row["input_token"])
    position = int(args.position) if args.position is not None else int(hip_row["position"])
    llama_values = _llama_values_by_label(llama_record, token_id=token_id, position=position)

    layer_values = hip_row.get("layer_output_hidden_values")
    if not isinstance(layer_values, dict):
        raise SystemExit("hipEngine row is missing layer_output_hidden_values")
    comparisons: list[dict[str, Any]] = []
    for layer_id in layers:
        llama_label = f"verify_layer_output_{layer_id}"
        hip_key = str(layer_id)
        if llama_label not in llama_values:
            raise SystemExit(f"llama.cpp trace missing {llama_label}")
        if hip_key not in layer_values:
            raise SystemExit(f"hipEngine trace missing layer output {hip_key}")
        comparisons.append(
            {
                "name": llama_label,
                "llamacpp_label": llama_label,
                "hipengine_label": f"target_verify_layer_output_{layer_id}",
                "row": int(args.row),
                "token_id": token_id,
                "position": position,
                "delta": _delta(llama_values[llama_label], layer_values[hip_key]),
            }
        )

    if "verify_pre_output_norm" in llama_values and "pre_output_norm_hidden_values" in hip_row:
        comparisons.append(
            {
                "name": "verify_pre_output_norm",
                "llamacpp_label": "verify_pre_output_norm",
                "hipengine_label": "target_verify_pre_output_norm_hidden",
                "row": int(args.row),
                "token_id": token_id,
                "position": position,
                "delta": _delta(llama_values["verify_pre_output_norm"], hip_row["pre_output_norm_hidden_values"]),
            }
        )
    if "verify_h" in llama_values and "hidden_seed_values" in hip_row:
        comparisons.append(
            {
                "name": "verify_h_vs_target_verify_hidden_seed",
                "llamacpp_label": "verify_h",
                "hipengine_label": "target_verify_hidden_seed",
                "row": int(args.row),
                "token_id": token_id,
                "position": position,
                "delta": _delta(llama_values["verify_h"], hip_row["hidden_seed_values"]),
            }
        )

    boundary_comparisons: list[dict[str, Any]] = []
    for layer_id in boundary_layers:
        hip_capture = _hip_boundary_capture(
            hip_artifact,
            layer=layer_id,
            row_index=int(args.row),
            source=args.boundary_source,
        )
        hip_values = hip_capture["values"]
        for name, hip_key, llama_pattern in boundary_pairs:
            llama_label = llama_pattern.format(layer=layer_id)
            if llama_label not in llama_values:
                raise SystemExit(f"llama.cpp trace missing {llama_label}")
            if hip_key not in hip_values:
                raise SystemExit(
                    f"hipEngine boundary capture layer={layer_id} row={args.row} missing {hip_key}"
                )
            boundary_comparisons.append(
                {
                    "name": name,
                    "llamacpp_label": llama_label,
                    "hipengine_label": hip_key,
                    "layer": int(layer_id),
                    "row": int(args.row),
                    "token_id": token_id,
                    "position": position,
                    "hipengine_source": hip_capture["_source_key"],
                    "delta": _delta(llama_values[llama_label], hip_values[hip_key]),
                }
            )

    layer_comparisons = [row for row in comparisons if row["name"].startswith("verify_layer_output_")]
    first_mae_ge_1e3 = next(
        (
            {"name": row["name"], "mean_abs_diff": row["delta"]["mean_abs_diff"]}
            for row in layer_comparisons
            if float(row["delta"]["mean_abs_diff"]) >= 1.0e-3
        ),
        None,
    )
    target_sample_trace = list(llama_record.get("target_sample_trace", []))
    if int(args.row) >= len(target_sample_trace):
        raise SystemExit(f"llama.cpp target_sample_trace has no row {args.row}")

    return {
        "schema": SCHEMA,
        "kind": "diagnostic",
        "status": "completed",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "llamacpp_jsonl": str(args.llamacpp_jsonl),
            "hipengine_json": str(args.hipengine_json),
            "cycle": int(args.cycle),
            "task_id": args.task_id,
            "draft_tokens": draft_tokens,
            "row": int(args.row),
            "token_id": token_id,
            "position": position,
            "layers": layers,
            "boundary_layers": boundary_layers,
            "boundary_source": args.boundary_source,
            "boundary_pairs": [
                {"name": name, "hipengine": hip_key, "llamacpp": llama_pattern}
                for name, hip_key, llama_pattern in boundary_pairs
            ],
        },
        "llamacpp_cycle": {
            "task_id": llama_record.get("task_id"),
            "cycle": llama_record.get("cycle"),
            "draft_token_ids": llama_record.get("draft_token_ids"),
            "accepted_draft_tokens": llama_record.get("accepted_draft_tokens"),
            "output_token_ids": llama_record.get("output_token_ids"),
            "bonus_token_id": llama_record.get("bonus_token_id"),
        },
        "hipengine_cycle": {
            "cycle": (hip_artifact.get("probe") or {}).get("cycle"),
            "trace_draft_tokens": (hip_artifact.get("probe") or {}).get("trace_draft_tokens"),
            "sampled_tokens": (hip_artifact.get("result") or {}).get("sampled_tokens"),
            "accepted_draft_tokens": (hip_artifact.get("result") or {}).get("accepted_draft_tokens"),
        },
        "token_margin": _token_margin(
            target_sample_trace[int(args.row)],
            hip_row,
            token_a=candidate_tokens[0],
            token_b=candidate_tokens[1],
        ),
        "available_llamacpp_value_labels": sorted(llama_values),
        "comparisons": comparisons,
        "boundary_comparisons": boundary_comparisons,
        "summary": {
            "first_layer_mean_abs_diff_ge_1e-3": first_mae_ge_1e3,
            "largest_layer_mean_abs_diff": max(
                (
                    {"name": row["name"], "mean_abs_diff": row["delta"]["mean_abs_diff"]}
                    for row in layer_comparisons
                ),
                key=lambda item: float(item["mean_abs_diff"]),
                default=None,
            ),
            "pre_output_norm_mean_abs_diff": next(
                (
                    row["delta"]["mean_abs_diff"]
                    for row in comparisons
                    if row["name"] == "verify_pre_output_norm"
                ),
                None,
            ),
            "verify_h_mean_abs_diff": next(
                (
                    row["delta"]["mean_abs_diff"]
                    for row in comparisons
                    if row["name"] == "verify_h_vs_target_verify_hidden_seed"
                ),
                None,
            ),
            "largest_boundary_mean_abs_diff": max(
                (
                    {
                        "layer": row["layer"],
                        "name": row["name"],
                        "mean_abs_diff": row["delta"]["mean_abs_diff"],
                    }
                    for row in boundary_comparisons
                ),
                key=lambda item: float(item["mean_abs_diff"]),
                default=None,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--hipengine-json", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--draft-tokens")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--token-id", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--layers", default="", help="Comma-separated layer-output IDs to compare.")
    parser.add_argument(
        "--boundary-layers",
        default="",
        help="Comma-separated layer IDs whose boundary tensors should be compared.",
    )
    parser.add_argument(
        "--boundary-source",
        choices=("auto", "scored", "isolated"),
        default="auto",
        help="Which hipEngine boundary capture list to use for --boundary-layers.",
    )
    parser.add_argument(
        "--boundary-pairs",
        default="",
        help=(
            "Comma-separated hip_key=llamacpp_label_pattern mappings. "
            "The pattern may include {layer}."
        ),
    )
    parser.add_argument(
        "--candidate-tokens",
        required=True,
        help="Two comma-separated token IDs. The artifact reports token_a_minus_token_b logits.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
