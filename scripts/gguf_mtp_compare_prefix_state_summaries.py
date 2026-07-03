#!/usr/bin/env python3
"""Compare compact hipEngine MTP prefix-state numeric summaries."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gguf_mtp_prefix_state_summary_compare.v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prefix_state(artifact: dict[str, Any], *, label: str) -> dict[str, Any]:
    state = ((artifact.get("result") or {}).get("prefix_state_fingerprint") or {})
    if not state:
        raise SystemExit(f"{label} artifact has no result.prefix_state_fingerprint")
    return state


def _summary(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    summary = payload.get("numeric_summary")
    if not isinstance(summary, dict):
        raise SystemExit(f"{label} has no numeric_summary; rerun probe with --prefix-state-numeric-summary")
    return summary


def _float(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if value is None:
        raise SystemExit(f"numeric_summary is missing {key}")
    return float(value)


def _mae(left: list[Any], right: list[Any]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right)) / len(left))


def _sample_delta(base_summary: dict[str, Any], cand_summary: dict[str, Any], key: str) -> float | None:
    return _mae(list(base_summary.get(key) or []), list(cand_summary.get(key) or []))


def _compare_numeric_payload(
    component: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    layer: int | None = None,
    part: str | None = None,
) -> dict[str, Any]:
    base_summary = _summary(baseline, label=f"baseline {component}")
    cand_summary = _summary(candidate, label=f"candidate {component}")
    size = int(base_summary.get("size", 0))
    cand_size = int(cand_summary.get("size", 0))
    if size != cand_size:
        raise SystemExit(f"{component} size mismatch: {size} vs {cand_size}")
    first8_mae = _sample_delta(base_summary, cand_summary, "first8")
    last8_mae = _sample_delta(base_summary, cand_summary, "last8")
    scalar_deltas = {
        "mean": abs(_float(base_summary, "mean") - _float(cand_summary, "mean")),
        "rms": abs(_float(base_summary, "rms") - _float(cand_summary, "rms")),
        "min": abs(_float(base_summary, "min") - _float(cand_summary, "min")),
        "max": abs(_float(base_summary, "max") - _float(cand_summary, "max")),
    }
    summary_score = max(
        [
            scalar_deltas["mean"],
            scalar_deltas["rms"],
            first8_mae or 0.0,
            last8_mae or 0.0,
        ]
    )
    result: dict[str, Any] = {
        "component": component,
        "layer": layer,
        "part": part,
        "nbytes": int(baseline.get("nbytes", 0)),
        "hash_equal": baseline.get("blake2b_128") == candidate.get("blake2b_128"),
        "baseline_blake2b_128": baseline.get("blake2b_128"),
        "candidate_blake2b_128": candidate.get("blake2b_128"),
        "size": size,
        "baseline_summary_sha256_16": base_summary.get("sha256_16"),
        "candidate_summary_sha256_16": cand_summary.get("sha256_16"),
        "summary_scalar_abs_deltas": scalar_deltas,
        "first8_mae": first8_mae,
        "last8_mae": last8_mae,
        "summary_delta_score": float(summary_score),
        "baseline_first8": base_summary.get("first8"),
        "candidate_first8": cand_summary.get("first8"),
    }
    pairwise_delta = _raw_pairwise_delta(baseline, candidate, label=component)
    if pairwise_delta is not None:
        result["pairwise_delta"] = pairwise_delta
    return result


def _f32_from_bf16_raw(raw: bytes) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return np.ascontiguousarray((words << np.uint32(16)).view(np.float32), dtype=np.float32)


def _raw_values(payload: dict[str, Any], *, label: str) -> np.ndarray | None:
    raw_payload = payload.get("raw_data")
    if raw_payload is None:
        return None
    if not isinstance(raw_payload, dict):
        raise SystemExit(f"{label} raw_data is not an object")
    if str(raw_payload.get("encoding")) != "base64":
        raise SystemExit(f"{label} raw_data encoding is not base64")
    raw = base64.b64decode(str(raw_payload.get("data_b64", "")), validate=True)
    expected_nbytes = int(raw_payload.get("nbytes", -1))
    if len(raw) != expected_nbytes:
        raise SystemExit(f"{label} raw_data nbytes mismatch: {len(raw)} vs {expected_nbytes}")
    expected_digest = raw_payload.get("blake2b_128")
    digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
    if expected_digest is not None and str(expected_digest) != digest:
        raise SystemExit(f"{label} raw_data digest mismatch")
    dtype = str(raw_payload.get("dtype"))
    if dtype == "fp32":
        if len(raw) % np.dtype(np.float32).itemsize != 0:
            raise SystemExit(f"{label} raw_data is not FP32-aligned")
        return np.ascontiguousarray(np.frombuffer(raw, dtype="<f4"), dtype=np.float32)
    if dtype == "bf16":
        if len(raw) % np.dtype(np.uint16).itemsize != 0:
            raise SystemExit(f"{label} raw_data is not BF16-aligned")
        return _f32_from_bf16_raw(raw)
    raise SystemExit(f"{label} raw_data has unsupported dtype {dtype!r}")


def _array_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise SystemExit(f"raw shape mismatch: {reference.shape} vs {candidate.shape}")
    ref64 = reference.astype(np.float64)
    cand64 = candidate.astype(np.float64)
    diff = ref64 - cand64
    ref_norm = float(np.linalg.norm(ref64))
    cand_norm = float(np.linalg.norm(cand64))
    return {
        "size": int(reference.size),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "cosine": float(np.dot(ref64, cand64) / (ref_norm * cand_norm)) if ref_norm and cand_norm else None,
        "first8_diff": [float(value) for value in diff[:8]],
    }


def _raw_pairwise_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any] | None:
    base_values = _raw_values(baseline, label=f"baseline {label}")
    cand_values = _raw_values(candidate, label=f"candidate {label}")
    if base_values is None and cand_values is None:
        return None
    if base_values is None or cand_values is None:
        raise SystemExit(f"{label} raw_data is present in only one input")
    return _array_delta(base_values, cand_values)


def _layer_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["layer"]): row for row in rows}


def _compare_linear_layers(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    base_layers = _layer_map(list(baseline.get("linear_state_layers") or []))
    cand_layers = _layer_map(list(candidate.get("linear_state_layers") or []))
    results: list[dict[str, Any]] = []
    for layer_id in sorted(set(base_layers) | set(cand_layers)):
        if layer_id not in base_layers or layer_id not in cand_layers:
            results.append(
                {
                    "component": "linear_state_layer_presence",
                    "layer": int(layer_id),
                    "baseline_present": layer_id in base_layers,
                    "candidate_present": layer_id in cand_layers,
                    "summary_delta_score": None,
                }
            )
            continue
        for part in ("conv", "recurrent"):
            results.append(
                _compare_numeric_payload(
                    f"linear_state_{part}",
                    dict(base_layers[layer_id][part]),
                    dict(cand_layers[layer_id][part]),
                    layer=int(layer_id),
                    part=part,
                )
            )
    return results


def _compare_kv_layers(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    base_layers = _layer_map(list(((baseline.get("kv_state") or {}).get("layers") or [])))
    cand_layers = _layer_map(list(((candidate.get("kv_state") or {}).get("layers") or [])))
    results: list[dict[str, Any]] = []
    for layer_id in sorted(set(base_layers) | set(cand_layers)):
        if layer_id not in base_layers or layer_id not in cand_layers:
            results.append(
                {
                    "component": "full_attn_kv_layer_presence",
                    "layer": int(layer_id),
                    "baseline_present": layer_id in base_layers,
                    "candidate_present": layer_id in cand_layers,
                    "summary_delta_score": None,
                }
            )
            continue
        for part in ("key", "value"):
            results.append(
                _compare_numeric_payload(
                    f"full_attn_kv_{part}",
                    dict(base_layers[layer_id][part]),
                    dict(cand_layers[layer_id][part]),
                    layer=int(layer_id),
                    part=part,
                )
            )
    return results


def _row(artifact: dict[str, Any], row_index: int) -> dict[str, Any]:
    for row in ((artifact.get("result") or {}).get("rows") or []):
        if int(row.get("row", -1)) == int(row_index):
            return row
    raise SystemExit(f"artifact has no result row {row_index}")


def _score_for_token(row: dict[str, Any], token_id: int) -> dict[str, Any]:
    for key in ("candidate_scores", "top_k"):
        for score in row.get(key, []) or []:
            if int(score.get("token_id", -1)) == int(token_id):
                return dict(score)
    raise SystemExit(f"row {row.get('row')} has no score for token {token_id}")


def _token_margin(
    artifact: dict[str, Any],
    *,
    row_index: int,
    token_a: int,
    token_b: int,
) -> dict[str, Any]:
    row = _row(artifact, row_index)
    score_a = _score_for_token(row, token_a)
    score_b = _score_for_token(row, token_b)
    return {
        "row": int(row_index),
        "sampled_token": row.get("sampled_token"),
        "logits": {
            str(token_a): float(score_a["logit"]),
            str(token_b): float(score_b["logit"]),
        },
        f"{token_a}_minus_{token_b}": float(score_a["logit"]) - float(score_b["logit"]),
        "ranks": {
            str(token_a): score_a.get("rank"),
            str(token_b): score_b.get("rank"),
        },
    }


def _top_deltas(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sortable = [row for row in rows if row.get("summary_delta_score") is not None]
    return sorted(sortable, key=lambda row: float(row["summary_delta_score"]), reverse=True)[:limit]


def _top_pairwise_deltas(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sortable = [row for row in rows if isinstance(row.get("pairwise_delta"), dict)]
    return sorted(
        sortable,
        key=lambda row: float(row["pairwise_delta"]["mean_abs_diff"]),
        reverse=True,
    )[:limit]


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    baseline_artifact = _load_json(args.baseline_json)
    candidate_artifact = _load_json(args.candidate_json)
    baseline_state = _prefix_state(baseline_artifact, label=args.baseline_label)
    candidate_state = _prefix_state(candidate_artifact, label=args.candidate_label)
    candidate_tokens = [int(part.strip()) for part in str(args.candidate_tokens).split(",") if part.strip()]
    if len(candidate_tokens) != 2:
        raise SystemExit("--candidate-tokens must contain exactly two token IDs")
    linear_rows = _compare_linear_layers(baseline_state, candidate_state)
    kv_rows = _compare_kv_layers(baseline_state, candidate_state)
    linear_hash_changed = [row for row in linear_rows if row.get("hash_equal") is False]
    kv_hash_changed = [row for row in kv_rows if row.get("hash_equal") is False]
    return {
        "schema": SCHEMA,
        "kind": "diagnostic",
        "status": "completed",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline": {"label": args.baseline_label, "path": str(args.baseline_json)},
            "candidate": {"label": args.candidate_label, "path": str(args.candidate_json)},
            "row": int(args.row),
            "candidate_tokens": candidate_tokens,
        },
        "contract": {
            "note": (
                "This compares compact per-buffer numeric summaries. When selected raw_data "
                "payloads are present, it also computes full pairwise MAE/RMSE/max/cosine for "
                "those buffers only."
            )
        },
        "prefix_metadata": {
            args.baseline_label: {
                "position": baseline_state.get("position"),
                "current_prev": baseline_state.get("current_prev"),
                "hidden_seed_blake2b_128": (baseline_state.get("hidden_seed") or {}).get("blake2b_128"),
                "hidden_seed_summary": (baseline_state.get("hidden_seed") or {}).get("summary"),
            },
            args.candidate_label: {
                "position": candidate_state.get("position"),
                "current_prev": candidate_state.get("current_prev"),
                "hidden_seed_blake2b_128": (candidate_state.get("hidden_seed") or {}).get("blake2b_128"),
                "hidden_seed_summary": (candidate_state.get("hidden_seed") or {}).get("summary"),
            },
        },
        "token_margin": {
            args.baseline_label: _token_margin(
                baseline_artifact,
                row_index=int(args.row),
                token_a=candidate_tokens[0],
                token_b=candidate_tokens[1],
            ),
            args.candidate_label: _token_margin(
                candidate_artifact,
                row_index=int(args.row),
                token_a=candidate_tokens[0],
                token_b=candidate_tokens[1],
            ),
        },
        "summary": {
            "linear_components": len(linear_rows),
            "linear_hash_changed": len(linear_hash_changed),
            "kv_components": len(kv_rows),
            "kv_hash_changed": len(kv_hash_changed),
            "top_linear_summary_deltas": _top_deltas(linear_rows, int(args.top_n)),
            "top_kv_summary_deltas": _top_deltas(kv_rows, int(args.top_n)),
            "raw_linear_components": len([row for row in linear_rows if isinstance(row.get("pairwise_delta"), dict)]),
            "raw_kv_components": len([row for row in kv_rows if isinstance(row.get("pairwise_delta"), dict)]),
            "top_linear_pairwise_deltas": _top_pairwise_deltas(linear_rows, int(args.top_n)),
            "top_kv_pairwise_deltas": _top_pairwise_deltas(kv_rows, int(args.top_n)),
        },
        "linear_state_summary_deltas": linear_rows,
        "kv_state_summary_deltas": kv_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--candidate-tokens", required=True, help="TOKEN_A,TOKEN_B for row margin comparison")
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
