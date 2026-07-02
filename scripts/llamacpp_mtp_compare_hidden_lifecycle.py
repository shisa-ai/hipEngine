#!/usr/bin/env python3
"""Compare llama.cpp MTP hidden lifecycle rows against hipEngine probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "llamacpp_mtp_hidden_lifecycle_compare.v1"


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit("--hipengine-json must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit("--hipengine-json must be LABEL=PATH")
    return label, Path(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _trace_values(
    record: dict[str, Any],
    *,
    label: str,
    row_index: int,
    depth: int | None = None,
) -> dict[str, Any]:
    matches = []
    for trace in record.get("draft_hidden_state_trace", []):
        if str(trace.get("label")) != label:
            continue
        if int(trace.get("row_index", -999)) != int(row_index):
            continue
        if depth is not None and int(trace.get("depth", -999)) != int(depth):
            continue
        matches.append(trace)
    if not matches:
        detail = f"label={label} row_index={row_index}"
        if depth is not None:
            detail += f" depth={depth}"
        raise SystemExit(f"llama.cpp trace missing {detail}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous llama.cpp trace: label={label} row_index={row_index}")
    values = matches[0].get("values")
    if not isinstance(values, list) or not values:
        raise SystemExit(f"llama.cpp trace {label} row_index={row_index} has no raw values")
    return matches[0]


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


def _self_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return _delta(left["values"], right["values"])


def _hip_row(artifact: dict[str, Any], row_index: int) -> dict[str, Any]:
    rows = ((artifact.get("result") or {}).get("rows") or [])
    for row in rows:
        if int(row.get("row", -1)) == int(row_index):
            return row
    raise SystemExit(f"hipEngine artifact has no result row {row_index}")


def _hip_hidden_values(artifact: dict[str, Any], *, row_index: int) -> list[float]:
    row = _hip_row(artifact, row_index)
    values = row.get("hidden_seed_values")
    if not isinstance(values, list) or not values:
        raise SystemExit(f"hipEngine row {row_index} has no hidden_seed_values")
    return values


def _hip_prefix_values(artifact: dict[str, Any]) -> list[float]:
    values = (
        ((artifact.get("result") or {}).get("prefix_state_fingerprint") or {})
        .get("hidden_seed", {})
        .get("values")
    )
    if not isinstance(values, list) or not values:
        raise SystemExit("hipEngine artifact has no prefix_state_fingerprint.hidden_seed.values")
    return values


def _score_for_token(rows: list[dict[str, Any]], token_id: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("token_id", -1)) == int(token_id):
            return row
    return None


def _candidate_score(source: dict[str, Any], token_id: int) -> dict[str, Any]:
    for key in ("candidate_scores", "top_k"):
        found = _score_for_token(list(source.get(key, [])), token_id)
        if found is not None:
            return found
    raise SystemExit(f"token {token_id} missing from candidate_scores/top_k")


def _token_margin_from_scores(source: dict[str, Any], *, token_a: int, token_b: int) -> dict[str, Any]:
    score_a = _candidate_score(source, token_a)
    score_b = _candidate_score(source, token_b)
    logit_a = float(score_a["logit"])
    logit_b = float(score_b["logit"])
    return {
        "sampled_token": source.get("sampled_token"),
        "logits": {str(token_a): logit_a, str(token_b): logit_b},
        f"{token_a}_minus_{token_b}": float(logit_a - logit_b),
        "ranks": {str(token_a): score_a.get("rank"), str(token_b): score_b.get("rank")},
    }


def _target_sample_row(record: dict[str, Any], row_index: int) -> dict[str, Any]:
    rows = list(record.get("target_sample_trace", []))
    for row in rows:
        if int(row.get("row", -1)) == int(row_index):
            return row
    raise SystemExit(f"llama.cpp target_sample_trace has no row {row_index}")


def _hip_token_margin(artifact: dict[str, Any], *, row_index: int, token_a: int, token_b: int) -> dict[str, Any]:
    return _token_margin_from_scores(_hip_row(artifact, row_index), token_a=token_a, token_b=token_b)


def _hip_comparison(
    label: str,
    artifact: dict[str, Any],
    *,
    llama_seed: dict[str, Any],
    llama_process_row0: dict[str, Any],
    llama_verify_rows: list[dict[str, Any]],
    row_index: int,
    token_a: int,
    token_b: int,
) -> dict[str, Any]:
    prefix_values = _hip_prefix_values(artifact)
    row_comparisons = []
    for idx, llama_row in enumerate(llama_verify_rows):
        row_comparisons.append(
            {
                "row": idx,
                "llamacpp_label": "verify_h",
                "llamacpp_token_id": llama_row.get("token_id"),
                "llamacpp_position": llama_row.get("position"),
                "hipengine_input_token": _hip_row(artifact, idx).get("input_token"),
                "hipengine_position": _hip_row(artifact, idx).get("position"),
                "delta": _delta(llama_row["values"], _hip_hidden_values(artifact, row_index=idx)),
            }
        )
    return {
        "label": label,
        "hipengine_cycle": {
            "cycle": (artifact.get("probe") or {}).get("cycle"),
            "sampled_tokens": (artifact.get("result") or {}).get("sampled_tokens"),
            "accepted_draft_tokens": (artifact.get("result") or {}).get("accepted_draft_tokens"),
            "prefix_position": ((artifact.get("result") or {}).get("prefix_state_fingerprint") or {}).get("position"),
            "prefix_current_prev": ((artifact.get("result") or {}).get("prefix_state_fingerprint") or {}).get("current_prev"),
        },
        "prefix_vs_llama_draft_seed_input": _delta(llama_seed["values"], prefix_values),
        "prefix_vs_llama_process_h_input_row0": _delta(llama_process_row0["values"], prefix_values),
        "verify_h_row_comparisons": row_comparisons,
        "token_margin": _hip_token_margin(artifact, row_index=row_index, token_a=token_a, token_b=token_b),
    }


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    draft_tokens = _parse_int_list(args.draft_tokens)
    candidate_tokens = _parse_int_list(args.candidate_tokens)
    if candidate_tokens is None or len(candidate_tokens) != 2:
        raise SystemExit("--candidate-tokens must contain exactly two token IDs")
    llama_record = _find_llama_record(
        args.llamacpp_jsonl,
        cycle=int(args.cycle),
        task_id=args.task_id,
        draft_tokens=draft_tokens,
    )
    llama_seed = _trace_values(llama_record, label="draft_seed_input", row_index=-1, depth=-1)
    llama_process_row0 = _trace_values(llama_record, label="process_h_input", row_index=0, depth=-2)
    llama_verify_rows = [
        _trace_values(llama_record, label="verify_h", row_index=row, depth=row)
        for row in range(int(args.verify_rows))
    ]
    llama_target_row = _target_sample_row(llama_record, int(args.row))
    hip_inputs = [_parse_named_path(value) for value in args.hipengine_json]
    hip_comparisons = [
        _hip_comparison(
            label,
            _load_json(path),
            llama_seed=llama_seed,
            llama_process_row0=llama_process_row0,
            llama_verify_rows=llama_verify_rows,
            row_index=int(args.row),
            token_a=candidate_tokens[0],
            token_b=candidate_tokens[1],
        )
        for label, path in hip_inputs
    ]
    row_key = f"row{int(args.row)}_verify_h_mean_abs_diff"
    nearest_row = min(
        (
            {
                "label": item["label"],
                "mean_abs_diff": item["verify_h_row_comparisons"][int(args.row)]["delta"]["mean_abs_diff"],
            }
            for item in hip_comparisons
        ),
        key=lambda item: float(item["mean_abs_diff"]),
    )
    nearest_prefix = min(
        (
            {
                "label": item["label"],
                "mean_abs_diff": item["prefix_vs_llama_draft_seed_input"]["mean_abs_diff"],
            }
            for item in hip_comparisons
        ),
        key=lambda item: float(item["mean_abs_diff"]),
    )
    return {
        "schema": SCHEMA,
        "kind": "diagnostic",
        "status": "completed",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "llamacpp_jsonl": str(args.llamacpp_jsonl),
            "hipengine_json": [{"label": label, "path": str(path)} for label, path in hip_inputs],
            "cycle": int(args.cycle),
            "task_id": args.task_id,
            "draft_tokens": draft_tokens,
            "row": int(args.row),
            "verify_rows": int(args.verify_rows),
            "candidate_tokens": candidate_tokens,
        },
        "llamacpp_cycle": {
            "task_id": llama_record.get("task_id"),
            "cycle": llama_record.get("cycle"),
            "draft_token_ids": llama_record.get("draft_token_ids"),
            "accepted_draft_tokens": llama_record.get("accepted_draft_tokens"),
            "output_token_ids": llama_record.get("output_token_ids"),
            "bonus_token_id": llama_record.get("bonus_token_id"),
            "rejected_draft_token_id": llama_record.get("rejected_draft_token_id"),
            "seed_token_id": llama_seed.get("token_id"),
            "seed_position": llama_seed.get("position"),
        },
        "llamacpp_handoff_checks": {
            "draft_seed_input_vs_process_h_input_row0": _self_delta(llama_seed, llama_process_row0),
            "verify_h_row0_vs_process_h_input_row1": _self_delta(
                llama_verify_rows[0],
                _trace_values(llama_record, label="process_h_input", row_index=1, depth=-2),
            ),
            "verify_h_row1_vs_process_h_input_row2": _self_delta(
                llama_verify_rows[1],
                _trace_values(llama_record, label="process_h_input", row_index=2, depth=-2),
            ),
        },
        "llamacpp_token_margin": _token_margin_from_scores(
            llama_target_row,
            token_a=candidate_tokens[0],
            token_b=candidate_tokens[1],
        ),
        "hipengine_comparisons": hip_comparisons,
        "summary": {
            "nearest_prefix_seed": nearest_prefix,
            row_key: nearest_row,
            "decision_reading": (
                "Compare prefix closeness separately from the decisive verifier row; "
                "the row token margin determines accept/reject parity."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument(
        "--hipengine-json",
        action="append",
        required=True,
        help="Named hipEngine artifact as LABEL=PATH. Repeat for each lane.",
    )
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--draft-tokens")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--verify-rows", type=int, default=3)
    parser.add_argument(
        "--candidate-tokens",
        required=True,
        help="Two comma-separated token IDs. The artifact reports token_a_minus_token_b logits.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
