#!/usr/bin/env python3
"""Compare hipEngine live target-score rows against llama.cpp MTP traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def _hip_cycles(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    cycles = artifact.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("hip artifact does not contain a cycles list")
    return [cycle for cycle in cycles if isinstance(cycle, dict)]


def _find_hip_cycle(artifact: dict[str, Any], *, cycle: int) -> dict[str, Any]:
    for row in _hip_cycles(artifact):
        if int(row.get("cycle", -1)) == int(cycle):
            return row
    raise ValueError(f"hip artifact does not contain cycle {cycle}")


def _find_hip_score_row(artifact: dict[str, Any], *, cycle: int, row: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cycle_row = _find_hip_cycle(artifact, cycle=cycle)
    rows = cycle_row.get("target_lm_head_score_rows")
    if not isinstance(rows, list):
        raise ValueError(f"hip cycle {cycle} does not contain target_lm_head_score_rows")
    for score_row in rows:
        if isinstance(score_row, dict) and int(score_row.get("row", -1)) == int(row):
            return cycle_row, score_row
    raise ValueError(f"hip cycle {cycle} does not contain target score row {row}")


def _find_hip_hidden_row(cycle_row: dict[str, Any], *, row: int) -> dict[str, Any] | None:
    rows = cycle_row.get("target_hidden_seed_rows")
    if not isinstance(rows, list):
        return None
    for hidden_row in rows:
        if isinstance(hidden_row, dict) and int(hidden_row.get("row", -1)) == int(row):
            return hidden_row
    return None


def _find_llama_cycle(
    rows: list[dict[str, Any]],
    *,
    cycle: int,
    task_id: int | None,
    draft_tokens: list[int] | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("cycle", -1)) != int(cycle):
            continue
        if task_id is not None and int(row.get("task_id", -1)) != int(task_id):
            continue
        if draft_tokens is not None:
            got = [int(token) for token in row.get("draft_token_ids", [])]
            if got != draft_tokens:
                continue
        candidates.append(row)
    if len(candidates) != 1:
        hint = f"cycle={cycle} task_id={task_id} draft_tokens={draft_tokens}"
        raise ValueError(f"expected one llama cycle for {hint}, found {len(candidates)}")
    return candidates[0]


def _find_llama_score_row(
    rows: list[dict[str, Any]],
    *,
    cycle: int,
    row: int,
    task_id: int | None,
    draft_tokens: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cycle_row = _find_llama_cycle(rows, cycle=cycle, task_id=task_id, draft_tokens=draft_tokens)
    score_rows = cycle_row.get("target_sample_trace")
    if not isinstance(score_rows, list):
        raise ValueError("llama cycle does not contain target_sample_trace")
    for score_row in score_rows:
        if isinstance(score_row, dict) and int(score_row.get("row", -1)) == int(row):
            return cycle_row, score_row
    raise ValueError(f"llama cycle {cycle} does not contain target sample row {row}")


def _normal_score_map(score_row: dict[str, Any], *, source: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for entry in score_row.get("top_k", []) or []:
        if not isinstance(entry, dict):
            continue
        token = int(entry.get("token", entry.get("token_id")))
        if source == "llama":
            score = float(entry["logit"])
            delta = -float(entry.get("margin_from_top", 0.0))
        else:
            score = float(entry["score"])
            delta = float(entry.get("delta_vs_top1", 0.0))
        result[token] = {
            "token": token,
            "rank": int(entry.get("rank", 0)),
            "score": score,
            "delta_vs_top1": delta,
            "source": "top_k",
        }
    for entry in score_row.get("candidate_scores", []) or []:
        if not isinstance(entry, dict):
            continue
        token = int(entry.get("token", entry.get("token_id")))
        if entry.get("score") is None and entry.get("logit") is None:
            continue
        if source == "llama":
            score = float(entry["logit"])
            delta = -float(entry.get("margin_from_top", 0.0))
        else:
            score = float(entry["score"])
            delta = float(entry.get("delta_vs_top1", 0.0))
        result.setdefault(
            token,
            {
                "token": token,
                "rank": entry.get("rank"),
                "score": score,
                "delta_vs_top1": delta,
                "source": "candidate_scores",
            },
        )
    return result


def _score(map_: dict[int, dict[str, Any]], token: int) -> float | None:
    row = map_.get(int(token))
    if row is None:
        return None
    return float(row["score"])


def _margin(map_: dict[int, dict[str, Any]], a: int, b: int) -> float | None:
    score_a = _score(map_, a)
    score_b = _score(map_, b)
    if score_a is None or score_b is None:
        return None
    return float(score_a - score_b)


def build_comparison(
    *,
    hip: dict[str, Any],
    llama_rows: list[dict[str, Any]],
    cycle: int,
    row: int,
    task_id: int | None,
    candidate_tokens: list[int],
    pair: tuple[int, int] | None,
) -> dict[str, Any]:
    hip_cycle, hip_score_row = _find_hip_score_row(hip, cycle=cycle, row=row)
    draft_tokens = [int(token) for token in hip_cycle.get("draft_tokens", [])]
    llama_cycle, llama_score_row = _find_llama_score_row(
        llama_rows,
        cycle=cycle,
        row=row,
        task_id=task_id,
        draft_tokens=draft_tokens,
    )
    hip_scores = _normal_score_map(hip_score_row, source="hip")
    llama_scores = _normal_score_map(llama_score_row, source="llama")
    tokens: list[int] = []
    for token in [
        *candidate_tokens,
        *hip_scores.keys(),
        *llama_scores.keys(),
    ]:
        token = int(token)
        if token not in tokens:
            tokens.append(token)
    token_rows: list[dict[str, Any]] = []
    for token in tokens:
        hip_row = hip_scores.get(token)
        llama_row = llama_scores.get(token)
        token_rows.append(
            {
                "token": token,
                "hip": None if hip_row is None else {k: hip_row.get(k) for k in ("rank", "score", "delta_vs_top1", "source")},
                "llama": None
                if llama_row is None
                else {k: llama_row.get(k) for k in ("rank", "score", "delta_vs_top1", "source")},
                "score_delta_hip_minus_llama": (
                    None
                    if hip_row is None or llama_row is None
                    else float(float(hip_row["score"]) - float(llama_row["score"]))
                ),
            }
        )
    margin = None
    if pair is not None:
        a, b = pair
        hip_margin = _margin(hip_scores, a, b)
        llama_margin = _margin(llama_scores, a, b)
        margin = {
            "token_a": int(a),
            "token_b": int(b),
            "definition": "token_a_score - token_b_score",
            "hip": hip_margin,
            "llama": llama_margin,
            "delta_hip_minus_llama": (
                None if hip_margin is None or llama_margin is None else float(hip_margin - llama_margin)
            ),
        }
    return {
        "schema": "hipengine.mtp_target_score_compare.v1",
        "performance_claim": False,
        "selection": {
            "cycle": int(cycle),
            "row": int(row),
            "task_id": task_id,
            "draft_tokens": draft_tokens,
        },
        "hip": {
            "target_tokens": [int(token) for token in hip_cycle.get("target_tokens", [])],
            "sampled_token": int(hip_score_row.get("target_token")),
            "input_token": int(hip_score_row.get("input_token")),
            "hidden_seed_row": _find_hip_hidden_row(hip_cycle, row=row),
        },
        "llama": {
            "sampled_token_ids": [int(token) for token in llama_cycle.get("sampled_token_ids", [])],
            "sampled_token": int(llama_score_row.get("sampled_token")),
            "input_token": int(llama_score_row.get("draft_token_at_depth") or hip_score_row.get("input_token")),
        },
        "pair_margin": margin,
        "token_scores": token_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip", type=Path, required=True, help="hipEngine JSON artifact with target_lm_head_score_rows")
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True, help="llama.cpp token trace JSONL")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--candidate-tokens", default="", help="Comma-separated extra token IDs to include")
    parser.add_argument("--pair", default="", help="Two comma-separated token IDs for a focused margin")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = _parse_int_csv(args.candidate_tokens)
    pair_values = _parse_int_csv(args.pair) if args.pair else []
    if pair_values and len(pair_values) != 2:
        parser.error("--pair must contain exactly two token IDs")
    result = build_comparison(
        hip=_load_json(args.hip),
        llama_rows=_load_jsonl(args.llamacpp_jsonl),
        cycle=int(args.cycle),
        row=int(args.row),
        task_id=args.task_id,
        candidate_tokens=candidates,
        pair=(pair_values[0], pair_values[1]) if pair_values else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
