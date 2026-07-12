#!/usr/bin/env python3
"""Reconcile hipEngine and llama.cpp MTP economy denominators.

The parity dashboard compares request-level tok/s with stage-timing buckets.
Those buckets do not always share the same denominator: llama.cpp's request
summary counts all predicted tokens, while the instrumented stage summary may
exclude warmup rows or omit the final server token. This reducer makes that
explicit so row-economy work targets real gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def _round(value: Any, ndigits: int = 6) -> Any:
    return round(float(value), ndigits) if isinstance(value, (int, float)) else value


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hist_get(histograms: dict[str, Any] | None, key: str) -> dict[str, int]:
    if not isinstance(histograms, dict):
        return {}
    raw = histograms.get(key)
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items()}


def hip_request_summary(suite: dict[str, Any], *, budget: str) -> dict[str, Any]:
    row = suite["mtp_by_budget"][budget]
    accepted = int(row["total_accepted"])
    output = int(row["total_output_tokens"])
    drafts = int(row.get("total_drafts") or 0)
    if drafts == 0:
        hist = row.get("cycle_histograms", {}).get("generated_draft_tokens", {})
        drafts = sum(int(k) * int(v) for k, v in hist.items())
    return {
        "source": "hipengine_full_suite",
        "tok_s": _round(row["decode_tok_s_weighted"]),
        "output_tokens": output,
        "accepted_draft_tokens": accepted,
        "generated_draft_tokens": drafts,
        "accepted_per_output": _round(row.get("accepted_per_output") or _safe_div(accepted, output)),
        "draft_acceptance": _round(row.get("draft_acceptance") or _safe_div(accepted, drafts)),
        "cycles": sum(int(v) for v in row.get("cycle_histograms", {}).get("visible_output_tokens", {}).values()),
        "cycle_wall_ms_per_output": _round(row.get("cycle_wall_ms_per_output")),
        "target_rows_per_output": _round(row.get("target_verify_rows_per_output")),
        "discarded_rows": int(row.get("target_verify_discarded_rows") or 0),
        "discarded_rows_per_output": _round(_safe_div(row.get("target_verify_discarded_rows"), output)),
    }


def llama_request_summary(payload: dict[str, Any], *, protocol: str) -> dict[str, Any]:
    rows = payload["runs"]["mtp"]["protocols"][protocol]["rows"]
    output = sum(int(row["timings"]["predicted_n"]) for row in rows)
    accepted = sum(int(row["timings"].get("draft_n_accepted") or 0) for row in rows)
    drafts = sum(int(row["timings"].get("draft_n") or 0) for row in rows)
    summary = payload["summary"][protocol]
    return {
        "source": "llamacpp_request_rows",
        "tok_s": _round(summary["mtp_weighted_predicted_per_second"]),
        "output_tokens": output,
        "accepted_draft_tokens": accepted,
        "generated_draft_tokens": drafts,
        "accepted_per_output": _round(summary.get("mtp_accepted_per_output") or _safe_div(accepted, output)),
        "draft_acceptance": _round(summary.get("mtp_draft_acceptance") or _safe_div(accepted, drafts)),
        "predicted_ms_total": _round(sum(float(row["timings"]["predicted_ms"]) for row in rows)),
        "request_count": len(rows),
    }


def llama_stage_summary(payload: dict[str, Any], *, section: str) -> dict[str, Any] | None:
    stage = payload.get("stage_timing_summary")
    if not isinstance(stage, dict):
        return None
    row = stage.get(section)
    if not isinstance(row, dict):
        return None
    return {
        "source": f"llamacpp_stage_{section}",
        "cycles": int(row.get("cycles") or 0),
        "output_tokens": int(row.get("total_output_tokens") or 0),
        "accepted_draft_tokens": int(row.get("total_accepted") or 0),
        "generated_draft_tokens": int(row.get("total_drafts") or 0),
        "accepted_per_output": _round(row.get("accepted_per_output")),
        "draft_acceptance": _round(row.get("draft_acceptance")),
        "cycle_wall_ms_per_output": _round(row.get("cycle_wall_ms_per_output")),
        "target_rows_per_output": _round(row.get("target_verify_rows_per_output")),
        "discarded_rows_per_output": _round(row.get("target_verify_discarded_rows_per_output")),
    }


def _hip_prompt_rows(category_payload: dict[str, Any], *, budget: str) -> dict[str, dict[str, Any]]:
    raw_root = Path(category_payload["raw_root"]) / budget
    rows: dict[str, dict[str, Any]] = {}
    for prompt in category_payload.get("prompts", []):
        prompt_id = str(prompt["id"])
        prompt_path = raw_root / f"{prompt_id}.json"
        if not prompt_path.exists():
            continue
        payload = _load(prompt_path)
        metrics = payload["metrics"]
        cycles = payload.get("cycles") or []
        rows[prompt_id] = {
            "id": prompt_id,
            "category": prompt.get("category"),
            "output_tokens": int(metrics["total_output_tokens"]),
            "accepted_draft_tokens": int(metrics["total_accepted"]),
            "generated_draft_tokens": int(metrics["total_drafts"]),
            "cycles": len(cycles),
            "tok_s": _round(metrics.get("tokens_per_sec")),
            "accepted_per_output": _round(metrics.get("accepted_per_output")),
            "draft_acceptance": _round(metrics.get("accept_per_draft")),
            "target_rows_per_output": _round(metrics.get("target_verify_rows_per_output")),
            "discarded_rows": int(metrics.get("target_verify_discarded_rows") or 0),
            "accepted_histogram": _hist_get(_cycle_histograms(cycles), "accepted_draft_tokens"),
        }
    return rows


def _cycle_histograms(cycles: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    hist: dict[str, dict[str, int]] = {
        "generated_draft_tokens": {},
        "accepted_draft_tokens": {},
        "visible_output_tokens": {},
    }
    for cycle in cycles:
        for key in hist:
            value = str(int(cycle.get(key) or 0))
            hist[key][value] = hist[key].get(value, 0) + 1
    return hist


def _llama_prompt_rows(payload: dict[str, Any], *, protocol: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload["runs"]["mtp"]["protocols"][protocol]["rows"]:
        timings = row["timings"]
        output = int(timings["predicted_n"])
        accepted = int(timings.get("draft_n_accepted") or 0)
        drafts = int(timings.get("draft_n") or 0)
        rows[str(row["id"])] = {
            "id": str(row["id"]),
            "category": row.get("category"),
            "output_tokens": output,
            "accepted_draft_tokens": accepted,
            "generated_draft_tokens": drafts,
            "tok_s": _round(timings.get("predicted_per_second")),
            "predicted_ms": _round(timings.get("predicted_ms")),
            "accepted_per_output": _round(row.get("accepted_per_output") or _safe_div(accepted, output)),
            "draft_acceptance": _round(row.get("draft_acceptance") or _safe_div(accepted, drafts)),
        }
    return rows


def prompt_reconciliation(
    hip_category: dict[str, Any] | None,
    llama_payload: dict[str, Any],
    *,
    budget: str,
    protocol: str,
) -> list[dict[str, Any]]:
    if hip_category is None:
        return []
    hip_rows = _hip_prompt_rows(hip_category, budget=budget)
    llama_rows = _llama_prompt_rows(llama_payload, protocol=protocol)
    result = []
    for prompt_id in sorted(set(hip_rows) & set(llama_rows)):
        hip = hip_rows[prompt_id]
        llama = llama_rows[prompt_id]
        result.append(
            {
                "id": prompt_id,
                "category": hip.get("category") or llama.get("category"),
                "hipengine": hip,
                "llamacpp_request": llama,
                "delta_hip_minus_llama": {
                    "accepted_draft_tokens": hip["accepted_draft_tokens"] - llama["accepted_draft_tokens"],
                    "generated_draft_tokens": hip["generated_draft_tokens"] - llama["generated_draft_tokens"],
                    "accepted_per_output": _round(hip["accepted_per_output"] - llama["accepted_per_output"]),
                    "draft_acceptance": _round(hip["draft_acceptance"] - llama["draft_acceptance"]),
                    "tok_s": _round(hip["tok_s"] - llama["tok_s"]),
                },
            }
        )
    result.sort(
        key=lambda row: (
            row["delta_hip_minus_llama"]["accepted_draft_tokens"],
            row["delta_hip_minus_llama"]["tok_s"],
        )
    )
    return result


def build_reconciliation(
    hip_suite: dict[str, Any],
    llama_payload: dict[str, Any],
    *,
    hip_category: dict[str, Any] | None,
    budget: str,
    protocol: str,
) -> dict[str, Any]:
    hip_req = hip_request_summary(hip_suite, budget=budget)
    llama_req = llama_request_summary(llama_payload, protocol=protocol)
    llama_stage_all = llama_stage_summary(llama_payload, section="all")
    llama_stage_measured = llama_stage_summary(llama_payload, section="measured_excluding_first_task")

    request_acc_gap = hip_req["accepted_per_output"] - llama_req["accepted_per_output"]
    measured_acc_gap = (
        None
        if llama_stage_measured is None
        else hip_req["accepted_per_output"] - llama_stage_measured["accepted_per_output"]
    )
    prompt_rows = prompt_reconciliation(
        hip_category,
        llama_payload,
        budget=budget,
        protocol=protocol,
    )
    return {
        "schema": "hipengine.mtp_economy_reconcile.v1",
        "performance_claim": False,
        "budget": budget,
        "protocol": protocol,
        "hipengine_full_request": hip_req,
        "llamacpp_full_request": llama_req,
        "llamacpp_stage_all": llama_stage_all,
        "llamacpp_stage_measured_excluding_first_task": llama_stage_measured,
        "gaps_hip_minus_llama": {
            "request_tok_s": _round(hip_req["tok_s"] - llama_req["tok_s"]),
            "request_accepted_per_output": _round(request_acc_gap),
            "request_draft_acceptance": _round(hip_req["draft_acceptance"] - llama_req["draft_acceptance"]),
            "stage_measured_cycle_wall_ms_per_output": (
                None
                if llama_stage_measured is None
                else _round(hip_req["cycle_wall_ms_per_output"] - llama_stage_measured["cycle_wall_ms_per_output"])
            ),
            "stage_measured_accepted_per_output": None if measured_acc_gap is None else _round(measured_acc_gap),
            "stage_measured_target_rows_per_output": (
                None
                if llama_stage_measured is None
                else _round(hip_req["target_rows_per_output"] - llama_stage_measured["target_rows_per_output"])
            ),
        },
        "denominator_readout": {
            "llamacpp_request_and_stage_accepted_per_output_match": (
                None
                if llama_stage_measured is None
                else abs(llama_req["accepted_per_output"] - llama_stage_measured["accepted_per_output"]) < 1e-9
            ),
            "llamacpp_request_accepted_per_output": llama_req["accepted_per_output"],
            "llamacpp_stage_measured_accepted_per_output": (
                None if llama_stage_measured is None else llama_stage_measured["accepted_per_output"]
            ),
            "hipengine_has_full_request_acceptance_deficit": request_acc_gap < 0.0,
            "stage_acceptance_gap_uses_different_denominator": (
                False
                if llama_stage_measured is None
                else abs(llama_req["accepted_per_output"] - llama_stage_measured["accepted_per_output"]) > 1e-9
            ),
        },
        "prompt_rows_by_worst_acceptance_delta": prompt_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip-suite", type=Path, required=True)
    parser.add_argument("--hip-category", type=Path, default=None)
    parser.add_argument("--llamacpp", type=Path, required=True)
    parser.add_argument("--budget", default="b2")
    parser.add_argument("--protocol", default="natural")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = build_reconciliation(
        _load(args.hip_suite),
        _load(args.llamacpp),
        hip_category=_load(args.hip_category) if args.hip_category else None,
        budget=args.budget,
        protocol=args.protocol,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
