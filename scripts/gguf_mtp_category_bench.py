#!/usr/bin/env python3
"""Run hipEngine native GGUF-MTP diagnostics over the category prompt suite.

This is a thin wrapper around ``scripts/gguf_mtp_bench.py``.  It deliberately
uses the existing native GGUF MTP diagnostic path and aggregates its per-prompt
JSON outputs into llama.cpp-style total/category tables for off/B1..B5.

Important: ``off`` is the target-AR verifier timing measured inside the B1
diagnostic run (sum visible output tokens / sum target AR verify time), not an
independent no-MTP autoregressive generation path.  Therefore this wrapper is
acceptance/economics diagnostic evidence only: its speed ratios are not eligible
as retained "MTP beats AR" claims until the benchmark script measures a true AR
baseline under the same prompt suite and timing protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
DEFAULT_BUDGETS = "1,2,3,4,5"
DEFAULT_HELDOUT_PROMPT_IDS = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)


class BenchError(RuntimeError):
    pass


def parse_budgets(text: str) -> list[int]:
    budgets = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not budgets:
        raise BenchError("--budgets resolved to an empty list")
    bad = [b for b in budgets if b < 1 or b > 5]
    if bad:
        raise BenchError(f"budgets must be in 1..5 for gguf_mtp_bench.py: {bad}")
    return budgets


def load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt_id = str(raw.get("id") or raw.get("name") or f"prompt_{line_no}")
            category = str(raw.get("category") or "uncategorized")
            if "prompt" in raw:
                prompt_text = str(raw["prompt"])
            else:
                messages = raw.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise BenchError(f"{path}:{line_no}: expected prompt or messages[]")
                user_parts = [str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"]
                prompt_text = "\n\n".join(part for part in user_parts if part)
            if not prompt_text:
                raise BenchError(f"{path}:{line_no}: prompt text is empty")
            rows.append({"id": prompt_id, "category": category, "prompt": prompt_text, "source": raw})
    if not rows:
        raise BenchError(f"{path} contained no prompt rows")
    return rows


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def run_one(
    *,
    python: str,
    model: Path,
    prompt: str,
    budget: int,
    cycles: int,
    output: Path,
    log_path: Path,
    extra_args: list[str],
    dry_run: bool,
) -> dict[str, Any] | None:
    cmd = [
        python,
        "scripts/gguf_mtp_bench.py",
        "--model",
        str(model),
        "--draft-n-max",
        str(budget),
        "--cycles",
        str(cycles),
        "--prompt",
        prompt,
        "--output",
        str(output),
        *extra_args,
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"command": quote_command(cmd), "output": str(output), "log": str(log_path)}
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise BenchError(f"command failed with {completed.returncode}: {quote_command(cmd)}\n" + "\n".join(tail))
    data = json.loads(output.read_text(encoding="utf-8"))
    data.setdefault("wrapper", {})["subprocess_wall_seconds"] = wall
    data["wrapper"]["command"] = quote_command(cmd)
    data["wrapper"]["log"] = str(log_path)
    return data


def quote_command(cmd: list[str]) -> str:
    return " ".join(shell_quote(part) for part in cmd)


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def cycle_sum(row: dict[str, Any], key: str) -> float:
    return float(sum(float(c.get(key, 0.0) or 0.0) for c in row.get("cycles", [])))


def aggregate_rows(rows: list[dict[str, Any]], *, off_tps: float | None = None) -> dict[str, Any]:
    total_output = sum(int(row["metrics"]["total_output_tokens"]) for row in rows)
    total_accepted = sum(int(row["metrics"]["total_accepted"]) for row in rows)
    total_drafts = sum(int(row["metrics"]["total_drafts"]) for row in rows)
    total_cycle_ms = sum(float(row["metrics"].get("total_cycle_ms") or cycle_sum(row, "ar_decode_ms") + cycle_sum(row, "mtp_draft_ms")) for row in rows)
    total_ar_ms = sum(cycle_sum(row, "ar_decode_ms") for row in rows)
    decode_tps = 1000.0 * total_output / total_cycle_ms if total_cycle_ms > 0 else 0.0
    ar_tps = 1000.0 * total_output / total_ar_ms if total_ar_ms > 0 else 0.0
    return {
        "prompts": len(rows),
        "total_output_tokens": total_output,
        "total_accepted": total_accepted,
        "total_drafts": total_drafts,
        "decode_tok_s_weighted": decode_tps,
        "ar_baseline_tok_s_weighted_for_this_budget": ar_tps,
        "mtp_vs_ar_decode_ratio": decode_tps / off_tps if off_tps else (decode_tps / ar_tps if ar_tps else None),
        "draft_acceptance": total_accepted / total_drafts if total_drafts else None,
        "accepted_per_output": total_accepted / total_output if total_output else None,
        "avg_output_tokens_per_prompt": total_output / len(rows) if rows else 0.0,
    }


def aggregate_off_from_b1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_output = sum(int(row["metrics"]["total_output_tokens"]) for row in rows)
    total_ar_ms = sum(cycle_sum(row, "ar_decode_ms") for row in rows)
    return {
        "prompts": len(rows),
        "total_output_tokens": total_output,
        "decode_tok_s_weighted": 1000.0 * total_output / total_ar_ms if total_ar_ms > 0 else 0.0,
        "source": "B1 native target-AR verifier time over the same prompt rows",
        "baseline_kind": "verifier_derived_from_b1_target_ar",
        "true_autoregressive_path": False,
    }


def row_prompt_id(row: dict[str, Any]) -> str:
    return str(row.get("suite_id") or row.get("prompt_id") or row.get("id") or "")


def build_split_contract(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_ids = [str(row["id"]) for row in prompts]
    categories = sorted({str(row["category"]) for row in prompts})
    heldout_ids = [prompt_id for prompt_id in prompt_ids if prompt_id in DEFAULT_HELDOUT_PROMPT_IDS]
    train_ids = [prompt_id for prompt_id in prompt_ids if prompt_id not in DEFAULT_HELDOUT_PROMPT_IDS]
    prompt_by_id = {str(row["id"]): row for row in prompts}
    heldout_categories = sorted({str(prompt_by_id[prompt_id]["category"]) for prompt_id in heldout_ids})
    missing_default_heldouts = sorted(DEFAULT_HELDOUT_PROMPT_IDS.difference(prompt_ids))
    return {
        "strategy": "fixed_category_heldout_v1",
        "purpose": "Detect train-only acceptance/speed gains before resuming MTP optimization.",
        "default_heldout_ids": sorted(DEFAULT_HELDOUT_PROMPT_IDS),
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "full_ids": prompt_ids,
        "categories": categories,
        "heldout_categories": heldout_categories,
        "heldout_has_all_present_categories": set(heldout_categories) == set(categories),
        "missing_default_heldout_ids": missing_default_heldouts,
        "required_for_keep_decisions": True,
        "regression_rule": "Train improvements are not wins if heldout or full-suite acceptance/true-AR speed ratio regresses.",
    }


def filter_rows_by_prompt_ids(rows: list[dict[str, Any]], prompt_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row_prompt_id(row) in prompt_ids]


def aggregate_split(raw: dict[int, list[dict[str, Any]]], prompt_ids: list[str]) -> dict[str, Any]:
    prompt_id_set = set(prompt_ids)
    b1_rows = filter_rows_by_prompt_ids(raw[min(raw)], prompt_id_set)
    off = aggregate_off_from_b1(b1_rows)
    metrics: dict[str, Any] = {"off": {"label": "off", "budget": 0, **off, "mtp_vs_ar_decode_ratio": 1.0}}
    for budget, rows in sorted(raw.items()):
        split_rows = filter_rows_by_prompt_ids(rows, prompt_id_set)
        metrics[f"b{budget}"] = {
            "label": f"b{budget}",
            "budget": budget,
            **aggregate_rows(split_rows, off_tps=off["decode_tok_s_weighted"]),
        }
    return {
        "prompt_ids": prompt_ids,
        "metrics": metrics,
    }


def build_split_summaries(prompts: list[dict[str, Any]], raw: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    contract = build_split_contract(prompts)
    return {
        "contract": contract,
        "full": aggregate_split(raw, contract["full_ids"]),
        "train": aggregate_split(raw, contract["train_ids"]),
        "heldout": aggregate_split(raw, contract["heldout_ids"]),
    }


def true_ar_rows_from_artifact(path: Path) -> list[dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise BenchError("true AR baseline artifact must be a JSON object")
    if artifact.get("true_autoregressive_path") is not True:
        raise BenchError("true AR baseline artifact must set true_autoregressive_path=true")
    if artifact.get("same_prompt_suite") is not True:
        raise BenchError("true AR baseline artifact must set same_prompt_suite=true")
    if artifact.get("same_timing_protocol") is not True:
        raise BenchError("true AR baseline artifact must set same_timing_protocol=true")
    rows = artifact.get("prompt_metrics")
    if not isinstance(rows, list) or not rows:
        raise BenchError("true AR baseline artifact must contain non-empty prompt_metrics[]")
    return rows


def validate_true_ar_prompt_rows(*, rows: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_by_id = {str(row["id"]): row for row in prompts}
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = str(row.get("id") or row.get("prompt_id") or "")
        if not prompt_id:
            raise BenchError("true AR prompt_metrics row missing id")
        if prompt_id in seen:
            raise BenchError(f"duplicate true AR prompt_metrics id: {prompt_id}")
        if prompt_id not in prompt_by_id:
            raise BenchError(f"true AR prompt id not in selected prompt suite: {prompt_id}")
        category = str(row.get("category") or "")
        expected_category = str(prompt_by_id[prompt_id]["category"])
        if category != expected_category:
            raise BenchError(f"true AR category mismatch for {prompt_id}: {category!r} != {expected_category!r}")
        output_tokens = int(row.get("output_tokens") or 0)
        decode_ms = float(row.get("decode_ms") or 0.0)
        if output_tokens <= 0 or decode_ms <= 0.0:
            raise BenchError(f"true AR row for {prompt_id} must have positive output_tokens and decode_ms")
        seen[prompt_id] = {"id": prompt_id, "category": category, "output_tokens": output_tokens, "decode_ms": decode_ms}
    expected_ids = set(prompt_by_id)
    seen_ids = set(seen)
    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        extra = sorted(seen_ids - expected_ids)
        raise BenchError(f"true AR prompt_metrics must exactly match selected prompts; missing={missing}, extra={extra}")
    return seen


def aggregate_true_ar_rows(rows_by_id: dict[str, dict[str, Any]], prompt_ids: list[str]) -> dict[str, Any]:
    rows = [rows_by_id[prompt_id] for prompt_id in prompt_ids]
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    return {
        "prompts": len(rows),
        "total_output_tokens": output_tokens,
        "decode_ms": decode_ms,
        "decode_tok_s_weighted": 1000.0 * output_tokens / decode_ms if decode_ms > 0 else 0.0,
    }


def attach_true_ar_baseline(summary: dict[str, Any], *, rows_by_id: dict[str, dict[str, Any]], source: Path) -> dict[str, Any]:
    prompt_ids = [str(row["id"]) for row in summary["prompts"]]
    full_metric = aggregate_true_ar_rows(rows_by_id, prompt_ids)
    category_metrics = {
        category: aggregate_true_ar_rows(rows_by_id, [row["id"] for row in summary["prompts"] if row["category"] == category])
        for category in sorted(summary["categories"])
    }
    split_metrics = {
        split_name: aggregate_true_ar_rows(rows_by_id, payload["prompt_ids"])
        for split_name, payload in summary["splits"].items()
        if split_name != "contract"
    }

    summary["true_ar_baseline"] = {
        "available": True,
        "true_autoregressive_path": True,
        "same_prompt_suite": True,
        "same_timing_protocol": True,
        "source": str(source),
        "prompt_count": len(prompt_ids),
        "totals": full_metric,
        "categories": category_metrics,
        "splits": split_metrics,
    }
    summary["true_ar_comparison_available"] = True
    summary["speed_claim_eligible"] = False
    summary["promotion_blocker"] = (
        "true_ar_baseline is attached and mtp_vs_true_ar_decode_ratio is available, "
        "but this diagnostic category artifact is not a retained speed claim. "
        "Promotion still requires the retained benchmark protocol: full validation shape, "
        "hermetic/warm timing, and benchmark rollup evidence."
    )
    summary["diagnostic_notes"].append(
        "A true no-MTP AR baseline artifact was attached; mtp_vs_true_ar_decode_ratio is available for same-protocol diagnostics, but speed_claim_eligible remains false until the retained benchmark protocol is satisfied."
    )

    true_ar_tps = full_metric["decode_tok_s_weighted"]
    for label, row in summary["totals"].items():
        row["true_ar_decode_tok_s_weighted"] = true_ar_tps
        if label != "off":
            row["mtp_vs_true_ar_decode_ratio"] = row["decode_tok_s_weighted"] / true_ar_tps if true_ar_tps else None

    for category, payload in summary["categories"].items():
        category_tps = category_metrics[category]["decode_tok_s_weighted"]
        for label, row in payload.items():
            row["true_ar_decode_tok_s_weighted"] = category_tps
            if label != "off":
                row["mtp_vs_true_ar_decode_ratio"] = row["decode_tok_s_weighted"] / category_tps if category_tps else None

    for split_name, payload in summary["splits"].items():
        if split_name == "contract":
            continue
        split_tps = split_metrics[split_name]["decode_tok_s_weighted"]
        for label, row in payload["metrics"].items():
            row["true_ar_decode_tok_s_weighted"] = split_tps
            if label != "off":
                row["mtp_vs_true_ar_decode_ratio"] = row["decode_tok_s_weighted"] / split_tps if split_tps else None

    return validate_speed_claim_contract(summary)


def validate_speed_claim_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Ensure speed-promotable artifacts contain a real AR baseline.

    Verifier-derived ``off`` rows are useful economics telemetry, but they are
    not an independently measured no-MTP autoregressive path. Keep this as a
    machine-checkable invariant so future harness edits cannot accidentally flip
    ``speed_claim_eligible`` back on without adding the true baseline evidence.
    """
    if not summary.get("speed_claim_eligible", False):
        return summary
    true_ar = summary.get("true_ar_baseline")
    if not isinstance(true_ar, dict):
        raise BenchError("speed_claim_eligible=true requires true_ar_baseline metadata")
    if true_ar.get("true_autoregressive_path") is not True:
        raise BenchError("speed_claim_eligible=true requires a true no-MTP autoregressive baseline")
    if true_ar.get("same_prompt_suite") is not True:
        raise BenchError("speed_claim_eligible=true requires true AR measured on the same prompt suite")
    if true_ar.get("same_timing_protocol") is not True:
        raise BenchError("speed_claim_eligible=true requires true AR measured with the same timing protocol")
    return summary


def build_summary(*, args: argparse.Namespace, prompts: list[dict[str, Any]], raw: dict[int, list[dict[str, Any]]], commands: list[str]) -> dict[str, Any]:
    b1_rows = raw[min(raw)]
    off_total = aggregate_off_from_b1(b1_rows)
    categories = sorted({row["category"] for row in prompts})
    prompt_meta = {row["id"]: row for row in prompts}

    totals: dict[str, Any] = {"off": {"label": "off", "budget": 0, **off_total, "mtp_vs_ar_decode_ratio": 1.0}}
    off_by_category: dict[str, dict[str, Any]] = {}
    for category in categories:
        b1_cat = [row for row in b1_rows if row["suite_category"] == category]
        off_by_category[category] = aggregate_off_from_b1(b1_cat)

    category_summary: dict[str, dict[str, Any]] = {
        category: {
            "off": {
                "label": "off",
                "budget": 0,
                **off_by_category[category],
                "mtp_vs_ar_decode_ratio": 1.0,
            }
        }
        for category in categories
    }

    for budget, rows in sorted(raw.items()):
        totals[f"b{budget}"] = {
            "label": f"b{budget}",
            "budget": budget,
            **aggregate_rows(rows, off_tps=off_total["decode_tok_s_weighted"]),
        }
        for category in categories:
            cat_rows = [row for row in rows if row["suite_category"] == category]
            category_summary[category][f"b{budget}"] = {
                "label": f"b{budget}",
                "budget": budget,
                **aggregate_rows(cat_rows, off_tps=off_by_category[category]["decode_tok_s_weighted"]),
            }

    best = {
        "total_by_decode_tok_s": max((v for k, v in totals.items() if k != "off"), key=lambda x: x["decode_tok_s_weighted"])["label"],
        "total_by_accepted_per_output": max((v for k, v in totals.items() if k != "off"), key=lambda x: x["accepted_per_output"])["label"],
        "categories_by_decode_tok_s": {},
        "categories_by_accepted_per_output": {},
    }
    for category, payload in category_summary.items():
        best["categories_by_decode_tok_s"][category] = max((v for k, v in payload.items() if k != "off"), key=lambda x: x["decode_tok_s_weighted"])["label"]
        best["categories_by_accepted_per_output"][category] = max((v for k, v in payload.items() if k != "off"), key=lambda x: x["accepted_per_output"])["label"]

    split_summaries = build_split_summaries(prompts, raw)

    true_ar_baseline_json = getattr(args, "true_ar_baseline_json", None)

    summary = {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_category_matrix",
        "status": "diagnostic_retained",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "true_ar_comparison_available": False,
        "promotion_blocker": (
            "off/AR is derived from B1 target-verifier timing, not a true no-MTP "
            "autoregressive generation path. MTP speed promotion requires a separate "
            "true AR baseline measured by the benchmark script over the same prompt suite."
        ),
        "ar_baseline_contract": {
            "required_for_speed_claims": "true_no_mtp_autoregressive_generation",
            "current_off_kind": "verifier_derived_from_b1_target_ar",
            "current_off_true_autoregressive_path": False,
        },
        "true_ar_baseline": {
            "available": False,
            "true_autoregressive_path": False,
            "same_prompt_suite": False,
            "same_timing_protocol": False,
            "source": None,
        },
        "diagnostic_notes": [
            "Native GGUF-MTP category wrapper around scripts/gguf_mtp_bench.py.",
            "Each prompt/budget is a separate process and model load; tok/s metrics use child JSON cycle timings only, not wrapper subprocess wall time.",
            "off/AR row is derived from B1 target-AR verifier timing and is not a true no-MTP autoregressive baseline.",
            "MTP-vs-AR speed claims are blocked until this harness measures a true AR/no-MTP path over the same prompt suite and timing protocol.",
            "cycles is verify-cycle count, not llama.cpp max_tokens; accepted drafts add visible output tokens.",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "quant": "UD-Q4_K_M GGUF with MTP blocks",
        "prompt_file": str(args.prompts),
        "cycles": int(args.cycles),
        "budgets": sorted(raw),
        "raw_root": str(args.raw_root),
        "commands": commands,
        "totals": totals,
        "categories": category_summary,
        "splits": split_summaries,
        "best": best,
        "prompts": [
            {"id": row["id"], "category": row["category"], "prompt_chars": len(row["prompt"])}
            for row in prompts
        ],
    }
    if true_ar_baseline_json:
        true_ar_path = Path(true_ar_baseline_json)
        rows_by_id = validate_true_ar_prompt_rows(rows=true_ar_rows_from_artifact(true_ar_path), prompts=prompts)
        return attach_true_ar_baseline(summary, rows_by_id=rows_by_id, source=true_ar_path)
    return validate_speed_claim_contract(summary)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    def fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines: list[str] = []
    has_true_ar = (summary.get("true_ar_baseline") or {}).get("available") is True
    lines.append("# hipEngine GGUF-MTP category matrix")
    lines.append("")
    lines.append(f"Raw root: `{summary['raw_root']}`")
    if has_true_ar:
        lines.append("")
        lines.append("> **Diagnostic only:** true no-MTP AR baseline attached, so `vs true AR` is available for same-protocol diagnostics. This artifact is still not a retained speed claim unless `speed_claim_eligible=true` and `performance_claim=true`; `vs verifier off` remains diagnostic telemetry.")
    elif not summary.get("speed_claim_eligible", True):
        lines.append("")
        lines.append("> **Diagnostic only:** the `off` row is derived from B1 target-verifier timing, not a true no-MTP autoregressive run. Do not use `vs verifier off` as a retained MTP speedup claim until a true AR baseline is measured by this harness.")
    splits = summary.get("splits") or {}
    contract = splits.get("contract") or {}
    if contract:
        lines.append("")
        lines.append("## Train / heldout split")
        lines.append(f"Strategy: `{contract.get('strategy')}`")
        lines.append(f"Train prompts: `{', '.join(contract.get('train_ids', []))}`")
        lines.append(f"Heldout prompts: `{', '.join(contract.get('heldout_ids', []))}`")
        lines.append(f"Heldout covers all present categories: `{contract.get('heldout_has_all_present_categories')}`")
        lines.append("")
        true_ar_header = " | vs true AR" if has_true_ar else ""
        true_ar_align = " |---:" if has_true_ar else ""
        lines.append(f"| split | budget | decode tok/s | vs verifier off{true_ar_header} | draft accept | accepted/output | prompts |")
        lines.append(f"|---|---|---:|---:{true_ar_align}|---:|---:|---:|")
        for split_name in ("full", "train", "heldout"):
            metrics = (splits.get(split_name) or {}).get("metrics") or {}
            for label, row in metrics.items():
                if label == "off":
                    continue
                true_ar_cell = f" | {fmt(row.get('mtp_vs_true_ar_decode_ratio'), 3)}" if has_true_ar else ""
                lines.append(
                    f"| {split_name} | {label} | {fmt(row['decode_tok_s_weighted'])} | {fmt(row['mtp_vs_ar_decode_ratio'], 3)}{true_ar_cell} | "
                    f"{fmt(row.get('draft_acceptance'), 4)} | {fmt(row.get('accepted_per_output'), 4)} | {row['prompts']} |"
                )
    lines.append("")
    lines.append("## Total")
    true_ar_header = " | vs true AR" if has_true_ar else ""
    true_ar_align = " |---:" if has_true_ar else ""
    lines.append(f"| budget | decode tok/s | vs verifier off{true_ar_header} | draft accept | accepted/output | output tokens |")
    lines.append(f"|---|---:|---:{true_ar_align}|---:|---:|---:|")
    for label, row in summary["totals"].items():
        true_ar_cell = f" | {fmt(row.get('mtp_vs_true_ar_decode_ratio'), 3)}" if has_true_ar else ""
        lines.append(
            f"| {label} | {fmt(row['decode_tok_s_weighted'])} | {fmt(row['mtp_vs_ar_decode_ratio'], 3)}{true_ar_cell} | "
            f"{fmt(row.get('draft_acceptance'), 4)} | {fmt(row.get('accepted_per_output'), 4)} | {row['total_output_tokens']} |"
        )
    for category, payload in sorted(summary["categories"].items()):
        lines.append("")
        lines.append(f"## {category}")
        lines.append(f"| budget | decode tok/s | vs verifier off{true_ar_header} | draft accept | accepted/output | output tokens |")
        lines.append(f"|---|---:|---:{true_ar_align}|---:|---:|---:|")
        for label, row in payload.items():
            true_ar_cell = f" | {fmt(row.get('mtp_vs_true_ar_decode_ratio'), 3)}" if has_true_ar else ""
            lines.append(
                f"| {label} | {fmt(row['decode_tok_s_weighted'])} | {fmt(row['mtp_vs_ar_decode_ratio'], 3)}{true_ar_cell} | "
                f"{fmt(row.get('draft_acceptance'), 4)} | {fmt(row.get('accepted_per_output'), 4)} | {row['total_output_tokens']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument(
        "--true-ar-baseline-json",
        type=Path,
        default=None,
        help=(
            "Optional same-protocol true no-MTP AR baseline artifact with prompt_metrics[]. "
            "When valid, the summary includes mtp_vs_true_ar_decode_ratio and may be speed-claim eligible."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    prompts = load_prompt_rows(args.prompts)
    if args.limit is not None:
        prompts = prompts[: max(0, int(args.limit))]
    if not prompts:
        raise BenchError("selected prompt list is empty")
    if not args.model.exists():
        raise BenchError(f"model not found: {args.model}")
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.raw_root is None:
        args.raw_root = Path(f"/tmp/hipengine-gguf-mtp-category-{run_tag}")
    if args.output is None:
        args.output = args.raw_root / "summary-category-off-b1-b5.json"
    args.raw_root.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    raw: dict[int, list[dict[str, Any]]] = {}
    for budget in budgets:
        raw[budget] = []
        for row in prompts:
            out = args.raw_root / f"b{budget}" / f"{safe_name(row['id'])}.json"
            log = args.raw_root / f"b{budget}" / f"{safe_name(row['id'])}.log"
            result = run_one(
                python=args.python,
                model=args.model,
                prompt=row["prompt"],
                budget=budget,
                cycles=args.cycles,
                output=out,
                log_path=log,
                extra_args=list(args.extra_arg),
                dry_run=bool(args.dry_run),
            )
            cmd = result["command"] if args.dry_run and isinstance(result, dict) else result["wrapper"]["command"] if isinstance(result, dict) else ""
            commands.append(cmd)
            print(f"B{budget} {row['id']}: {out}", flush=True)
            if args.dry_run:
                continue
            assert result is not None
            result["suite_id"] = row["id"]
            result["suite_category"] = row["category"]
            result["suite_prompt_chars"] = len(row["prompt"])
            # Persist metadata into the child artifact too.
            child = json.loads(out.read_text(encoding="utf-8"))
            child["suite"] = {
                "id": row["id"],
                "category": row["category"],
                "prompt_file": str(args.prompts),
            }
            child.setdefault("wrapper", {})["command"] = result["wrapper"]["command"]
            child["wrapper"]["log"] = result["wrapper"]["log"]
            child["wrapper"]["subprocess_wall_seconds"] = result["wrapper"]["subprocess_wall_seconds"]
            out.write_text(json.dumps(child, indent=2) + "\n", encoding="utf-8")
            result = child
            result["suite_id"] = row["id"]
            result["suite_category"] = row["category"]
            result["suite_prompt_chars"] = len(row["prompt"])
            raw[budget].append(result)

    if args.dry_run:
        dry = {"dry_run": True, "commands": commands, "raw_root": str(args.raw_root)}
        args.output.write_text(json.dumps(dry, indent=2) + "\n", encoding="utf-8")
        return 0

    summary = build_summary(args=args, prompts=prompts, raw=raw, commands=commands)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.output.with_suffix(".md"))
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
