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
import hashlib
import json
import math
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
DEFAULT_FULL_PROMPT_IDS = (
    "code_merge_intervals",
    "code_topological_sort",
    "code_lru_cache",
    "code_markdown_table",
    "general_en_plan",
    "general_en_explain",
    "general_ja_plan",
    "general_ja_explain",
    "mixed_ja_en_translate",
    "mixed_ja_en_review",
)
DEFAULT_HELDOUT_PROMPT_IDS = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)
REPO_PROVENANCE_FIELDS = ("repo_root", "git_commit", "git_branch", "git_tracked_dirty", "git_untracked_count")
MTP_CATEGORY_SCHEMA = 1
MTP_CATEGORY_KIND = "hipengine_gguf_mtp_category_matrix"
TRUE_AR_SCHEMA = 1
TRUE_AR_KIND = "hipengine_gguf_true_ar_category_baseline"
TRUE_AR_PROTOCOL_FIELDS = ("model", "quant", "prompt_file", "prompt_count", "decode_tokens", "warmup_decode_tokens")


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


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def git_output(args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def repo_provenance() -> dict[str, Any]:
    tracked_dirty = None
    diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    staged = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if diff.returncode in {0, 1} and staged.returncode in {0, 1}:
        tracked_dirty = diff.returncode == 1 or staged.returncode == 1
    untracked = git_output(["ls-files", "--others", "--exclude-standard"])
    untracked_count = None if untracked is None else len([line for line in untracked.splitlines() if line.strip()])
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_tracked_dirty": tracked_dirty,
        "git_untracked_count": untracked_count,
    }


def validate_artifact_schema(payload: dict[str, Any], *, label: str, kind: str, schema: int) -> dict[str, Any]:
    if payload.get("kind") != kind:
        raise BenchError(f"{label} requires kind={kind!r}")
    if payload.get("schema") != schema:
        raise BenchError(f"{label} requires schema={schema}")
    return {"schema": schema, "kind": kind}


def validate_category_summary_schema(summary: dict[str, Any], *, label: str) -> dict[str, Any]:
    return validate_artifact_schema(summary, label=label, kind=MTP_CATEGORY_KIND, schema=MTP_CATEGORY_SCHEMA)


def validate_true_ar_artifact_schema(artifact: dict[str, Any], *, label: str) -> dict[str, Any]:
    return validate_artifact_schema(artifact, label=label, kind=TRUE_AR_KIND, schema=TRUE_AR_SCHEMA)


def validate_attached_true_ar_artifact_schema(true_ar: dict[str, Any], *, label: str) -> dict[str, Any]:
    payload = {"kind": true_ar.get("artifact_kind"), "schema": true_ar.get("artifact_schema")}
    return validate_true_ar_artifact_schema(payload, label=label)


def validate_repo_provenance(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    repo = payload.get("repo")
    if not isinstance(repo, dict):
        raise BenchError(f"{label} requires repo provenance metadata")
    missing = [field for field in REPO_PROVENANCE_FIELDS if field not in repo]
    if missing:
        raise BenchError(f"{label} repo provenance missing fields: {missing}")
    if not isinstance(repo.get("repo_root"), str) or not repo["repo_root"]:
        raise BenchError(f"{label} repo provenance requires non-empty repo_root")
    if repo.get("git_commit") is not None and not isinstance(repo.get("git_commit"), str):
        raise BenchError(f"{label} repo provenance git_commit must be a string or null")
    if repo.get("git_branch") is not None and not isinstance(repo.get("git_branch"), str):
        raise BenchError(f"{label} repo provenance git_branch must be a string or null")
    if repo.get("git_tracked_dirty") is not None and not isinstance(repo.get("git_tracked_dirty"), bool):
        raise BenchError(f"{label} repo provenance git_tracked_dirty must be a bool or null")
    if repo.get("git_untracked_count") is not None:
        if not isinstance(repo.get("git_untracked_count"), int) or repo["git_untracked_count"] < 0:
            raise BenchError(f"{label} repo provenance git_untracked_count must be a non-negative integer or null")
    return {field: repo.get(field) for field in REPO_PROVENANCE_FIELDS}


def validate_command_provenance(payload: dict[str, Any], *, label: str) -> list[str]:
    commands = payload.get("commands")
    if not isinstance(commands, list) or not commands:
        raise BenchError(f"{label} requires non-empty commands provenance")
    out: list[str] = []
    for index, command in enumerate(commands):
        if not isinstance(command, str) or not command.strip():
            raise BenchError(f"{label} commands[{index}] must be a non-empty string")
        out.append(command)
    return out


def validate_sha256_hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise BenchError(f"{label} requires 64-character SHA-256 hex")
    return value.lower()


def validate_summary_prompt_metadata(summary: dict[str, Any], *, label: str) -> dict[str, Any]:
    prompts = summary.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise BenchError(f"{label} requires non-empty prompt metadata")
    prompt_ids: list[str] = []
    prompt_hashes: dict[str, str] = {}
    categories: list[str] = []
    category_counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for index, row in enumerate(prompts):
        if not isinstance(row, dict):
            raise BenchError(f"{label} prompts[{index}] must be an object")
        prompt_id = row.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise BenchError(f"{label} prompts[{index}] requires non-empty id")
        if prompt_id in seen:
            raise BenchError(f"{label} contains duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        category = row.get("category")
        if not isinstance(category, str) or not category:
            raise BenchError(f"{label} prompt {prompt_id} requires non-empty category")
        prompt_chars = row.get("prompt_chars")
        if type(prompt_chars) is not int or prompt_chars <= 0:
            raise BenchError(f"{label} prompt {prompt_id} requires positive prompt_chars")
        prompt_hash = validate_sha256_hex(row.get("prompt_sha256"), label=f"{label} prompt {prompt_id}")
        prompt_ids.append(prompt_id)
        prompt_hashes[prompt_id] = prompt_hash
        categories.append(category)
        category_counts[category] += 1

    summary_categories = summary.get("categories")
    if not isinstance(summary_categories, dict) or not summary_categories:
        raise BenchError(f"{label} requires category summary metadata")
    if set(categories) != set(map(str, summary_categories.keys())):
        raise BenchError(f"{label} prompt categories do not match category summary keys")
    for category in sorted(set(categories)):
        category_payload = summary_categories.get(category)
        if not isinstance(category_payload, dict) or not category_payload:
            raise BenchError(f"{label} requires non-empty category metrics for {category}")
    split_contract = (summary.get("splits") or {}).get("contract") if isinstance(summary.get("splits"), dict) else None
    if isinstance(split_contract, dict) and "full_ids" in split_contract:
        full_ids = [str(prompt_id) for prompt_id in split_contract.get("full_ids") or []]
        if prompt_ids != full_ids:
            raise BenchError(f"{label} prompt ids must match splits.contract.full_ids")
    return {
        "prompt_count": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "prompt_hashes": prompt_hashes,
        "categories": sorted(set(categories)),
        "category_counts": dict(sorted(category_counts.items())),
    }


def validate_summary_category_budget_metrics(
    summary: dict[str, Any],
    *,
    label: str,
    budget_label: str,
    category_counts: dict[str, int],
) -> dict[str, int]:
    categories_payload = summary.get("categories")
    if not isinstance(categories_payload, dict):
        raise BenchError(f"{label} requires category summary metadata")
    for category, expected_count in sorted(category_counts.items()):
        table = categories_payload.get(category)
        if not isinstance(table, dict):
            raise BenchError(f"{label} requires category summary metadata for {category}")
        row = table.get(budget_label)
        if not isinstance(row, dict):
            raise BenchError(f"{label} category {category} requires {budget_label} metrics")
        required = (
            "accepted_per_output",
            "draft_acceptance",
            "decode_tok_s_weighted",
            "mtp_vs_true_ar_decode_ratio",
            "prompts",
        )
        missing = [field for field in required if field not in row or row[field] is None]
        if missing:
            raise BenchError(f"{label} category {category}.{budget_label} missing fields: {missing}")
        try:
            prompts_count = int(row.get("prompts"))
        except (TypeError, ValueError) as exc:
            raise BenchError(f"{label} category {category}.{budget_label}.prompts must match prompt metadata") from exc
        if prompts_count != expected_count:
            raise BenchError(f"{label} category {category}.{budget_label}.prompts must match prompt metadata")
        finite_unit_interval_objective(row["accepted_per_output"], label=f"{label} category {category}.{budget_label}.accepted_per_output")
        finite_unit_interval_objective(row["draft_acceptance"], label=f"{label} category {category}.{budget_label}.draft_acceptance")
        finite_nonnegative_objective(row["decode_tok_s_weighted"], label=f"{label} category {category}.{budget_label}.decode_tok_s_weighted")
        finite_nonnegative_objective(row["mtp_vs_true_ar_decode_ratio"], label=f"{label} category {category}.{budget_label}.mtp_vs_true_ar_decode_ratio")
    return dict(sorted(category_counts.items()))


def normalize_protocol_path(value: Any) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve(strict=False))


def normalize_quant_label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchError("quant protocol metadata requires non-empty quant")
    label = " ".join(value.split())
    suffix = " with MTP blocks"
    if label.endswith(suffix):
        label = label[: -len(suffix)].strip()
    return label


def validate_true_ar_protocol_metadata(*, artifact: dict[str, Any], args: argparse.Namespace, prompt_count: int, expected_quant: str) -> dict[str, Any]:
    missing = [field for field in TRUE_AR_PROTOCOL_FIELDS if field not in artifact]
    if missing:
        raise BenchError(f"true AR baseline artifact protocol metadata missing fields: {missing}")
    model = artifact.get("model")
    if not isinstance(model, str) or not model:
        raise BenchError("true AR baseline artifact protocol metadata requires non-empty model")
    quant = artifact.get("quant")
    quant_normalized = normalize_quant_label(quant)
    expected_quant_normalized = normalize_quant_label(expected_quant)
    if quant_normalized != expected_quant_normalized:
        raise BenchError(f"true AR quant mismatch: {quant!r} != {expected_quant!r}")
    prompt_file = artifact.get("prompt_file")
    if not isinstance(prompt_file, str) or not prompt_file:
        raise BenchError("true AR baseline artifact protocol metadata requires non-empty prompt_file")
    artifact_prompt_count = artifact.get("prompt_count")
    if not isinstance(artifact_prompt_count, int) or artifact_prompt_count <= 0:
        raise BenchError("true AR baseline artifact protocol metadata requires positive prompt_count")
    decode_tokens = artifact.get("decode_tokens")
    if not isinstance(decode_tokens, int) or decode_tokens <= 0:
        raise BenchError("true AR baseline artifact protocol metadata requires positive decode_tokens")
    warmup_decode_tokens = artifact.get("warmup_decode_tokens")
    if not isinstance(warmup_decode_tokens, int) or warmup_decode_tokens < 0:
        raise BenchError("true AR baseline artifact protocol metadata requires non-negative warmup_decode_tokens")
    if artifact_prompt_count != prompt_count:
        raise BenchError(f"true AR prompt_count mismatch: {artifact_prompt_count} != {prompt_count}")
    model_normalized = normalize_protocol_path(model)
    expected_model_normalized = normalize_protocol_path(getattr(args, "model"))
    if model_normalized != expected_model_normalized:
        raise BenchError(f"true AR model path mismatch: {model!r} != {str(getattr(args, 'model'))!r}")
    prompt_file_normalized = normalize_protocol_path(prompt_file)
    expected_prompt_file_normalized = normalize_protocol_path(getattr(args, "prompts"))
    if prompt_file_normalized != expected_prompt_file_normalized:
        raise BenchError(f"true AR prompt_file mismatch: {prompt_file!r} != {str(getattr(args, 'prompts'))!r}")
    return {
        "model": model,
        "model_normalized": model_normalized,
        "quant": quant,
        "quant_normalized": quant_normalized,
        "prompt_file": prompt_file,
        "prompt_file_normalized": prompt_file_normalized,
        "decode_tokens": decode_tokens,
        "warmup_decode_tokens": warmup_decode_tokens,
        "prompt_count": artifact_prompt_count,
    }


def validate_attached_true_ar_protocol(true_ar: dict[str, Any], *, label: str) -> dict[str, Any]:
    protocol = true_ar.get("protocol")
    if not isinstance(protocol, dict):
        raise BenchError(f"{label} requires true AR protocol metadata")
    for field in ("model", "model_normalized", "quant", "quant_normalized", "prompt_file", "prompt_file_normalized"):
        if not isinstance(protocol.get(field), str) or not protocol[field]:
            raise BenchError(f"{label} protocol metadata requires non-empty {field}")
    prompt_count = protocol.get("prompt_count")
    if not isinstance(prompt_count, int) or prompt_count <= 0:
        raise BenchError(f"{label} protocol metadata requires positive prompt_count")
    decode_tokens = protocol.get("decode_tokens")
    if not isinstance(decode_tokens, int) or decode_tokens <= 0:
        raise BenchError(f"{label} protocol metadata requires positive decode_tokens")
    warmup_decode_tokens = protocol.get("warmup_decode_tokens")
    if not isinstance(warmup_decode_tokens, int) or warmup_decode_tokens < 0:
        raise BenchError(f"{label} protocol metadata requires non-negative warmup_decode_tokens")
    return {
        "model": protocol["model"],
        "model_normalized": protocol["model_normalized"],
        "quant": protocol["quant"],
        "quant_normalized": protocol["quant_normalized"],
        "prompt_file": protocol["prompt_file"],
        "prompt_file_normalized": protocol["prompt_file_normalized"],
        "decode_tokens": decode_tokens,
        "warmup_decode_tokens": warmup_decode_tokens,
        "prompt_count": prompt_count,
    }


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


def finite_float(value: Any, *, prompt_id: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchError(f"non-numeric timing field {field} for {prompt_id}: {value!r}") from exc
    if not math.isfinite(result):
        raise BenchError(f"non-finite timing field {field} for {prompt_id}: {value!r}")
    return result


def finite_nonnegative_objective(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchError(f"objective metrics require finite non-negative {label}") from exc
    if not math.isfinite(result) or result < 0.0:
        raise BenchError(f"objective metrics require finite non-negative {label}")
    return result


def finite_positive_objective(value: Any, *, label: str) -> float:
    result = finite_nonnegative_objective(value, label=label)
    if result <= 0.0:
        raise BenchError(f"objective metrics require positive {label}")
    return result


def finite_unit_interval_objective(value: Any, *, label: str) -> float:
    result = finite_nonnegative_objective(value, label=label)
    if result > 1.0:
        raise BenchError(f"objective metrics require 0<= {label} <=1")
    return result


def validate_metric_row(row: dict[str, Any]) -> None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise BenchError("category row missing metrics")
    prompt_id = str(row.get("prompt_id") or row.get("suite_id") or row.get("id") or "<unknown>")
    total_output = int(metrics.get("total_output_tokens") or 0)
    total_accepted = int(metrics.get("total_accepted") or 0)
    total_drafts = int(metrics.get("total_drafts") or 0)
    if total_output <= 0:
        raise BenchError(f"non-positive output token count for {prompt_id}: {total_output}")
    if total_accepted < 0 or total_drafts < 0:
        raise BenchError(f"negative metric in row {prompt_id}: accepted={total_accepted}, drafts={total_drafts}")
    if total_accepted > total_drafts:
        raise BenchError(f"accepted draft tokens exceed proposed drafts for {prompt_id}: {total_accepted} > {total_drafts}")

    cycles = row.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise BenchError(f"category row missing non-empty cycles for {prompt_id}")
    total_ar_ms = 0.0
    total_mtp_ms = 0.0
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict):
            raise BenchError(f"cycle {index} for {prompt_id} is not an object")
        ar_ms = finite_float(cycle.get("ar_decode_ms", 0.0) or 0.0, prompt_id=prompt_id, field=f"cycles[{index}].ar_decode_ms")
        mtp_ms = finite_float(cycle.get("mtp_draft_ms", 0.0) or 0.0, prompt_id=prompt_id, field=f"cycles[{index}].mtp_draft_ms")
        if ar_ms < 0.0 or mtp_ms < 0.0:
            raise BenchError(f"negative timing in row {prompt_id}: ar_decode_ms={ar_ms}, mtp_draft_ms={mtp_ms}")
        total_ar_ms += ar_ms
        total_mtp_ms += mtp_ms
    if total_ar_ms <= 0.0:
        raise BenchError(f"non-positive total ar_decode_ms for {prompt_id}: {total_ar_ms}")

    total_cycle_raw = metrics.get("total_cycle_ms")
    total_cycle_ms = finite_float(total_cycle_raw, prompt_id=prompt_id, field="metrics.total_cycle_ms") if total_cycle_raw is not None else total_ar_ms + total_mtp_ms
    if total_cycle_ms <= 0.0:
        raise BenchError(f"non-positive total_cycle_ms for {prompt_id}: {total_cycle_ms}")


def aggregate_rows(rows: list[dict[str, Any]], *, off_tps: float | None = None) -> dict[str, Any]:
    for row in rows:
        validate_metric_row(row)
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
    for row in rows:
        validate_metric_row(row)
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


def row_category(row: dict[str, Any]) -> str:
    return str(row.get("suite_category") or row.get("category") or "")


def validate_raw_prompt_coverage(*, prompts: list[dict[str, Any]], raw: dict[int, list[dict[str, Any]]]) -> None:
    expected_ids = [str(row["id"]) for row in prompts]
    expected_set = set(expected_ids)
    expected_category = {str(row["id"]): str(row["category"]) for row in prompts}
    if not raw:
        raise BenchError("category summary requires at least one MTP budget")
    for budget, rows in sorted(raw.items()):
        seen: dict[str, int] = {}
        for row in rows:
            prompt_id = row_prompt_id(row)
            if not prompt_id:
                raise BenchError(f"budget b{budget} contains row without prompt id")
            seen[prompt_id] = seen.get(prompt_id, 0) + 1
            if prompt_id not in expected_set:
                raise BenchError(f"budget b{budget} contains unexpected prompt id: {prompt_id}")
            category = row_category(row)
            if category != expected_category[prompt_id]:
                raise BenchError(
                    f"budget b{budget} category mismatch for {prompt_id}: {category!r} != {expected_category[prompt_id]!r}"
                )
        duplicates = sorted(prompt_id for prompt_id, count in seen.items() if count > 1)
        if duplicates:
            raise BenchError(f"budget b{budget} contains duplicate prompt rows: {duplicates}")
        missing = [prompt_id for prompt_id in expected_ids if prompt_id not in seen]
        if missing:
            raise BenchError(f"budget b{budget} missing prompt rows: {missing}")


def build_split_contract(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_ids = [str(row["id"]) for row in prompts]
    categories = sorted({str(row["category"]) for row in prompts})
    heldout_ids = [prompt_id for prompt_id in prompt_ids if prompt_id in DEFAULT_HELDOUT_PROMPT_IDS]
    train_ids = [prompt_id for prompt_id in prompt_ids if prompt_id not in DEFAULT_HELDOUT_PROMPT_IDS]
    prompt_by_id = {str(row["id"]): row for row in prompts}
    heldout_categories = sorted({str(prompt_by_id[prompt_id]["category"]) for prompt_id in heldout_ids})
    missing_default_heldouts = sorted(DEFAULT_HELDOUT_PROMPT_IDS.difference(prompt_ids))
    missing_default_full_ids = [prompt_id for prompt_id in DEFAULT_FULL_PROMPT_IDS if prompt_id not in prompt_ids]
    extra_vs_default_full_ids = [prompt_id for prompt_id in prompt_ids if prompt_id not in DEFAULT_FULL_PROMPT_IDS]
    return {
        "strategy": "fixed_category_heldout_v1",
        "purpose": "Detect train-only acceptance/speed gains before resuming MTP optimization.",
        "default_full_ids": list(DEFAULT_FULL_PROMPT_IDS),
        "default_heldout_ids": sorted(DEFAULT_HELDOUT_PROMPT_IDS),
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "full_ids": prompt_ids,
        "categories": categories,
        "heldout_categories": heldout_categories,
        "heldout_has_all_present_categories": set(heldout_categories) == set(categories),
        "full_suite_matches_default": tuple(prompt_ids) == DEFAULT_FULL_PROMPT_IDS,
        "missing_default_heldout_ids": missing_default_heldouts,
        "missing_default_full_ids": missing_default_full_ids,
        "extra_vs_default_full_ids": extra_vs_default_full_ids,
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


def true_ar_artifact_from_path(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise BenchError("true AR baseline artifact must be a JSON object")
    validate_true_ar_artifact_schema(artifact, label="true AR baseline artifact")
    if artifact.get("true_autoregressive_path") is not True:
        raise BenchError("true AR baseline artifact must set true_autoregressive_path=true")
    if artifact.get("same_prompt_suite") is not True:
        raise BenchError("true AR baseline artifact must set same_prompt_suite=true")
    if artifact.get("same_timing_protocol") is not True:
        raise BenchError("true AR baseline artifact must set same_timing_protocol=true")
    validate_repo_provenance(artifact, label="true AR baseline artifact")
    rows = artifact.get("prompt_metrics")
    if not isinstance(rows, list) or not rows:
        raise BenchError("true AR baseline artifact must contain non-empty prompt_metrics[]")
    return artifact


def true_ar_rows_from_artifact(path: Path) -> list[dict[str, Any]]:
    return true_ar_artifact_from_path(path)["prompt_metrics"]


def validate_true_ar_prompt_rows(*, artifact: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_by_id = {str(row["id"]): row for row in prompts}
    expected_hashes = {prompt_id: prompt_sha256(str(row["prompt"])) for prompt_id, row in prompt_by_id.items()}
    prompt_hashes = artifact.get("prompt_hashes")
    if not isinstance(prompt_hashes, dict):
        raise BenchError("true AR baseline artifact requires prompt_hashes metadata")
    normalized_hashes = {str(prompt_id): str(value) for prompt_id, value in prompt_hashes.items()}
    if normalized_hashes != expected_hashes:
        missing = sorted(set(expected_hashes) - set(normalized_hashes))
        extra = sorted(set(normalized_hashes) - set(expected_hashes))
        mismatched = sorted(
            prompt_id
            for prompt_id in set(expected_hashes).intersection(normalized_hashes)
            if normalized_hashes[prompt_id] != expected_hashes[prompt_id]
        )
        raise BenchError(
            "true AR prompt_hashes must exactly match selected prompt text; "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )

    expected_decode_tokens = artifact.get("decode_tokens")
    if not isinstance(expected_decode_tokens, int) or expected_decode_tokens <= 0:
        raise BenchError("true AR baseline artifact protocol metadata requires positive decode_tokens")
    expected_warmup_decode_tokens = artifact.get("warmup_decode_tokens")
    if not isinstance(expected_warmup_decode_tokens, int) or expected_warmup_decode_tokens < 0:
        raise BenchError("true AR baseline artifact protocol metadata requires non-negative warmup_decode_tokens")
    rows = artifact.get("prompt_metrics")
    if not isinstance(rows, list) or not rows:
        raise BenchError("true AR baseline artifact must contain non-empty prompt_metrics[]")
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
        row_hash = row.get("prompt_sha256")
        if row_hash is None:
            raise BenchError(f"true AR prompt_metrics row requires prompt_sha256 for {prompt_id}")
        if str(row_hash) != expected_hashes[prompt_id]:
            raise BenchError(f"true AR prompt hash mismatch for {prompt_id}: {row_hash!r} != {expected_hashes[prompt_id]!r}")
        if row.get("finite_final_logits") is not True:
            raise BenchError(f"true AR prompt_metrics row requires finite_final_logits=true for {prompt_id}")
        output_tokens = int(row.get("output_tokens") or 0)
        if output_tokens <= 0:
            raise BenchError(f"true AR row for {prompt_id} must have positive output_tokens")
        if output_tokens != expected_decode_tokens:
            raise BenchError(f"true AR row for {prompt_id} output_tokens must match artifact decode_tokens")
        warmup_decode_tokens = row.get("warmup_decode_tokens")
        if not isinstance(warmup_decode_tokens, int) or warmup_decode_tokens != expected_warmup_decode_tokens:
            raise BenchError(f"true AR row for {prompt_id} warmup_decode_tokens must match artifact warmup_decode_tokens")
        decode_ms = finite_float(row.get("decode_ms"), prompt_id=prompt_id, field="true_ar.prompt_metrics[].decode_ms")
        if decode_ms <= 0.0:
            raise BenchError(f"true AR row for {prompt_id} must have positive decode_ms")
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


def attach_true_ar_baseline(summary: dict[str, Any], *, rows_by_id: dict[str, dict[str, Any]], source: Path, artifact_schema: dict[str, Any], repo: dict[str, Any], commands: list[str], protocol: dict[str, Any]) -> dict[str, Any]:
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
        "artifact_schema": artifact_schema["schema"],
        "artifact_kind": artifact_schema["kind"],
        "repo": repo,
        "commands": commands,
        "protocol": protocol,
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

    return validate_speed_claim_contract(populate_objective_metrics(summary))


def validate_speed_claim_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Ensure speed-promotable artifacts contain a real AR baseline.

    Verifier-derived ``off`` rows are useful economics telemetry, but they are
    not an independently measured no-MTP autoregressive path. Keep this as a
    machine-checkable invariant so future harness edits cannot accidentally flip
    ``speed_claim_eligible`` back on without adding the true baseline evidence.
    """
    if not summary.get("speed_claim_eligible", False):
        return summary
    validate_category_summary_schema(summary, label="speed-claim summary")
    validate_repo_provenance(summary, label="speed-claim summary")
    validate_command_provenance(summary, label="speed-claim summary")
    validate_summary_prompt_metadata(summary, label="speed-claim summary")
    true_ar = summary.get("true_ar_baseline")
    if not isinstance(true_ar, dict):
        raise BenchError("speed_claim_eligible=true requires true_ar_baseline metadata")
    if true_ar.get("true_autoregressive_path") is not True:
        raise BenchError("speed_claim_eligible=true requires a true no-MTP autoregressive baseline")
    if true_ar.get("same_prompt_suite") is not True:
        raise BenchError("speed_claim_eligible=true requires true AR measured on the same prompt suite")
    if true_ar.get("same_timing_protocol") is not True:
        raise BenchError("speed_claim_eligible=true requires true AR measured with the same timing protocol")
    validate_attached_true_ar_artifact_schema(true_ar, label="speed-claim true_ar_baseline")
    validate_repo_provenance(true_ar, label="speed-claim true_ar_baseline")
    validate_command_provenance(true_ar, label="speed-claim true_ar_baseline")
    validate_attached_true_ar_protocol(true_ar, label="speed-claim true_ar_baseline")
    return summary


def objective_metrics_for_budget(summary: dict[str, Any], budget_label: str | int) -> dict[str, Any]:
    """Extract honest optimization metrics for one MTP budget.

    Future optimization loops should consume this compact view instead of ad-hoc
    JSON paths. It deliberately requires true-AR comparison metadata and heldout
    coverage, so verifier-derived ``off`` / ``B0`` rows cannot become the speed
    denominator by accident.
    """
    label = f"b{budget_label}" if isinstance(budget_label, int) or str(budget_label).isdigit() else str(budget_label)
    if label == "off" or not label.startswith("b"):
        raise BenchError(f"objective budget must be an MTP budget label like b5, got {budget_label!r}")
    summary_artifact = validate_category_summary_schema(summary, label="objective summary")
    validate_repo_provenance(summary, label="objective summary")
    summary_prompts = validate_summary_prompt_metadata(summary, label="objective summary")
    if summary.get("true_ar_comparison_available") is not True:
        raise BenchError("objective metrics require true_ar_comparison_available=true")
    true_ar = summary.get("true_ar_baseline")
    if not isinstance(true_ar, dict) or true_ar.get("available") is not True:
        raise BenchError("objective metrics require an attached true_ar_baseline")
    true_ar_artifact = validate_attached_true_ar_artifact_schema(true_ar, label="attached true_ar_baseline")
    validate_repo_provenance(true_ar, label="attached true_ar_baseline")
    summary_commands = validate_command_provenance(summary, label="objective summary")
    true_ar_commands = validate_command_provenance(true_ar, label="attached true_ar_baseline")
    true_ar_protocol = validate_attached_true_ar_protocol(true_ar, label="attached true_ar_baseline")
    summary_categories = validate_summary_category_budget_metrics(
        summary,
        label="objective summary",
        budget_label=label,
        category_counts=summary_prompts["category_counts"],
    )
    splits = summary.get("splits")
    if not isinstance(splits, dict):
        raise BenchError("objective metrics require splits")
    contract = splits.get("contract")
    if not isinstance(contract, dict) or contract.get("heldout_has_all_present_categories") is not True:
        raise BenchError("objective metrics require heldout coverage for all present categories")
    if contract.get("full_suite_matches_default") is not True:
        raise BenchError("objective metrics require the full default mtp-bench category prompt suite")
    expected_full_ids = list(DEFAULT_FULL_PROMPT_IDS)
    actual_full_ids = [str(prompt_id) for prompt_id in contract.get("full_ids") or []]
    if actual_full_ids != expected_full_ids:
        raise BenchError("objective metrics require splits.contract.full_ids to match the default full prompt order")
    expected_heldout_ids = [prompt_id for prompt_id in expected_full_ids if prompt_id in DEFAULT_HELDOUT_PROMPT_IDS]
    actual_heldout_ids = [str(prompt_id) for prompt_id in contract.get("heldout_ids") or []]
    if actual_heldout_ids != expected_heldout_ids:
        raise BenchError("objective metrics require the fixed default heldout prompt IDs")
    expected_train_ids = [prompt_id for prompt_id in expected_full_ids if prompt_id not in DEFAULT_HELDOUT_PROMPT_IDS]
    actual_train_ids = [str(prompt_id) for prompt_id in contract.get("train_ids") or []]
    if actual_train_ids != expected_train_ids:
        raise BenchError("objective metrics require train prompt IDs to be the default full-minus-heldout complement")

    out: dict[str, Any] = {
        "budget": label,
        "true_ar_comparison_available": True,
        "speed_claim_eligible": bool(summary.get("speed_claim_eligible", False)),
        "performance_claim": bool(summary.get("performance_claim", False)),
        "summary_artifact": summary_artifact,
        "true_ar_artifact": true_ar_artifact,
        "summary_repo": validate_repo_provenance(summary, label="objective summary"),
        "true_ar_repo": validate_repo_provenance(true_ar, label="attached true_ar_baseline"),
        "summary_prompts": summary_prompts,
        "summary_categories": summary_categories,
        "summary_commands": summary_commands,
        "true_ar_commands": true_ar_commands,
        "true_ar_protocol": true_ar_protocol,
        "heldout_ids": list(contract.get("heldout_ids", [])),
        "train_ids": list(contract.get("train_ids", [])),
    }
    true_ar_splits = true_ar.get("splits")
    if not isinstance(true_ar_splits, dict):
        raise BenchError("objective metrics require attached true_ar_baseline.splits")
    split_contract_keys = {"full": "full_ids", "train": "train_ids", "heldout": "heldout_ids"}
    for split_name in ("full", "train", "heldout"):
        split = splits.get(split_name)
        if not isinstance(split, dict):
            raise BenchError(f"objective metrics require splits.{split_name}")
        contract_key = split_contract_keys[split_name]
        expected_split_ids = contract.get(contract_key)
        if not isinstance(expected_split_ids, list) or not expected_split_ids:
            raise BenchError(f"objective metrics require splits.contract.{contract_key}")
        split_prompt_ids = split.get("prompt_ids")
        if not isinstance(split_prompt_ids, list) or [str(prompt_id) for prompt_id in split_prompt_ids] != [str(prompt_id) for prompt_id in expected_split_ids]:
            raise BenchError(f"objective metrics require splits.{split_name}.prompt_ids to match splits.contract.{contract_key}")
        split_prompt_count = len(split_prompt_ids)
        metrics = split.get("metrics")
        if not isinstance(metrics, dict) or label not in metrics:
            raise BenchError(f"objective metrics missing {label} for split {split_name}")
        row = metrics[label]
        required = (
            "accepted_per_output",
            "draft_acceptance",
            "decode_tok_s_weighted",
            "mtp_vs_true_ar_decode_ratio",
            "prompts",
        )
        missing = [field for field in required if field not in row or row[field] is None]
        if missing:
            raise BenchError(f"objective metrics missing {missing} for {split_name}.{label}")
        try:
            prompts_count = int(row["prompts"])
        except (TypeError, ValueError) as exc:
            raise BenchError(f"objective metrics require positive {split_name}.{label}.prompts") from exc
        if prompts_count <= 0:
            raise BenchError(f"objective metrics require positive {split_name}.{label}.prompts")
        if prompts_count != split_prompt_count:
            raise BenchError(f"objective metrics require {split_name}.{label}.prompts to match splits.{split_name}.prompt_ids length")
        true_ar_split = true_ar_splits.get(split_name)
        if not isinstance(true_ar_split, dict):
            raise BenchError(f"objective metrics require attached true_ar_baseline.splits.{split_name}")
        try:
            true_ar_prompts_count = int(true_ar_split.get("prompts"))
        except (TypeError, ValueError) as exc:
            raise BenchError(f"objective metrics require positive attached true_ar_baseline.splits.{split_name}.prompts") from exc
        if true_ar_prompts_count <= 0:
            raise BenchError(f"objective metrics require positive attached true_ar_baseline.splits.{split_name}.prompts")
        if true_ar_prompts_count != split_prompt_count:
            raise BenchError(f"objective metrics require attached true_ar_baseline.splits.{split_name}.prompts to match splits.{split_name}.prompt_ids length")
        if true_ar_prompts_count != prompts_count:
            raise BenchError(
                f"objective metrics require {split_name}.{label}.prompts to match attached true_ar_baseline.splits.{split_name}.prompts"
            )
        decode_tok_s = finite_nonnegative_objective(row["decode_tok_s_weighted"], label=f"{split_name}.{label}.decode_tok_s_weighted")
        ratio = finite_nonnegative_objective(row["mtp_vs_true_ar_decode_ratio"], label=f"{split_name}.{label}.mtp_vs_true_ar_decode_ratio")
        true_ar_tok_s = finite_positive_objective(
            true_ar_split.get("decode_tok_s_weighted"),
            label=f"attached true_ar_baseline.splits.{split_name}.decode_tok_s_weighted",
        )
        expected_ratio = decode_tok_s / true_ar_tok_s
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12):
            raise BenchError(
                f"objective metrics require {split_name}.{label}.mtp_vs_true_ar_decode_ratio to match attached true_ar_baseline.splits.{split_name}"
            )
        out[split_name] = {
            "accepted_per_output": finite_unit_interval_objective(row["accepted_per_output"], label=f"{split_name}.{label}.accepted_per_output"),
            "draft_acceptance": finite_unit_interval_objective(row["draft_acceptance"], label=f"{split_name}.{label}.draft_acceptance"),
            "decode_tok_s_weighted": decode_tok_s,
            "mtp_vs_true_ar_decode_ratio": ratio,
            "prompts": prompts_count,
        }
    return out


def summary_protocol_signature(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": summary.get("schema"),
        "kind": summary.get("kind"),
        "model": summary.get("model"),
        "quant": summary.get("quant"),
        "prompt_file": summary.get("prompt_file"),
        "cycles": summary.get("cycles"),
        "prompts": summary.get("prompts"),
        "default_full_ids": (summary.get("splits") or {}).get("contract", {}).get("default_full_ids"),
        "heldout_ids": (summary.get("splits") or {}).get("contract", {}).get("heldout_ids"),
        "train_ids": (summary.get("splits") or {}).get("contract", {}).get("train_ids"),
    }


def true_ar_baseline_signature(summary: dict[str, Any]) -> dict[str, Any]:
    true_ar = summary.get("true_ar_baseline")
    if not isinstance(true_ar, dict) or true_ar.get("available") is not True:
        raise BenchError("objective comparison requires attached true_ar_baseline metadata")
    return {
        "source": true_ar.get("source"),
        "artifact": validate_attached_true_ar_artifact_schema(true_ar, label="attached true_ar_baseline"),
        "repo": validate_repo_provenance(true_ar, label="attached true_ar_baseline"),
        "commands": validate_command_provenance(true_ar, label="attached true_ar_baseline"),
        "protocol": validate_attached_true_ar_protocol(true_ar, label="attached true_ar_baseline"),
        "prompt_count": true_ar.get("prompt_count"),
        "totals": true_ar.get("totals"),
        "categories": true_ar.get("categories"),
        "splits": true_ar.get("splits"),
    }


def compare_objective_metrics(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    budget_label: str | int,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare candidate objective metrics against a baseline.

    The returned `passed` flag is intentionally conservative for future
    keep/revert loops: full-suite and heldout acceptance plus true-AR speed ratio
    must not regress. Train deltas are reported, but train-only gains cannot pass
    if full or heldout regress.
    """
    if tolerance < 0.0:
        raise BenchError("objective comparison tolerance must be non-negative")
    baseline = objective_metrics_for_budget(baseline_summary, budget_label)
    candidate = objective_metrics_for_budget(candidate_summary, budget_label)
    if baseline["heldout_ids"] != candidate["heldout_ids"]:
        raise BenchError("objective comparison requires identical heldout_ids")
    if baseline["train_ids"] != candidate["train_ids"]:
        raise BenchError("objective comparison requires identical train_ids")
    baseline_protocol = summary_protocol_signature(baseline_summary)
    candidate_protocol = summary_protocol_signature(candidate_summary)
    if baseline_protocol != candidate_protocol:
        raise BenchError("objective comparison requires identical benchmark protocol metadata")
    baseline_true_ar = true_ar_baseline_signature(baseline_summary)
    candidate_true_ar = true_ar_baseline_signature(candidate_summary)
    if baseline_true_ar != candidate_true_ar:
        raise BenchError("objective comparison requires identical attached true_ar_baseline metadata")

    fields = ("accepted_per_output", "draft_acceptance", "decode_tok_s_weighted", "mtp_vs_true_ar_decode_ratio")
    deltas: dict[str, dict[str, float]] = {}
    for split_name in ("full", "train", "heldout"):
        deltas[split_name] = {
            field: float(candidate[split_name][field]) - float(baseline[split_name][field])
            for field in fields
        }

    regressions: list[dict[str, Any]] = []
    gated_fields = ("accepted_per_output", "draft_acceptance", "mtp_vs_true_ar_decode_ratio")
    for split_name in ("full", "heldout"):
        for field in gated_fields:
            delta = deltas[split_name][field]
            if delta < -tolerance:
                regressions.append(
                    {
                        "split": split_name,
                        "field": field,
                        "baseline": float(baseline[split_name][field]),
                        "candidate": float(candidate[split_name][field]),
                        "delta": delta,
                    }
                )

    return {
        "budget": baseline["budget"],
        "passed": not regressions,
        "regressions": regressions,
        "tolerance": tolerance,
        "benchmark_protocol": baseline_protocol,
        "true_ar_baseline": baseline_true_ar,
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "decision_rule": (
            "full and heldout accepted_per_output, draft_acceptance, and "
            "mtp_vs_true_ar_decode_ratio must not regress; train deltas are report-only"
        ),
    }


def populate_objective_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Populate top-level objective metrics when all guardrails are satisfied."""
    summary["objective_metrics_available"] = False
    summary["objective_metrics_blocker"] = None
    summary["objectives"] = {}
    labels = [label for label in summary.get("totals", {}) if str(label).startswith("b")]
    for label in labels:
        try:
            summary["objectives"][label] = objective_metrics_for_budget(summary, label)
        except BenchError as exc:
            summary["objectives"] = {}
            summary["objective_metrics_blocker"] = str(exc)
            return summary
    summary["objective_metrics_available"] = bool(labels)
    if not labels:
        summary["objective_metrics_blocker"] = "no MTP budget rows available"
    return summary


def build_summary(*, args: argparse.Namespace, prompts: list[dict[str, Any]], raw: dict[int, list[dict[str, Any]]], commands: list[str]) -> dict[str, Any]:
    validate_raw_prompt_coverage(prompts=prompts, raw=raw)
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
        "schema": MTP_CATEGORY_SCHEMA,
        "kind": MTP_CATEGORY_KIND,
        "status": "diagnostic_retained",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "true_ar_comparison_available": False,
        "objective_metrics_available": False,
        "objective_metrics_blocker": "true AR comparison is not attached",
        "objectives": {},
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
        "repo": repo_provenance(),
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
            {"id": row["id"], "category": row["category"], "prompt_chars": len(row["prompt"]), "prompt_sha256": prompt_sha256(row["prompt"])}
            for row in prompts
        ],
    }
    if true_ar_baseline_json:
        true_ar_path = Path(true_ar_baseline_json)
        true_ar_artifact = true_ar_artifact_from_path(true_ar_path)
        true_ar_schema = validate_true_ar_artifact_schema(true_ar_artifact, label="true AR baseline artifact")
        true_ar_repo = validate_repo_provenance(true_ar_artifact, label="true AR baseline artifact")
        true_ar_commands = validate_command_provenance(true_ar_artifact, label="true AR baseline artifact")
        true_ar_protocol = validate_true_ar_protocol_metadata(artifact=true_ar_artifact, args=args, prompt_count=len(prompts), expected_quant=str(summary["quant"]))
        rows_by_id = validate_true_ar_prompt_rows(artifact=true_ar_artifact, prompts=prompts)
        return attach_true_ar_baseline(summary, rows_by_id=rows_by_id, source=true_ar_path, artifact_schema=true_ar_schema, repo=true_ar_repo, commands=true_ar_commands, protocol=true_ar_protocol)
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
            "When valid, the summary includes mtp_vs_true_ar_decode_ratio for diagnostics."
        ),
    )
    parser.add_argument(
        "--objective-summary-json",
        type=Path,
        default=None,
        help="Read an existing category summary JSON and print guarded objective metrics instead of running benchmarks.",
    )
    parser.add_argument(
        "--objective-budget",
        default=None,
        help="Budget label for --objective-summary-json, e.g. b5 or 5.",
    )
    parser.add_argument(
        "--objective-split",
        choices=("full", "train", "heldout"),
        default=None,
        help="With --objective-summary-json, print only one numeric metric from this split instead of JSON.",
    )
    parser.add_argument(
        "--objective-field",
        choices=("accepted_per_output", "draft_acceptance", "decode_tok_s_weighted", "mtp_vs_true_ar_decode_ratio"),
        default=None,
        help="With --objective-split, print this numeric objective field as a scalar.",
    )
    parser.add_argument(
        "--compare-baseline-summary-json",
        type=Path,
        default=None,
        help="Read an existing baseline category summary JSON for guarded objective comparison.",
    )
    parser.add_argument(
        "--compare-candidate-summary-json",
        type=Path,
        default=None,
        help="Read an existing candidate category summary JSON for guarded objective comparison.",
    )
    parser.add_argument(
        "--compare-budget",
        default=None,
        help="Budget label for compare mode, e.g. b5 or 5.",
    )
    parser.add_argument(
        "--compare-require-pass",
        action="store_true",
        help="In compare mode, exit non-zero when the guarded comparison does not pass.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.compare_baseline_summary_json is not None or args.compare_candidate_summary_json is not None:
        if args.compare_baseline_summary_json is None or args.compare_candidate_summary_json is None or args.compare_budget is None:
            raise BenchError("compare mode requires --compare-baseline-summary-json, --compare-candidate-summary-json, and --compare-budget")
        baseline = json.loads(args.compare_baseline_summary_json.read_text(encoding="utf-8"))
        candidate = json.loads(args.compare_candidate_summary_json.read_text(encoding="utf-8"))
        comparison = compare_objective_metrics(baseline, candidate, args.compare_budget)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0 if comparison["passed"] or not args.compare_require_pass else 2

    if args.objective_summary_json is not None:
        if args.objective_budget is None:
            raise BenchError("--objective-summary-json requires --objective-budget")
        if (args.objective_split is None) != (args.objective_field is None):
            raise BenchError("--objective-split and --objective-field must be provided together")
        summary = json.loads(args.objective_summary_json.read_text(encoding="utf-8"))
        metrics = objective_metrics_for_budget(summary, args.objective_budget)
        if args.objective_split is not None and args.objective_field is not None:
            print(metrics[args.objective_split][args.objective_field])
        else:
            print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0

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
