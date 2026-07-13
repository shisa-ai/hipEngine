#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a bounded long-context task smoke for PARO BF16 versus candidate KV.

The suite is deliberately small: one deterministic retrieval, multihop,
aggregation, long-document, and code task by default. It is a regression smoke,
not a replacement for a full long-context benchmark. Prompts are expanded to an
exact token length with neutral filler while authoritative evidence remains at
fixed relative positions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import token_ids_sha256
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kvcache import ResolvedKVPolicy, resolve_kv_policy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_SUITE = REPO_ROOT / "benchmarks" / "prompts" / "kv-int8-long-context-smoke.jsonl"
CATEGORIES = ("retrieval", "multihop", "aggregation", "long_doc", "code")


class SuiteError(ValueError):
    """Raised for an invalid bounded quality suite."""


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


def _load_suite(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SuiteError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SuiteError(f"{path}:{line_number}: row must be an object")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise SuiteError(f"{path}:{line_number}: id must be non-empty text")
        if task_id in seen:
            raise SuiteError(f"{path}:{line_number}: duplicate id {task_id!r}")
        seen.add(task_id)
        category = row.get("category")
        if category not in CATEGORIES:
            raise SuiteError(f"{path}:{line_number}: category must be one of {', '.join(CATEGORIES)}")
        for field in ("prefix", "filler", "suffix"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise SuiteError(f"{path}:{line_number}: {field} must be non-empty text")
        target = row.get("target_context_tokens")
        if type(target) is not int or target <= 0:
            raise SuiteError(f"{path}:{line_number}: target_context_tokens must be positive")
        expected = row.get("expected")
        if not isinstance(expected, list) or not expected or any(not isinstance(item, str) or not item for item in expected):
            raise SuiteError(f"{path}:{line_number}: expected must contain non-empty strings")
        if row.get("scorer") != "final_exact":
            raise SuiteError(f"{path}:{line_number}: only scorer='final_exact' is supported")
        if row.get("prompt_format", "qwen_chat") not in {"qwen_chat", "raw"}:
            raise SuiteError(f"{path}:{line_number}: prompt_format must be qwen_chat or raw")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise SuiteError(f"{path}:{line_number}: evidence must be a non-empty list")
        last_position = -1.0
        for item in evidence:
            if not isinstance(item, dict):
                raise SuiteError(f"{path}:{line_number}: evidence entries must be objects")
            position = item.get("position")
            text = item.get("text")
            if not isinstance(position, (int, float)) or not 0.0 <= float(position) <= 1.0:
                raise SuiteError(f"{path}:{line_number}: evidence position must be in [0,1]")
            if float(position) < last_position:
                raise SuiteError(f"{path}:{line_number}: evidence positions must be sorted")
            if not isinstance(text, str) or not text:
                raise SuiteError(f"{path}:{line_number}: evidence text must be non-empty")
            last_position = float(position)
        rows.append(row)
    if not rows:
        raise SuiteError(f"{path} contains no tasks")
    return rows


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(item) for item in tokenizer.encode(text).ids]


def _build_prompt_tokens(
    tokenizer: Any,
    task: dict[str, Any],
    *,
    context_tokens: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    target = int(context_tokens or task["target_context_tokens"])
    prompt_format = str(task.get("prompt_format", "qwen_chat"))
    prefix_text = str(task["prefix"])
    suffix_text = str(task["suffix"])
    if prompt_format == "qwen_chat":
        prefix_text = "<|im_start|>user\n" + prefix_text
        suffix_text = suffix_text + "<|im_end|>\n<|im_start|>assistant\n"
    elif prompt_format != "raw":
        raise SuiteError(f"task {task['id']!r} has unsupported prompt_format {prompt_format!r}")
    prefix = _encode(tokenizer, prefix_text)
    suffix = _encode(tokenizer, suffix_text)
    filler_unit = _encode(tokenizer, str(task["filler"]))
    evidence = [
        (float(item["position"]), _encode(tokenizer, str(item["text"])))
        for item in task["evidence"]
    ]
    if not filler_unit:
        raise SuiteError(f"task {task['id']!r} filler tokenized to an empty sequence")
    fixed_tokens = len(prefix) + len(suffix) + sum(len(ids) for _, ids in evidence)
    filler_budget = target - fixed_tokens
    if filler_budget < 0:
        raise SuiteError(
            f"task {task['id']!r} fixed prompt requires {fixed_tokens} tokens, above target {target}"
        )
    repeats = (filler_budget + len(filler_unit) - 1) // len(filler_unit)
    filler = (filler_unit * repeats)[:filler_budget]
    prompt = list(prefix)
    evidence_offsets: list[dict[str, int | float]] = []
    cursor = 0
    for relative_position, evidence_ids in evidence:
        insert_at = min(filler_budget, max(cursor, int(round(relative_position * filler_budget))))
        prompt.extend(filler[cursor:insert_at])
        absolute_offset = len(prompt)
        prompt.extend(evidence_ids)
        evidence_offsets.append(
            {
                "relative_position": relative_position,
                "token_offset": absolute_offset,
                "token_count": len(evidence_ids),
            }
        )
        cursor = insert_at
    prompt.extend(filler[cursor:])
    prompt.extend(suffix)
    if len(prompt) != target:
        raise AssertionError(f"task {task['id']!r} expanded to {len(prompt)} tokens, expected {target}")
    return prompt, {
        "context_tokens": target,
        "prompt_format": prompt_format,
        "fixed_tokens": fixed_tokens,
        "filler_tokens": filler_budget,
        "evidence_offsets": evidence_offsets,
        "prompt_token_ids_sha256": token_ids_sha256(prompt),
    }


def _normalize_answer(text: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z-]+", " ", text.casefold()).strip()
    return re.sub(r"\s*-\s*", "-", normalized)


def _score_output(output: str, expected: Sequence[str]) -> dict[str, Any]:
    matches = re.findall(r"FINAL\s*:\s*([^\r\n<]+)", output, flags=re.IGNORECASE)
    extracted = matches[-1].strip(" `\t.\"'") if matches else None
    normalized = None if extracted is None else _normalize_answer(extracted)
    normalized_expected = [_normalize_answer(item) for item in expected]
    passed = normalized in normalized_expected if normalized is not None else False
    return {
        "passed": bool(passed),
        "extracted_answer": extracted,
        "normalized_answer": normalized,
        "normalized_expected": normalized_expected,
        "expected_mentioned_anywhere": any(item in _normalize_answer(output) for item in normalized_expected),
    }


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def _run_task(
    session: Qwen35ParoResidentSession,
    *,
    tokenizer: Any,
    task: dict[str, Any],
    prompt_tokens: Sequence[int],
    prompt_metadata: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    session.reset()
    session._resolve_prefill_config_for_length(len(prompt_tokens))
    started = time.perf_counter()
    seed = session.prefill_native(list(prompt_tokens), sample=True)
    if seed is None:
        raise RuntimeError(f"task {task['id']!r} prefill did not produce a seed token")
    output_ids = [int(seed.token_id)]
    current = seed
    for offset in range(max_new_tokens - 1):
        current = session.step(
            int(current.token_id),
            position=len(prompt_tokens) + offset,
            sample=True,
        )
        if current is None:
            raise RuntimeError(f"task {task['id']!r} decode stopped at token {offset}")
        output_ids.append(int(current.token_id))
    elapsed = time.perf_counter() - started
    output = _decode(tokenizer, output_ids)
    return {
        "id": str(task["id"]),
        "category": str(task["category"]),
        **prompt_metadata,
        "max_new_tokens": int(max_new_tokens),
        "output_token_ids": output_ids,
        "output_token_ids_sha256": token_ids_sha256(output_ids),
        "output_text": output,
        "output_text_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "elapsed_seconds": float(elapsed),
        "score": _score_output(output, [str(item) for item in task["expected"]]),
    }


def _run_policy(
    *,
    runner: Qwen35ParoNextTokenRunner,
    tokenizer: Any,
    policy: ResolvedKVPolicy,
    tasks: Sequence[dict[str, Any]],
    expanded_prompts: dict[str, tuple[list[int], dict[str, Any]]],
    max_sequence_length: int,
    max_new_tokens: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> dict[str, Any]:
    reset_memory_stats()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512, auto_tune_chunk_sizes=True),
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        for task in tasks:
            prompt_tokens, prompt_metadata = expanded_prompts[str(task["id"])]
            rows.append(
                _run_task(
                    session,
                    tokenizer=tokenizer,
                    task=task,
                    prompt_tokens=prompt_tokens,
                    prompt_metadata=prompt_metadata,
                    max_new_tokens=max_new_tokens,
                )
            )
        owned_summary = session.owned_buffer_summary()
    elapsed = time.perf_counter() - started
    gc.collect()
    return {
        "kv_policy": kv_policy_json(policy),
        "rows": rows,
        "summary": {
            "passed": sum(bool(row["score"]["passed"]) for row in rows),
            "total": len(rows),
            "score": float(sum(bool(row["score"]["passed"]) for row in rows) / len(rows)),
            "elapsed_seconds": float(elapsed),
        },
        "memory": memory_stats(),
        "owned_buffer_summary": {
            key: owned_summary.get(key)
            for key in (
                "allocation_bytes",
                "buffer_bytes",
                "full_attention_kv_payload_bytes",
                "full_attention_kv_scale_bytes",
                "full_attention_kv_total_bytes",
                "kv_storage_dtype",
                "kv_scale_dtype",
                "kv_scale_granularity",
            )
        },
    }


def _select_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    categories: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    requested = {item.strip() for item in categories.split(",") if item.strip()}
    unknown = requested - set(CATEGORIES)
    if unknown:
        raise SuiteError(f"unknown categories: {', '.join(sorted(unknown))}")
    selected = [task for task in tasks if not requested or task["category"] in requested]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SuiteError("task selection is empty")
    return selected


def _pair_results(
    tasks: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    reference_by_id = {str(row["id"]): row for row in reference_rows}
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    expected_ids = [str(task["id"]) for task in tasks]
    if set(reference_by_id) != set(expected_ids) or set(candidate_by_id) != set(expected_ids):
        raise ValueError("reference/candidate task IDs must exactly match the selected suite")
    paired: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        reference_passed = bool(reference_by_id[task_id]["score"]["passed"])
        candidate_passed = bool(candidate_by_id[task_id]["score"]["passed"])
        paired.append(
            {
                "id": task_id,
                "category": str(task["category"]),
                "reference_passed": reference_passed,
                "candidate_passed": candidate_passed,
                "candidate_regression": bool(reference_passed and not candidate_passed),
                "candidate_improvement": bool(candidate_passed and not reference_passed),
                "output_token_ids_match": (
                    reference_by_id[task_id]["output_token_ids"]
                    == candidate_by_id[task_id]["output_token_ids"]
                ),
            }
        )
    reference_failures = [row["id"] for row in paired if not row["reference_passed"]]
    candidate_regressions = [row["id"] for row in paired if row["candidate_regression"]]
    if reference_failures:
        status = "reference_unscorable"
    elif candidate_regressions:
        status = "candidate_quality_regression"
    else:
        status = "accepted_smoke"
    summary = {
        "reference_failures": reference_failures,
        "candidate_regressions": candidate_regressions,
        "candidate_improvements": [row["id"] for row in paired if row["candidate_improvement"]],
        "paired_non_regression": not candidate_regressions,
    }
    return paired, status, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _select_tasks(_load_suite(args.suite), categories=args.categories, limit=args.limit)
    candidate_policy = resolve_args_kv_policy(args, block_size=256)
    reference_policy = resolve_kv_policy("bf16", block_size=256)
    compiler_version = _read_compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(
        args.model,
        shared_expert_format=None if args.shared_expert_format == "auto" else args.shared_expert_format,
        backend=args.backend,
    )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    expanded = {
        str(task["id"]): _build_prompt_tokens(
            tokenizer,
            task,
            context_tokens=args.context_tokens or None,
        )
        for task in tasks
    }
    max_context = max(len(tokens) for tokens, _metadata in expanded.values())
    max_sequence_length = max_context + int(args.max_new_tokens) + 2
    reference = _run_policy(
        runner=runner,
        tokenizer=tokenizer,
        policy=reference_policy,
        tasks=tasks,
        expanded_prompts=expanded,
        max_sequence_length=max_sequence_length,
        max_new_tokens=args.max_new_tokens,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    candidate = _run_policy(
        runner=runner,
        tokenizer=tokenizer,
        policy=candidate_policy,
        tasks=tasks,
        expanded_prompts=expanded,
        max_sequence_length=max_sequence_length,
        max_new_tokens=args.max_new_tokens,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    paired, status, pair_summary = _pair_results(tasks, reference["rows"], candidate["rows"])
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=args.backend,
        resolved_backend=runner.backend,
        target_arch=runner.target_arch,
        model_path=args.model,
        quant="w4_paro",
        kv_dtype=f"bf16_vs_{candidate_policy.storage_dtype.value}",
        command=(sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]),
        environment={
            key: os.environ.get(key)
            for key in ("HIP_VISIBLE_DEVICES", "HIPENGINE_HIP_ARCH", "HIPENGINE_BACKEND")
        },
        build_profile="kv_int8_quality_smoke",
        timing_protocol="serial_task_generation_no_warmup",
        warmups=0,
        repetitions=1,
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": "qwen35_paro_kv_bounded_quality_smoke",
        "status": status,
        "performance_claim": False,
        "full_benchmark_claim": False,
        "suite": {
            "path": str(args.suite),
            "sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "selected_ids": [str(task["id"]) for task in tasks],
            "categories": [str(task["category"]) for task in tasks],
            "context_tokens_override": int(args.context_tokens) if args.context_tokens else None,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "model": str(args.model),
        "backend": runner.backend,
        "target_arch": runner.target_arch,
        "provenance": provenance,
        "reference": reference,
        "candidate": candidate,
        "paired": paired,
        "summary": pair_summary,
        "notes": [
            "This five-category suite is a bounded regression smoke, not a full long-context benchmark.",
            "Task-score retention is the product signal; exact output-token equality is diagnostic only.",
            "A reference failure makes the row unscorable for candidate regression rather than a candidate pass.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--context-tokens", type=int, default=0, help="Override every selected task context length")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument(
        "--shared-expert-format",
        choices=("auto", "legacy_fp16", "packed_paro_w4"),
        default="packed_paro_w4",
    )
    add_kv_policy_args(
        parser,
        default_storage="int8_per_token_head",
        help_prefix="Candidate KV storage for the bounded BF16-vs-candidate quality smoke",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.context_tokens < 0:
        parser.error("--context-tokens must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_layers < 0:
        parser.error("--max-layers must be non-negative")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["status"] == "accepted_smoke" else 1


if __name__ == "__main__":
    raise SystemExit(main())
