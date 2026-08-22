#!/usr/bin/env python3
"""Run the committed RF1 long-context task packet through eager GGUF MTP.

Each task is expanded to an exact token length, generated once by true AR and
once by host-proposal/eager-native MTP, and scored against its declared choice.
The gate binds task correctness, AR-ID equality, eager/split-K ownership, direct
cycle records, and hipEngine-owned allocation high-water. It is correctness
evidence, not a performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.gguf import scan_gguf
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    borrow_qwen35_gguf_nextn_fallback_weights,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_long_context_gate import _atomic_write_json, _provenance
from scripts.qwen35_paro_kv_quality_smoke import _build_prompt_tokens

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/mtp-realworld-long-context.jsonl"
CATEGORIES = (
    "retrieval",
    "multihop",
    "aggregation",
    "long_doc",
    "code",
    "mixed_ja_en",
)
CHOICES = ("A", "B", "C", "D")


class TaskSuiteError(ValueError):
    """Raised when the committed RF1 task fixture is malformed."""


def load_tasks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskSuiteError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise TaskSuiteError(f"{path}:{line_number}: row must be an object")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id.strip() or task_id in seen:
            raise TaskSuiteError(f"{path}:{line_number}: task id must be unique non-empty text")
        seen.add(task_id)
        if row.get("category") not in CATEGORIES:
            raise TaskSuiteError(f"{path}:{line_number}: unsupported category")
        for field in ("prefix", "filler", "suffix"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise TaskSuiteError(f"{path}:{line_number}: {field} must be non-empty text")
        target = row.get("target_context_tokens")
        if type(target) is not int or target <= 0:
            raise TaskSuiteError(f"{path}:{line_number}: target_context_tokens must be positive")
        choices = row.get("choices")
        if not isinstance(choices, dict) or tuple(choices) != CHOICES:
            raise TaskSuiteError(f"{path}:{line_number}: choices must be ordered A/B/C/D")
        expected_choice = row.get("expected_choice")
        if expected_choice not in CHOICES:
            raise TaskSuiteError(f"{path}:{line_number}: expected_choice must be A/B/C/D")
        if row.get("scorer") != "choice_exact":
            raise TaskSuiteError(f"{path}:{line_number}: scorer must be choice_exact")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise TaskSuiteError(f"{path}:{line_number}: evidence must be non-empty")
        previous = -1.0
        for item in evidence:
            if not isinstance(item, dict):
                raise TaskSuiteError(f"{path}:{line_number}: evidence entries must be objects")
            position = item.get("position")
            text = item.get("text")
            if not isinstance(position, (int, float)) or not 0.0 <= float(position) <= 1.0:
                raise TaskSuiteError(f"{path}:{line_number}: invalid evidence position")
            if float(position) < previous:
                raise TaskSuiteError(f"{path}:{line_number}: evidence positions must be sorted")
            if not isinstance(text, str) or not text:
                raise TaskSuiteError(f"{path}:{line_number}: evidence text must be non-empty")
            previous = float(position)
        rows.append(row)
    if not rows:
        raise TaskSuiteError(f"{path} contains no tasks")
    if tuple(dict.fromkeys(str(row["category"]) for row in rows)) != CATEGORIES:
        raise TaskSuiteError("task fixture must contain every RF1 category in canonical order")
    return rows


class _TokenizerAdapter:
    """Expose the tokenizers-style interface used by the shared prompt builder."""

    def __init__(self, tokenizer: Qwen35GGUFTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[int(token) for token in self.tokenizer.encode(text)])


def score_task_output(text: str, task: dict[str, Any]) -> dict[str, Any]:
    upper = text.upper()
    standalone = re.findall(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])", upper)
    selected = standalone[0] if standalone else None
    expected_choice = str(task["expected_choice"])
    expected_text = str(task["choices"][expected_choice])
    text_match = expected_text.casefold() in text.casefold()
    passed = selected == expected_choice or text_match
    return {
        "passed": bool(passed),
        "selected_choice": selected,
        "expected_choice": expected_choice,
        "expected_text": expected_text,
        "expected_text_present": bool(text_match),
    }


def _decode(tokenizer: Qwen35GGUFTokenizer, token_ids: Sequence[int]) -> str:
    return str(tokenizer.decode([int(token) for token in token_ids]))


def finalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the RF1 binding contract to captured task rows.

    RF1 binds MTP-vs-true-AR IDs and eager ownership. Absolute task correctness
    is retained separately and remains a production-quality input for RF6; it
    cannot turn an AR-identical MTP row into a functional regression.
    """

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TaskSuiteError("captured task payload requires non-empty rows")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("score"), dict):
            raise TaskSuiteError("captured task rows require score objects")
        binding_passed = bool(
            row.get("output_ids_exact")
            and row.get("gpu_accept_match_cpu")
            and row.get("all_cycles_eager")
        )
        row["task_score_passed"] = bool(row["score"].get("passed", False))
        row["binding_passed"] = binding_passed
        row["passed"] = binding_passed
    binding_passed = all(bool(row["binding_passed"]) for row in rows)
    task_correct = sum(bool(row["task_score_passed"]) for row in rows)
    summary = dict(payload.get("summary") or {})
    summary.update(
        {
            "passed": sum(bool(row["binding_passed"]) for row in rows),
            "binding_passed": sum(bool(row["binding_passed"]) for row in rows),
            "task_correct": task_correct,
            "total": len(rows),
            "absolute_task_quality_passed": task_correct == len(rows),
        }
    )
    payload["summary"] = summary
    payload["status"] = "passed" if binding_passed else "failed"
    payload["verdict"] = "pass" if binding_passed else "fail"
    payload["passed"] = binding_passed
    payload["production_quality_claim"] = False
    payload["task_quality_role"] = "diagnostic_rf1_non_regression_input_to_rf6"
    return payload


def _run_ar(target: Qwen35GGUFResidentSession, prompt: Sequence[int], count: int) -> list[int]:
    target.reset()
    first = target.prefill(prompt, use_bulk=True, return_logits=False)
    output = [int(first.token_id)]
    while len(output) < int(count):
        output.append(int(target.step(output[-1], return_logits=False).token_id))
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_tasks(args.suite)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    adapter = _TokenizerAdapter(tokenizer)
    expanded = {
        str(task["id"]): _build_prompt_tokens(
            adapter,
            task,
            context_tokens=int(args.context_tokens or task["target_context_tokens"]),
        )
        for task in tasks
    }
    maximum_prompt = max(len(tokens) for tokens, _metadata in expanded.values())
    required_sequence = maximum_prompt + int(args.max_new_tokens)
    max_sequence_length = int(args.max_sequence_length or required_sequence)
    if max_sequence_length < required_sequence:
        raise TaskSuiteError("--max-sequence-length does not cover prompt plus output")

    reset_memory_stats()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def checkpoint(event: str, details: dict[str, Any]) -> None:
        print(json.dumps({"event": event, **details}, sort_keys=True), file=sys.stderr, flush=True)
        if args.out is not None:
            _atomic_write_json(
                args.out,
                {
                    "schema": 1,
                    "kind": "gguf_mtp_long_context_task_checkpoint",
                    "status": "running",
                    "active_event": event,
                    "active_details": details,
                    "rows": rows,
                    "summary": {"completed": len(rows), "total": len(tasks)},
                },
            )

    checkpoint("run_start", {})
    with Qwen35GGUFResidentSession(
        args.model,
        max_sequence_length=max_sequence_length,
        require_cached_build=bool(args.require_cached_build),
    ) as target:
        target.select_prefill_quant("gguf_q4_k_m")
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            args.model,
            max_positions=max_sequence_length,
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=bool(args.require_cached_build),
            borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(target),
        )
        try:
            with Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=int(args.candidate_budget),
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            ) as decoder:
                for index, task in enumerate(tasks):
                    task_id = str(task["id"])
                    prompt, prompt_metadata = expanded[task_id]
                    checkpoint("task_start", {"id": task_id, "category": task["category"]})
                    task_started = time.perf_counter()
                    expected = _run_ar(target, prompt, int(args.max_new_tokens))
                    request_id = 50_000 + index
                    actual = decoder.generate(
                        prompt,
                        max_new_tokens=int(args.max_new_tokens),
                        request_id=request_id,
                        return_cycle_logits=True,
                        use_bulk_prefill=True,
                    )
                    provider.release_request(request_id)
                    cycles = tuple(actual.cycle_records)
                    output_text = _decode(tokenizer, actual.token_ids)
                    score = score_task_output(output_text, task)
                    row = {
                        "id": task_id,
                        "category": str(task["category"]),
                        **prompt_metadata,
                        "max_new_tokens": int(args.max_new_tokens),
                        "expected_choice": str(task["expected_choice"]),
                        "output_token_ids": list(actual.token_ids),
                        "output_token_ids_sha256": hashlib.sha256(
                            json.dumps(list(actual.token_ids), separators=(",", ":")).encode()
                        ).hexdigest(),
                        "output_text": output_text,
                        "output_ids_exact": tuple(actual.token_ids) == tuple(expected),
                        "gpu_accept_match_cpu": bool(actual.gpu_accept_match_cpu),
                        "all_cycles_eager": bool(cycles)
                        and all(
                            not bool(cycle["target_native_graph_submitted"])
                            and not bool(cycle["proposal_target_device_chained"])
                            for cycle in cycles
                        ),
                        "split_k_observed": bool(cycles)
                        and all(
                            cycle.get("target_native_graph_fallback_reason") is not None
                            for cycle in cycles
                        ),
                        "accepted_counts": [int(value) for value in actual.accepted_counts],
                        "cycle_count": len(cycles),
                        "score": score,
                        "wall_seconds": time.perf_counter() - task_started,
                    }
                    # Split-K execution itself is already mechanically bound by
                    # RF1 direct/generation artifacts. Here the long context and
                    # eager fallback reason bind task-route ownership. Absolute
                    # answer quality is retained separately for RF6.
                    row["task_score_passed"] = bool(score["passed"])
                    row["binding_passed"] = bool(
                        row["output_ids_exact"]
                        and row["gpu_accept_match_cpu"]
                        and row["all_cycles_eager"]
                    )
                    row["passed"] = bool(row["binding_passed"])
                    rows.append(row)
                    checkpoint(
                        "task_complete",
                        {
                            "id": task_id,
                            "binding_passed": row["binding_passed"],
                            "task_score_passed": row["task_score_passed"],
                        },
                    )
        finally:
            provider.close()

    payload = {
        "schema": 1,
        "kind": "gguf_mtp_long_context_task_gate",
        "status": "unfinalized",
        "verdict": None,
        "performance_claim": False,
        "profile_contract": "strict_ar_id_and_task_choice",
        "model_quant": "gguf_q4_k_m",
        "kv_storage": "bf16",
        "command": [sys.executable, *sys.argv],
        "provenance": _provenance(args.model, hash_model=bool(args.hash_model)),
        "configuration": {
            "suite": str(args.suite),
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "context_tokens": int(args.context_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "candidate_budget": int(args.candidate_budget),
            "max_sequence_length": max_sequence_length,
            "require_cached_build": bool(args.require_cached_build),
        },
        "rows": rows,
        "memory": memory_stats(),
        "summary": {
            "total": len(rows),
            "categories": list(CATEGORIES),
            "wall_seconds": time.perf_counter() - started,
        },
        "passed": False,
    }
    payload = finalize_payload(payload)
    passed = bool(payload["passed"])
    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.out, payload)
        print(f"wrote {args.out}: passed={passed} tasks={len(rows)}", flush=True)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--finalize-existing",
        type=Path,
        help="Finalize a captured task artifact under the RF1 binding contract without GPU work.",
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--candidate-budget", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=65544)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_existing is not None:
        source = args.finalize_existing
        payload = json.loads(source.read_text(encoding="utf-8"))
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        payload = finalize_payload(payload)
        payload["source_artifact"] = str(source)
        payload["source_artifact_sha256"] = source_sha256
        payload["finalization_command"] = [sys.executable, *sys.argv]
        destination = args.out or source
        _atomic_write_json(destination, payload)
        print(f"wrote {destination}: passed={payload['passed']} tasks={len(payload['rows'])}")
        return 1 if args.fail_on_fail and not payload["passed"] else 0
    if not args.model.is_file() or not args.suite.is_file():
        raise SystemExit("model and suite must exist")
    if args.context_tokens <= 0 or args.max_new_tokens < 2:
        raise SystemExit("context tokens must be positive and max-new-tokens must be >=2")
    if args.candidate_budget not in {1, 2, 3}:
        raise SystemExit("candidate budget must be 1, 2, or 3")
    payload = run(args)
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
