#!/usr/bin/env python3
"""Run one standalone Poolside Laguna DFlash proposal from live target taps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.speculative.laguna_dflash import LagunaDFlashResidentDrafter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Laguna S 2.1 target GGUF")
    parser.add_argument("drafter", type=Path, help="Poolside Laguna S 2.1 DFlash directory")
    parser.add_argument(
        "--template-fixture",
        type=Path,
        default=Path("tests/fixtures/laguna_poolside_v1_template.json"),
    )
    parser.add_argument("--prompt-case", default="oracle_no_thinking")
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("tests/fixtures/laguna_dflash_poolside_oracle.json"),
    )
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _prompt_ids(path: Path, case: str) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload["cases"]:
        if item["name"] == case:
            return tuple(int(value) for value in item["token_ids"])
    raise ValueError(f"prompt case {case!r} not found in {path}")


def main() -> int:
    args = parse_args()
    prompt_ids = _prompt_ids(args.template_fixture, args.prompt_case)
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    if oracle["prompt_case"] != args.prompt_case:
        raise ValueError(
            f"oracle prompt case {oracle['prompt_case']!r} does not match {args.prompt_case!r}"
        )
    compiler_version = (
        args.compiler_version_file.read_text(encoding="utf-8")
        if args.compiler_version_file is not None
        else None
    )
    reset_memory_stats()
    started = time.perf_counter()
    session_started = time.perf_counter()
    result: dict[str, object]
    with LagunaGGUFResidentSession(
        args.model,
        context_length=args.context_length,
        backend=args.backend,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        repacked_cache=args.repacked_cache,
        model_sha256=args.model_sha256,
        prefill_chunk_size=64,
    ) as target:
        target_loaded = time.perf_counter()
        with LagunaDFlashResidentDrafter(
            target,
            args.drafter,
            candidate_budget=args.candidate_budget,
            top_k=args.top_k,
            max_append_rows=1,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        ) as drafter:
            drafter_loaded = time.perf_counter()
            with drafter.allocate_captures(rows=1) as captures:
                target_result = None
                context_started = time.perf_counter()
                for position, token_id in enumerate(prompt_ids):
                    target_result = target.forward_token(
                        token_id,
                        captures=captures.targets,
                    )
                    drafter.append_target_hidden(captures, positions=(position,))
                context_ready = time.perf_counter()
                assert target_result is not None
                proposal_started = time.perf_counter()
                proposal = drafter.propose(
                    root_token_id=target_result.next_token_id,
                    root_position=len(prompt_ids),
                )
                proposal_done = time.perf_counter()
                expected_topk_rows = tuple(
                    tuple(int(value) for value in row)
                    for row in oracle["candidate_topk_token_ids"]
                )
                admitted_budget = int(oracle["admitted_candidate_budget"])
                compared_rows = min(args.candidate_budget, admitted_budget)
                actual_topk_rows = proposal.topk_token_ids[:compared_rows]
                expected_admitted_rows = expected_topk_rows[:compared_rows]
                rows_match = actual_topk_rows == expected_admitted_rows
                budget_admitted = args.candidate_budget <= admitted_budget
                oracle_match = bool(
                    int(target_result.next_token_id) == int(oracle["root_token_id"])
                    and rows_match
                    and budget_admitted
                )
                result = {
                    "schema": 1,
                    "passed": bool(
                        target_result.next_token_id >= 0
                        and len(proposal.candidate_token_ids) == args.candidate_budget
                        and oracle_match
                    ),
                    "model": str(args.model.resolve()),
                    "drafter": str(args.drafter.resolve()),
                    "backend": args.backend,
                    "prompt_case": args.prompt_case,
                    "prompt_tokens": len(prompt_ids),
                    "root_token_id": int(target_result.next_token_id),
                    "root_position": len(prompt_ids),
                    "candidate_budget": args.candidate_budget,
                    "top_k": args.top_k,
                    "candidate_token_ids": list(proposal.candidate_token_ids),
                    "candidate_values": list(proposal.candidate_values),
                    "topk_token_ids": [list(row) for row in proposal.topk_token_ids],
                    "topk_values": [list(row) for row in proposal.topk_values],
                    "oracle": {
                        "path": str(args.oracle.resolve()),
                        "expected_root_token_id": int(oracle["root_token_id"]),
                        "admitted_candidate_budget": admitted_budget,
                        "budget_admitted": budget_admitted,
                        "compared_rows": compared_rows,
                        "expected_topk_rows": [
                            list(row) for row in expected_admitted_rows
                        ],
                        "root_match": int(target_result.next_token_id)
                        == int(oracle["root_token_id"]),
                        "topk_rows_match": rows_match,
                        "passed": oracle_match,
                    },
                    "capture_depths": list(drafter.capture_depths),
                    "committed_context_tokens": drafter.committed_context_tokens,
                    "drafter_resident_nbytes": drafter.resident_nbytes,
                    "timing_seconds": {
                        "target_load": target_loaded - session_started,
                        "drafter_load": drafter_loaded - target_loaded,
                        "serial_target_and_draft_context": context_ready - context_started,
                        "proposal": proposal_done - proposal_started,
                        "total_inside_session": proposal_done - session_started,
                        "process": proposal_done - started,
                    },
                    "memory": memory_stats(),
                }
    after_close = memory_stats()
    result["memory_after_close"] = after_close
    result["lifecycle_passed"] = (
        after_close["current_allocated_bytes"] == 0
        and after_close["active_allocations"] == 0
    )
    result["passed"] = bool(result["passed"] and result["lifecycle_passed"])
    result["timing_seconds"]["close"] = time.perf_counter() - proposal_done  # type: ignore[index]
    result["timing_seconds"]["process_with_close"] = time.perf_counter() - started  # type: ignore[index]
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
