#!/usr/bin/env python3
"""Validate Laguna DFlash B+1 accept/commit against same-session target AR."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.speculative.laguna_dflash import (
    LagunaDFlashResidentCycle,
    LagunaDFlashResidentDrafter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("drafter", type=Path)
    parser.add_argument(
        "--template-fixture",
        type=Path,
        default=Path("tests/fixtures/laguna_poolside_v1_template.json"),
    )
    parser.add_argument("--prompt-case", default="oracle_no_thinking")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--budgets", default="1,2,4,7,15")
    parser.add_argument("--max-new-tokens", type=int, default=12)
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


def _parse_budgets(raw: str) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in raw.split(",") if value.strip())
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be a non-empty unique comma-separated list")
    if any(value <= 0 or value >= 16 for value in budgets):
        raise ValueError("Laguna DFlash budgets must be within [1, 15]")
    return budgets


def _serial_target_ar(
    target: LagunaGGUFResidentSession,
    prompt_ids: tuple[int, ...],
    *,
    max_new_tokens: int,
) -> tuple[int, ...]:
    result = None
    for token_id in prompt_ids:
        result = target.forward_token(token_id)
    assert result is not None
    generated: list[int] = []
    while len(generated) < max_new_tokens:
        token = int(result.next_token_id)
        generated.append(token)
        if len(generated) == max_new_tokens:
            break
        result = target.forward_token(token)
    return tuple(generated)


def _address_digest(signature: dict[str, int]) -> str:
    payload = "\n".join(
        f"{name}:{pointer}" for name, pointer in sorted(signature.items())
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    budgets = _parse_budgets(args.budgets)
    prompt_ids = _prompt_ids(args.template_fixture, args.prompt_case)
    max_new_tokens = int(args.max_new_tokens)
    if max_new_tokens <= 1:
        raise ValueError("max_new_tokens must exceed one for a speculative cycle")
    compiler_version = (
        args.compiler_version_file.read_text(encoding="utf-8")
        if args.compiler_version_file is not None
        else None
    )
    reset_memory_stats()
    process_started = time.perf_counter()
    rows: list[dict[str, object]] = []
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
        baseline_started = time.perf_counter()
        ar_ids = _serial_target_ar(
            target,
            prompt_ids,
            max_new_tokens=max_new_tokens,
        )
        baseline_seconds = time.perf_counter() - baseline_started
        target.reset_state()

        for budget in budgets:
            budget_started = time.perf_counter()
            cycles: list[dict[str, object]] = []
            committed_generation_inputs: list[int] = []
            with LagunaDFlashResidentDrafter(
                target,
                args.drafter,
                candidate_budget=budget,
                top_k=1,
                max_append_rows=budget + 1,
                compiler_version=compiler_version,
                require_cached_build=args.require_cached_build,
            ) as drafter:
                with LagunaDFlashResidentCycle(target, drafter) as cycle:
                    context_started = time.perf_counter()
                    prompt_result = cycle.prefill(prompt_ids)
                    context_seconds = time.perf_counter() - context_started
                    generated = [int(prompt_result.next_token_id)]
                    root = generated[0]
                    while len(generated) < max_new_tokens:
                        remaining = max_new_tokens - len(generated)
                        cycle_started = time.perf_counter()
                        result = cycle.run_cycle(root, remaining_decode=remaining)
                        cycle_seconds = time.perf_counter() - cycle_started
                        committed_generation_inputs.extend(
                            result.target_result.committed_input_ids
                        )
                        generated.extend(result.visible_output_ids)
                        if len(generated) > max_new_tokens:
                            raise RuntimeError("DFlash visible output exceeded remaining decode budget")
                        committed_prefix_exact = tuple(committed_generation_inputs) == ar_ids[
                            : len(committed_generation_inputs)
                        ]
                        expected_position = len(prompt_ids) + len(committed_generation_inputs) - 1
                        state_aligned = bool(
                            committed_prefix_exact
                            and target.position == expected_position
                            and drafter.committed_context_tokens == target.position + 1
                        )
                        cycles.append(
                            {
                                "root_token_id": root,
                                "candidate_token_ids": list(
                                    result.proposal.candidate_token_ids
                                ),
                                "target_top1_ids": list(
                                    result.target_result.target_top1_ids
                                ),
                                "accepted_draft_tokens": result.target_result.accepted_draft_count,
                                "accepted_token_ids": list(
                                    result.target_result.accepted_token_ids
                                ),
                                "visible_output_ids": list(result.visible_output_ids),
                                "next_token_id": result.target_result.next_token_id,
                                "full_accept": result.target_result.full_accept,
                                "commit_row": result.target_result.commit_row,
                                "commit_position": result.target_result.commit_position,
                                "committed_rows": result.target_result.committed_rows,
                                "rejected_rows": budget + 1
                                - result.target_result.committed_rows,
                                "packed_accept_payload": list(
                                    result.target_result.packed_payload
                                ),
                                "committed_prefix_exact_ar": committed_prefix_exact,
                                "target_drafter_state_aligned": state_aligned,
                                "verifier_addresses_stable": result.verifier_addresses_stable,
                                "seconds": cycle_seconds,
                            }
                        )
                        if not state_aligned:
                            raise RuntimeError("DFlash committed state diverged from target AR prefix")
                        next_token = result.target_result.next_token_id
                        if next_token is None:
                            break
                        root = next_token
                    generated_ids = tuple(generated)
                    exact = generated_ids == ar_ids
                    row = {
                        "candidate_budget": budget,
                        "passed": bool(
                            exact
                            and len(generated_ids) == max_new_tokens
                            and all(
                                cycle_row["target_drafter_state_aligned"]
                                and cycle_row["verifier_addresses_stable"]
                                for cycle_row in cycles
                            )
                        ),
                        "generated_ids": list(generated_ids),
                        "ar_ids": list(ar_ids),
                        "exact_match_ar": exact,
                        "cycles": cycles,
                        "accepted_draft_tokens": sum(
                            int(cycle_row["accepted_draft_tokens"])
                            for cycle_row in cycles
                        ),
                        "generated_draft_tokens": budget * len(cycles),
                        "target_verify_rows": (budget + 1) * len(cycles),
                        "rejected_target_rows": sum(
                            int(cycle_row["rejected_rows"]) for cycle_row in cycles
                        ),
                        "context_seconds": context_seconds,
                        "total_seconds": time.perf_counter() - budget_started,
                        "verifier_address_digest_sha256": _address_digest(
                            cycle.address_signature()
                        ),
                        "target_position": target.position,
                        "drafter_context_tokens": drafter.committed_context_tokens,
                        "resident_bytes": target.resident_nbytes + drafter.resident_nbytes,
                    }
                    rows.append(row)
            target.reset_state()

        inside_memory = memory_stats()
    after_close = memory_stats()
    result = {
        "schema": 1,
        "passed": bool(
            rows
            and all(bool(row["passed"]) for row in rows)
            and after_close["current_allocated_bytes"] == 0
            and after_close["active_allocations"] == 0
        ),
        "model": str(args.model.resolve()),
        "drafter": str(args.drafter.resolve()),
        "backend": args.backend,
        "prompt_case": args.prompt_case,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "budgets": list(budgets),
        "true_ar_ids": list(ar_ids),
        "true_ar_seconds": baseline_seconds,
        "target_load_seconds": target_loaded - process_started,
        "rows": rows,
        "memory_inside_target": inside_memory,
        "memory_after_close": after_close,
        "process_seconds": time.perf_counter() - process_started,
    }
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
