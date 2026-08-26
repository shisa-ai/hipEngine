#!/usr/bin/env python3
"""Inject recoverable SPECDEC2 phase failures and prove later exact health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from hipengine import LLM
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _request(prompt: str) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def _ids(handles: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token) for token in handle.result(timeout=180).generated_token_ids)
        for handle in handles
    )


def _recent_rows(snapshot: dict[str, Any], request_ids: set[int]) -> list[dict[str, Any]]:
    return [
        row
        for row in snapshot["runner"]["routes"]["recent_completed"]
        if int(row["request_id"]) in request_ids
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    llm = LLM(
        str(args.model),
        backend="hip_gfx1151",
        execution_profile=str(args.execution_profile),
        max_active_requests=2,
        max_sequence_length=256,
    )
    service = llm._get_text_generator()
    greeting = _request("Write one short greeting.")
    health = _request("Write one short farewell.")
    original_proposal = Qwen35GGUFNextNDraftProvider.propose_batch_device
    original_target = Qwen35GGUFResidentSession.verify_target_blocks_batch
    original_readback = Qwen35GGUFNextNDraftProvider.materialize_batch_device_proposal
    results: list[dict[str, Any]] = []
    try:
        ar_health = _ids(service.submit_children((health, health)))
        next_request_id = 2
        phases: tuple[tuple[str, object, str, Callable[..., Any]], ...] = (
            (
                "proposal",
                Qwen35GGUFNextNDraftProvider,
                "propose_batch_device",
                original_proposal,
            ),
            (
                "target",
                Qwen35GGUFResidentSession,
                "verify_target_blocks_batch",
                original_target,
            ),
            (
                "readback",
                Qwen35GGUFNextNDraftProvider,
                "materialize_batch_device_proposal",
                original_readback,
            ),
        )
        for phase, owner, method_name, original in phases:
            raised = False

            def fail_once(*call_args, __original=original, __phase=phase, **call_kwargs):
                nonlocal raised
                if not raised:
                    raised = True
                    raise RuntimeError(f"injected SPECDEC2 {__phase} failure")
                return __original(*call_args, **call_kwargs)

            setattr(owner, method_name, fail_once)
            fault_ids = {next_request_id, next_request_id + 1}
            fault_output = _ids(
                service.submit_speculative_children((greeting, greeting))
            )
            setattr(owner, method_name, original)
            next_request_id += 2
            health_ids = {next_request_id, next_request_id + 1}
            health_output = _ids(
                service.submit_speculative_children((health, health))
            )
            next_request_id += 2
            snapshot = service.live_loop_snapshot()
            fault_rows = _recent_rows(snapshot, fault_ids)
            health_rows = _recent_rows(snapshot, health_ids)
            phase_passed = bool(
                raised
                and fault_output == ((271, 9419, 0, 2500, 628),) * 2
                and health_output == ar_health
                and len(fault_rows) == 2
                and all(
                    int(row["specdec2_mtp2_recoverable_failures"]) == 1
                    and row["specdec2_mtp2_failure_reasons"]
                    == ["precommit_failure_ar_fallback"]
                    and int(row["specdec2_mtp2_cycles"]) == 0
                    for row in fault_rows
                )
                and len(health_rows) == 2
                and all(
                    int(row["specdec2_mtp2_cycles"]) > 0
                    and int(row["specdec2_mtp2_recoverable_failures"]) == 0
                    for row in health_rows
                )
            )
            results.append(
                {
                    "phase": phase,
                    "raised": raised,
                    "fault_output": fault_output,
                    "health_output": health_output,
                    "ar_health": ar_health,
                    "fault_rows": fault_rows,
                    "health_rows": health_rows,
                    "passed": phase_passed,
                }
            )
        final = service.live_loop_snapshot()
    finally:
        Qwen35GGUFNextNDraftProvider.propose_batch_device = original_proposal
        Qwen35GGUFResidentSession.verify_target_blocks_batch = original_target
        Qwen35GGUFNextNDraftProvider.materialize_batch_device_proposal = original_readback
        llm.close()

    passed = bool(
        all(row["passed"] for row in results)
        and final["loop"]["requests"]["active"] == 0
        and final["loop"]["requests"]["pending"] == 0
        and final["engine_service"]["active_children"] == 0
    )
    return {
        "schema": 1,
        "kind": "specdec2_s6_recoverable_failure_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "execution_profile": str(args.execution_profile),
        "phases": results,
        "final_snapshot": final,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
    )
    parser.add_argument(
        "--execution-profile",
        choices=("strict", "production"),
        default="strict",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "phases": {
                    row["phase"]: row["passed"] for row in payload["phases"]
                },
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
