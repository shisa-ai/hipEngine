#!/usr/bin/env python3
"""Inject recoverable SPECDEC2 phase failures and prove later exact health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from hipengine import LLM
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _invoke_original_args(
    call_args: tuple[Any, ...],
    *,
    staticmethod_owner: bool,
) -> tuple[Any, ...]:
    return call_args[1:] if staticmethod_owner else call_args


def _resident_capacity(concurrency: int, requested: int | None) -> int:
    capacity = int(concurrency) if requested is None else int(requested)
    if capacity < int(concurrency):
        raise ValueError("resident capacity cannot be smaller than concurrency")
    return capacity


def _request(prompt: str, *, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def _outcomes(
    handles: Sequence[Any],
) -> tuple[tuple[tuple[int, ...] | None, ...], tuple[str | None, ...]]:
    outputs: list[tuple[int, ...] | None] = []
    errors: list[str | None] = []
    for handle in handles:
        try:
            result = handle.result(timeout=180)
        except BaseException as error:
            outputs.append(None)
            errors.append(f"{type(error).__name__}:{error}")
        else:
            outputs.append(
                tuple(int(token) for token in result.generated_token_ids)
            )
            errors.append(None)
    return tuple(outputs), tuple(errors)


def _ids(handles: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    outputs, errors = _outcomes(handles)
    if any(error is not None for error in errors):
        raise RuntimeError(f"generation failed: {errors}")
    return tuple(output for output in outputs if output is not None)


def _recent_rows(snapshot: dict[str, Any], request_ids: set[int]) -> list[dict[str, Any]]:
    return [
        row
        for row in snapshot["runner"]["routes"]["recent_completed"]
        if int(row["request_id"]) in request_ids
    ]


def _failure_phase_specs(
    concurrency: int = 1,
) -> tuple[
    tuple[str, type[Any], str, Callable[..., Any], bool], ...
]:
    """Return active-path owners and whether failure follows target commit."""

    if int(concurrency) > 1:
        return (
            (
                "proposal",
                Qwen35GGUFNextNDraftProvider,
                "propose_batch_device",
                Qwen35GGUFNextNDraftProvider.propose_batch_device,
                False,
            ),
            (
                "target",
                Qwen35GGUFResidentSession,
                "verify_target_blocks_batch",
                Qwen35GGUFResidentSession.verify_target_blocks_batch,
                False,
            ),
            (
                "readback",
                Qwen35GGUFMTP2Adapter,
                "_read_target_batch_accept",
                Qwen35GGUFMTP2Adapter._read_target_batch_accept,
                True,
            ),
        )
    return (
        (
            "proposal",
            Qwen35GGUFNextNDraftProvider,
            "launch_device_proposal",
            Qwen35GGUFNextNDraftProvider.launch_device_proposal,
            False,
        ),
        (
            "target",
            Qwen35GGUFResidentSession,
            "verify_target_from_device_proposal",
            Qwen35GGUFResidentSession.verify_target_from_device_proposal,
            False,
        ),
        (
            "readback",
            Qwen35GGUFTransactionalVerifier,
            "prepare",
            Qwen35GGUFTransactionalVerifier.prepare,
            True,
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    concurrency = int(args.concurrency)
    resident_capacity = _resident_capacity(
        concurrency,
        getattr(args, "resident_capacity", None),
    )
    candidate_budget = int(getattr(args, "candidate_budget", 2))
    llm = LLM(
        str(args.model),
        backend="hip_gfx1151",
        execution_profile=str(args.execution_profile),
        max_active_requests=resident_capacity,
        max_sequence_length=int(args.max_sequence_length),
        speculative_candidate_budget=candidate_budget,
    )
    service = llm._get_text_generator()
    greeting = _request(
        "Write one short greeting.",
        max_tokens=int(args.max_tokens),
    )
    health = _request(
        "Write one short farewell.",
        max_tokens=int(args.max_tokens),
    )
    phases = _failure_phase_specs(concurrency)
    results: list[dict[str, Any]] = []
    try:
        ar_greeting = _ids(
            service.submit_children((greeting,) * concurrency)
        )
        ar_health = _ids(service.submit_children((health,) * concurrency))
        next_request_id = 2 * concurrency
        for phase, owner, method_name, original, fail_after in phases:
            raised = False

            descriptor = owner.__dict__.get(method_name)
            staticmethod_owner = isinstance(descriptor, staticmethod)

            def fail_once(
                *call_args,
                __original=original,
                __phase=phase,
                __fail_after=fail_after,
                __staticmethod_owner=staticmethod_owner,
                **call_kwargs,
            ):
                nonlocal raised
                original_args = _invoke_original_args(
                    call_args,
                    staticmethod_owner=__staticmethod_owner,
                )
                if __fail_after:
                    result = __original(*original_args, **call_kwargs)
                    if not raised:
                        raised = True
                        raise RuntimeError(f"injected SPECDEC2 {__phase} failure")
                    return result
                if not raised:
                    raised = True
                    raise RuntimeError(f"injected SPECDEC2 {__phase} failure")
                return __original(*original_args, **call_kwargs)

            setattr(owner, method_name, fail_once)
            fault_ids = set(
                range(next_request_id, next_request_id + concurrency)
            )
            fault_output, fault_errors = _outcomes(
                service.submit_speculative_children((greeting,) * concurrency)
            )
            setattr(owner, method_name, original)
            next_request_id += concurrency
            health_ids = set(
                range(next_request_id, next_request_id + concurrency)
            )
            health_output, health_errors = _outcomes(
                service.submit_speculative_children((health,) * concurrency)
            )
            next_request_id += concurrency
            snapshot = service.live_loop_snapshot()
            fault_rows = _recent_rows(snapshot, fault_ids)
            health_rows = _recent_rows(snapshot, health_ids)
            # C1 readback remains inside the transactional verifier journal and
            # restores the canonical target cursor. Physical C>1 commits target
            # rows before its packed readback and therefore needs target rebuild.
            precommit = phase != "readback" or concurrency == 1
            fault_contract_passed = bool(
                (
                    precommit
                    and fault_output == ar_greeting
                    and fault_errors == (None,) * concurrency
                    and len(fault_rows) == concurrency
                    and all(
                        int(row["specdec2_mtp2_recoverable_failures"]) == 1
                        and row["specdec2_mtp2_failure_reasons"]
                        == [
                            "precommit_failure_ar_fallback",
                            f"RuntimeError:injected SPECDEC2 {phase} failure",
                        ]
                        and int(row["specdec2_mtp2_cycles"]) == 0
                        for row in fault_rows
                    )
                )
                or (
                    not precommit
                    and fault_output == ar_greeting
                    and fault_errors == (None,) * concurrency
                    and len(fault_rows) == concurrency
                    and all(
                        int(row["specdec2_mtp2_recoverable_failures"]) == 1
                        and row["specdec2_mtp2_failure_reasons"][0]
                        == "postcommit_target_rebuild_ar_fallback"
                        and "RuntimeError:injected SPECDEC2 readback failure"
                        in row["specdec2_mtp2_failure_reasons"][1]
                        and int(row["specdec2_mtp2_cycles"]) == 0
                        for row in fault_rows
                    )
                )
            )
            phase_passed = bool(
                raised
                and fault_contract_passed
                and health_output == ar_health
                and health_errors == (None,) * concurrency
                and len(health_rows) == concurrency
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
                    "fault_errors": fault_errors,
                    "fault_contract": (
                        "precommit_exact_ar_recovery"
                        if precommit
                        else "postcommit_target_rebuild_ar_recovery"
                    ),
                    "health_output": health_output,
                    "health_errors": health_errors,
                    "ar_greeting": ar_greeting,
                    "ar_health": ar_health,
                    "fault_rows": fault_rows,
                    "health_rows": health_rows,
                    "passed": phase_passed,
                }
            )
        final = service.live_loop_snapshot()
    finally:
        for _, owner, method_name, original, _fail_after in phases:
            setattr(owner, method_name, original)
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
        "workload": {
            "concurrency": concurrency,
            "resident_capacity": resident_capacity,
            "candidate_budget": candidate_budget,
            "max_sequence_length": int(args.max_sequence_length),
            "max_tokens": int(args.max_tokens),
        },
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
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resident-capacity", type=int, default=None)
    parser.add_argument("--candidate-budget", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=5)
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
