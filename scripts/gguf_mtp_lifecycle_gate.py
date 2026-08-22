#!/usr/bin/env python3
"""RF3 real-GPU lifecycle/fault gate for dense GGUF MTP.

The gate injects faults at stable cycle phases, proves cancellation observed
during a cycle is delayed until target commit + draft repair, exercises EOS
termination, and requires a clean AR then MTP health request after every fault.
It emits direct structured evidence and makes no performance claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.generation.deadline import (
    GenerationCancelled,
    GenerationDeadlineExceeded,
)
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    borrow_qwen35_gguf_nextn_fallback_weights,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.gguf_mtp_long_context_gate import _atomic_write_json, _prompt, _provenance

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
FAULT_PHASES = (
    "before_proposal",
    "after_proposal_before_target",
    "after_target_prepare",
    "after_target_commit",
    "after_draft_repair",
    "before_output_publication",
)
EXPECTED_SUCCESS_PHASES = (
    "before_proposal",
    "after_proposal_before_target",
    "after_target_prepare",
    "after_target_commit",
    "after_draft_repair",
    "before_output_publication",
    "after_output_publication",
)


class InjectedLifecycleFault(RuntimeError):
    pass


def _run_ar(target: Qwen35GGUFResidentSession, prompt: Sequence[int], count: int) -> tuple[int, ...]:
    target.reset()
    first = target.prefill(prompt, use_bulk=True, return_logits=False)
    output = [int(first.token_id)]
    while len(output) < int(count):
        output.append(int(target.step(output[-1], return_logits=False).token_id))
    return tuple(output)


def _release(provider: Qwen35GGUFNextNDraftProvider, request_id: int) -> None:
    try:
        provider.release_request(int(request_id))
    except Exception:
        # A fault before draft admission may leave no request slot to release.
        pass


def _mtp_health(
    target: Qwen35GGUFResidentSession,
    provider: Qwen35GGUFNextNDraftProvider,
    prompt: Sequence[int],
    expected: Sequence[int],
    *,
    request_id: int,
) -> dict[str, Any]:
    with Qwen35GGUFMTPDecodeSession(
        target,
        provider,
        candidate_budget=3,
        quant="gguf_q4_k_m",
        target_verify_mode="native",
    ) as decoder:
        result = decoder.generate(
            prompt,
            max_new_tokens=len(expected),
            request_id=int(request_id),
            return_cycle_logits=True,
            use_bulk_prefill=True,
        )
    _release(provider, request_id)
    phases = [tuple(row.get("lifecycle_phases", ())) for row in result.cycle_records]
    return {
        "output_ids_exact": tuple(result.token_ids) == tuple(expected),
        "gpu_accept_match_cpu": bool(result.gpu_accept_match_cpu),
        "cycles": int(result.cycles),
        "lifecycle_phases": phases,
        "lifecycle_phases_exact": bool(phases)
        and all(row == EXPECTED_SUCCESS_PHASES for row in phases),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _prompt(int(args.prompt_tokens))
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
                    "kind": "gguf_mtp_lifecycle_checkpoint",
                    "status": "running",
                    "active_event": event,
                    "active_details": details,
                    "rows": rows,
                },
            )

    with Qwen35GGUFResidentSession(
        args.model,
        max_sequence_length=int(args.max_sequence_length),
        require_cached_build=bool(args.require_cached_build),
    ) as target:
        target.select_prefill_quant("gguf_q4_k_m")
        expected = _run_ar(target, prompt, int(args.max_new_tokens))
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            args.model,
            max_positions=int(args.max_sequence_length),
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=bool(args.require_cached_build),
            borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(target),
        )
        try:
            for index, fault_phase in enumerate(FAULT_PHASES):
                request_id = 60_000 + index
                seen: list[str] = []

                def fault_hook(phase: str, *, selected=fault_phase) -> None:
                    seen.append(str(phase))
                    if phase == selected:
                        raise InjectedLifecycleFault(f"injected:{phase}")

                checkpoint("fault_start", {"phase": fault_phase})
                error = None
                decoder = Qwen35GGUFMTPDecodeSession(
                    target,
                    provider,
                    candidate_budget=3,
                    quant="gguf_q4_k_m",
                    target_verify_mode="native",
                )
                try:
                    decoder.generate(
                        prompt,
                        max_new_tokens=int(args.max_new_tokens),
                        request_id=request_id,
                        return_cycle_logits=True,
                        use_bulk_prefill=True,
                        lifecycle_hook=fault_hook,
                    )
                except InjectedLifecycleFault as exc:
                    error = str(exc)
                finally:
                    prepared_open = decoder.verifier._prepared is not None
                    decoder.close()
                    _release(provider, request_id)
                ar_health = _run_ar(target, prompt, len(expected))
                mtp_health = _mtp_health(
                    target,
                    provider,
                    prompt,
                    expected,
                    request_id=61_000 + index,
                )
                row = {
                    "kind": "phase_fault",
                    "phase": fault_phase,
                    "error": error,
                    "seen_phases": seen,
                    "verifier_prepared_open_after_fault": prepared_open,
                    "ar_health_exact": tuple(ar_health) == tuple(expected),
                    "mtp_health": mtp_health,
                }
                row["passed"] = bool(
                    error == f"injected:{fault_phase}"
                    and not prepared_open
                    and row["ar_health_exact"]
                    and mtp_health["output_ids_exact"]
                    and mtp_health["gpu_accept_match_cpu"]
                    and mtp_health["lifecycle_phases_exact"]
                )
                rows.append(row)
                checkpoint("fault_complete", {"phase": fault_phase, "passed": row["passed"]})

            # Simulate cancellation arriving while the owned cycle is in flight:
            # the first checkpoint admits the cycle; the second is after target
            # commit and draft repair and must suppress response publication.
            checkpoint_calls = 0
            seen: list[str] = []

            def cancel_after_cycle() -> None:
                nonlocal checkpoint_calls
                checkpoint_calls += 1
                if checkpoint_calls == 2:
                    raise GenerationCancelled()

            cancel_error = None
            decoder = Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=3,
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            )
            cancel_request_id = 62_000
            try:
                decoder.generate(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    request_id=cancel_request_id,
                    return_cycle_logits=True,
                    use_bulk_prefill=True,
                    checkpoint=cancel_after_cycle,
                    lifecycle_hook=seen.append,
                )
            except GenerationCancelled as exc:
                cancel_error = exc.finish_details.reason
            finally:
                prepared_open = decoder.verifier._prepared is not None
                decoder.close()
                _release(provider, cancel_request_id)
            cancel_health = _mtp_health(
                target,
                provider,
                prompt,
                expected,
                request_id=62_001,
            )
            cancel_row = {
                "kind": "inflight_cancel",
                "checkpoint_calls": checkpoint_calls,
                "seen_phases": seen,
                "error_reason": cancel_error,
                "verifier_prepared_open_after_cancel": prepared_open,
                "mtp_health": cancel_health,
            }
            cancel_row["passed"] = bool(
                cancel_error == "cancelled"
                and checkpoint_calls == 2
                and seen[-1:] == ["after_draft_repair"]
                and not prepared_open
                and cancel_health["output_ids_exact"]
                and cancel_health["lifecycle_phases_exact"]
            )
            rows.append(cancel_row)
            checkpoint("cancel_complete", {"passed": cancel_row["passed"]})

            deadline_calls = 0
            deadline_seen: list[str] = []

            def deadline_after_cycle() -> None:
                nonlocal deadline_calls
                deadline_calls += 1
                if deadline_calls == 2:
                    raise GenerationDeadlineExceeded(deadline_at=time.perf_counter())

            deadline_error = None
            deadline_request_id = 62_100
            decoder = Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=3,
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            )
            try:
                decoder.generate(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    request_id=deadline_request_id,
                    return_cycle_logits=True,
                    use_bulk_prefill=True,
                    checkpoint=deadline_after_cycle,
                    lifecycle_hook=deadline_seen.append,
                )
            except GenerationDeadlineExceeded as exc:
                deadline_error = exc.finish_details.reason
            finally:
                deadline_prepared_open = decoder.verifier._prepared is not None
                decoder.close()
                _release(provider, deadline_request_id)
            deadline_health = _mtp_health(
                target,
                provider,
                prompt,
                expected,
                request_id=62_101,
            )
            deadline_row = {
                "kind": "inflight_deadline",
                "checkpoint_calls": deadline_calls,
                "seen_phases": deadline_seen,
                "error_reason": deadline_error,
                "verifier_prepared_open_after_deadline": deadline_prepared_open,
                "mtp_health": deadline_health,
            }
            deadline_row["passed"] = bool(
                deadline_error == "deadline_exceeded"
                and deadline_calls == 2
                and deadline_seen[-1:] == ["after_draft_repair"]
                and not deadline_prepared_open
                and deadline_health["output_ids_exact"]
                and deadline_health["lifecycle_phases_exact"]
            )
            rows.append(deadline_row)
            checkpoint("deadline_complete", {"passed": deadline_row["passed"]})

            # EOS at the prefill seed is a zero-cycle terminal path. It must
            # release cleanly and leave the following MTP request healthy.
            eos_request_id = 63_000
            with Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=3,
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            ) as decoder:
                eos = decoder.generate(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    request_id=eos_request_id,
                    eos_token_id=int(expected[0]),
                    use_bulk_prefill=True,
                )
            _release(provider, eos_request_id)
            eos_health = _mtp_health(
                target,
                provider,
                prompt,
                expected,
                request_id=63_001,
            )
            eos_row = {
                "kind": "eos_prefill_terminal",
                "token_ids": list(eos.token_ids),
                "cycles": int(eos.cycles),
                "mtp_health": eos_health,
            }
            eos_row["passed"] = bool(
                tuple(eos.token_ids) == (int(expected[0]),)
                and eos.cycles == 0
                and eos_health["output_ids_exact"]
                and eos_health["lifecycle_phases_exact"]
            )
            rows.append(eos_row)
            checkpoint("eos_complete", {"passed": eos_row["passed"]})

            for offset, (kind, token_id) in enumerate(
                (
                    ("eos_in_cycle", int(expected[4])),
                    ("stop_token_in_cycle", int(expected[3])),
                )
            ):
                request_id = 63_100 + offset
                with Qwen35GGUFMTPDecodeSession(
                    target,
                    provider,
                    candidate_budget=3,
                    quant="gguf_q4_k_m",
                    target_verify_mode="native",
                ) as decoder:
                    terminal = decoder.generate(
                        prompt,
                        max_new_tokens=int(args.max_new_tokens),
                        request_id=request_id,
                        eos_token_id=(token_id if kind == "eos_in_cycle" else None),
                        stop_token_ids=(
                            (token_id,) if kind == "stop_token_in_cycle" else ()
                        ),
                        use_bulk_prefill=True,
                    )
                _release(provider, request_id)
                expected_length = 5 if kind == "eos_in_cycle" else 4
                terminal_health = _mtp_health(
                    target,
                    provider,
                    prompt,
                    expected,
                    request_id=63_200 + offset,
                )
                terminal_row = {
                    "kind": kind,
                    "terminal_token_id": token_id,
                    "token_ids": list(terminal.token_ids),
                    "cycles": int(terminal.cycles),
                    "mtp_health": terminal_health,
                }
                terminal_row["passed"] = bool(
                    tuple(terminal.token_ids) == tuple(expected[:expected_length])
                    and terminal.cycles == 1
                    and terminal_health["output_ids_exact"]
                    and terminal_health["lifecycle_phases_exact"]
                )
                rows.append(terminal_row)
                checkpoint(
                    "terminal_complete",
                    {"kind": kind, "passed": terminal_row["passed"]},
                )
        finally:
            provider.close()

    passed = bool(rows) and all(bool(row["passed"]) for row in rows)
    payload = {
        "schema": 1,
        "kind": "gguf_mtp_lifecycle_fault_gate",
        "status": "passed" if passed else "failed",
        "verdict": "pass" if passed else "fail",
        "performance_claim": False,
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "prompt_tokens": int(args.prompt_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "max_sequence_length": int(args.max_sequence_length),
            "fault_phases": list(FAULT_PHASES),
        },
        "provenance": _provenance(args.model, hash_model=bool(args.hash_model)),
        "rows": rows,
        "memory": memory_stats(),
        "summary": {
            "passed": sum(bool(row["passed"]) for row in rows),
            "total": len(rows),
            "wall_seconds": time.perf_counter() - started,
        },
        "passed": passed,
    }
    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.out, payload)
        print(f"wrote {args.out}: passed={passed} rows={len(rows)}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.prompt_tokens <= 0 or args.max_new_tokens < 2:
        raise SystemExit("prompt tokens must be positive and max-new-tokens must be >=2")
    payload = run(args)
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
