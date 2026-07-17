#!/usr/bin/env python3
"""Validate arbitrary-C GGUF retirement and new admission against c1.

The gate keeps scheduler slot identity stable, lowers logical C through the
production resident runner, and compares generated tokens plus every Conv/GDN
state and live BF16 KV byte against independent c1 checkpoints.  It is a
correctness diagnostic, not a throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hipengine import LLM, SamplingParams  # noqa: E402
from hipengine.generation import GenerationRequest  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
)
from scripts.gguf_packed_ar_state_oracle import (  # noqa: E402
    _capture_state,
    _compare_state_rows,
)


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _request(prompts: Sequence[Sequence[int]], *, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=tuple(tuple(int(token) for token in prompt) for prompt in prompts),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _run_reference(
    session: Any,
    prompt: Sequence[int],
    *,
    max_tokens: int,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    result = session.prefill(
        tuple(int(token) for token in prompt),
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    tokens = [int(result.token_id)]
    states = {1: _capture_state(session)}
    while len(tokens) < int(max_tokens):
        result = session.step(tokens[-1], return_logits=False)
        tokens.append(int(result.token_id))
        states[len(tokens)] = _capture_state(session)
    return tokens, states


def _compare_one(actual: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return _compare_state_rows([actual], [expected])


def _plan_summary(plan: Any) -> dict[str, Any] | None:
    if not isinstance(plan, dict) or not plan:
        return None
    return json.loads(json.dumps(plan))


def _poll_one(
    adapter: Any,
    runner: Any,
    *,
    label: str,
    token_counts: dict[int, int],
    timeline: list[dict[str, Any]],
) -> tuple[Any, ...]:
    events = tuple(adapter.poll(max_ticks=1))
    if not events:
        snapshot = adapter.live_loop_snapshot()
        debug_path = Path("/tmp/hipengine-gguf-arbitrary-c-stall.json")
        debug_path.write_text(
            json.dumps(
                {"label": str(label), "snapshot": snapshot, "timeline": timeline},
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"resident arbitrary-C loop stalled at {label}; debug={debug_path}"
        )
    for event in events:
        if event.kind == "token":
            token_counts[int(event.request_id)] = token_counts.get(int(event.request_id), 0) + 1
    work = [event.work_kind.value for event in events if event.kind == "work"]
    plan = (
        _plan_summary(runner._last_physical_group_plan)
        if work == ["decode"]
        else None
    )
    timeline.append(
        {
            "label": str(label),
            "work": work,
            "admitted": [int(event.request_id) for event in events if event.kind == "admitted"],
            "tokens": [int(event.request_id) for event in events if event.kind == "token"],
            "completed": [int(event.request_id) for event in events if event.kind == "completed"],
            "token_counts": {str(key): int(value) for key, value in sorted(token_counts.items())},
            "physical_group_plan": plan,
        }
    )
    return events


def _all_packed(plan: dict[str, Any] | None) -> bool:
    return bool(
        plan
        and int(plan.get("group_count", 0)) > 0
        and all(
            group.get("execution_path") == "packed_native"
            for group in plan.get("groups", ())
        )
    )


def _group_masks(plan: dict[str, Any] | None) -> list[str]:
    if plan is None:
        return []
    return [
        "".join("1" if active else "0" for active in group.get("active_mask", ()))
        for group in plan.get("groups", ())
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    logical_c = int(args.rows)
    if logical_c <= 8:
        raise ValueError("rows must be greater than 8 for the arbitrary-C lifecycle gate")
    cancel_slots = tuple(int(slot) for slot in args.cancel_slots)
    if len(cancel_slots) != 2 or len(set(cancel_slots)) != 2:
        raise ValueError("cancel-slots must contain two unique slots")
    if any(slot < 0 or slot >= logical_c for slot in cancel_slots):
        raise ValueError("cancel-slots must be within rows")
    original_max_tokens = int(args.original_max_tokens)
    newcomer_max_tokens = int(args.newcomer_max_tokens)
    if original_max_tokens < 5 or newcomer_max_tokens < 3:
        raise ValueError("original/newcomer max tokens must be at least 5/3")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")

    prompt_base = int(args.prompt_token_id)
    original_prompts = tuple(
        tuple([prompt_base + row] * (int(args.prompt_length) + row))
        for row in range(logical_c)
    )
    newcomer_prompts = tuple(
        tuple([prompt_base + logical_c + row] * (int(args.prompt_length) + row + 1))
        for row in range(len(cancel_slots))
    )
    max_sequence_length = max(
        *(len(prompt) + original_max_tokens for prompt in original_prompts),
        *(len(prompt) + newcomer_max_tokens for prompt in newcomer_prompts),
    ) + 2
    compiler_version_file = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.expanduser().resolve()
    )
    if compiler_version_file is not None and not compiler_version_file.is_file():
        raise ValueError(f"compiler version file does not exist: {compiler_version_file}")

    env = {
        "HIPENGINE_MAX_ACTIVE_REQUESTS": str(logical_c),
        "HIPENGINE_PREFILL_DECODE_POLICY": "protect_ttft",
        "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
        "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
    }
    if compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_version_file)

    started = time.perf_counter()
    with _temporary_env(env):
        llm = LLM(model, backend=str(args.backend))
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(max_tokens=original_max_tokens),
            )
            adapter._loop.prefill_chunk_size = int(args.prefill_chunk_size)
            runner = adapter._runner
            if int(runner.capacity) != logical_c:
                raise RuntimeError(
                    f"expected resident capacity {logical_c}, got {runner.capacity}"
                )

            reference_session = Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                runtime=runner._shared_runner.runtime,
                shared_runner=runner._shared_runner,
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                compiler_version=(
                    None
                    if compiler_version_file is None
                    else compiler_version_file.read_text(encoding="utf-8").strip()
                ),
                require_cached_build=bool(args.require_cached_build),
            )
            try:
                original_reference_tokens: list[list[int]] = []
                original_reference_states: list[dict[int, dict[str, Any]]] = []
                for prompt in original_prompts:
                    tokens, states = _run_reference(
                        reference_session,
                        prompt,
                        max_tokens=original_max_tokens,
                    )
                    original_reference_tokens.append(tokens)
                    original_reference_states.append(states)
                    reference_session.reset()
                newcomer_reference_tokens: list[list[int]] = []
                newcomer_reference_states: list[dict[int, dict[str, Any]]] = []
                for prompt in newcomer_prompts:
                    tokens, states = _run_reference(
                        reference_session,
                        prompt,
                        max_tokens=newcomer_max_tokens,
                    )
                    newcomer_reference_tokens.append(tokens)
                    newcomer_reference_states.append(states)
                    reference_session.reset()
            finally:
                reference_session.close()

            reclaimed_states: dict[int, dict[str, Any]] = {}
            reclaimed_session_ids: dict[int, int] = {}
            original_reclaim = runner.reclaim

            def capture_reclaim(self, completed):
                request_id = int(completed.request_id)
                row = self._rows.get(request_id)
                if row is not None and row.lease is not None:
                    if row.slot is not None:
                        self._flush_row_owner(row)
                        reclaimed_states[request_id] = _capture_state(row.lease.session)
                    reclaimed_session_ids[request_id] = id(row.lease.session)
                return original_reclaim(completed)

            runner.reclaim = MethodType(capture_reclaim, runner)
            timeline: list[dict[str, Any]] = []
            token_counts: dict[int, int] = {}

            original = adapter.submit_detailed(
                _request(original_prompts, max_tokens=original_max_tokens)
            )
            original_ids = tuple(int(request_id) for request_id in original.request_ids)
            token_counts.update({request_id: 0 for request_id in original_ids})
            initial_plan = None
            ticks = 0
            while initial_plan is None:
                _poll_one(
                    adapter,
                    runner,
                    label="initial_fill",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                ticks += 1
                candidate = timeline[-1]["physical_group_plan"]
                if (
                    candidate is not None
                    and int(candidate["logical_c"]) == logical_c
                    and all(token_counts[request_id] >= 2 for request_id in original_ids)
                ):
                    initial_plan = candidate
                if ticks > original.max_ticks:
                    raise RuntimeError("initial arbitrary-C fill exceeded tick budget")

            cancelled_ids = tuple(original_ids[slot] for slot in cancel_slots)
            cancelled_session_ids = tuple(
                id(runner._rows[request_id].lease.session)
                for request_id in cancelled_ids
            )
            for slot in cancel_slots:
                adapter.cancel_submission(original, row_index=slot, reason="cancel")
            inactive_sessions = {
                id(lease.session): lease.session
                for lease in runner._available
                if id(lease.session) in cancelled_session_ids
            }
            if set(inactive_sessions) != set(cancelled_session_ids):
                raise RuntimeError("cancelled sessions did not return to the available pool")
            inactive_before_hole = {
                session_id: _capture_state(session)
                for session_id, session in inactive_sessions.items()
            }

            hole_plan = None
            while hole_plan is None:
                _poll_one(
                    adapter,
                    runner,
                    label="middle_hole_decode",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                candidate = timeline[-1]["physical_group_plan"]
                if candidate is not None and int(candidate["logical_c"]) == logical_c - 2:
                    hole_plan = candidate
            runner._flush_all_packed_owners()
            survivor_hole_states = {
                request_id: _capture_state(runner._rows[request_id].lease.session)
                for request_id in original_ids
                if request_id not in cancelled_ids
            }
            survivor_hole_counts = {
                request_id: token_counts[request_id]
                for request_id in survivor_hole_states
            }
            inactive_after_hole = {
                session_id: _capture_state(session)
                for session_id, session in inactive_sessions.items()
            }
            inactive_hole_mismatches = {
                str(session_id): _compare_one(
                    inactive_after_hole[session_id],
                    inactive_before_hole[session_id],
                )
                for session_id in cancelled_session_ids
            }

            newcomers = adapter.submit_detailed(
                _request(newcomer_prompts, max_tokens=newcomer_max_tokens)
            )
            newcomer_ids = tuple(int(request_id) for request_id in newcomers.request_ids)
            token_counts.update({request_id: 0 for request_id in newcomer_ids})
            newcomer_slots: tuple[int, ...] | None = None
            refill_plan = None
            while refill_plan is None:
                _poll_one(
                    adapter,
                    runner,
                    label="new_admission",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                candidate = timeline[-1]["physical_group_plan"]
                if (
                    candidate is not None
                    and int(candidate["logical_c"]) == logical_c
                    and all(token_counts[request_id] >= 2 for request_id in newcomer_ids)
                ):
                    newcomer_slots = tuple(
                        adapter._loop.scheduler.active_batch.slot_for(request_id)
                        for request_id in newcomer_ids
                    )
                    refill_plan = candidate
            if newcomer_slots is None:  # pragma: no cover - guarded by refill_plan
                raise RuntimeError("newcomer physical slots were not observed")

            while not (
                adapter.generation_complete(original)
                and adapter.generation_complete(newcomers)
            ):
                _poll_one(
                    adapter,
                    runner,
                    label="drain",
                    token_counts=token_counts,
                    timeline=timeline,
                )
                ticks += 1
                if ticks > original.max_ticks + newcomers.max_ticks:
                    raise RuntimeError("arbitrary-C lifecycle drain exceeded tick budget")

            original_outputs = tuple(adapter.take_result(original))
            newcomer_outputs = tuple(adapter.take_result(newcomers))
            original_output_tokens = [
                list(output.generated_token_ids or ()) for output in original_outputs
            ]
            newcomer_output_tokens = [
                list(output.generated_token_ids or ()) for output in newcomer_outputs
            ]
            original_finish = [
                None
                if output.finish_details is None
                else output.finish_details.to_json_dict()
                for output in original_outputs
            ]

            cancelled_mismatches = {
                str(slot): _compare_one(
                    reclaimed_states[original_ids[slot]],
                    original_reference_states[slot][
                        len(original_output_tokens[slot])
                    ],
                )
                for slot in cancel_slots
            }
            survivor_hole_mismatches = {
                str(slot): _compare_one(
                    survivor_hole_states[request_id],
                    original_reference_states[slot][
                        survivor_hole_counts[request_id]
                    ],
                )
                for slot, request_id in enumerate(original_ids)
                if request_id not in cancelled_ids
            }
            survivor_final_mismatches = {
                str(slot): _compare_one(
                    reclaimed_states[request_id],
                    original_reference_states[slot][original_max_tokens],
                )
                for slot, request_id in enumerate(original_ids)
                if request_id not in cancelled_ids
            }
            newcomer_final_mismatches = {
                str(index): _compare_one(
                    reclaimed_states[request_id],
                    newcomer_reference_states[index][newcomer_max_tokens],
                )
                for index, request_id in enumerate(newcomer_ids)
            }

            original_tokens_exact = all(
                (
                    tokens == original_reference_tokens[index]
                    if index not in cancel_slots
                    else tokens
                    == original_reference_tokens[index][: len(tokens)]
                )
                for index, tokens in enumerate(original_output_tokens)
            )
            newcomer_tokens_exact = all(
                tokens == newcomer_reference_tokens[index]
                for index, tokens in enumerate(newcomer_output_tokens)
            )
            state_exact = all(
                not mismatches
                for family in (
                    cancelled_mismatches,
                    survivor_hole_mismatches,
                    survivor_final_mismatches,
                    newcomer_final_mismatches,
                    inactive_hole_mismatches,
                )
                for mismatches in family.values()
            )
            all_plans = [
                event["physical_group_plan"]
                for event in timeline
                if event["physical_group_plan"] is not None
            ]
            declared_widths_only = all(
                int(group["physical_rows"]) in {1, 2, 4, 8}
                for plan in all_plans
                for group in plan["groups"]
            )
            no_serial_fallback = all(_all_packed(plan) for plan in all_plans)
            expected_initial_masks = ["11111111", "11111000"]
            expected_hole_masks = [
                "11011111",
                "11011000",
            ]
            expected_refill_masks = expected_initial_masks
            routes = runner.observability_snapshot()["routes"]
            final_active = tuple(runner.active_request_ids)
            final_available = int(runner.available_session_count)
            scheduler_active = int(adapter._loop.active_count)
            cancelled_finish_ok = all(
                original_finish[slot] == {"reason": "cancelled", "cancelled": True}
                for slot in cancel_slots
            )
            newcomer_reclaimed_session_ids = tuple(
                reclaimed_session_ids[request_id] for request_id in newcomer_ids
            )
            session_reused = set(newcomer_reclaimed_session_ids) == set(
                cancelled_session_ids
            )
            passed = bool(
                original_tokens_exact
                and newcomer_tokens_exact
                and state_exact
                and cancelled_finish_ok
                and newcomer_slots == cancel_slots
                and session_reused
                and _group_masks(initial_plan) == expected_initial_masks
                and _group_masks(hole_plan) == expected_hole_masks
                and _group_masks(refill_plan) == expected_refill_masks
                and declared_widths_only
                and no_serial_fallback
                and routes["fallback_reasons"] == {}
                and final_active == ()
                and final_available == logical_c
                and scheduler_active == 0
            )
            return {
                "schema": 1,
                "kind": "gguf_arbitrary_c_lifecycle",
                "status": "passed" if passed else "failed",
                "passed": passed,
                "performance_claim": False,
                "model": str(model),
                "backend": str(args.backend),
                "target_arch": str(runner._shared_runner.target_arch),
                "shape": {
                    "logical_c": logical_c,
                    "cancel_slots": list(cancel_slots),
                    "newcomer_slots": list(newcomer_slots),
                    "prompt_lengths": [len(prompt) for prompt in original_prompts],
                    "newcomer_prompt_lengths": [len(prompt) for prompt in newcomer_prompts],
                    "original_max_tokens": original_max_tokens,
                    "newcomer_max_tokens": newcomer_max_tokens,
                    "prefill_chunk_size": int(args.prefill_chunk_size),
                },
                "initial_plan": initial_plan,
                "middle_hole_plan": hole_plan,
                "new_admission_plan": refill_plan,
                "all_decode_plans": all_plans,
                "declared_widths_only": declared_widths_only,
                "no_serial_fallback": no_serial_fallback,
                "original_request_ids": list(original_ids),
                "cancelled_request_ids": list(cancelled_ids),
                "newcomer_request_ids": list(newcomer_ids),
                "original_reference_tokens": original_reference_tokens,
                "original_output_tokens": original_output_tokens,
                "newcomer_reference_tokens": newcomer_reference_tokens,
                "newcomer_output_tokens": newcomer_output_tokens,
                "original_tokens_exact": original_tokens_exact,
                "newcomer_tokens_exact": newcomer_tokens_exact,
                "state_kv_exact": state_exact,
                "state_kv_mismatches": {
                    "cancelled": cancelled_mismatches,
                    "survivors_middle_hole": survivor_hole_mismatches,
                    "survivors_final": survivor_final_mismatches,
                    "newcomers_final": newcomer_final_mismatches,
                    "inactive_sessions_during_middle_hole": inactive_hole_mismatches,
                },
                "finish_details": original_finish,
                "cancelled_session_ids": list(cancelled_session_ids),
                "newcomer_reclaimed_session_ids": list(
                    newcomer_reclaimed_session_ids
                ),
                "cancelled_sessions_reused_by_newcomers": session_reused,
                "routes": routes,
                "timeline": timeline,
                "final_active_request_ids": list(final_active),
                "final_available_sessions": final_available,
                "final_scheduler_active": scheduler_active,
                "elapsed_seconds": time.perf_counter() - started,
                "notes": [
                    "The gate preserves scheduler slot identity; no physical compaction is performed.",
                    "Every decode group must use only declared c1/c2/c4/c8 physical widths.",
                    "Tokens, Conv/GDN state, and all live BF16 KV bytes are compared with c1 checkpoints.",
                ],
            }
        finally:
            llm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--cancel-slots", type=int, nargs=2, default=(2, 10))
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--original-max-tokens", type=int, default=5)
    parser.add_argument("--newcomer-max-tokens", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    command_args = list(sys.argv[1:] if argv is None else argv)
    payload["command"] = shlex.join(
        [sys.executable, "scripts/gguf_arbitrary_c_lifecycle.py", *command_args]
    )
    payload["environment"] = {
        key: os.environ.get(key)
        for key in (
            "HIPENGINE_HIP_ARCH",
            "HIP_VISIBLE_DEVICES",
            "HIPENGINE_COMPILER_VERSION_FILE",
        )
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
