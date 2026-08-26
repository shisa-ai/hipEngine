#!/usr/bin/env python3
"""Stagger, compact, substitute a neighbor, and prove SPECDEC2 isolation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Sequence

from hipengine import LLM
from hipengine.generation.engine_service import EngineService
from hipengine.generation.registry import GenerationRequest


def _request(max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=("Write one short greeting.",),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def _mtp_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in snapshot["runner"]["routes"]["recent_completed"]
        if row.get("specdec2_mtp2_used")
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    llm = LLM(
        str(args.model),
        backend="hip_gfx1151",
        execution_profile=str(args.execution_profile),
        max_active_requests=2,
        max_sequence_length=256,
    )
    adapter = llm._get_text_generator()
    adapter.reconfigure_engine_loop(
        replace(adapter._loop.config, prefill_decode_policy="protect_ttft")
    )
    owns_service = not isinstance(adapter, EngineService)
    service = (
        EngineService(adapter, idle_wait_seconds=0.001)
        if owns_service
        else adapter
    )
    try:
        short, survivor = service.submit_speculative_children(
            (_request(2), _request(64))
        )
        short_output = tuple(
            int(token) for token in short.result(timeout=120).generated_token_ids
        )
        before = service.live_loop_snapshot()
        moves = tuple(service.compact())
        after_compact = service.live_loop_snapshot()
        substitute = service.submit_speculative_child(_request(3))
        substitute_output = tuple(
            int(token) for token in substitute.result(timeout=120).generated_token_ids
        )
        after_substitute = service.live_loop_snapshot()
        cancelled = survivor.cancel()
        try:
            survivor.result(timeout=120)
            survivor_error = None
        except BaseException as error:
            survivor_error = type(error).__name__
        final = service.live_loop_snapshot()
    finally:
        if owns_service:
            service.close()
        llm.close()

    move_payload = [asdict(move) for move in moves]
    survivor_request_id = 1
    substitute_rows = [
        row for row in _mtp_rows(after_substitute) if int(row["request_id"]) == 2
    ]
    passed = bool(
        short_output == (271, 9419)
        and before["loop"]["physical_bucket"]["slot_to_request"] == [None, 1]
        and move_payload
        and move_payload[0] == {
            "request_id": survivor_request_id,
            "old_slot": 1,
            "new_slot": 0,
        }
        and after_compact["loop"]["physical_bucket"]["slot_to_request"] == [1, None]
        and substitute_output == (271, 9419, 0)
        and substitute_rows
        and int(substitute_rows[-1]["specdec2_mtp2_device_accept_calls"]) > 0
        and int(substitute_rows[-1]["specdec2_mtp2_selected_commit_batch_calls"]) > 0
        and cancelled
        and survivor_error == "GenerationCancelled"
        and final["loop"]["requests"]["active"] == 0
    )
    return {
        "schema": 1,
        "kind": "specdec2_s4_compaction_neighbor_substitution_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "execution_profile": str(args.execution_profile),
        "short_output": short_output,
        "moves": move_payload,
        "substitute_output": substitute_output,
        "survivor_cancelled": cancelled,
        "survivor_error": survivor_error,
        "before": before,
        "after_compact": after_compact,
        "after_substitute": after_substitute,
        "final": final,
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
    print(json.dumps({"moves": payload["moves"], "passed": payload["passed"]}))
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
