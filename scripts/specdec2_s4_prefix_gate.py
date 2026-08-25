#!/usr/bin/env python3
"""Qualify SPECDEC2 prefix restore/COW with pre-mutation K0 fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from hipengine import LLM
from hipengine.generation.registry import GenerationRequest


def _request(tokens: Sequence[int], max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=(tuple(int(token) for token in tokens),),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _completed_row(snapshot: dict[str, Any], request_id: int) -> dict[str, Any]:
    rows = snapshot["runner"]["routes"]["recent_completed"]
    matches = [row for row in rows if int(row["request_id"]) == int(request_id)]
    if not matches:
        raise RuntimeError(f"completed request {request_id} is missing")
    return matches[-1]


def run(args: argparse.Namespace) -> dict[str, Any]:
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    if oracle.get("passed") is not True:
        raise ValueError("independent prefix oracle did not pass")
    expected_source = int(
        oracle["prefill_oracle"]["source_predicted_token_id"]
    )
    expected_continuation = int(
        oracle["prefill_oracle"]["reference_predicted_token_id"]
    )
    prefix = (int(args.prefix_token_id),) * int(args.prefix_tokens)
    continuation = (*prefix, int(args.suffix_token_id))
    max_sequence_length = len(continuation) + int(args.max_tokens) + 4
    llm = LLM(
        str(args.model),
        backend="hip_gfx1151",
        execution_profile="strict",
        max_active_requests=2,
        max_sequence_length=max_sequence_length,
        prefix_cache="radix",
    )
    try:
        llm.prepare(max_sequence_length=max_sequence_length)
        service = llm._get_text_generator()
        source = service.submit_speculative_child(
            _request(prefix, 2)
        ).result(timeout=180)
        source_snapshot = service.live_loop_snapshot()
        cached = service.submit_speculative_child(
            _request(continuation, int(args.max_tokens))
        ).result(timeout=180)
        cached_snapshot = service.live_loop_snapshot()
    finally:
        llm.close()

    source_ids = tuple(int(token) for token in source.generated_token_ids)
    cached_ids = tuple(int(token) for token in cached.generated_token_ids)
    source_row = _completed_row(cached_snapshot, 0)
    cached_row = _completed_row(cached_snapshot, 1)
    cache = cached_snapshot["runner"]["prefix_cache"]
    mutation_keys = (
        "specdec2_mtp2_cycles",
        "specdec2_mtp2_candidate_device_handoffs",
        "specdec2_mtp2_candidate_d2h_after_target",
        "specdec2_mtp2_device_accept_calls",
        "specdec2_mtp2_selected_commit_batch_calls",
        "specdec2_mtp2_proposal_batch_calls",
        "specdec2_mtp2_target_batch_calls",
        "specdec2_mtp2_k0_catchups",
    )
    no_provider_mutation = all(int(cached_row[key]) == 0 for key in mutation_keys)
    passed = bool(
        source_ids
        and source_ids[0] == expected_source
        and cached_ids
        and cached_ids[0] == expected_continuation
        and cached_row["prefix_lookup"] is True
        and cached_row["prefix_snapshot_hit"] is True
        and cached_row["prefix_source_kind"] == "completed_snapshot"
        and int(cached_row["prefix_matched_tokens"]) == int(args.prefix_tokens)
        and int(cached_row["prefix_reused_tokens"]) == int(args.prefix_tokens)
        and int(cached_row["prefix_state_clone_bytes"]) > 0
        and int(cache["snapshot_hits"]) > 0
        and int(cache["reused_tokens"]) >= int(args.prefix_tokens)
        and no_provider_mutation
        and source_snapshot["loop"]["requests"]["active"] == 0
        and cached_snapshot["loop"]["requests"]["active"] == 0
    )
    return {
        "schema": 1,
        "kind": "specdec2_s4_prefix_restore_cow_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "model": str(args.model),
        "oracle": str(args.oracle),
        "source_ids": source_ids,
        "cached_ids": cached_ids,
        "expected_source_first_token": expected_source,
        "expected_continuation_first_token": expected_continuation,
        "source_row": source_row,
        "cached_row": cached_row,
        "prefix_cache": cache,
        "no_provider_mutation": no_provider_mutation,
        "source_snapshot": source_snapshot,
        "cached_snapshot": cached_snapshot,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--prefix-token-id", type=int, default=9707)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-token-id", type=int, default=9708)
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
                "cached_ids": payload["cached_ids"],
                "no_provider_mutation": payload["no_provider_mutation"],
                "passed": payload["passed"],
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
