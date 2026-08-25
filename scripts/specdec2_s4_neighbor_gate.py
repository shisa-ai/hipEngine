#!/usr/bin/env python3
"""Controlled reject-vs-full-accept isolation gate for physical SPECDEC2 S4.

This is a correctness-only fault/control injection. It never supplies benchmark
performance evidence and never changes production dispatch: one packed target
result is rewritten after target execution so request 0 follows a full draft
chain while request 1 rejects at its root. The following unmodified speculative
pair must recover the exact strict trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hipengine import LLM
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.generation.engine_service import EngineService
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _controlled_token_rows(
    results: Sequence[Any],
    candidate_rows: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> list[Any]:
    """Force request 0 full-accept and request 1 root-reject target IDs."""

    output = list(results)
    if len(output) != 2 or len(candidate_rows) != 2:
        raise ValueError("controlled neighbor gate requires exactly two requests")
    for index, (result, candidates) in enumerate(
        zip(output, candidate_rows, strict=True)
    ):
        candidate_ids = tuple(int(value) for value in candidates)
        token_ids = list(int(value) for value in result.token_ids)
        if not candidate_ids or len(token_ids) != len(candidate_ids) + 1:
            raise ValueError("controlled neighbor rows do not align")
        if index == 0:
            token_ids[: len(candidate_ids)] = candidate_ids
        else:
            token_ids[0] = (candidate_ids[0] + 1) % int(vocab_size)
            if token_ids[0] == candidate_ids[0]:
                raise RuntimeError("controlled reject token did not diverge")
        output[index] = replace(result, token_ids=token_ids)
    return output


def _request(max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=("Write one short greeting.",),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    llm = LLM(
        str(args.model),
        backend="hip_gfx1151",
        execution_profile="strict",
        max_active_requests=2,
        max_sequence_length=256,
    )
    adapter = llm._get_text_generator()
    service = EngineService(adapter, idle_wait_seconds=0.001)
    original = Qwen35GGUFResidentSession.verify_target_blocks_batch
    controlled_calls = 0

    def controlled(self, jobs, *, stream: int = 0):
        nonlocal controlled_calls
        results = original(self, jobs, stream=stream)
        if controlled_calls > 0:
            return results
        candidate_rows: list[tuple[int, ...]] = []
        for job in jobs:
            candidate = job.get("candidate_token_ids_device")
            if candidate is None:
                raise RuntimeError("controlled neighbor gate requires device candidates")
            values = np.empty(candidate.shape, dtype=np.int32)
            copy_device_to_host(
                host_array_ptr(values),
                DeviceBuffer(candidate.ptr, values.nbytes),
                values.nbytes,
                runtime=self.runtime,
            )
            candidate_rows.append(tuple(int(value) for value in values))
        controlled_calls += 1
        return _controlled_token_rows(
            results,
            candidate_rows,
            vocab_size=int(self.runner.vocab_size),
        )

    Qwen35GGUFResidentSession.verify_target_blocks_batch = controlled
    try:
        handles = service.submit_speculative_children((_request(4), _request(4)))
        controlled_outputs = tuple(
            tuple(int(token) for token in handle.result(timeout=120).generated_token_ids)
            for handle in handles
        )
        controlled_snapshot = service.live_loop_snapshot()
        Qwen35GGUFResidentSession.verify_target_blocks_batch = original
        health_handles = service.submit_speculative_children((_request(5), _request(5)))
        health_outputs = tuple(
            tuple(int(token) for token in handle.result(timeout=120).generated_token_ids)
            for handle in health_handles
        )
        health_snapshot = service.live_loop_snapshot()
    finally:
        Qwen35GGUFResidentSession.verify_target_blocks_batch = original
        service.close()
        llm.close()

    recent = health_snapshot["runner"]["routes"]["recent_completed"]
    controlled_rows = [
        row for row in recent if int(row["request_id"]) in {0, 1}
    ]
    accepted = sorted(
        int(row["specdec2_mtp2_accepted_counts"][0])
        for row in controlled_rows
        if row["specdec2_mtp2_accepted_counts"]
    )
    expected = (271, 9419, 0, 2500, 628)
    passed = bool(
        controlled_calls == 1
        and accepted == [0, 2]
        and len(controlled_outputs) == 2
        and all(len(output) == 4 for output in controlled_outputs)
        and health_outputs == (expected, expected)
        and all(
            int(row["specdec2_mtp2_device_accept_calls"]) > 0
            and int(row["specdec2_mtp2_selected_commit_batch_calls"]) > 0
            for row in controlled_rows
        )
        and health_snapshot["loop"]["requests"]["active"] == 0
    )
    return {
        "schema": 1,
        "kind": "specdec2_s4_controlled_neighbor_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "model": str(args.model),
        "controlled_calls": controlled_calls,
        "controlled_outputs": controlled_outputs,
        "controlled_accepted_counts": accepted,
        "health_outputs": health_outputs,
        "controlled_snapshot": controlled_snapshot,
        "health_snapshot": health_snapshot,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
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
                "controlled_accepted_counts": payload[
                    "controlled_accepted_counts"
                ],
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
