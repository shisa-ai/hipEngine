#!/usr/bin/env python3
"""Alternating AR/K0/MTP2 product soak with C8 and stream accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

from hipengine import LLM, SamplingParams
from hipengine.generation.registry import GenerationRequest


def _request(prompt: str, max_tokens: int = 3) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def _outputs(handles: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token) for token in handle.result(timeout=180).generated_token_ids)
        for handle in handles
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    llm = LLM(
        str(args.model),
        backend="hip_gfx1151",
        execution_profile=str(args.execution_profile),
        max_active_requests=4,
        max_sequence_length=256,
    )
    service = llm._get_text_generator()
    greeting = _request("Write one short greeting.")
    farewell = _request("Write one short farewell.")
    wave_results: list[dict[str, Any]] = []
    total_requests = 0
    started = time.perf_counter()
    try:
        ar_reference = _outputs(service.submit_children((greeting, farewell)))
        references = {
            "greeting": ar_reference[0],
            "farewell": ar_reference[1],
        }
        for wave in range(int(args.waves)):
            requests = (greeting, farewell, greeting, farewell)
            expected = (
                references["greeting"],
                references["farewell"],
                references["greeting"],
                references["farewell"],
            )
            if wave % 5 == 4:
                first = service.submit_children(requests[:2])
                second = service.submit_speculative_children(requests[2:])
                outputs = (*_outputs(first), *_outputs(second))
                route = "mixed_ar_mtp"
            elif wave % 2 == 0:
                outputs = _outputs(service.submit_speculative_children(requests))
                route = "explicit_mtp_c4"
            else:
                outputs = _outputs(service.submit_children(requests))
                route = "automatic_k0_c4"
            total_requests += 4
            wave_results.append(
                {
                    "wave": wave,
                    "route": route,
                    "outputs": outputs,
                    "exact": outputs == expected,
                }
            )
        c8_requests = tuple(greeting if index % 2 == 0 else farewell for index in range(8))
        c8_outputs = _outputs(service.submit_children(c8_requests))
        c8_expected = tuple(
            references["greeting"] if index % 2 == 0 else references["farewell"]
            for index in range(8)
        )
        total_requests += 8

        blocking = service.submit_speculative_child(_request("Write one short greeting.", 5)).result(
            timeout=180
        )
        stream_chunks = tuple(
            service.stream_speculative_mtp_detailed(
                _request("Write one short greeting.", 5)
            )
        )
        streamed_text = "".join(chunk.text for chunk in stream_chunks)
        final_stream_ids = next(
            (
                tuple(int(token) for token in chunk.generated_token_ids)
                for chunk in reversed(stream_chunks)
                if chunk.generated_token_ids is not None
            ),
            (),
        )
        total_requests += 2
        snapshot = service.live_loop_snapshot()
        runner = service.inner._runner
        mtp_adapter = runner._resolved_mtp2_adapter()
        request_scoped_pages = int(snapshot["runner"]["kv_pool"]["refcounted_pages"]) - int(
            snapshot["runner"]["kv_pool"]["pinned_pages"]
        )
    finally:
        llm.close()

    passed = bool(
        total_requests >= 100
        and all(row["exact"] for row in wave_results)
        and c8_outputs == c8_expected
        and tuple(int(token) for token in blocking.generated_token_ids) == final_stream_ids
        and streamed_text == blocking.text
        and snapshot["loop"]["requests"]["active"] == 0
        and snapshot["loop"]["requests"]["pending"] == 0
        and snapshot["runner"]["model_runner"]["active_requests"] == 0
        and snapshot["engine_service"]["active_children"] == 0
        and request_scoped_pages == 0
        and (mtp_adapter is None or not mtp_adapter._states)
        and (mtp_adapter is None or mtp_adapter._active_claims is None)
    )
    return {
        "schema": 1,
        "kind": "specdec2_s6_alternating_soak",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "execution_profile": str(args.execution_profile),
        "waves": int(args.waves),
        "total_requests": total_requests,
        "references": references,
        "wave_results": wave_results,
        "c8_outputs": c8_outputs,
        "c8_expected": c8_expected,
        "blocking_ids": tuple(int(token) for token in blocking.generated_token_ids),
        "stream_ids": final_stream_ids,
        "stream_chunk_count": len(stream_chunks),
        "stream_text_exact": streamed_text == blocking.text,
        "request_scoped_pages": request_scoped_pages,
        "final_snapshot": snapshot,
        "elapsed_seconds": time.perf_counter() - started,
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
    parser.add_argument("--waves", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if args.waves < 23:
        raise SystemExit("waves must be >=23 so the packet covers 100+ requests")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["elapsed_seconds"],
                "passed": payload["passed"],
                "total_requests": payload["total_requests"],
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
