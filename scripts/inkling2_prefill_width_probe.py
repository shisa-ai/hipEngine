#!/usr/bin/env python3
"""Time packed prefill at several block widths for the Inkling 2 GGUF route.

One width per process invocation; run the sweep from the shell. Reports
time-to-first-text, total stream time, and generated token ids for a fixed
575-token prompt at ``execution_profile=fast`` (the campaign's prefill path).
The generated ids are comparable across widths: any mismatch means the width
change altered outputs, which is a correctness failure, not a speed result.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import closing

from hipengine import LLM, SamplingParams
from hipengine.generation.inkling2_gguf import Inkling2GGUFGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=575)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--backend", default="hip_gfx1100")
    args = parser.parse_args()

    if args.width >= 1:
        Inkling2GGUFGenerator.prefill_block_width = args.width
        Inkling2GGUFGenerator.engine_loop_config_defaults = {
            **Inkling2GGUFGenerator.engine_loop_config_defaults,
            "max_prefill_chunk_tokens": args.width,
            "round_prefill_token_budget": args.width,
        }

    prompt_text = (
        "Explain why the day gets longer in spring, using plain language and a few practical examples. "
        "Distinguish what an observer sees from the motion of the Earth. "
    )

    llm = LLM(model=args.model, backend=args.backend, execution_profile="fast",
              max_sequence_length=args.tokens + args.output_tokens,
              prefix_cache="off", speculative_mtp_serving="off")
    try:
        seed = llm.tokenize(prompt_text)
        ids = (seed * ((args.tokens + len(seed) - 1) // len(seed)))[: args.tokens]
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        started = time.perf_counter()
        first_text_at = None
        output_ids = None
        diagnostics = None
        with closing(llm.stream_detailed(ids, sampling)) as stream:
            for chunk in stream:
                if first_text_at is None and chunk.text:
                    first_text_at = time.perf_counter()
                if getattr(chunk, "generated_token_ids", None) is not None:
                    output_ids = list(chunk.generated_token_ids)
                if getattr(chunk, "finish_details", None) and getattr(chunk.telemetry, "diagnostics", None):
                    diagnostics = dict(chunk.telemetry.diagnostics)
        total = time.perf_counter() - started
        print(json.dumps({
            "width": args.width,
            "prompt_tokens": len(ids),
            "seconds_to_first_text": (first_text_at - started) if first_text_at else None,
            "total_seconds": total,
            "output_ids": output_ids,
            "diagnostics": diagnostics,
        }))
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
