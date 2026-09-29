"""Compare a saved Q8 T16 WMMA library with the current build on Gemma 4.

This diagnostic keeps one model residency and alternates library order. Both
arms use the public dense-WMMA route; the baseline library must be captured
before the overflow repair. It also checks public generation and records the
actual resolved kernel variants. It does not certify task quality.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gemma4_campaign_bench import (
    _provenance, _public_wall, _resolve_generator, exact_prompt_ids,
    resolve_artifact, run_instrumented,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--prompt-lengths", default="512,1024,2048,4096")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(value) for value in args.prompt_lengths.split(",")]
    if args.pairs < 1 or args.output_tokens < 1 or not lengths or min(lengths) < 1:
        parser.error("prompt lengths, output tokens, and pairs must be positive")
    import numpy as np
    import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill as kernel
    import hipengine.runtime.gguf_linear as linear

    build = kernel.build_gguf_q8_0_t16_prefill
    resolve = linear.resolve
    libraries = {"before": ctypes.CDLL(str(args.baseline_library.resolve())), "after": build()}
    variants = set()

    def recording_resolve(**kwargs):
        variants.add((kwargs.get("quant"), kwargs.get("variant")))
        return resolve(**kwargs)

    linear.resolve = recording_resolve
    artifact = resolve_artifact()
    llm, runner, loading = _resolve_generator(artifact, max(lengths) + args.output_tokens + 32)
    generator = llm._get_text_generator()
    report = {
        "command": sys.argv, "host": platform.node(), "provenance": _provenance(artifact),
        "loading": loading, "baseline_library_sha256": hashlib.sha256(args.baseline_library.read_bytes()).hexdigest(),
        "scope": "same-residency diagnostic; unchanged WMMA route, not task-quality certification",
        "shapes": [],
    }
    try:
        for length in lengths:
            prompt = exact_prompt_ids(generator.tokenize, length)
            arms = {key: [] for key in libraries}
            public = {}
            for label, library in libraries.items():
                kernel.build_gguf_q8_0_t16_prefill = lambda **kwargs: library
                run_instrumented(runner, prompt, args.output_tokens)
            for pair in range(args.pairs):
                order = ("before", "after") if pair % 2 == 0 else ("after", "before")
                for label in order:
                    library = libraries[label]
                    kernel.build_gguf_q8_0_t16_prefill = lambda **kwargs: library
                    arms[label].append(run_instrumented(runner, prompt, args.output_tokens))
            for label, library in libraries.items():
                kernel.build_gguf_q8_0_t16_prefill = lambda **kwargs: library
                variants.clear()
                public[label] = _public_wall(llm, prompt, args.output_tokens)
                public[label]["variants"] = sorted(variants)
            expected = arms["before"][0]["generated_token_ids"]
            equal = all(row["generated_token_ids"] == expected for rows in arms.values() for row in rows)
            equal = equal and all(row["generated_token_ids"] == expected for row in public.values())
            raw_hashes = {}
            for label, library in libraries.items():
                kernel.build_gguf_q8_0_t16_prefill = lambda **kwargs: library
                runner.reset()
                hashes = []
                for tokens in [prompt] + [[token] for token in expected[:-1]]:
                    logits = np.asarray(runner.forward(tokens, apply_softcap=False))
                    if not np.isfinite(logits).all():
                        raise AssertionError(f"non-finite raw logits in {label} at {length}")
                    hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
                raw_hashes[label] = hashes
            raw_equal = raw_hashes["before"] == raw_hashes["after"]
            row = {"prompt_tokens": length, "arms": arms, "public": public,
                   "tokens_identical": equal, "raw_logit_hashes": raw_hashes,
                   "raw_logits_identical": raw_equal}
            report["shapes"].append(row)
            print(length, {key: [round(length / r["prefill_s"], 1) for r in rows] for key, rows in arms.items()}, "tokens_identical", equal, flush=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            if not equal or not raw_equal:
                raise AssertionError(f"generated tokens or raw logits differ at prompt {length}")
    finally:
        kernel.build_gguf_q8_0_t16_prefill = build
        linear.resolve = resolve
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
