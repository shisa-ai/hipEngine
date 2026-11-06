#!/usr/bin/env python3
"""Muse Glimmer AR prefill/decode bench matched to the P0 llama-bench protocol.

Workloads: 512/128, 1024/128, 4096/128 (prompt tokens / generated tokens).
Timing boundary: the resident session's synchronized prefill and decode-step
walls (the model-step boundary llama-bench reports as pp/tg); sampling and
detokenization are outside the timed window. Warmup prebuilds every kernel;
the timed process never spawns hipcc. >=3 measured repetitions, median reported.

This is the Gate A measurement instrument (campaign MUSE-GLIMMER.md). It is a
diagnostic until a retained rollup attaches the correctness gate + provenance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [str(REPO_ROOT), *(e for e in sys.path if e != REPO_ROOT)]

import numpy as np  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
SEED_TEXT = (
    "The study of inference engines for large language models spans computer "
    "architecture, numerical analysis, and compiler design. Modern accelerators "
    "expose wide vector units and high-bandwidth memory, and the task of the "
    "engine is to keep those units fed while preserving the numerical contract "
    "of the model. Quantization reduces the bytes moved per weight, attention "
    "patterns determine the cache traffic, and kernel fusion removes the round "
    "trips that would otherwise dominate the step latency. "
)


def build_prompt_ids(tokenizer, n_tokens: int) -> list[int]:
    """A natural-text prompt of exactly n_tokens (llama-bench-style synthetic)."""

    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(tokenizer.encode(SEED_TEXT, add_special_tokens=False))
    return ids[:n_tokens]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--workloads", default="512/128,1024/128,4096/128")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    import hipengine  # noqa: F401
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.loading.gguf import scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF

    workloads = []
    for spec in args.workloads.split(","):
        pp, tg = spec.split("/")
        workloads.append((int(pp), int(tg)))

    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=args.model, weight_index=scan_gguf(args.model),
        model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=args.max_context,
    )
    session = gen._ensure_session()
    tokenizer = gen.tokenizer

    # Warmup: prebuild every kernel on a short pass outside the timed window.
    warm = build_prompt_ids(tokenizer, 32)
    session.reset()
    session.prefill(warm)
    for _ in range(4):
        session.decode_step(13796)

    rows = []
    artifact = {
        "schema": 1,
        "kind": "muse_glimmer_hipengine_ar_bench",
        "status": "diagnostic",
        "performance_claim": False,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": str(args.model),
        "quant": "gguf_q4_k_m",
        "backend": "hip_gfx1151",
        "timing_boundary": "session prefill / decode-step synchronized wall; sampling excluded",
        "repeats": args.repeats,
        "warmup": "32-token prefill + 4 decode steps (kernel prebuild)",
        "execution_path": session.execution_path(),
        "rows": rows,
    }
    for pp, tg in workloads:
        prompt = build_prompt_ids(tokenizer, pp)
        prefill_samples, decode_samples = [], []
        for rep in range(args.repeats):
            session.reset()
            t0 = time.perf_counter()
            logits = session.prefill(prompt)
            t1 = time.perf_counter()
            token = int(np.argmax(logits))
            for _ in range(tg):
                logits = session.decode_step(token)
                token = int(np.argmax(logits))
            t2 = time.perf_counter()
            prefill_samples.append(t1 - t0)
            decode_samples.append(t2 - t1)
        pp_med = statistics.median(prefill_samples)
        tg_med = statistics.median(decode_samples)
        row = {
            "prompt_tokens": pp,
            "gen_tokens": tg,
            "prefill_tok_s": pp / pp_med,
            "decode_tok_s": tg / tg_med,
            "prefill_s_samples": prefill_samples,
            "decode_s_samples": decode_samples,
            "prefill_s_median": pp_med,
            "decode_s_median": tg_med,
        }
        rows.append(row)
        print(f"{pp}/{tg}: prefill {row['prefill_tok_s']:.3f} tok/s, "
              f"decode {row['decode_tok_s']:.3f} tok/s "
              f"(medians of {args.repeats})", flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(artifact, indent=1))

    if args.output:
        print(f"wrote {args.output}", flush=True)
    gen.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())