#!/usr/bin/env python3
"""The promoted route through the public ``LLM.generate`` surface, two profiles.

Every other harness for this change drives the runner directly. This one takes
the path a user takes -- ``hipengine.LLM(...).generate_detailed(...)`` with a
chat-templated prompt -- under ``production`` and under ``strict``, and reports
the decode-selection counter the launcher publishes, so the row says which route
the public call actually ran rather than which one was intended.

Run it once per profile; the two token sequences are the product-level
comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
PROMPT = "Explain in one sentence why the sky is blue."


def chat(text: str) -> str:
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="repeat the prompt sentence; the decode route needs a key count "
             "past the split threshold (1024) to leave the block kernel",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import hipengine
    from hipengine.llm import SamplingParams
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    llm = hipengine.LLM(model=str(MODEL), max_sequence_length=args.max_sequence_length)
    try:
        prompt = chat(PROMPT * args.repeat)
        generator = llm._get_text_generator()
        prompt_tokens = len(generator.tokenize(prompt)) if hasattr(generator, "tokenize") else None
        out = llm.generate_detailed(
            [prompt],
            SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
        )
        first = out[0]
        token_ids = list(getattr(first, "token_ids", None) or ())
        row = {
            "kind": "gemma4_generate_route_smoke",
            "profile_env": os.environ.get("HIPENGINE_EXECUTION_PROFILE"),
            "max_tokens": args.max_tokens,
            "max_sequence_length": args.max_sequence_length,
            "repeat": args.repeat,
            "prompt_tokens": prompt_tokens,
            "text": getattr(first, "text", str(first)),
            "token_ids": token_ids,
            "token_ids_sha256": hashlib.sha256(
                b"".join(int(t).to_bytes(4, "little", signed=True) for t in token_ids)
            ).hexdigest(),
            "decode_selection": attention.decode_selection(),
            "finish_reason": str(getattr(first, "finish_reason", "")),
        }
    finally:
        llm.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
