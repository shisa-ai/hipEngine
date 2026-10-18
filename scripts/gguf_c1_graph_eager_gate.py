#!/usr/bin/env python3
"""Check low-level GGUF C1 graph replay against eager on category and heldout prompts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.loading.gguf import scan_gguf
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen38_gfx1151_readme_sweep import verify_graph_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--max-sequence-length", type=int, default=4226)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.decode_steps < 32 or args.max_sequence_length <= args.decode_steps:
        parser.error("requires at least 32 decode steps and a larger session capacity")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    rows = []
    with Qwen35GGUFResidentSession(
        args.model, backend=args.backend, max_sequence_length=args.max_sequence_length,
        use_wmma_prefill=True, use_gemv_decode=True,
        compiler_version=Path(compiler_file).read_text() if compiler_file else None,
    ) as session:
        for prompt in _load_suites(DEFAULT_PROMPTS):
            tokens = build_chat_prompt(tokenizer, str(prompt["prompt"]))
            if len(tokens) + args.decode_steps + 1 >= args.max_sequence_length:
                parser.error(f"prompt {prompt['id']} exceeds the configured capacity")
            row = verify_graph_prompt(session, tokens, steps=args.decode_steps)
            rows.append(dict(row, prompt_id=prompt["id"], category=prompt["category"]))
            print(prompt["id"], row["passed"], flush=True)
    payload = {
        "kind": "gguf_c1_graph_eager_gate",
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "provenance": collect_artifact_provenance(
            repo_root=ROOT, configured_backend=args.backend,
            model_path=args.model, quant="gguf_q4_k_m", kv_dtype="bf16",
            command=[sys.executable, *sys.argv], build_profile="low_level_c1_readme",
            timing_protocol="correctness_only", warmups=0, repetitions=1,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
