#!/usr/bin/env python3
"""Build the long-form category-heldout fixture in the plane-gate fixture schema.

Why this exists
---------------
``X-7`` needs category-heldout numerical evidence for the promoted 16-47 scope
before that scope can be extended, and its blocker is that the 18-prompt
admission suite cannot engage a prefill route at all: its longest prompt is 71
tokens against ``dense_wide``'s 257-row floor, so a heldout run over it is a
control between two runs of the same kernel, not qualification.

That blocker was verified genuine rather than assumed. The floor is read from
``_q8_dense_wide_dispatch``'s ``rows <= 256``, and ``rows`` at that guard is
``len(tokens)`` (``laguna_gguf_runner.py:5059``, ``_execute_rows``) - the real
prefill token count, not the scratch row count that ``allocate_scratch`` sizes to
``min(chunk_size, max(768, prompt_tokens + 1))``. So ``prompt_tokens`` is a
faithful proxy for what the guard sees.

This script renders ``benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl`` -
four natural long tasks, one per category, all above that floor - into the same
schema, giving the heldout protocol an input that can actually exercise a prefill
route.

Why a sibling rather than an edit
---------------------------------
The admission suite is shared: the same 18 prompts are what
``scripts/qwen4exp_layer2_profile_gate.py`` and the 2026-08-16 threshold
calibration consume, and
``scripts/execution_profile_q8_wmma_prefill_layers_gate.py`` hardcodes both its
SHA256 and its expected case-id set. Extending it in place would change a shared
fixture and invalidate every artifact citing that hash. A sibling fixture leaves
the admission protocol's identity intact and gives the gate a second,
separately-identified heldout protocol.

It performs no measurement and makes no claim. Every prompt is rendered at its
natural length with no padding or repetition, so ``prompt_tokens`` is the true
prompt length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import discover_gguf_files, scan_gguf
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_bench import build_chat_prompt

DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_PROMPTS = [REPO_ROOT / "benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl"]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/fixtures/qwen4exp_longform_heldout_4prompt.json"

# The four categories the canonical protocol requires coverage of. A long-form
# suite that silently dropped one would pass a "categories covered" count while
# leaving a category unqualified, so the count is asserted against this set
# rather than checked as a bare length.
REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _user_text(row: dict) -> str:
    for message in row.get("messages") or ():
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError(f"row {row.get('id')!r} has no user message")


def build(model_root: Path, decode_transitions: int) -> dict:
    gguf = discover_gguf_files(model_root.resolve())[0]
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(gguf))
    cases = []
    for path in DEFAULT_PROMPTS:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids = [int(t) for t in build_chat_prompt(tokenizer, _user_text(row))]
            cases.append(
                {
                    "id": str(row["id"]),
                    "category": str(row["category"]),
                    "prompt_tokens": len(ids),
                    "prompt_token_ids": ids,
                    "prompt_token_ids_sha256": hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "suite": str(path.relative_to(REPO_ROOT)),
                }
            )
    categories = sorted({case["category"] for case in cases})
    if categories != sorted(REQUIRED_CATEGORIES):
        raise SystemExit(
            f"long-form suite covers {categories}, expected "
            f"{sorted(REQUIRED_CATEGORIES)}; a missing category would leave it "
            "unqualified while still reporting full coverage of what it ran"
        )
    return {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_exact_token_fixture",
        "name": (
            "Qwen3.8-Flash-Next long-form category heldout (4 prompts, one per "
            "category, natural length)"
        ),
        "categories": categories,
        "decode_transitions": int(decode_transitions),
        "source": {
            "prompt_suites": {
                str(path.relative_to(REPO_ROOT)): _sha256_path(path)
                for path in DEFAULT_PROMPTS
            },
            "chat_prompt_builder": "scripts.gguf_mtp_bench.build_chat_prompt(reasoning='off')",
        },
        "model": {
            "id": "Qwen/Qwen3.8-Flash-Next",
            "quant": "Unsloth UD-Q4_K_XL",
            "tokenizer_source": str(gguf.name),
        },
        "construction": (
            "The four natural long tasks are rendered through the retained Qwen "
            "chat-prompt builder at their natural length with no padding or "
            "repetition, so prompt_tokens is the true prompt length. The suite "
            "exists because the 18-prompt admission suite tops out at 71 tokens "
            "and cannot reach a prefill route's row floor, which made every "
            "heldout run over it a control rather than qualification. Cases are "
            "disjoint from the canonical 12-case fixture."
        ),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--decode-transitions", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    fixture = build(args.model_root, args.decode_transitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    lengths = [case["prompt_tokens"] for case in fixture["cases"]]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(fixture["cases"]),
                "prompt_tokens_min": min(lengths),
                "prompt_tokens_max": max(lengths),
                "categories": fixture["categories"],
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
