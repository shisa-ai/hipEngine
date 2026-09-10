#!/usr/bin/env python3
"""Build the 18-prompt admission-suite fixture in the plane-gate fixture schema.

The 2026-08-29 Qwen4Exp production admissions were measured over the 18-prompt
calibration suite (``benchmarks/prompts/mtpbench-code-general-ja.jsonl`` plus
``benchmarks/prompts/gdn-prefill-category-heldouts.jsonl``), the same suite the
2026-08-16 threshold calibration and ``scripts/qwen4exp_layer2_profile_gate.py``
consume.  ``scripts/execution_profile_q8_mmq_plane_gate.py`` consumes the
canonical exact-token fixture schema instead, so this script renders the same
18 prompts into that schema.

Emitting both fixtures in one schema makes the prompt suite the only variable
between the 2026-08-29 admission basis and the 2026-09-03 canonical-fixture
gate runs.  It performs no measurement and makes no claim.
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
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt

DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/fixtures/qwen4exp_admission_suite_18prompt.json"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(model_root: Path, decode_transitions: int) -> dict:
    gguf = discover_gguf_files(model_root.resolve())[0]
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(gguf))
    rows = _load_suites(list(DEFAULT_PROMPTS))
    cases = []
    for row in rows:
        ids = [int(t) for t in build_chat_prompt(tokenizer, str(row["prompt"]))]
        cases.append(
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "prompt_tokens": len(ids),
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": hashlib.sha256(
                    np.asarray(ids, dtype="<i8").tobytes()
                ).hexdigest(),
                "suite": str(Path(row["suite"]).relative_to(REPO_ROOT)),
            }
        )
    return {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_exact_token_fixture",
        "name": "Qwen3.8-Flash-Next 2026-08-29 admission suite (18 prompts, natural length)",
        "categories": sorted({case["category"] for case in cases}),
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
            "The 18 calibration/heldout prompts are rendered through the retained "
            "Qwen chat-prompt builder at their natural length with no padding or "
            "repetition, so prompt_tokens is the true prompt length. This is the "
            "prompt material behind the 450-row 2026-08-29 admission packets."
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
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
