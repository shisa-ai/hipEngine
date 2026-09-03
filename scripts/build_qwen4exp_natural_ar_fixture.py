#!/usr/bin/env python3
"""Build the PF-0 natural-text route-covering fixture in the canonical schema.

PF-0 (``docs/QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md`` section 6, queue row 1)
requires a fixture that is both natural (no token-level repetition) and
route-covering (every prompt reaches the ``rows >= 64`` Q8 MMQ policy).  The
2026-08-29 admission suite is natural but too short (11.1% route coverage) and
the canonical p512/p1024/p4096 fixture is route-covering but synthetic (short
prompts repeated at the token-ID level).

This builder renders the committed natural excerpts under
``benchmarks/fixtures/natural_sources/`` (provenance in that directory's
``PROVENANCE.md``) through the retained Qwen chat-prompt builder into the
canonical exact-token fixture schema.  Every case is a contiguous, unpadded
natural excerpt whose full chat prompt clears the 512-token route floor, so
the route dispatches on every prefill row.  Cases keep the four canonical
categories; per-case construction provenance and token-ID hashes are recorded
inside the fixture.

It performs no measurement and makes no claim.
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
DEFAULT_SOURCES = REPO_ROOT / "benchmarks" / "fixtures" / "natural_sources"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "fixtures" / "qwen4exp_natural_ar_pf0.json"

CANONICAL_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
SPANS_PER_CATEGORY = 3
ROUTE_FLOOR_TOKENS = 512

# Case ordering matches the canonical fixture: three cases per canonical
# category, category-major.
CASE_IDS = [
    f"{category}-{index}"
    for category in CANONICAL_CATEGORIES
    for index in range(1, SPANS_PER_CATEGORY + 1)
]


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        np.asarray(token_ids, dtype="<i8").tobytes()
    ).hexdigest()


def build(model_root: Path, decode_transitions: int, sources_dir: Path) -> dict:
    gguf = discover_gguf_files(model_root.resolve())[0]
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(gguf))

    case_paths = {p.stem: p for p in sorted(sources_dir.glob("*.txt"))}
    expected = set(CASE_IDS)
    observed = set(case_paths)
    if observed != expected:
        raise ValueError(
            "natural_sources must contain exactly one body file per case id "
            f"(missing: {sorted(expected - observed)}, "
            f"extra: {sorted(observed - expected)})"
        )

    cases = []
    for case_id in CASE_IDS:
        body_path = case_paths[case_id]
        body = body_path.read_text(encoding="utf-8")
        ids = [int(t) for t in build_chat_prompt(tokenizer, body)]
        if len(ids) < ROUTE_FLOOR_TOKENS:
            raise ValueError(
                f"case {case_id!r} is {len(ids)} tokens, below the "
                f"{ROUTE_FLOOR_TOKENS}-token route-covering floor"
            )
        cases.append(
            {
                "id": case_id,
                "category": case_id.rsplit("-", 1)[0],
                "prompt_tokens": len(ids),
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": _token_ids_sha256(ids),
                "unique_token_ratio": round(len(set(ids)) / len(ids), 4),
                "source_file": str(body_path.relative_to(REPO_ROOT)),
                "source_sha256": _sha256_path(body_path),
            }
        )

    return {
        "schema": 1,
        "kind": "qwen4exp_natural_ar_exact_token_fixture",
        "name": "Qwen3.8-Flash-Next PF-0 natural-text route-covering AR fixture",
        "categories": list(CANONICAL_CATEGORIES),
        "decode_transitions": int(decode_transitions),
        "source": {
            "provenance": "benchmarks/fixtures/natural_sources/PROVENANCE.md",
            "bodies": {
                case["source_file"]: case["source_sha256"] for case in cases
            },
            "chat_prompt_builder": (
                "scripts.gguf_mtp_bench.build_chat_prompt(reasoning='off')"
            ),
        },
        "model": {
            "id": "Qwen/Qwen3.8-Flash-Next",
            "quant": "Unsloth UD-Q4_K_XL",
            "tokenizer_source": str(gguf.name),
        },
        "construction": (
            "Three contiguous natural excerpts per canonical category "
            "(public-domain or liberally licensed material; see "
            "natural_sources/PROVENANCE.md), rendered through the retained "
            "Qwen chat-prompt builder at their natural length with no padding "
            "or repetition. Every case clears the rows >= 64 Q8 MMQ policy "
            "floor, so the route dispatches on every prefill row. Engines "
            "consume the committed token arrays directly."
        ),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--decode-transitions", type=int, default=128)
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    fixture = build(args.model_root, args.decode_transitions, args.sources_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(fixture["cases"]),
                "prompt_tokens_min": min(c["prompt_tokens"] for c in fixture["cases"]),
                "prompt_tokens_max": max(c["prompt_tokens"] for c in fixture["cases"]),
                "unique_token_ratio_min": min(
                    c["unique_token_ratio"] for c in fixture["cases"]
                ),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
