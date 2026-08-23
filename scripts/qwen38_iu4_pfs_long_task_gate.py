#!/usr/bin/env python3
"""Bounded five-category 4K task gate for the explicit IU4 FFN product."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from hipengine.loading.gguf import load_gguf_index
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.runtime.prefill import PrefillConfig
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.qwen35_paro_kv_quality_smoke import (
    _build_prompt_tokens,
    _decode,
    _load_suite,
    _score_output,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
DEFAULT_SUITE = Path("benchmarks/prompts/kv-int8-long-context-smoke.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pfs", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run(session, product, *, candidate: bool, prompt: list[int], max_new_tokens: int, tokenizer, task):
    session.reset()
    session.runner._iu4_ffn_product = product if candidate else None
    before_launches = product.launch_count
    before_fallbacks = product.fallback_count
    started = time.perf_counter()
    current = session.prefill(
        prompt,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    output_ids = [int(current.token_id)]
    eos_ids = {248044, 248046}
    for _ in range(max_new_tokens - 1):
        if int(current.token_id) in eos_ids:
            break
        current = session.step(int(current.token_id), return_logits=False)
        output_ids.append(int(current.token_id))
    session.runtime.device_synchronize()
    elapsed = time.perf_counter() - started
    text = _decode(tokenizer, output_ids)
    return {
        "output_token_ids": output_ids,
        "output_text": text,
        "score": _score_output(text, [str(item) for item in task["expected"]]),
        "elapsed_seconds": elapsed,
        "route": {
            "launches": product.launch_count - before_launches,
            "fallbacks": product.fallback_count - before_fallbacks,
        },
    }


def main() -> int:
    args = _parse_args()
    if min(args.context_tokens, args.max_new_tokens) <= 0:
        raise ValueError("context/max-new-tokens must be positive")
    compiler_version = None
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()
    tasks = _load_suite(args.suite)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(args.model))

    class _Encoded:
        def __init__(self, ids):
            self.ids = ids

    class _TokenizerAdapter:
        def encode(self, text):
            encoded = tokenizer.encode(text)
            return _Encoded(encoded if isinstance(encoded, list) else encoded.ids)

        def decode(self, token_ids, **kwargs):
            return tokenizer.decode(token_ids, **kwargs)

    tokenizer_adapter = _TokenizerAdapter()
    expanded = {
        str(task["id"]): _build_prompt_tokens(
            tokenizer_adapter,
            task,
            context_tokens=args.context_tokens,
        )
        for task in tasks
    }
    rows = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend="hip_gfx1151",
        max_sequence_length=args.context_tokens + args.max_new_tokens + 8,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        use_wmma_prefill=True,
        prefill_config=PrefillConfig(linear_chunk_size=2048, moe_chunk_size=2048),
        iu4_ffn_pfs_path=args.pfs,
    ) as session:
        product = session.runner._iu4_ffn_product
        if product is None:
            raise RuntimeError("IU4 FFN product was not loaded")
        for task in tasks:
            prompt, metadata = expanded[str(task["id"])]
            control = _run(
                session,
                product,
                candidate=False,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer_adapter,
                task=task,
            )
            candidate = _run(
                session,
                product,
                candidate=True,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer_adapter,
                task=task,
            )
            rows.append(
                {
                    "id": str(task["id"]),
                    "category": str(task["category"]),
                    **metadata,
                    "control": control,
                    "candidate": candidate,
                    "candidate_noninferior": (
                        bool(candidate["score"]["passed"])
                        or not bool(control["score"]["passed"])
                    ),
                }
            )
        session.runner._iu4_ffn_product = product
    all_candidate_pass = all(bool(row["candidate"]["score"]["passed"]) for row in rows)
    all_noninferior = all(bool(row["candidate_noninferior"]) for row in rows)
    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "qwen38_gfx1151_kairic_pfs_iu4_long_task_gate",
        "status": "accepted" if all_candidate_pass and all_noninferior else "rejected_correctness",
        "performance_claim": False,
        "model": str(args.model.resolve()),
        "pfs": str(args.pfs.resolve()),
        "suite": str(args.suite),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "context_tokens": args.context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "rows": rows,
        "gates": {
            "all_candidate_tasks_pass": all_candidate_pass,
            "all_candidate_noninferior": all_noninferior,
            "runtime_default_authorized": False,
        },
        "software": {
            "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "tracked_dirty": subprocess.check_output(
                ("git", "status", "--short", "--untracked-files=no"), cwd=ROOT, text=True
            ).splitlines(),
        },
        "command": " ".join(
            [
                f"HIPENGINE_HIP_ARCH={os.environ.get('HIPENGINE_HIP_ARCH', '')}",
                "PYTHONPATH=.",
                Path(os.sys.executable).name,
                *os.sys.argv,
            ]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "rows": [
                    {
                        "id": row["id"],
                        "control": row["control"]["score"]["passed"],
                        "candidate": row["candidate"]["score"]["passed"],
                        "control_s": row["control"]["elapsed_seconds"],
                        "candidate_s": row["candidate"]["elapsed_seconds"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
