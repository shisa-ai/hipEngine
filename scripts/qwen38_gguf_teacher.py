#!/usr/bin/env python3
"""Capture an unqualified hipEngine BF16 diagnostic (not a quality oracle)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.loading import load_gguf_index
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows
from scripts.quant_quality.qwen36_teacher import (
    PROTOCOL_ID as QUANT_QUALITY_PROTOCOL_ID,
    TEACHER_STEPS as QUANT_QUALITY_TEACHER_STEPS,
)
from scripts.qwen36_dense_gguf_suite import HELDOUT_PROMPT_IDS

# Reuse the established 90-row cache/compare schema. Despite its historical
# qwen36 name, the protocol is model-bound by fixture/model hashes.
TEACHER_STEPS = QUANT_QUALITY_TEACHER_STEPS
PROTOCOL_ID = QUANT_QUALITY_PROTOCOL_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def capture(args: argparse.Namespace) -> int:
    if not args.allow_unqualified_diagnostic:
        raise RuntimeError(
            "hipEngine Qwen3.8 BF16 trunk execution is not a qualified quality oracle; "
            "use scripts/qwen38_llama_teacher.py or pass the explicit diagnostic override"
        )
    model = Path(args.model).resolve()
    prompts_path = Path(args.prompts).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "teacher_fixture.json"
    logits_path = output_dir / "bf16_logits.npy"
    manifest_path = output_dir / "bf16_logits.manifest.json"
    compiler_version = (
        None
        if args.compiler_version_file is None
        else Path(args.compiler_version_file).read_text(encoding="utf-8")
    )

    info = load_gguf_index(model)
    if info.file_type_name != "MOSTLY_BF16":
        raise ValueError("teacher model must be one merged MOSTLY_BF16 GGUF")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    prompts = load_prompt_rows(prompts_path)
    encoded = {
        str(row["id"]): tuple(
            int(token)
            for token in build_chat_prompt(tokenizer, str(row["prompt"]), reasoning="off")
        )
        for row in prompts
    }
    max_sequence_length = max(len(tokens) for tokens in encoded.values()) + TEACHER_STEPS + 4
    n_rows = len(prompts) * TEACHER_STEPS
    vocab_size = len(tokenizer.tokens)
    cache = np.lib.format.open_memmap(
        logits_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_rows, vocab_size),
    )
    fixture_prompts: list[dict[str, Any]] = []
    row_index = 0
    started = time.perf_counter()
    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        max_sequence_length=max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        for row in prompts:
            prompt_ids = encoded[str(row["id"])]
            session.reset()
            result = session.prefill(
                prompt_ids,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
            )
            teacher: list[int] = []
            for step in range(TEACHER_STEPS):
                logits = np.asarray(result.logits, dtype=np.float32).reshape(-1)
                if logits.shape != (vocab_size,) or not np.isfinite(logits).all():
                    raise RuntimeError(f"malformed BF16 logits at {row['id']} step {step}")
                cache[row_index] = logits.astype(np.float16)
                token_id = int(np.argmax(logits))
                teacher.append(token_id)
                row_index += 1
                if step + 1 < TEACHER_STEPS:
                    result = session.step(token_id, return_logits=True)
            fixture_prompts.append(
                {
                    "id": str(row["id"]),
                    "category": str(row["category"]),
                    "split": "heldout" if str(row["id"]) in HELDOUT_PROMPT_IDS else "train",
                    "prompt": str(row["prompt"]),
                    "prompt_token_ids": list(prompt_ids),
                    "prompt_token_ids_sha256": _sha256_json(prompt_ids),
                    "teacher_token_ids": teacher,
                    "teacher_text": tokenizer.decode(teacher),
                }
            )
            cache.flush()
            print(f"BF16 {row['id']}: {teacher}", flush=True)
    cache.flush()
    del cache
    elapsed = time.perf_counter() - started
    fixture = {
        "schema": 1,
        "kind": "quant_quality_teacher_fixture",
        "protocol_id": PROTOCOL_ID,
        "teacher_steps": TEACHER_STEPS,
        "reference_model": str(model),
        "reference_model_sha256": str(args.model_sha256),
        "prompt_source": str(prompts_path),
        "prompt_source_sha256": _sha256(prompts_path),
        "rendering": "GGUF Qwen chat template; reasoning off",
        "reference_scoring_execution": "hipEngine BF16 bulk prefill plus cached one-token steps",
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "prompts": fixture_prompts,
    }
    _write_json(fixture_path, fixture)
    manifest = {
        "schema": 1,
        "kind": "quant_quality_full_logits_cache",
        "protocol_id": PROTOCOL_ID,
        "name": "UNQUALIFIED Qwen3.8 merged BF16 GGUF / hipEngine diagnostic",
        "runtime": f"hipEngine ({args.backend})",
        "model_path": str(model),
        "model_sha256": str(args.model_sha256),
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256(fixture_path),
        "logits_path": str(logits_path),
        "logits_sha256": _sha256(logits_path),
        "shape": [n_rows, vocab_size],
        "dtype": "float16",
        "elapsed_seconds": elapsed,
        "role": "diagnostic_not_reference",
        "teacher_generation": "greedy argmax",
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"fixture": str(fixture_path), "manifest": str(manifest_path)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"),
    )
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--allow-unqualified-diagnostic",
        action="store_true",
        help="capture the known-unqualified hipEngine BF16 path for debugging only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    return capture(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
