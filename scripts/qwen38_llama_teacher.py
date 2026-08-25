#!/usr/bin/env python3
"""Prepare/finalize a llama.cpp Qwen3.8 BF16 full-logit teacher capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.loading import load_gguf_index
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows
from scripts.quant_quality.qwen36_teacher import PROTOCOL_ID, TEACHER_STEPS
from scripts.qwen36_dense_gguf_suite import HELDOUT_PROMPT_IDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    model = Path(args.model).resolve()
    prompts_path = Path(args.prompts).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
    prompts = load_prompt_rows(prompts_path)
    metadata_rows = []
    binary_path = output_dir / "prompts.bin"
    with binary_path.open("wb") as handle:
        handle.write(b"Q38Q")
        handle.write(struct.pack("<II", 1, len(prompts)))
        for row in prompts:
            prompt_ids = [
                int(token)
                for token in build_chat_prompt(tokenizer, str(row["prompt"]), reasoning="off")
            ]
            handle.write(struct.pack("<I", len(prompt_ids)))
            handle.write(np.asarray(prompt_ids, dtype="<i4").tobytes())
            metadata_rows.append(
                {
                    "id": str(row["id"]),
                    "category": str(row["category"]),
                    "split": (
                        "heldout" if str(row["id"]) in HELDOUT_PROMPT_IDS else "train"
                    ),
                    "prompt": str(row["prompt"]),
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": _sha256_json(prompt_ids),
                }
            )
    metadata = {
        "schema": 1,
        "kind": "qwen38_llama_teacher_capture_input",
        "protocol_id": PROTOCOL_ID,
        "teacher_steps": TEACHER_STEPS,
        "model": str(model),
        "prompt_source": str(prompts_path),
        "prompt_source_sha256": _sha256(prompts_path),
        "vocab_size": len(tokenizer.tokens),
        "prompts": metadata_rows,
        "binary_path": str(binary_path),
        "binary_sha256": _sha256(binary_path),
    }
    metadata_path = output_dir / "capture_input.json"
    _write_json(metadata_path, metadata)
    print(json.dumps({"input": str(binary_path), "metadata": str(metadata_path)}))
    return 0


def _teacher_rows(path: Path, prompt_count: int) -> list[list[int]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) != prompt_count:
        raise ValueError(f"teacher token rows {len(lines)} != prompt count {prompt_count}")
    rows = [[int(token) for token in line.split(",")] for line in lines]
    if any(len(row) != TEACHER_STEPS for row in rows):
        raise ValueError("every teacher token row must have the fixed teacher-step count")
    return rows


def finalize(args: argparse.Namespace) -> int:
    input_path = Path(args.capture_input).resolve()
    capture = json.loads(input_path.read_text(encoding="utf-8"))
    if capture.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("capture input protocol differs from quant-quality tooling")
    prompt_count = len(capture["prompts"])
    vocab_size = int(capture["vocab_size"])
    teacher_rows = _teacher_rows(Path(args.teacher_tokens), prompt_count)
    raw_path = Path(args.raw_logits).resolve()
    shape = (prompt_count * TEACHER_STEPS, vocab_size)
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(f"raw logits bytes {raw_path.stat().st_size} != {expected_bytes}")
    raw = np.memmap(raw_path, mode="r", dtype=np.float32, shape=shape)
    if not np.isfinite(raw).all():
        raise ValueError("BF16 teacher raw logits contain non-finite values")
    flat_tokens = np.asarray(
        [token for row in teacher_rows for token in row], dtype=np.int64
    )
    if not np.array_equal(np.argmax(raw, axis=1), flat_tokens):
        raise ValueError("teacher token rows do not equal full-logit argmax")

    output_dir = input_path.parent
    logits_path = output_dir / "bf16.npy"
    cache = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float16, shape=shape)
    cache[:] = raw.astype(np.float16)
    cache.flush()
    del cache, raw

    model = Path(args.model).resolve()
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
    fixture_prompts = []
    for prompt, teacher in zip(capture["prompts"], teacher_rows, strict=True):
        fixture_prompts.append(
            {
                **prompt,
                "teacher_token_ids": teacher,
                "teacher_text": tokenizer.decode(teacher),
            }
        )
    fixture = {
        "schema": 1,
        "kind": "quant_quality_teacher_fixture",
        "protocol_id": PROTOCOL_ID,
        "teacher_steps": TEACHER_STEPS,
        "reference_model": str(model),
        "reference_model_sha256": str(args.model_sha256),
        "prompt_source": capture["prompt_source"],
        "prompt_source_sha256": capture["prompt_source_sha256"],
        "rendering": "GGUF Qwen chat template; reasoning off",
        "reference_scoring_execution": (
            "llama.cpp Vulkan BF16 greedy prefill plus cached one-token steps"
        ),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "prompts": fixture_prompts,
    }
    fixture_path = output_dir / "fixture.json"
    _write_json(fixture_path, fixture)
    manifest = {
        "schema": 1,
        "kind": "quant_quality_full_logits_cache",
        "protocol_id": PROTOCOL_ID,
        "name": "Original Qwen3.8 BF16 GGUF / llama.cpp Vulkan",
        "runtime": f"llama.cpp Vulkan {args.runtime_revision}",
        "model_path": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": str(args.model_sha256),
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256(fixture_path),
        "logits_path": str(logits_path),
        "logits_sha256": _sha256(logits_path),
        "shape": list(shape),
        "dtype": "float16",
        "elapsed_seconds_diagnostic": float(args.elapsed_seconds),
        "host": platform.node(),
        "role": "reference",
        "teacher_generation": "greedy argmax",
        "raw_capture": str(raw_path),
        "raw_capture_sha256": _sha256(raw_path),
    }
    manifest_path = output_dir / "bf16.manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"fixture": str(fixture_path), "manifest": str(manifest_path)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"),
    )
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.set_defaults(func=prepare)
    final = sub.add_parser("finalize")
    final.add_argument("--capture-input", type=Path, required=True)
    final.add_argument("--raw-logits", type=Path, required=True)
    final.add_argument("--teacher-tokens", type=Path, required=True)
    final.add_argument("--model", type=Path, required=True)
    final.add_argument("--model-sha256", required=True)
    final.add_argument("--runtime-revision", required=True)
    final.add_argument("--elapsed-seconds", type=float, required=True)
    final.set_defaults(func=finalize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
