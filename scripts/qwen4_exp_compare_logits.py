#!/usr/bin/env python3
"""Compare full Qwen4Exp last-token logits with the frozen llama.cpp oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

import numpy as np

from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model

_DEFAULT_LLAMA_DEBUG = Path(
    "/home/lhl/llama.cpp/llama.cpp-qwen4exp/build-qwen4exp-hip/bin/llama-debug"
)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, dtype=np.float64)


def compare_logits(teacher: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    teacher_values = np.asarray(teacher, dtype=np.float32)
    actual_values = np.asarray(actual, dtype=np.float32)
    if teacher_values.ndim != 1 or actual_values.shape != teacher_values.shape:
        raise ValueError("teacher and actual logits must have the same 1D shape")
    if not np.all(np.isfinite(teacher_values)) or not np.all(np.isfinite(actual_values)):
        raise ValueError("teacher and actual logits must be finite")
    teacher_probability = _probabilities(teacher_values)
    actual_probability = _probabilities(actual_values)
    tiny = np.finfo(np.float64).tiny
    kl = float(
        np.sum(
            teacher_probability
            * (
                np.log(np.maximum(teacher_probability, tiny))
                - np.log(np.maximum(actual_probability, tiny))
            ),
            dtype=np.float64,
        )
    )
    teacher_top1 = int(np.argmax(teacher_values))
    actual_top1 = int(np.argmax(actual_values))
    difference = actual_values.astype(np.float64) - teacher_values.astype(np.float64)
    return {
        "kl_teacher_to_hipengine": kl,
        "teacher_top1": teacher_top1,
        "hipengine_top1": actual_top1,
        "top1_agreement": teacher_top1 == actual_top1,
        "mean_absolute_logit_error": float(np.mean(np.abs(difference))),
        "max_absolute_logit_error": float(np.max(np.abs(difference))),
    }


def _run_llama_debug(
    executable: Path,
    first_part: Path,
    prompt: str,
    output_directory: Path,
    context: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    command = [
        str(executable),
        "-m",
        str(first_part),
        "-p",
        prompt,
        "--save-logits",
        "--logits-output-dir",
        str(output_directory),
        "-c",
        str(context),
        "-ngl",
        "99",
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "llama-debug failed with exit code "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    logits_files = sorted(output_directory.glob("llamacpp-*.bin"))
    logits_files = [path for path in logits_files if not path.name.endswith("-tokens.bin")]
    token_files = sorted(output_directory.glob("llamacpp-*-tokens.bin"))
    if len(logits_files) != 1 or len(token_files) != 1:
        raise RuntimeError("llama-debug did not emit exactly one logits/token pair")
    logits = np.fromfile(logits_files[0], dtype=np.float32)
    tokens = np.fromfile(token_files[0], dtype=np.int32)
    return logits, tokens, command


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="split GGUF directory or a GGUF file")
    parser.add_argument("--prompt", default="The answer to 2 + 2 is")
    parser.add_argument("--context", type=int, default=2051)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--llama-debug", type=Path, default=_DEFAULT_LLAMA_DEBUG)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--keep-llama-output", type=Path)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--require-top1", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    model_path = args.model.expanduser().resolve()
    parts = discover_gguf_files(model_path)
    first_part = parts[0]
    executable = args.llama_debug.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"llama-debug is not executable: {executable}")
    if args.context <= 0:
        raise ValueError("--context must be positive")

    temporary = None
    if args.keep_llama_output is None:
        temporary = tempfile.TemporaryDirectory(prefix="qwen4exp-llama-logits-")
        llama_output = Path(temporary.name)
    else:
        llama_output = args.keep_llama_output.expanduser().resolve()
        llama_output.mkdir(parents=True, exist_ok=True)

    generator = None
    try:
        teacher_logits, teacher_tokens, command = _run_llama_debug(
            executable,
            first_part,
            args.prompt,
            llama_output,
            args.context,
        )
        index = load_gguf_index(first_part)
        plugin = resolve_model(index.architecture or "")
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model_path,
            weight_index=index,
            model_plugin=plugin,
            backend=args.backend,
            max_sequence_length=args.context,
        )
        hip_tokens = np.asarray(generator.tokenizer.encode(args.prompt), dtype=np.int32)
        if not np.array_equal(teacher_tokens, hip_tokens):
            raise RuntimeError(
                "tokenizer mismatch: "
                f"llama={teacher_tokens.tolist()} hipengine={hip_tokens.tolist()}"
            )
        actual = generator.runner.prefill(hip_tokens.tolist())
        metrics = compare_logits(teacher_logits, actual.logits)
        report = {
            "schema": 1,
            "model_path": str(model_path),
            "parts": [
                {"path": str(path), "bytes": path.stat().st_size} for path in parts
            ],
            "prompt": args.prompt,
            "token_ids": hip_tokens.tolist(),
            "vocab_size": int(actual.logits.size),
            "backend": args.backend,
            "llama_debug": str(executable),
            "llama_command": command,
            **metrics,
            "max_kl": float(args.max_kl),
            "require_top1": bool(args.require_top1),
        }
        passed = bool(metrics["kl_teacher_to_hipengine"] <= args.max_kl)
        if args.require_top1:
            passed = passed and bool(metrics["top1_agreement"])
        report["passed"] = passed
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.json_out is not None:
            output = args.json_out.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if passed else 1
    finally:
        if generator is not None:
            generator.close()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
