#!/usr/bin/env python3
"""Compare full Qwen4Exp last-token logits with the frozen llama.cpp oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
from time import perf_counter
from typing import Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
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
    llama_batch: int = 0,
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
        "-ctk",
        "bf16",
        "-ctv",
        "bf16",
        "-ngl",
        "99",
    ]
    if llama_batch > 0:
        command.extend(("-b", str(llama_batch)))
    result = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
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
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--context", type=int, default=2051)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--llama-debug", type=Path, default=_DEFAULT_LLAMA_DEBUG)
    parser.add_argument("--llama-batch", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--keep-llama-output", type=Path)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument(
        "--prefill-mode", choices=("serial", "chunked", "both"), default="serial"
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=2)
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
    prompt = (
        args.prompt_file.expanduser().resolve().read_text(encoding="utf-8")
        if args.prompt_file is not None
        else args.prompt
    )
    if args.llama_batch < 0:
        raise ValueError("--llama-batch must be non-negative")

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
            prompt,
            llama_output,
            args.context,
            args.llama_batch,
        )
        index = load_gguf_index(first_part)
        plugin = resolve_model(index.architecture or "")
        runtime = get_hip_runtime()
        free_before, total_device_bytes = runtime.mem_get_info()
        reset_memory_stats()
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model_path,
            weight_index=index,
            model_plugin=plugin,
            backend=args.backend,
            max_sequence_length=args.context,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        hip_tokens = np.asarray(generator.tokenizer.encode(prompt), dtype=np.int32)
        if not np.array_equal(teacher_tokens, hip_tokens):
            raise RuntimeError(
                "tokenizer mismatch: "
                f"llama={teacher_tokens.tolist()} hipengine={hip_tokens.tolist()}"
            )
        free_after_residency, _ = runtime.mem_get_info()
        teacher_to_serial = None
        serial_to_chunked = None
        prefill_timings: dict[str, float] = {}
        if args.prefill_mode == "both":
            started = perf_counter()
            serial = generator.runner.prefill_serial(hip_tokens.tolist())
            prefill_timings["serial_seconds"] = perf_counter() - started
            serial_logits = serial.logits.copy()
            teacher_to_serial = compare_logits(teacher_logits, serial_logits)
            started = perf_counter()
            actual = generator.runner.prefill_chunked(hip_tokens.tolist())
            prefill_timings["chunked_seconds"] = perf_counter() - started
            serial_to_chunked = compare_logits(serial_logits, actual.logits)
        else:
            prefill = (
                generator.runner.prefill_chunked
                if args.prefill_mode == "chunked"
                else generator.runner.prefill_serial
            )
            started = perf_counter()
            actual = prefill(hip_tokens.tolist())
            prefill_timings[f"{args.prefill_mode}_seconds"] = (
                perf_counter() - started
            )
        free_after_inference, _ = runtime.mem_get_info()
        owned_peak = memory_stats()
        max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        metrics = compare_logits(teacher_logits, actual.logits)
        generator.close()
        generator = None
        free_after_close, _ = runtime.mem_get_info()
        owned_after_close = memory_stats()
        teardown_passed = owned_after_close["current_allocated_bytes"] == 0
        report = {
            "schema": 1,
            "prefill_mode": args.prefill_mode,
            "prefill_chunk_size": args.prefill_chunk_size,
            "prefill_timings": prefill_timings,
            "model_path": str(model_path),
            "parts": [
                {"path": str(path), "bytes": path.stat().st_size} for path in parts
            ],
            "prompt": prompt,
            "prompt_file": (
                str(args.prompt_file.expanduser().resolve())
                if args.prompt_file is not None
                else None
            ),
            "token_ids": hip_tokens.tolist(),
            "vocab_size": int(actual.logits.size),
            "backend": args.backend,
            "llama_debug": str(executable),
            "llama_command": command,
            "memory": {
                "device_total_bytes": total_device_bytes,
                "device_free_before_bytes": free_before,
                "device_free_after_residency_bytes": free_after_residency,
                "device_free_after_inference_bytes": free_after_inference,
                "device_free_after_close_bytes": free_after_close,
                "hipengine_owned_peak": owned_peak,
                "hipengine_owned_after_close": owned_after_close,
                "process_max_rss_kib": max_rss_kib,
                "tracked_teardown_passed": teardown_passed,
            },
            **metrics,
            "max_kl": float(args.max_kl),
            "require_top1": bool(args.require_top1),
        }
        if teacher_to_serial is not None:
            report["teacher_to_serial"] = teacher_to_serial
        if serial_to_chunked is not None:
            report["serial_to_chunked"] = serial_to_chunked
        passed = bool(metrics["kl_teacher_to_hipengine"] <= args.max_kl) and teardown_passed
        if args.require_top1:
            passed = passed and bool(metrics["top1_agreement"])
        if teacher_to_serial is not None:
            passed = passed and bool(
                teacher_to_serial["kl_teacher_to_hipengine"] <= args.max_kl
            )
            if args.require_top1:
                passed = passed and bool(teacher_to_serial["top1_agreement"])
        if serial_to_chunked is not None:
            passed = passed and bool(
                serial_to_chunked["kl_teacher_to_hipengine"] <= args.max_kl
            )
            if args.require_top1:
                passed = passed and bool(serial_to_chunked["top1_agreement"])
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
