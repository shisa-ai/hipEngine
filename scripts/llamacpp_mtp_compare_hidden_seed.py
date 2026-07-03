#!/usr/bin/env python3
"""Compare llama.cpp embeddings_nextn hidden seed to hipEngine FP32 seed."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType  # noqa: E402
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr  # noqa: E402
from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    compare_all_rows,
    pack_float32,
    run_logged,
    sha256_bytes,
    summarize_capture,
    summarize_floats,
    top_abs_diff_entries,
    unpack_float32,
    with_library_path,
)

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter302-llamacpp-hidden-seed-capture-harness-compile.json"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "/tmp/hipengine-llamacpp-mtp-iter303-hidden-seed/pos16"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter303-hidden-seed-compare.json"
)

HipCaptureFn = Callable[[Path, tuple[int, ...], int, int | None], dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--exact-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=303)
    args = parser.parse_args()

    artifact = compare_hidden_seed(
        compile_artifact_path=args.compile_artifact,
        model_path=args.model,
        prompt_tokens=parse_prompt_tokens(args.prompt_tokens),
        position=args.position,
        output_prefix=args.output_prefix,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        all_rows=bool(args.all_rows),
        max_sequence_length=args.max_sequence_length,
        timeout_seconds=args.timeout_seconds,
        exact_atol=args.exact_atol,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "llamacpp_rc": artifact["llamacpp_run"]["returncode"],
                "hipengine_status": artifact["hipengine_capture"]["status"],
                "rmse": artifact["numeric_delta"].get("rmse"),
                "max_abs_diff": artifact["numeric_delta"].get("max_abs_diff"),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compare_hidden_seed(
    *,
    compile_artifact_path: Path,
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    output_prefix: Path,
    n_gpu_layers: int = 999,
    threads: int = 8,
    all_rows: bool = False,
    max_sequence_length: int | None = None,
    timeout_seconds: int = 1800,
    exact_atol: float = 0.0,
    env: Mapping[str, str] | None = None,
    iteration: int = 303,
    hip_capture_fn: HipCaptureFn | None = None,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must be non-empty")
    if int(position) != len(prompt_tokens) - 1:
        raise ValueError(
            "hipEngine FP32 hidden seed capture currently targets the final prompt token"
        )
    env_map = dict(os.environ if env is None else env)
    compile_artifact = json.loads(compile_artifact_path.read_text())
    executable = Path(compile_artifact["outputs"]["executable"])
    lib_dir = Path(compile_artifact["lib_dir"])
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    llama_run = run_llamacpp_hidden_seed_capture(
        executable=executable,
        lib_dir=lib_dir,
        model_path=model_path,
        prompt_tokens=prompt_tokens,
        position=position,
        output_prefix=output_prefix,
        n_gpu_layers=n_gpu_layers,
        threads=threads,
        all_rows=all_rows,
        timeout_seconds=timeout_seconds,
        env=env_map,
    )
    llama_capture = summarize_capture(
        binary_path=output_prefix.with_suffix(".f32"),
        meta_path=output_prefix.with_suffix(".json"),
    )
    hip_capture = (hip_capture_fn or capture_hipengine_hidden_seed)(
        model_path,
        prompt_tokens,
        int(position),
        max_sequence_length,
    )
    numeric_delta = compare_capture_vectors(
        llamacpp_capture=llama_capture,
        hipengine_capture=hip_capture,
        exact_atol=exact_atol,
    )
    if numeric_delta.get("shape_match"):
        scan = compare_all_rows(llama_capture, hip_capture["values"])
        if scan is not None:
            numeric_delta["all_rows_scan"] = scan
    status = status_from_results(
        llamacpp_run=llama_run,
        llama_capture=llama_capture,
        hip_capture=hip_capture,
        numeric_delta=numeric_delta,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_hidden_seed_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "compile_artifact_path": str(compile_artifact_path),
        "executable": str(executable),
        "lib_dir": str(lib_dir),
        "model": str(model_path),
        "prompt_tokens": list(prompt_tokens),
        "position": int(position),
        "token_id": int(prompt_tokens[position]),
        "all_rows_requested": bool(all_rows),
        "max_sequence_length": max_sequence_length,
        "exact_atol": float(exact_atol),
        "llamacpp_run": llama_run,
        "llamacpp_capture": llama_capture,
        "hipengine_capture": redact_values(hip_capture),
        "numeric_delta": numeric_delta,
        "external_checkout_modified": False,
        "next_action": next_action(status, numeric_delta),
    }


def run_llamacpp_hidden_seed_capture(
    *,
    executable: Path,
    lib_dir: Path,
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    output_prefix: Path,
    n_gpu_layers: int,
    threads: int,
    all_rows: bool,
    timeout_seconds: int,
    env: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        str(executable),
        "--model",
        str(model_path),
        "--prompt-tokens",
        ",".join(str(token) for token in prompt_tokens),
        "--position",
        str(position),
        "--output-prefix",
        str(output_prefix),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--threads",
        str(threads),
    ]
    if all_rows:
        command.append("--all-rows")
    return run_logged(
        command,
        env=with_library_path(env, lib_dir),
        stdout_path=output_prefix.with_suffix(".stdout.log"),
        stderr_path=output_prefix.with_suffix(".stderr.log"),
        timeout_seconds=timeout_seconds,
    )


def capture_hipengine_hidden_seed(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "values": []}
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        result = session.prefill(
            list(prompt_tokens),
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        contract = session.fp32_hidden_seed_contract(rows=1).as_dict()
        ptr = session.fp32_hidden_seed_ptr()
        runtime = session.runtime
        if runtime is None:
            raise RuntimeError("GGUF resident session runtime is unavailable")
        values = np.empty((session.runner.hidden_size,), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(values),
            DeviceBuffer(int(ptr), values.nbytes),
            values.nbytes,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return {
            "status": "captured",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "next_token_id": int(result.token_id),
            "contract": contract,
            "dtype": DType.FP32.name,
            "values": [float(value) for value in values.tolist()],
        }


def compare_capture_vectors(
    *,
    llamacpp_capture: dict[str, Any],
    hipengine_capture: dict[str, Any],
    exact_atol: float,
) -> dict[str, Any]:
    if not llamacpp_capture.get("binary_exists"):
        return {"available": False, "reason": "llamacpp_capture_missing"}
    if hipengine_capture.get("status") != "captured":
        return {"available": False, "reason": hipengine_capture.get("status")}
    actual = unpack_float32(Path(llamacpp_capture["binary_path"]).read_bytes())
    reference = [float(value) for value in hipengine_capture["values"]]
    if len(actual) != len(reference):
        return {
            "available": True,
            "shape_match": False,
            "llamacpp_count": len(actual),
            "hipengine_count": len(reference),
        }
    diffs = [a - b for a, b in zip(actual, reference)]
    abs_diffs = [abs(value) for value in diffs]
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    return {
        "available": True,
        "shape_match": True,
        "count": len(actual),
        "llamacpp_sha256": sha256_bytes(pack_float32(actual)),
        "hipengine_sha256": sha256_bytes(pack_float32(reference)),
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs))
        if diffs
        else 0.0,
        "llamacpp_l2": math.sqrt(sum(value * value for value in actual)),
        "hipengine_l2": math.sqrt(sum(value * value for value in reference)),
        "diff_samples": [round(value, 8) for value in diffs[:8]],
        "top_abs_diff": top_abs_diff_entries(actual, reference, limit=8),
        "exact_match": max_abs <= exact_atol,
    }


def status_from_results(
    *,
    llamacpp_run: dict[str, Any],
    llama_capture: dict[str, Any],
    hip_capture: dict[str, Any],
    numeric_delta: dict[str, Any],
) -> str:
    if llamacpp_run["returncode"] != 0:
        return "llamacpp_capture_failed"
    if not llama_capture.get("binary_exists"):
        return "llamacpp_capture_missing_output"
    if hip_capture.get("status") != "captured":
        return str(hip_capture.get("status", "hipengine_capture_failed"))
    if not numeric_delta.get("available") or not numeric_delta.get("shape_match"):
        return "comparison_unavailable"
    if numeric_delta.get("exact_match"):
        return "matched"
    return "mismatched"


def redact_values(capture: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in capture.items() if key != "values"}
    values = [float(value) for value in capture.get("values", [])]
    if values:
        result["summary"] = {
            "count": len(values),
            "sha256": sha256_bytes(pack_float32(values)),
            "stats": summarize_floats(values),
            "samples": [round(value, 8) for value in values[:8]],
        }
    return result


def next_action(status: str, numeric_delta: dict[str, Any]) -> str:
    if status == "matched":
        return "promote_fp32_hidden_seed_oracle_to_nextn_parity_gate"
    if status == "mismatched":
        return "inspect_upstream_bf16_activation_propagation_before_fp32_seed"
    if status == "llamacpp_capture_failed":
        return "inspect_llamacpp_hidden_seed_capture_logs"
    if status == "skipped_no_hip_runtime":
        return "rerun_on_rocm_host"
    if not numeric_delta.get("shape_match", True):
        return "fix_hidden_seed_width_mismatch"
    return "inspect_hidden_seed_compare_failure"


def parse_prompt_tokens(csv: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in csv.split(",") if item.strip())
    if not tokens:
        raise ValueError("prompt token list is empty")
    return tokens


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
