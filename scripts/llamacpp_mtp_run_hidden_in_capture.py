#!/usr/bin/env python3
"""Run the llama.cpp hidden-in capture harness and summarize the row hash."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter292-llamacpp-hidden-in-capture-harness-tokenids-compile.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter293-llamacpp-hidden-in-capture-result.json"
)
DEFAULT_OUTPUT_PREFIX = Path("/tmp/hipengine-llamacpp-mtp-iter293-hidden-in/layer3-pos16")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPT_TOKENS = (
    "248045,846,198,7734,264,2716,40719,13,248046,198,248045,74455,198,"
    "248068,271,248069,271"
)
DEFAULT_EXPECTED_SHA256 = "f6a6539866a1153c0d2e684a69d4004deabe83c1da4721225e78dd1a2ee74e07"
DEFAULT_REFERENCE_ARRAYS = Path(
    "benchmarks/results/mtp-gguf-iter280-layer3-full-attn-actual-routing-full-arrays.json"
)
DEFAULT_REFERENCE_KEY = "hidden_in_f32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--expected-sha256", default=DEFAULT_EXPECTED_SHA256)
    parser.add_argument("--reference-arrays", type=Path, default=DEFAULT_REFERENCE_ARRAYS)
    parser.add_argument("--reference-key", default=DEFAULT_REFERENCE_KEY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--iteration", type=int, default=293)
    args = parser.parse_args()

    artifact = run_hidden_in_capture(
        compile_artifact_path=args.compile_artifact,
        model_path=args.model,
        prompt_tokens=args.prompt_tokens,
        layer=args.layer,
        position=args.position,
        expected_sha256=args.expected_sha256,
        reference_arrays_path=args.reference_arrays,
        reference_key=args.reference_key,
        output_prefix=args.output_prefix,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        all_rows=args.all_rows,
        timeout_seconds=args.timeout_seconds,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "returncode": artifact["run"]["returncode"],
                "sha256": artifact["capture"].get("sha256"),
                "matches_expected": artifact["comparison"].get("matches_expected"),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def run_hidden_in_capture(
    *,
    compile_artifact_path: Path,
    model_path: Path,
    prompt_tokens: str,
    layer: int,
    position: int,
    expected_sha256: str,
    output_prefix: Path,
    reference_arrays_path: Path | None = None,
    reference_key: str = DEFAULT_REFERENCE_KEY,
    n_gpu_layers: int = 999,
    threads: int = 8,
    all_rows: bool = False,
    timeout_seconds: int = 1800,
    env: Mapping[str, str] | None = None,
    iteration: int = 293,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    compile_artifact = json.loads(compile_artifact_path.read_text())
    executable = Path(compile_artifact["outputs"]["executable"])
    lib_dir = Path(compile_artifact["lib_dir"])
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--model",
        str(model_path),
        "--prompt-tokens",
        prompt_tokens,
        "--layer",
        str(layer),
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
    run_env = with_library_path(env_map, lib_dir)
    run = run_logged(
        command,
        env=run_env,
        stdout_path=output_prefix.with_suffix(".stdout.log"),
        stderr_path=output_prefix.with_suffix(".stderr.log"),
        timeout_seconds=timeout_seconds,
    )
    binary_path = output_prefix.with_suffix(".f32")
    meta_path = output_prefix.with_suffix(".json")
    capture = summarize_capture(binary_path=binary_path, meta_path=meta_path)
    comparison = compare_capture(capture, expected_sha256=expected_sha256)
    numeric_delta = compare_numeric_reference(
        capture,
        reference_arrays_path=reference_arrays_path,
        reference_key=reference_key,
    )
    status = status_from_run(run=run, comparison=comparison)
    return {
        "schema": 1,
        "kind": "llamacpp_hidden_in_capture_result",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "compile_artifact_path": str(compile_artifact_path),
        "executable": str(executable),
        "lib_dir": str(lib_dir),
        "model": str(model_path),
        "prompt_tokens": parse_prompt_tokens(prompt_tokens),
        "layer": int(layer),
        "position": int(position),
        "all_rows_requested": bool(all_rows),
        "expected_sha256": expected_sha256,
        "run": run,
        "capture": capture,
        "comparison": comparison,
        "numeric_delta": numeric_delta,
        "external_checkout_modified": False,
        "next_action": next_action(status, comparison),
    }


def summarize_capture(*, binary_path: Path, meta_path: Path) -> dict[str, Any]:
    capture: dict[str, Any] = {
        "binary_path": str(binary_path),
        "binary_exists": binary_path.exists(),
        "metadata_path": str(meta_path),
        "metadata_exists": meta_path.exists(),
    }
    if meta_path.exists():
        try:
            capture["metadata"] = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            capture["metadata_error"] = str(exc)
    metadata = capture.get("metadata") or {}
    all_rows_path = Path(metadata.get("all_rows_binary", "")) if metadata else None
    if all_rows_path and str(all_rows_path) != "." and all_rows_path.exists():
        all_data = all_rows_path.read_bytes()
        all_floats = unpack_float32(all_data)
        capture["all_rows"] = {
            "binary_path": str(all_rows_path),
            "bytes": len(all_data),
            "float_count": len(all_floats),
            "sha256": sha256_bytes(all_data),
        }
    if not binary_path.exists():
        return capture
    data = binary_path.read_bytes()
    floats = unpack_float32(data)
    capture.update(
        {
            "bytes": len(data),
            "float_count": len(floats),
            "sha256": sha256_bytes(data),
            "stats": summarize_floats(floats),
            "samples": [round(value, 8) for value in floats[:8]],
            "top_abs": top_abs_entries(floats, limit=8),
        }
    )
    return capture


def compare_capture(capture: dict[str, Any], *, expected_sha256: str) -> dict[str, Any]:
    actual = capture.get("sha256")
    return {
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "matches_expected": actual == expected_sha256,
        "comparable": actual is not None,
    }


def compare_numeric_reference(
    capture: dict[str, Any],
    *,
    reference_arrays_path: Path | None,
    reference_key: str,
) -> dict[str, Any]:
    if reference_arrays_path is None or not reference_arrays_path.exists():
        return {"available": False, "reason": "reference_arrays_missing"}
    if not capture.get("binary_exists"):
        return {"available": False, "reason": "capture_binary_missing"}
    reference_doc = json.loads(reference_arrays_path.read_text())
    arrays = reference_doc.get("arrays") or {}
    if reference_key not in arrays:
        return {"available": False, "reason": "reference_key_missing"}
    reference = [float(value) for value in arrays[reference_key]]
    actual = unpack_float32(Path(capture["binary_path"]).read_bytes())
    if len(reference) != len(actual):
        return {
            "available": True,
            "shape_match": False,
            "reference_count": len(reference),
            "actual_count": len(actual),
        }
    diffs = [a - b for a, b in zip(actual, reference)]
    abs_diffs = [abs(value) for value in diffs]
    result = {
        "available": True,
        "shape_match": True,
        "reference_path": str(reference_arrays_path),
        "reference_key": reference_key,
        "reference_sha256": sha256_bytes(pack_float32(reference)),
        "actual_sha256": capture.get("sha256"),
        "count": len(actual),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
        "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else 0.0,
        "actual_l2": math.sqrt(sum(value * value for value in actual)),
        "reference_l2": math.sqrt(sum(value * value for value in reference)),
        "diff_samples": [round(value, 8) for value in diffs[:8]],
        "top_abs_diff": top_abs_diff_entries(actual, reference, limit=8),
    }
    row_scan = compare_all_rows(capture, reference)
    if row_scan is not None:
        result["all_rows_scan"] = row_scan
    return result


def compare_all_rows(
    capture: dict[str, Any], reference: list[float]
) -> dict[str, Any] | None:
    metadata = capture.get("metadata") or {}
    all_rows = capture.get("all_rows") or {}
    all_path = all_rows.get("binary_path") or metadata.get("all_rows_binary")
    n_embd = int(metadata.get("n_embd") or len(reference))
    if not all_path or not Path(all_path).exists() or n_embd <= 0:
        return None
    values = unpack_float32(Path(all_path).read_bytes())
    if len(values) % n_embd != 0:
        return {"available": False, "reason": "all_rows_shape_mismatch"}
    rows = len(values) // n_embd
    if n_embd != len(reference):
        return {
            "available": False,
            "reason": "reference_width_mismatch",
            "row_width": n_embd,
            "reference_count": len(reference),
        }
    metrics = []
    for row in range(rows):
        start = row * n_embd
        current = values[start : start + n_embd]
        diffs = [a - b for a, b in zip(current, reference)]
        abs_diffs = [abs(value) for value in diffs]
        metrics.append(
            {
                "row": row,
                "sha256": sha256_bytes(pack_float32(current)),
                "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
                "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
                "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs))
                if diffs
                else 0.0,
            }
        )
    best = min(metrics, key=lambda item: item["rmse"]) if metrics else None
    return {
        "available": True,
        "rows": rows,
        "row_width": n_embd,
        "best_by_rmse": best,
        "matches": [item for item in metrics if item["max_abs_diff"] == 0.0],
        "rows_by_rmse": sorted(metrics, key=lambda item: item["rmse"])[:8],
    }


def status_from_run(*, run: dict[str, Any], comparison: dict[str, Any]) -> str:
    if run["returncode"] != 0:
        return "capture_failed"
    if not comparison["comparable"]:
        return "capture_missing_output"
    if comparison["matches_expected"]:
        return "matched"
    return "mismatched"


def next_action(status: str, comparison: dict[str, Any]) -> str:
    if status == "matched":
        return "promote_llamacpp_hidden_in_oracle_to_layer_checkpoint_compare"
    if status == "mismatched":
        return "inspect_llamacpp_vs_hipengine_hidden_in_numeric_delta"
    if comparison.get("comparable"):
        return "inspect_capture_status_and_hash"
    return "inspect_capture_harness_logs"


def run_logged(
    command: list[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = decode_timeout_stream(exc.stdout)
        stderr = decode_timeout_stream(exc.stderr)
        stderr += f"\nTIMEOUT after {timeout_seconds}s"
        timed_out = True
    elapsed = time.monotonic() - start
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "command": command,
        "command_shell": subprocess.list2cmdline(command),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def with_library_path(env: Mapping[str, str], lib_dir: Path) -> dict[str, str]:
    result = dict(env)
    old = result.get("LD_LIBRARY_PATH", "")
    result["LD_LIBRARY_PATH"] = str(lib_dir) + (os.pathsep + old if old else "")
    return result


def parse_prompt_tokens(csv: str) -> list[int]:
    return [int(item) for item in csv.split(",") if item]


def unpack_float32(data: bytes) -> list[float]:
    if len(data) % 4 != 0:
        raise ValueError(f"float32 byte length is not divisible by 4: {len(data)}")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def summarize_floats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "l2": math.sqrt(sum(value * value for value in values)),
    }


def pack_float32(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def top_abs_entries(values: list[float], *, limit: int) -> list[dict[str, Any]]:
    entries = sorted(
        ((index, value) for index, value in enumerate(values)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]
    return [
        {"index": index, "value": round(value, 8), "abs": round(abs(value), 8)}
        for index, value in entries
    ]


def top_abs_diff_entries(
    actual: list[float], reference: list[float], *, limit: int
) -> list[dict[str, Any]]:
    entries = sorted(
        ((index, a, b, a - b) for index, (a, b) in enumerate(zip(actual, reference))),
        key=lambda item: abs(item[3]),
        reverse=True,
    )[:limit]
    return [
        {
            "index": index,
            "actual": round(actual_value, 8),
            "reference": round(reference_value, 8),
            "diff": round(diff, 8),
            "abs_diff": round(abs(diff), 8),
        }
        for index, actual_value, reference_value, diff in entries
    ]


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def tail_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    main()
