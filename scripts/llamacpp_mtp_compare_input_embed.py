#!/usr/bin/env python3
"""Compare llama.cpp input_embed h_nextn row to hipEngine hidden_in_f32."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_compare_hidden_seed import (  # noqa: E402
    compare_capture_vectors,
    parse_prompt_tokens,
    redact_values,
    run_llamacpp_hidden_seed_capture,
    status_from_results,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    compare_all_rows,
    pack_float32,
    summarize_capture,
    unpack_float32,
)

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter318-input-embed-harness-compile.json"
)
DEFAULT_LAYER0_REFERENCE = Path("benchmarks/results/mtp-gguf-iter316-layer0-compare.json")
DEFAULT_OUTPUT_PREFIX = Path("/tmp/hipengine-llamacpp-mtp-iter318-input-embed/pos16")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter318-input-embed-compare.json")

HipInputCaptureFn = Callable[[Path, tuple[int, ...], int, int | None], dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--layer0-reference", type=Path, default=DEFAULT_LAYER0_REFERENCE)
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
    parser.add_argument("--iteration", type=int, default=318)
    args = parser.parse_args()

    artifact = compare_input_embed(
        compile_artifact_path=args.compile_artifact,
        layer0_reference_path=args.layer0_reference,
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
                "exact_rmse": artifact["numeric_delta"].get("rmse"),
                "bf16_roundtrip_rmse": artifact["bf16_rounded_delta"].get("rmse"),
                "classification": artifact["classification"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compare_input_embed(
    *,
    compile_artifact_path: Path,
    layer0_reference_path: Path,
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
    iteration: int = 318,
    hip_capture_fn: HipInputCaptureFn | None = None,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must be non-empty")
    if position < 0 or position >= len(prompt_tokens):
        raise ValueError("position outside prompt token range")
    compile_artifact = json.loads(compile_artifact_path.read_text())
    layer0_reference = json.loads(layer0_reference_path.read_text())
    executable = Path(compile_artifact["outputs"]["executable"])
    lib_dir = Path(compile_artifact["lib_dir"])
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    env_map = dict(os.environ if env is None else env)
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
    annotate_effective_input_tap(llama_capture)
    hip_capture = (hip_capture_fn or capture_hipengine_hidden_in)(
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
    bf16_capture = build_bf16_roundtrip_capture(
        llama_capture,
        output_prefix=output_prefix,
    )
    bf16_delta = compare_capture_vectors(
        llamacpp_capture=bf16_capture,
        hipengine_capture=hip_capture,
        exact_atol=exact_atol,
    )
    status = status_from_results(
        llamacpp_run=llama_run,
        llama_capture=llama_capture,
        hip_capture=hip_capture,
        numeric_delta=numeric_delta,
    )
    layer0 = summarize_layer0_reference(layer0_reference)
    classification = classify_input_embed(status, numeric_delta, bf16_delta)
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_input_embed_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "compile_artifact_path": str(compile_artifact_path),
        "layer0_reference_path": str(layer0_reference_path),
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
        "llamacpp_bf16_roundtrip_capture": bf16_capture,
        "hipengine_capture": redact_values(hip_capture),
        "numeric_delta": numeric_delta,
        "bf16_rounded_delta": bf16_delta,
        "prior_layer0_reference": layer0,
        "classification": classification,
        "external_checkout_modified": False,
        "next_action": next_action(status, classification),
    }


def annotate_effective_input_tap(capture: dict[str, Any]) -> None:
    metadata = capture.get("metadata")
    capture["effective_tap"] = "h_nextn_input_embed"
    if not isinstance(metadata, dict):
        return
    if metadata.get("tap") != "h_nextn_input_embed":
        capture["metadata_tap_note"] = (
            "generic hidden-seed harness writes its original tap label; the patched "
            "llama.cpp graph keeps res->t_h_nextn from model.input_embed"
        )


def capture_hipengine_hidden_in(
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
        for index, token_id in enumerate(prompt_tokens[:position]):
            session.step(int(token_id), position=index, return_logits=False)
        capture = session.capture_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=0,
            run_preceding_layers=False,
        )
        return {
            "status": "captured",
            "mode": "capture_attention_layer_hidden_in",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": 0,
            "preceding_layer_count": int(capture.preceding_layer_count),
            "dtype": "BF16_to_F32",
            "provenance": "capture_attention_layer.hidden_in_f32",
            "capture_summary": capture.as_summary_dict(),
            "values": [float(value) for value in capture.hidden_in_f32.tolist()],
        }


def build_bf16_roundtrip_capture(
    capture: dict[str, Any],
    *,
    output_prefix: Path,
) -> dict[str, Any]:
    binary_path = Path(capture.get("binary_path", ""))
    rounded_path = output_prefix.with_suffix(".llama_bf16_roundtrip.f32")
    meta_path = output_prefix.with_suffix(".llama_bf16_roundtrip.json")
    if not capture.get("binary_exists") or not binary_path.exists():
        return {
            "binary_path": str(rounded_path),
            "binary_exists": False,
            "metadata_path": str(meta_path),
            "metadata_exists": False,
            "rounding": "bf16_roundtrip",
        }
    values = unpack_float32(binary_path.read_bytes())
    rounded = [f32_to_bf16_roundtrip(value) for value in values]
    rounded_path.write_bytes(pack_float32(rounded))
    meta_path.write_text(
        json.dumps(
            {
                "kind": "llamacpp_bf16_roundtrip_capture",
                "source_binary": str(binary_path),
                "binary": str(rounded_path),
                "rounding": "round_to_nearest_even_bf16_then_f32",
            },
            indent=2,
        )
        + "\n"
    )
    result = summarize_capture(binary_path=rounded_path, meta_path=meta_path)
    result["rounding"] = "round_to_nearest_even_bf16_then_f32"
    result["source_sha256"] = capture.get("sha256")
    return result


def f32_to_bf16_roundtrip(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000 and mantissa != 0:
        rounded = bits | 0x00400000
    else:
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def summarize_layer0_reference(layer0: Mapping[str, Any]) -> dict[str, Any]:
    numeric = layer0.get("numeric_delta") or {}
    return {
        "status": layer0.get("status"),
        "classification": layer0.get("classification"),
        "layer_id": layer0.get("layer_id"),
        "rmse": numeric.get("rmse"),
        "max_abs_diff": numeric.get("max_abs_diff"),
        "mean_abs_diff": numeric.get("mean_abs_diff"),
        "llamacpp_sha256": numeric.get("llamacpp_sha256"),
        "hipengine_sha256": numeric.get("hipengine_sha256"),
    }


def classify_input_embed(
    status: str,
    exact_delta: Mapping[str, Any],
    bf16_delta: Mapping[str, Any],
) -> str:
    if status == "matched":
        return "input_embed_exact_match"
    if status != "mismatched":
        return "input_embed_comparison_unavailable"
    if bf16_delta.get("available") and bf16_delta.get("exact_match"):
        return "input_embed_matches_after_bf16_roundtrip"
    if exact_delta.get("shape_match"):
        return "input_embed_mismatch_after_bf16_roundtrip"
    return "input_embed_comparison_unavailable"


def next_action(status: str, classification: str) -> str:
    if status == "matched":
        return "investigate_layer0_implementation_after_embedding"
    if status == "llamacpp_capture_failed":
        return "inspect_input_embed_llamacpp_capture_logs"
    if status == "skipped_no_hip_runtime":
        return "rerun_on_rocm_host"
    if classification == "input_embed_matches_after_bf16_roundtrip":
        return "investigate_layer0_implementation_after_embedding"
    if classification == "input_embed_mismatch_after_bf16_roundtrip":
        return "audit_token_embedding_lookup_or_bf16_conversion"
    return "inspect_input_embed_compare_failure"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
