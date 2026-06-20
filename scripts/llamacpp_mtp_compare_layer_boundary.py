#!/usr/bin/env python3
"""Compare llama.cpp layer-boundary h_nextn row to hipEngine layer_out_f32."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
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
    summarize_capture,
)

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter311-layer39-harness-compile.json"
)
DEFAULT_PRIOR_PRE_OUTPUT = Path("benchmarks/results/mtp-gguf-iter309-pre-output-norm-compare.json")
DEFAULT_OUTPUT_PREFIX = Path("/tmp/hipengine-llamacpp-mtp-iter311-layer39/pos16")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter311-layer39-compare.json")

HipLayerCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--prior-pre-output", type=Path, default=DEFAULT_PRIOR_PRE_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--layer-id", type=int, default=39)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--exact-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=311)
    args = parser.parse_args()

    artifact = compare_layer_boundary(
        compile_artifact_path=args.compile_artifact,
        prior_pre_output_path=args.prior_pre_output,
        model_path=args.model,
        prompt_tokens=parse_prompt_tokens(args.prompt_tokens),
        position=args.position,
        layer_id=args.layer_id,
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
                "layer_id": artifact["layer_id"],
                "llamacpp_rc": artifact["llamacpp_run"]["returncode"],
                "hipengine_status": artifact["hipengine_capture"]["status"],
                "rmse": artifact["numeric_delta"].get("rmse"),
                "max_abs_diff": artifact["numeric_delta"].get("max_abs_diff"),
                "classification": artifact["classification"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compare_layer_boundary(
    *,
    compile_artifact_path: Path,
    prior_pre_output_path: Path,
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
    output_prefix: Path,
    n_gpu_layers: int = 999,
    threads: int = 8,
    all_rows: bool = False,
    max_sequence_length: int | None = None,
    timeout_seconds: int = 1800,
    exact_atol: float = 0.0,
    env: Mapping[str, str] | None = None,
    iteration: int = 311,
    hip_capture_fn: HipLayerCaptureFn | None = None,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must be non-empty")
    if position < 0 or position >= len(prompt_tokens):
        raise ValueError("position outside prompt token range")
    if layer_id < 0:
        raise ValueError("layer_id must be non-negative")
    compile_artifact = json.loads(compile_artifact_path.read_text())
    prior_pre_output = json.loads(prior_pre_output_path.read_text())
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
    annotate_effective_layer_tap(llama_capture, layer_id=layer_id)
    hip_capture = (hip_capture_fn or capture_hipengine_layer_out)(
        model_path,
        prompt_tokens,
        int(position),
        int(layer_id),
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
    prior = summarize_prior_pre_output(prior_pre_output)
    alignment = classify_prior_alignment(numeric_delta, prior)
    classification = classify_layer_boundary(status, numeric_delta, alignment)
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_layer_boundary_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "layer_id": int(layer_id),
        "compile_artifact_path": str(compile_artifact_path),
        "prior_pre_output_path": str(prior_pre_output_path),
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
        "prior_pre_output_reference": prior,
        "prior_alignment": alignment,
        "classification": classification,
        "external_checkout_modified": False,
        "next_action": next_action(status, classification, layer_id=int(layer_id)),
    }


def annotate_effective_layer_tap(capture: dict[str, Any], *, layer_id: int) -> None:
    metadata = capture.get("metadata")
    capture["effective_tap"] = "h_nextn_layer_out"
    capture["effective_layer_id"] = int(layer_id)
    if not isinstance(metadata, dict):
        return
    if metadata.get("tap") != "h_nextn_layer_out":
        capture["metadata_tap_note"] = (
            "generic hidden-seed harness writes its original tap label; the patched "
            "llama.cpp graph keeps res->t_h_nextn from the selected layer output"
        )


def capture_hipengine_layer_out(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
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
            layer_id=int(layer_id),
            run_preceding_layers=True,
        )
        return {
            "status": "captured",
            "mode": "capture_attention_layer_layer_out",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": int(layer_id),
            "preceding_layer_count": int(capture.preceding_layer_count),
            "dtype": "BF16_to_F32",
            "provenance": "capture_attention_layer.layer_out_f32",
            "capture_summary": capture.as_summary_dict(),
            "values": [float(value) for value in capture.layer_out_f32.tolist()],
        }


def summarize_prior_pre_output(prior: Mapping[str, Any]) -> dict[str, Any]:
    numeric = prior.get("numeric_delta") or {}
    return {
        "status": prior.get("status"),
        "classification": prior.get("classification"),
        "rmse": numeric.get("rmse"),
        "max_abs_diff": numeric.get("max_abs_diff"),
        "mean_abs_diff": numeric.get("mean_abs_diff"),
        "llamacpp_sha256": numeric.get("llamacpp_sha256"),
        "hipengine_sha256": numeric.get("hipengine_sha256"),
    }


def classify_prior_alignment(delta: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    llama_match = bool(
        delta.get("llamacpp_sha256")
        and delta.get("llamacpp_sha256") == prior.get("llamacpp_sha256")
    )
    hip_match = bool(
        delta.get("hipengine_sha256")
        and delta.get("hipengine_sha256") == prior.get("hipengine_sha256")
    )
    return {
        "llamacpp_matches_prior_pre_output": llama_match,
        "hipengine_matches_prior_pre_output": hip_match,
        "both_match_prior_pre_output": bool(llama_match and hip_match),
        "rmse_matches_prior_pre_output": floats_equal(delta.get("rmse"), prior.get("rmse")),
        "max_abs_matches_prior_pre_output": floats_equal(
            delta.get("max_abs_diff"), prior.get("max_abs_diff")
        ),
    }


def classify_layer_boundary(
    status: str,
    delta: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> str:
    if status == "matched":
        return "layer_boundary_matches"
    if status != "mismatched":
        return "layer_boundary_comparison_unavailable"
    if alignment.get("both_match_prior_pre_output"):
        return "final_layer_reproduces_pre_output_mismatch"
    if alignment.get("llamacpp_matches_prior_pre_output"):
        return "hipengine_layer_out_differs_from_serial_pre_output"
    if alignment.get("hipengine_matches_prior_pre_output"):
        return "llamacpp_layer_out_differs_from_pre_output_oracle"
    if delta.get("shape_match"):
        return "layer_boundary_mismatch"
    return "layer_boundary_comparison_unavailable"


def next_action(status: str, classification: str, *, layer_id: int | None = None) -> str:
    if status == "matched":
        return "continue_bisect_with_earlier_midpoint_layer"
    if status == "llamacpp_capture_failed":
        return "inspect_layer_boundary_llamacpp_capture_logs"
    if status == "skipped_no_hip_runtime":
        return "rerun_on_rocm_host"
    if classification == "final_layer_reproduces_pre_output_mismatch":
        return "continue_bisect_with_layer_19"
    if classification == "layer_boundary_mismatch":
        earlier = earlier_midpoint(layer_id)
        if earlier is not None:
            return f"continue_bisect_with_layer_{earlier}"
        return "inspect_initial_embedding_or_token_capture"
    if classification == "hipengine_layer_out_differs_from_serial_pre_output":
        return "audit_hipengine_capture_attention_layer_vs_serial_loop"
    if classification == "llamacpp_layer_out_differs_from_pre_output_oracle":
        return "audit_llamacpp_layer_patch_alignment"
    return "inspect_layer_boundary_compare_failure"


def earlier_midpoint(layer_id: int | None) -> int | None:
    if layer_id is None or layer_id <= 0:
        return None
    return (int(layer_id) - 1) // 2


def floats_equal(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= atol


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
