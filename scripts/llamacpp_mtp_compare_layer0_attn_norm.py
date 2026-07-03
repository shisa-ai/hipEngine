#!/usr/bin/env python3
"""Compare llama.cpp layer-0 attn_norm h_nextn row to hipEngine attn_norm_f32."""

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
from scripts.llamacpp_mtp_compare_input_embed import (  # noqa: E402
    build_bf16_roundtrip_capture,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    compare_all_rows,
    summarize_capture,
)

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter320-layer0-attn-norm-harness-compile.json"
)
DEFAULT_PLAN_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter319-layer0-subboundary-plan.json"
)
DEFAULT_OUTPUT_PREFIX = Path("/tmp/hipengine-llamacpp-mtp-iter320-layer0-attn-norm/pos16")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter320-layer0-attn-norm-compare.json")

HipAttnNormCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--plan-artifact", type=Path, default=DEFAULT_PLAN_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--exact-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=320)
    args = parser.parse_args()

    artifact = compare_layer0_attn_norm(
        compile_artifact_path=args.compile_artifact,
        plan_artifact_path=args.plan_artifact,
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
                "exact_rmse": artifact["numeric_delta"].get("rmse"),
                "bf16_roundtrip_rmse": artifact["bf16_rounded_delta"].get("rmse"),
                "classification": artifact["classification"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compare_layer0_attn_norm(
    *,
    compile_artifact_path: Path,
    plan_artifact_path: Path,
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
    iteration: int = 320,
    hip_capture_fn: HipAttnNormCaptureFn | None = None,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must be non-empty")
    if position < 0 or position >= len(prompt_tokens):
        raise ValueError("position outside prompt token range")
    if layer_id != 0:
        raise ValueError("layer0 attn_norm diagnostic currently requires layer_id=0")
    compile_artifact = json.loads(compile_artifact_path.read_text())
    plan_artifact = json.loads(plan_artifact_path.read_text())
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
    annotate_effective_attn_norm_tap(llama_capture, layer_id=layer_id)
    hip_capture = (hip_capture_fn or capture_hipengine_layer0_attn_norm)(
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
    plan_summary = summarize_plan_reference(plan_artifact)
    classification = classify_layer0_attn_norm(status, numeric_delta, bf16_delta)
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_layer0_attn_norm_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "layer_id": int(layer_id),
        "compile_artifact_path": str(compile_artifact_path),
        "plan_artifact_path": str(plan_artifact_path),
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
        "plan_reference": plan_summary,
        "classification": classification,
        "external_checkout_modified": False,
        "next_action": next_action(status, classification),
    }


def annotate_effective_attn_norm_tap(capture: dict[str, Any], *, layer_id: int) -> None:
    metadata = capture.get("metadata")
    capture["effective_tap"] = "h_nextn_layer0_attn_norm"
    capture["effective_layer_id"] = int(layer_id)
    if not isinstance(metadata, dict):
        return
    if metadata.get("tap") != "h_nextn_layer0_attn_norm":
        capture["metadata_tap_note"] = (
            "generic hidden-seed harness writes its original tap label; the patched "
            "llama.cpp graph keeps res->t_h_nextn from layer-0 attn_norm"
        )


def capture_hipengine_layer0_attn_norm(
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
        capture = session.capture_linear_attention_boundary(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
        )
        return {
            "status": "captured",
            "mode": "capture_linear_attention_boundary_attn_norm",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": int(layer_id),
            "dtype": "BF16_to_F32",
            "provenance": "capture_linear_attention_boundary.attn_norm_f32",
            "capture_summary": capture.as_summary_dict(),
            "values": [float(value) for value in capture.attn_norm_f32.tolist()],
        }


def summarize_plan_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    comparison = plan.get("comparison_plan") or {}
    prior_input = plan.get("prior_input_embed_result") or {}
    prior_layer0 = plan.get("prior_layer0_result") or {}
    return {
        "status": plan.get("status"),
        "conclusion": plan.get("conclusion"),
        "first_probe": comparison.get("first_probe"),
        "llamacpp_effective_tap": comparison.get("llamacpp_effective_tap"),
        "hipengine_value_field": comparison.get("hipengine_value_field"),
        "prior_input_bf16_exact_match": prior_input.get("bf16_exact_match"),
        "prior_layer0_rmse": prior_layer0.get("rmse"),
        "next_action": plan.get("next_action"),
    }


def classify_layer0_attn_norm(
    status: str,
    exact_delta: Mapping[str, Any],
    bf16_delta: Mapping[str, Any],
) -> str:
    if status == "matched":
        return "layer0_attn_norm_exact_match"
    if status != "mismatched":
        return "layer0_attn_norm_comparison_unavailable"
    if bf16_delta.get("available") and bf16_delta.get("exact_match"):
        return "layer0_attn_norm_matches_after_bf16_roundtrip"
    if exact_delta.get("shape_match"):
        return "layer0_attn_norm_mismatch_after_bf16_roundtrip"
    return "layer0_attn_norm_comparison_unavailable"


def next_action(status: str, classification: str) -> str:
    if status == "matched":
        return "continue_layer0_subboundary_bisect_inside_linear_attention"
    if status == "llamacpp_capture_failed":
        return "inspect_layer0_attn_norm_llamacpp_capture_logs"
    if status == "skipped_no_hip_runtime":
        return "rerun_on_rocm_host"
    if classification == "layer0_attn_norm_matches_after_bf16_roundtrip":
        return "continue_layer0_subboundary_bisect_inside_linear_attention"
    if classification == "layer0_attn_norm_mismatch_after_bf16_roundtrip":
        return "audit_layer0_attn_norm_rmsnorm_or_weight_materialization"
    return "inspect_layer0_attn_norm_compare_failure"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
