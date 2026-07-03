#!/usr/bin/env python3
"""Compare patched llama.cpp pre-output_norm row to hipEngine serial pre-norm row."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
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
    pack_float32,
    sha256_bytes,
    summarize_capture,
)

DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter307-pre-output-norm-harness-compile.json"
)
DEFAULT_POST_OUTPUT_ARTIFACT = Path("benchmarks/results/mtp-gguf-iter303-hidden-seed-compare.json")
DEFAULT_OUTPUT_PREFIX = Path("/tmp/hipengine-llamacpp-mtp-iter308-pre-output-norm/pos16")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter308-pre-output-norm-compare.json")

HipPreCaptureFn = Callable[[Path, tuple[int, ...], int, int | None], dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--post-output-artifact", type=Path, default=DEFAULT_POST_OUTPUT_ARTIFACT)
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
    parser.add_argument("--iteration", type=int, default=308)
    args = parser.parse_args()

    artifact = compare_pre_output_norm(
        compile_artifact_path=args.compile_artifact,
        post_output_artifact_path=args.post_output_artifact,
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
                "pre_rmse": artifact["numeric_delta"].get("rmse"),
                "pre_max_abs_diff": artifact["numeric_delta"].get("max_abs_diff"),
                "classification": artifact["classification"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compare_pre_output_norm(
    *,
    compile_artifact_path: Path,
    post_output_artifact_path: Path,
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
    iteration: int = 308,
    hip_capture_fn: HipPreCaptureFn | None = None,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must be non-empty")
    if int(position) != len(prompt_tokens) - 1:
        raise ValueError("pre-output_norm comparison currently targets final prompt token")
    compile_artifact = json.loads(compile_artifact_path.read_text())
    post_artifact = json.loads(post_output_artifact_path.read_text())
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
    annotate_effective_pre_output_tap(llama_capture)
    hip_capture = (hip_capture_fn or capture_hipengine_pre_output_norm)(
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
    classification = classify_pre_vs_post(
        pre_delta=numeric_delta,
        post_artifact=post_artifact,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_vs_hipengine_pre_output_norm_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "compile_artifact_path": str(compile_artifact_path),
        "post_output_artifact_path": str(post_output_artifact_path),
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
        "post_output_reference": summarize_post_output_delta(post_artifact),
        "classification": classification,
        "external_checkout_modified": False,
        "next_action": next_action(status, classification),
    }


def annotate_effective_pre_output_tap(capture: dict[str, Any]) -> None:
    metadata = capture.get("metadata")
    if not isinstance(metadata, dict):
        return
    written_tap = metadata.get("tap")
    capture["effective_tap"] = "h_nextn_pre_output_norm"
    if written_tap != "h_nextn_pre_output_norm":
        capture["metadata_tap_note"] = (
            "generic hidden-seed harness writes its original tap label; the patched "
            "llama.cpp graph now keeps res->t_h_nextn from before output_norm"
        )


def capture_hipengine_pre_output_norm(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "values": []}
    from hipengine.runtime.qwen35_gguf_runner import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        Qwen35GGUFResidentSession,
        _copy_bf16_ptr_to_host_f32,
    )

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        for index, token_id in enumerate(prompt_tokens[:position]):
            session.step(int(token_id), position=index, return_logits=False)
        if session.runner is None or session.runner.weights is None or session.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if session._hidden_a is None or session._hidden_b is None:  # noqa: SLF001
            raise RuntimeError("GGUF resident session buffers are closed")
        runtime = session.runtime
        if runtime is None:
            raise RuntimeError("GGUF resident session runtime is unavailable")
        stream = 0
        session._set_full_attention_position_device(position, stream=stream)  # noqa: SLF001
        session._set_token_id_device(int(prompt_tokens[position]), stream=stream)  # noqa: SLF001
        session.scratch.position_host[0] = int(position)
        session.scratch.context_host[0] = int(position) + 1
        src = session._hidden_a  # noqa: SLF001
        dst = session._hidden_b  # noqa: SLF001
        for layer_id, layer_type in enumerate(session.runner.weights.config.layer_types):
            if layer_type == LINEAR_ATTENTION:
                session.runner._run_linear_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    session.scratch,
                    stream=stream,
                )
            elif layer_type == FULL_ATTENTION:
                session.runner._run_full_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    session.scratch,
                    position=position,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            src, dst = dst, src
        runtime.device_synchronize()
        values = _copy_bf16_ptr_to_host_f32(
            int(src.ptr),
            int(session.runner.hidden_size),
            runtime=runtime,
        )
        return {
            "status": "captured",
            "mode": "step-serial-pre-output_norm",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "dtype": "BF16_to_F32",
            "provenance": "final_decoder_output_before_output_norm",
            "values": [float(value) for value in values.tolist()],
        }


def classify_pre_vs_post(*, pre_delta: dict[str, Any], post_artifact: Mapping[str, Any]) -> str:
    if not pre_delta.get("available") or not pre_delta.get("shape_match"):
        return "pre_output_comparison_unavailable"
    post_delta = post_artifact.get("numeric_delta") or {}
    if (
        pre_delta.get("llamacpp_sha256")
        and pre_delta.get("llamacpp_sha256") == post_delta.get("llamacpp_sha256")
    ):
        return "llamacpp_pre_output_patch_overwritten_by_post_output_h_nextn"
    if pre_delta.get("exact_match"):
        return "pre_output_matches_post_output_mismatch_is_output_norm"
    pre_rmse = float(pre_delta.get("rmse") or 0.0)
    post_rmse = float(post_delta.get("rmse") or 0.0)
    if post_rmse > 0.0 and pre_rmse <= post_rmse * 0.1:
        return "pre_output_much_closer_output_norm_suspect"
    return "pre_output_mismatch_already_present"


def summarize_post_output_delta(post_artifact: Mapping[str, Any]) -> dict[str, Any]:
    delta = post_artifact.get("numeric_delta") or {}
    return {
        "status": post_artifact.get("status"),
        "rmse": delta.get("rmse"),
        "max_abs_diff": delta.get("max_abs_diff"),
        "mean_abs_diff": delta.get("mean_abs_diff"),
        "llamacpp_sha256": delta.get("llamacpp_sha256"),
        "hipengine_sha256": delta.get("hipengine_sha256"),
    }


def next_action(status: str, classification: str) -> str:
    if status == "matched":
        return "fix_or_match_output_norm_precision_for_mtp_seed"
    if classification == "pre_output_much_closer_output_norm_suspect":
        return "audit_output_norm_kernel_precision_against_llamacpp"
    if classification == "llamacpp_pre_output_patch_overwritten_by_post_output_h_nextn":
        return "move_or_replace_post_output_h_nextn_assignment_in_llamacpp_patch"
    if classification == "pre_output_mismatch_already_present":
        return "bisect_final_decoder_layer_output_before_output_norm"
    if status == "llamacpp_capture_failed":
        return "inspect_pre_output_norm_llamacpp_capture_logs"
    if status == "skipped_no_hip_runtime":
        return "rerun_on_rocm_host"
    return "inspect_pre_output_norm_compare_failure"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
