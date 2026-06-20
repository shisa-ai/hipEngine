#!/usr/bin/env python3
"""Plan the layer-0 sub-boundary diagnostic after embedding alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_layer_boundary_bisect_plan import (  # noqa: E402
    POST_OUTPUT_NEW,
    POST_OUTPUT_OLD,
    extract_function_body,
    find_line,
    read_json,
)

DEFAULT_QWEN35MOE = Path("/home/lhl/llama.cpp/llama.cpp-hip/src/models/qwen35moe.cpp")
DEFAULT_CONTEXT = Path("/home/lhl/llama.cpp/llama.cpp-hip/src/llama-context.cpp")
DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_INPUT_ARTIFACT = Path("benchmarks/results/mtp-gguf-iter318-input-embed-compare.json")
DEFAULT_LAYER0_ARTIFACT = Path("benchmarks/results/mtp-gguf-iter316-layer0-compare.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter319-layer0-subboundary-plan.json")

ATTN_NORM_CAPTURE_LABEL = "h_nextn_layer0_attn_norm"
ATTN_NORM_PATCH_ANCHOR = (
    "        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, "
    "LLM_NORM_RMS, il);\n"
    "        cb(cur, \"attn_norm\", il);\n\n"
    "        ggml_build_forward_expand(gf, cur);"
)
ATTN_NORM_PATCH_REPLACEMENT = (
    "        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, "
    "LLM_NORM_RMS, il);\n"
    "        cb(cur, \"attn_norm\", il);\n\n"
    "        if (il == 0) {\n"
    f"            cb(cur, \"{ATTN_NORM_CAPTURE_LABEL}\", il);\n"
    "            res->t_h_nextn = cur;\n"
    "        }\n\n"
    "        ggml_build_forward_expand(gf, cur);"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35moe", type=Path, default=DEFAULT_QWEN35MOE)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--input-artifact", type=Path, default=DEFAULT_INPUT_ARTIFACT)
    parser.add_argument("--layer0-artifact", type=Path, default=DEFAULT_LAYER0_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=319)
    args = parser.parse_args()

    artifact = build_layer0_subboundary_plan(
        qwen35moe_path=args.qwen35moe,
        context_path=args.context,
        runner_path=args.runner,
        input_artifact_path=args.input_artifact,
        layer0_artifact_path=args.layer0_artifact,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "llamacpp_attn_norm_patch_ready": artifact[
                    "llamacpp_attn_norm_patch"
                ]["ready"],
                "hipengine_linear_boundary_ready": artifact[
                    "hipengine_linear_boundary_capture"
                ]["ready"],
                "first_probe": artifact["comparison_plan"]["first_probe"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_layer0_subboundary_plan(
    *,
    qwen35moe_path: Path,
    context_path: Path,
    runner_path: Path,
    input_artifact_path: Path,
    layer0_artifact_path: Path,
    iteration: int = 319,
) -> dict[str, Any]:
    qwen_text = qwen35moe_path.read_text()
    context_text = context_path.read_text()
    runner_text = runner_path.read_text()
    input_artifact = read_json(input_artifact_path)
    layer0_artifact = read_json(layer0_artifact_path)
    llama_patch = audit_llamacpp_attn_norm_patch(qwen_text, context_text)
    hip_capture = audit_hipengine_linear_boundary_capture(runner_text)
    prior_input = audit_prior_input_embed_result(input_artifact)
    prior_layer0 = audit_prior_layer0_result(layer0_artifact)
    decision = decide_plan(
        prior_input=prior_input,
        prior_layer0=prior_layer0,
        llama_patch=llama_patch,
        hip_capture=hip_capture,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_hipengine_layer0_subboundary_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": decision["status"],
        "qwen35moe_path": str(qwen35moe_path),
        "context_path": str(context_path),
        "runner_path": str(runner_path),
        "input_artifact_path": str(input_artifact_path),
        "layer0_artifact_path": str(layer0_artifact_path),
        "model": input_artifact.get("model"),
        "prompt_tokens": input_artifact.get("prompt_tokens"),
        "position": input_artifact.get("position"),
        "token_id": input_artifact.get("token_id"),
        "llamacpp_attn_norm_patch": llama_patch,
        "hipengine_linear_boundary_capture": hip_capture,
        "prior_input_embed_result": prior_input,
        "prior_layer0_result": prior_layer0,
        "comparison_plan": build_comparison_plan(),
        "decision": decision,
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def audit_llamacpp_attn_norm_patch(qwen_text: str, context_text: str) -> dict[str, Any]:
    attn_norm_anchor_count = qwen_text.count(ATTN_NORM_PATCH_ANCHOR)
    post_old_count = qwen_text.count(POST_OUTPUT_OLD)
    post_new_count = qwen_text.count(POST_OUTPUT_NEW)
    input_capture_count = qwen_text.count("h_nextn_input_embed")
    layer_capture_count = qwen_text.count("h_nextn_layer_out")
    final_pre_output_count = qwen_text.count("h_nextn_pre_output_norm")
    context_ready = (
        "unmasked: nextn rows are stored densely" in context_text
        and "return embd_nextn.data + (size_t) i * n_embd" in context_text
    )
    preserve_ready = post_old_count == 1 or post_new_count == 1
    preserve_state = "needs_patch" if post_old_count == 1 else "already_patched"
    return {
        "ready": attn_norm_anchor_count == 1 and preserve_ready and context_ready,
        "capture_label": ATTN_NORM_CAPTURE_LABEL,
        "patch_scope": "temporary copied llama.cpp source tree only",
        "attn_norm_anchor_count": attn_norm_anchor_count,
        "attn_norm_capture_old_text": ATTN_NORM_PATCH_ANCHOR,
        "attn_norm_capture_new_text": ATTN_NORM_PATCH_REPLACEMENT,
        "post_output_preserve_state": preserve_state if preserve_ready else "missing_anchor",
        "post_output_old_count": post_old_count,
        "post_output_new_count": post_new_count,
        "post_output_preserve_old_text": POST_OUTPUT_OLD,
        "post_output_preserve_new_text": POST_OUTPUT_NEW,
        "input_capture_patch_must_be_absent": True,
        "input_capture_patch_count": input_capture_count,
        "layer_capture_patch_must_be_absent": True,
        "layer_capture_patch_count": layer_capture_count,
        "final_pre_output_patch_must_be_absent": True,
        "final_pre_output_patch_count": final_pre_output_count,
        "expected_shape": "hidden_size",
        "anchors": {
            "layer_loop": find_line(qwen_text, "for (int il = 0; il < n_layer; ++il)"),
            "attn_norm_build": find_line(qwen_text, "cur = build_norm(inpL"),
            "attn_norm_cb": find_line(qwen_text, 'cb(cur, "attn_norm", il)'),
            "linear_attn_branch": find_line(qwen_text, "build_layer_attn_linear"),
            "post_output_h_nextn": find_line(qwen_text, 'cb(cur, "h_nextn", -1)'),
            "context_unmasked_get_ith": find_line(
                context_text,
                "unmasked: nextn rows are stored densely",
            ),
        },
        "notes": [
            "Use a clean base source without input/layer/final diagnostic taps so the "
            "attn_norm tap cannot be overwritten.",
            "Apply the post-output preserve patch so the later h_nextn assignment "
            "cannot overwrite the selected attn_norm tensor.",
        ],
    }


def audit_hipengine_linear_boundary_capture(runner_text: str) -> dict[str, Any]:
    body = extract_function_body(runner_text, "capture_linear_attention_boundary")
    facts = {
        "capture_linear_attention_boundary_available": bool(body),
        "requires_linear_attention_layer": "not a linear_attention layer" in body,
        "sets_token_id_device": "_set_token_id_device" in body,
        "runs_attn_only": "_run_linear_attention_attn_only" in body,
        "captures_attn_norm_f32": "attn_norm_f32=" in body,
        "attn_norm_copied_from_bf16_norm_scratch": (
            "int(self.scratch.norm.ptr), hidden_size" in body
            and "_copy_bf16_ptr_to_host_f32" in body
        ),
        "captures_linear_qkv_f32": "linear_qkv_f32=" in body,
        "captures_linear_z_f32": "linear_z_f32=" in body,
        "captures_attn_out_f32": "attn_out_f32=" in body,
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "capture_strategy": (
            "Warm prior prompt tokens with session.step(...). Then call "
            "session.capture_linear_attention_boundary(token_id, position=position, "
            "layer_id=0) and compare capture.attn_norm_f32 to the patched llama.cpp "
            "h_nextn_layer0_attn_norm row."
        ),
        "first_value_field": "attn_norm_f32",
        "later_value_fields": [
            "linear_qkv_f32",
            "linear_z_f32",
            "conv_out_f32",
            "recurrent_out_f32",
            "attn_out_f32",
        ],
        "dtype_note": (
            "hipEngine attn_norm_f32 copies the BF16 scratch norm back to F32. The "
            "next comparison should report both exact F32 delta and a BF16-rounded "
            "llama.cpp attn_norm row delta, matching the input-embedding diagnostic."
        ),
        "anchors": {
            "capture_linear_attention_boundary": find_line(
                runner_text,
                "def capture_linear_attention_boundary",
            ),
            "run_attn_only": find_line(runner_text, "_run_linear_attention_attn_only"),
            "attn_norm_f32": find_line(runner_text, "attn_norm_f32="),
            "linear_qkv_f32": find_line(runner_text, "linear_qkv_f32="),
            "attn_out_f32": find_line(runner_text, "attn_out_f32="),
        },
    }


def audit_prior_input_embed_result(input_artifact: Mapping[str, Any]) -> dict[str, Any]:
    exact = input_artifact.get("numeric_delta") or {}
    bf16 = input_artifact.get("bf16_rounded_delta") or {}
    return {
        "ready": input_artifact.get("classification")
        == "input_embed_matches_after_bf16_roundtrip",
        "status": input_artifact.get("status"),
        "classification": input_artifact.get("classification"),
        "exact_rmse": exact.get("rmse"),
        "exact_max_abs_diff": exact.get("max_abs_diff"),
        "bf16_exact_match": bf16.get("exact_match"),
        "bf16_rmse": bf16.get("rmse"),
        "bf16_max_abs_diff": bf16.get("max_abs_diff"),
        "llamacpp_input_sha256": exact.get("llamacpp_sha256"),
        "hipengine_hidden_in_sha256": exact.get("hipengine_sha256"),
        "llamacpp_bf16_roundtrip_sha256": bf16.get("llamacpp_sha256"),
    }


def audit_prior_layer0_result(layer0_artifact: Mapping[str, Any]) -> dict[str, Any]:
    numeric = layer0_artifact.get("numeric_delta") or {}
    return {
        "ready": layer0_artifact.get("classification") == "layer_boundary_mismatch"
        and layer0_artifact.get("layer_id") == 0,
        "status": layer0_artifact.get("status"),
        "classification": layer0_artifact.get("classification"),
        "layer_id": layer0_artifact.get("layer_id"),
        "rmse": numeric.get("rmse"),
        "max_abs_diff": numeric.get("max_abs_diff"),
        "mean_abs_diff": numeric.get("mean_abs_diff"),
        "next_action": layer0_artifact.get("next_action"),
    }


def build_comparison_plan() -> dict[str, Any]:
    return {
        "first_probe": "layer0_attn_norm_vs_linear_boundary_attn_norm_f32",
        "llamacpp_effective_tap": ATTN_NORM_CAPTURE_LABEL,
        "hipengine_value_field": "attn_norm_f32",
        "expected_artifacts": [
            "mtp-gguf-iter320-layer0-attn-norm-harness-build.json",
            "mtp-gguf-iter320-layer0-attn-norm-llamacpp-build-result.json",
            "mtp-gguf-iter320-layer0-attn-norm-harness-compile.json",
            "mtp-gguf-iter320-layer0-attn-norm-compare.json",
        ],
        "interpretation": {
            "bf16_rounded_match": (
                "attn_norm is aligned; continue inside linear attention at qkv/z projections"
            ),
            "mismatch": "the first layer-0 difference is the attention RMSNorm output",
        },
    }


def decide_plan(
    *,
    prior_input: Mapping[str, Any],
    prior_layer0: Mapping[str, Any],
    llama_patch: Mapping[str, Any],
    hip_capture: Mapping[str, Any],
) -> dict[str, str]:
    if (
        prior_input.get("ready")
        and prior_layer0.get("ready")
        and llama_patch.get("ready")
        and hip_capture.get("ready")
    ):
        return {
            "status": "ready",
            "conclusion": "layer0_attn_norm_capture_plan_ready",
            "next_action": "build_layer0_attn_norm_capture_and_compare",
        }
    return {
        "status": "blocked",
        "conclusion": "layer0_attn_norm_capture_plan_missing_required_fact",
        "next_action": "inspect_layer0_subboundary_plan_blockers",
    }


if __name__ == "__main__":
    main()
