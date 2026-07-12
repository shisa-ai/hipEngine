#!/usr/bin/env python3
"""Plan llama.cpp/hipEngine initial token-embedding input capture."""

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
DEFAULT_EMBEDDING = Path("hipengine/runtime/gguf_embedding.py")
DEFAULT_LAYER0_ARTIFACT = Path("benchmarks/results/mtp-gguf-iter316-layer0-compare.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter317-initial-input-capture-plan.json")

INPUT_CAPTURE_LABEL = "h_nextn_input_embed"
INPUT_PATCH_ANCHOR = (
    "    inpL = build_inp_embd(model.tok_embd);\n\n"
    "    cb(inpL, \"model.input_embed\", -1);\n\n"
    "    auto * inp = build_inp_mem_hybrid();"
)
INPUT_PATCH_REPLACEMENT = (
    "    inpL = build_inp_embd(model.tok_embd);\n\n"
    "    cb(inpL, \"model.input_embed\", -1);\n"
    f"    cb(inpL, \"{INPUT_CAPTURE_LABEL}\", -1);\n"
    "    res->t_h_nextn = inpL;\n\n"
    "    auto * inp = build_inp_mem_hybrid();"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35moe", type=Path, default=DEFAULT_QWEN35MOE)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--layer0-artifact", type=Path, default=DEFAULT_LAYER0_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=317)
    args = parser.parse_args()

    artifact = build_initial_input_capture_plan(
        qwen35moe_path=args.qwen35moe,
        context_path=args.context,
        runner_path=args.runner,
        embedding_path=args.embedding,
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
                "llamacpp_input_patch_ready": artifact["llamacpp_input_patch"]["ready"],
                "hipengine_hidden_in_ready": artifact["hipengine_hidden_in_capture"]["ready"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_initial_input_capture_plan(
    *,
    qwen35moe_path: Path,
    context_path: Path,
    runner_path: Path,
    embedding_path: Path,
    layer0_artifact_path: Path,
    iteration: int = 317,
) -> dict[str, Any]:
    qwen_text = qwen35moe_path.read_text()
    context_text = context_path.read_text()
    runner_text = runner_path.read_text()
    embedding_text = embedding_path.read_text()
    layer0_artifact = read_json(layer0_artifact_path)
    llama_patch = audit_llamacpp_input_patch(qwen_text, context_text)
    hip_capture = audit_hipengine_hidden_in_capture(runner_text, embedding_text)
    prior = audit_prior_layer0_result(layer0_artifact)
    decision = decide_plan(
        prior=prior,
        llama_patch=llama_patch,
        hip_capture=hip_capture,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_hipengine_initial_input_capture_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": decision["status"],
        "qwen35moe_path": str(qwen35moe_path),
        "context_path": str(context_path),
        "runner_path": str(runner_path),
        "embedding_path": str(embedding_path),
        "layer0_artifact_path": str(layer0_artifact_path),
        "model": layer0_artifact.get("model"),
        "prompt_tokens": layer0_artifact.get("prompt_tokens"),
        "position": layer0_artifact.get("position"),
        "token_id": layer0_artifact.get("token_id"),
        "llamacpp_input_patch": llama_patch,
        "hipengine_hidden_in_capture": hip_capture,
        "prior_layer0_result": prior,
        "comparison_plan": build_comparison_plan(),
        "decision": decision,
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def audit_llamacpp_input_patch(qwen_text: str, context_text: str) -> dict[str, Any]:
    input_anchor_count = qwen_text.count(INPUT_PATCH_ANCHOR)
    post_old_count = qwen_text.count(POST_OUTPUT_OLD)
    post_new_count = qwen_text.count(POST_OUTPUT_NEW)
    layer_capture_count = qwen_text.count("h_nextn_layer_out")
    final_pre_output_count = qwen_text.count("h_nextn_pre_output_norm")
    context_ready = (
        "unmasked: nextn rows are stored densely" in context_text
        and "return embd_nextn.data + (size_t) i * n_embd" in context_text
    )
    preserve_ready = post_old_count == 1 or post_new_count == 1
    preserve_state = "needs_patch" if post_old_count == 1 else "already_patched"
    return {
        "ready": input_anchor_count == 1 and preserve_ready and context_ready,
        "input_anchor_count": input_anchor_count,
        "capture_label": INPUT_CAPTURE_LABEL,
        "patch_scope": "temporary copied llama.cpp source tree only",
        "input_capture_old_text": INPUT_PATCH_ANCHOR,
        "input_capture_new_text": INPUT_PATCH_REPLACEMENT,
        "post_output_preserve_state": preserve_state if preserve_ready else "missing_anchor",
        "post_output_old_count": post_old_count,
        "post_output_new_count": post_new_count,
        "post_output_preserve_old_text": POST_OUTPUT_OLD,
        "post_output_preserve_new_text": POST_OUTPUT_NEW,
        "layer_capture_patch_must_be_absent": True,
        "layer_capture_patch_count": layer_capture_count,
        "final_pre_output_patch_must_be_absent": True,
        "final_pre_output_patch_count": final_pre_output_count,
        "anchors": {
            "build_inp_embd": find_line(qwen_text, "inpL = build_inp_embd"),
            "model_input_embed_cb": find_line(qwen_text, 'cb(inpL, "model.input_embed", -1)'),
            "first_layer_loop": find_line(qwen_text, "for (int il = 0; il < n_layer; ++il)"),
            "post_output_h_nextn": find_line(qwen_text, 'cb(cur, "h_nextn", -1)'),
            "context_unmasked_get_ith": find_line(
                context_text,
                "unmasked: nextn rows are stored densely",
            ),
        },
        "notes": [
            "Use a clean base source without h_nextn_layer_out or "
            "h_nextn_pre_output_norm so the input embedding tap cannot be overwritten.",
            "Apply the post-output preserve patch so the later h_nextn assignment "
            "cannot overwrite the input embedding tensor.",
        ],
    }


def audit_hipengine_hidden_in_capture(
    runner_text: str,
    embedding_text: str,
) -> dict[str, Any]:
    body = extract_function_body(runner_text, "capture_attention_layer")
    facts = {
        "capture_attention_layer_available": bool(body),
        "captures_hidden_in_f32": "hidden_in_f32" in body,
        "copies_hidden_in_from_target_src_ptr": "target_src_ptr" in body
        and "hidden_in_f32=_copy_bf16_ptr_to_host_f32" in body,
        "sets_token_id_device": "_set_token_id_device" in body,
        "runs_layer_after_hidden_in_copy_pointer_selection": (
            "target_src_ptr = int(src.ptr)" in body
        ),
        "embedding_launcher_available": "def launch_gguf_embedding" in embedding_text,
        "embedding_output_bf16": "GGUF_EMBEDDING_OUTPUT_BF16" in embedding_text,
        "embedding_dispatch_registry_driven": (
            "KernelKey" in embedding_text and "resolve(" in embedding_text
        ),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "capture_strategy": (
            "Warm prior prompt tokens with session.step(...). Then call "
            "session.capture_attention_layer(token_id, position=position, layer_id=0, "
            "run_preceding_layers=False) and compare capture.hidden_in_f32 to the "
            "patched llama.cpp h_nextn_input_embed row."
        ),
        "value_field": "hidden_in_f32",
        "dtype_note": (
            "hipEngine launch_gguf_embedding writes BF16 and hidden_in_f32 copies it "
            "back as F32; the llama.cpp input_embed tensor is captured as F32. The "
            "next comparison should report both exact F32 delta and a BF16-rounded "
            "llama.cpp row delta before deciding this is a semantic mismatch."
        ),
        "anchors": {
            "capture_attention_layer": find_line(runner_text, "def capture_attention_layer"),
            "hidden_in_f32": find_line(runner_text, "hidden_in_f32"),
            "set_token_id_device": find_line(runner_text, "_set_token_id_device"),
            "launch_gguf_embedding": find_line(embedding_text, "def launch_gguf_embedding"),
            "embedding_output_bf16": find_line(embedding_text, "GGUF_EMBEDDING_OUTPUT_BF16"),
        },
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
        "llamacpp_layer0_sha256": numeric.get("llamacpp_sha256"),
        "hipengine_layer0_sha256": numeric.get("hipengine_sha256"),
        "next_action": layer0_artifact.get("next_action"),
    }


def build_comparison_plan() -> dict[str, Any]:
    return {
        "first_probe": "input_embed_vs_hidden_in_layer0",
        "llamacpp_effective_tap": INPUT_CAPTURE_LABEL,
        "hipengine_value_field": "hidden_in_f32",
        "expected_artifacts": [
            "mtp-gguf-iter318-input-embed-harness-build.json",
            "mtp-gguf-iter318-input-embed-llamacpp-build-result.json",
            "mtp-gguf-iter318-input-embed-harness-compile.json",
            "mtp-gguf-iter318-input-embed-compare.json",
        ],
        "interpretation": {
            "exact_or_bf16_rounded_match": (
                "initial embedding is aligned; investigate layer-0 implementation details"
            ),
            "mismatch": (
                "token embedding lookup/materialization differs before the first decoder layer"
            ),
        },
    }


def decide_plan(
    *,
    prior: Mapping[str, Any],
    llama_patch: Mapping[str, Any],
    hip_capture: Mapping[str, Any],
) -> dict[str, str]:
    if prior.get("ready") and llama_patch.get("ready") and hip_capture.get("ready"):
        return {
            "status": "ready",
            "conclusion": "initial_input_capture_plan_ready",
            "next_action": "build_input_embed_capture_and_compare_hidden_in_f32",
        }
    return {
        "status": "blocked",
        "conclusion": "initial_input_capture_plan_missing_required_fact",
        "next_action": "inspect_initial_input_capture_plan_blockers",
    }


if __name__ == "__main__":
    main()
