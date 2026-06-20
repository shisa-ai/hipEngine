#!/usr/bin/env python3
"""Plan llama.cpp/hipEngine final-decoder layer-boundary bisection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QWEN35MOE = Path("/home/lhl/llama.cpp/llama.cpp-hip/src/models/qwen35moe.cpp")
DEFAULT_CONTEXT = Path("/home/lhl/llama.cpp/llama.cpp-hip/src/llama-context.cpp")
DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_COMPARE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter309-pre-output-norm-compare.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter310-layer-boundary-bisect-plan.json")

LAYER_CAPTURE_LABEL = "h_nextn_layer_out"
LAYER_PATCH_ANCHOR = (
    "        cur = build_cvec(cur, il);\n"
    "        cb(cur, \"l_out\", il);\n\n"
    "        // Input for next layer\n"
    "        inpL = cur;"
)
LAYER_PATCH_TEMPLATE = (
    "        cur = build_cvec(cur, il);\n"
    "        cb(cur, \"l_out\", il);\n\n"
    "        if (il == {layer_id}) {{\n"
    f"            cb(cur, \"{LAYER_CAPTURE_LABEL}\", il);\n"
    "            res->t_h_nextn = cur;\n"
    "        }\n\n"
    "        // Input for next layer\n"
    "        inpL = cur;"
)
POST_OUTPUT_OLD = (
    "    cb(cur, \"h_nextn\", -1);\n"
    "    res->t_h_nextn = cur;\n\n"
    "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {"
)
POST_OUTPUT_NEW = (
    "    cb(cur, \"h_nextn_post_output_norm\", -1);\n"
    "    // PRE-output_norm diagnostic: keep res->t_h_nextn from before output_norm.\n\n"
    "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35moe", type=Path, default=DEFAULT_QWEN35MOE)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--compare-artifact", type=Path, default=DEFAULT_COMPARE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=310)
    args = parser.parse_args()

    artifact = build_layer_boundary_bisect_plan(
        qwen35moe_path=args.qwen35moe,
        context_path=args.context,
        runner_path=args.runner,
        compare_artifact_path=args.compare_artifact,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "llamacpp_layer_patch_ready": artifact["llamacpp_layer_patch"]["ready"],
                "hipengine_capture_ready": artifact["hipengine_layer_capture"]["ready"],
                "first_probe_layer": artifact["bisection_targets"]["first_probe_layer"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_layer_boundary_bisect_plan(
    *,
    qwen35moe_path: Path,
    context_path: Path,
    runner_path: Path,
    compare_artifact_path: Path,
    iteration: int = 310,
) -> dict[str, Any]:
    qwen_text = qwen35moe_path.read_text()
    context_text = context_path.read_text()
    runner_text = runner_path.read_text()
    compare_artifact = read_json(compare_artifact_path)
    layer_count = infer_model_layer_count(qwen_text, compare_artifact)
    llama_patch = audit_llamacpp_layer_patch(qwen_text, context_text)
    hip_capture = audit_hipengine_layer_capture(runner_text)
    prior = audit_prior_pre_output_result(compare_artifact)
    targets = build_bisection_targets(layer_count)
    decision = decide_plan(
        prior=prior,
        llama_patch=llama_patch,
        hip_capture=hip_capture,
        layer_count=layer_count,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_hipengine_layer_boundary_bisect_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": decision["status"],
        "qwen35moe_path": str(qwen35moe_path),
        "context_path": str(context_path),
        "runner_path": str(runner_path),
        "compare_artifact_path": str(compare_artifact_path),
        "model": compare_artifact.get("model"),
        "prompt_tokens": compare_artifact.get("prompt_tokens"),
        "position": compare_artifact.get("position"),
        "token_id": compare_artifact.get("token_id"),
        "selected_layer_count": layer_count,
        "llamacpp_layer_patch": llama_patch,
        "hipengine_layer_capture": hip_capture,
        "prior_pre_output_result": prior,
        "bisection_targets": targets,
        "execution_plan": build_execution_plan(targets),
        "decision": decision,
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def audit_llamacpp_layer_patch(qwen_text: str, context_text: str) -> dict[str, Any]:
    layer_anchor_count = qwen_text.count(LAYER_PATCH_ANCHOR)
    post_old_count = qwen_text.count(POST_OUTPUT_OLD)
    post_new_count = qwen_text.count(POST_OUTPUT_NEW)
    final_pre_output_patch_count = qwen_text.count("h_nextn_pre_output_norm")
    context_ready = (
        "unmasked: nextn rows are stored densely" in context_text
        and "return embd_nextn.data + (size_t) i * n_embd" in context_text
    )
    preserve_state = "needs_patch" if post_old_count == 1 else "already_patched"
    preserve_ready = post_old_count == 1 or post_new_count == 1
    return {
        "ready": layer_anchor_count == 1 and preserve_ready and context_ready,
        "layer_anchor_count": layer_anchor_count,
        "post_output_preserve_state": preserve_state if preserve_ready else "missing_anchor",
        "post_output_old_count": post_old_count,
        "post_output_new_count": post_new_count,
        "final_pre_output_patch_count": final_pre_output_patch_count,
        "final_pre_output_patch_must_be_absent": True,
        "capture_label": LAYER_CAPTURE_LABEL,
        "patch_scope": "temporary copied llama.cpp source tree only",
        "layer_capture_old_text": LAYER_PATCH_ANCHOR,
        "layer_capture_new_text_template": LAYER_PATCH_TEMPLATE,
        "post_output_preserve_old_text": POST_OUTPUT_OLD,
        "post_output_preserve_new_text": POST_OUTPUT_NEW,
        "anchors": {
            "layer_loop": find_line(qwen_text, "for (int il = 0; il < n_layer; ++il)"),
            "layer_out": find_line(qwen_text, 'cb(cur, "l_out", il)'),
            "next_layer_input": find_line(qwen_text, "inpL = cur;"),
            "post_output_h_nextn": find_line(qwen_text, 'cb(cur, "h_nextn", -1)'),
            "context_unmasked_get_ith": find_line(
                context_text,
                "unmasked: nextn rows are stored densely",
            ),
        },
        "notes": [
            "Use a clean base source or a source without h_nextn_pre_output_norm; "
            "the final pre-output patch would overwrite the selected layer tap.",
            "Apply the post-output preserve patch so the later h_nextn assignment "
            "cannot overwrite the selected layer-out tensor.",
        ],
    }


def audit_hipengine_layer_capture(runner_text: str) -> dict[str, Any]:
    body = extract_function_body(runner_text, "capture_attention_layer")
    facts = {
        "capture_attention_layer_available": bool(body),
        "supports_run_preceding_layers": "run_preceding_layers" in body,
        "captures_layer_out_f32": "layer_out_f32" in body,
        "captures_hidden_in_f32": "hidden_in_f32" in body,
        "supports_linear_attention": "_run_linear_attention_layer" in body,
        "supports_full_attention": "_run_full_attention_layer" in body,
        "sets_final_token_id": "_set_token_id_device" in body,
        "session_step_available_for_prior_tokens": "def step(" in runner_text,
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "capture_strategy": (
            "Warm the resident session by calling session.step(...) for prompt tokens "
            "before the target position. Then call session.capture_attention_layer("
            "token_id, position=position, layer_id=target_layer, "
            "run_preceding_layers=True) and compare layer_out_f32 to llama.cpp "
            "embeddings_nextn for the same target layer."
        ),
        "value_field": "layer_out_f32",
        "anchors": {
            "capture_attention_layer": find_line(runner_text, "def capture_attention_layer"),
            "run_preceding_layers": find_line(runner_text, "run_preceding_layers"),
            "layer_out_f32": find_line(runner_text, "layer_out_f32"),
            "session_step": find_line(runner_text, "def step("),
        },
    }


def audit_prior_pre_output_result(compare_artifact: Mapping[str, Any]) -> dict[str, Any]:
    numeric = compare_artifact.get("numeric_delta") or {}
    return {
        "ready": compare_artifact.get("classification") == "pre_output_mismatch_already_present",
        "status": compare_artifact.get("status"),
        "classification": compare_artifact.get("classification"),
        "rmse": numeric.get("rmse"),
        "max_abs_diff": numeric.get("max_abs_diff"),
        "mean_abs_diff": numeric.get("mean_abs_diff"),
        "llamacpp_pre_output_sha256": numeric.get("llamacpp_sha256"),
        "hipengine_pre_output_sha256": numeric.get("hipengine_sha256"),
        "next_action": compare_artifact.get("next_action"),
    }


def build_bisection_targets(layer_count: int) -> dict[str, Any]:
    if layer_count <= 0:
        return {"ready": False, "order": [], "first_probe_layer": None}
    final_layer = layer_count - 1
    order = [final_layer]
    for layer in binary_probe_order(0, final_layer - 1):
        if layer not in order:
            order.append(layer)
    return {
        "ready": True,
        "layer_count": layer_count,
        "first_probe_layer": final_layer,
        "order": order,
        "recommended_first_batch": order[:4],
        "interpretation": {
            "final_layer_matches_iter309_pre_output_delta": (
                "the layer-out tap is aligned; continue binary search earlier"
            ),
            "first_matching_midpoint": (
                "search after that layer; the first mismatch is in later layers"
            ),
            "first_mismatching_midpoint": (
                "search before or at that layer; the first mismatch is earlier"
            ),
        },
    }


def binary_probe_order(low: int, high: int) -> list[int]:
    if high < low:
        return []
    order: list[int] = []
    queue: list[tuple[int, int]] = [(int(low), int(high))]
    while queue:
        start, stop = queue.pop(0)
        if stop < start:
            continue
        mid = (start + stop) // 2
        order.append(mid)
        queue.append((start, mid - 1))
        queue.append((mid + 1, stop))
    return order


def build_execution_plan(targets: Mapping[str, Any]) -> dict[str, Any]:
    first_layer = targets.get("first_probe_layer")
    return {
        "steps": [
            "copy the clean llama.cpp source_dir to a new /tmp layer-boundary source tree",
            f"replace the unique l_out block with {LAYER_CAPTURE_LABEL} for one layer_id",
            "apply the post-output preserve patch; do not apply the final pre-output patch",
            "configure/build libllama in a new temporary build directory",
            "compile the existing hidden-seed capture harness against that build",
            "run the harness with --all-rows for the oracle prompt and target layer",
            "warm hipEngine with prior prompt tokens, then call capture_attention_layer "
            "with run_preceding_layers=True for the same target layer",
            "compare llama.cpp h_nextn_layer_out with hipEngine layer_out_f32",
        ],
        "first_probe_layer": first_layer,
        "first_probe_effective_tap": LAYER_CAPTURE_LABEL,
        "temporary_source_hint": f"/tmp/hipengine-llamacpp-mtp-iter311-layer{first_layer}-src",
        "temporary_build_hint": f"/tmp/hipengine-llamacpp-mtp-iter311-layer{first_layer}-build",
        "artifact_prefix_hint": f"/tmp/hipengine-llamacpp-mtp-iter311-layer{first_layer}/pos16",
    }


def decide_plan(
    *,
    prior: Mapping[str, Any],
    llama_patch: Mapping[str, Any],
    hip_capture: Mapping[str, Any],
    layer_count: int,
) -> dict[str, str]:
    if prior.get("ready") and llama_patch.get("ready") and hip_capture.get("ready"):
        return {
            "status": "ready",
            "conclusion": "layer_boundary_bisect_plan_ready",
            "next_action": "build_layer_boundary_capture_for_layer_39_and_compare",
        }
    return {
        "status": "blocked",
        "conclusion": "layer_boundary_bisect_plan_missing_required_fact",
        "next_action": "inspect_layer_boundary_bisect_plan_blockers",
    }


def infer_model_layer_count(qwen_text: str, compare_artifact: Mapping[str, Any]) -> int:
    model = str(compare_artifact.get("model") or "")
    if "35B" in model:
        return 40
    if "122B" in model:
        return 48
    if "397B" in model:
        return 60
    case_values = [int(value) for value in re.findall(r"case\s+(\d+)\s*:", qwen_text)]
    return case_values[0] if case_values else 0


def extract_function_body(text: str, name: str) -> str:
    pattern = re.compile(rf"^(?P<indent>\s*)def {re.escape(name)}\b.*$", re.M)
    match = pattern.search(text)
    if match is None:
        return ""
    start = match.start()
    indent = len(match.group("indent"))
    rest = text[match.end() :]
    end = len(text)
    for next_match in re.finditer(r"^\s*def \w+\b|^\s*@", rest, re.M):
        line = next_match.group(0)
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= indent:
            end = match.end() + next_match.start()
            break
    return text[start:end]


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
