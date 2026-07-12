#!/usr/bin/env python3
"""Plan the pre-output_norm hidden-row bisection for GGUF MTP parity."""

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
DEFAULT_COMPILE_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter302-llamacpp-hidden-seed-capture-harness-compile.json"
)
DEFAULT_SHARED_AUDIT = Path(
    "benchmarks/results/mtp-gguf-iter305-hidden-seed-shared-path-audit.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter306-pre-output-norm-capture-plan.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35moe", type=Path, default=DEFAULT_QWEN35MOE)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--compile-artifact", type=Path, default=DEFAULT_COMPILE_ARTIFACT)
    parser.add_argument("--shared-audit", type=Path, default=DEFAULT_SHARED_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=306)
    args = parser.parse_args()

    artifact = build_pre_output_norm_capture_plan(
        qwen35moe_path=args.qwen35moe,
        context_path=args.context,
        runner_path=args.runner,
        compile_artifact_path=args.compile_artifact,
        shared_audit_path=args.shared_audit,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "llamacpp_patch_ready": artifact["llamacpp_pre_output_patch"]["ready"],
                "hipengine_capture_ready": artifact["hipengine_pre_output_capture"]["ready"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_pre_output_norm_capture_plan(
    *,
    qwen35moe_path: Path,
    context_path: Path,
    runner_path: Path,
    compile_artifact_path: Path,
    shared_audit_path: Path,
    iteration: int = 306,
) -> dict[str, Any]:
    qwen_text = qwen35moe_path.read_text()
    context_text = context_path.read_text()
    runner_text = runner_path.read_text()
    compile_artifact = read_json(compile_artifact_path)
    shared_audit = read_json(shared_audit_path)
    llama_patch = audit_llamacpp_pre_output_patch(qwen_text, context_text)
    hip_capture = audit_hipengine_pre_output_capture_path(runner_text)
    build_inputs = audit_build_inputs(compile_artifact)
    decision = decide_plan(
        shared_audit=shared_audit,
        llama_patch=llama_patch,
        hip_capture=hip_capture,
        build_inputs=build_inputs,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_hipengine_pre_output_norm_capture_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": decision["status"],
        "qwen35moe_path": str(qwen35moe_path),
        "context_path": str(context_path),
        "runner_path": str(runner_path),
        "compile_artifact_path": str(compile_artifact_path),
        "shared_audit_path": str(shared_audit_path),
        "llamacpp_pre_output_patch": llama_patch,
        "hipengine_pre_output_capture": hip_capture,
        "build_inputs": build_inputs,
        "execution_plan": build_execution_plan(build_inputs),
        "decision": decision,
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def audit_llamacpp_pre_output_patch(qwen_text: str, context_text: str) -> dict[str, Any]:
    anchor = (
        "    cur = inpL;\n\n"
        "    // post-norm hidden state feeds both the LM head and the MTP seed below\n"
        "    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);"
    )
    replacement = (
        "    cur = inpL;\n\n"
        "    // PRE-output_norm diagnostic: expose final decoder output through h_nextn.\n"
        "    cb(cur, \"h_nextn_pre_output_norm\", -1);\n"
        "    res->t_h_nextn = cur;\n\n"
        "    // post-norm hidden state feeds both the LM head and the MTP seed below\n"
        "    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);"
    )
    count = qwen_text.count(anchor)
    post_norm_h_nextn = (
        "cur = build_norm(cur, model.output_norm" in qwen_text
        and 'cb(cur, "h_nextn", -1)' in qwen_text
        and "res->t_h_nextn = cur" in qwen_text
    )
    context_ready = (
        "unmasked: nextn rows are stored densely" in context_text
        and "return embd_nextn.data + (size_t) i * n_embd" in context_text
    )
    return {
        "ready": count == 1 and post_norm_h_nextn and context_ready,
        "anchor_count": count,
        "post_norm_h_nextn_currently_present": post_norm_h_nextn,
        "context_unmasked_get_ith_ready": context_ready,
        "patch_scope": "temporary copied llama.cpp source tree only",
        "old_text": anchor,
        "new_text": replacement,
        "anchors": {
            "cur_inpL_before_output_norm": find_line(qwen_text, "cur = inpL;"),
            "output_norm": find_line(qwen_text, "cur = build_norm(cur, model.output_norm"),
            "post_norm_h_nextn": find_line(qwen_text, 'cb(cur, "h_nextn", -1)'),
            "context_unmasked_get_ith": find_line(
                context_text,
                "unmasked: nextn rows are stored densely",
            ),
        },
    }


def audit_hipengine_pre_output_capture_path(runner_text: str) -> dict[str, Any]:
    current_body = extract_function_body(runner_text, "_run_current_hidden_to_final_hidden")
    output_norm_body = extract_function_body(runner_text, "_run_output_norm_hidden")
    facts = {
        "serial_loop_has_src_before_output_norm": "for layer_id, layer_type" in current_body
        and "src, dst = dst, src" in current_body,
        "output_norm_called_with_src_ptr": "return self._run_output_norm_hidden(" in current_body
        and "src.ptr" in current_body,
        "bf16_copy_helper_available": "def _copy_bf16_ptr_to_host_f32" in runner_text,
        "output_norm_fp32_seed_uses_same_src_ptr": "gguf_rmsnorm_bf16_f32_weight_out_f32"
        in output_norm_body
        and "src_ptr" in output_norm_body,
        "private_methods_can_replay_serial_path": "def _set_token_id_device" in runner_text
        and "def _set_full_attention_position_device" in runner_text,
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "capture_strategy": (
            "For tokens before the final position, call session.step(..., return_logits=False). "
            "For the final token, call the private token setup helpers, replay the same "
            "layer loop as _run_current_hidden_to_final_hidden, then copy the final src.ptr "
            "with _copy_bf16_ptr_to_host_f32 before _run_output_norm_hidden."
        ),
        "anchors": {
            "current_to_final": find_line(runner_text, "def _run_current_hidden_to_final_hidden"),
            "layer_loop": find_line(runner_text, "for layer_id, layer_type in enumerate"),
            "output_norm_call": find_line(runner_text, "return self._run_output_norm_hidden("),
            "output_norm": find_line(runner_text, "def _run_output_norm_hidden"),
            "bf16_copy_helper": find_line(runner_text, "def _copy_bf16_ptr_to_host_f32"),
        },
    }


def audit_build_inputs(compile_artifact: Mapping[str, Any]) -> dict[str, Any]:
    source_dir = Path(str(compile_artifact.get("source_dir", "")))
    build_dir = Path(str(compile_artifact.get("build_dir", "")))
    lib_dir = Path(str(compile_artifact.get("lib_dir", "")))
    executable = Path(
        str((compile_artifact.get("outputs") or {}).get("executable", ""))
    )
    ready = all(
        (source_dir.exists(), build_dir.exists(), lib_dir.exists(), executable.exists())
    )
    return {
        "ready": ready,
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "lib_dir": str(lib_dir),
        "capture_harness_executable": str(executable),
        "capture_harness_executable_exists": executable.exists(),
    }


def build_execution_plan(build_inputs: Mapping[str, Any]) -> dict[str, Any]:
    source_dir = str(build_inputs.get("source_dir"))
    return {
        "steps": [
            "copy the llama.cpp source_dir to a new /tmp pre-output_norm source tree",
            "apply llamacpp_pre_output_patch.old_text -> new_text in src/models/qwen35moe.cpp",
            "configure/build libllama in a new temporary build directory",
            "compile the existing hidden-seed capture harness against that patched build",
            "run the harness with --all-rows for the oracle prompt at position 16",
            "capture hipEngine serial pre-output_norm row via the private serial replay strategy",
            "compare pre-output_norm rows, then compare post-output_norm rows from iter303/304",
        ],
        "source_dir_to_copy": source_dir,
        "temporary_source_hint": "/tmp/hipengine-llamacpp-mtp-iter306-pre-output-norm-src",
        "temporary_build_hint": "/tmp/hipengine-llamacpp-mtp-iter306-pre-output-norm-build",
        "artifact_prefix_hint": "/tmp/hipengine-llamacpp-mtp-iter306-pre-output-norm/pos16",
    }


def decide_plan(
    *,
    shared_audit: Mapping[str, Any],
    llama_patch: Mapping[str, Any],
    hip_capture: Mapping[str, Any],
    build_inputs: Mapping[str, Any],
) -> dict[str, str]:
    shared_ready = shared_audit.get("conclusion") == (
        "shared_serial_path_mismatch_before_or_at_output_norm"
    )
    if (
        shared_ready
        and llama_patch.get("ready")
        and hip_capture.get("ready")
        and build_inputs.get("ready")
    ):
        return {
            "status": "ready",
            "conclusion": "pre_output_norm_capture_plan_ready",
            "next_action": "build_patched_llamacpp_pre_output_norm_harness_and_compare_serial_rows",
        }
    return {
        "status": "blocked",
        "conclusion": "pre_output_norm_capture_plan_missing_required_fact",
        "next_action": "inspect_pre_output_norm_plan_blockers",
    }


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
