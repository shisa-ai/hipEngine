#!/usr/bin/env python3
"""Audit shared hipEngine hidden-seed path after mode sweep."""

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

DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_QWEN35MOE = Path("/home/lhl/llama.cpp/llama.cpp-hip/src/models/qwen35moe.cpp")
DEFAULT_CONTEXT = Path("/home/lhl/llama.cpp/llama.cpp-hip/src/llama-context.cpp")
DEFAULT_MODE_SWEEP = Path("benchmarks/results/mtp-gguf-iter304-hidden-seed-mode-sweep.json")
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter305-hidden-seed-shared-path-audit.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--qwen35moe", type=Path, default=DEFAULT_QWEN35MOE)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--mode-sweep", type=Path, default=DEFAULT_MODE_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=305)
    args = parser.parse_args()

    artifact = build_shared_path_audit(
        runner_path=args.runner,
        qwen35moe_path=args.qwen35moe,
        context_path=args.context,
        mode_sweep_path=args.mode_sweep,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "shared_serial_modes_exact": artifact["mode_evidence"][
                    "native_serial_step_exact"
                ],
                "bulk_secondary_rmse": artifact["mode_evidence"].get(
                    "bulk_vs_native_rmse"
                ),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_shared_path_audit(
    *,
    runner_path: Path,
    qwen35moe_path: Path,
    context_path: Path,
    mode_sweep_path: Path,
    iteration: int = 305,
) -> dict[str, Any]:
    runner_text = runner_path.read_text()
    qwen_text = qwen35moe_path.read_text()
    context_text = context_path.read_text()
    mode_sweep = read_json(mode_sweep_path)
    mode_evidence = summarize_mode_evidence(mode_sweep)
    hipengine_path = audit_hipengine_shared_seed_path(runner_text)
    llamacpp_path = audit_llamacpp_h_nextn_path(qwen_text, context_text)
    decision = decide_shared_path(
        mode_evidence=mode_evidence,
        hipengine_path=hipengine_path,
        llamacpp_path=llamacpp_path,
    )
    return {
        "schema": 1,
        "kind": "hipengine_hidden_seed_shared_path_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": decision["status"],
        "runner_path": str(runner_path),
        "qwen35moe_path": str(qwen35moe_path),
        "context_path": str(context_path),
        "mode_sweep_path": str(mode_sweep_path),
        "mode_evidence": mode_evidence,
        "hipengine_shared_seed_path": hipengine_path,
        "llamacpp_h_nextn_path": llamacpp_path,
        "decision": decision,
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def summarize_mode_evidence(mode_sweep: Mapping[str, Any]) -> dict[str, Any]:
    rows = {row.get("mode"): row for row in mode_sweep.get("mode_results", [])}
    pairs = {
        (row.get("left"), row.get("right")): row
        for row in mode_sweep.get("hipengine_pairwise", {}).get("pairs", [])
    }
    native_sha = capture_sha(rows.get("prefill-native"))
    serial_sha = capture_sha(rows.get("prefill-serial"))
    step_sha = capture_sha(rows.get("step-serial"))
    bulk_sha = capture_sha(rows.get("prefill-bulk"))
    native_delta = (rows.get("prefill-native") or {}).get("numeric_delta") or {}
    bulk_delta = (rows.get("prefill-bulk") or {}).get("numeric_delta") or {}
    bulk_pair = pairs.get(("prefill-bulk", "prefill-native")) or pairs.get(
        ("prefill-native", "prefill-bulk"),
        {},
    )
    return {
        "source_artifact_status": mode_sweep.get("status"),
        "source_conclusion": mode_sweep.get("conclusion"),
        "native_serial_step_exact": bool(
            native_sha and native_sha == serial_sha and native_sha == step_sha
        ),
        "native_serial_step_sha256": native_sha if native_sha == serial_sha == step_sha else None,
        "bulk_sha256": bulk_sha,
        "bulk_differs_from_native": bool(bulk_sha and native_sha and bulk_sha != native_sha),
        "native_vs_llamacpp_rmse": native_delta.get("rmse"),
        "native_vs_llamacpp_max_abs": native_delta.get("max_abs_diff"),
        "bulk_vs_llamacpp_rmse": bulk_delta.get("rmse"),
        "bulk_vs_llamacpp_max_abs": bulk_delta.get("max_abs_diff"),
        "bulk_vs_native_rmse": bulk_pair.get("rmse"),
        "bulk_vs_native_max_abs": bulk_pair.get("max_abs_diff"),
        "all_modes_exact_vs_llamacpp": all(
            (row.get("numeric_delta") or {}).get("exact_match") is True
            for row in rows.values()
        ),
    }


def audit_hipengine_shared_seed_path(text: str) -> dict[str, Any]:
    prefill_body = extract_function_body(text, "prefill")
    bulk_body = extract_function_body(text, "_run_bulk_prefill_and_sample")
    token_body = extract_function_body(text, "_run_token_to_final_hidden")
    current_body = extract_function_body(text, "_run_current_hidden_to_final_hidden")
    norm_body = extract_function_body(text, "_run_output_norm_hidden")
    step_body = extract_function_body(text, "step")
    facts = {
        "prefill_serial_captures_only_final_token": "index == final_index" in prefill_body
        and "capture_hidden_seed_fp32" in prefill_body,
        "step_serial_calls_token_to_final_hidden": (
            "_run_token_to_final_hidden" in step_body
        ),
        "token_path_sets_position_and_embedding": (
            "_set_full_attention_position_device" in token_body
            and "_set_token_id_device" in token_body
        ),
        "serial_path_loops_all_layers_before_output_norm": "for layer_id, layer_type"
        in current_body
        and "_run_output_norm_hidden" in current_body,
        "bulk_native_and_bulk_share_output_norm_call": "bulk_attention_mode == \"native\""
        in bulk_body
        and "_run_output_norm_hidden" in bulk_body,
        "output_norm_writes_bf16_logits_path": "gguf_rmsnorm_bf16_f32_weight(" in norm_body,
        "hidden_seed_f32_recomputed_from_same_bf16_source": "gguf_rmsnorm_bf16_f32_weight_out_f32"
        in norm_body
        and "src_ptr" in norm_body
        and "self.scratch.hidden_seed_fp32.ptr" in norm_body,
        "fp32_seed_not_source_for_logits": "return int(out_ptr)" in norm_body,
    }
    ready = all(facts.values())
    return {
        "ready": ready,
        "facts": facts,
        "anchors": {
            "prefill": find_line(text, "def prefill("),
            "prefill_final_index_capture": find_line(text, "index == final_index"),
            "bulk_output_norm_call": find_line(text, "last_src_ptr = src.ptr"),
            "step": find_line(text, "def step("),
            "token_to_final": find_line(text, "def _run_token_to_final_hidden"),
            "current_to_final": find_line(text, "def _run_current_hidden_to_final_hidden"),
            "output_norm": find_line(text, "def _run_output_norm_hidden"),
            "out_f32_kernel": find_line(text, "gguf_rmsnorm_bf16_f32_weight_out_f32"),
        },
    }


def audit_llamacpp_h_nextn_path(qwen_text: str, context_text: str) -> dict[str, Any]:
    trunk_region = slice_region(qwen_text, "// post-norm hidden state", "// LM head")
    facts = {
        "qwen_trunk_h_nextn_after_output_norm": (
            "build_norm(cur, model.output_norm" in trunk_region
            and 'cb(cur, "h_nextn", -1)' in trunk_region
            and "res->t_h_nextn = cur" in trunk_region
        ),
        "context_unmasked_rows_by_raw_position": (
            "unmasked: nextn rows are stored densely" in context_text
            and "return embd_nextn.data + (size_t) i * n_embd" in context_text
        ),
        "decode_copies_unmasked_rows_with_token_offset": (
            "offset = masked ? n_outputs_prev  : n_tokens_prev" in context_text
            and "ggml_backend_tensor_get_async(backend_h, t_h_nextn" in context_text
        ),
        "buffer_sized_by_batch_when_unmasked": (
            "embd_nextn.size = (size_t) n_embd_out * n_batch" in context_text
        ),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "anchors": {
            "qwen_output_norm": find_line(qwen_text, "cur = build_norm(cur, model.output_norm"),
            "qwen_h_nextn": find_line(qwen_text, 'cb(cur, "h_nextn", -1)'),
            "qwen_t_h_nextn": find_line(qwen_text, "res->t_h_nextn = cur"),
            "context_unmasked_get_ith": find_line(
                context_text,
                "unmasked: nextn rows are stored densely",
            ),
            "context_decode_nextn_copy": find_line(
                context_text,
                "ggml_backend_tensor_get_async(backend_h, t_h_nextn",
            ),
        },
    }


def decide_shared_path(
    *,
    mode_evidence: Mapping[str, Any],
    hipengine_path: Mapping[str, Any],
    llamacpp_path: Mapping[str, Any],
) -> dict[str, Any]:
    serials_agree = mode_evidence.get("native_serial_step_exact") is True
    serials_far = float(mode_evidence.get("native_vs_llamacpp_rmse") or 0.0) > 1.0
    bulk_secondary = float(mode_evidence.get("bulk_vs_native_rmse") or 0.0) > 0.0
    if serials_agree and serials_far and hipengine_path.get("ready") and llamacpp_path.get("ready"):
        return {
            "status": "audited",
            "conclusion": "shared_serial_path_mismatch_before_or_at_output_norm",
            "secondary_issue": "prefill_bulk_differs_from_serial" if bulk_secondary else None,
            "reason": (
                "prefill-native, prefill-serial, and step-serial produce identical "
                "FP32 seeds but all are far from the llama.cpp post-output_norm h_nextn row; "
                "therefore row/mode mixups are unlikely and the next bisection should "
                "compare the final layer output before output_norm versus the output_norm result."
            ),
            "next_action": "capture_pre_output_norm_rows_in_llamacpp_and_hipengine_serial_path",
        }
    return {
        "status": "inconclusive",
        "conclusion": "shared_path_audit_needs_more_evidence",
        "secondary_issue": None,
        "reason": "Mode evidence or source path facts did not satisfy the expected pattern.",
        "next_action": "rerun_hidden_seed_mode_sweep_and_source_audit",
    }


def capture_sha(row: Mapping[str, Any] | None) -> str | None:
    if row is None:
        return None
    return ((row.get("hipengine_capture") or {}).get("summary") or {}).get("sha256")


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


def slice_region(text: str, start_needle: str, end_needle: str) -> str:
    start = text.find(start_needle)
    if start < 0:
        return ""
    end = text.find(end_needle, start)
    return text[start : end if end >= 0 else len(text)]


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
