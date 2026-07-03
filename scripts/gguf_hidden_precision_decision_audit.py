#!/usr/bin/env python3
"""Decide the GGUF hidden precision path needed for llama.cpp parity."""

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
DEFAULT_DOC = Path("docs/MTP-gguf.md")
DEFAULT_TOKEN_AUDIT = Path(
    "benchmarks/results/mtp-gguf-iter299-token-embedding-parity-audit.json"
)
DEFAULT_EARLIEST = Path(
    "benchmarks/results/mtp-gguf-iter298-hidden-in-earliest-divergence.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter300-hidden-precision-decision-audit.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--earliest", type=Path, default=DEFAULT_EARLIEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=300)
    args = parser.parse_args()

    artifact = build_hidden_precision_decision_artifact(
        runner_path=args.runner,
        doc_path=args.doc,
        token_audit_path=args.token_audit,
        earliest_path=args.earliest,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "current_seed_dtype": artifact["runner_abi"]["facts"][
                    "current_hidden_seed_contract_dtype"
                ],
                "fp32_seed_dtype": artifact["runner_abi"]["facts"][
                    "fp32_hidden_seed_contract_dtype"
                ],
                "default_activation_buffer_dtype": artifact["runner_abi"]["facts"][
                    "default_activation_buffer_dtype"
                ],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_hidden_precision_decision_artifact(
    *,
    runner_path: Path,
    doc_path: Path,
    token_audit_path: Path,
    earliest_path: Path,
    iteration: int = 300,
) -> dict[str, Any]:
    runner_text = runner_path.read_text()
    doc_text = doc_path.read_text()
    token_audit = read_json(token_audit_path)
    earliest = read_json(earliest_path)
    runner_abi = audit_runner_hidden_precision(runner_text)
    doc_contract = audit_doc_hidden_seed_contract(doc_text)
    numeric = summarize_numeric_evidence(token_audit=token_audit, earliest=earliest)
    decision = decide_precision_path(
        runner_facts=runner_abi["facts"],
        doc_contract=doc_contract,
        numeric=numeric,
    )
    return {
        "schema": 1,
        "kind": "gguf_hidden_precision_decision_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": decision["status"],
        "runner_path": str(runner_path),
        "doc_path": str(doc_path),
        "token_audit_path": str(token_audit_path),
        "earliest_divergence_path": str(earliest_path),
        "runner_abi": runner_abi,
        "doc_contract": doc_contract,
        "numeric_evidence": numeric,
        "decision": decision,
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def audit_runner_hidden_precision(text: str) -> dict[str, Any]:
    current_seed_body = extract_function_body(
        text,
        "qwen35_gguf_current_hidden_seed_contract",
    )
    fp32_seed_body = extract_function_body(
        text,
        "qwen35_gguf_fp32_hidden_seed_contract",
    )
    facts = {
        "current_hidden_seed_contract_dtype": dtype_literal(
            current_seed_body,
            default="unknown",
        ),
        "current_hidden_seed_llama_compatible": "llama_cpp_compatible=True"
        in current_seed_body,
        "current_hidden_seed_marks_llama_incompatible": "llama_cpp_compatible=False"
        in current_seed_body,
        "fp32_hidden_seed_contract_dtype": dtype_literal(
            fp32_seed_body,
            default="unknown",
        ),
        "fp32_hidden_seed_contract_present": "scratch.hidden_seed_fp32" in fp32_seed_body,
        "fp32_hidden_seed_populated_guard": "ready_for_mtp" in text
        and "capture_hidden_seed_fp32=True" in text,
        "fp32_seed_populated_from_bf16_source": "gguf_rmsnorm_bf16_f32_weight_out_f32"
        in text,
        "default_activation_buffer_dtype": "BF16",
        "session_hidden_buffer_nbytes_expression": find_expression(
            text,
            r"hidden_bytes\s*=\s*([^\n]+)",
        ),
        "session_hidden_buffers_allocated": all(
            needle in text for needle in ("self._hidden_a = malloc", "self._hidden_b = malloc")
        ),
        "prefill_hidden_buffers_allocated": all(
            needle in text
            for needle in ("self._prefill_hidden_a = malloc", "self._prefill_hidden_b = malloc")
        ),
        "run_prompt_hidden_returns_uint16_bits": "dtype=np.uint16" in text,
        "session_embedding_writes_hidden_a_bf16_buffer": "launch_gguf_embedding" in text
        and "self._hidden_a.ptr" in text,
        "capture_hidden_in_uses_bf16_copy": "hidden_in_f32=_copy_bf16_ptr_to_host_f32" in text,
        "logits_api_expects_hidden_bits_uint16": "hidden_bits" in text
        and "np.ascontiguousarray(hidden_bits, dtype=np.uint16)" in text,
    }
    facts["current_runtime_activation_lane"] = current_runtime_activation_lane(facts)
    return {
        "facts": facts,
        "anchors": {
            "current_hidden_seed_contract": find_line(
                text,
                "def qwen35_gguf_current_hidden_seed_contract",
            ),
            "fp32_hidden_seed_contract": find_line(
                text,
                "def qwen35_gguf_fp32_hidden_seed_contract",
            ),
            "current_hidden_seed_dtype": find_line(text, "dtype=DType.BF16"),
            "fp32_hidden_seed_dtype": find_line(text, "dtype=DType.FP32"),
            "session_hidden_bytes": find_line(text, "hidden_bytes = self.runner.hidden_size * 2"),
            "session_embedding_launch": find_line(text, "launch_gguf_embedding("),
            "capture_hidden_in_bf16_copy": find_line(
                text,
                "hidden_in_f32=_copy_bf16_ptr_to_host_f32",
            ),
        },
    }


def audit_doc_hidden_seed_contract(text: str) -> dict[str, Any]:
    fp32_terms = [
        "Target hidden-row seed = POST output-norm hidden, at fp32",
        "GGML_TYPE_F32",
        "post-norm fp32 hidden seed",
    ]
    matched = [term for term in fp32_terms if term in text]
    return {
        "requires_fp32_seed": bool(matched),
        "matched_terms": matched,
        "anchors": {
            "target_hidden_seed_section": find_line(
                text,
                "Target hidden-row seed = POST output-norm hidden, at fp32",
            ),
            "ggml_type_f32": find_line(text, "GGML_TYPE_F32"),
        },
    }


def summarize_numeric_evidence(
    *, token_audit: Mapping[str, Any], earliest: Mapping[str, Any]
) -> dict[str, Any]:
    comparisons = token_audit.get("comparisons") or {}
    earliest_rows = {row.get("layer"): row for row in earliest.get("layer_results", [])}
    layer0 = earliest_rows.get(0) or {}
    layer0_delta = layer0.get("numeric_delta") or {}
    return {
        "token_audit_status": token_audit.get("status"),
        "token_audit_conclusion": token_audit.get("conclusion"),
        "llamacpp_matches_raw_dequant": bool(
            comparisons.get("llamacpp_vs_raw_dequant", {}).get("exact_match")
        ),
        "hipengine_matches_bf16_round": bool(
            comparisons.get("hipengine_vs_bf16_round", {}).get("exact_match")
        ),
        "llamacpp_vs_hipengine_embedding_rmse": comparisons.get(
            "llamacpp_vs_hipengine",
            {},
        ).get("rmse"),
        "llamacpp_vs_hipengine_embedding_max_abs": comparisons.get(
            "llamacpp_vs_hipengine",
            {},
        ).get("max_abs_diff"),
        "earliest_first_mismatch_layer": earliest.get("ranking", {}).get(
            "first_mismatch_layer"
        ),
        "earliest_layer0_rmse": layer0_delta.get("rmse"),
        "earliest_layer0_max_abs": layer0_delta.get("max_abs_diff"),
        "earliest_layer0_preceding_precision_contractions": layer0.get(
            "preceding_precision_contractions",
            {},
        ).get("count"),
    }


def decide_precision_path(
    *,
    runner_facts: Mapping[str, Any],
    doc_contract: Mapping[str, Any],
    numeric: Mapping[str, Any],
) -> dict[str, Any]:
    numeric_proves_bf16 = all(
        (
            numeric.get("llamacpp_matches_raw_dequant") is True,
            numeric.get("hipengine_matches_bf16_round") is True,
            numeric.get("earliest_first_mismatch_layer") == 0,
            numeric.get("earliest_layer0_preceding_precision_contractions") == 0,
        )
    )
    runner_is_bf16 = runner_facts.get("current_runtime_activation_lane") == "bf16"
    doc_requires_fp32 = bool(doc_contract.get("requires_fp32_seed"))
    fp32_seed_target_exists = all(
        (
            runner_facts.get("fp32_hidden_seed_contract_dtype") == "FP32",
            runner_facts.get("fp32_hidden_seed_contract_present") is True,
            runner_facts.get("fp32_hidden_seed_populated_guard") is True,
        )
    )
    if numeric_proves_bf16 and runner_is_bf16 and doc_requires_fp32:
        conclusion = (
            "fp32_seed_target_exists_but_activation_lane_is_bf16"
            if fp32_seed_target_exists
            else "exact_parity_requires_explicit_f32_activation_or_seed_lane"
        )
        next_step = (
            "capture_fp32_hidden_seed_vs_llamacpp_post_output_norm_oracle"
            if fp32_seed_target_exists
            else "prototype_f32_hidden_seed_capture_or_comparable_bf16_llamacpp_trace"
        )
        return {
            "status": "decided",
            "conclusion": conclusion,
            "keep_default_runtime": "bf16_activation_buffers",
            "reason": (
                "llama.cpp layer0 hidden_in equals raw dequantized F32 embedding, "
                "hipEngine embedding equals BF16-rounded output, the documented "
                "MTP seed contract requires FP32 post-output_norm hidden, and the "
                "current runtime activation buffers remain BF16."
            ),
            "next_action": next_step,
        }
    if numeric_proves_bf16 and runner_is_bf16:
        return {
            "status": "partial",
            "conclusion": "bf16_embedding_output_explains_layer0_without_doc_contract",
            "keep_default_runtime": "bf16_activation_buffers",
            "reason": (
                "Numeric evidence identifies BF16 embedding output, but doc "
                "FP32 seed terms were not found."
            ),
            "next_action": "reconcile_docs_before_precision_promotion",
        }
    return {
        "status": "blocked",
        "conclusion": "hidden_precision_decision_needs_more_evidence",
        "keep_default_runtime": "unchanged",
        "reason": "Numeric BF16 proof, runner BF16 ABI, or doc FP32 seed contract was missing.",
        "next_action": "rerun_token_embedding_and_source_audits",
    }


def current_runtime_activation_lane(facts: Mapping[str, Any]) -> str:
    if (
        facts.get("session_hidden_buffer_nbytes_expression") in {
            "self.runner.hidden_size * 2",
            "runner.hidden_size * 2",
        }
        and facts.get("session_hidden_buffers_allocated")
        and facts.get("run_prompt_hidden_returns_uint16_bits")
    ):
        return "bf16"
    return "unknown"


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


def dtype_literal(text: str, *, default: str) -> str:
    match = re.search(r"dtype=DType\.([A-Z0-9_]+)", text)
    return match.group(1) if match else default


def find_expression(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
