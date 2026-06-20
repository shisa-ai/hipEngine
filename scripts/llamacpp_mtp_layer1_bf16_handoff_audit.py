#!/usr/bin/env python3
"""Audit the BF16 handoff from layer-0 output into layer-1 hidden input."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_layer0_bisection_conclusion import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER0_CONCLUSION,
)

DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter331-layer1-bf16-handoff.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, bool, int | None],
    dict[str, Any],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer0-conclusion", type=Path, default=DEFAULT_LAYER0_CONCLUSION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--handoff-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=331)
    args = parser.parse_args()

    artifact = audit_layer1_bf16_handoff(
        layer0_conclusion_path=args.layer0_conclusion,
        model_path=args.model,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        max_sequence_length=args.max_sequence_length,
        handoff_atol=args.handoff_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "source_layer": artifact["source_layer"],
                "target_layer": artifact["target_layer"],
                "handoff_max_abs": artifact.get("handoff_delta", {}).get("max_abs_diff"),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer1_bf16_handoff(
    *,
    layer0_conclusion_path: Path,
    model_path: Path | None = None,
    source_layer: int = 0,
    target_layer: int = 1,
    max_sequence_length: int | None = None,
    handoff_atol: float = 0.0,
    iteration: int = 331,
    layer_capture_fn: LayerCaptureFn | None = None,
) -> dict[str, Any]:
    conclusion = json.loads(layer0_conclusion_path.read_text())
    validate_layer0_conclusion(conclusion)
    if target_layer != source_layer + 1:
        raise ValueError("target_layer must be the immediate successor of source_layer")
    resolved_model = Path(model_path or conclusion["model"])
    prompt_tokens = tuple(int(token) for token in conclusion["prompt_tokens"])
    position = int(conclusion["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = layer_capture_fn or capture_hipengine_attention_layer
    source_capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(source_layer),
        False,
        max_sequence_length,
    )
    target_capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(target_layer),
        True,
        max_sequence_length,
    )
    if source_capture.get("status") != "captured" or target_capture.get("status") != "captured":
        classification = "layer1_bf16_handoff_capture_unavailable"
        return {
            "schema": 1,
            "kind": "layer1_bf16_handoff_audit",
            "date": "2026-06-20",
            "loop": "mtp-gguf/run-20260615-103738",
            "iteration": int(iteration),
            "status": "unavailable",
            "classification": classification,
            "layer0_conclusion_path": str(layer0_conclusion_path),
            "model": str(resolved_model),
            "source_layer": int(source_layer),
            "target_layer": int(target_layer),
            "position": position,
            "token_id": token_id,
            "prompt_tokens": list(prompt_tokens),
            "source_capture": summarize_capture(source_capture),
            "target_capture": summarize_capture(target_capture),
            "external_checkout_modified": False,
            "next_action": next_action(classification),
        }
    handoff_delta = delta_summary(
        np.asarray(source_capture["fields"]["layer_out_f32"], dtype=np.float32),
        np.asarray(target_capture["fields"]["hidden_in_f32"], dtype=np.float32),
    )
    classification = classify_handoff(
        handoff_delta,
        target_capture=target_capture,
        target_layer=int(target_layer),
        handoff_atol=float(handoff_atol),
    )
    return {
        "schema": 1,
        "kind": "layer1_bf16_handoff_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer0_conclusion_path": str(layer0_conclusion_path),
        "model": str(resolved_model),
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer0 layer_out_f32 under resident-BF16 contract",
            "target": "layer1 hidden_in_f32 captured after run_preceding_layers=True",
            "expectation": "exact BF16 handoff before auditing layer1 sub-boundaries",
            "layer0_conclusion": conclusion.get("classification"),
        },
        "source_capture": summarize_capture(source_capture),
        "target_capture": summarize_capture(target_capture),
        "handoff_delta": handoff_delta,
        "handoff_atol": float(handoff_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer0_conclusion(conclusion: Mapping[str, Any]) -> None:
    if conclusion.get("status") != "ready":
        raise ValueError("layer0 conclusion must be ready")
    if conclusion.get("classification") != (
        "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split"
    ):
        raise ValueError("layer0 conclusion must close the BF16-oracle chain")
    if conclusion.get("next_action") != (
        "advance_bisection_to_layer1_or_next_layer_boundary_under_bf16_contract"
    ):
        raise ValueError("layer0 conclusion must point to layer1/next-boundary work")
    internal = conclusion.get("internal_bf16_oracle_chain") or {}
    final_delta = internal.get("final_layer_out_delta") or {}
    if final_delta.get("exact_match") is not True:
        raise ValueError("layer0 final layer_out oracle must be exact")


def capture_hipengine_attention_layer(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
    run_preceding_layers: bool,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "fields": {}}
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        for index, token_id in enumerate(prompt_tokens[:position]):
            session.step(int(token_id), position=index, return_logits=False)
        capture = session.capture_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
            run_preceding_layers=bool(run_preceding_layers),
        )
    fields = {
        "hidden_in_f32": np.asarray(capture.hidden_in_f32, dtype=np.float32),
        "layer_out_f32": np.asarray(capture.layer_out_f32, dtype=np.float32),
    }
    return {
        "status": "captured",
        "summary": capture.as_summary_dict(),
        "fields": fields,
    }


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def classify_handoff(
    delta: Mapping[str, Any],
    *,
    target_capture: Mapping[str, Any],
    target_layer: int,
    handoff_atol: float,
) -> str:
    target_summary = target_capture.get("summary") or {}
    if not delta.get("available") or not delta.get("shape_match"):
        return "layer1_bf16_handoff_unavailable"
    if int(target_summary.get("preceding_layer_count", -1)) != int(target_layer):
        return "layer1_bf16_handoff_wrong_preceding_layer_count"
    if delta.get("exact_match"):
        return "layer1_hidden_in_matches_layer0_layer_out_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(handoff_atol):
        return "layer1_hidden_in_matches_layer0_layer_out_within_tolerance"
    return "layer1_hidden_in_mismatch_after_layer0_bf16_handoff"


def status_from_classification(classification: str) -> str:
    if "unavailable" in classification:
        return "unavailable"
    if "wrong_preceding" in classification or "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer1_hidden_in_matches_layer0_layer_out_exactly",
        "layer1_hidden_in_matches_layer0_layer_out_within_tolerance",
    }:
        return "audit_layer1_attn_norm_under_bf16_contract"
    if classification == "layer1_bf16_handoff_wrong_preceding_layer_count":
        return "inspect_capture_attention_layer_run_preceding_layers"
    if classification == "layer1_hidden_in_mismatch_after_layer0_bf16_handoff":
        return "inspect_layer0_to_layer1_hidden_buffer_handoff"
    return "rerun_layer1_bf16_handoff_on_rocm_host"


def summarize_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(capture.get("summary") or {})
    fields = capture.get("fields") or {}
    field_summaries: dict[str, Any] = {}
    for name, values in fields.items():
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        field_summaries[name] = {
            "summary": summarize_array(arr),
            "sha256": sha256_float32(arr),
        }
    return {
        "status": capture.get("status"),
        "summary": summary,
        "fields": field_summaries,
    }


if __name__ == "__main__":
    main()
