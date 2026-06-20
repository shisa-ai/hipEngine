#!/usr/bin/env python3
"""Audit layer-1 attention RMSNorm under the resident-BF16 contract."""

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
    bf16_roundtrip_array,
    delta_summary,
    load_layer0_attn_norm_weight,
    rmsnorm_f32,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_layer1_bf16_handoff_audit import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER1_HANDOFF,
)

DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter332-layer1-attn-norm-oracle.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
WeightLoaderFn = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1-handoff", type=Path, default=DEFAULT_LAYER1_HANDOFF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--attn-norm-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=332)
    args = parser.parse_args()

    artifact = audit_layer1_attn_norm_oracle(
        layer1_handoff_path=args.layer1_handoff,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        attn_norm_atol=args.attn_norm_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "layer_id": artifact["layer_id"],
                "attn_norm_max_abs": artifact.get("attn_norm_delta", {}).get(
                    "max_abs_diff"
                ),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer1_attn_norm_oracle(
    *,
    layer1_handoff_path: Path,
    model_path: Path | None = None,
    layer_id: int = 1,
    max_sequence_length: int | None = None,
    attn_norm_atol: float = 0.0,
    iteration: int = 332,
    layer_capture_fn: LayerCaptureFn | None = None,
    weight_loader: WeightLoaderFn | None = None,
) -> dict[str, Any]:
    handoff = json.loads(layer1_handoff_path.read_text())
    validate_layer1_handoff(handoff, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or handoff["model"])
    prompt_tokens = tuple(int(token) for token in handoff["prompt_tokens"])
    position = int(handoff["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = layer_capture_fn or capture_layer_attn_norm
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        classification = "layer1_attn_norm_oracle_unavailable"
        return {
            "schema": 1,
            "kind": "layer1_attn_norm_oracle",
            "date": "2026-06-20",
            "loop": "mtp-gguf/run-20260615-103738",
            "iteration": int(iteration),
            "status": str(capture.get("status", "unavailable")),
            "classification": classification,
            "layer1_handoff_path": str(layer1_handoff_path),
            "model": str(resolved_model),
            "layer_id": int(layer_id),
            "position": position,
            "token_id": token_id,
            "prompt_tokens": list(prompt_tokens),
            "hipengine_capture": summarize_capture(capture),
            "external_checkout_modified": False,
            "next_action": next_action(classification),
        }
    selected_loader = weight_loader or load_layer0_attn_norm_weight
    weight, eps, weight_metadata = selected_loader(resolved_model, int(layer_id))
    hidden_in = np.asarray(capture["fields"]["hidden_in_f32"], dtype=np.float32)
    expected = bf16_roundtrip_array(rmsnorm_f32(hidden_in, weight, float(eps)))
    actual = np.asarray(capture["fields"]["attn_norm_f32"], dtype=np.float32)
    attn_norm_delta = delta_summary(expected, actual)
    classification = classify_attn_norm(
        attn_norm_delta,
        capture=capture,
        layer_id=int(layer_id),
        attn_norm_atol=float(attn_norm_atol),
    )
    return {
        "schema": 1,
        "kind": "layer1_attn_norm_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer1_handoff_path": str(layer1_handoff_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer1 hidden_in_f32 from exact BF16 handoff artifact",
            "formula": "BF16(RMSNorm(hidden_in_f32, attn_norm.weight_f32, eps_model))",
            "handoff_classification": handoff.get("classification"),
            "expectation": "exact resident-BF16 attn_norm before layer1 projection audits",
        },
        "weight": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "oracle_attn_norm": {
            "summary": summarize_array(expected),
            "sha256": sha256_float32(expected),
        },
        "attn_norm_delta": attn_norm_delta,
        "attn_norm_atol": float(attn_norm_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer1_handoff(
    handoff: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if handoff.get("status") != "ready":
        raise ValueError("layer1 handoff artifact must be ready")
    if handoff.get("classification") != "layer1_hidden_in_matches_layer0_layer_out_exactly":
        raise ValueError("layer1 handoff artifact must be exact")
    if int(handoff.get("target_layer", -1)) != int(expected_layer_id):
        raise ValueError("layer1 handoff target_layer does not match requested layer")
    delta = handoff.get("handoff_delta") or {}
    if delta.get("exact_match") is not True or float(delta.get("max_abs_diff", 1.0)) != 0.0:
        raise ValueError("layer1 handoff delta must be exact")
    if handoff.get("next_action") != "audit_layer1_attn_norm_under_bf16_contract":
        raise ValueError("layer1 handoff artifact must point to attn_norm audit")


def capture_layer_attn_norm(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "fields": {}}
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _copy_bf16_ptr_to_host_f32,
    )

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        for index, token in enumerate(prompt_tokens[:position]):
            session.step(int(token), position=index, return_logits=False)
        layer_capture = session.capture_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
            run_preceding_layers=True,
        )
        runtime = session.runtime
        hidden_size = int(layer_capture.hidden_size)
        attn_norm = _copy_bf16_ptr_to_host_f32(
            int(session.scratch.norm.ptr),
            hidden_size,
            runtime=runtime,
        )
    return {
        "status": "captured",
        "summary": {
            **layer_capture.as_summary_dict(),
            "attn_norm_shape": list(attn_norm.shape),
        },
        "fields": {
            "hidden_in_f32": np.asarray(layer_capture.hidden_in_f32, dtype=np.float32),
            "attn_norm_f32": np.asarray(attn_norm, dtype=np.float32),
        },
    }


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def classify_attn_norm(
    delta: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    layer_id: int,
    attn_norm_atol: float,
) -> str:
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer1_attn_norm_wrong_layer_capture"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer1_attn_norm_wrong_preceding_layer_count"
    if not delta.get("available") or not delta.get("shape_match"):
        return "layer1_attn_norm_oracle_unavailable"
    if delta.get("exact_match"):
        return "layer1_attn_norm_matches_bf16_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(attn_norm_atol):
        return "layer1_attn_norm_matches_bf16_oracle_within_tolerance"
    return "layer1_attn_norm_mismatch_after_bf16_oracle"


def status_from_classification(classification: str) -> str:
    if "unavailable" in classification:
        return "unavailable"
    if "wrong" in classification or "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer1_attn_norm_matches_bf16_oracle_exactly",
        "layer1_attn_norm_matches_bf16_oracle_within_tolerance",
    }:
        return "audit_layer1_projection_or_conv_gdn_under_bf16_contract"
    if classification == "layer1_attn_norm_wrong_preceding_layer_count":
        return "inspect_layer1_attn_norm_capture_preceding_layers"
    if classification == "layer1_attn_norm_wrong_layer_capture":
        return "inspect_layer1_attn_norm_capture_layer_id"
    if classification == "layer1_attn_norm_mismatch_after_bf16_oracle":
        return "inspect_layer1_attn_norm_weight_or_rmsnorm_semantics"
    return "rerun_layer1_attn_norm_oracle_on_rocm_host"


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
