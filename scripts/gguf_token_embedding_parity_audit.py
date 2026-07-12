#!/usr/bin/env python3
"""Audit GGUF token-embedding parity for llama.cpp vs hipEngine."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32, dequantize_gguf_data  # noqa: E402
from scripts.gguf_hidden_in_earliest_divergence import (  # noqa: E402
    DEFAULT_LAYER_SWEEP,
    compare_vectors,
    load_llamacpp_layer_values,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOKENS,
    pack_float32,
    sha256_bytes,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter299-token-embedding-parity-audit.json"
)
DEFAULT_LAYER = 0
DEFAULT_POSITION = 16
DEFAULT_TOKEN_TENSOR = "token_embd.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer-sweep", type=Path, default=DEFAULT_LAYER_SWEEP)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--position", type=int, default=DEFAULT_POSITION)
    parser.add_argument("--token-id", type=int)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--token-tensor", default=DEFAULT_TOKEN_TENSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=299)
    parser.add_argument("--skip-hip", action="store_true")
    args = parser.parse_args()

    prompt_tokens = parse_prompt_tokens(args.prompt_tokens)
    token_id = int(args.token_id if args.token_id is not None else prompt_tokens[args.position])
    artifact = build_token_embedding_parity_artifact(
        model_path=args.model,
        layer_sweep_path=args.layer_sweep,
        layer=args.layer,
        position=args.position,
        token_id=token_id,
        token_tensor=args.token_tensor,
        iteration=args.iteration,
        skip_hip=bool(args.skip_hip),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "raw_matches_llamacpp": artifact["comparisons"][
                    "llamacpp_vs_raw_dequant"
                ]["exact_match"],
                "hip_matches_bf16_round": artifact["comparisons"].get(
                    "hipengine_vs_bf16_round"
                ),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_token_embedding_parity_artifact(
    *,
    model_path: Path,
    layer_sweep_path: Path,
    layer: int,
    position: int,
    token_id: int,
    token_tensor: str = DEFAULT_TOKEN_TENSOR,
    iteration: int = 299,
    skip_hip: bool = False,
    llamacpp_values: Sequence[float] | None = None,
    raw_dequant_values: Sequence[float] | None = None,
    hipengine_values: Sequence[float] | None = None,
    tensor_info_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    llama_values = list(
        map(float, llamacpp_values)
        if llamacpp_values is not None
        else load_llamacpp_layer_values(read_json(layer_sweep_path), layer=layer)["values"]
    )
    raw_values, tensor_info = _raw_values_and_info(
        model_path=model_path,
        token_tensor=token_tensor,
        token_id=token_id,
        raw_dequant_values=raw_dequant_values,
        tensor_info_override=tensor_info_override,
    )
    bf16_values = bf16_round_to_f32(raw_values)
    hip_values: list[float] | None
    hip_status = "skipped_by_flag" if skip_hip else "not_run"
    if hipengine_values is not None:
        hip_values = [float(value) for value in hipengine_values]
        hip_status = "provided"
    elif skip_hip:
        hip_values = None
    elif not hip_available():
        hip_values = None
        hip_status = "skipped_no_hip_runtime"
    else:
        hip_values = capture_hipengine_embedding_output(
            model_path=model_path,
            token_id=token_id,
            hidden_size=int(tensor_info["shape"][1]),
            vocab_size=int(tensor_info["shape"][0]),
        )
        hip_status = "captured"

    comparisons = {
        "llamacpp_vs_raw_dequant": compare_vectors(
            llama_values, raw_values, exact_atol=0.0
        ),
        "llamacpp_vs_bf16_round": compare_vectors(
            llama_values, bf16_values, exact_atol=0.0
        ),
        "bf16_round_vs_raw_dequant": compare_vectors(
            bf16_values, raw_values, exact_atol=0.0
        ),
    }
    if hip_values is not None:
        comparisons["hipengine_vs_raw_dequant"] = compare_vectors(
            hip_values, raw_values, exact_atol=0.0
        )
        comparisons["hipengine_vs_bf16_round"] = compare_vectors(
            hip_values, bf16_values, exact_atol=0.0
        )
        comparisons["llamacpp_vs_hipengine"] = compare_vectors(
            llama_values, hip_values, exact_atol=0.0
        )
    conclusion = conclude(comparisons=comparisons, hip_status=hip_status)
    return {
        "schema": 1,
        "kind": "gguf_token_embedding_parity_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_conclusion(conclusion),
        "model": str(model_path),
        "layer_sweep_path": str(layer_sweep_path),
        "layer": int(layer),
        "position": int(position),
        "token_id": int(token_id),
        "token_tensor": token_tensor,
        "token_tensor_info": tensor_info,
        "hipengine_capture_status": hip_status,
        "vectors": {
            "llamacpp": summarize_vector(llama_values),
            "raw_dequant": summarize_vector(raw_values),
            "bf16_round": summarize_vector(bf16_values),
            "hipengine_embedding": summarize_vector(hip_values) if hip_values is not None else None,
        },
        "comparisons": comparisons,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion),
    }


def _raw_values_and_info(
    *,
    model_path: Path,
    token_tensor: str,
    token_id: int,
    raw_dequant_values: Sequence[float] | None,
    tensor_info_override: Mapping[str, Any] | None,
) -> tuple[list[float], dict[str, Any]]:
    if raw_dequant_values is not None:
        info = dict(tensor_info_override or {})
        info.setdefault("shape", [1, len(raw_dequant_values)])
        info.setdefault("ggml_type_name", "synthetic")
        info.setdefault("source", "provided")
        return [float(value) for value in raw_dequant_values], info
    reader = GGUFReader(model_path)
    info = reader.tensor_info(token_tensor)
    raw = reader.tensor_data(token_tensor)
    row = raw[int(token_id)]
    dequant = dequantize_gguf_data(row, info.ggml_type).reshape(-1)
    return [float(value) for value in dequant], {
        "name": info.name,
        "shape": list(info.shape),
        "byte_shape": list(info.byte_shape),
        "ggml_type": int(info.ggml_type),
        "ggml_type_name": info.ggml_type_name,
        "row_byte_shape": list(row.shape),
        "dequant_count": int(dequant.size),
    }


def capture_hipengine_embedding_output(
    *, model_path: Path, token_id: int, hidden_size: int, vocab_size: int
) -> list[float]:
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.loading.qwen35_gguf_materialize import materialize_qwen35_gguf_weights
    from hipengine.runtime.gguf_embedding import launch_gguf_embedding

    runtime = get_hip_runtime()
    reader = GGUFReader(model_path)
    weights = materialize_qwen35_gguf_weights(
        reader,
        selected_slots={"root.token_embedding"},
        runtime=runtime,
    )
    token = np.asarray([int(token_id)], dtype=np.int64)
    token_buf = malloc(token.nbytes, runtime=runtime)
    out_buf = malloc(int(hidden_size) * DType.BF16.itemsize, runtime=runtime)
    try:
        copy_host_to_device(token_buf, host_array_ptr(token), token.nbytes, runtime=runtime)
        launch_gguf_embedding(
            weights.root("token_embedding"),
            token_buf.ptr,
            out_buf.ptr,
            rows=1,
            hidden_size=int(hidden_size),
            vocab_size=int(vocab_size),
            runtime=runtime,
        )
        bits = np.empty((int(hidden_size),), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(bits),
            DeviceBuffer(out_buf.ptr, bits.nbytes),
            bits.nbytes,
            runtime=runtime,
        )
        return [float(value) for value in bf16_to_float32(bits)]
    finally:
        free(out_buf, runtime=runtime)
        free(token_buf, runtime=runtime)
        weights.free(runtime=runtime)


def bf16_round_to_f32(values: Sequence[float]) -> list[float]:
    bits = float_array_to_bf16_bits(np.asarray(values, dtype=np.float32))
    return [float(value) for value in bf16_to_float32(bits).reshape(-1)]


def summarize_vector(values: Sequence[float] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    array = [float(value) for value in values]
    return {
        "count": len(array),
        "sha256": sha256_bytes(pack_float32(array)),
        "samples": [round(value, 8) for value in array[:8]],
        "stats": summarize_floats(array),
    }


def summarize_floats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "l2": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
        "l2": math.sqrt(sum(float(value) * float(value) for value in values)),
    }


def conclude(*, comparisons: Mapping[str, dict[str, Any]], hip_status: str) -> str:
    raw_match = bool(comparisons["llamacpp_vs_raw_dequant"].get("exact_match"))
    llama_bf16 = comparisons["llamacpp_vs_bf16_round"]
    bf16_changes_raw = not bool(comparisons["bf16_round_vs_raw_dequant"].get("exact_match"))
    hip_vs_bf16 = comparisons.get("hipengine_vs_bf16_round")
    if not raw_match:
        return "llamacpp_layer0_does_not_match_raw_token_embedding"
    if hip_vs_bf16 is None:
        if bf16_changes_raw and hip_status.startswith("skipped"):
            return "raw_match_bf16_rounding_suspect_hip_skipped"
        return "raw_token_embedding_matches_llamacpp"
    if hip_vs_bf16.get("exact_match") and bf16_changes_raw and not llama_bf16.get("exact_match"):
        return "layer0_drift_is_bf16_embedding_output"
    if not hip_vs_bf16.get("exact_match"):
        return "hip_embedding_kernel_differs_from_bf16_round"
    return "token_embedding_paths_match"


def status_from_conclusion(conclusion: str) -> str:
    if conclusion == "layer0_drift_is_bf16_embedding_output":
        return "explained"
    if conclusion.endswith("hip_skipped"):
        return "partial"
    if conclusion == "token_embedding_paths_match":
        return "matched"
    return "mismatched"


def next_action(conclusion: str) -> str:
    if conclusion == "layer0_drift_is_bf16_embedding_output":
        return "decide_embedding_hidden_precision_for_llamacpp_exact_parity"
    if conclusion == "raw_match_bf16_rounding_suspect_hip_skipped":
        return "rerun_token_embedding_audit_with_hip_capture"
    if conclusion == "llamacpp_layer0_does_not_match_raw_token_embedding":
        return "inspect_llamacpp_layer0_capture_row_selection"
    if conclusion == "hip_embedding_kernel_differs_from_bf16_round":
        return "debug_gguf_embedding_kernel_dequantization"
    if conclusion == "token_embedding_paths_match":
        return "continue_layer0_internal_tap_comparison"
    return "inspect_token_embedding_audit"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def parse_prompt_tokens(text: str) -> tuple[int, ...]:
    tokens = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not tokens:
        raise ValueError("prompt token list is empty")
    return tokens


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
