#!/usr/bin/env python3
"""Gate Moonshine NumPy oracles against a model-derived FP16 fixture archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_apply_partial_rope,
    moonshine_attention,
    moonshine_decoder_mlp,
    moonshine_layernorm,
    moonshine_projection,
    moonshine_residual,
    moonshine_rope_tables,
    moonshine_stable_argmax,
    moonshine_tied_lm_logits,
    moonshine_triple_projection,
)


def parse_positions(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("positions must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or min(result) < 0 or max(result) > 193:
        raise argparse.ArgumentTypeError("positions must be unique and in 0..193")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--positions", type=parse_positions, default=parse_positions("0,1,8,32,64,128,193"))
    parser.add_argument("--max-abs", type=float, default=0.5)
    parser.add_argument("--max-relative-l2", type=float, default=0.005)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual32 = np.asarray(actual, dtype=np.float32)
    expected32 = np.asarray(expected, dtype=np.float32)
    if actual32.shape != expected32.shape:
        raise ValueError(f"{name}: shape {actual32.shape} != {expected32.shape}")
    finite = bool(np.isfinite(actual32).all() and np.isfinite(expected32).all())
    difference = actual32 - expected32
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = max(float(np.linalg.norm(expected32.ravel())), 1.0e-12)
    relative_l2 = float(np.linalg.norm(difference.ravel()) / denominator)
    return {
        "name": name,
        "shape": list(actual32.shape),
        "finite": finite,
        "exact_fp16": bool(np.array_equal(actual, expected)),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
    }


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted).astype(np.float32)
    return exponential / np.sum(exponential, axis=-1, keepdims=True, dtype=np.float32)


def logits_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | bool]:
    actual_probability = softmax(actual)
    expected_probability = softmax(expected)
    floor = np.float32(1.0e-30)
    kl = np.sum(
        expected_probability
        * (
            np.log(np.maximum(expected_probability, floor))
            - np.log(np.maximum(actual_probability, floor))
        ),
        axis=-1,
        dtype=np.float32,
    )
    actual_top = moonshine_stable_argmax(actual)
    expected_top = moonshine_stable_argmax(expected)
    return {
        "kl": float(np.max(kl)),
        "top1_match": bool(np.array_equal(actual_top, expected_top)),
    }


def load_decoder_weights(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="numpy") as handle:
        return {
            name: handle.get_tensor(name).astype(np.float16)
            for name in handle.keys()
            if name.startswith("model.decoder")
        }


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file() or not args.fixture.is_file():
        raise FileNotFoundError("checkpoint and fixture must exist")
    arrays = np.load(args.fixture, allow_pickle=False)
    weights = load_decoder_weights(args.checkpoint)
    required_positions = {
        int(name.split(".")[1].removeprefix("position_"))
        for name in arrays.files
        if name.startswith("decoder.position_")
    }
    missing_positions = sorted(set(args.positions) - required_positions)
    if missing_positions:
        raise ValueError(f"fixture is missing positions {missing_positions}")

    cos, sin = moonshine_rope_tables(194, rotary_dim=32, theta=10_000.0)
    comparisons = []
    logit_rows = []
    selected_token_matches = []
    embedding = weights["model.decoder.embed_tokens.weight"]

    for position in args.positions:
        input_token = int(arrays[f"decoder.position_{position}.input_token"].reshape(-1)[0])
        residual = embedding[input_token][None, None, :]
        for layer in range(8):
            prefix = f"model.decoder.layers.{layer}"
            if layer:
                residual = arrays[f"decoder.position_{position}.layer_{layer - 1}.after_mlp"]

            normalized = moonshine_layernorm(
                residual,
                weights[f"{prefix}.input_layernorm.weight"],
            )
            query, key, _ = moonshine_triple_projection(
                normalized,
                weights[f"{prefix}.self_attn.q_proj.weight"],
                weights[f"{prefix}.self_attn.k_proj.weight"],
                weights[f"{prefix}.self_attn.v_proj.weight"],
            )
            query = query.reshape(1, 1, 8, 52).transpose(0, 2, 1, 3)
            key = key.reshape(1, 1, 8, 52).transpose(0, 2, 1, 3)
            query, key = moonshine_apply_partial_rope(
                query,
                key,
                cos,
                sin,
                position_ids=np.asarray([[position]], dtype=np.int64),
                rotary_dim=32,
            )
            self_key = arrays[f"decoder.position_{position}.layer_{layer}.self_key"]
            self_value = arrays[f"decoder.position_{position}.layer_{layer}.self_value"]
            comparisons.append(
                comparison(
                    f"position_{position}.layer_{layer}.self_key_current",
                    key,
                    self_key[:, :, -1:, :],
                )
            )
            self_context = moonshine_attention(query, self_key, self_value)
            self_output = moonshine_projection(
                self_context.transpose(0, 2, 1, 3).reshape(1, 1, 416),
                weights[f"{prefix}.self_attn.o_proj.weight"],
            )
            after_self = moonshine_residual(residual, self_output)
            expected_self = arrays[
                f"decoder.position_{position}.layer_{layer}.after_self_attention"
            ]
            comparisons.append(
                comparison(
                    f"position_{position}.layer_{layer}.after_self_attention",
                    after_self,
                    expected_self,
                )
            )

            cross_normalized = moonshine_layernorm(
                expected_self,
                weights[f"{prefix}.post_attention_layernorm.weight"],
            )
            cross_query = moonshine_projection(
                cross_normalized,
                weights[f"{prefix}.encoder_attn.q_proj.weight"],
            ).reshape(1, 1, 8, 52).transpose(0, 2, 1, 3)
            cross_context = moonshine_attention(
                cross_query,
                arrays[f"cross.layer_{layer}.key"],
                arrays[f"cross.layer_{layer}.value"],
            )
            cross_output = moonshine_projection(
                cross_context.transpose(0, 2, 1, 3).reshape(1, 1, 416),
                weights[f"{prefix}.encoder_attn.o_proj.weight"],
            )
            after_cross = moonshine_residual(expected_self, cross_output)
            expected_cross = arrays[
                f"decoder.position_{position}.layer_{layer}.after_cross_attention"
            ]
            comparisons.append(
                comparison(
                    f"position_{position}.layer_{layer}.after_cross_attention",
                    after_cross,
                    expected_cross,
                )
            )

            mlp_normalized = moonshine_layernorm(
                expected_cross,
                weights[f"{prefix}.final_layernorm.weight"],
            )
            mlp_output = moonshine_decoder_mlp(
                mlp_normalized,
                weights[f"{prefix}.mlp.fc1.weight"],
                weights[f"{prefix}.mlp.fc1.bias"],
                weights[f"{prefix}.mlp.fc2.weight"],
                weights[f"{prefix}.mlp.fc2.bias"],
            )
            after_mlp = moonshine_residual(expected_cross, mlp_output)
            comparisons.append(
                comparison(
                    f"position_{position}.layer_{layer}.after_mlp",
                    after_mlp,
                    arrays[f"decoder.position_{position}.layer_{layer}.after_mlp"],
                )
            )

        logits = moonshine_tied_lm_logits(
            arrays[f"decoder.position_{position}.final_hidden"],
            embedding,
        )[:, -1, :]
        expected_logits = arrays[f"decoder.position_{position}.logits"]
        row = comparison(f"position_{position}.logits", logits, expected_logits)
        row.update(logits_metrics(logits, expected_logits))
        comparisons.append(row)
        logit_rows.append(row)
        selected = int(moonshine_stable_argmax(logits).reshape(-1)[0])
        expected_selected = int(
            arrays[f"decoder.position_{position}.selected_token"].reshape(-1)[0]
        )
        selected_token_matches.append(selected == expected_selected)

    max_abs = max(float(row["max_abs"]) for row in comparisons)
    max_relative_l2 = max(float(row["relative_l2"]) for row in comparisons)
    max_kl = max(float(row["kl"]) for row in logit_rows)
    top1_agreement = sum(bool(row["top1_match"]) for row in logit_rows) / len(logit_rows)
    finite = all(bool(row["finite"]) for row in comparisons)
    passed = bool(
        finite
        and max_abs <= args.max_abs
        and max_relative_l2 <= args.max_relative_l2
        and max_kl <= args.max_kl
        and top1_agreement >= 0.9
        and all(selected_token_matches)
    )
    report = {
        "schema": 1,
        "kind": "moonshine_cpu_reference_fixture_gate",
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "fixture": str(args.fixture),
            "fixture_sha256": sha256_file(args.fixture),
            "positions": list(args.positions),
        },
        "thresholds": {
            "max_abs": args.max_abs,
            "max_relative_l2": args.max_relative_l2,
            "max_kl": args.max_kl,
            "min_top1_agreement": 0.9,
            "exact_selected_tokens": True,
        },
        "summary": {
            "passed": passed,
            "comparison_count": len(comparisons),
            "finite": finite,
            "max_abs": max_abs,
            "max_relative_l2": max_relative_l2,
            "max_kl": max_kl,
            "top1_agreement": top1_agreement,
            "selected_tokens_exact": all(selected_token_matches),
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
