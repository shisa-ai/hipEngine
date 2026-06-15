#!/usr/bin/env python3
"""Run GGUF MTP CPU-reference oracle fixtures and report exactness metrics.

The gate is intentionally fixture-driven.  It resolves the kernel through the
four-axis registry, recomputes logits, and reports KL/top-1 agreement against the
fixture's expected logits.  It does not load model weights or run generation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hipengine.kernels.cpu_reference  # noqa: F401,E402 - self-registers CPU kernels
from hipengine.kernels.registry import resolve  # noqa: E402
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402


DEFAULT_FIXTURE = Path("benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json")


def run_oracle_gate(
    fixture_path: Path,
    *,
    max_kl: float = 0.05,
    min_top1_agreement: float = 0.90,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text())
    actual_logits = _run_nextn_fixture(fixture)
    expected_logits = _f32(fixture["expected"]["logits"])
    if actual_logits.shape != expected_logits.shape:
        raise ValueError(
            f"actual logits shape {actual_logits.shape} did not match expected {expected_logits.shape}"
        )
    kl_values = _kl_divergence(_softmax(expected_logits), _softmax(actual_logits))
    actual_top1 = np.argmax(actual_logits, axis=-1)
    expected_top1 = np.argmax(expected_logits, axis=-1)
    top1_agreement = float(np.mean(actual_top1 == expected_top1))
    top_k = int(fixture.get("top_k", 1))
    actual_top_k = np.argsort(-actual_logits, axis=-1, kind="stable")[:, :top_k]
    expected_top_k = np.asarray(fixture["expected"].get("top_k_token_ids", []), dtype=np.int64)
    if expected_top_k.ndim == 1:
        expected_top_k = expected_top_k[None, :]
    top_k_match = bool(expected_top_k.shape == actual_top_k.shape and np.array_equal(actual_top_k, expected_top_k))
    metrics = {
        "max_kl": float(np.max(kl_values)) if kl_values.size else 0.0,
        "mean_kl": float(np.mean(kl_values)) if kl_values.size else 0.0,
        "top1_agreement": top1_agreement,
        "top_k_match": top_k_match,
        "rows": int(actual_logits.shape[0]),
        "vocab_size": int(actual_logits.shape[1]),
    }
    passed = metrics["max_kl"] <= float(max_kl) and metrics["top1_agreement"] >= float(
        min_top1_agreement
    )
    return {
        "schema": 1,
        "kind": "gguf_mtp_oracle_gate",
        "fixture": str(fixture_path),
        "cpu_reference_kernel": fixture["cpu_reference_kernel"],
        "thresholds": {
            "max_kl": float(max_kl),
            "min_top1_agreement": float(min_top1_agreement),
        },
        "metrics": metrics,
        "actual_top1_token_ids": actual_top1.astype(int).tolist(),
        "expected_top1_token_ids": expected_top1.astype(int).tolist(),
        "actual_top_k_token_ids": actual_top_k.astype(int).tolist(),
        "expected_top_k_token_ids": expected_top_k.astype(int).tolist(),
        "passed": bool(passed),
    }


def _run_nextn_fixture(fixture: dict[str, Any]) -> np.ndarray:
    backend, layer, quant, variant = fixture["cpu_reference_kernel"]
    kernel = resolve(backend=backend, layer=layer, quant=quant, variant=variant)
    inputs = fixture["inputs"]
    logits = kernel(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        _f32(inputs["attn_norm_weight"]),
        _f32(inputs["wq_weight"]),
        _f32(inputs["wk_weight"]),
        _f32(inputs["wv_weight"]),
        _f32(inputs["wo_weight"]),
        _f32(inputs["q_norm_weight"]),
        _f32(inputs["k_norm_weight"]),
        _f32(inputs["attn_post_norm_weight"]),
        _f32(inputs["router_weight"]),
        _f32(inputs["gate_qweight"]),
        _f32(inputs["up_qweight"]),
        _f32(inputs["down_qweight"]),
        GGMLQuantizationType[str(inputs["gate_qtype"])],
        GGMLQuantizationType[str(inputs["up_qtype"])],
        GGMLQuantizationType[str(inputs["down_qtype"])],
        _f32(inputs["shared_gate_logit_weight"]),
        _f32(inputs["shared_gate_qweight"]),
        _f32(inputs["shared_up_qweight"]),
        _f32(inputs["shared_down_qweight"]),
        GGMLQuantizationType[str(inputs["shared_qtype"])],
        _f32(inputs["shared_head_norm_weight"]),
        _f32(inputs["shared_head_weight"]),
        **fixture["kwargs"],
    )
    return np.asarray(logits, dtype=np.float32)


def _f32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).tiny
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--min-top1-agreement", type=float, default=0.90)
    parser.add_argument("--out", type=Path, help="write JSON artifact to this path")
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)

    if not math.isfinite(args.max_kl) or args.max_kl < 0:
        raise SystemExit("--max-kl must be finite and non-negative")
    if not math.isfinite(args.min_top1_agreement) or not (0.0 <= args.min_top1_agreement <= 1.0):
        raise SystemExit("--min-top1-agreement must be in [0, 1]")
    result = run_oracle_gate(
        args.fixture,
        max_kl=args.max_kl,
        min_top1_agreement=args.min_top1_agreement,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload, end="")
    if args.fail_on_fail and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
