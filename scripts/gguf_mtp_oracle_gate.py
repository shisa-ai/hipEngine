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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hipengine.kernels.cpu_reference  # noqa: F401,E402 - self-registers CPU kernels
from hipengine.kernels.registry import resolve  # noqa: E402
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402
from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPContext  # noqa: E402


DEFAULT_FIXTURE = Path("benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json")


@dataclass(frozen=True)
class _FixtureHiddenContract:
    ready_for_mtp: bool
    rows: int
    hidden_size: int


@dataclass(frozen=True)
class _FixtureSeed:
    token_id: int
    position: int
    hidden_ptr: int
    hidden_contract: _FixtureHiddenContract


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
    topk_kernel = resolve(
        backend="cpu_reference",
        layer="mtp_draft_topk",
        quant="w4_gguf",
        variant="full_vocab_d2h",
    )
    actual_top_k, actual_top_k_logits = topk_kernel(actual_logits, k=top_k)
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
    draft_execution_plan = _build_draft_execution_plan_summary(
        fixture,
        actual_logits=actual_logits,
        top_k=top_k,
    )
    passed = metrics["max_kl"] <= float(max_kl) and metrics["top1_agreement"] >= float(
        min_top1_agreement
    )
    return {
        "schema": 1,
        "kind": "gguf_mtp_oracle_gate",
        "fixture": str(fixture_path),
        "cpu_reference_kernel": fixture["cpu_reference_kernel"],
        "draft_topk_kernel": ["cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h"],
        "thresholds": {
            "max_kl": float(max_kl),
            "min_top1_agreement": float(min_top1_agreement),
        },
        "metrics": metrics,
        "actual_top1_token_ids": actual_top1.astype(int).tolist(),
        "expected_top1_token_ids": expected_top1.astype(int).tolist(),
        "actual_top_k_token_ids": actual_top_k.astype(int).tolist(),
        "actual_top_k_logits": actual_top_k_logits.astype(float).tolist(),
        "expected_top_k_token_ids": expected_top_k.astype(int).tolist(),
        "draft_execution_plan": draft_execution_plan,
        "passed": bool(passed),
    }


def _build_draft_execution_plan_summary(
    fixture: dict[str, Any],
    *,
    actual_logits: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    hidden_seed = _f32(fixture["inputs"]["hidden_seed"])
    if hidden_seed.ndim != 2 or hidden_seed.shape[0] < 1:
        raise ValueError("hidden_seed must have shape [rows, hidden_size]")
    seed = _FixtureSeed(
        token_id=int(fixture.get("seed_token_id", 0)),
        position=int(fixture.get("seed_position", 0)),
        hidden_ptr=int(fixture.get("seed_hidden_ptr", 1)),
        hidden_contract=_FixtureHiddenContract(
            ready_for_mtp=True,
            rows=1,
            hidden_size=int(hidden_seed.shape[1]),
        ),
    )
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(seed, source="oracle_fixture")
    plan = context.build_draft_execution_plan_from_logits(
        request_id=int(fixture.get("request_id", 0)),
        logits=actual_logits,
        top_k=int(top_k),
        block_size=int(fixture.get("block_size", 256)),
        storage_dtype=str(fixture.get("kv_storage_dtype", "bf16")),
    )
    return plan.as_dict()


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
