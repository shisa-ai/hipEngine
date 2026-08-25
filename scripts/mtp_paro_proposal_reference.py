#!/usr/bin/env python3
"""Compare an uv-native PARO MTP capture with an independent CPU/torch reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from scripts.mtp_paro_proposal_parity import (
    _bf16_bits_to_f32,
    _load_torch_reference,
    _stable_topk,
    _torch_w8_logits,
)


def run(*, model: Path, capture: Path) -> dict[str, Any]:
    started = time.perf_counter()
    payload = np.load(capture)
    prompt_tokens = [int(token) for token in payload["prompt_tokens"].tolist()]
    target_hidden_bits = np.asarray(payload["target_hidden_bits"], dtype=np.uint16)
    native_hidden_bits = np.asarray(payload["native_hidden_bits"], dtype=np.uint16)
    native_logits = np.asarray(payload["native_logits"], dtype=np.float32)
    native_stage_bits = {
        "input_fusion": np.asarray(payload["native_fused_bits"], dtype=np.uint16),
        "fc": np.asarray(payload["native_fc_bits"], dtype=np.uint16),
        "attn_input_norm": np.asarray(payload["native_attn_in_bits"], dtype=np.uint16),
    }
    root_token = int(payload["root_token"][0])
    native_token = int(payload["native_token"][0])
    tail_start = int(payload["tail_start"][0])

    import torch
    import torch.nn.functional as F
    from scripts.mtp_torch_proposal_smoke import (
        _advance,
        _linear_bf16,
        _rmsnorm,
        _rope_tables,
    )

    device = torch.device("cpu")
    cfg, embed, lm_head, weights = _load_torch_reference(model, device)
    cos, sin = _rope_tables(
        len(prompt_tokens) + 8,
        int(cfg.rotary_dim or cfg.head_dim),
        float(cfg.rope_theta),
        device=device,
    )
    state = None
    reference_stage_f32: dict[str, np.ndarray] = {}
    for idx in range(tail_start, len(prompt_tokens)):
        input_token = prompt_tokens[idx + 1] if idx + 1 < len(prompt_tokens) else root_token
        hidden_row = torch.from_numpy(
            _bf16_bits_to_f32(target_hidden_bits[idx : idx + 1]).copy()
        ).to(device=device, dtype=torch.bfloat16)
        input_id = torch.tensor([int(input_token)], dtype=torch.long, device=device)
        input_embed = F.embedding(input_id, embed.to(dtype=hidden_row.dtype))
        embed_norm = _rmsnorm(
            input_embed, weights["mtp.pre_fc_norm_embedding.weight"], cfg.rms_norm_eps
        )
        hidden_norm = _rmsnorm(
            hidden_row, weights["mtp.pre_fc_norm_hidden.weight"], cfg.rms_norm_eps
        )
        fused = torch.cat((embed_norm, hidden_norm), dim=-1)
        fc = _linear_bf16(fused, weights["mtp.fc.weight"])
        attn_in = _rmsnorm(
            fc, weights["mtp.layers.0.input_layernorm.weight"], cfg.rms_norm_eps
        )
        reference_stage_f32 = {
            "input_fusion": fused.float().numpy(),
            "fc": fc.float().numpy(),
            "attn_input_norm": attn_in.float().numpy(),
        }
        state = _advance(
            token=int(input_token),
            target_hidden=hidden_row,
            state=state,
            embed_tokens=embed,
            lm_head=lm_head[:1],
            weights=weights,
            position=idx + 1,
            cfg=cfg,
            cos=cos,
            sin=sin,
        )
    if state is None:
        raise RuntimeError("capture contained no replay transition")
    reference_hidden_f32 = state.hidden.float().numpy()
    native_hidden_f32 = _bf16_bits_to_f32(native_hidden_bits)
    reference_w8_logits = _torch_w8_logits(lm_head, state.hidden).numpy()
    from scripts.quant_quality.metrics import per_row_metrics

    distribution = per_row_metrics(
        reference_w8_logits.reshape(1, -1),
        native_logits.reshape(1, -1),
        np.asarray([int(np.argmax(reference_w8_logits))], dtype=np.int64),
        top_k=5,
    )
    native_top8 = _stable_topk(native_logits, 8)
    reference_top8 = _stable_topk(reference_w8_logits, 8)
    hidden_delta = np.abs(native_hidden_f32 - reference_hidden_f32)
    hidden_norm = float(np.linalg.norm(reference_hidden_f32.reshape(-1)))
    hidden_relative_l2 = float(
        np.linalg.norm((native_hidden_f32 - reference_hidden_f32).reshape(-1))
        / max(hidden_norm, 1.0e-12)
    )
    hidden_cosine = float(
        np.dot(native_hidden_f32.reshape(-1), reference_hidden_f32.reshape(-1))
        / max(
            float(np.linalg.norm(native_hidden_f32.reshape(-1))) * hidden_norm,
            1.0e-12,
        )
    )
    stages: dict[str, Any] = {}
    for name, native_bits in native_stage_bits.items():
        native_f32 = _bf16_bits_to_f32(native_bits)
        reference_f32 = reference_stage_f32[name]
        delta = np.abs(native_f32 - reference_f32)
        stages[name] = {
            "bits_equal": bool(
                np.array_equal(
                    native_bits,
                    (reference_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16),
                )
            ),
            "max_abs": float(delta.max(initial=0.0)),
            "mean_abs": float(delta.mean()),
        }
    first_stage_mismatch = next(
        (name for name, summary in stages.items() if not summary["bits_equal"]), None
    )
    thresholds = EvaluationThresholds()
    fixed_chain_checks = {
        "native_fused_matches_materialized": native_token == native_top8[0],
        "native_reference_top1": native_top8[0] == reference_top8[0],
        "kl_outer_floor": float(distribution["kl_nats"][0]) <= 0.05,
        "top5_set_overlap": float(distribution["topk_set_overlap"][0]) == 1.0,
        "finite": bool(np.isfinite(native_logits).all() and np.isfinite(reference_w8_logits).all()),
    }
    checks = {
        **fixed_chain_checks,
        "native_reference_top8": native_top8 == reference_top8,
        "native_hidden_finite": bool(np.isfinite(native_hidden_f32).all()),
        "reference_hidden_finite": bool(np.isfinite(reference_hidden_f32).all()),
        "native_logits_finite": bool(np.isfinite(native_logits).all()),
        "reference_logits_finite": bool(np.isfinite(reference_w8_logits).all()),
    }
    strict_parity = all(checks.values()) and first_stage_mismatch is None
    fixed_chain_passed = all(fixed_chain_checks.values())
    return {
        "schema": "hipengine.paro_mtp_proposal_two_process_parity.v2",
        "status": (
            "passed_strict" if strict_parity else (
                "passed_fixed_chain" if fixed_chain_passed else "mismatch"
            )
        ),
        "performance_claim": False,
        "model": str(model),
        "capture": str(capture),
        "prompt_tokens": len(prompt_tokens),
        "tail_replay_tokens": len(prompt_tokens) - tail_start,
        "root_token": root_token,
        "native_token": native_token,
        "native_top8": native_top8,
        "cpu_reference_top8": reference_top8,
        "distribution": {
            "kl_nats": float(distribution["kl_nats"][0]),
            "top1_equal": bool(distribution["top1_equal"][0]),
            "top5_set_overlap": float(distribution["topk_set_overlap"][0]),
            "max_abs_logit_delta": float(distribution["max_abs_logit_delta"][0]),
            "production_mean_kl_threshold": float(thresholds.mean_kl_max),
            "outer_kl_floor": 0.05,
        },
        "hidden": {
            "max_abs_native_vs_reference": float(hidden_delta.max(initial=0.0)),
            "mean_abs_native_vs_reference": float(hidden_delta.mean()),
            "relative_l2": hidden_relative_l2,
            "cosine": hidden_cosine,
        },
        "stages": stages,
        "first_stage_mismatch": first_stage_mismatch,
        "fixed_chain_checks": fixed_chain_checks,
        "strict_parity_checks": checks,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(model=args.model, capture=args.capture)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "native_top8": result["native_top8"], "reference_top8": result["cpu_reference_top8"], "distribution": result["distribution"], "hidden": result["hidden"]}, sort_keys=True))
    return 0 if str(result["status"]).startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
