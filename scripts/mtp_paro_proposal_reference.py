#!/usr/bin/env python3
"""Compare an uv-native PARO MTP capture with an independent CPU/torch reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

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
    root_token = int(payload["root_token"][0])
    native_token = int(payload["native_token"][0])
    tail_start = int(payload["tail_start"][0])

    import torch
    from scripts.mtp_torch_proposal_smoke import _advance, _rope_tables

    device = torch.device("cpu")
    cfg, embed, lm_head, weights = _load_torch_reference(model, device)
    cos, sin = _rope_tables(
        len(prompt_tokens) + 8,
        int(cfg.rotary_dim or cfg.head_dim),
        float(cfg.rope_theta),
        device=device,
    )
    state = None
    for idx in range(tail_start, len(prompt_tokens)):
        input_token = prompt_tokens[idx + 1] if idx + 1 < len(prompt_tokens) else root_token
        hidden_row = torch.from_numpy(
            _bf16_bits_to_f32(target_hidden_bits[idx : idx + 1]).copy()
        ).to(device=device, dtype=torch.bfloat16)
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
    native_top8 = _stable_topk(native_logits, 8)
    reference_top8 = _stable_topk(reference_w8_logits, 8)
    hidden_delta = np.abs(native_hidden_f32 - reference_hidden_f32)
    checks = {
        "native_fused_matches_materialized": native_token == native_top8[0],
        "native_reference_top1": native_top8[0] == reference_top8[0],
        "native_reference_top8": native_top8 == reference_top8,
        "native_hidden_finite": bool(np.isfinite(native_hidden_f32).all()),
        "reference_hidden_finite": bool(np.isfinite(reference_hidden_f32).all()),
        "native_logits_finite": bool(np.isfinite(native_logits).all()),
        "reference_logits_finite": bool(np.isfinite(reference_w8_logits).all()),
    }
    return {
        "schema": "hipengine.paro_mtp_proposal_two_process_parity.v1",
        "status": "passed" if all(checks.values()) else "mismatch",
        "performance_claim": False,
        "model": str(model),
        "capture": str(capture),
        "prompt_tokens": len(prompt_tokens),
        "tail_replay_tokens": len(prompt_tokens) - tail_start,
        "root_token": root_token,
        "native_token": native_token,
        "native_top8": native_top8,
        "cpu_reference_top8": reference_top8,
        "hidden": {
            "max_abs_native_vs_reference": float(hidden_delta.max(initial=0.0)),
            "mean_abs_native_vs_reference": float(hidden_delta.mean()),
        },
        "checks": checks,
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
    print(json.dumps({"status": result["status"], "native_top8": result["native_top8"], "reference_top8": result["cpu_reference_top8"], "hidden": result["hidden"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
