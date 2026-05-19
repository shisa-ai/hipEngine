#!/usr/bin/env python3
"""Torch reference smoke for the full one-layer Qwen3.6 MTP proposal path.

This is an optional bring-up script, not the hipEngine hot path.  It exercises the
assembled PARO+MTP-BF16 weights end-to-end through the MTP proposal head and
emits the same candidate-only DraftBatch / TargetVerifyBatch metadata that the
native provider must produce.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading import load_weight_index, qwen35_paro_config_from_hf, validate_qwen35_mtp_model
from hipengine.loading.qwen35_paro import normalize_qwen35_weight_name
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


@dataclass
class MtpState:
    hidden: torch.Tensor
    logits: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    position: int


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Qwen3.5/Qwen3.6 centered RMSNorm weights are applied as (1 + weight).
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    return (xf * inv * (1.0 + weight.float())).to(x.dtype)


def _add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = (x.to(torch.bfloat16) + residual.to(torch.bfloat16)).to(torch.bfloat16)
    return _rmsnorm(residual_out, weight, eps), residual_out


def _rope_tables(max_positions: int, rotary_dim: int, base: float, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_positions, dtype=torch.float32, device=device)[:, None]
    dims = torch.arange(rotary_dim // 2, dtype=torch.float32, device=device)[None, :]
    inv_freq = torch.pow(torch.tensor(float(base), dtype=torch.float32, device=device), -2.0 * dims / float(rotary_dim))
    freqs = positions * inv_freq
    cos = torch.cat((torch.cos(freqs), torch.cos(freqs)), dim=1)
    sin = torch.cat((torch.sin(freqs), torch.sin(freqs)), dim=1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position: int, rotary_dim: int) -> torch.Tensor:
    if rotary_dim <= 0:
        return x
    x_rot = x[..., :rotary_dim].float()
    x_pass = x[..., rotary_dim:]
    c = cos[int(position)].view(*([1] * (x_rot.dim() - 1)), rotary_dim)
    s = sin[int(position)].view(*([1] * (x_rot.dim() - 1)), rotary_dim)
    out = x_rot * c + _rotate_half(x_rot) * s
    return torch.cat((out.to(x.dtype), x_pass), dim=-1) if x_pass.numel() else out.to(x.dtype)


def _linear_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x.to(torch.bfloat16), weight.to(dtype=torch.bfloat16)).to(torch.bfloat16)


def _attention_decode(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    position: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_proj = _linear_bf16(hidden, weights["mtp.layers.0.self_attn.q_proj.weight"])
    k_proj = _linear_bf16(hidden, weights["mtp.layers.0.self_attn.k_proj.weight"])
    v_proj = _linear_bf16(hidden, weights["mtp.layers.0.self_attn.v_proj.weight"])
    q_proj = q_proj.view(1, num_q_heads, 2 * head_dim)
    query, gate = q_proj.split(head_dim, dim=-1)
    gate = gate.reshape(1, num_q_heads * head_dim)
    key = k_proj.view(1, num_kv_heads, head_dim)
    value = v_proj.view(1, num_kv_heads, head_dim)
    query = _rmsnorm(query, weights["mtp.layers.0.self_attn.q_norm.weight"], eps)
    key = _rmsnorm(key, weights["mtp.layers.0.self_attn.k_norm.weight"], eps)
    query = _apply_rope(query, cos, sin, position, rotary_dim)
    key = _apply_rope(key, cos, sin, position, rotary_dim)
    key_cache = key if key_cache is None else torch.cat((key_cache, key), dim=0)
    value_cache = value if value_cache is None else torch.cat((value_cache, value), dim=0)

    group = num_q_heads // num_kv_heads
    key_for_q = key_cache[:, torch.arange(num_q_heads, device=hidden.device) // group, :]  # [T, qh, hd]
    value_for_q = value_cache[:, torch.arange(num_q_heads, device=hidden.device) // group, :]
    q = query.squeeze(0).float()  # [q_heads, head_dim]
    scores = torch.einsum("qh,tqh->qt", q, key_for_q.float())
    scores = scores * (float(head_dim) ** -0.5)
    attn = torch.softmax(scores, dim=-1)
    context = torch.einsum("qt,tqh->qh", attn, value_for_q.float()).reshape(1, num_q_heads * head_dim)
    gated = context * torch.sigmoid(gate.float())
    out = _linear_bf16(gated.to(torch.bfloat16), weights["mtp.layers.0.self_attn.o_proj.weight"])
    return out, key_cache.contiguous(), value_cache.contiguous()


def _moe(hidden: torch.Tensor, weights: dict[str, torch.Tensor], *, top_k: int, intermediate_size: int) -> torch.Tensor:
    router = F.linear(hidden.float(), weights["mtp.layers.0.mlp.gate.weight"].float())
    topk_logits, selected = torch.topk(router, int(top_k), dim=-1)
    routing = torch.softmax(topk_logits.float(), dim=-1)
    out = torch.zeros_like(hidden.float())
    gate_up_all = weights["mtp.layers.0.mlp.experts.gate_up_proj"]
    down_all = weights["mtp.layers.0.mlp.experts.down_proj"]
    for route in range(int(top_k)):
        expert = int(selected[0, route].item())
        gate_up = _linear_bf16(hidden, gate_up_all[expert])
        gate, up = gate_up.split(int(intermediate_size), dim=-1)
        intermediate = F.silu(gate.float()) * up.float()
        down = _linear_bf16(intermediate.to(torch.bfloat16), down_all[expert]).float()
        out += down * routing[0, route].float()
    shared_gate = F.linear(hidden.float(), weights["mtp.layers.0.mlp.shared_expert_gate.weight"].float())
    shared_gate_proj = _linear_bf16(hidden, weights["mtp.layers.0.mlp.shared_expert.gate_proj.weight"])
    shared_up_proj = _linear_bf16(hidden, weights["mtp.layers.0.mlp.shared_expert.up_proj.weight"])
    shared_intermediate = F.silu(shared_gate_proj.float()) * shared_up_proj.float()
    shared_down = _linear_bf16(shared_intermediate.to(torch.bfloat16), weights["mtp.layers.0.mlp.shared_expert.down_proj.weight"]).float()
    out = out + torch.sigmoid(shared_gate.float()) * shared_down
    return out.to(torch.bfloat16)


def _advance(
    *,
    token: int,
    target_hidden: torch.Tensor,
    state: MtpState | None,
    embed_tokens: torch.Tensor,
    lm_head: torch.Tensor,
    weights: dict[str, torch.Tensor],
    position: int,
    cfg: Any,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> MtpState:
    input_id = torch.tensor([int(token)], dtype=torch.long, device=embed_tokens.device)
    input_embed = F.embedding(input_id, embed_tokens.to(dtype=target_hidden.dtype))
    embed_norm = _rmsnorm(input_embed, weights["mtp.pre_fc_norm_embedding.weight"], cfg.rms_norm_eps)
    hidden_norm = _rmsnorm(target_hidden.to(torch.bfloat16), weights["mtp.pre_fc_norm_hidden.weight"], cfg.rms_norm_eps)
    fused = _linear_bf16(torch.cat((embed_norm, hidden_norm), dim=-1), weights["mtp.fc.weight"])
    residual = fused.to(torch.bfloat16)
    attn_in = _rmsnorm(residual, weights["mtp.layers.0.input_layernorm.weight"], cfg.rms_norm_eps)
    attn_out, key_cache, value_cache = _attention_decode(
        attn_in,
        weights,
        key_cache=None if state is None else state.key_cache,
        value_cache=None if state is None else state.value_cache,
        position=position,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rotary_dim=cfg.rotary_dim or cfg.head_dim,
        cos=cos,
        sin=sin,
        eps=cfg.rms_norm_eps,
    )
    moe_in, residual2 = _add_rmsnorm(attn_out, residual, weights["mtp.layers.0.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    moe_out = _moe(moe_in, weights, top_k=cfg.num_experts_per_tok, intermediate_size=cfg.moe_intermediate_size)
    hidden, _ = _add_rmsnorm(moe_out, residual2, weights["mtp.norm.weight"], cfg.rms_norm_eps)
    logits = F.linear(hidden.float(), lm_head.float())
    return MtpState(hidden=hidden.contiguous(), logits=logits.contiguous(), key_cache=key_cache, value_cache=value_cache, position=int(position))


def _load_model(model: Path, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    validation = validate_qwen35_mtp_model(model)
    validation.raise_for_errors()
    index = load_weight_index(model)
    cfg = qwen35_paro_config_from_hf(index.config)
    infos = {normalize_qwen35_weight_name(name): info for name, info in index.tensors.items()}
    embed_info = infos["embed_tokens.weight"]
    lm_info = infos["lm_head.weight"]
    with safe_open(str(embed_info.shard_path), framework="pt", device=str(device)) as handle:
        embed = handle.get_tensor(embed_info.name).contiguous()
    with safe_open(str(lm_info.shard_path), framework="pt", device=str(device)) as handle:
        lm_head = handle.get_tensor(lm_info.name).contiguous()
    sidecar = model / "mtp-bf16.safetensors"
    weights = {name: tensor.to(device=device).contiguous() for name, tensor in load_file(str(sidecar), device=str(device)).items()}
    return cfg, embed, lm_head, weights


def run(model: Path, *, root_token: int, root_position: int, budget: int, device: torch.device) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg, embed, lm_head, weights = _load_model(model, device)
    load_seconds = time.perf_counter() - t0
    cos, sin = _rope_tables(root_position + budget + 2, cfg.rotary_dim or cfg.head_dim, cfg.rope_theta, device=device)
    # Synthetic target-hidden root row for this proposal-path smoke.  The E2E
    # benchmark will pass the captured target final hidden from Qwen35Paro.
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    target_hidden = (torch.randn((1, cfg.hidden_size), dtype=torch.float32, device=device, generator=gen) * 0.25).to(torch.bfloat16)
    proposal_state: MtpState | None = None
    current_token = int(root_token)
    current_hidden = target_hidden
    candidates: list[int] = []
    proposal_seconds = 0.0
    for depth in range(int(budget)):
        started = time.perf_counter()
        proposal_state = _advance(
            token=current_token,
            target_hidden=current_hidden,
            state=proposal_state,
            embed_tokens=embed,
            lm_head=lm_head,
            weights=weights,
            position=int(root_position) + depth,
            cfg=cfg,
            cos=cos,
            sin=sin,
        )
        torch.cuda.synchronize()
        proposal_seconds += time.perf_counter() - started
        next_token = int(torch.argmax(proposal_state.logits, dim=-1).item())
        candidates.append(next_token)
        current_token = next_token
        current_hidden = proposal_state.hidden.detach()
    draft = compile_mtp_chain(
        [MtpDraftRequest(request_id=0, root_position=int(root_position), candidate_tokens=tuple(candidates), active_count=len(candidates))],
        candidate_budget=int(budget),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(int(root_token),), root_positions=(int(root_position),))
    return {
        "status": "passed",
        "model": str(model),
        "device": str(device),
        "root_token": int(root_token),
        "root_position": int(root_position),
        "candidate_budget": int(budget),
        "candidate_tokens": candidates,
        "load_seconds": load_seconds,
        "proposal_seconds": proposal_seconds,
        "finite_logits": bool(torch.isfinite(proposal_state.logits).all().item()) if proposal_state is not None else False,
        "draft_batch": {
            "request_ids": list(draft.request_ids),
            "candidate_tokens": list(draft.candidate_tokens),
            "parent_positions": list(draft.parent_positions),
            "draft_depths": list(draft.draft_depths),
            "row_to_request": list(draft.row_to_request),
            "active_mask": list(draft.active_mask),
            "mode": draft.mode,
        },
        "target_verify_batch": {
            "tokens": list(target.tokens),
            "positions": list(target.positions),
            "parent_rows": list(target.parent_rows),
            "draft_depths": list(target.draft_depths),
            "active_mask": list(target.active_mask),
            "mode": target.mode,
        },
        "note": "Torch reference full MTP proposal smoke; E2E shared-verifier benchmark still needs captured target hidden wiring or native decoder port.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--root-token", type=int, default=151646)
    parser.add_argument("--root-position", type=int, default=0)
    parser.add_argument("--draft-budget", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(args.model, root_token=args.root_token, root_position=args.root_position, budget=args.draft_budget, device=torch.device(args.device))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "budget": result["candidate_budget"], "candidates": result["candidate_tokens"], "proposal_seconds": result["proposal_seconds"], "finite_logits": result["finite_logits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
