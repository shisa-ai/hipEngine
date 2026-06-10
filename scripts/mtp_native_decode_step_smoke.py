#!/usr/bin/env python3
"""Native Qwen3.6 MTP proposal-chain smoke.

This script is a correctness-first bring-up harness for task #41.  It executes
one or more MTP decode/proposal steps with torch-free hipEngine kernels for the
runtime path, then optionally compares the produced candidate chain with the
optional torch reference from ``scripts/mtp_torch_proposal_smoke.py``.

It is not a throughput benchmark: the harness intentionally keeps route/expert
orchestration simple and copies the selected expert ids to the host so existing
raw-pointer BF16 GEMV kernels can be reused per selected expert.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head, lm_head_argmax_stage1_blocks, lm_head_fp16_argmax_bf16, topk_f32_rows_i32
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_head_rmsnorm_rotary_f32,
    dflash_qkv_proj_bf16_mixed,
    dflash_silu_mul_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative.mtp import (
    build_mtp_speculative,
    mtp_accumulate_route_bf16_to_f32,
    mtp_accumulate_sigmoid_gate_bf16_to_f32,
    mtp_add_rmsnorm_bf16_oneplus,
    mtp_finalize_f32_to_bf16,
    mtp_fuse_inputs_f16_bf16,
    mtp_gate_mul_bf16,
    mtp_rmsnorm_bf16_oneplus,
    mtp_softmax_topk_f32,
    mtp_split_q_gate_f32_bf16,
)
from hipengine.loading import TensorInfo, load_tensor_info_to_device, load_weight_index, qwen35_paro_config_from_hf, validate_qwen35_mtp_model
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_paro import normalize_qwen35_weight_name
from hipengine.loading.safetensors import read_tensor_storage_bytes
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


def _f32_to_bf16_bits(array: np.ndarray) -> np.ndarray:
    f32 = np.asarray(array, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & np.uint32(1)
    u32 += np.uint32(0x7FFF) + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    u32 = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _rope_tables(max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim), dtype=np.float32)
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32)
    sin_half = np.sin(freqs).astype(np.float32)
    return np.concatenate([cos_half, cos_half], axis=1), np.concatenate([sin_half, sin_half], axis=1)


def _normalized_infos(model: str | Path) -> tuple[Any, dict[str, TensorInfo]]:
    index = load_weight_index(model)
    return index, {normalize_qwen35_weight_name(name): info for name, info in index.tensors.items()}


def _load_info(infos: dict[str, TensorInfo], name: str, allocations: list[DeviceTensorAllocation]) -> DeviceTensorAllocation:
    alloc = load_tensor_info_to_device(infos[name])
    allocations.append(alloc)
    return alloc


def _load_oneplus_bf16_weight(infos: dict[str, TensorInfo], name: str, buffers: list[DeviceBuffer]) -> DeviceBuffer:
    bits = np.frombuffer(read_tensor_storage_bytes(infos[name]), dtype=np.uint16).copy().reshape(infos[name].shape)
    transformed = _f32_to_bf16_bits(1.0 + _bf16_bits_to_f32(bits))
    buf = malloc(transformed.nbytes)
    copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(transformed)), transformed.nbytes)
    buffers.append(buf)
    return buf


def _device_array(array: np.ndarray, buffers: list[DeviceBuffer]) -> DeviceBuffer:
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes)
    copy_host_to_device(buf, host_array_ptr(contiguous), contiguous.nbytes)
    buffers.append(buf)
    return buf


def _empty_device(shape: tuple[int, ...], dtype: np.dtype[Any] | type[np.generic], buffers: list[DeviceBuffer]) -> tuple[np.ndarray, DeviceBuffer]:
    host = np.zeros(shape, dtype=dtype)
    buf = malloc(host.nbytes)
    buffers.append(buf)
    return host, buf


def _run_torch_reference(
    model: Path,
    *,
    root_token: int,
    root_position: int,
    target_hidden_bits: np.ndarray,
    draft_budget: int,
) -> dict[str, Any]:
    import torch

    from scripts.mtp_torch_proposal_smoke import _advance, _load_model, _rope_tables as _torch_rope_tables

    device = torch.device("cuda")
    cfg, embed, lm_head, weights = _load_model(model, device)
    cos, sin = _torch_rope_tables(int(root_position) + int(draft_budget) + 1, cfg.rotary_dim or cfg.head_dim, cfg.rope_theta, device=device)
    current_token = int(root_token)
    current_hidden = torch.from_numpy(_bf16_bits_to_f32(target_hidden_bits).copy()).to(device=device, dtype=torch.bfloat16)
    state = None
    candidates: list[int] = []
    finite_logits = True
    for depth in range(int(draft_budget)):
        state = _advance(
            token=current_token,
            target_hidden=current_hidden,
            state=state,
            embed_tokens=embed,
            lm_head=lm_head,
            weights=weights,
            position=int(root_position) + depth,
            cfg=cfg,
            cos=cos,
            sin=sin,
        )
        torch.cuda.synchronize()
        finite_logits = finite_logits and bool(torch.isfinite(state.logits).all().item())
        current_token = int(torch.argmax(state.logits, dim=-1).item())
        candidates.append(current_token)
        current_hidden = state.hidden.detach()
    return {"candidate_tokens": candidates, "finite_logits": finite_logits}


def _capture_target_hidden_from_session(
    model: Path,
    *,
    root_token: int,
    root_position: int,
    backend: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if root_position != 0:
        raise ValueError("target_session capture currently supports root_position=0 only; add prompt prefill before using larger positions")
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    runtime = runner.runtime
    capture_buf = None
    with Qwen35ParoResidentSession(runner, max_sequence_length=max(2, root_position + 2)) as session:
        capture_layer_id = int(session.layer_limit) - 1
        capture_host = np.zeros((1, session.config.hidden_size), dtype=np.uint16)
        capture_buf = malloc(capture_host.nbytes, runtime=runtime)
        try:
            capture_tensor = Tensor.from_handle(capture_buf.ptr, capture_host.shape, DType.BF16, Device("hip", 0))
            started = time.perf_counter()
            step_result = session.step_with_hidden_taps(
                int(root_token),
                position=int(root_position),
                capture_layer_ids=(capture_layer_id,),
                capture_hidden_concat=capture_tensor,
                capture_row=0,
                sample=True,
            )
            runtime.device_synchronize()
            capture_seconds = time.perf_counter() - started
            copy_device_to_host(host_array_ptr(capture_host), capture_buf, capture_host.nbytes, runtime=runtime)
        finally:
            if capture_buf is not None:
                free(capture_buf, runtime=runtime)
        return capture_host.copy(), {
            "source": "target_session_last_layer",
            "backend": session.backend,
            "target_arch": session.target_arch,
            "capture_layer_id": capture_layer_id,
            "capture_seconds": capture_seconds,
            "target_next_token": None if step_result is None else int(step_result.token_id),
            "target_next_logit": None if step_result is None else float(step_result.logit),
        }


def run_smoke(
    model: str | Path,
    *,
    root_token: int,
    root_position: int,
    draft_budget: int,
    torch_compare: bool,
    target_hidden_source: str = "synthetic",
    target_backend: str = "auto",
    target_hidden_bits_override: np.ndarray | None = None,
) -> dict[str, Any]:
    model = Path(model)
    validation = validate_qwen35_mtp_model(model)
    validation.raise_for_errors()
    index, infos = _normalized_infos(model)
    cfg = qwen35_paro_config_from_hf(index.config)
    hidden = int(cfg.hidden_size)
    vocab = int(cfg.vocab_size)
    q_heads = int(cfg.num_attention_heads)
    kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(cfg.head_dim)
    rotary_dim = int(cfg.rotary_dim or cfg.head_dim)
    q_proj_features = q_heads * 2 * head_dim
    q_features = q_heads * head_dim
    kv_features = kv_heads * head_dim
    top_k = int(cfg.num_experts_per_tok)
    intermediate = int(cfg.moe_intermediate_size)
    shared_intermediate = int(cfg.shared_expert_intermediate_size)
    eps = float(cfg.rms_norm_eps)
    if top_k > 8:
        raise ValueError("native smoke currently supports top_k <= 8")
    if draft_budget <= 0:
        raise ValueError("draft_budget must be positive")
    if not (0 <= root_token < vocab):
        raise ValueError(f"root_token must be in [0, {vocab})")

    if target_hidden_bits_override is not None:
        target_hidden_bits = np.ascontiguousarray(target_hidden_bits_override, dtype=np.uint16)
        if target_hidden_bits.shape != (1, hidden):
            raise ValueError(f"target_hidden_bits_override must have shape (1, {hidden}), got {target_hidden_bits.shape}")
        target_hidden_metadata: dict[str, Any] = {"source": "provided_bf16_bits"}
    elif target_hidden_source == "synthetic":
        rng = np.random.default_rng(1234)
        target_hidden_f32 = (rng.standard_normal((1, hidden), dtype=np.float32) * np.float32(0.25)).astype(np.float32)
        target_hidden_bits = _f32_to_bf16_bits(target_hidden_f32)
        target_hidden_metadata = {"source": "synthetic_rng_seed_1234"}
    elif target_hidden_source == "target_session":
        target_hidden_bits, target_hidden_metadata = _capture_target_hidden_from_session(
            model,
            root_token=int(root_token),
            root_position=int(root_position),
            backend=target_backend,
        )
    else:
        raise ValueError("target_hidden_source must be 'synthetic' or 'target_session'")
    token_ids = np.asarray([root_token], dtype=np.int64)
    positions = np.asarray([root_position], dtype=np.int32)
    max_positions = max(int(root_position) + int(draft_budget) + 1, 2)
    cos, sin = _rope_tables(max_positions, rotary_dim, float(cfg.rope_theta))

    allocations: list[DeviceTensorAllocation] = []
    buffers: list[DeviceBuffer] = []
    runtime = get_hip_runtime()
    try:
        weights = {name: _load_info(infos, name, allocations).tensor for name in [
            "embed_tokens.weight",
            "lm_head.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.fc.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
            "mtp.layers.0.mlp.shared_expert_gate.weight",
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            "mtp.norm.weight",
        ]}
        q_norm_oneplus = _load_oneplus_bf16_weight(infos, "mtp.layers.0.self_attn.q_norm.weight", buffers)
        k_norm_oneplus = _load_oneplus_bf16_weight(infos, "mtp.layers.0.self_attn.k_norm.weight", buffers)
        token_buf = _device_array(token_ids, buffers)
        target_hidden_buf = _device_array(target_hidden_bits, buffers)
        position_buf = _device_array(positions, buffers)
        cos_buf = _device_array(cos, buffers)
        sin_buf = _device_array(sin, buffers)

        _, fused_buf = _empty_device((1, 2 * hidden), np.uint16, buffers)
        _, fc_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, attn_in_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, q_proj_buf = _empty_device((1, q_proj_features), np.float32, buffers)
        _, k_proj_buf = _empty_device((1, kv_features), np.float32, buffers)
        _, v_proj_buf = _empty_device((1, kv_features), np.uint16, buffers)
        _, query_buf = _empty_device((1, q_features), np.float32, buffers)
        _, gate_buf = _empty_device((1, q_features), np.uint16, buffers)
        _, query_rot_buf = _empty_device((1, q_features), np.float32, buffers)
        _, key_rot_buf = _empty_device((1, kv_features), np.float32, buffers)
        _, key_cache_buf = _empty_device((draft_budget, kv_features), np.float32, buffers)
        _, value_cache_buf = _empty_device((draft_budget, kv_features), np.uint16, buffers)
        _, attn_out_buf = _empty_device((1, q_features), np.uint16, buffers)
        _, gated_buf = _empty_device((1, q_features), np.uint16, buffers)
        _, o_out_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, moe_in_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, residual2_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, router_logits_buf = _empty_device((1, int(cfg.num_experts)), np.float32, buffers)
        topk_values_host, topk_values_buf = _empty_device((1, top_k), np.float32, buffers)
        topk_ids_host, topk_ids_buf = _empty_device((1, top_k), np.int32, buffers)
        _, routing_buf = _empty_device((1, top_k), np.float32, buffers)
        _, gate_up_buf = _empty_device((1, 2 * intermediate), np.uint16, buffers)
        _, expert_intermediate_buf = _empty_device((1, intermediate), np.uint16, buffers)
        _, expert_down_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, moe_accum_buf = _empty_device((1, hidden), np.float32, buffers)
        _, shared_gate_buf = _empty_device((1, 1), np.float32, buffers)
        _, shared_gate_proj_buf = _empty_device((1, shared_intermediate), np.uint16, buffers)
        _, shared_up_proj_buf = _empty_device((1, shared_intermediate), np.uint16, buffers)
        _, shared_intermediate_buf = _empty_device((1, shared_intermediate), np.uint16, buffers)
        _, shared_down_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, moe_out_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, final_hidden_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, final_residual_buf = _empty_device((1, hidden), np.uint16, buffers)
        _, logits_buf = _empty_device((1, vocab), np.float32, buffers)
        block_count = lm_head_argmax_stage1_blocks(vocab, threads=256)
        _, block_values_buf = _empty_device((block_count,), np.float32, buffers)
        _, block_indices_buf = _empty_device((block_count,), np.int64, buffers)
        out_index_host, out_index_buf = _empty_device((1,), np.int64, buffers)
        out_value_host, out_value_buf = _empty_device((1,), np.float32, buffers)
        # Top-k oracle: full vocab top-k tokens per draft step (diagnostic).
        _ORACLE_K = 8
        token_topk_ids_host, token_topk_ids_buf = _empty_device((1, _ORACLE_K), np.int32, buffers)
        token_topk_values_host, token_topk_values_buf = _empty_device((1, _ORACLE_K), np.float32, buffers)

        mtp_lib = build_mtp_speculative(load=True)
        dflash_lib = build_dflash_drafter(load=True)
        lm_lib = build_lm_head(load=True)
        gate_up_base = weights["mtp.layers.0.mlp.experts.gate_up_proj"].ptr
        down_base = weights["mtp.layers.0.mlp.experts.down_proj"].ptr
        gate_up_expert_bytes = (2 * intermediate) * hidden * DType.BF16.itemsize
        down_expert_bytes = hidden * intermediate * DType.BF16.itemsize
        candidates: list[int] = []
        candidate_logits: list[float] = []
        candidate_topk: list[list[int]] = []
        candidate_topk_values: list[list[float]] = []
        topk_experts_by_step: list[list[int]] = []
        topk_logits_by_step: list[list[float]] = []
        started = time.perf_counter()
        for depth in range(int(draft_budget)):
            position_value = np.asarray([int(root_position) + depth], dtype=np.int32)
            copy_host_to_device(position_buf, host_array_ptr(position_value), position_value.nbytes)
            token_ptr = token_buf.ptr if depth == 0 else out_index_buf.ptr
            hidden_ptr = target_hidden_buf.ptr if depth == 0 else final_hidden_buf.ptr
            mtp_fuse_inputs_f16_bf16(
                token_ptr,
                weights["embed_tokens.weight"].ptr,
                hidden_ptr,
                weights["mtp.pre_fc_norm_embedding.weight"].ptr,
                weights["mtp.pre_fc_norm_hidden.weight"].ptr,
                fused_buf.ptr,
                1,
                hidden,
                vocab,
                eps=eps,
                threads=256,
                library=mtp_lib,
            )
            dflash_dense_bf16_to_bf16(fused_buf.ptr, weights["mtp.fc.weight"].ptr, fc_buf.ptr, 1, 2 * hidden, hidden, threads=128, library=dflash_lib)
            mtp_rmsnorm_bf16_oneplus(fc_buf.ptr, weights["mtp.layers.0.input_layernorm.weight"].ptr, attn_in_buf.ptr, 1, hidden, eps=eps, threads=256, library=mtp_lib)
            dflash_qkv_proj_bf16_mixed(
                attn_in_buf.ptr,
                weights["mtp.layers.0.self_attn.q_proj.weight"].ptr,
                weights["mtp.layers.0.self_attn.k_proj.weight"].ptr,
                weights["mtp.layers.0.self_attn.v_proj.weight"].ptr,
                q_proj_buf.ptr,
                k_proj_buf.ptr,
                v_proj_buf.ptr,
                1,
                hidden,
                q_proj_features,
                kv_features,
                threads=128,
                library=dflash_lib,
            )
            mtp_split_q_gate_f32_bf16(q_proj_buf.ptr, query_buf.ptr, gate_buf.ptr, 1, q_heads, head_dim, threads=256, library=mtp_lib)
            dflash_head_rmsnorm_rotary_f32(
                query_buf.ptr,
                k_proj_buf.ptr,
                q_norm_oneplus.ptr,
                k_norm_oneplus.ptr,
                cos_buf.ptr,
                sin_buf.ptr,
                position_buf.ptr,
                position_buf.ptr,
                query_rot_buf.ptr,
                key_rot_buf.ptr,
                1,
                1,
                1,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                max_positions,
                eps=eps,
                threads=128,
                library=dflash_lib,
            )
            runtime.memcpy(key_cache_buf.ptr + depth * kv_features * DType.FP32.itemsize, key_rot_buf.ptr, kv_features * DType.FP32.itemsize, HipMemcpyKind.DEVICE_TO_DEVICE)
            runtime.memcpy(value_cache_buf.ptr + depth * kv_features * DType.BF16.itemsize, v_proj_buf.ptr, kv_features * DType.BF16.itemsize, HipMemcpyKind.DEVICE_TO_DEVICE)
            dflash_gqa_attention_f32_bf16(query_rot_buf.ptr, key_cache_buf.ptr, value_cache_buf.ptr, attn_out_buf.ptr, 1, 1, depth + 1, q_heads, kv_heads, head_dim, threads=128, library=dflash_lib)
            mtp_gate_mul_bf16(attn_out_buf.ptr, gate_buf.ptr, gated_buf.ptr, q_features, threads=256, library=mtp_lib)
            dflash_dense_bf16_to_bf16(gated_buf.ptr, weights["mtp.layers.0.self_attn.o_proj.weight"].ptr, o_out_buf.ptr, 1, q_features, hidden, threads=128, library=dflash_lib)
            mtp_add_rmsnorm_bf16_oneplus(
                o_out_buf.ptr,
                fc_buf.ptr,
                weights["mtp.layers.0.post_attention_layernorm.weight"].ptr,
                moe_in_buf.ptr,
                residual2_buf.ptr,
                1,
                hidden,
                eps=eps,
                threads=256,
                library=mtp_lib,
            )
            dflash_dense_bf16_to_f32(moe_in_buf.ptr, weights["mtp.layers.0.mlp.gate.weight"].ptr, router_logits_buf.ptr, 1, hidden, int(cfg.num_experts), threads=128, library=dflash_lib)
            topk_f32_rows_i32(router_logits_buf.ptr, topk_values_buf.ptr, topk_ids_buf.ptr, 1, int(cfg.num_experts), top_k, threads=256, library=lm_lib)
            mtp_softmax_topk_f32(topk_values_buf.ptr, routing_buf.ptr, 1, top_k, library=mtp_lib)
            copy_device_to_host(host_array_ptr(topk_ids_host), topk_ids_buf, topk_ids_host.nbytes)
            copy_device_to_host(host_array_ptr(topk_values_host), topk_values_buf, topk_values_host.nbytes)
            topk_experts_by_step.append([int(x) for x in topk_ids_host.reshape(-1)])
            topk_logits_by_step.append([float(x) for x in topk_values_host.reshape(-1)])
            runtime.memset(moe_accum_buf.ptr, 0, moe_accum_buf.nbytes)
            for route, expert_id_value in enumerate(topk_ids_host.reshape(-1)):
                expert_id = int(expert_id_value)
                dflash_dense_bf16_to_bf16(
                    moe_in_buf.ptr,
                    gate_up_base + expert_id * gate_up_expert_bytes,
                    gate_up_buf.ptr,
                    1,
                    hidden,
                    2 * intermediate,
                    threads=128,
                    library=dflash_lib,
                )
                dflash_silu_mul_bf16(gate_up_buf.ptr, gate_up_buf.ptr + intermediate * DType.BF16.itemsize, expert_intermediate_buf.ptr, intermediate, threads=256, library=dflash_lib)
                dflash_dense_bf16_to_bf16(
                    expert_intermediate_buf.ptr,
                    down_base + expert_id * down_expert_bytes,
                    expert_down_buf.ptr,
                    1,
                    intermediate,
                    hidden,
                    threads=128,
                    library=dflash_lib,
                )
                mtp_accumulate_route_bf16_to_f32(expert_down_buf.ptr, routing_buf.ptr, moe_accum_buf.ptr, hidden, route, threads=256, library=mtp_lib)
            dflash_dense_bf16_to_f32(moe_in_buf.ptr, weights["mtp.layers.0.mlp.shared_expert_gate.weight"].ptr, shared_gate_buf.ptr, 1, hidden, 1, threads=128, library=dflash_lib)
            dflash_dense_bf16_to_bf16(moe_in_buf.ptr, weights["mtp.layers.0.mlp.shared_expert.gate_proj.weight"].ptr, shared_gate_proj_buf.ptr, 1, hidden, shared_intermediate, threads=128, library=dflash_lib)
            dflash_dense_bf16_to_bf16(moe_in_buf.ptr, weights["mtp.layers.0.mlp.shared_expert.up_proj.weight"].ptr, shared_up_proj_buf.ptr, 1, hidden, shared_intermediate, threads=128, library=dflash_lib)
            dflash_silu_mul_bf16(shared_gate_proj_buf.ptr, shared_up_proj_buf.ptr, shared_intermediate_buf.ptr, shared_intermediate, threads=256, library=dflash_lib)
            dflash_dense_bf16_to_bf16(shared_intermediate_buf.ptr, weights["mtp.layers.0.mlp.shared_expert.down_proj.weight"].ptr, shared_down_buf.ptr, 1, shared_intermediate, hidden, threads=128, library=dflash_lib)
            mtp_accumulate_sigmoid_gate_bf16_to_f32(shared_down_buf.ptr, shared_gate_buf.ptr, moe_accum_buf.ptr, hidden, threads=256, library=mtp_lib)
            mtp_finalize_f32_to_bf16(moe_accum_buf.ptr, moe_out_buf.ptr, hidden, threads=256, library=mtp_lib)
            mtp_add_rmsnorm_bf16_oneplus(moe_out_buf.ptr, residual2_buf.ptr, weights["mtp.norm.weight"].ptr, final_hidden_buf.ptr, final_residual_buf.ptr, 1, hidden, eps=eps, threads=256, library=mtp_lib)
            lm_head_fp16_argmax_bf16(
                final_hidden_buf.ptr,
                weights["lm_head.weight"].ptr,
                logits_buf.ptr,
                block_values_buf.ptr,
                block_indices_buf.ptr,
                out_index_buf.ptr,
                out_value_buf.ptr,
                hidden,
                vocab,
                threads=256,
                library=lm_lib,
            )
            topk_f32_rows_i32(logits_buf.ptr, token_topk_values_buf.ptr, token_topk_ids_buf.ptr, 1, vocab, _ORACLE_K, threads=256, library=lm_lib)
            copy_device_to_host(host_array_ptr(out_index_host), out_index_buf, out_index_host.nbytes)
            copy_device_to_host(host_array_ptr(out_value_host), out_value_buf, out_value_host.nbytes)
            copy_device_to_host(host_array_ptr(token_topk_ids_host), token_topk_ids_buf, token_topk_ids_host.nbytes)
            copy_device_to_host(host_array_ptr(token_topk_values_host), token_topk_values_buf, token_topk_values_host.nbytes)
            candidates.append(int(out_index_host[0]))
            candidate_logits.append(float(out_value_host[0]))
            candidate_topk.append([int(x) for x in token_topk_ids_host.reshape(-1)])
            candidate_topk_values.append([float(x) for x in token_topk_values_host.reshape(-1)])
        runtime.device_synchronize()
        native_seconds = time.perf_counter() - started
    finally:
        for buf in reversed(buffers):
            free(buf)
        for alloc in reversed(allocations):
            alloc.free()

    draft = compile_mtp_chain(
        [MtpDraftRequest(request_id=0, root_position=int(root_position), candidate_tokens=tuple(candidates), active_count=len(candidates))],
        candidate_budget=int(draft_budget),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(int(root_token),), root_positions=(int(root_position),))
    result: dict[str, Any] = {
        "status": "passed",
        "model": str(model),
        "root_token": int(root_token),
        "root_position": int(root_position),
        "draft_budget": int(draft_budget),
        "candidate_tokens": candidates,
        "candidate_logits": candidate_logits,
        "candidate_topk": candidate_topk,
        "candidate_topk_values": candidate_topk_values,
        "candidate_token": candidates[-1],
        "candidate_logit": candidate_logits[-1],
        "native_seconds": native_seconds,
        "target_hidden": target_hidden_metadata,
        "topk_experts": topk_experts_by_step,
        "topk_logits": topk_logits_by_step,
        "mtp_validation": validation.to_json_dict(),
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
        "note": "Native MTP proposal-chain smoke; expert ids are host-orchestrated and this is not a throughput benchmark.",
    }
    if torch_compare:
        reference = _run_torch_reference(model, root_token=root_token, root_position=root_position, target_hidden_bits=target_hidden_bits, draft_budget=int(draft_budget))
        result["torch_reference"] = reference
        result["candidate_matches_torch"] = bool(candidates == list(reference["candidate_tokens"]))
        if not result["candidate_matches_torch"]:
            result["status"] = "candidate_mismatch"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--root-token", type=int, default=151646)
    parser.add_argument("--root-position", type=int, default=0)
    parser.add_argument("--draft-budget", type=int, default=1)
    parser.add_argument("--target-hidden-source", choices=("synthetic", "target_session"), default="synthetic")
    parser.add_argument("--target-backend", default="auto", help="Backend for target-session hidden capture; default auto")
    parser.add_argument("--torch-compare", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = run_smoke(
        args.model,
        root_token=args.root_token,
        root_position=args.root_position,
        draft_budget=int(args.draft_budget),
        torch_compare=bool(args.torch_compare),
        target_hidden_source=str(args.target_hidden_source),
        target_backend=str(args.target_backend),
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "candidates": result["candidate_tokens"], "torch": result.get("torch_reference", {}).get("candidate_tokens"), "matches": result.get("candidate_matches_torch"), "native_seconds": result["native_seconds"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
