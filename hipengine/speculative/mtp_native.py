"""Native Qwen3.6 MTP proposal runner utilities.

This module is a bring-up bridge, not a final public API.  It keeps MTP weights
and scratch resident across proposal steps so benchmark scripts can exercise the
shared verifier without per-proposal weight reloads or target-hidden D2H copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import os

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head, lm_head_argmax_stage1_blocks, lm_head_fp16_argmax_bf16, topk_f32_rows_i32
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_bf16_expert,
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


def _device_array(array: np.ndarray, buffers: list[DeviceBuffer], *, runtime: HipRuntime) -> DeviceBuffer:
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(contiguous), contiguous.nbytes, runtime=runtime)
    buffers.append(buf)
    return buf


def _empty_device(shape: tuple[int, ...], dtype: np.dtype[Any] | type[np.generic], buffers: list[DeviceBuffer], *, runtime: HipRuntime) -> tuple[np.ndarray, DeviceBuffer]:
    host = np.zeros(shape, dtype=dtype)
    buf = malloc(host.nbytes, runtime=runtime)
    buffers.append(buf)
    return host, buf


@dataclass(frozen=True, slots=True)
class NativeMtpStepResult:
    token: int
    logit: float
    topk_experts: tuple[int, ...]
    topk_logits: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NativeMtpStateSnapshot:
    slot: int
    cache_len: int
    position: int
    current: NativeMtpStepResult


class NativeMtpChainProposer:
    """Resident native MTP proposal runner for one request.

    The initial target-hidden rows are supplied as device pointers captured from
    the target session.  Selected expert ids are still copied to the host for
    pointer selection; this is explicitly a bring-up limitation.
    """

    def __init__(self, model: str | Path, *, max_positions: int, max_mtp_tokens: int, runtime: HipRuntime | None = None) -> None:
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if max_mtp_tokens <= 0:
            raise ValueError("max_mtp_tokens must be positive")
        self.model = Path(model)
        validation = validate_qwen35_mtp_model(self.model)
        validation.raise_for_errors()
        self.validation = validation
        index, infos = _normalized_infos(self.model)
        self.config = qwen35_paro_config_from_hf(index.config)
        self.infos = infos
        self.runtime = runtime or get_hip_runtime()
        self.allocations: list[DeviceTensorAllocation] = []
        self.buffers: list[DeviceBuffer] = []
        self.closed = False
        self.hidden = int(self.config.hidden_size)
        self.vocab = int(self.config.vocab_size)
        # Draft-only vocab cap: BPE token ids are merge-frequency-ordered, so
        # the first N rows of lm_head cover the hot tokens. Draft argmax over a
        # capped row range cuts the dominant per-advance GEMV bytes (~1.7 ms at
        # full 248k vocab); exactness is unaffected (drafts only steer
        # acceptance — verify commits target tokens over the FULL vocab).
        cap = int(os.environ.get("HIPENGINE_MTP_DRAFT_VOCAB_CAP", "0") or 0)
        self.draft_vocab = cap if 0 < cap < self.vocab else self.vocab
        self._vocab_topk_host: tuple | None = None
        self.q_heads = int(self.config.num_attention_heads)
        self.kv_heads = int(self.config.num_key_value_heads)
        self.head_dim = int(self.config.head_dim)
        self.rotary_dim = int(self.config.rotary_dim or self.config.head_dim)
        self.q_proj_features = self.q_heads * 2 * self.head_dim
        self.q_features = self.q_heads * self.head_dim
        self.kv_features = self.kv_heads * self.head_dim
        self.top_k = int(self.config.num_experts_per_tok)
        self.intermediate = int(self.config.moe_intermediate_size)
        self.shared_intermediate = int(self.config.shared_expert_intermediate_size)
        self.eps = float(self.config.rms_norm_eps)
        self.max_positions = int(max_positions)
        self.max_mtp_tokens = int(max_mtp_tokens)
        if self.top_k > 8:
            raise ValueError("native MTP proposer currently supports top_k <= 8")
        self.weights = {
            name: self._load_info(name).tensor
            for name in [
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
            ]
        }
        self.q_norm_oneplus = self._load_oneplus_bf16_weight("mtp.layers.0.self_attn.q_norm.weight")
        self.k_norm_oneplus = self._load_oneplus_bf16_weight("mtp.layers.0.self_attn.k_norm.weight")
        cos, sin = _rope_tables(self.max_positions, self.rotary_dim, float(self.config.rope_theta))
        self.token_host = np.zeros((1,), dtype=np.int64)
        self.position_host = np.zeros((1,), dtype=np.int32)
        self.token_buf = _device_array(self.token_host, self.buffers, runtime=self.runtime)
        self.position_buf = _device_array(self.position_host, self.buffers, runtime=self.runtime)
        self.cos_buf = _device_array(cos, self.buffers, runtime=self.runtime)
        self.sin_buf = _device_array(sin, self.buffers, runtime=self.runtime)
        self._allocate_scratch()
        self.mtp_lib = build_mtp_speculative(load=True)
        self.dflash_lib = build_dflash_drafter(load=True)
        self.lm_lib = build_lm_head(load=True)
        self.gate_up_base = self.weights["mtp.layers.0.mlp.experts.gate_up_proj"].ptr
        self.down_base = self.weights["mtp.layers.0.mlp.experts.down_proj"].ptr
        self.gate_up_expert_bytes = (2 * self.intermediate) * self.hidden * DType.BF16.itemsize
        self.down_expert_bytes = self.hidden * self.intermediate * DType.BF16.itemsize
        self.cache_len = 0
        self.position = -1
        self.current = NativeMtpStepResult(token=0, logit=float("nan"), topk_experts=(), topk_logits=())

    def _load_info(self, name: str) -> DeviceTensorAllocation:
        alloc = load_tensor_info_to_device(self.infos[name], runtime=self.runtime)
        self.allocations.append(alloc)
        return alloc

    def _load_oneplus_bf16_weight(self, name: str) -> DeviceBuffer:
        bits = np.frombuffer(read_tensor_storage_bytes(self.infos[name]), dtype=np.uint16).copy().reshape(self.infos[name].shape)
        transformed = _f32_to_bf16_bits(1.0 + _bf16_bits_to_f32(bits))
        return _device_array(transformed, self.buffers, runtime=self.runtime)

    def _allocate_scratch(self) -> None:
        _, self.fused_buf = _empty_device((1, 2 * self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.fc_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.attn_in_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.q_proj_buf = _empty_device((1, self.q_proj_features), np.float32, self.buffers, runtime=self.runtime)
        _, self.k_proj_buf = _empty_device((1, self.kv_features), np.float32, self.buffers, runtime=self.runtime)
        _, self.v_proj_buf = _empty_device((1, self.kv_features), np.uint16, self.buffers, runtime=self.runtime)
        _, self.query_buf = _empty_device((1, self.q_features), np.float32, self.buffers, runtime=self.runtime)
        _, self.gate_buf = _empty_device((1, self.q_features), np.uint16, self.buffers, runtime=self.runtime)
        _, self.query_rot_buf = _empty_device((1, self.q_features), np.float32, self.buffers, runtime=self.runtime)
        _, self.key_rot_buf = _empty_device((1, self.kv_features), np.float32, self.buffers, runtime=self.runtime)
        _, self.key_cache_buf = _empty_device((self.max_mtp_tokens, self.kv_features), np.float32, self.buffers, runtime=self.runtime)
        _, self.value_cache_buf = _empty_device((self.max_mtp_tokens, self.kv_features), np.uint16, self.buffers, runtime=self.runtime)
        _, self.attn_out_buf = _empty_device((1, self.q_features), np.uint16, self.buffers, runtime=self.runtime)
        _, self.gated_buf = _empty_device((1, self.q_features), np.uint16, self.buffers, runtime=self.runtime)
        _, self.o_out_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.moe_in_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.residual2_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.router_logits_buf = _empty_device((1, int(self.config.num_experts)), np.float32, self.buffers, runtime=self.runtime)
        self.topk_values_host, self.topk_values_buf = _empty_device((1, self.top_k), np.float32, self.buffers, runtime=self.runtime)
        self.topk_ids_host, self.topk_ids_buf = _empty_device((1, self.top_k), np.int32, self.buffers, runtime=self.runtime)
        _, self.routing_buf = _empty_device((1, self.top_k), np.float32, self.buffers, runtime=self.runtime)
        _, self.gate_up_buf = _empty_device((1, 2 * self.intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.expert_intermediate_buf = _empty_device((1, self.intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.expert_down_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.moe_accum_buf = _empty_device((1, self.hidden), np.float32, self.buffers, runtime=self.runtime)
        _, self.shared_gate_buf = _empty_device((1, 1), np.float32, self.buffers, runtime=self.runtime)
        _, self.shared_gate_proj_buf = _empty_device((1, self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.shared_up_proj_buf = _empty_device((1, self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.shared_intermediate_buf = _empty_device((1, self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.shared_down_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.moe_out_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.final_hidden_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.snapshot_hidden_buf = _empty_device((self.max_mtp_tokens, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.final_residual_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.logits_buf = _empty_device((1, self.vocab), np.float32, self.buffers, runtime=self.runtime)
        block_count = lm_head_argmax_stage1_blocks(self.vocab, threads=256)
        self.block_values_host, self.block_values_buf = _empty_device((block_count,), np.float32, self.buffers, runtime=self.runtime)
        _, self.block_indices_buf = _empty_device((block_count,), np.int64, self.buffers, runtime=self.runtime)
        self.out_index_host, self.out_index_buf = _empty_device((1,), np.int64, self.buffers, runtime=self.runtime)
        self.out_value_host, self.out_value_buf = _empty_device((1,), np.float32, self.buffers, runtime=self.runtime)

    def close(self) -> None:
        if self.closed:
            return
        self.runtime.device_synchronize()
        self.closed = True
        for buf in reversed(self.buffers):
            free(buf, runtime=self.runtime)
        for alloc in reversed(self.allocations):
            alloc.free(runtime=self.runtime)

    def __enter__(self) -> "NativeMtpChainProposer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> None:
        self.cache_len = 0
        self.position = -1
        self.current = NativeMtpStepResult(token=0, logit=float("nan"), topk_experts=(), topk_logits=())

    def prefill_from_target_hidden_rows(
        self,
        prompt_tokens: Sequence[int],
        *,
        capture_base_ptr: int,
        seed_token: int,
        capture_stride_hidden: int | None = None,
        read_expert_topk: bool = True,
    ) -> NativeMtpStepResult:
        """Run MTP prompt prefill using shifted prompt ids and target hidden rows."""

        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        self.reset()
        stride = self.hidden if capture_stride_hidden is None else int(capture_stride_hidden)
        result = self.current
        for idx, token in enumerate(prompt_tokens):
            input_token = int(prompt_tokens[idx + 1]) if idx + 1 < len(prompt_tokens) else int(seed_token)
            hidden_ptr = int(capture_base_ptr) + idx * stride * DType.BF16.itemsize
            # nano-vLLM reference uses position_offset=1 for MTP prompt prefill.
            result = self.advance(
                input_token=input_token,
                target_hidden_ptr=hidden_ptr,
                position=idx + 1,
                read_expert_topk=read_expert_topk,
            )
        return result

    def advance_with_previous_hidden(
        self,
        *,
        input_token: int,
        position: int,
        need_result: bool = True,
        read_expert_topk: bool = True,
    ) -> NativeMtpStepResult:
        return self.advance(
            input_token=int(input_token),
            target_hidden_ptr=self.final_hidden_buf.ptr,
            position=int(position),
            need_result=need_result,
            read_expert_topk=read_expert_topk,
        )

    def save_state(self, slot: int) -> NativeMtpStateSnapshot:
        if slot < 0 or slot >= self.max_mtp_tokens:
            raise ValueError("snapshot slot outside capacity")
        self.runtime.memcpy_async(
            self.snapshot_hidden_buf.ptr + int(slot) * self.hidden * DType.BF16.itemsize,
            self.final_hidden_buf.ptr,
            self.hidden * DType.BF16.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            0,
        )
        return NativeMtpStateSnapshot(slot=int(slot), cache_len=int(self.cache_len), position=int(self.position), current=self.current)

    def restore_state(self, snapshot: NativeMtpStateSnapshot) -> None:
        if snapshot.slot < 0 or snapshot.slot >= self.max_mtp_tokens:
            raise ValueError("snapshot slot outside capacity")
        self.runtime.memcpy_async(
            self.final_hidden_buf.ptr,
            self.snapshot_hidden_buf.ptr + int(snapshot.slot) * self.hidden * DType.BF16.itemsize,
            self.hidden * DType.BF16.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            0,
        )
        self.cache_len = int(snapshot.cache_len)
        self.position = int(snapshot.position)
        self.current = snapshot.current

    def vocab_topk(self, *, k: int = 8) -> tuple[list[int], list[float]]:
        """Exact top-k tokens+logits over the (capped) draft vocab.

        Runs the topk kernel over ``logits_buf`` (already populated by the
        advance's lm-head pass) and reads ids+values in one small D2H pair.
        Used for the gated branching tree (#99 -> persistent path).
        """

        if k <= 0 or k > 8:
            raise ValueError("vocab_topk supports 1..8")
        if self._vocab_topk_host is None:
            ids, ids_buf = _empty_device((1, 8), np.int32, self.buffers, runtime=self.runtime)
            vals, vals_buf = _empty_device((1, 8), np.float32, self.buffers, runtime=self.runtime)
            self._vocab_topk_host = (ids, ids_buf, vals, vals_buf)
        ids, ids_buf, vals, vals_buf = self._vocab_topk_host
        topk_f32_rows_i32(self.logits_buf.ptr, vals_buf.ptr, ids_buf.ptr, 1, self.draft_vocab, 8, threads=256, library=self.lm_lib)
        copy_device_to_host(host_array_ptr(ids), ids_buf, ids.nbytes, runtime=self.runtime)
        copy_device_to_host(host_array_ptr(vals), vals_buf, vals.nbytes, runtime=self.runtime)
        return [int(x) for x in ids.reshape(-1)[:k]], [float(x) for x in vals.reshape(-1)[:k]]

    def top1_prob_proxy(self, *, k: int = 8) -> float:
        """Softmax weight of the current top-1 over the top-k lm-head block maxima.

        Each lm-head stage-1 block covers a distinct vocab slice, so the top-k
        block maxima are a superset proxy for the vocab top-k logits — the same
        DFlash-style confidence proxy #100 used for per-position p-min draft
        truncation, without a vocab-sized D2H.
        """

        valid = lm_head_argmax_stage1_blocks(self.draft_vocab, threads=256)
        copy_device_to_host(host_array_ptr(self.block_values_host), self.block_values_buf, self.block_values_host.nbytes, runtime=self.runtime)
        vals = np.sort(self.block_values_host[:valid])[-int(k):].astype(np.float64)
        exp = np.exp(vals - vals.max())
        return float(exp[-1] / exp.sum())

    def advance(
        self,
        *,
        input_token: int,
        target_hidden_ptr: int,
        position: int,
        need_result: bool = True,
        read_expert_topk: bool = True,
    ) -> NativeMtpStepResult:
        """Advance one MTP step.

        ``need_result=False`` is for verifier repair/update advances where only
        the updated hidden/KV state is consumed by the following step. It skips
        the draft lm-head/argmax and host reads, so ``current`` is intentionally
        marked invalid until a later result-producing advance runs.
        """

        if self.closed:
            raise RuntimeError("NativeMtpChainProposer is closed")
        if self.cache_len >= self.max_mtp_tokens:
            raise RuntimeError("MTP cache capacity exceeded")
        if position < 0 or position >= self.max_positions:
            raise ValueError("position outside MTP RoPE table")
        self.token_host[0] = int(input_token)
        self.position_host[0] = int(position)
        copy_host_to_device(self.token_buf, host_array_ptr(self.token_host), self.token_host.nbytes, runtime=self.runtime)
        copy_host_to_device(self.position_buf, host_array_ptr(self.position_host), self.position_host.nbytes, runtime=self.runtime)
        mtp_fuse_inputs_f16_bf16(
            self.token_buf.ptr,
            self.weights["embed_tokens.weight"].ptr,
            int(target_hidden_ptr),
            self.weights["mtp.pre_fc_norm_embedding.weight"].ptr,
            self.weights["mtp.pre_fc_norm_hidden.weight"].ptr,
            self.fused_buf.ptr,
            1,
            self.hidden,
            self.vocab,
            eps=self.eps,
            threads=256,
            library=self.mtp_lib,
        )
        dflash_dense_bf16_to_bf16(self.fused_buf.ptr, self.weights["mtp.fc.weight"].ptr, self.fc_buf.ptr, 1, 2 * self.hidden, self.hidden, threads=128, library=self.dflash_lib)
        mtp_rmsnorm_bf16_oneplus(self.fc_buf.ptr, self.weights["mtp.layers.0.input_layernorm.weight"].ptr, self.attn_in_buf.ptr, 1, self.hidden, eps=self.eps, threads=256, library=self.mtp_lib)
        dflash_qkv_proj_bf16_mixed(
            self.attn_in_buf.ptr,
            self.weights["mtp.layers.0.self_attn.q_proj.weight"].ptr,
            self.weights["mtp.layers.0.self_attn.k_proj.weight"].ptr,
            self.weights["mtp.layers.0.self_attn.v_proj.weight"].ptr,
            self.q_proj_buf.ptr,
            self.k_proj_buf.ptr,
            self.v_proj_buf.ptr,
            1,
            self.hidden,
            self.q_proj_features,
            self.kv_features,
            threads=128,
            library=self.dflash_lib,
        )
        mtp_split_q_gate_f32_bf16(self.q_proj_buf.ptr, self.query_buf.ptr, self.gate_buf.ptr, 1, self.q_heads, self.head_dim, threads=256, library=self.mtp_lib)
        dflash_head_rmsnorm_rotary_f32(
            self.query_buf.ptr,
            self.k_proj_buf.ptr,
            self.q_norm_oneplus.ptr,
            self.k_norm_oneplus.ptr,
            self.cos_buf.ptr,
            self.sin_buf.ptr,
            self.position_buf.ptr,
            self.position_buf.ptr,
            self.query_rot_buf.ptr,
            self.key_rot_buf.ptr,
            1,
            1,
            1,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.max_positions,
            eps=self.eps,
            threads=128,
            library=self.dflash_lib,
        )
        self.runtime.memcpy_async(self.key_cache_buf.ptr + self.cache_len * self.kv_features * DType.FP32.itemsize, self.key_rot_buf.ptr, self.kv_features * DType.FP32.itemsize, HipMemcpyKind.DEVICE_TO_DEVICE, 0)
        self.runtime.memcpy_async(self.value_cache_buf.ptr + self.cache_len * self.kv_features * DType.BF16.itemsize, self.v_proj_buf.ptr, self.kv_features * DType.BF16.itemsize, HipMemcpyKind.DEVICE_TO_DEVICE, 0)
        context_len = self.cache_len + 1
        dflash_gqa_attention_f32_bf16(self.query_rot_buf.ptr, self.key_cache_buf.ptr, self.value_cache_buf.ptr, self.attn_out_buf.ptr, 1, 1, context_len, self.q_heads, self.kv_heads, self.head_dim, threads=128, library=self.dflash_lib)
        mtp_gate_mul_bf16(self.attn_out_buf.ptr, self.gate_buf.ptr, self.gated_buf.ptr, self.q_features, threads=256, library=self.mtp_lib)
        dflash_dense_bf16_to_bf16(self.gated_buf.ptr, self.weights["mtp.layers.0.self_attn.o_proj.weight"].ptr, self.o_out_buf.ptr, 1, self.q_features, self.hidden, threads=128, library=self.dflash_lib)
        mtp_add_rmsnorm_bf16_oneplus(
            self.o_out_buf.ptr,
            self.fc_buf.ptr,
            self.weights["mtp.layers.0.post_attention_layernorm.weight"].ptr,
            self.moe_in_buf.ptr,
            self.residual2_buf.ptr,
            1,
            self.hidden,
            eps=self.eps,
            threads=256,
            library=self.mtp_lib,
        )
        dflash_dense_bf16_to_f32(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.gate.weight"].ptr, self.router_logits_buf.ptr, 1, self.hidden, int(self.config.num_experts), threads=128, library=self.dflash_lib)
        topk_f32_rows_i32(self.router_logits_buf.ptr, self.topk_values_buf.ptr, self.topk_ids_buf.ptr, 1, int(self.config.num_experts), self.top_k, threads=256, library=self.lm_lib)
        mtp_softmax_topk_f32(self.topk_values_buf.ptr, self.routing_buf.ptr, 1, self.top_k, library=self.mtp_lib)
        # Expert-indexed GEMVs read `topk_ids[route]` on-device, so the MoE
        # loop no longer forces a mid-pass router D2H sync. Host-visible
        # ids/values are only read when the caller needs diagnostic metadata.
        self.runtime.memset_async(self.moe_accum_buf.ptr, 0, self.moe_accum_buf.nbytes, 0)
        for route in range(self.top_k):
            dflash_dense_bf16_to_bf16_expert(
                self.moe_in_buf.ptr,
                self.gate_up_base,
                self.topk_ids_buf.ptr,
                self.gate_up_buf.ptr,
                route,
                self.gate_up_expert_bytes // DType.BF16.itemsize,
                1,
                self.hidden,
                2 * self.intermediate,
                threads=128,
                library=self.dflash_lib,
            )
            dflash_silu_mul_bf16(self.gate_up_buf.ptr, self.gate_up_buf.ptr + self.intermediate * DType.BF16.itemsize, self.expert_intermediate_buf.ptr, self.intermediate, threads=256, library=self.dflash_lib)
            dflash_dense_bf16_to_bf16_expert(
                self.expert_intermediate_buf.ptr,
                self.down_base,
                self.topk_ids_buf.ptr,
                self.expert_down_buf.ptr,
                route,
                self.down_expert_bytes // DType.BF16.itemsize,
                1,
                self.intermediate,
                self.hidden,
                threads=128,
                library=self.dflash_lib,
            )
            mtp_accumulate_route_bf16_to_f32(self.expert_down_buf.ptr, self.routing_buf.ptr, self.moe_accum_buf.ptr, self.hidden, route, threads=256, library=self.mtp_lib)
        dflash_dense_bf16_to_f32(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert_gate.weight"].ptr, self.shared_gate_buf.ptr, 1, self.hidden, 1, threads=128, library=self.dflash_lib)
        dflash_dense_bf16_to_bf16(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert.gate_proj.weight"].ptr, self.shared_gate_proj_buf.ptr, 1, self.hidden, self.shared_intermediate, threads=128, library=self.dflash_lib)
        dflash_dense_bf16_to_bf16(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert.up_proj.weight"].ptr, self.shared_up_proj_buf.ptr, 1, self.hidden, self.shared_intermediate, threads=128, library=self.dflash_lib)
        dflash_silu_mul_bf16(self.shared_gate_proj_buf.ptr, self.shared_up_proj_buf.ptr, self.shared_intermediate_buf.ptr, self.shared_intermediate, threads=256, library=self.dflash_lib)
        dflash_dense_bf16_to_bf16(self.shared_intermediate_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert.down_proj.weight"].ptr, self.shared_down_buf.ptr, 1, self.shared_intermediate, self.hidden, threads=128, library=self.dflash_lib)
        mtp_accumulate_sigmoid_gate_bf16_to_f32(self.shared_down_buf.ptr, self.shared_gate_buf.ptr, self.moe_accum_buf.ptr, self.hidden, threads=256, library=self.mtp_lib)
        mtp_finalize_f32_to_bf16(self.moe_accum_buf.ptr, self.moe_out_buf.ptr, self.hidden, threads=256, library=self.mtp_lib)
        mtp_add_rmsnorm_bf16_oneplus(self.moe_out_buf.ptr, self.residual2_buf.ptr, self.weights["mtp.norm.weight"].ptr, self.final_hidden_buf.ptr, self.final_residual_buf.ptr, 1, self.hidden, eps=self.eps, threads=256, library=self.mtp_lib)
        self.cache_len += 1
        self.position = int(position)
        if not need_result:
            self.current = NativeMtpStepResult(token=-1, logit=float("nan"), topk_experts=(), topk_logits=())
            return self.current
        lm_head_fp16_argmax_bf16(
            self.final_hidden_buf.ptr,
            self.weights["lm_head.weight"].ptr,
            self.logits_buf.ptr,
            self.block_values_buf.ptr,
            self.block_indices_buf.ptr,
            self.out_index_buf.ptr,
            self.out_value_buf.ptr,
            self.hidden,
            self.draft_vocab,
            threads=256,
            library=self.lm_lib,
        )
        # Blocking D2H of the argmax pair implies a stream sync; the explicit
        # device_synchronize on top of it was pure host stall (#107 host-time trim).
        copy_device_to_host(host_array_ptr(self.out_index_host), self.out_index_buf, self.out_index_host.nbytes, runtime=self.runtime)
        copy_device_to_host(host_array_ptr(self.out_value_host), self.out_value_buf, self.out_value_host.nbytes, runtime=self.runtime)
        if read_expert_topk:
            copy_device_to_host(host_array_ptr(self.topk_ids_host), self.topk_ids_buf, self.topk_ids_host.nbytes, runtime=self.runtime)
            copy_device_to_host(host_array_ptr(self.topk_values_host), self.topk_values_buf, self.topk_values_host.nbytes, runtime=self.runtime)
            topk_experts = tuple(int(x) for x in self.topk_ids_host.reshape(-1))
            topk_logits = tuple(float(x) for x in self.topk_values_host.reshape(-1))
        else:
            topk_experts = ()
            topk_logits = ()
        self.current = NativeMtpStepResult(
            token=int(self.out_index_host[0]),
            logit=float(self.out_value_host[0]),
            topk_experts=topk_experts,
            topk_logits=topk_logits,
        )
        return self.current


__all__ = ["NativeMtpChainProposer", "NativeMtpStateSnapshot", "NativeMtpStepResult"]
