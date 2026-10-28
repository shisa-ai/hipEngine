"""Native Qwen3.6 MTP proposal runner utilities.

This module is a bring-up bridge, not a final public API.  It keeps MTP weights
and scratch resident across proposal steps so benchmark scripts can exercise the
shared verifier without per-proposal weight reloads or target-hidden D2H copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import os

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_array_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import build_dense_gemv, dense_dual_gemv_out_bf16_wmma
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    build_lm_head,
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
    topk_f32_rows_i32,
    w8a16_lm_head_argmax_rows_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_bf16_expert,
    dflash_dense_bf16_to_bf16_expert_routes,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_head_rmsnorm_rotary_f32,
    dflash_head_rmsnorm_rotary_indexed_key_f32,
    dflash_qkv_proj_bf16_mixed,
    dflash_qkv_proj_bf16_mixed_indexed_v,
    dflash_silu_mul_gate_up_routes_bf16,
    dflash_silu_mul_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative.mtp import (
    build_mtp_speculative,
    mtp_accumulate_route_bf16_to_f32,
    mtp_accumulate_routes_bf16_to_f32,
    mtp_accumulate_sigmoid_gate_bf16_to_f32,
    mtp_add_rmsnorm_bf16_oneplus,
    mtp_finalize_f32_to_bf16,
    mtp_fuse_inputs_f16_bf16,
    mtp_gate_mul_bf16,
    mtp_rmsnorm_bf16_oneplus,
    mtp_router_topk_softmax_f32,
    mtp_softmax_topk_f32,
    mtp_split_q_gate_f32_bf16,
)
from hipengine.loading import TensorInfo, load_tensor_info_to_device, load_weight_index, qwen35_paro_config_from_hf, validate_qwen35_mtp_model
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_paro import normalize_qwen35_weight_name
from hipengine.loading.safetensors import read_tensor_storage_bytes


_DEFAULT_DRAFT_VOCAB_CAP = 65536

# Row budget for the bulk historical-prompt K/V workspace. This bounds the
# scratch allocation, not the prompt: a longer prompt is appended in more
# chunks through the same fixed-size workspace.
_BULK_PROMPT_DEFAULT_ROWS = 16
_BULK_PROMPT_MAX_ROWS = 64


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
    copy_host_array_to_device(buf, contiguous, contiguous.nbytes, runtime=runtime)
    buffers.append(buf)
    return buf


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _route0_accum_init_enabled() -> bool:
    return _env_flag("HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT", True)


def _direct_kv_write_enabled() -> bool:
    return _env_flag("HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE", True)


def _indexed_kv_write_enabled() -> bool:
    return _env_flag("HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE", False)


def _router_topk_fused_enabled() -> bool:
    return _env_flag("HIPENGINE_MTP_PROPOSER_ROUTER_TOPK_FUSED", True)


def _route_batched_expert_enabled() -> bool:
    return _env_flag("HIPENGINE_MTP_PROPOSER_ROUTE_BATCHED_EXPERT", True)


def _shared_gate_up_dual_enabled() -> bool:
    return _env_flag("HIPENGINE_MTP_PROPOSER_SHARED_GATE_UP_DUAL", True)


def _draft_vocab_from_env(vocab: int) -> int:
    if vocab <= 0:
        raise ValueError("vocab must be positive")
    raw = os.environ.get("HIPENGINE_MTP_DRAFT_VOCAB_CAP", str(_DEFAULT_DRAFT_VOCAB_CAP))
    cap = int(raw or 0)
    return cap if 0 < cap < vocab else vocab


def _empty_device(shape: tuple[int, ...], dtype: np.dtype[Any] | type[np.generic], buffers: list[DeviceBuffer], *, runtime: HipRuntime) -> tuple[np.ndarray, DeviceBuffer]:
    host = np.zeros(shape, dtype=dtype)
    buf = malloc(host.nbytes, runtime=runtime)
    buffers.append(buf)
    return host, buf


@dataclass(slots=True)
class _BulkPromptWorkspace:
    """Fixed-capacity row scratch for bulk historical-prompt K/V appends.

    ``rows`` is the capacity this workspace was allocated for. Every buffer is
    ``[rows, features]`` and a chunk fills the prefix it needs, so the footprint
    is set once and does not grow with prompt length or chunk size.
    """

    rows: int
    host: np.ndarray
    tokens_host: np.ndarray
    positions_host: np.ndarray
    slots_host: np.ndarray
    tokens_buf: DeviceBuffer
    positions_buf: DeviceBuffer
    slots_buf: DeviceBuffer
    fused_buf: DeviceBuffer
    fc_buf: DeviceBuffer
    attn_in_buf: DeviceBuffer
    q_proj_buf: DeviceBuffer
    k_proj_buf: DeviceBuffer
    v_proj_buf: DeviceBuffer
    query_buf: DeviceBuffer
    gate_buf: DeviceBuffer
    query_rot_buf: DeviceBuffer
    key_rot_buf: DeviceBuffer


@dataclass(frozen=True, slots=True)
class NativeMtpW8A16Head:
    """Borrowed target-owned full-vocabulary W8A16 proposal scorer."""

    weight_int8_ptr: int
    scale_f32_ptr: int
    vocab_size: int
    threads: int = 256
    owner: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.weight_int8_ptr) <= 0 or int(self.scale_f32_ptr) <= 0:
            raise ValueError("W8A16 scorer pointers must be non-zero")
        if int(self.vocab_size) <= 0:
            raise ValueError("W8A16 scorer vocab_size must be positive")
        if int(self.threads) not in {128, 256, 512}:
            raise ValueError("W8A16 scorer threads must be 128, 256, or 512")

    def validate_live(self) -> None:
        if self.owner is not None and bool(getattr(self.owner, "closed", False)):
            raise RuntimeError("borrowed target W8A16 scorer owner is closed")


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
    # Whether ``final_hidden_buf`` held a hidden produced by an advance when the
    # snapshot was taken. A K/V-only row leaves no hidden behind, and restoring
    # a snapshot must not resurrect a carry that was never computed.
    hidden_valid: bool = True


class NativeMtpChainProposer:
    """Resident native MTP proposal runner for one request.

    The initial target-hidden rows are supplied as device pointers captured from
    the target session.  Selected expert ids are still copied to the host for
    pointer selection; this is explicitly a bring-up limitation.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        max_positions: int,
        max_mtp_tokens: int,
        runtime: HipRuntime | None = None,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
        scoring_head: NativeMtpW8A16Head | None = None,
    ) -> None:
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
        self.compiler_version = compiler_version
        self.require_cached_build = bool(require_cached_build)
        self.allocations: list[DeviceTensorAllocation] = []
        self.buffers: list[DeviceBuffer] = []
        self.closed = False
        self.hidden = int(self.config.hidden_size)
        self.vocab = int(self.config.vocab_size)
        self.scoring_head = scoring_head
        if scoring_head is not None and int(scoring_head.vocab_size) != self.vocab:
            raise ValueError("borrowed W8A16 scorer vocabulary does not match the MTP model")
        # The legacy private F16 scorer may retain a diagnostic prefix cap. A
        # borrowed target scorer always covers the target's complete decision
        # surface and avoids duplicating the target head allocation.
        self.draft_vocab = self.vocab if scoring_head is not None else _draft_vocab_from_env(self.vocab)
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
        weight_names = [
            "embed_tokens.weight",
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
        if scoring_head is None:
            weight_names.insert(1, "lm_head.weight")
        self.weights = {name: self._load_info(name).tensor for name in weight_names}
        self.q_norm_oneplus = self._load_oneplus_bf16_weight("mtp.layers.0.self_attn.q_norm.weight")
        self.k_norm_oneplus = self._load_oneplus_bf16_weight("mtp.layers.0.self_attn.k_norm.weight")
        cos, sin = _rope_tables(self.max_positions, self.rotary_dim, float(self.config.rope_theta))
        self.token_position_host = np.zeros((2,), dtype=np.int64)
        self.token_host = self.token_position_host[:1]
        self.position_host = self.token_position_host[1:].view(np.int32)[:1]
        self.cache_slot_host = self.token_position_host[1:].view(np.int32)[1:2]
        self.token_position_buf = _device_array(self.token_position_host, self.buffers, runtime=self.runtime)
        self.token_buf = DeviceBuffer(self.token_position_buf.ptr, np.dtype(np.int64).itemsize)
        self.position_buf = DeviceBuffer(
            self.token_position_buf.ptr + np.dtype(np.int64).itemsize,
            np.dtype(np.int32).itemsize,
        )
        self.cache_slot_buf = DeviceBuffer(
            self.token_position_buf.ptr + np.dtype(np.int64).itemsize + np.dtype(np.int32).itemsize,
            np.dtype(np.int32).itemsize,
        )
        self.cos_buf = _device_array(cos, self.buffers, runtime=self.runtime)
        self.sin_buf = _device_array(sin, self.buffers, runtime=self.runtime)
        self._allocate_scratch()
        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": self.require_cached_build,
        }
        self.mtp_lib = build_mtp_speculative(**build_kwargs)
        self.dflash_lib = build_dflash_drafter(**build_kwargs)
        self.dense_lib = None
        self.lm_lib = build_lm_head(**build_kwargs)
        self.gate_up_base = self.weights["mtp.layers.0.mlp.experts.gate_up_proj"].ptr
        self.down_base = self.weights["mtp.layers.0.mlp.experts.down_proj"].ptr
        self.gate_up_expert_bytes = (2 * self.intermediate) * self.hidden * DType.BF16.itemsize
        self.down_expert_bytes = self.hidden * self.intermediate * DType.BF16.itemsize
        self.cache_len = 0
        self.position = -1
        self.current = NativeMtpStepResult(token=0, logit=float("nan"), topk_experts=(), topk_logits=())
        self.final_hidden_valid = False
        self.result_valid = False
        # Allocated on the first bulk append and reused; it holds only scratch
        # for rows that have already been launched, so ``reset()`` keeps it.
        self._bulk_workspace: _BulkPromptWorkspace | None = None

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
        _, self.gate_up_routes_buf = _empty_device((self.top_k, 2 * self.intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.expert_intermediate_routes_buf = _empty_device((self.top_k, self.intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.expert_down_routes_buf = _empty_device((self.top_k, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.moe_accum_buf = _empty_device((1, self.hidden), np.float32, self.buffers, runtime=self.runtime)
        _, self.shared_gate_buf = _empty_device((1, 1), np.float32, self.buffers, runtime=self.runtime)
        _, self.shared_gate_proj_buf = _empty_device((1, self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.shared_up_proj_buf = _empty_device((1, self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        self.shared_gate_up_proj_buf = None
        _, self.shared_intermediate_buf = _empty_device((1, self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        _, self.shared_down_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.moe_out_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.final_hidden_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.snapshot_hidden_buf = _empty_device((self.max_mtp_tokens, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.final_residual_buf = _empty_device((1, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, self.logits_buf = _empty_device((1, self.vocab), np.float32, self.buffers, runtime=self.runtime)
        scoring_threads = 256 if self.scoring_head is None else int(self.scoring_head.threads)
        block_count = lm_head_argmax_stage1_blocks(self.vocab, threads=scoring_threads)
        self.block_values_host, self.block_values_buf = _empty_device((block_count,), np.float32, self.buffers, runtime=self.runtime)
        _, self.block_indices_buf = _empty_device((block_count,), np.int64, self.buffers, runtime=self.runtime)
        _, self.w8_block_indices_buf = _empty_device((block_count,), np.int32, self.buffers, runtime=self.runtime)
        self.out_index_host, self.out_index_buf = _empty_device((1,), np.int64, self.buffers, runtime=self.runtime)
        self.w8_out_index_host, self.w8_out_index_buf = _empty_device((1,), np.int32, self.buffers, runtime=self.runtime)
        self.out_value_host, self.out_value_buf = _empty_device((1,), np.float32, self.buffers, runtime=self.runtime)

    def _ensure_shared_gate_up_dual(self) -> DeviceBuffer:
        if self.shared_gate_up_proj_buf is None:
            _, self.shared_gate_up_proj_buf = _empty_device((1, 2 * self.shared_intermediate), np.uint16, self.buffers, runtime=self.runtime)
        if self.dense_lib is None:
            self.dense_lib = build_dense_gemv(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
        return self.shared_gate_up_proj_buf

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
        self.final_hidden_valid = False
        self.result_valid = False

    def _copy_device_to_host_array(self, host_array: np.ndarray, buffer: DeviceBuffer, *, nbytes: int | None = None, stream: int = 0) -> None:
        count = int(buffer.nbytes if nbytes is None else nbytes)
        if count < 0 or count > int(buffer.nbytes):
            raise ValueError("copy size exceeds device buffer")
        if int(stream) == 0:
            copy_device_to_host(host_array_ptr(host_array), buffer, count, runtime=self.runtime)
            return
        self.runtime.memcpy_async(
            host_array_ptr(host_array),
            buffer.ptr,
            count,
            HipMemcpyKind.DEVICE_TO_HOST,
            int(stream),
        )
        self.runtime.stream_synchronize(int(stream))

    def prefill_from_target_hidden_rows(
        self,
        prompt_tokens: Sequence[int],
        *,
        capture_base_ptr: int,
        seed_token: int,
        capture_stride_hidden: int | None = None,
        read_expert_topk: bool = True,
        read_lm_head_value: bool = True,
        stream: int = 0,
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
                read_lm_head_value=read_lm_head_value,
                stream=stream,
            )
        return result

    def advance_with_target_hidden(
        self,
        *,
        input_token: int,
        target_hidden_ptr: int,
        position: int,
        need_result: bool = True,
        kv_write_only: bool = False,
        read_token_id: bool = True,
        read_expert_topk: bool = True,
        read_lm_head_value: bool = True,
        stream: int = 0,
    ) -> NativeMtpStepResult:
        if int(target_hidden_ptr) <= 0:
            raise ValueError("target_hidden_ptr must be non-zero")
        forwarded: dict[str, Any] = {
            "input_token": int(input_token),
            "target_hidden_ptr": int(target_hidden_ptr),
            "position": int(position),
            "need_result": need_result,
            "read_token_id": read_token_id,
            "read_expert_topk": read_expert_topk,
            "read_lm_head_value": read_lm_head_value,
            "stream": stream,
        }
        if kv_write_only:
            forwarded["kv_write_only"] = True
        return self.advance(**forwarded)

    def append_prompt_kv(
        self,
        input_token: int,
        target_hidden_ptr: int,
        position: int,
        *,
        stream: int = 0,
    ) -> None:
        """Append one historical prompt row's K/V without producing a proposal.

        Historical prompt rows carry a target-supplied hidden state, so the MTP
        proposer only owes the cache that row's rotated key and projected value.
        This runs the same input fusion, projection, normalization, rotary, and
        cache write as :meth:`advance` and stops there: no attention, no expert
        block, no proposal head, and no host readback. ``cache_len`` and
        ``position`` advance exactly as a full step advances them.

        The row leaves no usable hidden or proposal behind. The next row must
        supply its own target hidden through this method or
        :meth:`advance_with_target_hidden`, and a result-producing advance must
        run before any proposal surface is read.
        """

        self.advance(
            input_token=int(input_token),
            target_hidden_ptr=int(target_hidden_ptr),
            position=int(position),
            need_result=False,
            kv_write_only=True,
            read_token_id=False,
            read_expert_topk=False,
            read_lm_head_value=False,
            stream=stream,
        )
        return None

    def append_prompt_kv_bulk(
        self,
        rows: Sequence[tuple[int, int, int]],
        *,
        chunk_rows: int | None = None,
        stream: int = 0,
    ) -> int:
        """Append several historical prompt rows' K/V with row-batched launches.

        ``rows`` is a sequence of ``(input_token, target_hidden_ptr, position)``
        triples in prompt order -- the same three arguments :meth:`append_prompt_kv`
        takes one at a time. This runs the same input fusion, projection,
        normalization, rotary, and cache write, with every kernel launched for a
        block of rows instead of one row. No attention, no expert block, no
        proposal head, and no host readback run, exactly as for the serial
        cache-only row.

        The target hidden rows must form one contiguous BF16 slab with the same
        row width as the proposer hidden (``rows[i][1] == rows[0][1] +
        i * hidden * 2``), because the fused input kernel reads the slab
        row-major. Rows that are not contiguous in that layout need the serial
        per-row primitive.

        The row workspace is fixed: ``min(_BULK_PROMPT_MAX_ROWS, max_mtp_tokens)``
        rows are allocated once on first use and reused for every later append,
        so the footprint never tracks prompt length or ``chunk_rows`` and no
        allocation is ever replaced while the proposer is live. At most
        ``chunk_rows`` rows (default ``_BULK_PROMPT_DEFAULT_ROWS``, hard cap
        ``_BULK_PROMPT_MAX_ROWS``) are launched per chunk.

        The request is validated in full before the first launch, so a rejected
        request appends nothing and leaves the previous carry and proposal
        intact. Once the first chunk is launched the previous carry and proposal
        are cleared, because a partially launched chunk may already have written
        cache rows the caller cannot observe: a device failure at any point then
        leaves the already-committed chunks published, no valid hidden or
        proposal behind, and the caller free to resume from ``cache_len``.

        Returns the number of appended rows.
        """

        if self.closed:
            raise RuntimeError("NativeMtpChainProposer is closed")
        if self.scoring_head is not None:
            self.scoring_head.validate_live()
        pending = self._validate_bulk_prompt_rows(rows)
        chunk = _BULK_PROMPT_DEFAULT_ROWS if chunk_rows is None else int(chunk_rows)
        if chunk <= 0 or chunk > _BULK_PROMPT_MAX_ROWS:
            raise ValueError(
                f"chunk_rows must be in 1..{_BULK_PROMPT_MAX_ROWS}; got {chunk}"
            )
        if self.cache_len + len(pending) > self.max_mtp_tokens:
            raise RuntimeError("MTP cache capacity exceeded")
        workspace = self._bulk_prompt_workspace()
        # From here on device work happens. Clear the carry and proposal before
        # the first launch, not after a chunk commits: a chunk that fails part
        # way through may already have written this row's K/V, so the previous
        # hidden and proposal stop describing a state the caller can observe.
        self.final_hidden_valid = False
        self.result_valid = False
        self.current = NativeMtpStepResult(token=-1, logit=float("nan"), topk_experts=(), topk_logits=())
        appended = 0
        for start in range(0, len(pending), chunk):
            block = pending[start : start + chunk]
            count = len(block)
            slot_base = int(self.cache_len)
            tokens = workspace.tokens_host[:count]
            positions = workspace.positions_host[:count]
            slots = workspace.slots_host[:count]
            for index, (token, _hidden_ptr, position) in enumerate(block):
                tokens[index] = token
                positions[index] = position
                slots[index] = slot_base + index
            # One packed upload carries the whole chunk's tokens, positions, and
            # cache slots; the sub-buffers are views into that one allocation.
            self.runtime.memcpy_async(
                workspace.tokens_buf.ptr,
                host_array_ptr(workspace.host),
                workspace.host.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
                int(stream),
            )
            self._append_prompt_kv_chunk(
                workspace,
                count=count,
                slot_base=slot_base,
                target_hidden_ptr=int(block[0][1]),
                stream=stream,
            )
            # Only the cursor is published per chunk; the validity clearing above
            # already covers every failure after the first launch.
            self.cache_len = slot_base + count
            self.position = int(block[-1][2])
            appended += count
        return appended

    def _validate_bulk_prompt_rows(
        self, rows: Sequence[tuple[int, int, int]]
    ) -> tuple[tuple[int, int, int], ...]:
        """Normalize and fully validate a bulk append before any device work."""

        pending: list[tuple[int, int, int]] = []
        for row in rows:
            try:
                token, hidden_ptr, position = row
            except (TypeError, ValueError):
                raise ValueError(
                    "bulk prompt rows must be (input_token, target_hidden_ptr, position) triples"
                ) from None
            pending.append((int(token), int(hidden_ptr), int(position)))
        if not pending:
            raise ValueError("bulk prompt rows must be non-empty")
        stride_bytes = self.hidden * DType.BF16.itemsize
        base_ptr = pending[0][1]
        for index, (_token, hidden_ptr, position) in enumerate(pending):
            if hidden_ptr <= 0:
                raise ValueError(
                    "bulk prompt append requires an explicit target hidden pointer per row"
                )
            if hidden_ptr == self.final_hidden_buf.ptr:
                raise ValueError(
                    "bulk prompt append requires target hidden rows, not the proposer's own final hidden"
                )
            if position < 0 or position >= self.max_positions:
                raise ValueError("position outside MTP RoPE table")
            expected = base_ptr + index * stride_bytes
            if hidden_ptr != expected:
                raise ValueError(
                    "bulk prompt target hidden rows must be one contiguous row-major slab: "
                    f"row {index} is at {hidden_ptr}, expected {expected}; pass a compact "
                    "stride-hidden slab or append the rows with append_prompt_kv"
                )
        return tuple(pending)

    def _bulk_prompt_workspace(self) -> _BulkPromptWorkspace:
        """Return the fixed row workspace, allocating it once for the proposer.

        Capacity is ``min(_BULK_PROMPT_MAX_ROWS, max_mtp_tokens)``: the hard cap,
        lowered to the cache capacity because one append can never stage more
        rows than the cache can hold. That is also why a chunk cannot overflow
        the workspace -- a chunk's row count is bounded by the rows the call may
        append, which the capacity check bounds by ``max_mtp_tokens``.

        The workspace is never replaced and never freed while the proposer is
        live, so no buffer can be released while an earlier chunk's launches are
        still in flight on a caller stream. ``close()`` synchronizes the device
        before freeing the whole buffer list.
        """

        existing = self._bulk_workspace
        if existing is not None:
            return existing
        rows = min(_BULK_PROMPT_MAX_ROWS, int(self.max_mtp_tokens))
        host, packed_buf = _empty_device((2 * rows,), np.int64, self.buffers, runtime=self.runtime)
        position_slot_host = host[rows:].view(np.int32)
        # Tokens are int64 and lead the packed buffer, so both int32 views that
        # follow are naturally aligned.
        _, fused_buf = _empty_device((rows, 2 * self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, fc_buf = _empty_device((rows, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, attn_in_buf = _empty_device((rows, self.hidden), np.uint16, self.buffers, runtime=self.runtime)
        _, q_proj_buf = _empty_device((rows, self.q_proj_features), np.float32, self.buffers, runtime=self.runtime)
        _, k_proj_buf = _empty_device((rows, self.kv_features), np.float32, self.buffers, runtime=self.runtime)
        _, v_proj_buf = _empty_device((rows, self.kv_features), np.uint16, self.buffers, runtime=self.runtime)
        _, query_buf = _empty_device((rows, self.q_features), np.float32, self.buffers, runtime=self.runtime)
        _, gate_buf = _empty_device((rows, self.q_features), np.uint16, self.buffers, runtime=self.runtime)
        _, query_rot_buf = _empty_device((rows, self.q_features), np.float32, self.buffers, runtime=self.runtime)
        _, key_rot_buf = _empty_device((rows, self.kv_features), np.float32, self.buffers, runtime=self.runtime)
        self._bulk_workspace = _BulkPromptWorkspace(
            rows=int(rows),
            host=host,
            tokens_host=host[:rows],
            positions_host=position_slot_host[:rows],
            slots_host=position_slot_host[rows : 2 * rows],
            tokens_buf=DeviceBuffer(packed_buf.ptr, rows * np.dtype(np.int64).itemsize),
            positions_buf=DeviceBuffer(
                packed_buf.ptr + rows * np.dtype(np.int64).itemsize,
                rows * np.dtype(np.int32).itemsize,
            ),
            slots_buf=DeviceBuffer(
                packed_buf.ptr
                + rows * np.dtype(np.int64).itemsize
                + rows * np.dtype(np.int32).itemsize,
                rows * np.dtype(np.int32).itemsize,
            ),
            fused_buf=fused_buf,
            fc_buf=fc_buf,
            attn_in_buf=attn_in_buf,
            q_proj_buf=q_proj_buf,
            k_proj_buf=k_proj_buf,
            v_proj_buf=v_proj_buf,
            query_buf=query_buf,
            gate_buf=gate_buf,
            query_rot_buf=query_rot_buf,
            key_rot_buf=key_rot_buf,
        )
        return self._bulk_workspace

    def _append_prompt_kv_chunk(
        self,
        workspace: _BulkPromptWorkspace,
        *,
        count: int,
        slot_base: int,
        target_hidden_ptr: int,
        stream: int,
    ) -> None:
        """Run the K/V producer prefix for ``count`` rows starting at ``slot_base``.

        Every kernel here writes an independent row: the fused input, RMSNorm,
        and split-q-gate blocks are per-row, the two GEMV kernels reduce over
        input features per output element, and the head rotary kernel is
        per-(row, head). Row ``r`` therefore receives the same accumulation the
        serial one-row launch gives it. Cache destinations are the block's first
        slot because both row strides equal the cache row stride.
        """

        hidden = self.hidden
        indexed_kv_write = _indexed_kv_write_enabled()
        key_cache_dst = self.key_cache_buf.ptr + slot_base * self.kv_features * DType.FP32.itemsize
        value_cache_dst = self.value_cache_buf.ptr + slot_base * self.kv_features * DType.BF16.itemsize
        direct_kv_write = _direct_kv_write_enabled() or indexed_kv_write
        v_out_ptr = value_cache_dst if direct_kv_write else workspace.v_proj_buf.ptr
        key_out_ptr = key_cache_dst if direct_kv_write else workspace.key_rot_buf.ptr
        mtp_fuse_inputs_f16_bf16(
            workspace.tokens_buf.ptr,
            self.weights["embed_tokens.weight"].ptr,
            int(target_hidden_ptr),
            self.weights["mtp.pre_fc_norm_embedding.weight"].ptr,
            self.weights["mtp.pre_fc_norm_hidden.weight"].ptr,
            workspace.fused_buf.ptr,
            count,
            hidden,
            self.vocab,
            eps=self.eps,
            threads=256,
            stream=stream,
            library=self.mtp_lib,
        )
        dflash_dense_bf16_to_bf16(workspace.fused_buf.ptr, self.weights["mtp.fc.weight"].ptr, workspace.fc_buf.ptr, count, 2 * hidden, hidden, threads=128, stream=stream, library=self.dflash_lib)
        mtp_rmsnorm_bf16_oneplus(workspace.fc_buf.ptr, self.weights["mtp.layers.0.input_layernorm.weight"].ptr, workspace.attn_in_buf.ptr, count, hidden, eps=self.eps, threads=256, stream=stream, library=self.mtp_lib)
        if indexed_kv_write:
            dflash_qkv_proj_bf16_mixed_indexed_v(
                workspace.attn_in_buf.ptr,
                self.weights["mtp.layers.0.self_attn.q_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.k_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.v_proj.weight"].ptr,
                workspace.q_proj_buf.ptr,
                workspace.k_proj_buf.ptr,
                self.value_cache_buf.ptr,
                workspace.slots_buf.ptr,
                self.max_mtp_tokens,
                count,
                hidden,
                self.q_proj_features,
                self.kv_features,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
        else:
            dflash_qkv_proj_bf16_mixed(
                workspace.attn_in_buf.ptr,
                self.weights["mtp.layers.0.self_attn.q_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.k_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.v_proj.weight"].ptr,
                workspace.q_proj_buf.ptr,
                workspace.k_proj_buf.ptr,
                v_out_ptr,
                count,
                hidden,
                self.q_proj_features,
                self.kv_features,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
        mtp_split_q_gate_f32_bf16(workspace.q_proj_buf.ptr, workspace.query_buf.ptr, workspace.gate_buf.ptr, count, self.q_heads, self.head_dim, threads=256, stream=stream, library=self.mtp_lib)
        if indexed_kv_write:
            dflash_head_rmsnorm_rotary_indexed_key_f32(
                workspace.query_buf.ptr,
                workspace.k_proj_buf.ptr,
                self.q_norm_oneplus.ptr,
                self.k_norm_oneplus.ptr,
                self.cos_buf.ptr,
                self.sin_buf.ptr,
                workspace.positions_buf.ptr,
                workspace.positions_buf.ptr,
                workspace.query_rot_buf.ptr,
                self.key_cache_buf.ptr,
                workspace.slots_buf.ptr,
                self.max_mtp_tokens,
                1,
                count,
                count,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                self.rotary_dim,
                self.max_positions,
                eps=self.eps,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
        else:
            dflash_head_rmsnorm_rotary_f32(
                workspace.query_buf.ptr,
                workspace.k_proj_buf.ptr,
                self.q_norm_oneplus.ptr,
                self.k_norm_oneplus.ptr,
                self.cos_buf.ptr,
                self.sin_buf.ptr,
                workspace.positions_buf.ptr,
                workspace.positions_buf.ptr,
                workspace.query_rot_buf.ptr,
                key_out_ptr,
                1,
                count,
                count,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                self.rotary_dim,
                self.max_positions,
                eps=self.eps,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
        if not direct_kv_write:
            # Both workspace rows and cache rows are contiguous, so one copy
            # covers the whole chunk.
            self.runtime.memcpy_async(
                key_cache_dst,
                workspace.key_rot_buf.ptr,
                count * self.kv_features * DType.FP32.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )
            self.runtime.memcpy_async(
                value_cache_dst,
                workspace.v_proj_buf.ptr,
                count * self.kv_features * DType.BF16.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )

    def advance_with_previous_hidden(
        self,
        *,
        input_token: int,
        position: int,
        need_result: bool = True,
        read_token_id: bool = True,
        read_expert_topk: bool = True,
        read_lm_head_value: bool = True,
        stream: int = 0,
    ) -> NativeMtpStepResult:
        if not self.final_hidden_valid:
            raise RuntimeError(
                "proposer final hidden is not valid; run a hidden-producing advance first"
            )
        return self.advance_with_target_hidden(
            input_token=int(input_token),
            target_hidden_ptr=self.final_hidden_buf.ptr,
            position=int(position),
            need_result=need_result,
            read_token_id=read_token_id,
            read_expert_topk=read_expert_topk,
            read_lm_head_value=read_lm_head_value,
            stream=stream,
        )

    def device_candidate_token_ids(self) -> Tensor:
        """Return the stable packed-PARO top-1 scalar without host readback."""

        if not self.result_valid:
            raise RuntimeError(
                "device candidate token requires a result-producing advance"
            )
        if self.scoring_head is None:
            raise NotImplementedError("device candidate descriptor requires W8A16 scoring")
        return Tensor.from_handle(
            self.w8_out_index_buf.ptr,
            (1,),
            DType.INT32,
            Device("hip", 0),
        )

    def read_final_hidden_bf16(self, *, stream: int = 0) -> np.ndarray:
        """Copy the current one-row BF16 MTP hidden for correctness diagnostics."""

        if not self.final_hidden_valid:
            raise RuntimeError(
                "proposer final hidden is not valid; run a hidden-producing advance first"
            )
        host = np.empty((1, self.hidden), dtype=np.uint16)
        self._copy_device_to_host_array(
            host,
            self.final_hidden_buf,
            nbytes=host.nbytes,
            stream=stream,
        )
        return host

    def save_state(self, slot: int, *, stream: int = 0) -> NativeMtpStateSnapshot:
        if slot < 0 or slot >= self.max_mtp_tokens:
            raise ValueError("snapshot slot outside capacity")
        self.runtime.memcpy_async(
            self.snapshot_hidden_buf.ptr + int(slot) * self.hidden * DType.BF16.itemsize,
            self.final_hidden_buf.ptr,
            self.hidden * DType.BF16.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )
        return NativeMtpStateSnapshot(
            slot=int(slot),
            cache_len=int(self.cache_len),
            position=int(self.position),
            current=self.current,
            hidden_valid=bool(self.final_hidden_valid),
        )

    def restore_state(self, snapshot: NativeMtpStateSnapshot, *, stream: int = 0) -> None:
        if snapshot.slot < 0 or snapshot.slot >= self.max_mtp_tokens:
            raise ValueError("snapshot slot outside capacity")
        self.runtime.memcpy_async(
            self.final_hidden_buf.ptr,
            self.snapshot_hidden_buf.ptr + int(snapshot.slot) * self.hidden * DType.BF16.itemsize,
            self.hidden * DType.BF16.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )
        self.cache_len = int(snapshot.cache_len)
        self.position = int(snapshot.position)
        self.current = snapshot.current
        self.final_hidden_valid = bool(snapshot.hidden_valid)
        # The snapshot carries the hidden row and the host readback, not the
        # device head buffers (top-1 index/value, block maxima, logits). Those
        # still hold the discarded advance's scores, so every score or candidate
        # read needs a new result-producing advance at the restored state.
        self.result_valid = False

    def vocab_topk(self, *, k: int = 8, stream: int = 0) -> tuple[list[int], list[float]]:
        """Exact top-k tokens+logits over the (capped) draft vocab.

        Runs the topk kernel over ``logits_buf`` (already populated by the
        advance's lm-head pass) and reads ids+values in one small D2H pair.
        Used for the gated branching tree (#99 -> persistent path).
        """

        if not self.result_valid:
            raise RuntimeError("vocab_topk requires a result-producing advance")
        if k <= 0 or k > 8:
            raise ValueError("vocab_topk supports 1..8")
        if self.scoring_head is not None:
            raise NotImplementedError("borrowed W8A16 scorer currently exposes top-1 only")
        if self._vocab_topk_host is None:
            ids, ids_buf = _empty_device((1, 8), np.int32, self.buffers, runtime=self.runtime)
            vals, vals_buf = _empty_device((1, 8), np.float32, self.buffers, runtime=self.runtime)
            self._vocab_topk_host = (ids, ids_buf, vals, vals_buf)
        ids, ids_buf, vals, vals_buf = self._vocab_topk_host
        topk_f32_rows_i32(self.logits_buf.ptr, vals_buf.ptr, ids_buf.ptr, 1, self.draft_vocab, 8, threads=256, stream=stream, library=self.lm_lib)
        self._copy_device_to_host_array(ids, ids_buf, nbytes=ids.nbytes, stream=stream)
        self._copy_device_to_host_array(vals, vals_buf, nbytes=vals.nbytes, stream=stream)
        return [int(x) for x in ids.reshape(-1)[:k]], [float(x) for x in vals.reshape(-1)[:k]]

    def top1_prob_proxy(self, *, k: int = 8, stream: int = 0) -> float:
        """Softmax weight of the current top-1 over the top-k lm-head block maxima.

        Each lm-head stage-1 block covers a distinct vocab slice, so the top-k
        block maxima are a superset proxy for the vocab top-k logits — the same
        DFlash-style confidence proxy #100 used for per-position p-min draft
        truncation, without a vocab-sized D2H.
        """

        if not self.result_valid:
            raise RuntimeError("top1_prob_proxy requires a result-producing advance")
        scoring_threads = 256 if self.scoring_head is None else int(self.scoring_head.threads)
        valid = lm_head_argmax_stage1_blocks(self.draft_vocab, threads=scoring_threads)
        self._copy_device_to_host_array(
            self.block_values_host,
            self.block_values_buf,
            nbytes=self.block_values_host.nbytes,
            stream=stream,
        )
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
        kv_write_only: bool = False,
        read_token_id: bool = True,
        read_expert_topk: bool = True,
        read_lm_head_value: bool = True,
        stream: int = 0,
    ) -> NativeMtpStepResult:
        """Advance one MTP step.

        ``need_result=False`` is for verifier repair/update advances where only
        the updated hidden/KV state is consumed by the following step. It skips
        the draft lm-head/argmax and host reads, so ``current`` is intentionally
        marked invalid until a later result-producing advance runs.

        ``kv_write_only=True`` is for historical prompt rows whose hidden comes
        from the target session. It appends this row's rotated key and projected
        value to the MTP cache and publishes the cursor, then returns before
        attention, the expert block, and the head. It requires an explicit
        target hidden, cannot be combined with ``need_result=True``, and leaves
        no valid hidden carry or proposal behind.
        """

        if self.closed:
            raise RuntimeError("NativeMtpChainProposer is closed")
        if self.scoring_head is not None:
            self.scoring_head.validate_live()
        if self.cache_len >= self.max_mtp_tokens:
            raise RuntimeError("MTP cache capacity exceeded")
        if position < 0 or position >= self.max_positions:
            raise ValueError("position outside MTP RoPE table")
        if kv_write_only:
            # A K/V-only row exists to append its own cache entry and nothing
            # else. Its hidden belongs to the target, so it can neither produce
            # a proposal nor leave a hidden behind for a later row to carry.
            if need_result:
                raise ValueError("kv_write_only advance cannot produce a proposal result")
            if int(target_hidden_ptr) <= 0:
                raise ValueError(
                    "kv_write_only advance requires an explicit target hidden pointer"
                )
            if int(target_hidden_ptr) == self.final_hidden_buf.ptr:
                raise ValueError(
                    "kv_write_only advance requires a target hidden, not the proposer's own final hidden"
                )
        self.token_host[0] = int(input_token)
        self.position_host[0] = int(position)
        self.cache_slot_host[0] = int(self.cache_len)
        indexed_kv_write = _indexed_kv_write_enabled()
        self.runtime.memcpy_async(
            self.token_position_buf.ptr,
            host_array_ptr(self.token_position_host),
            self.token_position_host.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
            int(stream),
        )
        key_cache_dst = self.key_cache_buf.ptr + self.cache_len * self.kv_features * DType.FP32.itemsize
        value_cache_dst = self.value_cache_buf.ptr + self.cache_len * self.kv_features * DType.BF16.itemsize
        direct_kv_write = _direct_kv_write_enabled() or indexed_kv_write
        v_out_ptr = value_cache_dst if direct_kv_write else self.v_proj_buf.ptr
        key_out_ptr = key_cache_dst if direct_kv_write else self.key_rot_buf.ptr
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
            stream=stream,
            library=self.mtp_lib,
        )
        dflash_dense_bf16_to_bf16(self.fused_buf.ptr, self.weights["mtp.fc.weight"].ptr, self.fc_buf.ptr, 1, 2 * self.hidden, self.hidden, threads=128, stream=stream, library=self.dflash_lib)
        mtp_rmsnorm_bf16_oneplus(self.fc_buf.ptr, self.weights["mtp.layers.0.input_layernorm.weight"].ptr, self.attn_in_buf.ptr, 1, self.hidden, eps=self.eps, threads=256, stream=stream, library=self.mtp_lib)
        if indexed_kv_write:
            dflash_qkv_proj_bf16_mixed_indexed_v(
                self.attn_in_buf.ptr,
                self.weights["mtp.layers.0.self_attn.q_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.k_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.v_proj.weight"].ptr,
                self.q_proj_buf.ptr,
                self.k_proj_buf.ptr,
                self.value_cache_buf.ptr,
                self.cache_slot_buf.ptr,
                self.max_mtp_tokens,
                1,
                self.hidden,
                self.q_proj_features,
                self.kv_features,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
        else:
            dflash_qkv_proj_bf16_mixed(
                self.attn_in_buf.ptr,
                self.weights["mtp.layers.0.self_attn.q_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.k_proj.weight"].ptr,
                self.weights["mtp.layers.0.self_attn.v_proj.weight"].ptr,
                self.q_proj_buf.ptr,
                self.k_proj_buf.ptr,
                v_out_ptr,
                1,
                self.hidden,
                self.q_proj_features,
                self.kv_features,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
        mtp_split_q_gate_f32_bf16(self.q_proj_buf.ptr, self.query_buf.ptr, self.gate_buf.ptr, 1, self.q_heads, self.head_dim, threads=256, stream=stream, library=self.mtp_lib)
        if indexed_kv_write:
            dflash_head_rmsnorm_rotary_indexed_key_f32(
                self.query_buf.ptr,
                self.k_proj_buf.ptr,
                self.q_norm_oneplus.ptr,
                self.k_norm_oneplus.ptr,
                self.cos_buf.ptr,
                self.sin_buf.ptr,
                self.position_buf.ptr,
                self.position_buf.ptr,
                self.query_rot_buf.ptr,
                self.key_cache_buf.ptr,
                self.cache_slot_buf.ptr,
                self.max_mtp_tokens,
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
                stream=stream,
                library=self.dflash_lib,
            )
        else:
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
                key_out_ptr,
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
                stream=stream,
                library=self.dflash_lib,
            )
        if not direct_kv_write:
            self.runtime.memcpy_async(key_cache_dst, self.key_rot_buf.ptr, self.kv_features * DType.FP32.itemsize, HipMemcpyKind.DEVICE_TO_DEVICE, int(stream))
            self.runtime.memcpy_async(value_cache_dst, self.v_proj_buf.ptr, self.kv_features * DType.BF16.itemsize, HipMemcpyKind.DEVICE_TO_DEVICE, int(stream))
        context_len = self.cache_len + 1
        if kv_write_only:
            # The remaining attention/expert/head work reads this cache but
            # does not write it. Historical rows use target-provided hidden
            # states, so their unused predictor hidden can be discarded.
            self.cache_len = context_len
            self.position = int(position)
            self.final_hidden_valid = False
            self.result_valid = False
            self.current = NativeMtpStepResult(token=-1, logit=float("nan"), topk_experts=(), topk_logits=())
            return self.current
        dflash_gqa_attention_f32_bf16(self.query_rot_buf.ptr, self.key_cache_buf.ptr, self.value_cache_buf.ptr, self.attn_out_buf.ptr, 1, 1, context_len, self.q_heads, self.kv_heads, self.head_dim, threads=128, stream=stream, library=self.dflash_lib)
        mtp_gate_mul_bf16(self.attn_out_buf.ptr, self.gate_buf.ptr, self.gated_buf.ptr, self.q_features, threads=256, stream=stream, library=self.mtp_lib)
        dflash_dense_bf16_to_bf16(self.gated_buf.ptr, self.weights["mtp.layers.0.self_attn.o_proj.weight"].ptr, self.o_out_buf.ptr, 1, self.q_features, self.hidden, threads=128, stream=stream, library=self.dflash_lib)
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
            stream=stream,
            library=self.mtp_lib,
        )
        dflash_dense_bf16_to_f32(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.gate.weight"].ptr, self.router_logits_buf.ptr, 1, self.hidden, int(self.config.num_experts), threads=128, stream=stream, library=self.dflash_lib)
        if _router_topk_fused_enabled() and int(self.config.num_experts) == 256 and int(self.top_k) == 8:
            mtp_router_topk_softmax_f32(
                self.router_logits_buf.ptr,
                self.topk_values_buf.ptr,
                self.topk_ids_buf.ptr,
                self.routing_buf.ptr,
                int(self.config.num_experts),
                self.top_k,
                stream=stream,
                library=self.mtp_lib,
            )
        else:
            topk_f32_rows_i32(self.router_logits_buf.ptr, self.topk_values_buf.ptr, self.topk_ids_buf.ptr, 1, int(self.config.num_experts), self.top_k, threads=256, stream=stream, library=self.lm_lib)
            mtp_softmax_topk_f32(self.topk_values_buf.ptr, self.routing_buf.ptr, 1, self.top_k, stream=stream, library=self.mtp_lib)
        # Expert-indexed GEMVs read `topk_ids[route]` on-device, so the MoE
        # loop no longer forces a mid-pass router D2H sync. Host-visible
        # ids/values are only read when the caller needs diagnostic metadata.
        if _route_batched_expert_enabled():
            dflash_dense_bf16_to_bf16_expert_routes(
                self.moe_in_buf.ptr,
                self.gate_up_base,
                self.topk_ids_buf.ptr,
                self.gate_up_routes_buf.ptr,
                self.top_k,
                0,
                self.gate_up_expert_bytes // DType.BF16.itemsize,
                1,
                self.hidden,
                2 * self.intermediate,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
            dflash_silu_mul_gate_up_routes_bf16(
                self.gate_up_routes_buf.ptr,
                self.expert_intermediate_routes_buf.ptr,
                self.top_k,
                self.intermediate,
                threads=256,
                stream=stream,
                library=self.dflash_lib,
            )
            dflash_dense_bf16_to_bf16_expert_routes(
                self.expert_intermediate_routes_buf.ptr,
                self.down_base,
                self.topk_ids_buf.ptr,
                self.expert_down_routes_buf.ptr,
                self.top_k,
                self.intermediate,
                self.down_expert_bytes // DType.BF16.itemsize,
                1,
                self.intermediate,
                self.hidden,
                threads=128,
                stream=stream,
                library=self.dflash_lib,
            )
            mtp_accumulate_routes_bf16_to_f32(
                self.expert_down_routes_buf.ptr,
                self.routing_buf.ptr,
                self.moe_accum_buf.ptr,
                self.top_k,
                self.hidden,
                threads=256,
                stream=stream,
                library=self.mtp_lib,
            )
        else:
            route0_init = _route0_accum_init_enabled()
            if not route0_init:
                self.runtime.memset_async(self.moe_accum_buf.ptr, 0, self.moe_accum_buf.nbytes, int(stream))
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
                    stream=stream,
                    library=self.dflash_lib,
                )
                dflash_silu_mul_bf16(self.gate_up_buf.ptr, self.gate_up_buf.ptr + self.intermediate * DType.BF16.itemsize, self.expert_intermediate_buf.ptr, self.intermediate, threads=256, stream=stream, library=self.dflash_lib)
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
                    stream=stream,
                    library=self.dflash_lib,
                )
                mtp_accumulate_route_bf16_to_f32(
                    self.expert_down_buf.ptr,
                    self.routing_buf.ptr,
                    self.moe_accum_buf.ptr,
                    self.hidden,
                    route,
                    reset_output=route0_init and route == 0,
                    threads=256,
                    stream=stream,
                    library=self.mtp_lib,
                )
        dflash_dense_bf16_to_f32(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert_gate.weight"].ptr, self.shared_gate_buf.ptr, 1, self.hidden, 1, threads=128, stream=stream, library=self.dflash_lib)
        if _shared_gate_up_dual_enabled():
            shared_gate_up_proj_buf = self._ensure_shared_gate_up_dual()
            dense_dual_gemv_out_bf16_wmma(
                self.moe_in_buf.ptr,
                self.weights["mtp.layers.0.mlp.shared_expert.gate_proj.weight"].ptr,
                self.weights["mtp.layers.0.mlp.shared_expert.up_proj.weight"].ptr,
                shared_gate_up_proj_buf.ptr,
                1,
                self.hidden,
                self.shared_intermediate,
                self.shared_intermediate,
                library=self.dense_lib,
                runtime=self.runtime,
                stream=stream,
            )
            shared_gate_proj_ptr = shared_gate_up_proj_buf.ptr
            shared_up_proj_ptr = shared_gate_up_proj_buf.ptr + self.shared_intermediate * DType.BF16.itemsize
        else:
            dflash_dense_bf16_to_bf16(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert.gate_proj.weight"].ptr, self.shared_gate_proj_buf.ptr, 1, self.hidden, self.shared_intermediate, threads=128, stream=stream, library=self.dflash_lib)
            dflash_dense_bf16_to_bf16(self.moe_in_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert.up_proj.weight"].ptr, self.shared_up_proj_buf.ptr, 1, self.hidden, self.shared_intermediate, threads=128, stream=stream, library=self.dflash_lib)
            shared_gate_proj_ptr = self.shared_gate_proj_buf.ptr
            shared_up_proj_ptr = self.shared_up_proj_buf.ptr
        dflash_silu_mul_bf16(shared_gate_proj_ptr, shared_up_proj_ptr, self.shared_intermediate_buf.ptr, self.shared_intermediate, threads=256, stream=stream, library=self.dflash_lib)
        dflash_dense_bf16_to_bf16(self.shared_intermediate_buf.ptr, self.weights["mtp.layers.0.mlp.shared_expert.down_proj.weight"].ptr, self.shared_down_buf.ptr, 1, self.shared_intermediate, self.hidden, threads=128, stream=stream, library=self.dflash_lib)
        mtp_accumulate_sigmoid_gate_bf16_to_f32(self.shared_down_buf.ptr, self.shared_gate_buf.ptr, self.moe_accum_buf.ptr, self.hidden, threads=256, stream=stream, library=self.mtp_lib)
        mtp_finalize_f32_to_bf16(self.moe_accum_buf.ptr, self.moe_out_buf.ptr, self.hidden, threads=256, stream=stream, library=self.mtp_lib)
        mtp_add_rmsnorm_bf16_oneplus(self.moe_out_buf.ptr, self.residual2_buf.ptr, self.weights["mtp.norm.weight"].ptr, self.final_hidden_buf.ptr, self.final_residual_buf.ptr, 1, self.hidden, eps=self.eps, threads=256, stream=stream, library=self.mtp_lib)
        self.cache_len += 1
        self.position = int(position)
        self.final_hidden_valid = True
        if not need_result:
            self.result_valid = False
            self.current = NativeMtpStepResult(token=-1, logit=float("nan"), topk_experts=(), topk_logits=())
            return self.current
        if self.scoring_head is None:
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
                stream=stream,
                library=self.lm_lib,
            )
            index_host = self.out_index_host
            index_buf = self.out_index_buf
        else:
            w8a16_lm_head_argmax_rows_bf16(
                self.final_hidden_buf.ptr,
                int(self.scoring_head.weight_int8_ptr),
                int(self.scoring_head.scale_f32_ptr),
                self.block_values_buf.ptr,
                self.w8_block_indices_buf.ptr,
                self.w8_out_index_buf.ptr,
                self.out_value_buf.ptr,
                1,
                self.hidden,
                self.vocab,
                threads=int(self.scoring_head.threads),
                stream=stream,
                library=self.lm_lib,
                runtime=self.runtime,
            )
            index_host = self.w8_out_index_host
            index_buf = self.w8_out_index_buf
        # Blocking D2H of the argmax pair implies a stream sync; the explicit
        # device_synchronize on top of it was pure host stall (#107 host-time trim).
        if read_token_id:
            self._copy_device_to_host_array(
                index_host,
                index_buf,
                nbytes=index_host.nbytes,
                stream=stream,
            )
        else:
            index_host[0] = -1
        logit = float("nan")
        if read_lm_head_value:
            self._copy_device_to_host_array(self.out_value_host, self.out_value_buf, nbytes=self.out_value_host.nbytes, stream=stream)
            logit = float(self.out_value_host[0])
        if read_expert_topk:
            self._copy_device_to_host_array(self.topk_ids_host, self.topk_ids_buf, nbytes=self.topk_ids_host.nbytes, stream=stream)
            self._copy_device_to_host_array(self.topk_values_host, self.topk_values_buf, nbytes=self.topk_values_host.nbytes, stream=stream)
            topk_experts = tuple(int(x) for x in self.topk_ids_host.reshape(-1))
            topk_logits = tuple(float(x) for x in self.topk_values_host.reshape(-1))
        else:
            topk_experts = ()
            topk_logits = ()
        self.result_valid = True
        self.current = NativeMtpStepResult(
            token=int(index_host[0]),
            logit=logit,
            topk_experts=topk_experts,
            topk_logits=topk_logits,
        )
        return self.current


__all__ = ["NativeMtpChainProposer", "NativeMtpStateSnapshot", "NativeMtpStepResult", "NativeMtpW8A16Head"]
