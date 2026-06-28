"""Resident GGUF MTP NextN draft runner.

This is the production-shaped companion to the correctness-first NumPy wrapper
in ``hip_gfx1100.speculative.mtp_nextn``.  It keeps the one-layer NextN draft
chain on device across depths and returns only the small top-k token IDs needed
by the Python acceptance harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.convert.cast import (
    bf16_to_f32,
    build_cast,
    f32_to_bf16,
)
from hipengine.kernels.hip_gfx1100.convert.gather import (
    build_gather,
    gather_f32_rows_by_i32id,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    build_paro_combine,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    build_lm_head,
    topk_f32_rows_i32,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    build_qwen35_router,
    qwen35_router_select,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_gemv_f32_f32_out,
    gguf_q5_k_selected_gemv_bf16_bf16_out,
    gguf_q8_0_gemv_f32_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
    gguf_q6_k_pack8_gemv_decode_bf16_f32_out,
)
from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
    _cached_upload,
    build_mtp_nextn,
    mtp_add_f32,
    mtp_dense_attn_f32,
    mtp_linear_f32,
    mtp_rmsnorm_f32,
    mtp_rope_f32,
    mtp_scale_f32,
    mtp_sigmoid_gate_mul_f32,
    mtp_sigmoid_row_scale_from_logits_f32,
    mtp_silu_mul_f32,
    mtp_split_q_gate_f32,
)
from hipengine.quant.gguf import GGMLQuantizationType


def _bf16_host_to_f32(array: np.ndarray) -> np.ndarray:
    raw = np.asarray(array)
    if raw.dtype == np.float32:
        return np.ascontiguousarray(raw, dtype=np.float32)
    if raw.dtype not in (np.uint16, np.int16):
        return np.ascontiguousarray(raw.astype(np.float32), dtype=np.float32)
    out = np.empty(raw.shape, dtype=np.float32)
    out.view(np.uint32)[:] = raw.astype(np.uint32) << 16
    return np.ascontiguousarray(out, dtype=np.float32)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def apply_moe_down_combine(
    *,
    gate_bf16_ptr: int,
    up_bf16_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    shared_out_ptr: int,
    shared_gate_logit_ptr: int,
    residual_ptr: int,
    down_exps_ptr: int,
    inter_bf16_ptr: int,
    down_out_bf16_ptr: int,
    attended_bf16_ptr: int,
    shared_bf16_ptr: int,
    ffn_out_bf16_ptr: int,
    ffn_out_f32_ptr: int,
    top_k: int,
    inter: int,
    hidden: int,
    num_experts: int,
    silu_lib,
    k_lib,
    combine_lib,
    cast_lib,
    runtime: HipRuntime | None = None,
) -> None:
    """Device-resident NextN selected-MoE down + combine (no host readback).

    Mirrors the verifier's proven sequence so the draft path matches its bf16
    precision: one ``silu_mul`` over all ``top_k`` experts, one selected-down
    GEMV that reads the device ``selected`` indices (raw Q5_K, selected order:
    ``out[k] = down[selected[k]] @ inter[k]``), and one combine that folds the
    routing-weighted expert sum + sigmoid-gated shared expert + residual.  The
    expert indices and routing weights never leave the GPU.

    ``residual`` / ``shared_out`` are the draft's f32 buffers; they are cast to
    bf16 to feed the bf16 combine kernel, whose bf16 output is cast back to f32
    in ``ffn_out_f32_ptr`` for the downstream RMSNorm.
    """
    runtime = runtime or get_hip_runtime()
    # SiLU(gate) * up over all top_k experts at once (bf16 in/out).
    silu_mul_separate_out_bf16(
        gate_bf16_ptr, up_bf16_ptr, inter_bf16_ptr, top_k, inter,
        library=silu_lib, runtime=runtime,
    )
    # Selected-down GEMV: each expert consumes its own intermediate row
    # (x_rows == rows == top_k  =>  lanes_per_x_row == 1  =>  x_row == row).
    gguf_q5_k_selected_gemv_bf16_bf16_out(
        inter_bf16_ptr, selected_ptr, down_exps_ptr, down_out_bf16_ptr,
        top_k, top_k, num_experts, inter, hidden,
        library=k_lib, runtime=runtime,
    )
    # Cast the f32 residual + shared-expert output to bf16 for the combine.
    f32_to_bf16(residual_ptr, attended_bf16_ptr, hidden, library=cast_lib, runtime=runtime)
    f32_to_bf16(shared_out_ptr, shared_bf16_ptr, hidden, library=cast_lib, runtime=runtime)
    # routing-weighted expert sum + sigmoid(gate)*shared + residual, in one launch.
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
        down_out_bf16_ptr, routing_ptr, shared_bf16_ptr, shared_gate_logit_ptr,
        attended_bf16_ptr, ffn_out_bf16_ptr, top_k, hidden,
        library=combine_lib, runtime=runtime,
    )
    bf16_to_f32(ffn_out_bf16_ptr, ffn_out_f32_ptr, hidden, library=cast_lib, runtime=runtime)


@dataclass
class Qwen35GGUFResidentMTPDraftRunner:
    """Device-resident chain runner for the real Qwen3.6 GGUF NextN block."""

    weights: dict[str, tuple[np.ndarray, object, object]]
    token_embd_f32: np.ndarray
    runtime: HipRuntime | None = None
    vocab_cap: int = 32768
    num_heads: int = 16
    num_kv_heads: int = 2
    experts_used: int = 8
    eps: float = 1e-6
    _buffers: list[DeviceBuffer] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime or get_hip_runtime()
        self.hidden_size = int(self.token_embd_f32.shape[1])
        self.qk_head_dim = int(np.asarray(self._get("blk.40.attn_q_norm.weight")).shape[0])
        self.value_head_dim = self.qk_head_dim
        self.inter_dim = int(self._get("blk.40.ffn_gate_exps.weight").shape[1])
        self.vocab = min(int(self.vocab_cap), int(self._get("output.weight").shape[0]))
        if self.vocab <= 0 or self.vocab % 8 != 0:
            raise ValueError("resident GGUF MTP vocab cap must be positive and divisible by 8")
        self._check_real_model_qtypes()
        self._mtp_lib = build_mtp_nextn(load=True)
        self._k_lib = build_gguf_k_gemv(load=True)
        self._q4_lib = build_gguf_q4_k_gemv(load=True)
        self._q6_pack8_lib = build_gguf_q6_k_pack8_gemv(load=True)
        self._cast_lib = build_cast(load=True)
        self._router_lib = build_qwen35_router(load=True)
        self._lm_head_lib = build_lm_head(load=True)
        self._silu_lib = build_paro_silu(load=True)
        self._combine_lib = build_paro_combine(load=True)
        self._gather_lib = build_gather(load=True)
        self._device_moe_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE", True)
        self._device_chain_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN", False)
        # Max draft depth the precomputed-rope / topk-accumulator buffers cover.
        self._draft_chain_cap = 16
        self._embed_table_f32: DeviceBuffer | None = None
        self._upload_weights()
        self._allocate_buffers()

    def _get(self, name: str) -> np.ndarray:
        return self.weights[name][0]

    def _qt(self, name: str) -> GGMLQuantizationType:
        return GGMLQuantizationType(self.weights[name][1])

    def _check_real_model_qtypes(self) -> None:
        expected = {
            "blk.40.nextn.eh_proj.weight": GGMLQuantizationType.Q8_0,
            "blk.40.attn_q.weight": GGMLQuantizationType.Q8_0,
            "blk.40.attn_k.weight": GGMLQuantizationType.Q8_0,
            "blk.40.attn_v.weight": GGMLQuantizationType.Q8_0,
            "blk.40.attn_output.weight": GGMLQuantizationType.Q8_0,
            "blk.40.ffn_gate_exps.weight": GGMLQuantizationType.Q4_K,
            "blk.40.ffn_up_exps.weight": GGMLQuantizationType.Q4_K,
            "blk.40.ffn_down_exps.weight": GGMLQuantizationType.Q5_K,
            "blk.40.ffn_gate_shexp.weight": GGMLQuantizationType.Q8_0,
            "blk.40.ffn_up_shexp.weight": GGMLQuantizationType.Q8_0,
            "blk.40.ffn_down_shexp.weight": GGMLQuantizationType.Q8_0,
            "output.weight": GGMLQuantizationType.Q6_K,
        }
        for name, qtype in expected.items():
            if self._qt(name) != qtype:
                raise NotImplementedError(f"resident GGUF MTP draft expects {name}={qtype.name}")

    def _upload(self, name: str, data: np.ndarray | None = None) -> DeviceBuffer:
        array = self._get(name) if data is None else data
        return _cached_upload(f"resident_mtp:{name}", np.ascontiguousarray(array), runtime=self.runtime)

    def _upload_weights(self) -> None:
        self.eh_proj = self._upload("blk.40.nextn.eh_proj.weight")
        self.hnorm = self._upload("blk.40.nextn.hnorm.weight")
        self.enorm = self._upload("blk.40.nextn.enorm.weight")
        self.attn_norm = self._upload("blk.40.attn_norm.weight")
        self.wq = self._upload("blk.40.attn_q.weight")
        self.wk = self._upload("blk.40.attn_k.weight")
        self.wv = self._upload("blk.40.attn_v.weight")
        self.wo = self._upload("blk.40.attn_output.weight")
        self.q_norm = self._upload("blk.40.attn_q_norm.weight")
        self.k_norm = self._upload("blk.40.attn_k_norm.weight")
        self.post_norm_weight = self._upload("blk.40.post_attention_norm.weight")
        self.router_weight_f32 = self._upload(
            "blk.40.ffn_gate_inp.weight:f32",
            _bf16_host_to_f32(self._get("blk.40.ffn_gate_inp.weight")),
        )
        self.shared_gate_vec_f32 = self._upload(
            "blk.40.ffn_gate_inp_shexp.weight:f32",
            _bf16_host_to_f32(self._get("blk.40.ffn_gate_inp_shexp.weight")).reshape(1, self.hidden_size),
        )
        self.gate_exps = self._upload("blk.40.ffn_gate_exps.weight")
        self.up_exps = self._upload("blk.40.ffn_up_exps.weight")
        self.down_exps = self._upload("blk.40.ffn_down_exps.weight")
        self.shared_gate = self._upload("blk.40.ffn_gate_shexp.weight")
        self.shared_up = self._upload("blk.40.ffn_up_shexp.weight")
        self.shared_down = self._upload("blk.40.ffn_down_shexp.weight")
        self.shared_head_norm = self._upload("blk.40.nextn.shared_head_norm.weight")
        self.shared_head = self._upload("output.weight")

    def _malloc(self, nbytes: int) -> DeviceBuffer:
        buf = malloc(int(nbytes), runtime=self.runtime)
        self._buffers.append(buf)
        return buf

    def _allocate_buffers(self) -> None:
        h = self.hidden_size
        heads = self.num_heads
        kv_heads = self.num_kv_heads
        d = self.qk_head_dim
        top_k = self.experts_used
        inter = self.inter_dim
        self.seed_a = self._malloc(h * 4)
        self.seed_b = self._malloc(h * 4)
        self.token_embed = self._malloc(h * 4)
        self.e_norm = self._malloc(h * 4)
        self.h_norm = self._malloc(h * 4)
        self.concat = self._malloc(h * 2 * 4)
        self.projected = self._malloc(h * 4)
        self.attn_normed = self._malloc(h * 4)
        self.q_full = self._malloc(heads * 2 * d * 4)
        self.query = self._malloc(heads * d * 4)
        self.gate = self._malloc(heads * d * 4)
        self.key_cur = self._malloc(kv_heads * d * 4)
        self.value_cur = self._malloc(kv_heads * d * 4)
        self.cos = self._malloc(d * 4)
        self.sin = self._malloc(d * 4)
        self.position_i64 = self._malloc(8)
        self.context_i64 = self._malloc(8)
        self.attn = self._malloc(heads * d * 4)
        self.gated = self._malloc(heads * d * 4)
        self.wo_out = self._malloc(h * 4)
        self.attended = self._malloc(h * 4)
        self.post_norm = self._malloc(h * 4)
        self.post_norm_bf16 = self._malloc(h * 2)
        self.router_logits = self._malloc(256 * 4)
        self.selected = self._malloc(top_k * 8)
        self.routing = self._malloc(top_k * 4)
        self.selected_out = self._malloc(h * 4)
        self.gate_bf16 = self._malloc(top_k * inter * 2)
        self.up_bf16 = self._malloc(top_k * inter * 2)
        self.gate_f32 = self._malloc(top_k * inter * 4)
        self.up_f32 = self._malloc(top_k * inter * 4)
        self.inter_f32 = self._malloc(top_k * inter * 4)
        self.down_out = self._malloc(h * 4)
        self.scaled = self._malloc(h * 4)
        self.shared_gate_out = self._malloc(inter * 4)
        self.shared_up_out = self._malloc(inter * 4)
        self.shared_inter = self._malloc(inter * 4)
        self.shared_out = self._malloc(h * 4)
        self.shared_gate_logit = self._malloc(4)
        self.gated_shared = self._malloc(h * 4)
        self.tmp = self._malloc(h * 4)
        self.ffn_out = self._malloc(h * 4)
        # Device-resident MoE-down + combine scratch (bf16, matches the verifier).
        self.inter_bf16 = self._malloc(top_k * inter * 2)
        self.down_out_bf16 = self._malloc(top_k * h * 2)
        self.attended_bf16 = self._malloc(h * 2)
        self.shared_bf16 = self._malloc(h * 2)
        self.ffn_out_bf16 = self._malloc(h * 2)
        self.head_normed_bf16 = self._malloc(h * 2)
        self.logits = self._malloc(self.vocab * 4)
        self.topk_values = self._malloc(64 * 4)
        self.topk_indices = self._malloc(64 * 4)
        # Device-chain (sub-win B) scratch: per-depth rope/pos/ctx precomputed
        # once, top-k accumulated on device, single D->H at chain end.
        cap = self._draft_chain_cap
        self.cos_all = self._malloc(cap * d * 4)
        self.sin_all = self._malloc(cap * d * 4)
        self.pos_all = self._malloc(cap * 8)
        self.ctx_all = self._malloc(cap * 8)
        self.topk_all = self._malloc(cap * top_k * 4)

    def close(self) -> None:
        runtime = self.runtime or get_hip_runtime()
        for buf in reversed(self._buffers):
            free(buf, runtime=runtime)
        self._buffers.clear()

    def __enter__(self) -> "Qwen35GGUFResidentMTPDraftRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def propose_chain(
        self,
        hidden_seed: np.ndarray,
        *,
        start_token: int,
        start_position: int,
        draft_n_max: int,
        top_k: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        dense_key_cache: DeviceBuffer | None = None,
        dense_value_cache: DeviceBuffer | None = None,
        dense_cache_len: int = 0,
        draft_p_min: float = 0.0,
    ) -> tuple[list[int], list[list[int]], int]:
        if draft_n_max <= 0:
            raise ValueError("draft_n_max must be positive")
        if top_k <= 0 or top_k > 8:
            raise ValueError("resident GGUF MTP top_k must be in 1..8")
        hidden = np.ascontiguousarray(hidden_seed, dtype=np.float32)
        if hidden.shape != (1, self.hidden_size):
            raise ValueError("hidden_seed must have shape [1, hidden_size]")
        copy_host_to_device(self.seed_a, host_array_ptr(hidden), hidden.nbytes, runtime=self.runtime)
        if (
            self._device_chain_enabled
            and draft_p_min <= 0.0
            and int(draft_n_max) <= self._draft_chain_cap
        ):
            return self._propose_chain_device(
                current_seed=self.seed_a,
                next_seed=self.seed_b,
                start_token=int(start_token),
                start_position=int(start_position),
                draft_n_max=int(draft_n_max),
                top_k=int(top_k),
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                dense_key_cache=dense_key_cache,
                dense_value_cache=dense_value_cache,
                dense_cache_len=int(dense_cache_len),
            )
        current_seed = self.seed_a
        next_seed = self.seed_b
        current_token = int(start_token)
        current_pos = int(start_position)
        current_cache_len = int(dense_cache_len)
        tokens: list[int] = []
        topk_rows: list[list[int]] = []
        for depth in range(int(draft_n_max)):
            if current_token < 0 or current_token >= int(self.token_embd_f32.shape[0]):
                raise ValueError("draft token id outside embedding table")
            embed = np.ascontiguousarray(self.token_embd_f32[current_token:current_token + 1], dtype=np.float32)
            copy_host_to_device(self.token_embed, host_array_ptr(embed), embed.nbytes, runtime=self.runtime)
            pos = np.asarray([current_pos], dtype=np.int64)
            ctx = np.asarray([current_cache_len + 1], dtype=np.int64)
            cos = np.ascontiguousarray(rope_cos[pos], dtype=np.float32)
            sin = np.ascontiguousarray(rope_sin[pos], dtype=np.float32)
            copy_host_to_device(self.cos, host_array_ptr(cos), cos.nbytes, runtime=self.runtime)
            copy_host_to_device(self.sin, host_array_ptr(sin), sin.nbytes, runtime=self.runtime)
            copy_host_to_device(self.position_i64, host_array_ptr(pos), pos.nbytes, runtime=self.runtime)
            copy_host_to_device(self.context_i64, host_array_ptr(ctx), ctx.nbytes, runtime=self.runtime)
            self._run_one(
                current_seed,
                next_seed,
                cos_ptr=self.cos.ptr,
                sin_ptr=self.sin.ptr,
                pos_ptr=self.position_i64.ptr,
                ctx_ptr=self.context_i64.ptr,
                dense_key_cache=dense_key_cache,
                dense_value_cache=dense_value_cache,
                dense_cache_len=current_cache_len,
            )
            if dense_key_cache is not None:
                current_cache_len += 1
            top_ids = self._read_topk(top_k)
            draft_token = int(top_ids[0])
            tokens.append(draft_token)
            topk_rows.append([int(token) for token in top_ids])
            current_token = draft_token
            current_pos += 1
            current_seed, next_seed = next_seed, current_seed
            if draft_p_min > 0.0 and depth + 1 < int(draft_n_max):
                # The resident production path is greedy/top-k only.  Preserve
                # the old path for probability-threshold diagnostics.
                raise NotImplementedError("resident GGUF MTP draft does not support draft_p_min")
        return tokens, topk_rows, current_cache_len

    def _ensure_embed_table(self) -> None:
        """Lazily upload the vocab-capped FP32 embedding rows for device gather.

        Draft tokens are always top-k of the capped LM head (``< self.vocab``),
        so the resident table covers every reachable depth-1+ token id.  The
        rows are an exact copy of ``token_embd_f32`` -> the device-chain draft
        embeddings are bit-identical to the host-gather path.
        """
        if self._embed_table_f32 is not None:
            return
        rows = int(self.vocab)
        table = np.ascontiguousarray(self.token_embd_f32[:rows], dtype=np.float32)
        buf = self._malloc(table.nbytes)
        copy_host_to_device(buf, host_array_ptr(table), table.nbytes, runtime=self.runtime)
        self._embed_table_f32 = buf

    def _propose_chain_device(
        self,
        *,
        current_seed: DeviceBuffer,
        next_seed: DeviceBuffer,
        start_token: int,
        start_position: int,
        draft_n_max: int,
        top_k: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        dense_key_cache: DeviceBuffer | None,
        dense_value_cache: DeviceBuffer | None,
        dense_cache_len: int,
    ) -> tuple[list[int], list[list[int]], int]:
        """Device-chained NextN draft: one drain + one D->H for the whole chain.

        Removes the per-depth ``device_synchronize`` + top-1 readback + host
        embedding re-upload (each a blocking transfer).  Per-depth rope / pos /
        ctx are precomputed and uploaded once; each depth's top-1 is gathered
        device-side from the resident embedding table to feed the next depth.
        """
        runtime = self.runtime or get_hip_runtime()
        n = int(draft_n_max)
        d = self.qk_head_dim
        self._ensure_embed_table()
        if start_token < 0 or start_token >= int(self.token_embd_f32.shape[0]):
            raise ValueError("draft token id outside embedding table")
        # Depth-0 embedding: single host gather + upload (start_token may exceed
        # the capped resident table; depths 1+ are device-gathered).
        embed0 = np.ascontiguousarray(self.token_embd_f32[start_token:start_token + 1], dtype=np.float32)
        copy_host_to_device(self.token_embed, host_array_ptr(embed0), embed0.nbytes, runtime=runtime)
        # Precompute per-depth rope / position / context once; upload once.
        # The rope kernel reads ``qk_head_dim/2`` cos/sin values per token but the
        # table only supplies ``rope_w`` (rotary_dim) real columns; the legacy
        # path leaves the remainder of its ``self.cos`` scratch zeroed, so each
        # depth's slot is real[:rope_w] + zeros, strided by ``d`` -> bit-identical.
        rope_w = int(np.asarray(rope_cos).shape[1])
        if rope_w > d:
            raise ValueError("rope table width exceeds qk_head_dim scratch stride")
        positions = np.arange(n, dtype=np.int64) + int(start_position)
        if dense_key_cache is not None:
            ctxs = np.arange(n, dtype=np.int64) + int(dense_cache_len) + 1
        else:
            ctxs = np.full(n, int(dense_cache_len) + 1, dtype=np.int64)
        cos_strided = np.zeros((n, d), dtype=np.float32)
        sin_strided = np.zeros((n, d), dtype=np.float32)
        cos_strided[:, :rope_w] = np.asarray(rope_cos[positions], dtype=np.float32)
        sin_strided[:, :rope_w] = np.asarray(rope_sin[positions], dtype=np.float32)
        copy_host_to_device(self.cos_all, host_array_ptr(np.ascontiguousarray(cos_strided)), cos_strided.nbytes, runtime=runtime)
        copy_host_to_device(self.sin_all, host_array_ptr(np.ascontiguousarray(sin_strided)), sin_strided.nbytes, runtime=runtime)
        copy_host_to_device(self.pos_all, host_array_ptr(np.ascontiguousarray(positions)), positions.nbytes, runtime=runtime)
        copy_host_to_device(self.ctx_all, host_array_ptr(np.ascontiguousarray(ctxs)), ctxs.nbytes, runtime=runtime)
        current_cache_len = int(dense_cache_len)
        for depth in range(n):
            self._run_one(
                current_seed,
                next_seed,
                cos_ptr=self.cos_all.ptr + depth * d * 4,
                sin_ptr=self.sin_all.ptr + depth * d * 4,
                pos_ptr=self.pos_all.ptr + depth * 8,
                ctx_ptr=self.ctx_all.ptr + depth * 8,
                dense_key_cache=dense_key_cache,
                dense_value_cache=dense_value_cache,
                dense_cache_len=current_cache_len,
            )
            if dense_key_cache is not None:
                current_cache_len += 1
            # Record this depth's top-k on device (no sync, no readback).
            self._topk_indices_into(self.topk_all.ptr + depth * top_k * 4, top_k)
            # Device-gather the next depth's embedding from this depth's top-1.
            if depth + 1 < n:
                gather_f32_rows_by_i32id(
                    self._embed_table_f32.ptr,
                    self.topk_all.ptr + depth * top_k * 4,
                    self.token_embed.ptr,
                    1,
                    self.hidden_size,
                    self.vocab,
                    library=self._gather_lib,
                    runtime=runtime,
                )
            current_seed, next_seed = next_seed, current_seed
        # Single drain + readback of the whole chain's top-k.
        runtime.device_synchronize()
        topk_host = np.empty((n, int(top_k)), dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(topk_host),
            DeviceBuffer(self.topk_all.ptr, topk_host.nbytes),
            topk_host.nbytes,
            runtime=runtime,
        )
        tokens = [int(topk_host[depth, 0]) for depth in range(n)]
        topk_rows = [[int(token) for token in topk_host[depth].tolist()] for depth in range(n)]
        return tokens, topk_rows, current_cache_len

    def write_kv_rows(
        self,
        hidden_seed_rows: np.ndarray,
        token_ids: np.ndarray,
        *,
        positions: np.ndarray,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        dense_key_cache: DeviceBuffer,
        dense_value_cache: DeviceBuffer,
        dense_cache_len: int,
    ) -> int:
        hidden = np.ascontiguousarray(hidden_seed_rows, dtype=np.float32)
        tokens = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        pos = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError("hidden_seed_rows must have shape [rows, hidden_size]")
        if hidden.shape[0] != tokens.shape[0] or tokens.shape[0] != pos.shape[0]:
            raise ValueError("hidden rows, token ids, and positions must have the same length")
        current_len = int(dense_cache_len)
        for row in range(int(tokens.shape[0])):
            token = int(tokens[row])
            if token < 0 or token >= int(self.token_embd_f32.shape[0]):
                raise ValueError("commit token id outside embedding table")
            hidden_row = np.ascontiguousarray(hidden[row:row + 1], dtype=np.float32)
            embed = np.ascontiguousarray(self.token_embd_f32[token:token + 1], dtype=np.float32)
            cos = np.ascontiguousarray(rope_cos[pos[row:row + 1]], dtype=np.float32)
            sin = np.ascontiguousarray(rope_sin[pos[row:row + 1]], dtype=np.float32)
            copy_host_to_device(self.seed_a, host_array_ptr(hidden_row), hidden_row.nbytes, runtime=self.runtime)
            copy_host_to_device(self.token_embed, host_array_ptr(embed), embed.nbytes, runtime=self.runtime)
            self._write_one_kv(
                dense_key_cache=dense_key_cache,
                dense_value_cache=dense_value_cache,
                dense_cache_len=current_len,
                cos=cos,
                sin=sin,
            )
            current_len += 1
        return current_len

    def _project_current_to_attn_normed(self, hidden_seed: DeviceBuffer) -> None:
        runtime = self.runtime or get_hip_runtime()
        h = self.hidden_size
        mtp_rmsnorm_f32(self.token_embed.ptr, self.enorm.ptr, self.e_norm.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        mtp_rmsnorm_f32(hidden_seed.ptr, self.hnorm.ptr, self.h_norm.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        runtime.memcpy(self.concat.ptr, self.e_norm.ptr, h * 4, HipMemcpyKind.DEVICE_TO_DEVICE)
        runtime.memcpy(self.concat.ptr + h * 4, self.h_norm.ptr, h * 4, HipMemcpyKind.DEVICE_TO_DEVICE)
        gguf_q8_0_gemv_f32_f32_out(
            self.concat.ptr,
            self.eh_proj.ptr,
            self.projected.ptr,
            1,
            h * 2,
            h,
            library=self._k_lib,
            runtime=runtime,
        )
        mtp_rmsnorm_f32(self.projected.ptr, self.attn_norm.ptr, self.attn_normed.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)

    def _write_one_kv(
        self,
        *,
        dense_key_cache: DeviceBuffer,
        dense_value_cache: DeviceBuffer,
        dense_cache_len: int,
        cos: np.ndarray,
        sin: np.ndarray,
    ) -> None:
        runtime = self.runtime or get_hip_runtime()
        h = self.hidden_size
        kv_heads = self.num_kv_heads
        d = self.qk_head_dim
        self._project_current_to_attn_normed(self.seed_a)
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wk.ptr, self.key_cur.ptr, 1, h, kv_heads * d, library=self._k_lib, runtime=runtime)
        mtp_rmsnorm_f32(self.key_cur.ptr, self.k_norm.ptr, self.key_cur.ptr, kv_heads, d, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        copy_host_to_device(self.cos, host_array_ptr(cos), cos.nbytes, runtime=runtime)
        copy_host_to_device(self.sin, host_array_ptr(sin), sin.nbytes, runtime=runtime)
        mtp_rope_f32(self.key_cur.ptr, self.cos.ptr, self.sin.ptr, self.key_cur.ptr, 1, kv_heads, d, d, d // 2, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wv.ptr, self.value_cur.ptr, 1, h, kv_heads * d, library=self._k_lib, runtime=runtime)
        key_row_bytes = kv_heads * d * 4
        value_row_bytes = kv_heads * d * 4
        runtime.memcpy(
            dense_key_cache.ptr + int(dense_cache_len) * key_row_bytes,
            self.key_cur.ptr,
            key_row_bytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
        )
        runtime.memcpy(
            dense_value_cache.ptr + int(dense_cache_len) * value_row_bytes,
            self.value_cur.ptr,
            value_row_bytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
        )

    def _run_one(
        self,
        hidden_seed: DeviceBuffer,
        next_seed: DeviceBuffer,
        *,
        cos_ptr: int,
        sin_ptr: int,
        pos_ptr: int,
        ctx_ptr: int,
        dense_key_cache: DeviceBuffer | None,
        dense_value_cache: DeviceBuffer | None,
        dense_cache_len: int,
    ) -> None:
        runtime = self.runtime or get_hip_runtime()
        h = self.hidden_size
        heads = self.num_heads
        kv_heads = self.num_kv_heads
        d = self.qk_head_dim
        top_k = self.experts_used
        inter = self.inter_dim
        runtime.memset(self.selected_out.ptr, 0, self.selected_out.nbytes)

        self._project_current_to_attn_normed(hidden_seed)
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wq.ptr, self.q_full.ptr, 1, h, heads * 2 * d, library=self._k_lib, runtime=runtime)
        mtp_split_q_gate_f32(self.q_full.ptr, self.query.ptr, self.gate.ptr, 1, heads, d, library=self._mtp_lib, runtime=runtime)
        mtp_rmsnorm_f32(self.query.ptr, self.q_norm.ptr, self.query.ptr, heads, d, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wk.ptr, self.key_cur.ptr, 1, h, kv_heads * d, library=self._k_lib, runtime=runtime)
        mtp_rmsnorm_f32(self.key_cur.ptr, self.k_norm.ptr, self.key_cur.ptr, kv_heads, d, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        mtp_rope_f32(self.query.ptr, cos_ptr, sin_ptr, self.query.ptr, 1, heads, d, d, d // 2, runtime=runtime)
        mtp_rope_f32(self.key_cur.ptr, cos_ptr, sin_ptr, self.key_cur.ptr, 1, kv_heads, d, d, d // 2, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wv.ptr, self.value_cur.ptr, 1, h, kv_heads * d, library=self._k_lib, runtime=runtime)
        if dense_key_cache is not None:
            if dense_value_cache is None:
                raise ValueError("dense_value_cache is required with dense_key_cache")
            key_row_bytes = kv_heads * d * 4
            value_row_bytes = kv_heads * d * 4
            runtime.memcpy(
                dense_key_cache.ptr + int(dense_cache_len) * key_row_bytes,
                self.key_cur.ptr,
                key_row_bytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            runtime.memcpy(
                dense_value_cache.ptr + int(dense_cache_len) * value_row_bytes,
                self.value_cur.ptr,
                value_row_bytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            key_ptr = dense_key_cache.ptr
            value_ptr = dense_value_cache.ptr
            cache_tokens = int(dense_cache_len) + 1
        else:
            key_ptr = self.key_cur.ptr
            value_ptr = self.value_cur.ptr
            cache_tokens = 1
        mtp_dense_attn_f32(
            self.query.ptr,
            key_ptr,
            value_ptr,
            pos_ptr,
            ctx_ptr,
            self.attn.ptr,
            1,
            heads,
            kv_heads,
            d,
            d,
            cache_tokens,
            d ** -0.5,
            library=self._mtp_lib,
            runtime=runtime,
        )
        mtp_sigmoid_gate_mul_f32(self.attn.ptr, self.gate.ptr, self.gated.ptr, heads, d, library=self._mtp_lib, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.gated.ptr, self.wo.ptr, self.wo_out.ptr, 1, heads * d, h, library=self._k_lib, runtime=runtime)
        mtp_add_f32(self.projected.ptr, self.wo_out.ptr, self.attended.ptr, h, library=self._mtp_lib, runtime=runtime)

        mtp_rmsnorm_f32(self.attended.ptr, self.post_norm_weight.ptr, self.post_norm.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        mtp_linear_f32(self.post_norm.ptr, self.router_weight_f32.ptr, self.router_logits.ptr, 1, h, 256, library=self._mtp_lib, runtime=runtime)
        qwen35_router_select(
            self.router_logits.ptr,
            self.selected.ptr,
            self.routing.ptr,
            1,
            256,
            256,
            top_k,
            threads=256,
            library=self._router_lib,
            runtime=runtime,
        )
        f32_to_bf16(self.post_norm.ptr, self.post_norm_bf16.ptr, h, library=self._cast_lib, runtime=runtime)
        gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
            self.post_norm_bf16.ptr,
            self.selected.ptr,
            self.gate_exps.ptr,
            self.up_exps.ptr,
            self.gate_bf16.ptr,
            self.up_bf16.ptr,
            1,
            top_k,
            256,
            h,
            inter,
            library=self._q4_lib,
            runtime=runtime,
        )
        gguf_q8_0_gemv_f32_f32_out(self.post_norm.ptr, self.shared_gate.ptr, self.shared_gate_out.ptr, 1, h, inter, library=self._k_lib, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.post_norm.ptr, self.shared_up.ptr, self.shared_up_out.ptr, 1, h, inter, library=self._k_lib, runtime=runtime)
        mtp_silu_mul_f32(self.shared_gate_out.ptr, self.shared_up_out.ptr, self.shared_inter.ptr, inter, library=self._mtp_lib, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.shared_inter.ptr, self.shared_down.ptr, self.shared_out.ptr, 1, inter, h, library=self._k_lib, runtime=runtime)
        mtp_linear_f32(self.post_norm.ptr, self.shared_gate_vec_f32.ptr, self.shared_gate_logit.ptr, 1, h, 1, library=self._mtp_lib, runtime=runtime)

        if self._device_moe_enabled:
            # Device-resident selected-MoE down + combine: no host readback of
            # selected/routing, no per-expert Python loop (matches the verifier).
            apply_moe_down_combine(
                gate_bf16_ptr=self.gate_bf16.ptr,
                up_bf16_ptr=self.up_bf16.ptr,
                selected_ptr=self.selected.ptr,
                routing_ptr=self.routing.ptr,
                shared_out_ptr=self.shared_out.ptr,
                shared_gate_logit_ptr=self.shared_gate_logit.ptr,
                residual_ptr=self.attended.ptr,
                down_exps_ptr=self.down_exps.ptr,
                inter_bf16_ptr=self.inter_bf16.ptr,
                down_out_bf16_ptr=self.down_out_bf16.ptr,
                attended_bf16_ptr=self.attended_bf16.ptr,
                shared_bf16_ptr=self.shared_bf16.ptr,
                ffn_out_bf16_ptr=self.ffn_out_bf16.ptr,
                ffn_out_f32_ptr=self.ffn_out.ptr,
                top_k=top_k,
                inter=inter,
                hidden=h,
                num_experts=256,
                silu_lib=self._silu_lib,
                k_lib=self._k_lib,
                combine_lib=self._combine_lib,
                cast_lib=self._cast_lib,
                runtime=runtime,
            )
        else:
            # Legacy host-readback per-expert down loop + shared-gate combine.
            bf16_to_f32(self.gate_bf16.ptr, self.gate_f32.ptr, top_k * inter, library=self._cast_lib, runtime=runtime)
            bf16_to_f32(self.up_bf16.ptr, self.up_f32.ptr, top_k * inter, library=self._cast_lib, runtime=runtime)
            down_per_expert = int(self._get("blk.40.ffn_down_exps.weight").nbytes // 256)
            selected_host = np.empty((top_k,), dtype=np.int64)
            routing_host = np.empty((top_k,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(selected_host), self.selected, selected_host.nbytes, runtime=runtime)
            copy_device_to_host(host_array_ptr(routing_host), self.routing, routing_host.nbytes, runtime=runtime)
            for k in range(top_k):
                inter_ptr = self.inter_f32.ptr + k * inter * 4
                mtp_silu_mul_f32(
                    self.gate_f32.ptr + k * inter * 4,
                    self.up_f32.ptr + k * inter * 4,
                    inter_ptr,
                    inter,
                    library=self._mtp_lib,
                    runtime=runtime,
                )
                expert = int(selected_host[k])
                gguf_q5_k_gemv_f32_f32_out(
                    inter_ptr,
                    self.down_exps.ptr + expert * down_per_expert,
                    self.down_out.ptr,
                    1,
                    inter,
                    h,
                    library=self._k_lib,
                    runtime=runtime,
                )
                mtp_scale_f32(
                    self.down_out.ptr,
                    self.scaled.ptr,
                    float(routing_host[k]),
                    h,
                    library=self._mtp_lib,
                    runtime=runtime,
                )
                mtp_add_f32(
                    self.selected_out.ptr,
                    self.scaled.ptr,
                    self.selected_out.ptr,
                    h,
                    library=self._mtp_lib,
                    runtime=runtime,
                )
            mtp_sigmoid_row_scale_from_logits_f32(
                self.shared_gate_logit.ptr,
                self.shared_out.ptr,
                self.gated_shared.ptr,
                1,
                h,
                1,
                library=self._mtp_lib,
                runtime=runtime,
            )
            mtp_add_f32(self.attended.ptr, self.selected_out.ptr, self.tmp.ptr, h, library=self._mtp_lib, runtime=runtime)
            mtp_add_f32(self.tmp.ptr, self.gated_shared.ptr, self.ffn_out.ptr, h, library=self._mtp_lib, runtime=runtime)

        mtp_rmsnorm_f32(self.ffn_out.ptr, self.shared_head_norm.ptr, next_seed.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        f32_to_bf16(next_seed.ptr, self.head_normed_bf16.ptr, h, library=self._cast_lib, runtime=runtime)
        gguf_q6_k_pack8_gemv_decode_bf16_f32_out(
            self.head_normed_bf16.ptr,
            self.shared_head.ptr,
            self.logits.ptr,
            1,
            h,
            self.vocab,
            library=self._q6_pack8_lib,
            runtime=runtime,
        )

    def _topk_indices_into(self, out_indices_ptr: int, top_k: int) -> None:
        """Write the top-``top_k`` logit indices to a device buffer (no sync)."""
        runtime = self.runtime or get_hip_runtime()
        topk_f32_rows_i32(
            self.logits.ptr,
            self.topk_values.ptr,
            out_indices_ptr,
            1,
            self.vocab,
            int(top_k),
            threads=256,
            library=self._lm_head_lib,
            runtime=runtime,
        )

    def _read_topk(self, top_k: int) -> list[int]:
        runtime = self.runtime or get_hip_runtime()
        self._topk_indices_into(self.topk_indices.ptr, top_k)
        runtime.device_synchronize()
        out = np.empty((int(top_k),), dtype=np.int32)
        copy_device_to_host(host_array_ptr(out), DeviceBuffer(self.topk_indices.ptr, out.nbytes), out.nbytes, runtime=runtime)
        return [int(token) for token in out.tolist()]


__all__ = ["Qwen35GGUFResidentMTPDraftRunner"]
