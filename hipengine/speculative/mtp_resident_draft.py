"""Resident GGUF MTP NextN draft runner.

This is the production-shaped companion to the correctness-first NumPy wrapper
in ``hip_gfx1100.speculative.mtp_nextn``.  It keeps the one-layer NextN draft
chain on device across depths and returns only the small top-k token IDs needed
by the Python acceptance harness.
"""

from __future__ import annotations

import os
import time
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
    qwen35_router_logits_f32_f32w,
    qwen35_router_select,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_dual_gemv_f32_f32_out,
    gguf_q5_k_gemv_f32_f32_out,
    gguf_q5_k_selected_gemv_bf16_bf16_out,
    gguf_q8_0_gemv_f32_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
    gguf_q6_k_pack8_gemv_decode_bf16_f32_out,
    gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32,
    gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32,
    gguf_q6_k_pack8_top1_stage2_gather_f32,
    gguf_q6_k_x8_dscale_gemv_decode_q8_1_dp4a_top1_gather_f32,
    gguf_q6_k_x8_dscale_gemv_decode_q8_1_dp4a_top1_stage1_f32,
    gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32,
    gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_stage1_f32,
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
from hipengine.quant.gguf_x8 import repack_gguf_q6_k_x8, repack_gguf_q6_k_x8_dscale_f32


def _stage_add(timings: dict[str, float] | None, name: str, ms: float) -> None:
    if timings is None:
        return
    if ms < 0.0:
        raise RuntimeError(f"negative resident MTP stage timing for {name}: {ms}")
    timings[name] = timings.get(name, 0.0) + float(ms)


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


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name, default).strip().lower()
    if raw not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return raw


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
    stage_marker=None,
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
    if stage_marker is not None:
        stage_marker("draft_run_moe_selected_silu")
    # Selected-down GEMV: each expert consumes its own intermediate row
    # (x_rows == rows == top_k  =>  lanes_per_x_row == 1  =>  x_row == row).
    gguf_q5_k_selected_gemv_bf16_bf16_out(
        inter_bf16_ptr, selected_ptr, down_exps_ptr, down_out_bf16_ptr,
        top_k, top_k, num_experts, inter, hidden,
        library=k_lib, runtime=runtime,
    )
    if stage_marker is not None:
        stage_marker("draft_run_moe_selected_down")
    # Cast the f32 residual + shared-expert output to bf16 for the combine.
    f32_to_bf16(residual_ptr, attended_bf16_ptr, hidden, library=cast_lib, runtime=runtime)
    f32_to_bf16(shared_out_ptr, shared_bf16_ptr, hidden, library=cast_lib, runtime=runtime)
    if stage_marker is not None:
        stage_marker("draft_run_moe_combine_cast_inputs")
    # routing-weighted expert sum + sigmoid(gate)*shared + residual, in one launch.
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
        down_out_bf16_ptr, routing_ptr, shared_bf16_ptr, shared_gate_logit_ptr,
        attended_bf16_ptr, ffn_out_bf16_ptr, top_k, hidden,
        library=combine_lib, runtime=runtime,
    )
    bf16_to_f32(ffn_out_bf16_ptr, ffn_out_f32_ptr, hidden, library=cast_lib, runtime=runtime)
    if stage_marker is not None:
        stage_marker("draft_run_moe_weighted_combine")


@dataclass
class Qwen35GGUFResidentMTPDraftRunner:
    """Device-resident chain runner for the real Qwen3.6 GGUF NextN block."""

    weights: dict[str, tuple[np.ndarray, object, object]]
    token_embd_f32: np.ndarray
    runtime: HipRuntime | None = None
    vocab_cap: int = 32768
    device_chain_enabled: bool | None = None
    prewarm_device_chain: bool = False
    sync_stage_timings: bool = False
    compiler_version: str | None = None
    require_cached_build: bool = False
    num_heads: int = 16
    num_kv_heads: int = 2
    experts_used: int = 8
    eps: float = 1e-6
    _buffers: list[DeviceBuffer] = field(default_factory=list, init=False)
    last_top1_probs: list[float] = field(default_factory=list, init=False)
    last_stage_timings_ms: dict[str, float] = field(default_factory=dict, init=False)

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
        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": bool(self.require_cached_build),
        }
        self._mtp_lib = build_mtp_nextn(**build_kwargs)
        self._k_lib = build_gguf_k_gemv(**build_kwargs)
        self._q4_lib = build_gguf_q4_k_gemv(**build_kwargs)
        self._q6_pack8_lib = build_gguf_q6_k_pack8_gemv(**build_kwargs)
        self._cast_lib = build_cast(**build_kwargs)
        self._router_lib = build_qwen35_router(**build_kwargs)
        self._lm_head_lib = build_lm_head(**build_kwargs)
        self._silu_lib = build_paro_silu(**build_kwargs)
        self._combine_lib = build_paro_combine(**build_kwargs)
        self._gather_lib = build_gather(**build_kwargs)
        self._device_moe_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE", True)
        self._q6_top1_gather_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER", True)
        self._q6_top1_dp4a_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A", False)
        self._q6_top1_stage1_shape = _env_choice(
            "HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE",
            "pack8",
            {"pack8", "pack16", "pack8_llama", "pack8_scalehoist", "row", "x8", "x8_dscale"},
        )
        if self._q6_top1_stage1_shape == "pack16" and self.vocab % 16 != 0:
            raise ValueError("pack16 Q6 top-1 diagnostic requires vocab divisible by 16")
        self._q8_shared_dual_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", True)
        self._router_row_parallel_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL", False)
        if self.device_chain_enabled is None:
            self._device_chain_enabled = _env_flag("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN", False)
        else:
            self._device_chain_enabled = bool(self.device_chain_enabled)
        # Max draft depth the precomputed-rope / topk-accumulator buffers cover.
        self._draft_chain_cap = 16
        self._embed_table_f32: DeviceBuffer | None = None
        self._upload_weights()
        self._allocate_buffers()
        if self.prewarm_device_chain and self._device_chain_enabled:
            self.ensure_device_chain_ready()

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
        self.shared_head_x8: DeviceBuffer | None = None
        self.shared_head_x8_dscale: DeviceBuffer | None = None
        if self._q6_top1_stage1_shape in {"x8", "x8_dscale"}:
            raw_head = np.ascontiguousarray(self._get("output.weight")[: self.vocab], dtype=np.uint8)
            packed_head = np.ascontiguousarray(repack_gguf_q6_k_x8(raw_head.reshape(1, self.vocab, -1)).tiles[0])
            self.shared_head_x8 = self._upload(f"output.weight:q6_x8_top1:vocab{self.vocab}", packed_head)
            if self._q6_top1_stage1_shape == "x8_dscale":
                packed_dscale = repack_gguf_q6_k_x8_dscale_f32(raw_head.reshape(1, self.vocab, -1))[0]
                self.shared_head_x8_dscale = self._upload(
                    f"output.weight:q6_x8_dscale_top1:vocab{self.vocab}",
                    np.ascontiguousarray(packed_dscale, dtype=np.float32),
                )

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
        self.head_normed_q8_1 = self._malloc((h // 32) * 36)
        self.logits = self._malloc(self.vocab * 4)
        self.q6_top1_block_values = self._malloc(self.vocab * 4)
        self.q6_top1_block_indices = self._malloc(self.vocab * 4)
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

    def ensure_device_chain_ready(self) -> None:
        """Preload resident state needed by the device-chained draft path."""

        self._ensure_embed_table()

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
        record_top1_probs: bool = False,
        record_stage_timings: bool = False,
    ) -> tuple[list[int], list[list[int]], int]:
        stage_timings: dict[str, float] | None = {} if record_stage_timings else None
        self.last_stage_timings_ms = stage_timings if stage_timings is not None else {}
        if draft_n_max <= 0:
            raise ValueError("draft_n_max must be positive")
        if top_k <= 0 or top_k > 64:
            raise ValueError("resident GGUF MTP top_k must be in 1..64")
        t_seed0 = time.perf_counter() if stage_timings is not None else 0.0
        hidden = np.ascontiguousarray(hidden_seed, dtype=np.float32)
        if hidden.shape != (1, self.hidden_size):
            raise ValueError("hidden_seed must have shape [1, hidden_size]")
        copy_host_to_device(self.seed_a, host_array_ptr(hidden), hidden.nbytes, runtime=self.runtime)
        if stage_timings is not None:
            _stage_add(stage_timings, "draft_seed_upload", (time.perf_counter() - t_seed0) * 1000)
        return self._propose_chain_from_seed_buffer(
            start_token=start_token,
            start_position=start_position,
            draft_n_max=draft_n_max,
            top_k=top_k,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dense_key_cache=dense_key_cache,
            dense_value_cache=dense_value_cache,
            dense_cache_len=dense_cache_len,
            draft_p_min=draft_p_min,
            record_top1_probs=record_top1_probs,
            stage_timings=stage_timings,
        )

    def propose_chain_from_device_seed(
        self,
        hidden_seed_ptr: int,
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
        record_top1_probs: bool = False,
        record_stage_timings: bool = False,
    ) -> tuple[list[int], list[list[int]], int]:
        """Run the draft chain from an already-resident target hidden seed."""

        stage_timings: dict[str, float] | None = {} if record_stage_timings else None
        self.last_stage_timings_ms = stage_timings if stage_timings is not None else {}
        ptr = int(hidden_seed_ptr)
        if ptr <= 0:
            raise ValueError("hidden_seed_ptr must be a non-zero device pointer")
        runtime = self.runtime or get_hip_runtime()
        t_seed0 = time.perf_counter() if stage_timings is not None else 0.0
        runtime.memcpy(self.seed_a.ptr, ptr, self.hidden_size * 4, HipMemcpyKind.DEVICE_TO_DEVICE)
        if stage_timings is not None:
            _stage_add(stage_timings, "draft_seed_upload", (time.perf_counter() - t_seed0) * 1000)
        return self._propose_chain_from_seed_buffer(
            start_token=start_token,
            start_position=start_position,
            draft_n_max=draft_n_max,
            top_k=top_k,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dense_key_cache=dense_key_cache,
            dense_value_cache=dense_value_cache,
            dense_cache_len=dense_cache_len,
            draft_p_min=draft_p_min,
            record_top1_probs=record_top1_probs,
            stage_timings=stage_timings,
        )

    def _propose_chain_from_seed_buffer(
        self,
        *,
        start_token: int,
        start_position: int,
        draft_n_max: int,
        top_k: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        dense_key_cache: DeviceBuffer | None,
        dense_value_cache: DeviceBuffer | None,
        dense_cache_len: int,
        draft_p_min: float,
        record_top1_probs: bool = False,
        stage_timings: dict[str, float] | None = None,
    ) -> tuple[list[int], list[list[int]], int]:
        self.last_top1_probs = []
        if draft_n_max <= 0:
            raise ValueError("draft_n_max must be positive")
        if top_k <= 0 or top_k > 64:
            raise ValueError("resident GGUF MTP top_k must be in 1..64")
        if (
            self._device_chain_enabled
            and draft_p_min <= 0.0
            and not bool(record_top1_probs)
            and int(draft_n_max) <= self._draft_chain_cap
            and int(top_k) <= self.experts_used
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
                stage_timings=stage_timings,
            )
        current_seed = self.seed_a
        next_seed = self.seed_b
        current_token = int(start_token)
        current_pos = int(start_position)
        current_cache_len = int(dense_cache_len)
        tokens: list[int] = []
        topk_rows: list[list[int]] = []
        for depth in range(int(draft_n_max)):
            t_prepare0 = time.perf_counter() if stage_timings is not None else 0.0
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
            if stage_timings is not None:
                _stage_add(stage_timings, "draft_prepare_inputs", (time.perf_counter() - t_prepare0) * 1000)
            t_forward0 = time.perf_counter() if stage_timings is not None else 0.0
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
                stage_timings=stage_timings,
            )
            if stage_timings is not None:
                _stage_add(stage_timings, "draft_mtp_layer_forward", (time.perf_counter() - t_forward0) * 1000)
            t_topk0 = time.perf_counter() if stage_timings is not None else 0.0
            if draft_p_min > 0.0 or record_top1_probs:
                top_ids, top1_prob = self._read_topk_with_prob(top_k)
                if record_top1_probs:
                    self.last_top1_probs.append(float(top1_prob))
                if top1_prob < float(draft_p_min):
                    break
            else:
                top_ids = self._read_topk(top_k)
            if stage_timings is not None:
                _stage_add(stage_timings, "draft_topk_readback", (time.perf_counter() - t_topk0) * 1000)
            draft_token = int(top_ids[0])
            tokens.append(draft_token)
            topk_rows.append([int(token) for token in top_ids])
            if dense_key_cache is not None:
                current_cache_len += 1
            current_token = draft_token
            current_pos += 1
            current_seed, next_seed = next_seed, current_seed
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
        self._embed_table_f32 = _cached_upload(
            f"resident_mtp:token_embd_f32:vocab{rows}:hidden{self.hidden_size}",
            table,
            runtime=self.runtime,
        )

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
        stage_timings: dict[str, float] | None = None,
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
        t_ensure0 = time.perf_counter() if stage_timings is not None else 0.0
        self._ensure_embed_table()
        if stage_timings is not None:
            _stage_add(
                stage_timings,
                "draft_device_chain_ensure_embed_table",
                (time.perf_counter() - t_ensure0) * 1000,
            )
        if start_token < 0 or start_token >= int(self.token_embd_f32.shape[0]):
            raise ValueError("draft token id outside embedding table")
        t_prepare0 = time.perf_counter() if stage_timings is not None else 0.0
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
        if stage_timings is not None:
            _stage_add(stage_timings, "draft_prepare_inputs", (time.perf_counter() - t_prepare0) * 1000)
        current_cache_len = int(dense_cache_len)
        q6_top1_gather_enabled = bool(getattr(self, "_q6_top1_gather_enabled", False))
        for depth in range(n):
            t_forward0 = time.perf_counter() if stage_timings is not None else 0.0
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
                stage_timings=stage_timings,
                top1_out_ptr=(self.topk_all.ptr + depth * top_k * 4) if (top_k == 1 and q6_top1_gather_enabled) else None,
                top1_next_embed_ptr=(
                    self.token_embed.ptr
                    if (top_k == 1 and q6_top1_gather_enabled and depth + 1 < n)
                    else None
                ),
            )
            if stage_timings is not None:
                _stage_add(stage_timings, "draft_mtp_layer_forward", (time.perf_counter() - t_forward0) * 1000)
            if dense_key_cache is not None:
                current_cache_len += 1
            # Record this depth's top-k on device (no sync, no readback).
            t_topk0 = time.perf_counter() if stage_timings is not None else 0.0
            if top_k == 1 and q6_top1_gather_enabled:
                # The Q6_K lm-head fast path already wrote the top-1 id and,
                # for non-final depths, gathered the next token embedding.
                pass
            else:
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
            if bool(getattr(self, "sync_stage_timings", False)) and stage_timings is not None:
                runtime.device_synchronize()
            if stage_timings is not None:
                _stage_add(stage_timings, "draft_device_topk_gather", (time.perf_counter() - t_topk0) * 1000)
            current_seed, next_seed = next_seed, current_seed
        # Single drain + readback of the whole chain's top-k.
        t_readback0 = time.perf_counter() if stage_timings is not None else 0.0
        t_drain0 = time.perf_counter() if stage_timings is not None else 0.0
        runtime.device_synchronize()
        if stage_timings is not None:
            _stage_add(stage_timings, "draft_device_chain_drain", (time.perf_counter() - t_drain0) * 1000)
        topk_host = np.empty((n, int(top_k)), dtype=np.int32)
        t_d2h0 = time.perf_counter() if stage_timings is not None else 0.0
        copy_device_to_host(
            host_array_ptr(topk_host),
            DeviceBuffer(self.topk_all.ptr, topk_host.nbytes),
            topk_host.nbytes,
            runtime=runtime,
        )
        if stage_timings is not None:
            _stage_add(stage_timings, "draft_topk_d2h", (time.perf_counter() - t_d2h0) * 1000)
        if stage_timings is not None:
            _stage_add(stage_timings, "draft_topk_readback", (time.perf_counter() - t_readback0) * 1000)
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

    def write_kv_rows_from_device_seed_base(
        self,
        hidden_seed_base_ptr: int,
        token_ids: np.ndarray,
        *,
        positions: np.ndarray,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        dense_key_cache: DeviceBuffer,
        dense_value_cache: DeviceBuffer,
        dense_cache_len: int,
        hidden_stride_bytes: int | None = None,
    ) -> int:
        """Write accepted MTP K/V rows from contiguous device hidden seeds."""

        base_ptr = int(hidden_seed_base_ptr)
        if base_ptr <= 0:
            raise ValueError("hidden_seed_base_ptr must be a non-zero device pointer")
        tokens = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        pos = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
        if tokens.shape[0] != pos.shape[0]:
            raise ValueError("token ids and positions must have the same length")
        stride = int(hidden_stride_bytes or (self.hidden_size * 4))
        if stride < self.hidden_size * 4:
            raise ValueError("hidden_stride_bytes is smaller than one fp32 hidden row")
        runtime = self.runtime or get_hip_runtime()
        current_len = int(dense_cache_len)
        for row in range(int(tokens.shape[0])):
            token = int(tokens[row])
            if token < 0 or token >= int(self.token_embd_f32.shape[0]):
                raise ValueError("commit token id outside embedding table")
            embed = np.ascontiguousarray(self.token_embd_f32[token:token + 1], dtype=np.float32)
            cos = np.ascontiguousarray(rope_cos[pos[row:row + 1]], dtype=np.float32)
            sin = np.ascontiguousarray(rope_sin[pos[row:row + 1]], dtype=np.float32)
            runtime.memcpy(
                self.seed_a.ptr,
                base_ptr + row * stride,
                self.hidden_size * 4,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
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

    def _project_current_to_attn_normed(self, hidden_seed: DeviceBuffer, *, stage_marker=None) -> None:
        runtime = self.runtime or get_hip_runtime()
        h = self.hidden_size
        mtp_rmsnorm_f32(self.token_embed.ptr, self.enorm.ptr, self.e_norm.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        mtp_rmsnorm_f32(hidden_seed.ptr, self.hnorm.ptr, self.h_norm.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        runtime.memcpy(self.concat.ptr, self.e_norm.ptr, h * 4, HipMemcpyKind.DEVICE_TO_DEVICE)
        runtime.memcpy(self.concat.ptr + h * 4, self.h_norm.ptr, h * 4, HipMemcpyKind.DEVICE_TO_DEVICE)
        if stage_marker is not None:
            stage_marker("draft_run_project_norm_concat")
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
        if stage_marker is not None:
            stage_marker("draft_run_project_eh_proj")
        mtp_rmsnorm_f32(self.projected.ptr, self.attn_norm.ptr, self.attn_normed.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        if stage_marker is not None:
            stage_marker("draft_run_project_attn_norm")

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
        stage_timings: dict[str, float] | None = None,
        top1_out_ptr: int | None = None,
        top1_next_embed_ptr: int | None = None,
    ) -> None:
        runtime = self.runtime or get_hip_runtime()
        h = self.hidden_size
        heads = self.num_heads
        kv_heads = self.num_kv_heads
        d = self.qk_head_dim
        top_k = self.experts_used
        inter = self.inter_dim
        sync_stages = bool(self.sync_stage_timings and stage_timings is not None)

        def mark_stage(name: str, t0: float) -> float:
            if sync_stages:
                runtime.device_synchronize()
                _stage_add(stage_timings, name, (time.perf_counter() - t0) * 1000)
                return time.perf_counter()
            return t0

        def add_aggregate_stage(name: str, t0: float) -> None:
            if sync_stages:
                _stage_add(stage_timings, name, (time.perf_counter() - t0) * 1000)

        def mark_substage(name: str) -> None:
            nonlocal t_stage
            t_stage = mark_stage(name, t_stage)

        t_stage = time.perf_counter() if sync_stages else 0.0
        runtime.memset(self.selected_out.ptr, 0, self.selected_out.nbytes)

        t_project0 = time.perf_counter() if sync_stages else 0.0
        self._project_current_to_attn_normed(hidden_seed, stage_marker=mark_substage if sync_stages else None)
        add_aggregate_stage("draft_run_project", t_project0)
        t_qkv0 = time.perf_counter() if sync_stages else 0.0
        t_stage = t_qkv0 if sync_stages else t_stage
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wq.ptr, self.q_full.ptr, 1, h, heads * 2 * d, library=self._k_lib, runtime=runtime)
        mtp_split_q_gate_f32(self.q_full.ptr, self.query.ptr, self.gate.ptr, 1, heads, d, library=self._mtp_lib, runtime=runtime)
        mtp_rmsnorm_f32(self.query.ptr, self.q_norm.ptr, self.query.ptr, heads, d, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_qkv_q_gate", t_stage)
        gguf_q8_0_gemv_f32_f32_out(self.attn_normed.ptr, self.wk.ptr, self.key_cur.ptr, 1, h, kv_heads * d, library=self._k_lib, runtime=runtime)
        mtp_rmsnorm_f32(self.key_cur.ptr, self.k_norm.ptr, self.key_cur.ptr, kv_heads, d, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        mtp_rope_f32(self.query.ptr, cos_ptr, sin_ptr, self.query.ptr, 1, heads, d, d, d // 2, runtime=runtime)
        mtp_rope_f32(self.key_cur.ptr, cos_ptr, sin_ptr, self.key_cur.ptr, 1, kv_heads, d, d, d // 2, runtime=runtime)
        t_stage = mark_stage("draft_run_qkv_k_rope", t_stage)
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
        t_stage = mark_stage("draft_run_qkv_v_kvwrite", t_stage)
        add_aggregate_stage("draft_run_qkv_kvwrite", t_qkv0)
        t_attention0 = time.perf_counter() if sync_stages else 0.0
        t_stage = t_attention0 if sync_stages else t_stage
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
        t_stage = mark_stage("draft_run_attention_core", t_stage)
        mtp_sigmoid_gate_mul_f32(self.attn.ptr, self.gate.ptr, self.gated.ptr, heads, d, library=self._mtp_lib, runtime=runtime)
        gguf_q8_0_gemv_f32_f32_out(self.gated.ptr, self.wo.ptr, self.wo_out.ptr, 1, heads * d, h, library=self._k_lib, runtime=runtime)
        mtp_add_f32(self.projected.ptr, self.wo_out.ptr, self.attended.ptr, h, library=self._mtp_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_attention_out", t_stage)
        add_aggregate_stage("draft_run_attention", t_attention0)

        t_ffn0 = time.perf_counter() if sync_stages else 0.0
        t_stage = t_ffn0 if sync_stages else t_stage
        mtp_rmsnorm_f32(self.attended.ptr, self.post_norm_weight.ptr, self.post_norm.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_ffn_post_norm", t_stage)
        if self._router_row_parallel_enabled:
            qwen35_router_logits_f32_f32w(
                self.post_norm.ptr,
                self.router_weight_f32.ptr,
                self.router_logits.ptr,
                1,
                h,
                256,
                threads=256,
                library=self._router_lib,
                runtime=runtime,
            )
        else:
            mtp_linear_f32(
                self.post_norm.ptr,
                self.router_weight_f32.ptr,
                self.router_logits.ptr,
                1,
                h,
                256,
                library=self._mtp_lib,
                runtime=runtime,
            )
        t_stage = mark_stage("draft_run_ffn_router_linear", t_stage)
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
        t_stage = mark_stage("draft_run_ffn_router_select_only", t_stage)
        f32_to_bf16(self.post_norm.ptr, self.post_norm_bf16.ptr, h, library=self._cast_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_ffn_post_norm_cast_bf16", t_stage)
        add_aggregate_stage("draft_run_ffn_router_select", t_ffn0)
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
        t_stage = mark_stage("draft_run_ffn_selected_gate_up", t_stage)
        if bool(getattr(self, "_q8_shared_dual_enabled", False)):
            gguf_q8_0_dual_gemv_f32_f32_out(
                self.post_norm.ptr,
                self.shared_gate.ptr,
                self.shared_up.ptr,
                self.shared_gate_out.ptr,
                self.shared_up_out.ptr,
                1,
                h,
                inter,
                library=self._k_lib,
                runtime=runtime,
            )
        else:
            gguf_q8_0_gemv_f32_f32_out(self.post_norm.ptr, self.shared_gate.ptr, self.shared_gate_out.ptr, 1, h, inter, library=self._k_lib, runtime=runtime)
            gguf_q8_0_gemv_f32_f32_out(self.post_norm.ptr, self.shared_up.ptr, self.shared_up_out.ptr, 1, h, inter, library=self._k_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_ffn_shared_gate_up", t_stage)
        mtp_silu_mul_f32(self.shared_gate_out.ptr, self.shared_up_out.ptr, self.shared_inter.ptr, inter, library=self._mtp_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_ffn_shared_silu", t_stage)
        gguf_q8_0_gemv_f32_f32_out(self.shared_inter.ptr, self.shared_down.ptr, self.shared_out.ptr, 1, inter, h, library=self._k_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_ffn_shared_down", t_stage)
        mtp_linear_f32(self.post_norm.ptr, self.shared_gate_vec_f32.ptr, self.shared_gate_logit.ptr, 1, h, 1, library=self._mtp_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_ffn_shared_gate_linear", t_stage)
        add_aggregate_stage("draft_run_ffn_up_shared", t_ffn0)

        t_moe_down0 = time.perf_counter() if sync_stages else 0.0
        t_stage = t_moe_down0 if sync_stages else t_stage
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
                stage_marker=mark_substage if sync_stages else None,
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

        add_aggregate_stage("draft_run_moe_down_combine", t_moe_down0)
        t_stage = time.perf_counter() if sync_stages else t_stage
        t_lm_head0 = time.perf_counter() if sync_stages else 0.0
        mtp_rmsnorm_f32(self.ffn_out.ptr, self.shared_head_norm.ptr, next_seed.ptr, 1, h, eps=self.eps, library=self._mtp_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_lm_head_norm", t_stage)
        f32_to_bf16(next_seed.ptr, self.head_normed_bf16.ptr, h, library=self._cast_lib, runtime=runtime)
        t_stage = mark_stage("draft_run_lm_head_cast_bf16", t_stage)
        if top1_out_ptr is not None:
            if bool(getattr(self, "_q6_top1_dp4a_enabled", False)):
                gguf_q4_k_quantize_bf16_q8_1(
                    self.head_normed_bf16.ptr,
                    self.head_normed_q8_1.ptr,
                    1,
                    h,
                    library=self._q4_lib,
                    runtime=runtime,
                )
                t_stage = mark_stage("draft_run_lm_head_quant_q8_1", t_stage)
                t_q6_top1 = t_stage
                q6_top1_stage1_shape = str(getattr(self, "_q6_top1_stage1_shape", "pack8"))
                q6_top1_stage2_blocks = (
                    self.vocab
                    if q6_top1_stage1_shape == "row"
                    else self.vocab // 16
                    if q6_top1_stage1_shape == "pack16"
                    else self.vocab // 8
                )
                q6_top1_stage_name = {
                    "row": "draft_run_lm_head_q6_top1_dp4a_row_stage1",
                    "pack16": "draft_run_lm_head_q6_top1_dp4a_pack16_stage1",
                    "x8": "draft_run_lm_head_q6_top1_dp4a_x8_stage1",
                    "x8_dscale": "draft_run_lm_head_q6_top1_dp4a_x8_dscale_stage1",
                    "pack8_llama": "draft_run_lm_head_q6_top1_dp4a_pack8_llama_stage1",
                    "pack8_scalehoist": "draft_run_lm_head_q6_top1_dp4a_scalehoist_stage1",
                }.get(q6_top1_stage1_shape, "draft_run_lm_head_q6_top1_dp4a_stage1")
                q6_top1_stage2_name = (
                    "draft_run_lm_head_q6_top1_dp4a_row_stage2_gather"
                    if q6_top1_stage1_shape == "row"
                    else "draft_run_lm_head_q6_top1_dp4a_stage2_gather"
                )
                q6_top1_gather_name = {
                    "row": "draft_run_lm_head_q6_top1_dp4a_row_gather",
                    "pack16": "draft_run_lm_head_q6_top1_dp4a_pack16_gather",
                    "x8": "draft_run_lm_head_q6_top1_dp4a_x8_gather",
                    "x8_dscale": "draft_run_lm_head_q6_top1_dp4a_x8_dscale_gather",
                    "pack8_llama": "draft_run_lm_head_q6_top1_dp4a_pack8_llama_gather",
                    "pack8_scalehoist": "draft_run_lm_head_q6_top1_dp4a_scalehoist_gather",
                }.get(q6_top1_stage1_shape, "draft_run_lm_head_q6_top1_dp4a_gather")
                if sync_stages:
                    if q6_top1_stage1_shape == "row":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    elif q6_top1_stage1_shape == "pack8_scalehoist":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    elif q6_top1_stage1_shape == "x8":
                        if self.shared_head_x8 is None:
                            raise RuntimeError("X8 Q6 top-1 sidecar was not materialized")
                        gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head_x8.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    elif q6_top1_stage1_shape == "x8_dscale":
                        if self.shared_head_x8 is None or self.shared_head_x8_dscale is None:
                            raise RuntimeError("X8 dscale Q6 top-1 sidecars were not materialized")
                        gguf_q6_k_x8_dscale_gemv_decode_q8_1_dp4a_top1_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head_x8.ptr,
                            self.shared_head_x8_dscale.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    elif q6_top1_stage1_shape == "pack16":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    elif q6_top1_stage1_shape == "pack8_llama":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    else:
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            1,
                            h,
                            self.vocab,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_stage_name, t_stage)
                    gguf_q6_k_pack8_top1_stage2_gather_f32(
                        self.q6_top1_block_values.ptr,
                        self.q6_top1_block_indices.ptr,
                        int(top1_out_ptr),
                        self.topk_values.ptr,
                        self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                        int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                        1,
                        q6_top1_stage2_blocks,
                        h if top1_next_embed_ptr is not None else 0,
                        self.vocab,
                        library=self._q6_pack8_lib,
                        runtime=runtime,
                    )
                    t_stage = mark_stage(q6_top1_stage2_name, t_stage)
                    _stage_add(
                        stage_timings,
                        q6_top1_gather_name,
                        (time.perf_counter() - t_q6_top1) * 1000,
                    )
                else:
                    if q6_top1_stage1_shape == "row":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
                    elif q6_top1_stage1_shape == "pack8_scalehoist":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
                    elif q6_top1_stage1_shape == "x8":
                        if self.shared_head_x8 is None:
                            raise RuntimeError("X8 Q6 top-1 sidecar was not materialized")
                        gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head_x8.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
                    elif q6_top1_stage1_shape == "x8_dscale":
                        if self.shared_head_x8 is None or self.shared_head_x8_dscale is None:
                            raise RuntimeError("X8 dscale Q6 top-1 sidecars were not materialized")
                        gguf_q6_k_x8_dscale_gemv_decode_q8_1_dp4a_top1_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head_x8.ptr,
                            self.shared_head_x8_dscale.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
                    elif q6_top1_stage1_shape == "pack16":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
                    elif q6_top1_stage1_shape == "pack8_llama":
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
                    else:
                        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32(
                            self.head_normed_q8_1.ptr,
                            self.shared_head.ptr,
                            self.q6_top1_block_values.ptr,
                            self.q6_top1_block_indices.ptr,
                            int(top1_out_ptr),
                            self.topk_values.ptr,
                            self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                            int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                            1,
                            h,
                            self.vocab,
                            h if top1_next_embed_ptr is not None else 0,
                            library=self._q6_pack8_lib,
                            runtime=runtime,
                        )
                        t_stage = mark_stage(q6_top1_gather_name, t_stage)
            else:
                t_q6_top1 = t_stage
                if sync_stages:
                    gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32(
                        self.head_normed_bf16.ptr,
                        self.shared_head.ptr,
                        self.q6_top1_block_values.ptr,
                        self.q6_top1_block_indices.ptr,
                        1,
                        h,
                        self.vocab,
                        library=self._q6_pack8_lib,
                        runtime=runtime,
                    )
                    t_stage = mark_stage("draft_run_lm_head_q6_top1_bf16_stage1", t_stage)
                    gguf_q6_k_pack8_top1_stage2_gather_f32(
                        self.q6_top1_block_values.ptr,
                        self.q6_top1_block_indices.ptr,
                        int(top1_out_ptr),
                        self.topk_values.ptr,
                        self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                        int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                        1,
                        self.vocab // 8,
                        h if top1_next_embed_ptr is not None else 0,
                        self.vocab,
                        library=self._q6_pack8_lib,
                        runtime=runtime,
                    )
                    t_stage = mark_stage("draft_run_lm_head_q6_top1_bf16_stage2_gather", t_stage)
                    _stage_add(
                        stage_timings,
                        "draft_run_lm_head_q6_top1_bf16_gather",
                        (time.perf_counter() - t_q6_top1) * 1000,
                    )
                else:
                    gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(
                        self.head_normed_bf16.ptr,
                        self.shared_head.ptr,
                        self.q6_top1_block_values.ptr,
                        self.q6_top1_block_indices.ptr,
                        int(top1_out_ptr),
                        self.topk_values.ptr,
                        self._embed_table_f32.ptr if top1_next_embed_ptr is not None else None,
                        int(top1_next_embed_ptr) if top1_next_embed_ptr is not None else None,
                        1,
                        h,
                        self.vocab,
                        h if top1_next_embed_ptr is not None else 0,
                        library=self._q6_pack8_lib,
                        runtime=runtime,
                    )
                    t_stage = mark_stage("draft_run_lm_head_q6_top1_bf16_gather", t_stage)
        else:
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
            t_stage = mark_stage("draft_run_lm_head_q6_full_logits", t_stage)
        if sync_stages:
            _stage_add(stage_timings, "draft_run_lm_head", (time.perf_counter() - t_lm_head0) * 1000)

    def _topk_indices_into(self, out_indices_ptr: int, top_k: int) -> None:
        """Write the top-``top_k`` logit indices to a device buffer (no sync)."""
        runtime = self.runtime or get_hip_runtime()
        if int(top_k) <= 8:
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
            return
        raise ValueError("device top-k is only supported for top_k <= 8")

    def _read_topk(self, top_k: int) -> list[int]:
        runtime = self.runtime or get_hip_runtime()
        if int(top_k) > 8:
            runtime.device_synchronize()
            logits = np.empty((self.vocab,), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(logits),
                DeviceBuffer(self.logits.ptr, logits.nbytes),
                logits.nbytes,
                runtime=runtime,
            )
            top_count = min(int(top_k), int(logits.shape[0]))
            top_idx = np.argpartition(logits, -top_count)[-top_count:]
            top_sorted = top_idx[np.argsort(logits[top_idx])[::-1]]
            return [int(token) for token in top_sorted.tolist()]
        self._topk_indices_into(self.topk_indices.ptr, top_k)
        runtime.device_synchronize()
        out = np.empty((int(top_k),), dtype=np.int32)
        copy_device_to_host(host_array_ptr(out), DeviceBuffer(self.topk_indices.ptr, out.nbytes), out.nbytes, runtime=runtime)
        return [int(token) for token in out.tolist()]

    def _read_topk_with_prob(self, top_k: int) -> tuple[list[int], float]:
        """Read full logits, return top-k ids and the top-1 softmax probability."""

        runtime = self.runtime or get_hip_runtime()
        runtime.device_synchronize()
        logits = np.empty((self.vocab,), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits),
            DeviceBuffer(self.logits.ptr, logits.nbytes),
            logits.nbytes,
            runtime=runtime,
        )
        top_count = min(int(top_k), int(logits.shape[0]))
        top_idx = np.argpartition(logits, -top_count)[-top_count:]
        top_sorted = top_idx[np.argsort(logits[top_idx])[::-1]]
        shifted = logits - float(logits[top_sorted[0]])
        exp = np.exp(shifted)
        top1_prob = float(exp[int(top_sorted[0])] / exp.sum())
        return [int(token) for token in top_sorted.tolist()], top1_prob


__all__ = ["Qwen35GGUFResidentMTPDraftRunner"]
