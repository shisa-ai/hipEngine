#!/usr/bin/env python3
"""Full-model DFlash chain E2E benchmark driver.

This is the first hipEngine runner that executes the real packed target model and
native DFlash drafter with a same-session AR control.  The verifier is
intentionally labelled ``serial_branch_state_copy``: it uses the resident target
model with per-slot state copies to verify a top-1 chain exactly before the
future bulk target verifier replaces it.  Rows from this mode are actual
full-model measurements, but they are not promotable unless the artifact says
the verifier is native bulk and the normal speed/correctness gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import DEFAULT_STABLE_PROMPT_FIXTURE, file_sha256, load_prompt_records
from hipengine.benchmark.speculative import DEFAULT_DFLASH_DRAFTER, DEFAULT_TARGET_MODEL, SpeculativeBenchmarkModels, build_speculative_artifact
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc, memory_stats, reset_memory_stats
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import hip_target_arch_environment
from hipengine.kernels.hip_gfx1100.convert import build_cast
from hipengine.kernels.hip_gfx1100.linear import build_lm_head, topk_f32_rows_i32
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import w8a16_linear_bf16_f32_out
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_drafter
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    dflash_add_bf16,
    dflash_concat_rows_bf16,
    dflash_concat_rows_f32,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_head_rmsnorm_rotary_f32,
    dflash_prepare_noise_inputs_bf16_i32,
    dflash_prepare_noise_inputs_f16_to_bf16_i32,
    dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
)
from hipengine.loading import load_weight_index
from hipengine.loading.dflash import load_dflash_drafter_bf16_weights, validate_dflash_artifact_pair
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_TARGET_PATH = "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
DEFAULT_DRAFTER_PATH = "/models/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719"
DEFAULT_TARGET_REVISION = "501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
DEFAULT_DRAFTER_REVISION = "42d3b34d588423cdae7ba8f53a8cf7789346a719"


@dataclass(frozen=True)
class DraftResult:
    candidate_tokens: tuple[int, ...]
    draft_seconds: float
    finite_logits: bool
    d2h_vector_reads: int
    d2h_vector_values: int
    phase_seconds: dict[str, float]


class NativeDFlashChainDrafter:
    """Correctness-first native DFlash top-1 chain proposer.

    The implementation rebuilds projected context and per-layer K/V from the
    current target-hidden context each draft call.  That keeps the first
    full-model E2E benchmark small and explicit; the append-only draft-KV owner
    remains the optimization target for promotable rows.
    """

    def __init__(
        self,
        *,
        session: Qwen35ParoResidentSession,
        drafter_model: str | Path,
        max_context_tokens: int,
        candidate_budget: int,
        compiler_version: str | None,
        require_cached_build: bool,
        sync_draft_phases: bool = False,
    ) -> None:
        self.session = session
        self.runtime = session.runtime
        self.device = Device("hip", 0)
        self.candidate_budget = int(candidate_budget)
        self.sync_draft_phases = bool(sync_draft_phases)
        self.drafter_index = load_weight_index(drafter_model)
        self.weights = load_dflash_drafter_bf16_weights(
            self.drafter_index,
            runtime=self.runtime,
            device=self.device,
            layer_limit=None,
        )
        self.config = self.weights.config
        if self.candidate_budget <= 0 or self.candidate_budget >= self.config.block_size:
            raise ValueError("candidate_budget must be in [1, block_size - 1]")
        self.max_context_tokens = int(max_context_tokens)
        self.hidden = int(self.config.hidden_size)
        self.intermediate = int(self.config.intermediate_size)
        self.q_heads = int(self.config.num_attention_heads)
        self.kv_heads = int(self.config.num_key_value_heads)
        self.head_dim = int(self.config.head_dim)
        self.attn_features = self.q_heads * self.head_dim
        self.kv_features = self.kv_heads * self.head_dim
        self.vocab_size = int(self.config.vocab_size)
        self.block_size = int(self.config.block_size)
        self.buffers: list[DeviceBuffer] = []
        with hip_target_arch_environment(session.target_arch):
            self.library = build_dflash_drafter(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
            self.lm_library = build_lm_head(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
            self.cast_library = build_cast(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
        self._allocate()

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        return self.config.target_layer_ids

    @property
    def target_hidden_concat(self) -> Tensor:
        return self._target_hidden_concat

    def close(self) -> None:
        self.weights.free(runtime=self.runtime)
        for buffer in reversed(self.buffers):
            free(buffer, runtime=self.runtime)
        self.buffers.clear()

    def __enter__(self) -> "NativeDFlashChainDrafter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def propose(self, *, root_token: int, root_position: int, context_tokens: int) -> DraftResult:
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive before DFlash draft")
        if context_tokens > self.max_context_tokens:
            raise ValueError("context_tokens exceeds DFlash context capacity")
        t0 = time.perf_counter()
        phases: dict[str, float] = {}
        phase_t = time.perf_counter()
        root = np.asarray([int(root_token)], dtype=np.int32)
        pos = np.asarray([int(root_position)], dtype=np.int32)
        copy_host_to_device(self._buffer_for(self.root_tokens), host_array_ptr(root), runtime=self.runtime)
        copy_host_to_device(self._buffer_for(self.root_positions), host_array_ptr(pos), runtime=self.runtime)
        prepare = (
            dflash_prepare_noise_inputs_bf16_i32
            if self.session.embedding.tensor.dtype == DType.BF16
            else dflash_prepare_noise_inputs_f16_to_bf16_i32
        )
        prepare(
            self.root_tokens.ptr,
            self.root_positions.ptr,
            self.session.embedding.tensor.ptr,
            self.noise_ids.ptr,
            self.query_positions.ptr,
            self.query_hidden_a.ptr,
            1,
            self.block_size,
            self.hidden,
            self.session.vocab_size,
            self.config.mask_token_id,
            threads=256,
            library=self.library,
            runtime=self.runtime,
        )
        self._record_phase(phases, "noise_prepare", phase_t)
        phase_t = time.perf_counter()
        self._write_key_positions(context_tokens, root_position)
        self._record_phase(phases, "key_positions_h2d", phase_t)
        phase_t = time.perf_counter()
        dflash_dense_bf16_to_bf16(
            self.target_hidden_concat.ptr,
            self.weights.tensor("fc.weight").ptr,
            self.projected_context.ptr,
            context_tokens,
            self.config.target_hidden_concat_size,
            self.hidden,
            threads=128,
            library=self.library,
            runtime=self.runtime,
        )
        dflash_rmsnorm_bf16(
            self.projected_context.ptr,
            self.weights.tensor("hidden_norm.weight").ptr,
            self.projected_context_norm.ptr,
            context_tokens,
            self.hidden,
            threads=128,
            library=self.library,
            runtime=self.runtime,
        )
        self._record_phase(phases, "context_projection", phase_t)
        query_in = self.query_hidden_a
        query_out = self.query_hidden_b
        layer_seconds: list[float] = []
        layers_t = time.perf_counter()
        for layer in range(self.config.num_hidden_layers):
            layer_t = time.perf_counter()
            query_out = self._run_layer(layer, context_tokens=context_tokens, query_in=query_in, query_out=query_out)
            query_in, query_out = query_out, query_in
            if self.sync_draft_phases:
                self.runtime.device_synchronize()
            layer_seconds.append(time.perf_counter() - layer_t)
        if self.sync_draft_phases:
            self.runtime.device_synchronize()
        phases["decoder_layers"] = time.perf_counter() - layers_t
        phases["slowest_decoder_layer"] = max(layer_seconds) if layer_seconds else 0.0
        phase_t = time.perf_counter()
        dflash_rmsnorm_bf16(
            query_in.ptr,
            self.weights.tensor("norm.weight").ptr,
            self.final_norm.ptr,
            self.block_size,
            self.hidden,
            threads=128,
            library=self.library,
            runtime=self.runtime,
        )
        self._record_phase(phases, "final_norm", phase_t)
        phase_t = time.perf_counter()
        logits_ptr = self.logits.ptr
        w8a16_linear_bf16_f32_out(
            self.final_norm.ptr + self.hidden * DType.BF16.itemsize,
            self.session.lm_head_weight.tensor.ptr,
            self.session.lm_head_scale.tensor.ptr,
            logits_ptr,
            self.candidate_budget,
            self.hidden,
            self.vocab_size,
            threads=128,
            library=self.session.libraries["w8a16"],
            runtime=self.runtime,
        )
        self._record_phase(phases, "lm_head", phase_t)
        phase_t = time.perf_counter()
        topk_f32_rows_i32(
            logits_ptr,
            self.top1_values.ptr,
            self.top1_ids.ptr,
            self.candidate_budget,
            self.vocab_size,
            1,
            threads=256,
            library=self.lm_library,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        top1 = np.empty((self.candidate_budget, 1), dtype=np.int32)
        top1_values = np.empty((self.candidate_budget, 1), dtype=np.float32)
        copy_device_to_host(host_array_ptr(top1), self._buffer_for(self.top1_ids), runtime=self.runtime)
        copy_device_to_host(host_array_ptr(top1_values), self._buffer_for(self.top1_values), runtime=self.runtime)
        phases["topk_and_readback"] = time.perf_counter() - phase_t
        draft_seconds = time.perf_counter() - t0
        phases["total"] = draft_seconds
        return DraftResult(
            candidate_tokens=tuple(int(x) for x in top1.reshape(-1).tolist()),
            draft_seconds=draft_seconds,
            finite_logits=bool(np.isfinite(top1_values).all()),
            d2h_vector_reads=2,
            d2h_vector_values=2 * self.candidate_budget,
            phase_seconds=phases,
        )

    def _record_phase(self, phases: dict[str, float], name: str, started_at: float) -> None:
        if self.sync_draft_phases:
            self.runtime.device_synchronize()
        phases[name] = time.perf_counter() - started_at

    def _allocate(self) -> None:
        self.root_tokens = self._empty((1,), DType.INT32)
        self.root_positions = self._empty((1,), DType.INT32)
        self.noise_ids = self._empty((1, self.block_size), DType.INT32)
        self.query_positions = self._empty((1, self.block_size), DType.INT32)
        self.key_positions = self._empty((1, self.max_context_tokens + self.block_size), DType.INT32)
        self._target_hidden_concat = self._empty((self.max_context_tokens, self.config.target_hidden_concat_size), DType.BF16)
        self.projected_context = self._empty((self.max_context_tokens, self.hidden), DType.BF16)
        self.projected_context_norm = self._empty((self.max_context_tokens, self.hidden), DType.BF16)
        self.query_hidden_a = self._empty((self.block_size, self.hidden), DType.BF16)
        self.query_hidden_b = self._empty((self.block_size, self.hidden), DType.BF16)
        self.norm = self._empty((self.block_size, self.hidden), DType.BF16)
        self.q_raw = self._empty((self.block_size, self.attn_features), DType.FP32)
        self.k_ctx = self._empty((self.max_context_tokens, self.kv_features), DType.FP32)
        self.k_q = self._empty((self.block_size, self.kv_features), DType.FP32)
        self.k_all = self._empty((1, self.max_context_tokens + self.block_size, self.kv_features), DType.FP32)
        self.v_ctx = self._empty((self.max_context_tokens, self.kv_features), DType.BF16)
        self.v_q = self._empty((self.block_size, self.kv_features), DType.BF16)
        self.v_all = self._empty((1, self.max_context_tokens + self.block_size, self.kv_features), DType.BF16)
        self.q_rot = self._empty((1, self.block_size, self.q_heads, self.head_dim), DType.FP32)
        self.k_rot = self._empty((1, self.max_context_tokens + self.block_size, self.kv_heads, self.head_dim), DType.FP32)
        self.attn = self._empty((1, self.block_size, self.q_heads, self.head_dim), DType.BF16)
        self.attn_proj = self._empty((self.block_size, self.hidden), DType.BF16)
        self.hidden_attn = self._empty((self.block_size, self.hidden), DType.BF16)
        self.post = self._empty((self.block_size, self.hidden), DType.BF16)
        self.gate = self._empty((self.block_size, self.intermediate), DType.BF16)
        self.up = self._empty((self.block_size, self.intermediate), DType.BF16)
        self.act = self._empty((self.block_size, self.intermediate), DType.BF16)
        self.mlp = self._empty((self.block_size, self.hidden), DType.BF16)
        self.final_norm = self._empty((self.block_size, self.hidden), DType.BF16)
        self.logits = self._empty((self.candidate_budget, self.vocab_size), DType.FP32)
        self.top1_values = self._empty((self.candidate_budget, 1), DType.FP32)
        self.top1_ids = self._empty((self.candidate_budget, 1), DType.INT32)
        cos, sin = _rotary_tables(self.max_context_tokens + self.block_size + 8, self.head_dim, theta=float(self.config.rope_theta))
        self.cos = self._load_array(cos, DType.FP32)
        self.sin = self._load_array(sin, DType.FP32)

    def _run_layer(self, layer: int, *, context_tokens: int, query_in: Tensor, query_out: Tensor) -> Tensor:
        prefix = f"layers.{layer}"
        total_kv = context_tokens + self.block_size
        dflash_rmsnorm_bf16(query_in.ptr, self.weights.tensor(f"{prefix}.input_layernorm.weight").ptr, self.norm.ptr, self.block_size, self.hidden, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.q_proj.weight").ptr, self.q_raw.ptr, self.block_size, self.hidden, self.attn_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.projected_context_norm.ptr, self.weights.tensor(f"{prefix}.self_attn.k_proj.weight").ptr, self.k_ctx.ptr, context_tokens, self.hidden, self.kv_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.k_proj.weight").ptr, self.k_q.ptr, self.block_size, self.hidden, self.kv_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.projected_context_norm.ptr, self.weights.tensor(f"{prefix}.self_attn.v_proj.weight").ptr, self.v_ctx.ptr, context_tokens, self.hidden, self.kv_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.v_proj.weight").ptr, self.v_q.ptr, self.block_size, self.hidden, self.kv_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_concat_rows_f32(self.k_ctx.ptr, self.k_q.ptr, self.k_all.ptr, 1, context_tokens, self.block_size, self.kv_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_concat_rows_bf16(self.v_ctx.ptr, self.v_q.ptr, self.v_all.ptr, 1, context_tokens, self.block_size, self.kv_features, threads=128, library=self.library, runtime=self.runtime)
        dflash_head_rmsnorm_rotary_f32(self.q_raw.ptr, self.k_all.ptr, self.weights.tensor(f"{prefix}.self_attn.q_norm.weight").ptr, self.weights.tensor(f"{prefix}.self_attn.k_norm.weight").ptr, self.cos.ptr, self.sin.ptr, self.query_positions.ptr, self.key_positions.ptr, self.q_rot.ptr, self.k_rot.ptr, 1, self.block_size, total_kv, self.q_heads, self.kv_heads, self.head_dim, self.head_dim, self.max_context_tokens + self.block_size + 8, threads=128, library=self.library, runtime=self.runtime)
        dflash_gqa_attention_f32_bf16(self.q_rot.ptr, self.k_rot.ptr, self.v_all.ptr, self.attn.ptr, 1, self.block_size, total_kv, self.q_heads, self.kv_heads, self.head_dim, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.attn.ptr, self.weights.tensor(f"{prefix}.self_attn.o_proj.weight").ptr, self.attn_proj.ptr, self.block_size, self.attn_features, self.hidden, threads=128, library=self.library, runtime=self.runtime)
        dflash_add_bf16(query_in.ptr, self.attn_proj.ptr, self.hidden_attn.ptr, self.block_size * self.hidden, threads=256, library=self.library, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.hidden_attn.ptr, self.weights.tensor(f"{prefix}.post_attention_layernorm.weight").ptr, self.post.ptr, self.block_size, self.hidden, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.post.ptr, self.weights.tensor(f"{prefix}.mlp.gate_proj.weight").ptr, self.gate.ptr, self.block_size, self.hidden, self.intermediate, threads=128, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.post.ptr, self.weights.tensor(f"{prefix}.mlp.up_proj.weight").ptr, self.up.ptr, self.block_size, self.hidden, self.intermediate, threads=128, library=self.library, runtime=self.runtime)
        dflash_silu_mul_bf16(self.gate.ptr, self.up.ptr, self.act.ptr, self.block_size * self.intermediate, threads=256, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.act.ptr, self.weights.tensor(f"{prefix}.mlp.down_proj.weight").ptr, self.mlp.ptr, self.block_size, self.intermediate, self.hidden, threads=128, library=self.library, runtime=self.runtime)
        dflash_add_bf16(self.hidden_attn.ptr, self.mlp.ptr, query_out.ptr, self.block_size * self.hidden, threads=256, library=self.library, runtime=self.runtime)
        return query_out

    def _write_key_positions(self, context_tokens: int, root_position: int) -> None:
        values = np.concatenate(
            [
                np.arange(context_tokens, dtype=np.int32),
                np.arange(root_position, root_position + self.block_size, dtype=np.int32),
            ]
        ).reshape(1, -1)
        copy_host_to_device(self._buffer_for(self.key_positions), host_array_ptr(values), values.nbytes, runtime=self.runtime)

    def _empty(self, shape: tuple[int, ...], dtype: DType) -> Tensor:
        nbytes = int(math.prod(shape)) * dtype.itemsize
        buf = malloc(nbytes, runtime=self.runtime)
        self.buffers.append(buf)
        return Tensor.from_handle(buf.ptr, shape, dtype, self.device)

    def _load_array(self, array: np.ndarray, dtype: DType) -> Tensor:
        tensor = self._empty(tuple(int(x) for x in array.shape), dtype)
        copy_host_to_device(self._buffer_for(tensor), host_array_ptr(np.ascontiguousarray(array)), runtime=self.runtime)
        return tensor

    def _buffer_for(self, tensor: Tensor) -> DeviceBuffer:
        for buffer in self.buffers:
            if buffer.ptr == tensor.ptr:
                return buffer
        raise KeyError(f"no owning buffer for tensor pointer 0x{tensor.ptr:x}")


def run_ar_tokens(
    *,
    model: Path,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    backend: str,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
) -> tuple[list[int], dict[str, Any]]:
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_ids) + decode_tokens + 1
    reset_memory_stats()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
    ) as session:
        t0 = time.perf_counter()
        next_result = None
        for pos, token in enumerate(prompt_ids):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_ids) - 1))
        if next_result is None:
            raise RuntimeError("AR prefill produced no token")
        prefill_seconds = time.perf_counter() - t0
        generated: list[int] = []
        next_token = int(next_result.token_id)
        t1 = time.perf_counter()
        finite = True
        for offset in range(decode_tokens):
            generated.append(next_token)
            result = session.step(next_token, position=len(prompt_ids) + offset, sample=True)
            if result is None:
                raise RuntimeError("AR decode step produced no token")
            finite = finite and math.isfinite(float(result.logit))
            next_token = int(result.token_id)
        decode_seconds = time.perf_counter() - t1
        memory = memory_stats()
        metadata = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "finite_logits": finite,
            "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
            "memory": memory,
            "backend": session.backend,
            "target_arch": session.target_arch,
        }
    return generated, metadata


def run_dflash_tokens(
    *,
    model: Path,
    drafter_model: Path,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
) -> tuple[list[int], dict[str, Any]]:
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_ids) + decode_tokens + candidate_budget + 2
    max_batch_size = candidate_budget + 2
    reset_memory_stats()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        max_batch_size=max_batch_size,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
    ) as session:
        with NativeDFlashChainDrafter(
            session=session,
            drafter_model=drafter_model,
            max_context_tokens=max_sequence,
            candidate_budget=candidate_budget,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
        ) as drafter:
            t0 = time.perf_counter()
            next_result = None
            for pos, token in enumerate(prompt_ids):
                next_result = session.step_with_hidden_taps(
                    int(token),
                    position=pos,
                    capture_layer_ids=drafter.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row=pos,
                    sample=(pos == len(prompt_ids) - 1),
                )
            if next_result is None:
                raise RuntimeError("DFlash prefill produced no root token")
            prefill_seconds = time.perf_counter() - t0
            root_token = int(next_result.token_id)
            context_tokens = len(prompt_ids)
            generated: list[int] = []
            accepted_lengths: list[int] = []
            draft_seconds_total = 0.0
            verify_seconds_total = 0.0
            commit_seconds_total = 0.0
            d2h_vector_reads = 0
            d2h_vector_values = 0
            cycles = 0
            verify_rows_total = 0
            finite = True
            t1 = time.perf_counter()
            while len(generated) < decode_tokens:
                cycles += 1
                remaining = decode_tokens - len(generated)
                active_budget = min(candidate_budget, max(0, remaining - 1))
                if active_budget <= 0:
                    verify_rows_total += 1
                    # Need only the current root output; advance target once to seed the next root.
                    t_verify = time.perf_counter()
                    session.copy_slot_state(0, 1)
                    result = _slot_step(
                        session,
                        root_token,
                        position=context_tokens,
                        slot=1,
                        drafter=drafter,
                        capture_row=context_tokens,
                    )
                    verify_seconds_total += time.perf_counter() - t_verify
                    t_commit = time.perf_counter()
                    session.copy_slot_state(1, 0)
                    commit_seconds_total += time.perf_counter() - t_commit
                    generated.append(root_token)
                    root_token = int(result.token_id)
                    context_tokens += 1
                    continue
                verify_rows_total += 1 + active_budget
                draft = drafter.propose(root_token=root_token, root_position=context_tokens, context_tokens=context_tokens)
                candidates = list(draft.candidate_tokens[:active_budget])
                draft_seconds_total += draft.draft_seconds
                d2h_vector_reads += draft.d2h_vector_reads
                d2h_vector_values += draft.d2h_vector_values
                t_verify = time.perf_counter()
                session.copy_slot_state(0, 1)
                parent_result = _slot_step(
                    session,
                    root_token,
                    position=context_tokens,
                    slot=1,
                    drafter=drafter,
                    capture_row=context_tokens,
                )
                target_top1 = [int(parent_result.token_id)]
                accepted = 0
                selected_slot = 1
                bonus = int(parent_result.token_id)
                finite = finite and math.isfinite(float(parent_result.logit))
                for idx, cand in enumerate(candidates):
                    if target_top1[-1] != int(cand):
                        bonus = target_top1[-1]
                        break
                    accepted += 1
                    parent_slot = idx + 1
                    child_slot = idx + 2
                    session.copy_slot_state(parent_slot, child_slot)
                    result = _slot_step(
                        session,
                        int(cand),
                        position=context_tokens + idx + 1,
                        slot=child_slot,
                        drafter=drafter,
                        capture_row=context_tokens + idx + 1,
                    )
                    finite = finite and math.isfinite(float(result.logit))
                    target_top1.append(int(result.token_id))
                    selected_slot = child_slot
                    bonus = int(result.token_id)
                verify_seconds_total += time.perf_counter() - t_verify
                accepted_lengths.append(accepted)
                t_commit = time.perf_counter()
                session.copy_slot_state(selected_slot, 0)
                committed = [root_token, *candidates[:accepted]]
                commit_seconds_total += time.perf_counter() - t_commit
                generated.extend(committed)
                root_token = int(bonus)
                context_tokens += len(committed)
            decode_seconds = time.perf_counter() - t1
            memory = memory_stats()
            metadata = {
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "draft_seconds": draft_seconds_total,
                "target_verify_seconds": verify_seconds_total,
                "commit_seconds": commit_seconds_total,
                "accepted_lengths": accepted_lengths,
                "target_verify_rows": verify_rows_total,
                "draft_calls": cycles,
                "finite_draft_logits": finite,
                "finite_verify_logits": finite,
                "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
                "d2h": {"scalar_reads": cycles, "vector_reads": d2h_vector_reads, "scalar_values": cycles, "vector_values": d2h_vector_values, "full_logits_readbacks": 0},
                "memory": memory,
                "backend": session.backend,
                "target_arch": session.target_arch,
                "verifier_mode": "serial_branch_state_copy",
                "native_bulk_verifier": False,
                "drafter_context_mode": "full_context_rebuild_per_cycle",
            }
    return generated[:decode_tokens], metadata


def run_same_session_pair(
    *,
    model: Path,
    drafter_model: Path,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    sync_draft_phases: bool = False,
) -> tuple[tuple[list[int], dict[str, Any]], tuple[list[int], dict[str, Any]]]:
    """Run AR control and DFlash chain in one resident target session.

    Slot 0 is reserved for the AR control.  Slot 1 is the DFlash committed
    state, and slots 2..N are serial branch-verifier scratch slots.  This keeps
    the target weights/libraries/session identical while preserving independent
    per-slot recurrent/KV state for exact token comparison.
    """

    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_ids) + decode_tokens + candidate_budget + 2
    max_batch_size = max(4, candidate_budget + 3)
    reset_memory_stats()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        max_batch_size=max_batch_size,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
    ) as session:
        t0 = time.perf_counter()
        next_result = None
        for pos, token in enumerate(prompt_ids):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_ids) - 1))
        if next_result is None:
            raise RuntimeError("same-session AR prefill produced no token")
        ar_prefill_seconds = time.perf_counter() - t0
        ar_generated: list[int] = []
        next_token = int(next_result.token_id)
        ar_finite = True
        t1 = time.perf_counter()
        for offset in range(decode_tokens):
            ar_generated.append(next_token)
            result = session.step(next_token, position=len(prompt_ids) + offset, sample=True)
            if result is None:
                raise RuntimeError("same-session AR decode step produced no token")
            ar_finite = ar_finite and math.isfinite(float(result.logit))
            next_token = int(result.token_id)
        ar_decode_seconds = time.perf_counter() - t1
        ar_meta = {
            "prefill_seconds": ar_prefill_seconds,
            "decode_seconds": ar_decode_seconds,
            "finite_logits": ar_finite,
            "decode_tok_s": decode_tokens / ar_decode_seconds if ar_decode_seconds > 0 else None,
            "memory": memory_stats(),
            "backend": session.backend,
            "target_arch": session.target_arch,
            "same_session_control": True,
            "same_process_control": True,
            "control_slot": 0,
        }
        with NativeDFlashChainDrafter(
            session=session,
            drafter_model=drafter_model,
            max_context_tokens=max_sequence,
            candidate_budget=candidate_budget,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            sync_draft_phases=sync_draft_phases,
        ) as drafter:
            spec_tokens, spec_meta = _run_dflash_chain_on_session(
                session=session,
                drafter=drafter,
                prompt_ids=prompt_ids,
                decode_tokens=decode_tokens,
                candidate_budget=candidate_budget,
                base_slot=1,
                branch_slot_start=2,
            )
        spec_meta["same_session_control"] = True
        spec_meta["same_process_control"] = True
        return (ar_generated, ar_meta), (spec_tokens, spec_meta)


def _run_dflash_chain_on_session(
    *,
    session: Qwen35ParoResidentSession,
    drafter: NativeDFlashChainDrafter,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    candidate_budget: int,
    base_slot: int,
    branch_slot_start: int,
) -> tuple[list[int], dict[str, Any]]:
    t0 = time.perf_counter()
    next_result = None
    for pos, token in enumerate(prompt_ids):
        next_result = _slot_step(
            session,
            int(token),
            position=pos,
            slot=base_slot,
            drafter=drafter,
            capture_row=pos,
            sample=(pos == len(prompt_ids) - 1),
        )
    if next_result is None:
        raise RuntimeError("DFlash prefill produced no root token")
    prefill_seconds = time.perf_counter() - t0
    root_token = int(next_result.token_id)
    context_tokens = len(prompt_ids)
    generated: list[int] = []
    accepted_lengths: list[int] = []
    draft_seconds_total = 0.0
    verify_seconds_total = 0.0
    commit_seconds_total = 0.0
    d2h_vector_reads = 0
    d2h_vector_values = 0
    cycles = 0
    draft_calls = 0
    draft_tokens_proposed = 0
    verify_rows_total = 0
    draft_phase_seconds: dict[str, float] = {}
    proposal_trace: list[dict[str, Any]] = []
    finite_draft = True
    finite_verify = True
    t1 = time.perf_counter()
    while len(generated) < decode_tokens:
        cycles += 1
        remaining = decode_tokens - len(generated)
        active_budget = min(candidate_budget, max(0, remaining - 1))
        if active_budget <= 0:
            verify_rows_total += 1
            t_verify = time.perf_counter()
            session.copy_slot_state(base_slot, branch_slot_start)
            result = _slot_step(
                session,
                root_token,
                position=context_tokens,
                slot=branch_slot_start,
                drafter=drafter,
                capture_row=context_tokens,
            )
            verify_seconds_total += time.perf_counter() - t_verify
            finite_verify = finite_verify and math.isfinite(float(result.logit))
            t_commit = time.perf_counter()
            session.copy_slot_state(branch_slot_start, base_slot)
            commit_seconds_total += time.perf_counter() - t_commit
            generated.append(root_token)
            root_token = int(result.token_id)
            context_tokens += 1
            continue
        verify_rows_total += 1 + active_budget
        draft = drafter.propose(root_token=root_token, root_position=context_tokens, context_tokens=context_tokens)
        candidates = list(draft.candidate_tokens[:active_budget])
        draft_calls += 1
        draft_tokens_proposed += active_budget
        draft_seconds_total += draft.draft_seconds
        for phase_name, phase_seconds in draft.phase_seconds.items():
            value = float(phase_seconds)
            if phase_name == "slowest_decoder_layer":
                draft_phase_seconds[phase_name] = max(draft_phase_seconds.get(phase_name, 0.0), value)
            else:
                draft_phase_seconds[phase_name] = draft_phase_seconds.get(phase_name, 0.0) + value
        finite_draft = finite_draft and draft.finite_logits
        d2h_vector_reads += draft.d2h_vector_reads
        d2h_vector_values += draft.d2h_vector_values
        t_verify = time.perf_counter()
        session.copy_slot_state(base_slot, branch_slot_start)
        parent_result = _slot_step(
            session,
            root_token,
            position=context_tokens,
            slot=branch_slot_start,
            drafter=drafter,
            capture_row=context_tokens,
        )
        target_top1 = [int(parent_result.token_id)]
        accepted = 0
        selected_slot = branch_slot_start
        bonus = int(parent_result.token_id)
        finite_verify = finite_verify and math.isfinite(float(parent_result.logit))
        for idx, cand in enumerate(candidates):
            if target_top1[-1] != int(cand):
                bonus = target_top1[-1]
                break
            accepted += 1
            parent_slot = branch_slot_start + idx
            child_slot = branch_slot_start + idx + 1
            session.copy_slot_state(parent_slot, child_slot)
            result = _slot_step(
                session,
                int(cand),
                position=context_tokens + idx + 1,
                slot=child_slot,
                drafter=drafter,
                capture_row=context_tokens + idx + 1,
            )
            finite_verify = finite_verify and math.isfinite(float(result.logit))
            target_top1.append(int(result.token_id))
            selected_slot = child_slot
            bonus = int(result.token_id)
        verify_seconds_total += time.perf_counter() - t_verify
        accepted_lengths.append(accepted)
        t_commit = time.perf_counter()
        session.copy_slot_state(selected_slot, base_slot)
        committed = [root_token, *candidates[:accepted]]
        if len(proposal_trace) < 16:
            proposal_trace.append(
                {
                    "cycle": cycles,
                    "root_position": context_tokens,
                    "root_token": int(root_token),
                    "draft_candidates": [int(token) for token in candidates],
                    "target_top1_path": [int(token) for token in target_top1],
                    "accepted": int(accepted),
                    "committed_tokens": [int(token) for token in committed],
                    "bonus_token": int(bonus),
                }
            )
        commit_seconds_total += time.perf_counter() - t_commit
        generated.extend(committed)
        root_token = int(bonus)
        context_tokens += len(committed)
    decode_seconds = time.perf_counter() - t1
    metadata = {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "draft_seconds": draft_seconds_total,
        "target_verify_seconds": verify_seconds_total,
        "commit_seconds": commit_seconds_total,
        "accepted_lengths": accepted_lengths,
        "target_verify_rows": verify_rows_total,
        "draft_calls": draft_calls,
        "decode_cycles": cycles,
        "draft_tokens_proposed": draft_tokens_proposed,
        "draft_native_phase_seconds": draft_phase_seconds,
        "proposal_trace_sample": proposal_trace,
        "proposal_trace_count": draft_calls,
        "finite_draft_logits": finite_draft,
        "finite_verify_logits": finite_verify,
        "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
        "d2h": {
            "scalar_reads": verify_rows_total,
            "vector_reads": d2h_vector_reads,
            "scalar_values": verify_rows_total,
            "vector_values": d2h_vector_values,
            "full_logits_readbacks": 0,
            "notes": ["draft finite check reads top-1 values only; full logits are not copied"],
        },
        "memory": memory_stats(),
        "backend": session.backend,
        "target_arch": session.target_arch,
        "verifier_mode": "serial_branch_state_copy",
        "native_bulk_verifier": False,
        "drafter_context_mode": "full_context_rebuild_per_cycle",
        "draft_phase_timing_mode": "synchronized" if drafter.sync_draft_phases else "enqueue_until_final_sync",
        "base_slot": base_slot,
        "branch_slot_start": branch_slot_start,
    }
    return generated[:decode_tokens], metadata


def _slot_step(
    session: Qwen35ParoResidentSession,
    token_id: int,
    *,
    position: int,
    slot: int,
    drafter: NativeDFlashChainDrafter | None = None,
    capture_row: int | None = None,
    sample: bool = True,
):
    session._set_slot_token_embedding(int(token_id), slot=slot)
    session._set_slot_position(int(position), slot=slot)
    kwargs: dict[str, Any] = {}
    if drafter is not None:
        if capture_row is None:
            raise ValueError("capture_row is required with drafter")
        kwargs = {
            "capture_layer_ids": drafter.target_layer_ids,
            "capture_hidden_concat": drafter.target_hidden_concat,
            "capture_row": int(capture_row),
        }
    hidden = session._run_layers(position=int(position), slot=slot, persist_aliases=False, stream=0, **kwargs)
    if not sample:
        return None
    return session._sample_from_hidden(hidden)


def _rotary_tables(max_positions: int, head_dim: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(head_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(theta), -2.0 * dims / np.float32(head_dim))
    angles = positions * inv_freq
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    return np.concatenate([cos_half, cos_half], axis=1), np.concatenate([sin_half, sin_half], axis=1)


def _row_for_artifact(prompt: dict[str, Any], budget: int, ar: tuple[list[int], dict[str, Any]], spec: tuple[list[int], dict[str, Any]]) -> dict[str, Any]:
    ar_tokens, ar_meta = ar
    spec_tokens, spec_meta = spec
    return {
        "prompt": {
            "id": prompt.get("id"),
            "dataset": prompt.get("dataset"),
            "category": prompt.get("benchmark_group"),
            "prompt_tokens": prompt.get("prompt_tokens"),
            "prompt_ids_sha256": prompt.get("prompt_ids_sha256"),
            "prompt_text_sha256": prompt.get("prompt_text_sha256"),
            "prompt_preview": prompt.get("prompt_preview"),
            "representative": bool(prompt.get("representative")),
        },
        "config": {"name": f"full_model_chain_b{budget}", "provider": "dflash", "proposal_mode": "chain", "verify_mode": "verify_chain", "draft_budget": budget, "topk": 1},
        "ar": {
            "same_session_control": bool(ar_meta.get("same_session_control", False)),
            "same_process_control": bool(ar_meta.get("same_process_control", True)),
            "decode_seconds": ar_meta["decode_seconds"],
            "finite_logits": ar_meta["finite_logits"],
            "generated_ids": ar_tokens,
        },
        "spec": {
            "decode_seconds": spec_meta["decode_seconds"],
            "draft_seconds": spec_meta["draft_seconds"],
            "draft_context_full_rebuild_seconds": spec_meta["draft_seconds"],
            "draft_context_append_seconds": 0.0,
            "draft_query_seconds": spec_meta["draft_seconds"],
            "draft_native_phase_seconds": spec_meta.get("draft_native_phase_seconds", {}),
            "drafter_context_mode": spec_meta.get("drafter_context_mode"),
            "draft_phase_timing_mode": spec_meta.get("draft_phase_timing_mode"),
            "proposal_trace_sample": spec_meta.get("proposal_trace_sample", []),
            "proposal_trace_count": spec_meta.get("proposal_trace_count", spec_meta["draft_calls"]),
            "target_verify_seconds": spec_meta["target_verify_seconds"],
            "commit_seconds": spec_meta["commit_seconds"],
            "target_verify_rows": spec_meta["target_verify_rows"],
            "draft_tokens_proposed": spec_meta.get("draft_tokens_proposed", spec_meta["draft_calls"] * budget),
            "draft_tokens": spec_meta.get("draft_tokens_proposed", spec_meta["draft_calls"] * budget),
            "accepted_draft_tokens": sum(int(x) for x in spec_meta["accepted_lengths"]),
            "accepted_lengths": spec_meta["accepted_lengths"],
            "draft_calls": spec_meta["draft_calls"],
            "finite_draft_logits": spec_meta["finite_draft_logits"],
            "finite_verify_logits": spec_meta["finite_verify_logits"],
            "generated_ids": spec_tokens,
            "d2h": spec_meta["d2h"],
            "graph": {"status": "not_captured", "replay_steps": 0, "bucket_key": {"mode": "verify_chain", "draft_budget": budget, "verifier": spec_meta["verifier_mode"]}, "validation_passed": None},
            "verifier_mode": spec_meta["verifier_mode"],
            "native_bulk_verifier": spec_meta["native_bulk_verifier"],
            "same_session_control": bool(spec_meta.get("same_session_control", False)),
            "same_process_control": bool(spec_meta.get("same_process_control", True)),
            "backend": spec_meta.get("backend"),
            "target_arch": spec_meta.get("target_arch"),
        },
        "quality_gate": {"exact_match_ar": ar_tokens == spec_tokens, "finite_ar_logits": ar_meta["finite_logits"], "finite_dflash_draft_logits": spec_meta["finite_draft_logits"], "finite_dflash_verify_logits": spec_meta["finite_verify_logits"]},
        "memory": {"peak_allocated_bytes": spec_meta["memory"].get("peak_allocated_bytes", 0), "peak_reserved_bytes": 0, "hip_used_peak_sampled_bytes": 0},
        "decode_tokens": len(ar_tokens),
    }


def _select_prompts(path: Path, *, groups: set[str], limit: int) -> list[dict[str, Any]]:
    rows = [row for row in load_prompt_records(path) if str(row.get("benchmark_group")) in groups]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no prompt rows selected")
    for row in rows:
        if not row.get("prompt_ids"):
            raise ValueError(f"prompt row {row.get('id')} lacks prompt_ids")
    return rows


def _git_context() -> dict[str, Any]:
    def run(cmd: list[str]) -> str | None:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None
    status = run(["git", "status", "--porcelain"])
    return {"hipengine_commit": run(["git", "rev-parse", "HEAD"]), "hipengine_branch": run(["git", "branch", "--show-current"]), "hipengine_dirty": bool(status), "hipengine_status_porcelain": status}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_PATH)
    parser.add_argument("--drafter-model", default=DEFAULT_DRAFTER_PATH)
    parser.add_argument("--prompt-fixture", type=Path, default=DEFAULT_STABLE_PROMPT_FIXTURE)
    parser.add_argument("--prompt-groups", default="code_promotion,robustness")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--draft-budgets", default="4")
    parser.add_argument("--backend", default="auto", choices=("auto", "hip_gfx1100", "hip_gfx1151"))
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--sync-draft-phases", action="store_true", help="Diagnostic only: synchronize after major drafter phases before timing them")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    target = Path(args.target_model)
    drafter = Path(args.drafter_model)
    validation = validate_dflash_artifact_pair(target_model=target, drafter_model=drafter, raise_on_error=True)
    prompts = _select_prompts(args.prompt_fixture, groups={x.strip() for x in args.prompt_groups.split(",") if x.strip()}, limit=args.max_prompts)
    budgets = [int(x) for x in args.draft_budgets.split(",") if x.strip()]
    prefill_config = PrefillConfig(auto_tune_chunk_sizes=True)
    rows: list[dict[str, Any]] = []
    commands = {"benchmark": " ".join(shlex.quote(part) for part in ["python3", "scripts/dflash_chain_e2e_bench.py", *(argv if argv is not None else sys.argv[1:])])}
    for prompt in prompts:
        prompt_ids = [int(x) for x in prompt["prompt_ids"]]
        for budget in budgets:
            ar, spec = run_same_session_pair(
                model=target,
                drafter_model=drafter,
                prompt_ids=prompt_ids,
                decode_tokens=args.decode_tokens,
                candidate_budget=budget,
                backend=args.backend,
                max_layers=args.max_layers,
                compiler_version=compiler_version,
                require_cached_build=args.require_cached_build,
                prefill_config=prefill_config,
                sync_draft_phases=args.sync_draft_phases,
            )
            rows.append(_row_for_artifact(prompt, budget, ar, spec))
    artifact = build_speculative_artifact(
        run_tag="dflash-chain-full-model-e2e",
        summary="Full-model hipEngine DFlash chain E2E run with same-session AR control, native drafter, and serial branch target verifier",
        rows=rows,
        models=SpeculativeBenchmarkModels(
            target_name=DEFAULT_TARGET_MODEL,
            target_path=str(target),
            target_revision=DEFAULT_TARGET_REVISION,
            drafter_name=DEFAULT_DFLASH_DRAFTER,
            drafter_path=str(drafter),
            drafter_revision=DEFAULT_DRAFTER_REVISION,
        ),
        status="diagnostic",
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware={"backend": rows[0]["spec"].get("backend") if rows else args.backend, "arch": rows[0]["spec"].get("target_arch") if rows else None, "gpu": None},
        software={**_git_context(), "python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        workload={
            "shape": "full_model_dflash_chain_e2e",
            "provider": "dflash",
            "verify_modes": ["verify_chain"],
            "draft_budgets": budgets,
            "decode_tokens": args.decode_tokens,
            "prompt_suite": str(args.prompt_fixture),
            "prompt_suite_sha256": file_sha256(args.prompt_fixture),
            "artifact_validation": validation,
            "verifier_mode": "serial_branch_state_copy",
            "native_bulk_verifier": False,
            "promotion_blocker": "serial branch verifier copies full per-slot state/KV; native bulk target verifier is required before promotion",
        },
        commands=commands,
        notes=[
            "Actual full-model target and native DFlash drafter execution with same-session AR control; diagnostic until native bulk verifier replaces serial branch state copies.",
            "Prompt fixture includes code/general/multilingual categories via fixtures/dflash/stable_prompts.jsonl.",
        ],
        decision_reason="full-model diagnostic only: serial_branch_state_copy verifier is not the promotable native bulk verifier",
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "all_correctness_passed": artifact["measurements"]["aggregate"]["all_correctness_passed"], "speedup_vs_ar": artifact["measurements"]["aggregate"].get("speedup_vs_ar"), "performance_claim": artifact["performance_claim"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
