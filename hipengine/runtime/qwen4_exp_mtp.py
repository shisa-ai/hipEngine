"""Correctness-first one-layer Qwen4Exp MTP sidecar runner.

The draft consumes a token ``x[p+1]`` together with the target's authoritative
widened hidden row ``h[p]``.  Token embedding and hidden are normalized
independently, fused through the sidecar's ``eh_proj``, then executed by one
Qwen4Exp hyper-connection + dense-attention + MoE block.  The draft owns its
K/V/index cursor independently of the target and returns its widened pre-final
hidden row for intra-cycle chaining.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
import time
from typing import Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.registry import resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFConfig
from hipengine.loading.qwen4_exp_mtp_materialize import Qwen4ExpMTPResidentWeights
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_BF16,
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDenseAttentionState,
    Qwen4ExpGRDeviceWeights,
    Qwen4ExpGRScratch,
    Qwen4ExpQSAIndexDeviceState,
    Qwen4ExpQSALayerDeviceWeights,
    Qwen4ExpQSALayerScratch,
    Qwen4ExpQSAMixerDeviceWeights,
    run_qwen4_exp_dense_qsa_layer,
    run_qwen4_exp_gr_read,
)


@dataclass(frozen=True)
class Qwen4ExpMTPDraftResult:
    token_id: int
    logits: np.ndarray | None
    hidden_seed: np.ndarray | None


@dataclass(frozen=True)
class Qwen4ExpMTPDraftSnapshot:
    position: int


class Qwen4ExpGGUFMTPDraftRunner:
    """One request-owned dense MTP draft at the 512/1K bring-up scope."""

    def __init__(
        self,
        resident: Qwen4ExpMTPResidentWeights,
        *,
        target_config: Qwen4ExpGGUFConfig,
        max_sequence_length: int = 1_024,
        backend: str = "hip_gfx1151",
        runtime: HipRuntime | None = None,
    ) -> None:
        self.resident = resident
        self.config = target_config
        self.backend = str(backend)
        self.runtime = runtime or resident.runtime or get_hip_runtime()
        self.max_sequence_length = int(max_sequence_length)
        if not 0 < self.max_sequence_length <= 1_024:
            raise ValueError("Qwen4Exp MTP bring-up capacity must be in 1..1024")
        sidecar = resident.plan.config
        for field in (
            "hidden_size",
            "residual_branch_count",
            "residual_low_rank",
            "attention_head_count",
            "attention_kv_head_count",
            "attention_key_length",
            "indexer_head_count",
            "indexer_key_length",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "shared_expert_feed_forward_length",
        ):
            if int(getattr(sidecar, field)) != int(getattr(target_config, field)):
                raise ValueError(f"MTP sidecar {field} does not match target")
        load_backend_kernel_package(self.backend)
        self.binding = self._bind_layer()
        self.attention = Qwen4ExpDenseAttentionState.allocate(
            max_positions=self.max_sequence_length,
            block_size=256,
            kv_heads=self.config.attention_kv_head_count,
            head_dim=self.config.attention_key_length,
            runtime=self.runtime,
        )
        self.index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=self.attention,
            index_heads=self.config.indexer_head_count,
            index_dim=self.config.indexer_key_length,
            compression_ratio=self.config.qsa_compression_ratio,
            block_budget=self.config.qsa_block_budget,
            runtime=self.runtime,
        )
        self.layer_scratch = Qwen4ExpQSALayerScratch.allocate(
            rows=1,
            branches=self.config.residual_branch_count,
            hidden=self.config.hidden_size,
            low_rank=self.config.residual_low_rank,
            query_heads=self.config.attention_head_count,
            kv_heads=self.config.attention_kv_head_count,
            head_dim=self.config.attention_key_length,
            ffn=self.config.expert_feed_forward_length,
            experts=self.config.expert_count,
            top_k=self.config.expert_used_count,
            index_heads=self.config.indexer_head_count,
            index_dim=self.config.indexer_key_length,
            runtime=self.runtime,
        )
        self.head_scratch = Qwen4ExpGRScratch.allocate(
            rows=1,
            branches=self.config.residual_branch_count,
            hidden=self.config.hidden_size,
            low_rank=self.config.residual_low_rank,
            runtime=self.runtime,
        )
        self._buffers: list[DeviceBuffer] = []
        self.position = 0
        self.closed = False
        self.last_forward_stage_timings_ms: dict[str, float] = {}
        self.last_proposal_stage_timings_ms: dict[str, float] = {}
        try:
            self._allocate_buffers()
        except Exception:
            self.close()
            raise

    def _allocate_buffers(self) -> None:
        hidden = self.config.hidden_size
        residual = self.config.residual_width
        argmax_blocks = lm_head_argmax_stage1_blocks(
            self.config.vocab_size, threads=256
        )
        sizes = (
            DType.INT64.itemsize,
            hidden * DType.BF16.itemsize,
            hidden * DType.BF16.itemsize,
            residual * DType.BF16.itemsize,
            self.config.residual_branch_count * 2 * hidden * DType.BF16.itemsize,
            residual * DType.BF16.itemsize,
            residual * DType.BF16.itemsize,
            self.config.vocab_size * DType.FP32.itemsize,
            argmax_blocks * DType.FP32.itemsize,
            argmax_blocks * DType.INT64.itemsize,
            DType.FP32.itemsize,
            4 * DType.INT64.itemsize,
        )
        self._buffers.extend(malloc(size, runtime=self.runtime) for size in sizes)

    @property
    def token_id_buffer(self) -> DeviceBuffer:
        return self._buffers[0]

    @property
    def embedding(self) -> DeviceBuffer:
        return self._buffers[1]

    @property
    def embedding_norm(self) -> DeviceBuffer:
        return self._buffers[2]

    @property
    def hidden_norm(self) -> DeviceBuffer:
        return self._buffers[3]

    @property
    def fused_input(self) -> DeviceBuffer:
        return self._buffers[4]

    @property
    def residual(self) -> DeviceBuffer:
        return self._buffers[5]

    @property
    def last_hidden(self) -> DeviceBuffer:
        return self._buffers[6]

    @property
    def logits_buffer(self) -> DeviceBuffer:
        return self._buffers[7]

    @property
    def argmax_block_values(self) -> DeviceBuffer:
        return self._buffers[8]

    @property
    def argmax_block_indices(self) -> DeviceBuffer:
        return self._buffers[9]

    @property
    def argmax_value(self) -> DeviceBuffer:
        return self._buffers[10]

    @property
    def candidate_packet(self) -> DeviceBuffer:
        return self._buffers[11]

    def _bind_layer(self) -> Qwen4ExpQSALayerDeviceWeights:
        def weight(slot: str):
            return self.resident.weight(f"layers.0.{slot}")

        def pointer(slot: str) -> int:
            return int(weight(slot).allocation("raw").tensor.ptr)

        def gr(prefix: str) -> Qwen4ExpGRDeviceWeights:
            return Qwen4ExpGRDeviceWeights(
                norm_weight_ptr=pointer(f"hc_{prefix}_norm"),
                down=weight(f"hc_{prefix}_down"),
                up=weight(f"hc_{prefix}_up"),
                inject=weight(f"hc_{prefix}_inject"),
            )

        projections = (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "index_q",
            "index_k",
        )
        moe_slots = {
            "router": "router",
            "expert_gate": "expert_gate",
            "expert_up": "expert_up",
            "expert_down": "expert_down",
            "shared_gate": "shared_gate",
            "shared_up": "shared_up",
            "shared_down": "shared_down",
            "shared_gate_weight": "shared_expert_gate",
        }
        return Qwen4ExpQSALayerDeviceWeights(
            attention_gr=gr("attn"),
            mixer=Qwen4ExpQSAMixerDeviceWeights(
                projections=MappingProxyType(
                    {slot: weight(slot) for slot in projections}
                ),
                q_norm_weight_ptr=pointer("attn_q_norm"),
                k_norm_weight_ptr=pointer("attn_k_norm"),
                index_q_norm_weight_ptr=pointer("index_q_norm"),
                index_k_norm_weight_ptr=pointer("index_k_norm"),
            ),
            ffn_gr=gr("ffn"),
            moe=MappingProxyType(
                {target: weight(source) for target, source in moe_slots.items()}
            ),
            layer_id=0,
            layer_type="qsa",
            qsa_state_index=0,
        )

    def reset(self) -> None:
        self._require_open()
        self.runtime.memset(
            self.attention.key_cache.ptr, 0, self.attention.key_cache.nbytes
        )
        self.runtime.memset(
            self.attention.value_cache.ptr, 0, self.attention.value_cache.nbytes
        )
        self.attention.set_position(0)
        self.index.reset()
        self.position = 0

    def snapshot(self) -> Qwen4ExpMTPDraftSnapshot:
        self._require_open()
        return Qwen4ExpMTPDraftSnapshot(self.position)

    def restore(self, snapshot: Qwen4ExpMTPDraftSnapshot) -> None:
        self.trim(snapshot.position)

    def trim(self, position: int) -> None:
        self._require_open()
        count = int(position)
        if count < 0 or count > self.position:
            raise ValueError("Qwen4Exp MTP trim must target the committed prefix")
        if count == 0:
            self.attention.set_position(0)
        else:
            self.attention.set_position(count - 1)
        self.index.restore_count(count)
        self.position = count

    def _upload_hidden_seed(self, hidden_seed: np.ndarray) -> None:
        values = np.ascontiguousarray(hidden_seed, dtype=np.float32).reshape(-1)
        if values.shape != (self.config.residual_width,):
            raise ValueError(
                "Qwen4Exp MTP hidden seed must have shape "
                f"({self.config.residual_width},)"
            )
        bits = np.ascontiguousarray(float_array_to_bf16_bits(values), dtype=np.uint16)
        copy_host_to_device(
            self.last_hidden,
            host_array_ptr(bits),
            bits.nbytes,
            runtime=self.runtime,
        )

    def _fuse_inputs(
        self,
        token_id: int,
        hidden_seed: np.ndarray | None,
        *,
        token_id_resident: bool = False,
        hidden_seed_resident: bool = False,
    ) -> None:
        token = int(token_id)
        if not token_id_resident:
            if token < 0 or token >= self.config.vocab_size:
                raise ValueError("Qwen4Exp MTP token is outside the vocabulary")
            token_host = np.asarray([token], dtype=np.int64)
            copy_host_to_device(
                self.token_id_buffer,
                host_array_ptr(token_host),
                runtime=self.runtime,
            )
        if not hidden_seed_resident:
            if hidden_seed is None:
                raise ValueError("Qwen4Exp MTP host hidden seed is required")
            self._upload_hidden_seed(hidden_seed)
        embedding_weight = self.resident.weight("root.token_embedding")
        embedding = resolve(
            backend=self.backend,
            layer="embedding",
            quant=embedding_weight.spec.quant_key,
            variant="lookup_bf16_out",
        )
        embedding(
            self.token_id_buffer.ptr,
            embedding_weight.allocation("raw").tensor.ptr,
            self.embedding.ptr,
            1,
            self.config.hidden_size,
            self.config.vocab_size,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            self.embedding.ptr,
            self.resident.weight("nextn.enorm").allocation("raw").tensor.ptr,
            self.embedding_norm.ptr,
            1,
            self.config.hidden_size,
            self.config.attention_rms_epsilon,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            self.last_hidden.ptr,
            self.resident.weight("nextn.hnorm").allocation("raw").tensor.ptr,
            self.hidden_norm.ptr,
            1,
            self.config.residual_width,
            self.config.attention_rms_epsilon,
            runtime=self.runtime,
        )
        row_bytes = 2 * self.config.hidden_size * DType.BF16.itemsize
        half_bytes = self.config.hidden_size * DType.BF16.itemsize
        for branch in range(self.config.residual_branch_count):
            destination = self.fused_input.ptr + branch * row_bytes
            self.runtime.memcpy(
                destination,
                self.embedding_norm.ptr,
                half_bytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            self.runtime.memcpy(
                destination + half_bytes,
                self.hidden_norm.ptr + branch * half_bytes,
                half_bytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        launch_gguf_linear(
            self.resident.weight("nextn.eh_proj"),
            self.fused_input.ptr,
            self.residual.ptr,
            self.config.residual_branch_count,
            2 * self.config.hidden_size,
            self.config.hidden_size,
            activation_dtype=GGUF_ACTIVATION_BF16,
            output_dtype=GGUF_OUTPUT_BF16,
            runtime=self.runtime,
        )

    def forward(
        self,
        token_id: int,
        hidden_seed: np.ndarray | None,
        *,
        capture_logits: bool = True,
        capture_hidden_seed: bool = True,
        capture_token_id: bool = True,
        token_id_resident: bool = False,
        hidden_seed_resident: bool = False,
    ) -> Qwen4ExpMTPDraftResult:
        self._require_open()
        if self.position >= self.max_sequence_length:
            raise ValueError("Qwen4Exp MTP draft capacity exceeded")
        stages: dict[str, float] = {}

        def finish_stage(name: str, started: float) -> None:
            stages[name] = (time.perf_counter() - started) * 1_000.0

        started = time.perf_counter()
        self._fuse_inputs(
            token_id,
            hidden_seed,
            token_id_resident=token_id_resident,
            hidden_seed_resident=hidden_seed_resident,
        )
        finish_stage("draft_input_fusion", started)
        started = time.perf_counter()
        output = run_qwen4_exp_dense_qsa_layer(
            self.residual.ptr,
            self.binding,
            attention_state=self.attention,
            index_state=self.index,
            scratch=self.layer_scratch,
            position=self.position,
            rows=1,
            branches=self.config.residual_branch_count,
            hidden=self.config.hidden_size,
            low_rank=self.config.residual_low_rank,
            query_heads=self.config.attention_head_count,
            kv_heads=self.config.attention_kv_head_count,
            head_dim=self.config.attention_key_length,
            rotary_dim=self.config.rope_dimension_count,
            theta=self.config.rope_freq_base,
            index_heads=self.config.indexer_head_count,
            index_dim=self.config.indexer_key_length,
            index_rotary_dim=self.config.rope_dimension_count,
            ffn=self.config.expert_feed_forward_length,
            experts=self.config.expert_count,
            top_k=self.config.expert_used_count,
            runtime=self.runtime,
        )
        finish_stage("draft_layer", started)
        self.runtime.memcpy(
            self.last_hidden.ptr,
            output.ptr,
            self.last_hidden.nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
        )
        started = time.perf_counter()
        head = run_qwen4_exp_gr_read(
            output.ptr,
            self.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
            self.resident.weight("root.head_hc_down"),
            self.resident.weight("root.head_hc_up"),
            None,
            self.head_scratch,
            rows=1,
            branches=self.config.residual_branch_count,
            hidden=self.config.hidden_size,
            low_rank=self.config.residual_low_rank,
            runtime=self.runtime,
        )
        finish_stage("draft_head_gr", started)
        started = time.perf_counter()
        launch_gguf_linear(
            self.resident.weight("root.lm_head"),
            head.mixed.ptr,
            self.logits_buffer.ptr,
            1,
            self.config.hidden_size,
            self.config.vocab_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
        )
        finish_stage("draft_lm_head", started)
        logits = None
        if capture_logits:
            started = time.perf_counter()
            self.runtime.device_synchronize()
            finish_stage("draft_device_synchronize", started)
            logits = np.empty(self.config.vocab_size, dtype=np.float32)
            started = time.perf_counter()
            copy_device_to_host(
                host_array_ptr(logits), self.logits_buffer, runtime=self.runtime
            )
            finish_stage("draft_logits_d2h", started)
            started = time.perf_counter()
            token_id_result = int(np.argmax(logits))
            finish_stage("draft_sampler", started)
        else:
            started = time.perf_counter()
            argmax_f32(
                self.logits_buffer.ptr,
                self.argmax_block_values.ptr,
                self.argmax_block_indices.ptr,
                self.token_id_buffer.ptr,
                self.argmax_value.ptr,
                self.config.vocab_size,
                runtime=self.runtime,
            )
            token_id_result = -1
            if capture_token_id:
                token_host = np.empty(1, dtype=np.int64)
                copy_device_to_host(
                    host_array_ptr(token_host),
                    self.token_id_buffer,
                    token_host.nbytes,
                    runtime=self.runtime,
                )
                token_id_result = int(token_host[0])
                finish_stage("draft_device_argmax_and_token_d2h", started)
            else:
                finish_stage("draft_device_argmax", started)
        chained_hidden = None
        if capture_hidden_seed:
            hidden_bits = np.empty(self.config.residual_width, dtype=np.uint16)
            started = time.perf_counter()
            copy_device_to_host(
                host_array_ptr(hidden_bits), self.last_hidden, runtime=self.runtime
            )
            finish_stage("draft_hidden_d2h", started)
            chained_hidden = np.ascontiguousarray(
                (hidden_bits.astype(np.uint32) << 16).view(np.float32)
            )
        self.last_forward_stage_timings_ms = stages
        self.position += 1
        return Qwen4ExpMTPDraftResult(
            token_id=token_id_result,
            logits=logits,
            hidden_seed=chained_hidden,
        )

    def prime_prompt(
        self,
        token_ids: Sequence[int],
        target_hidden_rows: np.ndarray,
    ) -> None:
        tokens = tuple(int(token) for token in token_ids)
        hidden = np.ascontiguousarray(target_hidden_rows, dtype=np.float32)
        if not tokens:
            raise ValueError("Qwen4Exp MTP prompt priming requires tokens")
        if len(tokens) > self.max_sequence_length:
            raise ValueError("Qwen4Exp MTP prompt exceeds capacity")
        if hidden.shape != (len(tokens), self.config.residual_width):
            raise ValueError(
                "Qwen4Exp MTP target hidden rows must have shape "
                f"({len(tokens)}, {self.config.residual_width})"
            )
        self.reset()
        zero = np.zeros(self.config.residual_width, dtype=np.float32)
        for row, token in enumerate(tokens):
            self.forward(token, zero if row == 0 else hidden[row - 1])

    def propose_chain(
        self,
        *,
        start_token: int,
        target_hidden_seed: np.ndarray,
        draft_n_max: int,
        compact_output: bool | None = None,
    ) -> tuple[Qwen4ExpMTPDraftResult, ...]:
        count = int(draft_n_max)
        if compact_output is None:
            compact_output = os.environ.get(
                "HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT", "0"
            ) not in {"", "0", "false", "False"}
        if count <= 0 or count > 4:
            raise ValueError("Qwen4Exp MTP draft_n_max must be in 1..4")
        if self.position + count > self.max_sequence_length:
            raise ValueError("Qwen4Exp MTP proposal exceeds capacity")
        results: list[Qwen4ExpMTPDraftResult] = []
        stage_totals: dict[str, float] = {}
        token = int(start_token)
        hidden: np.ndarray | None = np.ascontiguousarray(
            target_hidden_seed, dtype=np.float32
        )
        for depth in range(count):
            result = self.forward(
                token,
                hidden,
                capture_logits=not compact_output,
                capture_hidden_seed=not compact_output,
                capture_token_id=not compact_output,
                token_id_resident=compact_output and depth > 0,
                hidden_seed_resident=compact_output and depth > 0,
            )
            results.append(result)
            if compact_output:
                self.runtime.memcpy(
                    self.candidate_packet.ptr + depth * DType.INT64.itemsize,
                    self.token_id_buffer.ptr,
                    DType.INT64.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
            for name, elapsed_ms in self.last_forward_stage_timings_ms.items():
                stage_totals[name] = stage_totals.get(name, 0.0) + elapsed_ms
            token = result.token_id
            hidden = None if compact_output else result.hidden_seed
        if compact_output:
            packet = np.empty(count, dtype=np.int64)
            started = time.perf_counter()
            copy_device_to_host(
                host_array_ptr(packet),
                self.candidate_packet,
                packet.nbytes,
                runtime=self.runtime,
            )
            stage_totals["draft_candidate_packet_d2h"] = (
                time.perf_counter() - started
            ) * 1_000.0
            results = [
                Qwen4ExpMTPDraftResult(int(packet[index]), None, None)
                for index in range(count)
            ]
        self.last_proposal_stage_timings_ms = stage_totals
        return tuple(results)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for buffer in reversed(getattr(self, "_buffers", ())):
            free(buffer, runtime=self.runtime)
        if hasattr(self, "_buffers"):
            self._buffers.clear()
        if hasattr(self, "head_scratch"):
            self.head_scratch.close()
        if hasattr(self, "layer_scratch"):
            self.layer_scratch.close()
        if hasattr(self, "index"):
            self.index.close()
        if hasattr(self, "attention"):
            self.attention.close()

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp MTP draft runner is closed")


__all__ = [
    "Qwen4ExpGGUFMTPDraftRunner",
    "Qwen4ExpMTPDraftResult",
    "Qwen4ExpMTPDraftSnapshot",
]
