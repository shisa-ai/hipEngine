"""Bounded, request-owned predictor prompt projections using existing kernels."""
from __future__ import annotations

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_rmsnorm_bf16_f32_weight
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import qwen4_exp_gather_bf16_lanes
from hipengine.kernels.registry import resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.gguf_linear import launch_gguf_linear_decode_rows
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGRScratch, Qwen4ExpQSAScratch, Qwen4ExpQSAPrefillMetadata,
    run_qwen4_exp_gr_read, run_qwen4_exp_qsa_prefill_token_mixer,
)


def prompt_fusion_indices(rows: int, branches: int) -> np.ndarray:
    """Gather normalized embedding/hidden rows into paired predictor branches."""
    if rows <= 0 or branches <= 0:
        raise ValueError("prompt rows and branches must be positive")
    indices = np.empty((rows, branches, 2), dtype=np.int64)
    indices[:, :, 0] = np.arange(rows, dtype=np.int64)[:, None]
    indices[:, :, 1] = rows + np.arange(rows * branches, dtype=np.int64).reshape(rows, branches)
    return indices.reshape(-1)


class Qwen4ExpMTPPrefillWorkspace:
    """Project one bounded tile and append its raw KV/index rows.

    The owner controls logical commit and cancellation. This operation does not
    compute hidden carry or update the owner's prompt cursor. The final row
    must still run through the complete scalar predictor.
    """

    def __init__(self, owner, *, rows: int = 64):
        self.owner, self.runtime = owner, owner.runtime
        self.rows = min(int(rows), owner.max_sequence_length)
        if self.rows <= 0:
            raise ValueError("prompt workspace rows must be positive")
        self.closed = False
        self.append_calls = 0
        self.buffers = {}
        self.gr = self.qsa = self.metadata = None
        c = owner.config
        n, h, b = self.rows, c.hidden_size, c.residual_branch_count
        sizes = {"tokens": n * 8, "hidden": n * c.residual_width * 2,
                 "embedding": n * h * 2, "normalized": n * (1 + b) * h * 2,
                 "fused": n * b * 2 * h * 2, "indices": n * b * 2 * 8,
                 "residual": n * c.residual_width * 2}
        try:
            for name, size in sizes.items():
                self.buffers[name] = malloc(size, runtime=self.runtime)
            self.gr = Qwen4ExpGRScratch.allocate(rows=n, branches=b, hidden=h,
                low_rank=c.residual_low_rank, runtime=self.runtime)
            self.qsa = Qwen4ExpQSAScratch.allocate(rows=n, hidden=h,
                query_heads=c.attention_head_count, kv_heads=c.attention_kv_head_count,
                head_dim=c.attention_key_length, index_heads=c.indexer_head_count,
                index_dim=c.indexer_key_length, runtime=self.runtime)
            # Cache-only execution uses positions and write spans, never score
            # or selection storage. Keep those unused planes minimal.
            self.metadata = Qwen4ExpQSAPrefillMetadata.allocate(owner.attention,
                rows=n, selection_capacity=1, score_blocks=1)
        except BaseException:
            self.close()
            raise

    def append(self, token_ids, shifted_hidden_rows=None, *, target_hidden_buffer=None,
               row_offset=0, previous_hidden=None):
        if self.closed:
            raise RuntimeError("MTP prompt workspace is closed")
        owner, c = self.owner, self.owner.config
        tokens = np.ascontiguousarray(token_ids, dtype=np.int64)
        count = int(tokens.size)
        if tokens.ndim != 1 or not 0 < count <= self.rows:
            raise ValueError("MTP prompt tile must fit its row capacity")
        if target_hidden_buffer is None:
            hidden = np.ascontiguousarray(shifted_hidden_rows, dtype=np.float32)
            if hidden.shape != (count, c.residual_width):
                raise ValueError("MTP prompt tile needs one shifted hidden row per token")
            bits = np.ascontiguousarray(float_array_to_bf16_bits(hidden), dtype=np.uint16)
        else:
            stride = c.residual_width * 2
            if row_offset < 0 or target_hidden_buffer.ptr <= 0 or target_hidden_buffer.nbytes < (row_offset + count) * stride:
                raise ValueError("MTP target hidden buffer does not cover the prompt tile")
            if row_offset == 0:
                previous = np.asarray(previous_hidden, dtype=np.float32)
                if previous.shape != (c.residual_width,):
                    raise ValueError("MTP prompt tile needs its preceding hidden row")
                bits = np.ascontiguousarray(float_array_to_bf16_bits(previous), dtype=np.uint16)
            else:
                bits = None
        if np.any(tokens < 0) or np.any(tokens >= c.vocab_size):
            raise ValueError("MTP prompt token is outside the vocabulary")
        if owner.position != owner.index.count or owner.position + count > owner.max_sequence_length:
            raise ValueError("MTP prompt tile exceeds the contiguous cache capacity")
        if bits is not None:
            copy_host_to_device(self.buffers["hidden"], host_array_ptr(bits), bits.nbytes, runtime=self.runtime)
        if target_hidden_buffer is not None:
            first = int(row_offset == 0)
            if count > first:
                self.runtime.memcpy(self.buffers["hidden"].ptr + first * stride,
                    target_hidden_buffer.ptr + max(0, row_offset - 1) * stride,
                    (count - first) * stride, HipMemcpyKind.DEVICE_TO_DEVICE)
        indices = prompt_fusion_indices(count, c.residual_branch_count)
        for name, array in (("tokens", tokens), ("indices", indices)):
            copy_host_to_device(self.buffers[name], host_array_ptr(array), array.nbytes, runtime=self.runtime)
        embedding = owner.resident.weight("root.token_embedding")
        resolve(backend=owner.backend, layer="embedding", quant=embedding.spec.quant_key,
                variant="lookup_bf16_out")(
            self.buffers["tokens"].ptr, embedding.allocation("raw").tensor.ptr,
            self.buffers["embedding"].ptr, count, c.hidden_size, c.vocab_size, runtime=self.runtime)
        normalized = self.buffers["normalized"].ptr
        for source, destination, role, width in (
            (self.buffers["embedding"].ptr, normalized, "nextn.enorm", c.hidden_size),
            (self.buffers["hidden"].ptr, normalized + count * c.hidden_size * 2,
             "nextn.hnorm", c.residual_width),
        ):
            gguf_rmsnorm_bf16_f32_weight(source,
                owner.resident.weight(role).allocation("raw").tensor.ptr, destination,
                count, width, c.attention_rms_epsilon, runtime=self.runtime)
        qwen4_exp_gather_bf16_lanes(normalized, self.buffers["indices"].ptr,
            self.buffers["fused"].ptr, count * c.residual_branch_count * 2,
            c.hidden_size, runtime=self.runtime)
        launch_gguf_linear_decode_rows(owner.resident.weight("nextn.eh_proj"), self.buffers["fused"].ptr,
            self.buffers["residual"].ptr, count * c.residual_branch_count,
            2 * c.hidden_size, c.hidden_size, activation_dtype="bf16", output_dtype="bf16",
            runtime=self.runtime, reference_rows=c.residual_branch_count)
        weights = owner.binding.attention_gr
        mixed = run_qwen4_exp_gr_read(self.buffers["residual"].ptr,
            weights.norm_weight_ptr, weights.down, weights.up, weights.inject, self.gr,
            rows=count, branches=c.residual_branch_count, hidden=c.hidden_size,
            low_rank=c.residual_low_rank, runtime=self.runtime, decode_rows=True)
        run_qwen4_exp_qsa_prefill_token_mixer(mixed.mixed.ptr, owner.binding.mixer,
            attention_state=owner.attention, index_state=owner.index, scratch=self.qsa,
            metadata=self.metadata, start_position=owner.position, rows=count,
            hidden=c.hidden_size, query_heads=c.attention_head_count,
            kv_heads=c.attention_kv_head_count, head_dim=c.attention_key_length,
            rotary_dim=c.rope_dimension_count, theta=c.rope_freq_base,
            index_heads=c.indexer_head_count, index_dim=c.indexer_key_length,
            index_rotary_dim=c.rope_dimension_count, runtime=self.runtime, cache_only=True)
        self.append_calls += 1

    def close(self):
        if self.closed:
            return
        for resource in (self.metadata, self.qsa, self.gr):
            if resource is not None:
                resource.close()
        for buffer in reversed(tuple(self.buffers.values())):
            free(buffer, runtime=self.runtime)
        self.closed = True
