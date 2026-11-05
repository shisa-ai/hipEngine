"""Muse Glimmer resident text session on gfx11 (bring-up, eager).

Composes registered kernels only -- no model branches in dispatch. The layer
dataflow mirrors ``hipengine.kernels.cpu_reference.muse_glimmer`` stage for
stage; the campaign doc (``docs/campaigns/MUSE-GLIMMER.md`` §3.1) owns the
primitive map and the load-time NORM->NEOX row permutation that lets the
registered NEOX head-norm+rotary kernel compute Glimmer's interleaved RoPE.

Bring-up contracts:

* KV is a contiguous per-layer BF16 cache (the gemma4 attention family ABI).
  SWA is expressed by the caller-supplied keep mask, full layers are NoPE
  (identity rotary tables), and the paged KVLiveSpans migration is a recorded
  performance-phase decision, not a silent shortcut.
* Logits leave the device as F32 and take ``logit_scale`` + tanh softcap on the
  host in float32, matching the CPU reference exactly; the device softcap
  kernel is a performance-phase replacement with its own gate.
* Prefill runs in bounded chunks so the keep-mask and scratch stay small; each
  chunk attends over the full live cache through the same registered wrapper.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_bf16_add,
    gguf_gate_mul_bf16,
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_rmsnorm_f32w_bf16,
    gemma4_rmsnorm_weightless_bf16,
)
from hipengine.loading.muse_glimmer_gguf import MuseGlimmerLoadedWeights
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import launch_gguf_linear

#: Prefill chunk bound for the bring-up scratch and keep-mask sizing.
DEFAULT_PREFILL_CHUNK = 512


def _rope_tables_neox(
    max_positions: int, head_dim: int, freq_base: float
) -> tuple[np.ndarray, np.ndarray]:
    """Doubled NEOX cos/sin tables indexed by element, as the rotary kernel reads.

    The frequency math matches the CPU reference's interleaved tables; the
    load-time row permutation makes the paired rotation equivalent.
    """

    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(head_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(freq_base), -2.0 * dims / np.float32(head_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1)
    sin = np.concatenate([sin_half, sin_half], axis=1)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


@dataclass
class _SessionScratch:
    """Stream-owned scratch for one prefill chunk / decode step."""

    buffers: list[DeviceBuffer] = field(default_factory=list)

    def alloc(self, nbytes: int, runtime: HipRuntime) -> DeviceBuffer:
        buf = malloc(nbytes, runtime=runtime)
        self.buffers.append(buf)
        return buf

    def close(self, runtime: HipRuntime) -> None:
        for buf in reversed(self.buffers):
            free(buf, runtime=runtime)
        self.buffers.clear()


class MuseGlimmerResidentSession:
    """One resident Glimmer text model with a single-request KV cache."""

    def __init__(
        self,
        loaded: MuseGlimmerLoadedWeights,
        *,
        max_context: int,
        prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.runtime = runtime or get_hip_runtime()
        self.loaded = loaded
        self.config = loaded.config
        self.backend = loaded.backend
        self.max_context = int(max_context)
        self.prefill_chunk = int(prefill_chunk)
        if self.max_context <= 0 or self.prefill_chunk <= 0:
            raise ValueError("max_context and prefill_chunk must be positive")
        self._closed = False
        self._length = 0
        self._image_overrides: dict[int, np.ndarray] | None = None
        self._scratch: _SessionScratch | None = None
        self._kv: list[tuple[DeviceBuffer, DeviceBuffer]] = []
        self._buffers: list[DeviceBuffer] = []
        self._allocate()

    # -- allocation --------------------------------------------------------
    def _allocate(self) -> None:
        cfg, rt = self.config, self.runtime
        t_max = min(self.prefill_chunk, self.max_context)
        hidden = cfg.n_embd
        q_dim = cfg.qkv_dim
        kv_dim = cfg.kv_dim
        scratch = _SessionScratch()
        self._scratch = scratch
        try:
            bf16 = 2
            f32 = 4
            self.hidden_a = scratch.alloc(t_max * hidden * bf16, rt)
            self.hidden_b = scratch.alloc(t_max * hidden * bf16, rt)
            self.normed = scratch.alloc(t_max * hidden * bf16, rt)
            self.q_f32 = scratch.alloc(t_max * q_dim * f32, rt)
            self.k_f32 = scratch.alloc(t_max * kv_dim * f32, rt)
            self.q_rot_f32 = scratch.alloc(t_max * q_dim * f32, rt)
            self.k_rot_f32 = scratch.alloc(t_max * kv_dim * f32, rt)
            self.q_bf16 = scratch.alloc(t_max * q_dim * bf16, rt)
            self.k_bf16 = scratch.alloc(t_max * kv_dim * bf16, rt)
            self.v_bf16 = scratch.alloc(t_max * kv_dim * bf16, rt)
            self.gate_bf16 = scratch.alloc(t_max * q_dim * bf16, rt)
            self.attn_out = scratch.alloc(t_max * q_dim * bf16, rt)
            self.attn_proj = scratch.alloc(t_max * hidden * bf16, rt)
            self.post_normed = scratch.alloc(t_max * hidden * bf16, rt)
            self.ff_gate = scratch.alloc(t_max * cfg.n_ff * bf16, rt)
            self.ff_up = scratch.alloc(t_max * cfg.n_ff * bf16, rt)
            self.ff_act = scratch.alloc(t_max * cfg.n_ff * bf16, rt)
            self.ff_out = scratch.alloc(t_max * hidden * bf16, rt)
            self.logits_f32 = scratch.alloc(cfg.vocab_size * f32, rt)
            self.token_ids = scratch.alloc(t_max * 8, rt)
            self.positions = scratch.alloc(t_max * 8, rt)
            self.cos_table = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.sin_table = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.identity_cos = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.identity_sin = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.keep_mask_full = scratch.alloc(t_max * self.max_context, rt)
            self.keep_mask_swa = scratch.alloc(t_max * self.max_context, rt)
            self._logits_host = np.empty(cfg.vocab_size, dtype=np.float32)
            self._mask_host = np.empty((t_max, self.max_context), dtype=np.uint8)
            self._ids_host = np.empty(t_max, dtype=np.int64)
            self._pos_host = np.empty(t_max, dtype=np.int64)

            cos, sin = _rope_tables_neox(
                self.max_context, cfg.head_dim, cfg.rope_freq_base
            )
            copy_host_array_to_device(self.cos_table, cos, runtime=rt)
            copy_host_array_to_device(self.sin_table, sin, runtime=rt)
            copy_host_array_to_device(
                self.identity_cos, np.ones_like(cos), runtime=rt
            )
            copy_host_array_to_device(
                self.identity_sin, np.zeros_like(sin), runtime=rt
            )

            kv_row_bytes = kv_dim * bf16
            for _ in range(cfg.n_layers):
                k_cache = scratch.alloc(self.max_context * kv_row_bytes, rt)
                v_cache = scratch.alloc(self.max_context * kv_row_bytes, rt)
                self._kv.append((k_cache, v_cache))
        except Exception:
            scratch.close(rt)
            self._scratch = None
            self._kv.clear()
            raise

    # -- lifecycle ---------------------------------------------------------
    @property
    def length(self) -> int:
        return self._length

    def reset(self) -> None:
        """Drop the KV content; the next prefill starts at position 0."""

        self._require_open()
        self._length = 0

    @contextmanager
    def image_input(self, overrides: dict[int, np.ndarray]):
        """Replace embedding rows at absolute prompt positions for one prefill.

        Overrides are projected F32 vision rows; they enter *before* the
        weightless embedding norm, exactly as llama.cpp's external-embedding
        path feeds the graph (the norm applies to image rows too). Cleared on
        exit so a later text-only request cannot inherit stale rows.
        """

        self._require_open()
        if self._image_overrides is not None:
            raise RuntimeError("nested image_input spans are not supported")
        self._image_overrides = dict(overrides)
        try:
            yield self
        finally:
            self._image_overrides = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._scratch is not None:
            self._scratch.close(self.runtime)
            self._scratch = None
        self._kv.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the Muse Glimmer resident session is closed")

    # -- forward -----------------------------------------------------------
    def prefill(self, token_ids: Sequence[int]) -> np.ndarray:
        """Run the prompt in chunks; return the final row's logits (F32)."""

        self._require_open()
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("prefill needs a non-empty 1-D token sequence")
        if ids.size > self.max_context:
            raise ValueError(
                f"prompt of {ids.size} tokens exceeds the {self.max_context}-token cache"
            )
        self._length = 0
        logits: np.ndarray | None = None
        for begin in range(0, ids.size, self.prefill_chunk):
            block = ids[begin:begin + self.prefill_chunk]
            logits = self._forward_block(block, position_offset=begin)
        assert logits is not None
        return logits

    def decode_step(self, token_id: int) -> np.ndarray:
        """Append one token and return its logits (F32)."""

        self._require_open()
        if self._length >= self.max_context:
            raise ValueError("the KV cache is full; reset or raise max_context")
        return self._forward_block(
            np.asarray([int(token_id)], dtype=np.int64), position_offset=self._length
        )

    def _forward_block(self, ids: np.ndarray, *, position_offset: int) -> np.ndarray:
        cfg, rt = self.config, self.runtime
        tokens = int(ids.size)
        keys = position_offset + tokens
        if keys > self.max_context:
            raise ValueError("block overruns the KV cache")
        hidden = cfg.n_embd
        q_dim, kv_dim, d = cfg.qkv_dim, cfg.kv_dim, cfg.head_dim

        self._ids_host[:tokens] = ids
        copy_host_array_to_device(self.token_ids, self._ids_host[:tokens], runtime=rt)
        self._pos_host[:tokens] = np.arange(
            position_offset, position_offset + tokens, dtype=np.int64
        )
        copy_host_array_to_device(self.positions, self._pos_host[:tokens], runtime=rt)
        self._upload_keep_mask(tokens, position_offset, keys)

        launch_gguf_embedding(
            self.loaded.weight("token_embd.weight"),
            self.token_ids.ptr,
            self.hidden_a.ptr,
            rows=tokens,
            hidden_size=hidden,
            vocab_size=cfg.vocab_size,
            backend=self.backend,
            runtime=rt,
        )
        if self._image_overrides:
            from hipengine.core.memory import DeviceBuffer
            from hipengine.loading.materialize import float_array_to_bf16_bits

            for pos, row in self._image_overrides.items():
                if position_offset <= pos < position_offset + tokens:
                    bits = float_array_to_bf16_bits(row)
                    local = pos - position_offset
                    copy_host_array_to_device(
                        DeviceBuffer(self.hidden_a.ptr + local * hidden * 2, bits.nbytes),
                        bits, runtime=rt,
                    )
        gemma4_rmsnorm_weightless_bf16(
            self.hidden_a.ptr, self.hidden_b.ptr, tokens, hidden,
            cfg.rms_eps, runtime=rt,
        )
        src, dst = self.hidden_b, self.hidden_a

        for layer in range(cfg.n_layers):
            p = f"blk.{layer}."
            is_swa = cfg.swa_pattern[layer]
            gemma4_rmsnorm_f32w_bf16(
                src.ptr, self.loaded.ptr(p + "attn_norm.weight"), self.normed.ptr,
                tokens, hidden, cfg.rms_eps, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "attn_q.weight"), self.normed.ptr,
                self.q_f32.ptr, tokens, hidden, q_dim,
                output_dtype="f32", backend=self.backend, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "attn_k.weight"), self.normed.ptr,
                self.k_f32.ptr, tokens, hidden, kv_dim,
                output_dtype="f32", backend=self.backend, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "attn_v.weight"), self.normed.ptr,
                self.v_bf16.ptr, tokens, hidden, kv_dim,
                backend=self.backend, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "attn_gate.weight"), self.normed.ptr,
                self.gate_bf16.ptr, tokens, hidden, q_dim,
                backend=self.backend, runtime=rt,
            )
            # QK-norm always; rotation only on SWA layers. Full (NoPE) layers
            # take identity tables: cos=1/sin=0 leaves every element exact.
            gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
                self.q_f32.ptr, self.k_f32.ptr,
                self.loaded.ptr(p + "attn_q_norm.weight"),
                self.loaded.ptr(p + "attn_k_norm.weight"),
                (self.cos_table if is_swa else self.identity_cos).ptr,
                (self.sin_table if is_swa else self.identity_sin).ptr,
                self.positions.ptr,
                self.q_rot_f32.ptr, self.k_rot_f32.ptr,
                cfg.rms_eps, tokens, cfg.n_heads, cfg.n_kv_heads, d, d,
                self.max_context, runtime=rt,
            )
            f32_to_bf16(self.q_rot_f32.ptr, self.q_bf16.ptr, tokens * q_dim, runtime=rt)
            f32_to_bf16(self.k_rot_f32.ptr, self.k_bf16.ptr, tokens * kv_dim, runtime=rt)

            k_cache, v_cache = self._kv[layer]
            row_bytes = kv_dim * 2
            rt.memcpy(
                k_cache.ptr + position_offset * row_bytes, self.k_bf16.ptr,
                tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE,
            )
            rt.memcpy(
                v_cache.ptr + position_offset * row_bytes, self.v_bf16.ptr,
                tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE,
            )
            gemma4_attention_prefill_bf16(
                self.q_bf16.ptr, k_cache.ptr, v_cache.ptr,
                (self.keep_mask_swa if is_swa else self.keep_mask_full).ptr,
                self.attn_out.ptr,
                tokens=tokens, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads,
                head_dim=d, scale=float(d) ** -0.5, keys=keys,
                window=cfg.sliding_window if is_swa else 0,
                row_offset=position_offset, runtime=rt,
            )
            gguf_gate_mul_bf16(
                self.attn_out.ptr, self.gate_bf16.ptr, self.attn_out.ptr,
                tokens * q_dim, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "attn_output.weight"), self.attn_out.ptr,
                self.attn_proj.ptr, tokens, q_dim, hidden,
                backend=self.backend, runtime=rt,
            )
            gemma4_rmsnorm_f32w_bf16(
                self.attn_proj.ptr, self.loaded.ptr(p + "post_attention_norm.weight"),
                self.post_normed.ptr, tokens, hidden, cfg.post_norm_eps, runtime=rt,
            )
            gguf_bf16_add(src.ptr, self.post_normed.ptr, dst.ptr, tokens * hidden, runtime=rt)
            src, dst = dst, src

            gemma4_rmsnorm_f32w_bf16(
                src.ptr, self.loaded.ptr(p + "ffn_norm.weight"), self.normed.ptr,
                tokens, hidden, cfg.rms_eps, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "ffn_gate.weight"), self.normed.ptr,
                self.ff_gate.ptr, tokens, hidden, cfg.n_ff,
                backend=self.backend, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "ffn_up.weight"), self.normed.ptr,
                self.ff_up.ptr, tokens, hidden, cfg.n_ff,
                backend=self.backend, runtime=rt,
            )
            silu_mul_separate_out_bf16(
                self.ff_gate.ptr, self.ff_up.ptr, self.ff_act.ptr,
                tokens, cfg.n_ff, runtime=rt,
            )
            launch_gguf_linear(
                self.loaded.weight(p + "ffn_down.weight"), self.ff_act.ptr,
                self.ff_out.ptr, tokens, cfg.n_ff, hidden,
                backend=self.backend, runtime=rt,
            )
            gemma4_rmsnorm_f32w_bf16(
                self.ff_out.ptr, self.loaded.ptr(p + "post_ffw_norm.weight"),
                self.post_normed.ptr, tokens, hidden, cfg.post_norm_eps, runtime=rt,
            )
            gguf_bf16_add(src.ptr, self.post_normed.ptr, dst.ptr, tokens * hidden, runtime=rt)
            src, dst = dst, src

        self._length = keys
        # Only the final row feeds generation; the head runs on that row alone.
        last_row_ptr = src.ptr + (tokens - 1) * hidden * 2
        gemma4_rmsnorm_f32w_bf16(
            last_row_ptr, self.loaded.ptr("output_norm.weight"), self.normed.ptr,
            1, hidden, cfg.rms_eps, runtime=rt,
        )
        launch_gguf_linear(
            self.loaded.weight("output.weight"), self.normed.ptr,
            self.logits_f32.ptr, 1, hidden, cfg.vocab_size,
            output_dtype="f32", backend=self.backend, runtime=rt,
        )
        copy_device_to_host(
            self._logits_host.ctypes.data, self.logits_f32, runtime=rt
        )
        logits = self._logits_host * np.float32(cfg.logit_scale)
        cap = cfg.final_logit_softcapping
        if cap:
            logits = np.float32(cap) * np.tanh(logits / np.float32(cap))
        return np.ascontiguousarray(logits, dtype=np.float32)

    def _upload_keep_mask(self, tokens: int, position_offset: int, keys: int) -> None:
        """Build both layer-kind masks once per block and upload them.

        Full layers keep the causal prefix; SWA layers additionally drop keys
        at or below ``pos - window``. The kernel's ``window`` walk skips the
        leading zero columns on SWA rows, which this mask guarantees.
        """

        cfg = self.config
        q = np.arange(position_offset, position_offset + tokens, dtype=np.int64)[:, None]
        k = np.arange(keys, dtype=np.int64)[None, :]
        causal = k <= q
        mask = self._mask_host[:tokens, :keys]
        np.copyto(mask, causal.astype(np.uint8))
        copy_host_array_to_device(
            self.keep_mask_full, np.ascontiguousarray(mask), runtime=self.runtime
        )
        swa = causal & (k > q - int(cfg.sliding_window))
        np.copyto(mask, swa.astype(np.uint8))
        copy_host_array_to_device(
            self.keep_mask_swa, np.ascontiguousarray(mask), runtime=self.runtime
        )

    # -- introspection -----------------------------------------------------
    def execution_path(self) -> str:
        return "muse_glimmer_gguf_resident_eager"


__all__ = ["DEFAULT_PREFILL_CHUNK", "MuseGlimmerResidentSession"]