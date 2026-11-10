"""Muse Glimmer resident text session on gfx11 (bring-up, eager).

Composes registered kernels only -- no model branches in dispatch. The layer
dataflow mirrors ``hipengine.kernels.cpu_reference.muse_glimmer`` stage for
stage; the campaign doc (``docs/campaigns/MUSE-GLIMMER.md`` §3.1) owns the
primitive map and the load-time NORM->NEOX row permutation that lets the
registered NEOX head-norm+rotary kernel compute Glimmer's interleaved RoPE.

Bring-up contracts:

* KV is a contiguous per-layer BF16 cache (the gemma4 attention family ABI).
  SWA is applied inside the attention kernels from the layer window, full layers are NoPE
  (identity rotary tables), and the paged KVLiveSpans migration is a recorded
  performance-phase decision, not a silent shortcut.
* Logits leave the device as F32 and take ``logit_scale`` + tanh softcap on the
  host in float32, matching the CPU reference exactly; the device softcap
  kernel is a performance-phase replacement with its own gate.
* Prefill runs in bounded chunks so the scratch stays small; each
  chunk attends over the full live cache through the same registered wrapper.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

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
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_add_rmsnorm_scale_bf16,
    gemma4_rmsnorm_f32w_bf16,
    gemma4_rmsnorm_weightless_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_grouped_gemv import (
    Q4_GROUPED_VARIANT,
    Q6_HOISTED_VARIANT,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.muse_glimmer_gguf import MuseGlimmerLoadedWeights
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    launch_gguf_linear,
    launch_gguf_linear_exact_rows,
    prefill_f16_staging_session,
)

#: Prefill chunk bound for the bring-up scratch sizing. 1024
#: rows amortize the Q4/Q6 WMMA prefill projections better than 512 on
#: gfx1151 (1K prefill 327->339, 4K 293->305 tok/s with the WMMA attention);
#: 2048 adds only ~2% at 4K for double the logits scratch.
DEFAULT_PREFILL_CHUNK = 1024


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
        # Token id at each cached position and how many leading positions hold
        # text-token KV (image rows and torn writes end the reusable prefix).
        self._history = np.zeros(self.max_context, dtype=np.int64)
        self._history_valid = 0
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
            self.logits_bf16 = scratch.alloc(t_max * cfg.vocab_size * bf16, rt)
            self.token_ids = scratch.alloc(t_max * 8, rt)
            self.positions = scratch.alloc(t_max * 8, rt)
            self.cos_table = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.sin_table = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.identity_cos = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self.identity_sin = scratch.alloc(self.max_context * cfg.head_dim * f32, rt)
            self._attn_partial_rows = max(
                self._DECODE_ATTN_BLOCKS * self._MAX_VERIFY_ROWS, 4 * t_max
            )
            self.attn_partials = scratch.alloc(
                self._attn_partial_rows * cfg.n_heads * 130 * 4, rt
            )
            # Bounded F16 activation staging workspace for the WMMA prefill
            # pre-cast route (one row-block x the widest projection input).
            self._f16_staging = scratch.alloc(
                t_max * max(hidden, cfg.n_ff, q_dim) * bf16, rt
            )
            self._logits_host = np.empty(cfg.vocab_size, dtype=np.float32)
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
        self._history_valid = 0

    @property
    def history(self) -> np.ndarray:
        """Token ids whose KV is cached and reusable, oldest first."""

        return self._history[:min(self._history_valid, self._length)]

    def reusable_prefix(self, token_ids: Sequence[int], *, limit: int | None = None) -> int:
        """Leading tokens of ``token_ids`` a prefill may resume after.

        This is the cached common prefix rounded down to a prefill-chunk
        boundary. The prefill projections depend on the chunk's row count, so
        only a chunk-aligned resume runs the same arithmetic as a fresh prefill
        (bit-identical logits when the reused rows were themselves prefilled);
        the cost is at most one chunk of recompute. At least one prompt token
        is always left to prefill, since the prompt's last row produces the
        first logits. ``limit`` caps the reuse, for example at the first image
        position of a multimodal prompt.
        """

        ids = np.asarray(token_ids, dtype=np.int64)
        cached = self.history
        bound = min(cached.size, ids.size - 1)
        if limit is not None:
            bound = min(bound, int(limit))
        if bound <= 0:
            return 0
        mismatch = np.flatnonzero(cached[:bound] != ids[:bound])
        common = int(mismatch[0]) if mismatch.size else bound
        return self.align_resume(common)

    def align_resume(self, tokens: int) -> int:
        return (int(tokens) // self.prefill_chunk) * self.prefill_chunk

    def _record_history(self, ids: np.ndarray, position_offset: int) -> None:
        keys = position_offset + int(ids.size)
        continuous = self._history_valid >= position_offset
        self._history[position_offset:keys] = ids
        end = keys
        if self._image_overrides:
            hits = [p for p in self._image_overrides if position_offset <= p < keys]
            if hits:
                end = min(hits)
        self._history_valid = end if continuous else min(self._history_valid, position_offset)

    def _begin_prefill(self, ids: np.ndarray, start: int) -> None:
        if not 0 <= start < ids.size:
            raise ValueError("prefill start must leave at least one prompt token")
        if start:
            if start > self.history.size or not np.array_equal(self.history[:start], ids[:start]):
                raise ValueError("prefill start exceeds the cached prefix of this prompt")
        self._length = int(start)
        self._history_valid = min(self._history_valid, int(start))

    def _prefill_chunks(self, size: int, start: int):
        """Chunk bounds from ``start``, aligned to absolute multiples of the
        chunk so a resumed prefill runs the same chunk schedule as a fresh one."""

        chunk = self.prefill_chunk
        begin = start
        while begin < size:
            end = min(size, (begin // chunk + 1) * chunk)
            yield begin, end
            begin = end

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
        staging = getattr(self, "_g_staging", None)
        if staging is not None:
            free(staging, runtime=self.runtime)
            self._g_staging = None
        self._kv.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the Muse Glimmer resident session is closed")

    # -- forward -----------------------------------------------------------
    def prefill(self, token_ids: Sequence[int], *, start: int = 0) -> np.ndarray:
        """Run the prompt in chunks; return the final row's logits (F32).

        ``start`` resumes after a cached prefix (see :meth:`reusable_prefix`):
        positions below it keep their KV and only the suffix is computed.
        """

        self._require_open()
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("prefill needs a non-empty 1-D token sequence")
        if ids.size > self.max_context:
            raise ValueError(
                f"prompt of {ids.size} tokens exceeds the {self.max_context}-token cache"
            )
        self._begin_prefill(ids, int(start))
        logits: np.ndarray | None = None
        with prefill_f16_staging_session(
            True,
            workspace_ptr=int(self._f16_staging.ptr),
            workspace_nbytes=int(self._f16_staging.nbytes),
        ):
            for begin, end in self._prefill_chunks(ids.size, int(start)):
                logits = self._forward_block(ids[begin:end], position_offset=begin)
        assert logits is not None
        return logits

    def decode_step(self, token_id: int) -> np.ndarray:
        """Append one token and return its logits (F32)."""

        self._require_open()
        if self._length >= self.max_context:
            raise ValueError("the KV cache is full; reset or raise max_context")
        self._cross_attn_split_threshold()
        if self._image_overrides is None:
            graph = self._ensure_decode_graph()
            if graph is not None:
                return self._decode_step_graph(graph, int(token_id))
        return self._forward_block(
            np.asarray([int(token_id)], dtype=np.int64), position_offset=self._length
        )

    # -- HIP graph decode --------------------------------------------------
    def _attn_split_config(self):
        """Sticky (split_keys, n_blocks) shared by every decode/verify path.

        Memoized at first use so the decode graph, the verify graphs, and the
        eager fallback all run the same split config (the row bit-identity
        contract). Short live ranges use fewer blocks (min 16, one per
        64-key split at capture, capped at 64); the threshold crossing flips
        the session to the full 64-block config.
        """
        cfg = getattr(self, "_attn_split_cfg", None)
        if cfg is None:
            if self._length <= self._ATTN_SHORT_THRESHOLD:
                n_blocks = max(self._ATTN_SHORT_MIN_BLOCKS,
                               (self._length + self._DECODE_ATTN_SPLIT_KEYS - 1)
                               // self._DECODE_ATTN_SPLIT_KEYS)
                n_blocks = min(n_blocks, self._DECODE_ATTN_BLOCKS)
            else:
                n_blocks = self._DECODE_ATTN_BLOCKS
            cfg = (self._DECODE_ATTN_SPLIT_KEYS, n_blocks)
            self._attn_split_cfg = cfg
        return cfg

    def _cross_attn_split_threshold(self):
        """Invalidate both graph families when the live range crosses the
        short-mode threshold, so decode and verify re-capture together with
        the same split config. No-op before the first config memoization."""
        cfg = getattr(self, "_attn_split_cfg", None)
        if cfg is None:
            return
        want = (self._DECODE_ATTN_BLOCKS if self._length > self._ATTN_SHORT_THRESHOLD
                else cfg[1])
        if cfg[1] == self._DECODE_ATTN_BLOCKS or want == cfg[1]:
            return
        self._attn_split_cfg = (self._DECODE_ATTN_SPLIT_KEYS, self._DECODE_ATTN_BLOCKS)
        rt = self.runtime
        if getattr(self, "_graph_exec", 0):
            rt.graph_exec_destroy(self._graph_exec)
            self._graph_exec = 0
        self._graph_failed = False
        cache = getattr(self, "_verify_graph_exec", None)
        if cache:
            for g in cache.values():
                rt.graph_exec_destroy(g)
            cache.clear()
        self._verify_graph_failed = False

    def _ensure_decode_graph(self):
        """Capture the decode step once; replay eliminates the per-step host
        dispatch bubble (~17 ms at 1068 launches). Returns the exec handle or
        None if capture is unavailable (the eager path remains the fallback).
        """

        if getattr(self, "_graph_exec", 0):
            return self._graph_exec
        if getattr(self, "_graph_failed", False):
            return None
        rt = self.runtime
        try:
            from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
                muse_glimmer_decode_attention_shared, muse_glimmer_kv_append,
            )

            stream = getattr(self, "_graph_stream", 0) or rt.stream_create()
            # One staging allocation [token, pos, key_end] (int64) so each
            # replay uploads its position-dependent inputs in a single copy.
            # Reused across threshold-crossing re-captures.
            if getattr(self, "_g_staging", None) is None:
                self._g_staging = malloc(24, runtime=rt)
            base = self._g_staging.ptr
            self._g_token = DeviceBuffer(ptr=base, nbytes=8)
            self._g_pos = DeviceBuffer(ptr=base + 8, nbytes=8)
            self._g_key_end = DeviceBuffer(ptr=base + 16, nbytes=8)
            self._g_staging_host = np.zeros(3, dtype=np.int64)
            self._softcap_host = np.empty(self.config.vocab_size, dtype=np.float32)
            self._graph_stream = stream
            # Warm every kernel on the capture stream outside capture.
            self._g_staging_host[:] = (13796, self._length, self._length + 1)
            copy_host_array_to_device(self._g_staging, self._g_staging_host, runtime=rt)
            self._decode_graph_body(stream)
            rt.stream_synchronize(stream)
            rt.stream_begin_capture(stream, mode=0)
            try:
                self._decode_graph_body(stream)
                graph = rt.stream_end_capture(stream)
            except BaseException:
                if rt.stream_is_capturing(stream):
                    rt.stream_end_capture(stream)
                raise
            if not graph:
                raise RuntimeError("decode graph capture produced no graph")
            self._graph_exec = rt.graph_instantiate(graph)
            rt.graph_destroy(graph)
            return self._graph_exec
        except Exception:
            self._graph_failed = True
            buf = getattr(self, "_g_staging", None)
            if buf is not None:
                free(buf, runtime=rt)
            self._g_staging = self._g_token = self._g_pos = self._g_key_end = None
            return None

    def _decode_graph_body(self, stream: int) -> None:
        """Kernel-only decode sequence with graph-stable arguments.

        Position-dependent content lives in device staging buffers updated
        before each replay; the KV append and attention read the position and
        live extent from those buffers, so no kernel argument changes between
        steps. Mirrors the eager decode arithmetic exactly (fixed 64 splits).
        """

        from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
            muse_glimmer_add_rmsnorm_rmsnorm,
            muse_glimmer_decode_attention_shared,
            muse_glimmer_qk_norm_rope_kv_append,
        )
        from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

        cfg, rt = self.config, self.runtime
        hidden = cfg.n_embd
        q_dim, kv_dim, d = cfg.qkv_dim, cfg.kv_dim, cfg.head_dim
        kw = {"stream": stream, "runtime": rt}

        launch_gguf_embedding(
            self.loaded.weight("token_embd.weight"), self._g_token.ptr,
            self.hidden_a.ptr, rows=1, hidden_size=hidden,
            vocab_size=cfg.vocab_size, backend=self.backend, **kw,
        )
        gemma4_rmsnorm_weightless_bf16(
            self.hidden_a.ptr, self.hidden_b.ptr, 1, hidden, cfg.rms_eps, **kw
        )
        src, dst = self.hidden_b, self.hidden_a
        # Each residual boundary runs as one fused launch, bit-identical to
        # add_rmsnorm_scale -> rmsnorm: the branch post-norm + residual add,
        # then the next pre-norm (ffn_norm, the next layer's attn_norm, or
        # output_norm after the last layer) into ``normed``.
        gemma4_rmsnorm_f32w_bf16(
            src.ptr, self.loaded.ptr("blk.0.attn_norm.weight"), self.normed.ptr,
            1, hidden, cfg.rms_eps, **kw,
        )
        for layer in range(cfg.n_layers):
            p = f"blk.{layer}."
            is_swa = cfg.swa_pattern[layer]
            next_norm = (
                f"blk.{layer + 1}.attn_norm.weight"
                if layer + 1 < cfg.n_layers else "output_norm.weight"
            )
            self._decode_linears(self.normed.ptr, hidden, (
                (self.loaded.weight(p + "attn_q.weight"), self.q_bf16.ptr, q_dim),
                (self.loaded.weight(p + "attn_k.weight"), self.k_bf16.ptr, kv_dim),
                (self.loaded.weight(p + "attn_v.weight"), self.v_bf16.ptr, kv_dim),
                (self.loaded.weight(p + "attn_gate.weight"), self.gate_bf16.ptr, q_dim),
            ), waves=self._ATTN_GROUP_WAVES, kw=kw)
            # Fused bit-identical replacement for bf16_to_f32 x2 -> head-norm
            # rotary -> f32_to_bf16 x2 -> K/V append: rotated Q lands in the
            # decode-idle q_rot_f32 scratch (as BF16), K/V go to the cache row.
            k_cache, v_cache = self._kv[layer]
            q_rope = self.q_rot_f32
            muse_glimmer_qk_norm_rope_kv_append(
                self.q_bf16.ptr, self.k_bf16.ptr, self.v_bf16.ptr,
                self.loaded.ptr(p + "attn_q_norm.weight"),
                self.loaded.ptr(p + "attn_k_norm.weight"),
                (self.cos_table if is_swa else self.identity_cos).ptr,
                (self.sin_table if is_swa else self.identity_sin).ptr,
                self._g_pos.ptr, q_rope.ptr, k_cache.ptr, v_cache.ptr,
                eps=cfg.rms_eps, n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                head_dim=d, max_positions=self.max_context, **kw,
            )
            muse_glimmer_decode_attention_shared(
                q_rope.ptr, k_cache.ptr, v_cache.ptr, self.attn_out.ptr,
                self.attn_partials.ptr,
                n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                key_begin=0, key_end=self.max_context,
                split_keys=self._attn_split_config()[0], n_blocks=self._attn_split_config()[1],
                tokens=1, window=cfg.sliding_window if is_swa else 0,
                key_end_dev=self._g_key_end.ptr, gate_ptr=self.gate_bf16.ptr,
                scale=float(d) ** -0.5, **kw,
            )
            self._decode_linears(self.attn_out.ptr, q_dim, (
                (self.loaded.weight(p + "attn_output.weight"), self.attn_proj.ptr, hidden),
            ), waves=self._ATTN_OUTPUT_WAVES, kw=kw)
            muse_glimmer_add_rmsnorm_rmsnorm(
                self.attn_proj.ptr, src.ptr,
                self.loaded.ptr(p + "post_attention_norm.weight"),
                self.loaded.ptr(p + "ffn_norm.weight"), dst.ptr, self.normed.ptr,
                rows=1, hidden_size=hidden, post_eps=cfg.post_norm_eps,
                pre_eps=cfg.rms_eps, **kw,
            )
            src, dst = dst, src
            self._decode_linears(self.normed.ptr, hidden, (
                (self.loaded.weight(p + "ffn_gate.weight"), self.ff_gate.ptr, cfg.n_ff),
                (self.loaded.weight(p + "ffn_up.weight"), self.ff_up.ptr, cfg.n_ff),
            ), waves=self._FFN_GROUP_WAVES, kw=kw)
            silu_mul_separate_out_bf16(self.ff_gate.ptr, self.ff_up.ptr,
                                       self.ff_act.ptr, 1, cfg.n_ff, **kw)
            self._decode_linears(self.ff_act.ptr, cfg.n_ff, (
                (self.loaded.weight(p + "ffn_down.weight"), self.ff_out.ptr, hidden),
            ), waves=self._FFN_DOWN_WAVES, kw=kw)
            muse_glimmer_add_rmsnorm_rmsnorm(
                self.ff_out.ptr, src.ptr,
                self.loaded.ptr(p + "post_ffw_norm.weight"),
                self.loaded.ptr(next_norm), dst.ptr, self.normed.ptr,
                rows=1, hidden_size=hidden, post_eps=cfg.post_norm_eps,
                pre_eps=cfg.rms_eps, **kw,
            )
            src, dst = dst, src
        self._linear(self.loaded.weight("output.weight"), self.normed.ptr,
                     self.logits_bf16.ptr, 1, hidden, cfg.vocab_size,
                     output_dtype="bf16", backend=self.backend, **kw)
        bf16_to_f32(self.logits_bf16.ptr, self.logits_f32.ptr, cfg.vocab_size, **kw)

    def _decode_step_graph(self, graph_exec: int, token_id: int) -> np.ndarray:
        rt = self.runtime
        pos = self._length
        self._g_staging_host[:] = (token_id, pos, pos + 1)
        copy_host_array_to_device(self._g_staging, self._g_staging_host, runtime=rt)
        rt.graph_launch(graph_exec, self._graph_stream)
        rt.stream_synchronize(self._graph_stream)
        copy_device_to_host(self._logits_host.ctypes.data, self.logits_f32, runtime=rt)
        self._record_history(np.asarray([token_id], dtype=np.int64), pos)
        self._length = pos + 1
        # Same float32 ufunc sequence as the eager path, evaluated in place.
        logits = self._softcap_host
        np.multiply(self._logits_host, np.float32(self.config.logit_scale), out=logits)
        cap = self.config.final_logit_softcapping
        if cap:
            np.divide(logits, np.float32(cap), out=logits)
            np.tanh(logits, out=logits)
            np.multiply(np.float32(cap), logits, out=logits)
        return logits.copy()

    # -- HIP graph verify --------------------------------------------------
    def _ensure_verify_graph(self, tokens: int, tap_layers: tuple[int, ...]):
        """Capture the exact block-verify forward for a row count; replay
        removes the per-cycle host dispatch bubble (~740 eager launches, ~7 ms).
        Returns the exec handle or None (eager verify remains the fallback).
        Graphs are keyed by (tokens, tap_layers); the shipping DFlash budget
        verifies 4 or 5 rows depending on a pending root, so two graphs cover
        the default cycle. Mirrors the decode graph's dynamic-extent staging:
        the shared attention reads the live key end from a device buffer.
        """

        key = (int(tokens), tuple(int(l) for l in tap_layers))
        cache = getattr(self, "_verify_graph_exec", None)
        if cache is None:
            cache = self._verify_graph_exec = {}
        if key in cache:
            return cache[key]
        if getattr(self, "_verify_graph_failed", False):
            return None
        if not 2 <= tokens <= self._MAX_VERIFY_ROWS:
            return None
        # The fixed tap buffer is laid out for one tap_layers configuration;
        # a different one takes the eager path rather than re-laying it out.
        if getattr(self, "_vg_tap_layers", None) is not None and key[1] != self._vg_tap_layers:
            return None
        rt = self.runtime
        prev_exact = getattr(self, "_exact_multirow", False)
        try:
            from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (  # noqa: F401
                muse_glimmer_decode_attention_shared, muse_glimmer_qk_norm_rope_kv_append,
            )

            cfg = self.config
            hidden = cfg.n_embd
            n_taps = len(key[1])
            stream = rt.stream_create()
            if getattr(self, "_vg_key_end", None) is None:
                self._vg_key_end = malloc(8, runtime=rt)
                self._vg_tap_buf = malloc(
                    self._MAX_VERIFY_ROWS * max(n_taps, 1) * hidden * 4, runtime=rt)
                self._vg_logits_buf = malloc(
                    self._MAX_VERIFY_ROWS * cfg.vocab_size * 4, runtime=rt)
                self._vg_tap_layers = key[1]
            self._vg_stream = stream
            self._exact_multirow = True
            # Warm every kernel on the capture stream outside capture.
            self._vg_stage_inputs(np.zeros(tokens, dtype=np.int64), self._length, tokens)
            self._verify_graph_body(stream, tokens, key[1])
            rt.stream_synchronize(stream)
            rt.stream_begin_capture(stream, mode=0)
            try:
                self._verify_graph_body(stream, tokens, key[1])
                graph = rt.stream_end_capture(stream)
            except BaseException:
                if rt.stream_is_capturing(stream):
                    rt.stream_end_capture(stream)
                raise
            if not graph:
                raise RuntimeError("verify graph capture produced no graph")
            exec_handle = rt.graph_instantiate(graph)
            rt.graph_destroy(graph)
            cache[key] = exec_handle
            return exec_handle
        except Exception:
            self._verify_graph_failed = True
            return None
        finally:
            self._exact_multirow = prev_exact

    def _vg_stage_inputs(self, ids: np.ndarray, pos: int, tokens: int) -> None:
        rt = self.runtime
        self._ids_host[:tokens] = ids
        copy_host_array_to_device(self.token_ids, self._ids_host[:tokens], runtime=rt)
        self._pos_host[:tokens] = np.arange(pos, pos + tokens, dtype=np.int64)
        copy_host_array_to_device(self.positions, self._pos_host[:tokens], runtime=rt)
        copy_host_array_to_device(
            self._vg_key_end, np.asarray([pos + tokens], dtype=np.int64), runtime=rt)

    def _verify_graph_body(self, stream: int, tokens: int, tap_layers: tuple[int, ...]) -> None:
        """Kernel-only exact verify with graph-stable arguments.

        Mirrors the ``exact_multirow`` branch of ``_forward_block`` launch for
        launch (bit-identical row-independent projections + shared-row attention
        + fused residual boundaries). Token ids/positions live in the scratch
        staging buffers and the live key end in ``_vg_key_end``, all uploaded
        before replay; tap and logits land in fixed buffers. No host work.
        """

        from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
            muse_glimmer_add_rmsnorm_rmsnorm,
            muse_glimmer_decode_attention_shared,
            muse_glimmer_qk_norm_rope_kv_append,
        )
        from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

        cfg, rt = self.config, self.runtime
        hidden = cfg.n_embd
        q_dim, kv_dim, d = cfg.qkv_dim, cfg.kv_dim, cfg.head_dim
        kw = {"stream": stream, "runtime": rt}
        tap_index = {int(l): i for i, l in enumerate(tap_layers)}

        launch_gguf_embedding(
            self.loaded.weight("token_embd.weight"), self.token_ids.ptr,
            self.hidden_a.ptr, rows=tokens, hidden_size=hidden,
            vocab_size=cfg.vocab_size, backend=self.backend, **kw,
        )
        gemma4_rmsnorm_weightless_bf16(
            self.hidden_a.ptr, self.hidden_b.ptr, tokens, hidden, cfg.rms_eps, **kw)
        src, dst = self.hidden_b, self.hidden_a
        gemma4_rmsnorm_f32w_bf16(
            src.ptr, self.loaded.ptr("blk.0.attn_norm.weight"), self.normed.ptr,
            tokens, hidden, cfg.rms_eps, **kw)
        for layer in range(cfg.n_layers):
            p = f"blk.{layer}."
            is_swa = cfg.swa_pattern[layer]
            if layer in tap_index:
                slot = tap_index[layer]
                bf16_to_f32(
                    src.ptr, self._vg_tap_buf.ptr + slot * tokens * hidden * 4,
                    tokens * hidden, **kw)
            self._linear(self.loaded.weight(p + "attn_q.weight"), self.normed.ptr,
                         self.q_bf16.ptr, tokens, hidden, q_dim,
                         output_dtype="bf16", backend=self.backend, **kw)
            self._linear(self.loaded.weight(p + "attn_k.weight"), self.normed.ptr,
                         self.k_bf16.ptr, tokens, hidden, kv_dim,
                         output_dtype="bf16", backend=self.backend, **kw)
            self._linear(self.loaded.weight(p + "attn_v.weight"), self.normed.ptr,
                         self.v_bf16.ptr, tokens, hidden, kv_dim,
                         backend=self.backend, **kw)
            self._linear(self.loaded.weight(p + "attn_gate.weight"), self.normed.ptr,
                         self.gate_bf16.ptr, tokens, hidden, q_dim,
                         backend=self.backend, **kw)
            k_cache, v_cache = self._kv[layer]
            muse_glimmer_qk_norm_rope_kv_append(
                self.q_bf16.ptr, self.k_bf16.ptr, self.v_bf16.ptr,
                self.loaded.ptr(p + "attn_q_norm.weight"),
                self.loaded.ptr(p + "attn_k_norm.weight"),
                (self.cos_table if is_swa else self.identity_cos).ptr,
                (self.sin_table if is_swa else self.identity_sin).ptr,
                self.positions.ptr, self.q_rot_f32.ptr, k_cache.ptr, v_cache.ptr,
                eps=cfg.rms_eps, n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                head_dim=d, max_positions=self.max_context, tokens=tokens, **kw)
            muse_glimmer_decode_attention_shared(
                self.q_rot_f32.ptr, k_cache.ptr, v_cache.ptr, self.attn_out.ptr,
                self.attn_partials.ptr,
                n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                key_begin=0, key_end=self.max_context,
                split_keys=self._attn_split_config()[0], n_blocks=self._attn_split_config()[1],
                tokens=tokens, window=cfg.sliding_window if is_swa else 0,
                key_end_dev=self._vg_key_end.ptr, gate_ptr=self.gate_bf16.ptr,
                scale=float(d) ** -0.5, **kw)
            self._linear(self.loaded.weight(p + "attn_output.weight"), self.attn_out.ptr,
                         self.attn_proj.ptr, tokens, q_dim, hidden,
                         output_dtype="bf16", backend=self.backend, **kw)
            muse_glimmer_add_rmsnorm_rmsnorm(
                self.attn_proj.ptr, src.ptr,
                self.loaded.ptr(p + "post_attention_norm.weight"),
                self.loaded.ptr(p + "ffn_norm.weight"), dst.ptr, self.normed.ptr,
                rows=tokens, hidden_size=hidden, post_eps=cfg.post_norm_eps,
                pre_eps=cfg.rms_eps, **kw)
            src, dst = dst, src
            self._linear(self.loaded.weight(p + "ffn_gate.weight"), self.normed.ptr,
                         self.ff_gate.ptr, tokens, hidden, cfg.n_ff,
                         output_dtype="bf16", backend=self.backend, **kw)
            self._linear(self.loaded.weight(p + "ffn_up.weight"), self.normed.ptr,
                         self.ff_up.ptr, tokens, hidden, cfg.n_ff,
                         output_dtype="bf16", backend=self.backend, **kw)
            silu_mul_separate_out_bf16(
                self.ff_gate.ptr, self.ff_up.ptr, self.ff_act.ptr, tokens, cfg.n_ff, **kw)
            self._linear(self.loaded.weight(p + "ffn_down.weight"), self.ff_act.ptr,
                         self.ff_out.ptr, tokens, cfg.n_ff, hidden,
                         output_dtype="bf16", backend=self.backend, **kw)
            next_norm = (
                f"blk.{layer + 1}.attn_norm.weight"
                if layer + 1 < cfg.n_layers else "output_norm.weight"
            )
            muse_glimmer_add_rmsnorm_rmsnorm(
                self.ff_out.ptr, src.ptr,
                self.loaded.ptr(p + "post_ffw_norm.weight"),
                self.loaded.ptr(next_norm), dst.ptr, self.normed.ptr,
                rows=tokens, hidden_size=hidden, post_eps=cfg.post_norm_eps,
                pre_eps=cfg.rms_eps, **kw)
            src, dst = dst, src
        # output_norm already landed in ``normed`` at the last residual boundary.
        self._linear(self.loaded.weight("output.weight"), self.normed.ptr,
                     self.logits_bf16.ptr, tokens, hidden, cfg.vocab_size,
                     output_dtype="bf16", backend=self.backend, **kw)
        bf16_to_f32(self.logits_bf16.ptr, self._vg_logits_buf.ptr,
                    tokens * cfg.vocab_size, **kw)

    def _verify_with_taps_graph(self, graph_exec: int, ids: np.ndarray,
                                tap_layers: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        rt = self.runtime
        cfg = self.config
        tokens = int(ids.size)
        pos = self._length
        self._vg_stage_inputs(ids, pos, tokens)
        rt.graph_launch(graph_exec, self._vg_stream)
        rt.stream_synchronize(self._vg_stream)
        hidden = cfg.n_embd
        n_taps = len(tap_layers)
        logits_host = np.empty((tokens, cfg.vocab_size), dtype=np.float32)
        copy_device_to_host(
            logits_host.ctypes.data,
            DeviceBuffer(self._vg_logits_buf.ptr, tokens * cfg.vocab_size * 4), runtime=rt)
        logits = logits_host * np.float32(cfg.logit_scale)
        cap = cfg.final_logit_softcapping
        if cap:
            logits = np.float32(cap) * np.tanh(logits / np.float32(cap))
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        staging = np.empty((n_taps, tokens, hidden), dtype=np.float32)
        copy_device_to_host(
            staging.ctypes.data,
            DeviceBuffer(self._vg_tap_buf.ptr, n_taps * tokens * hidden * 4), runtime=rt)
        tap_host = np.ascontiguousarray(staging.transpose(1, 0, 2))
        self._record_history(ids, pos)
        self._length = pos + tokens
        return logits, tap_host

    # -- speculative (DFlash) surface --------------------------------------
    def prefill_with_taps(
        self, token_ids: Sequence[int], tap_layers: Sequence[int],
        tap_sink: Callable[[np.ndarray, int], None] | None = None,
        *, start: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prompt pass capturing layer-input taps; returns (last logits, taps).

        ``taps`` is [n_tokens, len(tap_layers), n_embd] F32 in tap_layers order
        (the DFlash feature-fusion input). With ``tap_sink`` each prefill
        chunk's taps go to ``tap_sink(taps, position_offset)`` as they are
        produced and the returned taps are empty: the whole-prompt array is
        n_taps * n_embd * 4 bytes per token (~35 GB of host memory at 256K).
        ``start`` resumes after a cached prefix as in :meth:`prefill`; taps
        then cover positions ``start`` onward only.
        """

        self._require_open()
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("prefill needs a non-empty 1-D token sequence")
        if ids.size > self.max_context:
            raise ValueError("prompt exceeds the KV cache")
        self._begin_prefill(ids, int(start))
        logits: np.ndarray | None = None
        tap_chunks: list[np.ndarray] = []
        with prefill_f16_staging_session(
            True,
            workspace_ptr=int(self._f16_staging.ptr),
            workspace_nbytes=int(self._f16_staging.nbytes),
        ):
            for begin, end in self._prefill_chunks(ids.size, int(start)):
                logits, taps = self._forward_block(
                    ids[begin:end], position_offset=begin, tap_layers=tap_layers
                )
                if tap_sink is not None:
                    tap_sink(taps, begin)
                else:
                    tap_chunks.append(taps)
        assert logits is not None
        if tap_sink is not None:
            return logits, np.empty((0, len(tap_layers), self.config.n_embd), np.float32)
        return logits, np.concatenate(tap_chunks, axis=0)

    def decode_with_taps(
        self, token_id: int, tap_layers: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_open()
        if self._length >= self.max_context:
            raise ValueError("the KV cache is full")
        return self._forward_block(
            np.asarray([int(token_id)], dtype=np.int64),
            position_offset=self._length, tap_layers=tap_layers,
        )

    def verify_with_taps(
        self, token_ids: Sequence[int], tap_layers: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append a draft block; return (logits for every row, taps).

        Exactness contract: the whole block runs in one forward whose
        projections use the bit-identical row-independent decode owners
        (weight-amortized rowtile / row-independent GEMV) and whose attention
        is decode-equivalent (per-row causal+SWA clamp at the decode split
        count). Each row's KV and logits therefore equal true AR decode
        bit-for-bit, while the weight reads amortize across the block. The
        caller compares each row's argmax against the draft and ``truncate``s
        to the accepted prefix (+1 corrected token).
        """

        self._require_open()
        self._cross_attn_split_threshold()
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("verify needs a non-empty block")
        if self._length + ids.size > self.max_context:
            raise ValueError("verify block overruns the KV cache")
        if self._image_overrides is None:
            graph = self._ensure_verify_graph(int(ids.size), tuple(int(l) for l in tap_layers))
            if graph is not None:
                return self._verify_with_taps_graph(graph, ids, tap_layers)
        logits, taps = self._forward_block(
            ids, position_offset=self._length, tap_layers=tap_layers,
            all_logits=True, exact_multirow=True,
        )
        return np.asarray(logits), np.asarray(taps)

    def truncate(self, length: int) -> None:
        """Roll the KV back to ``length`` (contiguous cache: data beyond is
        overwritten by the next append)."""

        self._require_open()
        if not 0 <= length <= self._length:
            raise ValueError("truncate must move the length backward")
        self._length = int(length)
        self._history_valid = min(self._history_valid, self._length)

    # -- context snapshots -------------------------------------------------
    @property
    def kv_geometry(self) -> dict[str, int]:
        return {"n_layers": int(self.config.n_layers), "kv_dim": int(self.config.kv_dim)}

    def save_kv(self, path: Path) -> int:
        """Write the reusable prefix's KV (see ``muse_glimmer_context``); returns its rows."""

        from hipengine.runtime.muse_glimmer_context import write_kv

        self._require_open()
        rows = int(self.history.size)
        write_kv(Path(path), self._kv, rows, self.config.kv_dim * 2, runtime=self.runtime)
        return rows

    def load_kv(self, path: Path, tokens: np.ndarray) -> None:
        """Restore ``save_kv`` output whose positions hold ``tokens``."""

        from hipengine.runtime.muse_glimmer_context import read_kv

        self._require_open()
        ids = np.asarray(tokens, dtype=np.int64)
        if ids.ndim != 1 or ids.size > self.max_context:
            raise ValueError(
                f"a {ids.size}-token snapshot does not fit the {self.max_context}-token cache"
            )
        self._length = 0
        self._history_valid = 0
        read_kv(Path(path), self._kv, ids.size, self.config.kv_dim * 2, runtime=self.runtime)
        self._history[:ids.size] = ids
        self._length = self._history_valid = int(ids.size)

    def _forward_block(
        self, ids: np.ndarray, *, position_offset: int,
        tap_layers: Sequence[int] | None = None, all_logits: bool = False,
        exact_multirow: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        cfg, rt = self.config, self.runtime
        # Exact multirow (speculative block verify): projections take the
        # bit-identical row-independent decode owners and attention uses the
        # decode split count, so every row's KV/logits match true AR decode.
        self._exact_multirow = bool(exact_multirow)
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

        tap_set = set(int(l) for l in tap_layers) if tap_layers is not None else set()
        tap_buf = None
        if tap_set:
            from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

            n_taps = len(tap_layers)
            tap_buf = malloc(tokens * n_taps * hidden * 4, runtime=rt)
        tap_index = {int(l): i for i, l in enumerate(tap_layers or ())}

        if self._exact_multirow:
            # Fused verify: the first attn pre-norm is standalone; every later
            # pre-norm is produced by the previous layer's fused residual
            # boundary (add_rmsnorm_rmsnorm), mirroring the decode graph.
            gemma4_rmsnorm_f32w_bf16(
                src.ptr, self.loaded.ptr("blk.0.attn_norm.weight"), self.normed.ptr,
                tokens, hidden, cfg.rms_eps, runtime=rt,
            )
        for layer in range(cfg.n_layers):
            p = f"blk.{layer}."
            is_swa = cfg.swa_pattern[layer]
            if layer in tap_set:
                slot = tap_index[layer]
                bf16_to_f32(
                    src.ptr, tap_buf.ptr + slot * tokens * hidden * 4,
                    tokens * hidden, runtime=rt,
                )
            if not self._exact_multirow:
                gemma4_rmsnorm_f32w_bf16(
                    src.ptr, self.loaded.ptr(p + "attn_norm.weight"), self.normed.ptr,
                    tokens, hidden, cfg.rms_eps, runtime=rt,
                )
            self._linear(
                self.loaded.weight(p + "attn_q.weight"), self.normed.ptr,
                self.q_bf16.ptr, tokens, hidden, q_dim,
                output_dtype="bf16", backend=self.backend, runtime=rt,
            )
            self._linear(
                self.loaded.weight(p + "attn_k.weight"), self.normed.ptr,
                self.k_bf16.ptr, tokens, hidden, kv_dim,
                output_dtype="bf16", backend=self.backend, runtime=rt,
            )
            self._linear(
                self.loaded.weight(p + "attn_v.weight"), self.normed.ptr,
                self.v_bf16.ptr, tokens, hidden, kv_dim,
                backend=self.backend, runtime=rt,
            )
            self._linear(
                self.loaded.weight(p + "attn_gate.weight"), self.normed.ptr,
                self.gate_bf16.ptr, tokens, hidden, q_dim,
                backend=self.backend, runtime=rt,
            )
            k_cache, v_cache = self._kv[layer]
            if self._exact_multirow:
                # Fused decode glue: Q/K head-norm + rotary + K/V append in one
                # launch, bit-identical per row to the seven-launch chain in the
                # else branch. Rotated Q lands in the decode-idle q_rot_f32
                # scratch (BF16); K/V go straight to the cache rows.
                from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
                    muse_glimmer_qk_norm_rope_kv_append,
                )
                muse_glimmer_qk_norm_rope_kv_append(
                    self.q_bf16.ptr, self.k_bf16.ptr, self.v_bf16.ptr,
                    self.loaded.ptr(p + "attn_q_norm.weight"),
                    self.loaded.ptr(p + "attn_k_norm.weight"),
                    (self.cos_table if is_swa else self.identity_cos).ptr,
                    (self.sin_table if is_swa else self.identity_sin).ptr,
                    self.positions.ptr, self.q_rot_f32.ptr, k_cache.ptr, v_cache.ptr,
                    eps=cfg.rms_eps, n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                    head_dim=d, max_positions=self.max_context, tokens=tokens,
                    runtime=rt,
                )
                q_attn_ptr = self.q_rot_f32.ptr
            else:
                from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

                bf16_to_f32(self.q_bf16.ptr, self.q_f32.ptr, tokens * q_dim, runtime=rt)
                bf16_to_f32(self.k_bf16.ptr, self.k_f32.ptr, tokens * kv_dim, runtime=rt)
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
                row_bytes = kv_dim * 2
                rt.memcpy(
                    k_cache.ptr + position_offset * row_bytes, self.k_bf16.ptr,
                    tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE,
                )
                rt.memcpy(
                    v_cache.ptr + position_offset * row_bytes, self.v_bf16.ptr,
                    tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE,
                )
                q_attn_ptr = self.q_bf16.ptr
            from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
                muse_glimmer_decode_attention_shared,
            )

            if tokens > 1 and not self._exact_multirow:
                # Bulk prefill: WMMA GQA-packed flash attention (16 Q heads
                # share each KV read; QK^T/PV on the 16x16x16 f16 WMMA).
                # Measured 1.3-1.6x over the scalar key-split owner.
                from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_flash_prefill import (
                    muse_glimmer_flash_prefill,
                )
                muse_glimmer_flash_prefill(
                    q_attn_ptr, k_cache.ptr, v_cache.ptr, self.attn_out.ptr,
                    n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                    key_begin=0, key_end=keys, tokens=tokens,
                    window=cfg.sliding_window if is_swa else 0,
                    scale=float(d) ** -0.5, runtime=rt,
                )
            else:
                # Decode and the decode-equivalent block verify share one
                # kernel: the 16 Q heads of a KV head and every verify row read
                # each K/V tile once, and absolute key splits make row i
                # bit-identical to a single decode step at position_offset+i.
                if tokens * self._DECODE_ATTN_BLOCKS > self._attn_partial_rows:
                    raise ValueError(
                        f"exact verify supports at most {self._MAX_VERIFY_ROWS} rows"
                    )
                muse_glimmer_decode_attention_shared(
                    q_attn_ptr, k_cache.ptr, v_cache.ptr, self.attn_out.ptr,
                    self.attn_partials.ptr,
                    n_q_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                    key_begin=0, key_end=keys,
                    split_keys=self._attn_split_config()[0],
                    n_blocks=self._attn_split_config()[1],
                    tokens=tokens, window=cfg.sliding_window if is_swa else 0,
                    gate_ptr=self.gate_bf16.ptr if self._exact_multirow else 0,
                    scale=float(d) ** -0.5, runtime=rt,
                )
            if not self._exact_multirow:
                # Verify fuses the sigmoid gate into the attention combine
                # (gate_ptr above, bit-identical); prefill/eager decode gate here.
                gguf_gate_mul_bf16(
                    self.attn_out.ptr, self.gate_bf16.ptr, self.attn_out.ptr,
                    tokens * q_dim, runtime=rt,
                )
            self._linear(
                self.loaded.weight(p + "attn_output.weight"), self.attn_out.ptr,
                self.attn_proj.ptr, tokens, q_dim, hidden,
                backend=self.backend, runtime=rt,
            )
            if self._exact_multirow:
                from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
                    muse_glimmer_add_rmsnorm_rmsnorm,
                )
                muse_glimmer_add_rmsnorm_rmsnorm(
                    self.attn_proj.ptr, src.ptr,
                    self.loaded.ptr(p + "post_attention_norm.weight"),
                    self.loaded.ptr(p + "ffn_norm.weight"), dst.ptr, self.normed.ptr,
                    rows=tokens, hidden_size=hidden, post_eps=cfg.post_norm_eps,
                    pre_eps=cfg.rms_eps, runtime=rt,
                )
            else:
                gemma4_add_rmsnorm_scale_bf16(
                    self.attn_proj.ptr, src.ptr,
                    self.loaded.ptr(p + "post_attention_norm.weight"), 0,
                    dst.ptr, tokens, hidden, cfg.post_norm_eps, runtime=rt,
                )
            src, dst = dst, src

            if not self._exact_multirow:
                gemma4_rmsnorm_f32w_bf16(
                    src.ptr, self.loaded.ptr(p + "ffn_norm.weight"), self.normed.ptr,
                    tokens, hidden, cfg.rms_eps, runtime=rt,
                )
            self._linear(
                self.loaded.weight(p + "ffn_gate.weight"), self.normed.ptr,
                self.ff_gate.ptr, tokens, hidden, cfg.n_ff,
                backend=self.backend, runtime=rt,
            )
            self._linear(
                self.loaded.weight(p + "ffn_up.weight"), self.normed.ptr,
                self.ff_up.ptr, tokens, hidden, cfg.n_ff,
                backend=self.backend, runtime=rt,
            )
            silu_mul_separate_out_bf16(
                self.ff_gate.ptr, self.ff_up.ptr, self.ff_act.ptr,
                tokens, cfg.n_ff, runtime=rt,
            )
            self._linear(
                self.loaded.weight(p + "ffn_down.weight"), self.ff_act.ptr,
                self.ff_out.ptr, tokens, cfg.n_ff, hidden,
                backend=self.backend, runtime=rt,
            )
            if self._exact_multirow:
                next_norm = (
                    f"blk.{layer + 1}.attn_norm.weight"
                    if layer + 1 < cfg.n_layers else "output_norm.weight"
                )
                muse_glimmer_add_rmsnorm_rmsnorm(
                    self.ff_out.ptr, src.ptr,
                    self.loaded.ptr(p + "post_ffw_norm.weight"),
                    self.loaded.ptr(next_norm), dst.ptr, self.normed.ptr,
                    rows=tokens, hidden_size=hidden, post_eps=cfg.post_norm_eps,
                    pre_eps=cfg.rms_eps, runtime=rt,
                )
            else:
                gemma4_add_rmsnorm_scale_bf16(
                    self.ff_out.ptr, src.ptr,
                    self.loaded.ptr(p + "post_ffw_norm.weight"), 0,
                    dst.ptr, tokens, hidden, cfg.post_norm_eps, runtime=rt,
                )
            src, dst = dst, src

        self._record_history(ids, position_offset)
        self._length = keys
        from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

        if all_logits:
            # Verification head: every block row (bounded by the draft block).
            # Fused verify already produced output_norm into ``normed`` at the
            # last layer's residual boundary; prefill computes it here.
            if not self._exact_multirow:
                gemma4_rmsnorm_f32w_bf16(
                    src.ptr, self.loaded.ptr("output_norm.weight"), self.normed.ptr,
                    tokens, hidden, cfg.rms_eps, runtime=rt,
                )
            self._linear(
                self.loaded.weight("output.weight"), self.normed.ptr,
                self.logits_bf16.ptr, tokens, hidden, cfg.vocab_size,
                output_dtype="bf16", backend=self.backend, runtime=rt,
            )
            logits_buf = malloc(tokens * cfg.vocab_size * 4, runtime=rt)
            try:
                bf16_to_f32(self.logits_bf16.ptr, logits_buf.ptr,
                            tokens * cfg.vocab_size, runtime=rt)
                logits_host = np.empty((tokens, cfg.vocab_size), dtype=np.float32)
                copy_device_to_host(logits_host.ctypes.data, logits_buf, runtime=rt)
            finally:
                free(logits_buf, runtime=rt)
        else:
            # Only the final row feeds generation; the head runs on that row alone.
            last_row_ptr = src.ptr + (tokens - 1) * hidden * 2
            gemma4_rmsnorm_f32w_bf16(
                last_row_ptr, self.loaded.ptr("output_norm.weight"), self.normed.ptr,
                1, hidden, cfg.rms_eps, runtime=rt,
            )
            self._linear(
                self.loaded.weight("output.weight"), self.normed.ptr,
                self.logits_bf16.ptr, 1, hidden, cfg.vocab_size,
                output_dtype="bf16", backend=self.backend, runtime=rt,
            )
            bf16_to_f32(self.logits_bf16.ptr, self.logits_f32.ptr,
                        cfg.vocab_size, runtime=rt)
            copy_device_to_host(
                self._logits_host.ctypes.data, self.logits_f32, runtime=rt
            )
            logits_host = self._logits_host[None, :]
        logits = logits_host * np.float32(cfg.logit_scale)
        cap = cfg.final_logit_softcapping
        if cap:
            logits = np.float32(cap) * np.tanh(logits / np.float32(cap))
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        if tap_layers is None:
            return logits[0] if not all_logits else logits
        if len(tap_layers) == 0:
            return (logits if all_logits else logits[0]), np.empty((tokens, 0, hidden), dtype=np.float32)
        assert tap_buf is not None
        try:
            staging = np.empty((len(tap_layers), tokens, hidden), dtype=np.float32)
            copy_device_to_host(staging.ctypes.data, tap_buf, runtime=rt)
            tap_host = np.ascontiguousarray(staging.transpose(1, 0, 2))
        finally:
            free(tap_buf, runtime=rt)
        return (logits if all_logits else logits[0]), tap_host

    # Rows-1 decode projection owners. Both are bit-identical to the
    # launch_gguf_linear decode owners they replace (dense_single_local32 Q4,
    # qmicro-planar Q6), so the exact-rows verify owners that match those
    # parents still match the AR decode KV. Wave counts are the per-launch
    # optimum measured on gfx1151 at the Glimmer shapes: grouped Q/K/V/gate
    # 1.19x at 2 waves, gate/up 1.04x at 1, attn_output 1.06x at 4, Q4 down
    # 1.02x at 1; the hoisted planar Q6 is 1.09x on down and 1.7x on narrow V.
    # Shared-row decode attention geometry, identical for AR decode and the
    # decode-equivalent verify (exactness depends on it): absolute key splits
    # of 64 keys owned round-robin by 64 blocks per KV head. On gfx1151 this
    # was the best verify setting at 8K-262K and on 2048-key SWA layers, and
    # within ~15% of the best single-row setting on full layers
    # (scripts/muse_glimmer_decode_attn_scale.py --kernel shared). Also the
    # largest verify block the partial scratch holds.
    _DECODE_ATTN_SPLIT_KEYS = 64
    _DECODE_ATTN_BLOCKS = 64
    # Short live ranges run fewer split blocks: empty blocks still launch and
    # the combine still scans, so a 512-key range on 64 blocks costs ~24 us
    # vs ~17 us on 16. The block count is chosen once per session (sticky,
    # shared by the decode graph, verify graphs, and eager fallback because
    # row bit-identity requires the same split config on both sides) and
    # crosses to the full 64 blocks once the live range outgrows the
    # threshold; both graph families re-capture together at the crossing.
    _ATTN_SHORT_THRESHOLD = 2048
    _ATTN_SHORT_MIN_BLOCKS = 16
    _MAX_VERIFY_ROWS = 32
    _ATTN_GROUP_WAVES = 2
    _ATTN_OUTPUT_WAVES = 4
    _FFN_GROUP_WAVES = 1
    _FFN_DOWN_WAVES = 1

    def _decode_linears(self, x_ptr, in_features, items, *, waves, kw):
        """Launch rows-1 projections that share ``x_ptr``.

        Weights whose quant has a registered ``linear_group`` owner run as one
        grouped launch per quant; a quant with the registered hoisted single
        owner runs that; anything else takes ``_linear``.
        """

        groups: dict[str, list[tuple[int, int, int]]] = {}
        for weight, out_ptr, out_features in items:
            quant = weight.spec.quant_key
            group_key = KernelKey(self.backend, "linear_group", quant, Q4_GROUPED_VARIANT)
            single_key = KernelKey(self.backend, "linear", quant, Q6_HOISTED_VARIANT)
            if is_registered(group_key):
                groups.setdefault(quant, []).append(
                    (weight.allocation("tiles").tensor.ptr, out_ptr, out_features)
                )
            elif is_registered(single_key):
                fn = resolve(backend=self.backend, layer="linear", quant=quant,
                             variant=Q6_HOISTED_VARIANT)
                fn(x_ptr, weight.allocation("tiles").tensor.ptr, out_ptr, 1,
                   in_features, out_features, **kw)
            else:
                self._linear(weight, x_ptr, out_ptr, 1, in_features, out_features,
                             backend=self.backend, **kw)
        for quant, segments in groups.items():
            fn = resolve(backend=self.backend, layer="linear_group", quant=quant,
                         variant=Q4_GROUPED_VARIANT)
            fn(x_ptr, segments, in_features, waves=waves, **kw)

    def _linear(self, weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        if getattr(self, "_exact_multirow", False):
            # Bit-identical row-independent decode owners (weight-amortized
            # rowtile / row-independent GEMV); never the reassociating WMMA
            # prefill family, so verify-written KV equals true AR.
            launch_gguf_linear_exact_rows(
                weight, x_ptr, out_ptr, rows, in_features, out_features,
                backend=kwargs.pop("backend", self.backend),
                runtime=kwargs.pop("runtime", self.runtime), **kwargs,
            )
            return
        # WMMA prefill opt-in: the dispatcher rewrites rows>1 raw-Q4_K/Q6_K
        # projections to the registered WMMA family; rows<=1 (decode) is a
        # no-op, so the GEMV decode path and its KV arithmetic are unchanged.
        launch_gguf_linear(
            weight, x_ptr, out_ptr, rows, in_features, out_features,
            use_wmma_prefill=True, use_q4_pack8_wmma=True, **kwargs,
        )

    # -- introspection -----------------------------------------------------
    def execution_path(self) -> str:
        return "muse_glimmer_gguf_resident_eager"


__all__ = ["DEFAULT_PREFILL_CHUNK", "MuseGlimmerResidentSession"]