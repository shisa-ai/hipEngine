"""Muse Glimmer DFlash resident drafter (gfx11, eager bring-up).

Dual-mode draft model per ``src/models/dflash.cpp`` (basic DFlash):

* ``inject``: target residual-tap features -> fc -> enc RMSNorm -> per-layer
  K/V -> K head-norm + NEOX RoPE -> draft KV at the committed positions;
* ``draft_block``: [anchor, MASK x (block-1)] through the shared target token
  table (no embedding norm), 5 pre-norm layers with QK-norm + NEOX RoPE and
  **non-causal** attention over [injected context + whole block], SwiGLU, final
  RMSNorm, shared target lm_head. Returns logits for every block row; the
  provider reads rows 1.. as the mask-position predictions.

Norm weights are raw HF tensors applied as-is (the draft conversion folds
nothing). RoPE is NEOX — the draft keeps HF layout, so no row permutation is
needed (unlike the target). The block's K/V are discarded after drafting; only
injected context persists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.cpu_reference.muse_glimmer_dflash import MuseGlimmerDFlashConfig
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_bf16_add,
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import gemma4_rmsnorm_f32w_bf16
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    load_host_array_to_device,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.muse_glimmer_gguf import (
    RAW_QUANT_KEYS,
    MuseGlimmerDeviceWeight,
    MuseGlimmerWeightSpec,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import launch_gguf_linear, launch_gguf_linear_exact_rows
from hipengine.kernels.registry import resolve
from hipengine.runtime.muse_glimmer import _rope_tables_neox

_RAW_QUANT_KEYS = RAW_QUANT_KEYS


class MuseGlimmerDFlashDrafter:
    """Resident draft weights + draft KV; shares the target's embd/head."""

    # Injection builds the draft KV from target taps over the whole prompt; it
    # owes no bit-exactness (the draft only proposes, verify corrects). Large
    # chunks let the WMMA prefill projections amortize each weight read over
    # many rows (~5x over the 8-row exact rowtile at 1024 rows), cutting the
    # linear fresh-conversation inject TTFT. Draft block stays block_size.
    _INJECT_CHUNK = 1024

    def __init__(
        self,
        reader: GGUFReader,
        config: MuseGlimmerDFlashConfig,
        *,
        target_weights,
        backend: str,
        max_context: int,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.runtime = runtime or get_hip_runtime()
        self.config = config
        self.backend = backend
        self.max_context = int(max_context)
        self.target_weights = target_weights
        self.vocab_size = int(target_weights.config.vocab_size)
        self._closed = False
        self._ctx_len = 0
        self._owned: list[DeviceBuffer] = []
        self._weights: dict[str, MuseGlimmerDeviceWeight] = {}
        self._kv: list[tuple[DeviceBuffer, DeviceBuffer]] = []
        allocations: dict[str, DeviceTensorAllocation] = {}
        rt = self.runtime
        try:
            from hipengine.loading.muse_glimmer_gguf import upload_glimmer_tensor

            for tensor in reader.info.tensors:
                name = tensor.name
                raw = np.ascontiguousarray(reader.tensor_data(name))
                weight, allocs = upload_glimmer_tensor(
                    name, raw, tensor.ggml_type_name, backend=backend, runtime=rt,
                )
                self._weights[name] = weight
                allocations.update(allocs)
            self._map = DeviceWeightMap(tensors=allocations)

            cfg = config
            h, q_dim, kv_dim, d = cfg.n_embd, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.head_dim
            b = cfg.block_size
            self.hidden_a = self._alloc(b * h * 2)
            self.hidden_b = self._alloc(b * h * 2)
            self.normed = self._alloc(b * h * 2)
            self.q_f32 = self._alloc(b * q_dim * 4)
            self.k_f32 = self._alloc(b * kv_dim * 4)
            self.q_rot_f32 = self._alloc(b * q_dim * 4)
            self.k_rot_f32 = self._alloc(b * kv_dim * 4)
            self.dummy_q = self._alloc(b * q_dim * 4)
            self.q_bf16 = self._alloc(b * q_dim * 2)
            self.k_bf16 = self._alloc(b * kv_dim * 2)
            self.v_bf16 = self._alloc(b * kv_dim * 2)
            self.attn_out = self._alloc(b * q_dim * 2)
            self.proj = self._alloc(b * h * 2)
            self.ff_gate = self._alloc(b * cfg.n_ff * 2)
            self.ff_up = self._alloc(b * cfg.n_ff * 2)
            self.ff_act = self._alloc(b * cfg.n_ff * 2)
            self.ff_out = self._alloc(b * h * 2)
            self.feat_buf = self._alloc(b * len(cfg.target_layers) * h * 2)
            self.fused = self._alloc(b * h * 4)
            self.logits_f32 = self._alloc(b * self.vocab_size * 4)
            self.logits_bf16 = self._alloc(b * self.vocab_size * 2)
            self.token_ids = self._alloc(b * 8)
            self.positions = self._alloc(b * 8)
            self.keep_mask = self._alloc(b * self.max_context)
            cos, sin = _rope_tables_neox(self.max_context, d, cfg.rope_freq_base)
            self.cos_table = self._alloc(self.max_context * d * 4)
            self.sin_table = self._alloc(self.max_context * d * 4)
            copy_host_array_to_device(self.cos_table, cos, runtime=rt)
            copy_host_array_to_device(self.sin_table, sin, runtime=rt)
            self._ids_host = np.empty(b, dtype=np.int64)
            self._pos_host = np.empty(b, dtype=np.int64)
            self._mask_host = np.empty((b, self.max_context), dtype=np.uint8)
            self._logits_host = np.empty((b, self.vocab_size), dtype=np.float32)
            # Dedicated large-chunk injection scratch (see _INJECT_CHUNK).
            ic = self._INJECT_CHUNK
            self._inj_feat = self._alloc(ic * len(cfg.target_layers) * h * 2)
            self._inj_hidden = self._alloc(ic * h * 2)
            self._inj_normed = self._alloc(ic * h * 2)
            self._inj_k_bf16 = self._alloc(ic * kv_dim * 2)
            self._inj_v_bf16 = self._alloc(ic * kv_dim * 2)
            self._inj_k_f32 = self._alloc(ic * kv_dim * 4)
            self._inj_k_rot = self._alloc(ic * kv_dim * 4)
            self._inj_dummy_q = self._alloc(ic * q_dim * 4)
            self._inj_positions = self._alloc(ic * 8)
            self._inj_pos_host = np.empty(ic, dtype=np.int64)
            for _ in range(cfg.n_layers):
                self._kv.append((
                    self._alloc(self.max_context * kv_dim * 2),
                    self._alloc(self.max_context * kv_dim * 2),
                ))
        except BaseException:
            self.close()
            raise

    def _alloc(self, nbytes: int) -> DeviceBuffer:
        buf = malloc(nbytes, runtime=self.runtime)
        self._owned.append(buf)
        return buf

    def _weight(self, name: str) -> MuseGlimmerDeviceWeight:
        return self._weights[name]

    def _ptr(self, name: str) -> int:
        return self._weights[name].allocation().tensor.ptr

    def _linear(self, x: int, name: str, out: int, rows: int, inf: int, outf: int,
                f32_out: bool = False, f32_in: bool = False) -> None:
        # Weight-amortized row-independent owners (Q4 rowtile ~198 GB/s; Q5/Q6
        # parent-parity rowtile). Measured far faster than the small-batch WMMA
        # prefill at the draft's row counts on gfx1151 (Q4 37->198 GB/s, Q5 head
        # 37->10 ms at rows=8). The draft only proposes, but these owners are
        # bit-identical to decode anyway.
        launch_gguf_linear_exact_rows(
            self._weight(name), x, out, rows, inf, outf,
            activation_dtype="f32" if f32_in else "bf16",
            output_dtype="f32" if f32_out else "bf16",
            backend=self.backend, runtime=self.runtime,
        )

    # -- lifecycle ---------------------------------------------------------
    @property
    def context_length(self) -> int:
        return self._ctx_len

    def reset(self) -> None:
        self._ctx_len = 0

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self._ctx_len:
            raise ValueError("draft truncate must move backward")
        self._ctx_len = int(length)

    def save_kv(self, path, rows: int) -> None:
        """Write the first ``rows`` draft-context rows (``muse_glimmer_context``)."""

        from hipengine.runtime.muse_glimmer_context import write_kv

        if not 0 <= rows <= self._ctx_len:
            raise ValueError("draft snapshot rows exceed the draft context")
        cfg = self.config
        write_kv(Path(path), self._kv, rows, cfg.n_kv_heads * cfg.head_dim * 2,
                 runtime=self.runtime)

    def load_kv(self, path, rows: int) -> None:
        from hipengine.runtime.muse_glimmer_context import read_kv

        if not 0 <= rows <= self.max_context:
            raise ValueError("draft snapshot exceeds the draft KV")
        cfg = self.config
        self._ctx_len = 0
        read_kv(Path(path), self._kv, rows, cfg.n_kv_heads * cfg.head_dim * 2,
                runtime=self.runtime)
        self._ctx_len = int(rows)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for buf in reversed(self._owned):
            free(buf, runtime=self.runtime)
        self._owned.clear()
        self._kv.clear()
        if getattr(self, "_map", None) is not None:
            self._map.free(runtime=self.runtime)
            self._map = None
        self._weights.clear()

    # -- injection mode ----------------------------------------------------
    def inject(self, features: np.ndarray, position_offset: int) -> None:
        """Append target-feature-derived K/V at ``position_offset``."""

        cfg, rt = self.config, self.runtime
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.size == 0:
            return
        if features.ndim == 3:
            features = features.reshape(features.shape[0], -1)
        tokens = features.shape[0]
        if features.shape[1] != len(cfg.target_layers) * cfg.n_embd:
            raise ValueError("feature width must be n_taps * n_embd")
        if position_offset != self._ctx_len:
            raise ValueError("injection must be contiguous with the draft context")
        if self._ctx_len + tokens > self.max_context:
            raise ValueError("draft KV overflow")
        if tokens > self._INJECT_CHUNK:
            # Chunk long prefill injections through the large inject scratch.
            for begin in range(0, tokens, self._INJECT_CHUNK):
                self.inject(features[begin:begin + self._INJECT_CHUNK], position_offset + begin)
            return
        h, kv_dim, d = cfg.n_embd, cfg.n_kv_heads * cfg.head_dim, cfg.head_dim
        # Taps are BF16-origin (target hidden states captured bf16->f32), so
        # the round trip back to BF16 bits is lossless and matches the pack8
        # activation contract.
        from hipengine.loading.materialize import float_array_to_bf16_bits

        bits = float_array_to_bf16_bits(features)
        copy_host_array_to_device(DeviceBuffer(self._inj_feat.ptr, bits.nbytes),
                                  bits, runtime=rt)
        self._inject_linear(self._inj_feat.ptr, "fc.weight", self._inj_hidden.ptr, tokens,
                            len(cfg.target_layers) * h, h)
        gemma4_rmsnorm_f32w_bf16(
            self._inj_hidden.ptr, self._ptr("enc.output_norm.weight"), self._inj_normed.ptr,
            tokens, h, cfg.rms_eps, runtime=rt,
        )
        self._inj_pos_host[:tokens] = np.arange(
            position_offset, position_offset + tokens, dtype=np.int64
        )
        copy_host_array_to_device(self._inj_positions, self._inj_pos_host[:tokens], runtime=rt)
        for il in range(cfg.n_layers):
            p = f"blk.{il}."
            self._inject_linear(self._inj_normed.ptr, p + "attn_k.weight", self._inj_k_bf16.ptr, tokens,
                                h, kv_dim)
            from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

            bf16_to_f32(self._inj_k_bf16.ptr, self._inj_k_f32.ptr, tokens * kv_dim, runtime=rt)
            self._inject_linear(self._inj_normed.ptr, p + "attn_v.weight", self._inj_v_bf16.ptr, tokens,
                                h, kv_dim)
            # K head-norm + NEOX rope via the fused F32 kernel; Q is a dummy
            # pass-through (same input, discarded output) since injection has
            # no query.
            gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
                self._inj_k_f32.ptr, self._inj_k_f32.ptr,
                self._ptr(p + "attn_k_norm.weight"), self._ptr(p + "attn_k_norm.weight"),
                self.cos_table.ptr, self.sin_table.ptr, self._inj_positions.ptr,
                self._inj_dummy_q.ptr, self._inj_k_rot.ptr,
                cfg.rms_eps, tokens, cfg.n_kv_heads, cfg.n_kv_heads, d, d,
                self.max_context, runtime=rt,
            )
            f32_to_bf16(self._inj_k_rot.ptr, self._inj_k_bf16.ptr, tokens * kv_dim, runtime=rt)
            k_cache, v_cache = self._kv[il]
            row_bytes = kv_dim * 2
            rt.memcpy(k_cache.ptr + position_offset * row_bytes, self._inj_k_bf16.ptr,
                      tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE)
            rt.memcpy(v_cache.ptr + position_offset * row_bytes, self._inj_v_bf16.ptr,
                      tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE)
        self._ctx_len += tokens

    def _inject_linear(self, x: int, name: str, out: int, rows: int, inf: int, outf: int) -> None:
        # Injection owes no bit-exactness (it builds the draft KV from target
        # taps; the draft only proposes and verify corrects). At the large
        # inject chunk the WMMA prefill amortizes each weight read over many
        # rows (~5x over the 8-row exact rowtile), unlike the draft block's
        # small-row exact owners.
        launch_gguf_linear(
            self._weight(name), x, out, rows, inf, outf,
            use_wmma_prefill=True, activation_dtype="bf16", output_dtype="bf16",
            backend=self.backend, runtime=self.runtime,
        )

    # -- token mode --------------------------------------------------------
    def draft_block(self, token_ids: np.ndarray) -> np.ndarray:
        """Run [anchor, MASK...] at the context edge; return [block, vocab] logits.

        The block's K/V are appended for the non-causal attention and discarded
        afterwards (the context length is restored). The block may be shorter
        than the trained block_size: only the requested mask rows are drafted,
        which cuts the draft projection compute (the draft FFN is full-width).
        """

        cfg, rt = self.config, self.runtime
        ids = np.asarray(token_ids, dtype=np.int64)
        tokens = int(ids.size)
        if not 2 <= tokens <= cfg.block_size:
            raise ValueError(f"draft block must be 2..{cfg.block_size} tokens")
        if self._ctx_len + tokens > self.max_context:
            raise ValueError("draft KV overflow")
        h, q_dim, kv_dim, d = cfg.n_embd, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.head_dim
        position_offset = self._ctx_len
        keys = position_offset + tokens

        self._ids_host[:tokens] = ids
        copy_host_array_to_device(self.token_ids, self._ids_host[:tokens], runtime=rt)
        self._pos_host[:tokens] = np.arange(position_offset, keys, dtype=np.int64)
        copy_host_array_to_device(self.positions, self._pos_host[:tokens], runtime=rt)
        # Non-causal keep mask within the SWA window over [context + block].
        # Only keys at or after the first row's window start can be kept, so
        # attention runs over that slice of the KV: the strict kernel's
        # resident logits scale with the key count, and the full context would
        # overflow its LDS past ~16K and walk masked keys for nothing.
        window = int(cfg.sliding_window)
        key_start = max(0, position_offset - window + 1) if window > 0 else 0
        span = keys - key_start
        q = np.arange(position_offset, keys, dtype=np.int64)[:, None]
        k = np.arange(key_start, keys, dtype=np.int64)[None, :]
        keep = k > q - window if window > 0 else np.ones((tokens, span), dtype=bool)
        mask = self._mask_host[:tokens, :span]
        np.copyto(mask, keep.astype(np.uint8))
        copy_host_array_to_device(
            DeviceBuffer(self.keep_mask.ptr, mask.nbytes),
            np.ascontiguousarray(mask), runtime=rt,
        )

        launch_gguf_embedding(
            self.target_weights.weight("token_embd.weight"),
            self.token_ids.ptr, self.hidden_a.ptr, tokens, h,
            self.target_weights.config.vocab_size,
            backend=self.backend, runtime=rt,
        )
        src, dst = self.hidden_a, self.hidden_b
        for il in range(cfg.n_layers):
            p = f"blk.{il}."
            gemma4_rmsnorm_f32w_bf16(
                src.ptr, self._ptr(p + "attn_norm.weight"), self.normed.ptr,
                tokens, h, cfg.rms_eps, runtime=rt,
            )
            self._linear(self.normed.ptr, p + "attn_q.weight", self.q_bf16.ptr, tokens,
                         h, q_dim)
            self._linear(self.normed.ptr, p + "attn_k.weight", self.k_bf16.ptr, tokens,
                         h, kv_dim)
            from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

            bf16_to_f32(self.q_bf16.ptr, self.q_f32.ptr, tokens * q_dim, runtime=rt)
            bf16_to_f32(self.k_bf16.ptr, self.k_f32.ptr, tokens * kv_dim, runtime=rt)
            self._linear(self.normed.ptr, p + "attn_v.weight", self.v_bf16.ptr, tokens,
                         h, kv_dim)
            gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
                self.q_f32.ptr, self.k_f32.ptr,
                self._ptr(p + "attn_q_norm.weight"), self._ptr(p + "attn_k_norm.weight"),
                self.cos_table.ptr, self.sin_table.ptr, self.positions.ptr,
                self.q_rot_f32.ptr, self.k_rot_f32.ptr,
                cfg.rms_eps, tokens, cfg.n_heads, cfg.n_kv_heads, d, d,
                self.max_context, runtime=rt,
            )
            f32_to_bf16(self.q_rot_f32.ptr, self.q_bf16.ptr, tokens * q_dim, runtime=rt)
            f32_to_bf16(self.k_rot_f32.ptr, self.k_bf16.ptr, tokens * kv_dim, runtime=rt)
            k_cache, v_cache = self._kv[il]
            row_bytes = kv_dim * 2
            rt.memcpy(k_cache.ptr + position_offset * row_bytes, self.k_bf16.ptr,
                      tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE)
            rt.memcpy(v_cache.ptr + position_offset * row_bytes, self.v_bf16.ptr,
                      tokens * row_bytes, MemcpyKind.DEVICE_TO_DEVICE)
            gemma4_attention_prefill_bf16(
                self.q_bf16.ptr, k_cache.ptr + key_start * row_bytes,
                v_cache.ptr + key_start * row_bytes, self.keep_mask.ptr,
                self.attn_out.ptr,
                tokens=tokens, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads,
                head_dim=d, scale=float(d) ** -0.5, keys=span,
                window=cfg.sliding_window, row_offset=position_offset - key_start,
                runtime=rt,
            )
            self._linear(self.attn_out.ptr, p + "attn_output.weight", self.proj.ptr,
                         tokens, q_dim, h)
            gguf_bf16_add(src.ptr, self.proj.ptr, dst.ptr, tokens * h, runtime=rt)
            src, dst = dst, src
            gemma4_rmsnorm_f32w_bf16(
                src.ptr, self._ptr(p + "ffn_norm.weight"), self.normed.ptr,
                tokens, h, cfg.rms_eps, runtime=rt,
            )
            self._linear(self.normed.ptr, p + "ffn_gate.weight", self.ff_gate.ptr,
                         tokens, h, cfg.n_ff)
            self._linear(self.normed.ptr, p + "ffn_up.weight", self.ff_up.ptr,
                         tokens, h, cfg.n_ff)
            silu_mul_separate_out_bf16(
                self.ff_gate.ptr, self.ff_up.ptr, self.ff_act.ptr, tokens, cfg.n_ff,
                runtime=rt,
            )
            self._linear(self.ff_act.ptr, p + "ffn_down.weight", self.ff_out.ptr,
                         tokens, cfg.n_ff, h)
            gguf_bf16_add(src.ptr, self.ff_out.ptr, dst.ptr, tokens * h, runtime=rt)
            src, dst = dst, src
        gemma4_rmsnorm_f32w_bf16(
            src.ptr, self._ptr("output_norm.weight"), self.normed.ptr,
            tokens, h, cfg.rms_eps, runtime=rt,
        )
        head_weight = self.target_weights.weight("output.weight")
        # The draft head only proposes tokens; it owes no bit-exactness to the
        # target decode. Prefer the faster single-wave production rowtile when
        # the head quant declares it (~1.8x over the parent-parity rowtile at
        # block rows); fall back to the exact rows owner otherwise. Verify stays
        # exact, so DFlash IDs still equal true AR regardless of draft numerics.
        head_fn = resolve(
            backend=self.backend, layer="linear", quant=head_weight.spec.quant_key,
            variant="t16_gemv_rowtile_single_wave_bf16_bf16_out", missing="none",
        )
        if head_fn is not None and 2 <= tokens <= 8:
            head_fn(
                self.normed.ptr, head_weight.allocation("tiles").tensor.ptr,
                self.logits_bf16.ptr, tokens, h, self.vocab_size, runtime=rt,
            )
        else:
            launch_gguf_linear_exact_rows(
                head_weight, self.normed.ptr,
                self.logits_bf16.ptr, tokens, h, self.vocab_size,
                output_dtype="bf16", backend=self.backend, runtime=rt,
            )
        from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32

        bf16_to_f32(self.logits_bf16.ptr, self.logits_f32.ptr,
                    tokens * self.vocab_size, runtime=rt)
        copy_device_to_host(
            self._logits_host.ctypes.data,
            DeviceBuffer(self.logits_f32.ptr, tokens * self.vocab_size * 4), runtime=rt,
        )
        # Discard the block KV; only injected context persists.
        return np.ascontiguousarray(self._logits_host[:tokens], dtype=np.float32)


__all__ = ["MuseGlimmerDFlashDrafter"]