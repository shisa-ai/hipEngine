"""Native GPU EAGLE-4 feature-autoregression drafter (clean-room).

Single Llama layer drafts at the feature level: fusion projects the three target
aux hidden states (each RMS-normed) to H; the layer attends over
concat(input_layernorm(embed), hidden_norm(feature)) (2H input, standard MHA with
RoPE and no q/k norm), then fused add+norm + SwiGLU; output norm(residual+mlp)
feeds the next step. The efficient head retrieves top-K candidates from a low-rank
estimate and re-scores the shortlist with full head rows. Reuses the DFlash2 native
helpers + DFlash dense/norm/attention primitives; EAGLE-specific RoPE-only + top-K
kernels are kernels/hip_gfx1100/speculative/eagle4.py.

Sequence-forward chain draft for bring-up parity checks; the production path
is a persistent-KV context drafter (prefill + rows=1 decode + append committed)
with a device-resident efficient head and draft step (no per-step host
round-trips). A dynamic tree is a speed follow-up.
"""
from __future__ import annotations

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_array_to_device, host_array_ptr
from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter, bf16_to_f32
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter, dflash_add_bf16, dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32, dflash_dense_bf16_to_f32_wmma, dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash2 import (
    build_dflash2, dflash2_sliding_attention_f32_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative.eagle4 import build_eagle4, eagle4_rope, eagle4_topk, eagle4_concat, eagle4_concat3
from hipengine.kernels.hip_gfx1100.convert.cast import build_cast, f32_to_bf16
from hipengine.kernels.hip_gfx1100.convert.gather import (
    build_gather, gather_bf16_rows_by_i32id, gather_i32_by_i32id,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import argmax_f32_rows_i32, build_lm_head
from hipengine.speculative.dflash2_native import DFlash2NativeDrafter, _to_bf16_bits


class EAGLE4NativeDrafter(DFlash2NativeDrafter):
    """EAGLE-4 single-layer feature drafter with efficient low-rank head."""

    def __init__(self, config: dict, weights: dict[str, np.ndarray], *,
                 runtime: HipRuntime | None = None, max_context_len: int = 4096,
                 max_seq: int = 64):
        self.runtime = runtime or get_hip_runtime()
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.q_features = self.num_heads * self.head_dim
        self.kv_features = self.num_kv_heads * self.head_dim
        self.intermediate_size = int(config["intermediate_size"])
        self.eps = float(config["rms_norm_eps"])
        self.fc_norm = bool(config.get("fc_norm", False))
        self.rope_theta = float(config.get("rope_parameters", {}).get("rope_theta", config.get("rope_theta", 1e7)))
        self.sliding_window = int(config.get("sliding_window") or 0)
        self.vocab_size = int(config["vocab_size"])
        hc = config["efficient_lm_head_config"]
        self.rank = int(hc["rank"])
        self.num_candidates = int(hc["num_candidates"])
        self.max_seq = int(max_seq)
        self.max_context_len = int(max_context_len)
        self.threads = 128
        self._lib1 = build_dflash_drafter(load=True)
        self._lib2 = build_dflash2(load=True)
        self._lib3 = build_eagle4(load=True)
        self._lib_gather = build_gather(load=True)
        self._lib_lm_head = build_lm_head(load=True)
        self._lib_cast = build_cast(load=True)
        self._numpy = EAGLE4NumpyDrafter(config, weights)
        max_wbytes = max((int(np.ascontiguousarray(a).nbytes) for a in weights.values()), default=0)
        # Cap the staging buffer; _staged_h2d chunks larger uploads. EAGLE-4's
        # lm_head/embed_tokens are multi-GB, so sizing to max_wbytes exhausts RAM.
        staging_bytes = int(np.ceil(min(max(max_wbytes, 16 * 1024 * 1024), 64 * 1024 * 1024) / 4)) * 4
        self._staging_np = np.empty(staging_bytes // 4, dtype=np.float32)
        self._staging_nbytes = staging_bytes
        self._bufs: list = []

        def _buf(nbytes):
            b = DeviceBuffer(ptr=self.runtime.malloc(nbytes), nbytes=nbytes)
            self._bufs.append(b)
            return b

        self.wdev: dict[str, int] = {}
        for name, arr in weights.items():
            if name.endswith("_version"):
                continue
            a = _to_bf16_bits(np.asarray(arr, dtype=np.float32))
            b = _buf(a.nbytes)
            self._staged_h2d(b.ptr, a)
            self.wdev[name] = b.ptr
        T = self.max_seq
        # Prefill runs the Q/K/V projection over the whole prompt (Tp rows); the
        # FFN + efficient head only ever run on the decode batch (T rows), so the
        # vocab-sized head scratch stays small. The persistent drafter K/V cache
        # is sized to the full context.
        Tp = self.max_context_len
        H, QF, KVF, I = self.hidden_size, self.q_features, self.kv_features, self.intermediate_size
        self.embed_seq = _buf(Tp * H * 2)
        self.feature_seq = _buf(Tp * H * 2)
        self.x_concat = _buf(Tp * 2 * H * 2)
        self.norm_embed = _buf(Tp * H * 2)
        self.norm_hidden = _buf(Tp * H * 2)
        self.q_out = _buf(Tp * QF * 4)
        self.k_out = _buf(Tp * KVF * 4)
        self.v_out = _buf(Tp * KVF * 2)
        self.attn_out = _buf(T * H * 2)
        self.o_proj_out = _buf(T * H * 2)
        self.residual = _buf(T * H * 2)
        self.hn = _buf(T * H * 2)
        self.mlp_gate = _buf(T * I * 2)
        self.mlp_up = _buf(T * I * 2)
        self.mlp_out = _buf(T * H * 2)
        self.summed = _buf(T * H * 2)
        self.out_feature = _buf(T * H * 2)
        self.positions_buf = _buf(Tp * 4)
        # Persistent drafter K/V cache (RoPE'd K f32, V bf16) + absolute positions.
        self.k_cache = _buf(Tp * KVF * 4)
        self.v_cache = _buf(Tp * KVF * 2)
        self.kv_positions = _buf(Tp * 4)
        # Key positions are always the identity arange (position i holds key i),
        # so fill once; decode steps read their query position from it via a
        # 4-byte D2D instead of a per-step H2D.
        self._staged_h2d(self.kv_positions.ptr, np.arange(Tp, dtype=np.int32))
        self._ctx_len = 0
        self._feature_ready = False
        # Efficient head scratch.
        self.rank_vec = _buf(T * self.rank * 2)
        self.scores = _buf(T * self.vocab_size * 4)
        self.scores_work = _buf(T * self.vocab_size * 4)
        self.cand_ids = _buf(T * self.num_candidates * 4)
        self.head_rows = _buf(T * self.num_candidates * H * 2)
        self.shortlist = _buf(T * self.num_candidates * 4)
        # Device-resident head outputs: top-1 token id, shortlist argmax index,
        # and the per-step draft id row the embedding gather reads directly.
        self.head_token = _buf(4)
        self.head_pick = _buf(4)
        self.draft_ids_dev = _buf(T * 4)
        nblk = max(1, (self.num_candidates + 128 * 4 - 1) // (128 * 4))
        self.argmax_block_values = _buf(nblk * 4)
        self.argmax_block_indices = _buf(nblk * 4)
        self.fc_normed = _buf(T * 3 * H * 2)
        self.fc_out = _buf(T * H * 2)
        # Device-resident fc projection scratch (append_committed_device): the
        # three aux chunks are cast f32->bf16, RMS-normed, concat'd to [T,3H],
        # then projected by fc.weight into feature_seq. Sized to the decode batch
        # (T=max_seq); prefill_context projects on the host (once per request).
        self.fc_chunk = [_buf(T * H * 2) for _ in range(3)]
        self.fc_normed_chunk = [_buf(T * H * 2) for _ in range(3)]

    def _w(self, name: str) -> int:
        return self.wdev[name]

    def _qkv(self, T: int) -> None:
        """Input norms + concat + Q/K/V projection + RoPE for T rows."""
        H, QF, KVF = self.hidden_size, self.q_features, self.kv_features
        L = "layers.0"
        dflash_rmsnorm_bf16(self.embed_seq.ptr, self._w(f"{L}.input_layernorm.weight"), self.norm_embed.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.feature_seq.ptr, self._w(f"{L}.hidden_norm.weight"), self.norm_hidden.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        eagle4_concat(self.norm_embed.ptr, self.norm_hidden.ptr, self.x_concat.ptr, T, H, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.x_concat.ptr, self._w(f"{L}.self_attn.q_proj.weight"), self.q_out.ptr, T, 2 * H, QF, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.x_concat.ptr, self._w(f"{L}.self_attn.k_proj.weight"), self.k_out.ptr, T, 2 * H, KVF, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.x_concat.ptr, self._w(f"{L}.self_attn.v_proj.weight"), self.v_out.ptr, T, 2 * H, KVF, library=self._lib1, runtime=self.runtime)
        eagle4_rope(self.q_out.ptr, self.positions_buf.ptr, T, self.num_heads, self.head_dim, self.rope_theta, runtime=self.runtime)
        eagle4_rope(self.k_out.ptr, self.positions_buf.ptr, T, self.num_kv_heads, self.head_dim, self.rope_theta, runtime=self.runtime)

    def _attend(self, T: int, kv_len: int, k_ptr: int, v_ptr: int, k_pos_ptr: int) -> None:
        """Causal GQA attention: T query rows against kv_len keys/values."""
        dflash2_sliding_attention_f32_bf16(self.q_out.ptr, k_ptr, v_ptr, self.positions_buf.ptr, k_pos_ptr, self.attn_out.ptr, 1, T, kv_len, self.num_heads, self.num_kv_heads, self.head_dim, sliding_window=self.sliding_window, is_causal=True, library=self._lib2, runtime=self.runtime)

    def _ffn(self, T: int) -> None:
        """o-proj + residual + SwiGLU + output norm for T rows."""
        H, QF, I = self.hidden_size, self.q_features, self.intermediate_size
        L = "layers.0"
        dflash_dense_bf16_to_bf16(self.attn_out.ptr, self._w(f"{L}.self_attn.o_proj.weight"), self.o_proj_out.ptr, T, QF, H, library=self._lib1, runtime=self.runtime)
        dflash_add_bf16(self.o_proj_out.ptr, self.feature_seq.ptr, self.residual.ptr, T * H, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.residual.ptr, self._w(f"{L}.post_attention_layernorm.weight"), self.hn.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.hn.ptr, self._w(f"{L}.mlp.gate_proj.weight"), self.mlp_gate.ptr, T, H, I, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.hn.ptr, self._w(f"{L}.mlp.up_proj.weight"), self.mlp_up.ptr, T, H, I, library=self._lib1, runtime=self.runtime)
        dflash_silu_mul_bf16(self.mlp_gate.ptr, self.mlp_up.ptr, self.mlp_gate.ptr, T * I, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.mlp_gate.ptr, self._w(f"{L}.mlp.down_proj.weight"), self.mlp_out.ptr, T, I, H, library=self._lib1, runtime=self.runtime)
        dflash_add_bf16(self.mlp_out.ptr, self.residual.ptr, self.summed.ptr, T * H, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.summed.ptr, self._w("norm.weight"), self.out_feature.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)

    def _layer(self, T: int) -> None:
        self._qkv(T)
        self._attend(T, T, self.k_out.ptr, self.v_out.ptr, self.positions_buf.ptr)
        self._ffn(T)

    def _efficient_head_last(self, T: int, token_out_ptr: int) -> None:
        """Device-resident efficient head over row T-1 of ``out_feature``.

        Low-rank retrieve -> top-K -> BF16 shortlist gather from the resident
        lm_head -> re-score -> row argmax -> token-id pick. No sync, D2H, or host
        gather; the top-1 token id lands at ``token_out_ptr`` (device i32).
        """
        H, V, R, C = self.hidden_size, self.vocab_size, self.rank, self.num_candidates
        last_ptr = self.out_feature.ptr + (T - 1) * H * 2
        dflash_dense_bf16_to_bf16(last_ptr, self._w("efficient_lm_head_down_proj.weight"), self.rank_vec.ptr, 1, H, R, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.rank_vec.ptr, self._w("efficient_lm_head.weight"), self.scores.ptr, 1, R, V, library=self._lib1, runtime=self.runtime)
        self._d2d(self.scores_work.ptr, self.scores.ptr, V * 4)
        eagle4_topk(self.scores_work.ptr, self.cand_ids.ptr, 1, V, C, runtime=self.runtime)
        gather_bf16_rows_by_i32id(self._w("lm_head.weight"), self.cand_ids.ptr, self.head_rows.ptr, C, H, V, library=self._lib_gather, runtime=self.runtime)
        dflash_dense_bf16_to_f32(last_ptr, self.head_rows.ptr, self.shortlist.ptr, 1, H, C, library=self._lib1, runtime=self.runtime)
        argmax_f32_rows_i32(self.shortlist.ptr, self.argmax_block_values.ptr, self.argmax_block_indices.ptr, self.head_pick.ptr, None, 1, C, threads=128, library=self._lib_lm_head, runtime=self.runtime)
        gather_i32_by_i32id(self.cand_ids.ptr, self.head_pick.ptr, token_out_ptr, 1, C, library=self._lib_gather, runtime=self.runtime)

    def draft_greedy_chain(self, target_aux: np.ndarray, root_token: int, root_pos: int, n_draft: int):
        H = self.hidden_size
        feature = self._numpy.project_target_features(target_aux[None, :])[0]
        features = [feature]; tokens = [int(root_token)]; draft_ids = []
        pos = int(root_pos)
        for _ in range(n_draft):
            T = len(features)
            self._staged_h2d(self.feature_seq.ptr, _to_bf16_bits(np.stack(features)))
            embed = self._numpy.w["embed_tokens.weight"][np.array(tokens, dtype=np.int64)]
            self._staged_h2d(self.embed_seq.ptr, _to_bf16_bits(embed))
            positions = np.arange(pos - T + 1, pos + 1, dtype=np.int32)
            self._staged_h2d(self.positions_buf.ptr, positions)
            self._layer(T)
            self._efficient_head_last(T, self.head_token.ptr)
            next_token = int(self._d2h(self.head_token.ptr, (1,), np.int32)[0])
            out_u16 = self._d2h(self.out_feature.ptr + (T - 1) * H * 2, (H,), np.uint16)
            feature = bf16_to_f32(out_u16)
            draft_ids.append(next_token); features.append(feature); tokens.append(next_token); pos += 1
        return np.array(draft_ids, dtype=np.int64), np.stack(features, axis=0)

    # -- persistent-KV context drafting -----------------------------------

    def _store_kv(self, base: int, rows: int) -> None:
        """Copy RoPE'd K/V from k_out/v_out into the persistent cache at ``base``."""
        KVF = self.kv_features
        self._d2d(self.k_cache.ptr + base * KVF * 4, self.k_out.ptr, rows * KVF * 4)
        self._d2d(self.v_cache.ptr + base * KVF * 2, self.v_out.ptr, rows * KVF * 2)

    def _finish_last_feature(self, q_row: int, query_pos: int, kv_len: int, feature_src_ptr: int) -> None:
        """Attend a single query row (q_out row ``q_row``, absolute position
        ``query_pos``) against the cache and run the FFN. The normed output
        feature stays device-resident in ``out_feature`` row 0;
        ``feature_src_ptr`` points at the input hidden row used as residual."""
        self._d2d(self.q_out.ptr, self.q_out.ptr + q_row * self.q_features * 4, self.q_features * 4)
        self._d2d(self.positions_buf.ptr, self.kv_positions.ptr + query_pos * 4, 4)
        self._attend(1, kv_len, self.k_cache.ptr, self.v_cache.ptr, self.kv_positions.ptr)
        self._d2d(self.feature_seq.ptr, feature_src_ptr, self.hidden_size * 2)
        self._ffn(1)

    def prefill_context(self, ctx_aux: np.ndarray, prompt_ids, bonus: int) -> None:
        """Build the drafter K/V cache over the prompt context.

        ``ctx_aux`` [P, 3H] is the target aux concat at each prompt position. EAGLE
        pairs feature_i with token_{i+1}, so the input tokens are prompt_ids[1:] +
        [bonus]. Stores K/V for positions 0..P-1 and leaves the last-position
        output feature device-resident in ``out_feature`` (it predicts the first
        draft token).
        """
        P = len(prompt_ids)
        if P > self.max_context_len:
            raise ValueError(f"prompt length {P} exceeds drafter max_context_len {self.max_context_len}")
        features = self._numpy.project_target_features(np.asarray(ctx_aux, dtype=np.float32))
        tokens = np.array(list(prompt_ids[1:]) + [int(bonus)], dtype=np.int64)
        self._staged_h2d(self.feature_seq.ptr, _to_bf16_bits(features))
        self._staged_h2d(self.embed_seq.ptr, _to_bf16_bits(self._numpy.w["embed_tokens.weight"][tokens]))
        positions = np.arange(P, dtype=np.int32)
        self._staged_h2d(self.positions_buf.ptr, positions)
        self._qkv(P)
        self._store_kv(0, P)
        self._ctx_len = P
        self._finish_last_feature(P - 1, P - 1, P, self.feature_seq.ptr + (P - 1) * self.hidden_size * 2)
        self._feature_ready = True

    def draft_greedy_chain_cached(self, n_draft: int):
        """Draft ``n_draft`` tokens as device-resident rows=1 decodes against the
        persistent K/V cache. Features, token ids, embeddings, and positions stay
        on device; a single D2H returns the draft ids at the end. Requires
        ``prefill_context``/``append_committed`` first; committed features are
        re-derived from the target aux by ``append_committed``."""
        H, V = self.hidden_size, self.vocab_size
        if not self._feature_ready:
            raise RuntimeError("draft_greedy_chain_cached requires prefill_context first")
        if not 1 <= n_draft <= self.max_seq:
            raise ValueError(f"n_draft {n_draft} outside [1, {self.max_seq}]")
        for step in range(n_draft):
            token_ptr = self.draft_ids_dev.ptr + step * 4
            self._efficient_head_last(1, token_ptr)
            if step < n_draft - 1:
                pos = self._ctx_len + step
                self._d2d(self.feature_seq.ptr, self.out_feature.ptr, H * 2)
                gather_bf16_rows_by_i32id(self._w("embed_tokens.weight"), token_ptr, self.embed_seq.ptr, 1, H, V, library=self._lib_gather, runtime=self.runtime)
                self._d2d(self.positions_buf.ptr, self.kv_positions.ptr + pos * 4, 4)
                self._qkv(1)
                self._store_kv(pos, 1)
                self._attend(1, pos + 1, self.k_cache.ptr, self.v_cache.ptr, self.kv_positions.ptr)
                self._ffn(1)
        ids = self._d2h(self.draft_ids_dev.ptr, (n_draft,), np.int32)
        return ids.astype(np.int64), None

    def append_committed(self, new_aux: np.ndarray, next_tokens) -> None:
        """Append K/V for committed rows using the TARGET aux (not draft features).

        ``new_aux`` [m, 3H] is the target aux concat for the m committed positions;
        ``next_tokens`` [m] is the input token at each (token_{i+1} alignment).
        Overwrites the temporary draft K/V from the finished block, advances
        ``_ctx_len``, and leaves the last-position feature device-resident for
        the next block's first draft.
        """
        m = int(new_aux.shape[0])
        base = self._ctx_len
        if base + m > self.max_context_len:
            raise ValueError(f"context length {base + m} exceeds drafter max_context_len {self.max_context_len}")
        features = self._numpy.project_target_features(np.asarray(new_aux, dtype=np.float32))
        tokens = np.asarray(next_tokens, dtype=np.int64)
        self._staged_h2d(self.feature_seq.ptr, _to_bf16_bits(features))
        self._staged_h2d(self.embed_seq.ptr, _to_bf16_bits(self._numpy.w["embed_tokens.weight"][tokens]))
        positions = np.arange(base, base + m, dtype=np.int32)
        self._staged_h2d(self.positions_buf.ptr, positions)
        self._qkv(m)
        self._store_kv(base, m)
        self._ctx_len = base + m
        self._finish_last_feature(m - 1, base + m - 1, self._ctx_len, self.feature_seq.ptr + (m - 1) * self.hidden_size * 2)
        self._feature_ready = True

    def _project_aux_device(self, aux_dev, m: int) -> None:
        """Project ``m`` committed target-aux rows into ``feature_seq`` on device.

        ``aux_dev`` is the list of NUM_AUX device FP32 buffers (in aux-layer
        order), each ``[rows, H]`` row-major; only the first ``m`` rows are read.
        Mirrors numpy ``project_target_features``: cast each chunk f32->bf16,
        RMS-norm per chunk when ``fc_norm``, concat to ``[m, 3H]``, fc GEMM -> H.
        Replaces the host projection + aux D2H + feature H2D on the hot path.
        """
        H = self.hidden_size
        if m > self.max_seq:
            raise ValueError(f"device projection rows {m} exceed max_seq {self.max_seq}")
        src = []
        for i in range(3):
            f32_to_bf16(int(aux_dev[i]), self.fc_chunk[i].ptr, m * H, library=self._lib_cast, runtime=self.runtime)
            if self.fc_norm:
                dflash_rmsnorm_bf16(self.fc_chunk[i].ptr, self._w(f"fc_norm.{i}.weight"), self.fc_normed_chunk[i].ptr, m, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
                src.append(self.fc_normed_chunk[i].ptr)
            else:
                src.append(self.fc_chunk[i].ptr)
        eagle4_concat3(src[0], src[1], src[2], self.fc_normed.ptr, m, H, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.fc_normed.ptr, self._w("fc.weight"), self.feature_seq.ptr, m, 3 * H, H, library=self._lib1, runtime=self.runtime)

    def append_committed_device(self, aux_dev, produced: int, next_tokens) -> None:
        """Device-resident variant of :meth:`append_committed`.

        ``aux_dev`` is the list of NUM_AUX device FP32 aux buffers from the target
        verify (``layer_output_hidden_device`` in aux-layer order); ``produced``
        is the number of committed rows (the first ``produced`` rows of each).
        The aux never leaves the device: the fc projection runs on device and
        writes ``feature_seq`` directly, skipping the host projection + D2H/H2D.
        """
        m = int(produced)
        base = self._ctx_len
        if base + m > self.max_context_len:
            raise ValueError(f"context length {base + m} exceeds drafter max_context_len {self.max_context_len}")
        self._project_aux_device(aux_dev, m)
        tokens = np.asarray(next_tokens, dtype=np.int64)
        self._staged_h2d(self.embed_seq.ptr, _to_bf16_bits(self._numpy.w["embed_tokens.weight"][tokens]))
        self._staged_h2d(self.positions_buf.ptr, np.arange(base, base + m, dtype=np.int32))
        self._qkv(m)
        self._store_kv(base, m)
        self._ctx_len = base + m
        self._finish_last_feature(m - 1, base + m - 1, self._ctx_len, self.feature_seq.ptr + (m - 1) * self.hidden_size * 2)
        self._feature_ready = True