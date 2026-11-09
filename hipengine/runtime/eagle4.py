"""Native GPU EAGLE-4 feature-autoregression drafter (clean-room).

Single Llama layer drafts at the feature level: fusion projects the three target
aux hidden states (each RMS-normed) to H; the layer attends over
concat(input_layernorm(embed), hidden_norm(feature)) (2H input, standard MHA with
RoPE and no q/k norm), then fused add+norm + SwiGLU; output norm(residual+mlp)
feeds the next step. The efficient head retrieves top-K candidates from a low-rank
estimate and re-scores the shortlist with full head rows. Reuses the DFlash2 native
helpers + DFlash dense/norm/attention primitives; EAGLE-specific RoPE-only + top-K
kernels are kernels/hip_gfx1100/speculative/eagle4.py.

Sequence-forward chain draft for bring-up (recompute attention over the draft
sequence each step); a KV-cache decode + dynamic tree are speed follow-ups.
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
from hipengine.kernels.hip_gfx1100.speculative.eagle4 import build_eagle4, eagle4_rope, eagle4_topk, eagle4_concat
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
        self._ctx_len = 0
        self._last_feature: np.ndarray | None = None
        # Efficient head scratch.
        self.rank_vec = _buf(T * self.rank * 2)
        self.scores = _buf(T * self.vocab_size * 4)
        self.scores_work = _buf(T * self.vocab_size * 4)
        self.cand_ids = _buf(T * self.num_candidates * 4)
        self.head_rows = _buf(T * self.num_candidates * H * 2)
        self.shortlist = _buf(T * self.num_candidates * 4)
        self.fc_normed = _buf(T * 3 * H * 2)
        self.fc_out = _buf(T * H * 2)

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

    def _efficient_head_last(self, T: int, k: int = 1):
        H, V, R, C = self.hidden_size, self.vocab_size, self.rank, self.num_candidates
        last_ptr = self.out_feature.ptr + (T - 1) * H * 2
        dflash_dense_bf16_to_bf16(last_ptr, self._w("efficient_lm_head_down_proj.weight"), self.rank_vec.ptr, 1, H, R, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.rank_vec.ptr, self._w("efficient_lm_head.weight"), self.scores.ptr, 1, R, V, library=self._lib1, runtime=self.runtime)
        self._d2d(self.scores_work.ptr, self.scores.ptr, V * 4)
        eagle4_topk(self.scores_work.ptr, self.cand_ids.ptr, 1, V, C, runtime=self.runtime)
        self.runtime.device_synchronize()
        cand = self._d2h(self.cand_ids.ptr, (C,), np.int32)
        head_rows = _to_bf16_bits(self._numpy.w["lm_head.weight"][cand])
        self._staged_h2d(self.head_rows.ptr, head_rows)
        dflash_dense_bf16_to_f32(last_ptr, self.head_rows.ptr, self.shortlist.ptr, 1, H, C, library=self._lib1, runtime=self.runtime)
        self.runtime.device_synchronize()
        logits = self._d2h(self.shortlist.ptr, (C,), np.float32)
        probs = np.exp(logits - np.max(logits)); probs = probs / np.sum(probs)
        pick = np.argsort(-probs)[:k]
        return probs[pick][None, :], cand[pick][None, :]

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
            _probs, ids = self._efficient_head_last(T, k=1)
            next_token = int(ids[0, 0])
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

    def _last_position_feature(self, q_row: int, query_pos: int, kv_len: int, residual_row: np.ndarray) -> np.ndarray:
        """Attend a single query row (q_out row ``q_row``, absolute position
        ``query_pos``) against the cache, run the FFN, and return the normed output
        feature. ``residual_row`` [1, H] is the input hidden for that position."""
        H = self.hidden_size
        self._d2d(self.q_out.ptr, self.q_out.ptr + q_row * self.q_features * 4, self.q_features * 4)
        self._staged_h2d(self.positions_buf.ptr, np.array([query_pos], dtype=np.int32))
        self._attend(1, kv_len, self.k_cache.ptr, self.v_cache.ptr, self.kv_positions.ptr)
        self._staged_h2d(self.feature_seq.ptr, _to_bf16_bits(residual_row))
        self._ffn(1)
        out_u16 = self._d2h(self.out_feature.ptr, (H,), np.uint16)
        return bf16_to_f32(out_u16)

    def prefill_context(self, ctx_aux: np.ndarray, prompt_ids, bonus: int) -> None:
        """Build the drafter K/V cache over the prompt context.

        ``ctx_aux`` [P, 3H] is the target aux concat at each prompt position. EAGLE
        pairs feature_i with token_{i+1}, so the input tokens are prompt_ids[1:] +
        [bonus]. Stores K/V for positions 0..P-1 and sets ``_last_feature`` to the
        last-position output feature (which predicts the first draft token).
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
        self._staged_h2d(self.kv_positions.ptr, positions)
        self._ctx_len = P
        self._last_feature = self._last_position_feature(P - 1, P - 1, P, features[P - 1:P])

    def draft_greedy_chain_cached(self, n_draft: int):
        """Draft ``n_draft`` tokens as rows=1 decodes against the persistent K/V
        cache. Requires ``prefill_context``/``append_committed`` to have set
        ``_last_feature``. Returns (draft_ids [n_draft], None); committed features
        are re-derived from the target aux by ``append_committed``."""
        H = self.hidden_size
        feature = self._last_feature
        if feature is None:
            raise RuntimeError("draft_greedy_chain_cached requires prefill_context first")
        draft_ids = []
        for step in range(n_draft):
            self._staged_h2d(self.out_feature.ptr, _to_bf16_bits(feature[None, :]))
            _probs, ids = self._efficient_head_last(1, k=1)
            nxt = int(ids[0, 0])
            draft_ids.append(nxt)
            if step < n_draft - 1:
                pos = self._ctx_len + step
                self._staged_h2d(self.feature_seq.ptr, _to_bf16_bits(feature[None, :]))
                embed = self._numpy.w["embed_tokens.weight"][np.array([nxt], dtype=np.int64)]
                self._staged_h2d(self.embed_seq.ptr, _to_bf16_bits(embed))
                self._staged_h2d(self.positions_buf.ptr, np.array([pos], dtype=np.int32))
                self._qkv(1)
                self._store_kv(pos, 1)
                self._staged_h2d(self.kv_positions.ptr + pos * 4, np.array([pos], dtype=np.int32))
                self._attend(1, pos + 1, self.k_cache.ptr, self.v_cache.ptr, self.kv_positions.ptr)
                self._ffn(1)
                out_u16 = self._d2h(self.out_feature.ptr, (H,), np.uint16)
                feature = bf16_to_f32(out_u16)
        return np.array(draft_ids, dtype=np.int64), None

    def append_committed(self, new_aux: np.ndarray, next_tokens) -> None:
        """Append K/V for committed rows using the TARGET aux (not draft features).

        ``new_aux`` [m, 3H] is the target aux concat for the m committed positions;
        ``next_tokens`` [m] is the input token at each (token_{i+1} alignment).
        Overwrites the temporary draft K/V from the finished block, advances
        ``_ctx_len``, and sets ``_last_feature`` for the next block's first draft.
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
        self._staged_h2d(self.kv_positions.ptr + base * 4, positions)
        self._ctx_len = base + m
        self._last_feature = self._last_position_feature(m - 1, base + m - 1, self._ctx_len, features[m - 1:m])