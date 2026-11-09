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
        H, QF, KVF, I = self.hidden_size, self.q_features, self.kv_features, self.intermediate_size
        self.embed_seq = _buf(T * H * 2)
        self.feature_seq = _buf(T * H * 2)
        self.x_concat = _buf(T * 2 * H * 2)
        self.norm_embed = _buf(T * H * 2)
        self.norm_hidden = _buf(T * H * 2)
        self.q_out = _buf(T * QF * 4)
        self.k_out = _buf(T * KVF * 4)
        self.v_out = _buf(T * KVF * 2)
        self.attn_out = _buf(T * H * 2)
        self.o_proj_out = _buf(T * H * 2)
        self.residual = _buf(T * H * 2)
        self.hn = _buf(T * H * 2)
        self.mlp_gate = _buf(T * I * 2)
        self.mlp_up = _buf(T * I * 2)
        self.mlp_out = _buf(T * H * 2)
        self.summed = _buf(T * H * 2)
        self.out_feature = _buf(T * H * 2)
        self.positions_buf = _buf(T * 4)
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

    def _layer(self, T: int) -> None:
        H, QF, KVF, I = self.hidden_size, self.q_features, self.kv_features, self.intermediate_size
        L = "layers.0"
        dflash_rmsnorm_bf16(self.embed_seq.ptr, self._w(f"{L}.input_layernorm.weight"), self.norm_embed.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.feature_seq.ptr, self._w(f"{L}.hidden_norm.weight"), self.norm_hidden.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        eagle4_concat(self.norm_embed.ptr, self.norm_hidden.ptr, self.x_concat.ptr, T, H, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.x_concat.ptr, self._w(f"{L}.self_attn.q_proj.weight"), self.q_out.ptr, T, 2 * H, QF, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.x_concat.ptr, self._w(f"{L}.self_attn.k_proj.weight"), self.k_out.ptr, T, 2 * H, KVF, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.x_concat.ptr, self._w(f"{L}.self_attn.v_proj.weight"), self.v_out.ptr, T, 2 * H, KVF, library=self._lib1, runtime=self.runtime)
        eagle4_rope(self.q_out.ptr, self.positions_buf.ptr, T, self.num_heads, self.head_dim, self.rope_theta, runtime=self.runtime)
        eagle4_rope(self.k_out.ptr, self.positions_buf.ptr, T, self.num_kv_heads, self.head_dim, self.rope_theta, runtime=self.runtime)
        dflash2_sliding_attention_f32_bf16(self.q_out.ptr, self.k_out.ptr, self.v_out.ptr, self.positions_buf.ptr, self.positions_buf.ptr, self.attn_out.ptr, 1, T, T, self.num_heads, self.num_kv_heads, self.head_dim, sliding_window=self.sliding_window, is_causal=True, library=self._lib2, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.attn_out.ptr, self._w(f"{L}.self_attn.o_proj.weight"), self.o_proj_out.ptr, T, QF, H, library=self._lib1, runtime=self.runtime)
        dflash_add_bf16(self.o_proj_out.ptr, self.feature_seq.ptr, self.residual.ptr, T * H, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.residual.ptr, self._w(f"{L}.post_attention_layernorm.weight"), self.hn.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.hn.ptr, self._w(f"{L}.mlp.gate_proj.weight"), self.mlp_gate.ptr, T, H, I, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.hn.ptr, self._w(f"{L}.mlp.up_proj.weight"), self.mlp_up.ptr, T, H, I, library=self._lib1, runtime=self.runtime)
        dflash_silu_mul_bf16(self.mlp_gate.ptr, self.mlp_up.ptr, self.mlp_gate.ptr, T * I, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.mlp_gate.ptr, self._w(f"{L}.mlp.down_proj.weight"), self.mlp_out.ptr, T, I, H, library=self._lib1, runtime=self.runtime)
        dflash_add_bf16(self.mlp_out.ptr, self.residual.ptr, self.summed.ptr, T * H, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.summed.ptr, self._w("norm.weight"), self.out_feature.ptr, T, H, eps=self.eps, library=self._lib1, runtime=self.runtime)

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