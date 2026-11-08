"""Native GPU DSpine drafter (clean-room), reusing DFlash2 backbone helpers.

DSpine = block-causal DFlash-style backbone (no grouped conv, no path selector)
+ per-layer adjacent causal injection (code bridge) + transition-cache refine.
Backbone projections/attention/norms reuse DFlash2 native kernel primitives; the
bridge uses kernels/hip_gfx1100/speculative/dspine.py. Greedy transition-cache
refinement runs on the CPU oracle for now (GPU refinement is a speed follow-up).
Clean-room: arXiv:2609.36173, no vendored reference code.
"""
from __future__ import annotations

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kernels.cpu_reference.dspine import DSpineNumpyDrafter
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter, dflash_add_bf16, dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32, dflash_rmsnorm_bf16, dflash_silu_mul_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash2 import build_dflash2
from hipengine.kernels.hip_gfx1100.speculative.dspine import (
    build_dspine, dspine_codes, dspine_gate_message, dspine_residual, dspine_rms_normalize,
)
from hipengine.speculative.dflash2_native import (
    DFlash2NativeDrafter, _dflash2_rope_tables, _to_bf16_bits,
)


class DSpineNativeDrafter(DFlash2NativeDrafter):
    """DSpine drafter: block-causal backbone + adjacent-injection bridge."""

    _attention_is_causal = True

    def __init__(self, config: dict, weights: dict[str, np.ndarray], *,
                 runtime: HipRuntime | None = None, max_context_len: int = 4096):
        self.runtime = runtime or get_hip_runtime()
        df = config["dflash_config"]
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_hidden_layers"])
        self.intermediate_size = int(config["intermediate_size"])
        self.head_dim = int(config["head_dim"])
        self.num_q_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.q_features = self.num_q_heads * self.head_dim
        self.kv_features = self.num_kv_heads * self.head_dim
        self.block_size = int(config["block_size"])
        self.sliding_window = int(config.get("sliding_window") or 0)
        self.rope_theta = float(config.get("rope_theta", 1e7))
        self.eps = float(config["rms_norm_eps"])
        self.max_context_len = int(max_context_len)
        self.vocab_size = int(config["vocab_size"])
        self.mask_token_id = int(df["mask_token_id"])
        self.code_size = int(df["selected_prefix_code_bridge"]["code_size"])
        self.rank = int(df["selected_prefix_code_bridge"]["rank"])
        self.head_topk = int(df["selected_prefix_last_write_refine"]["head_topk"])
        self.threads = 128
        self._lib1 = build_dflash_drafter(load=True)
        self._lib2 = build_dflash2(load=True)
        self._lib3 = build_dspine(load=True)
        max_wbytes = max((int(np.ascontiguousarray(a).nbytes) for a in weights.values()), default=0)
        staging_bytes = int(np.ceil(max(max_wbytes, 16 * 1024 * 1024) / 4)) * 4
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
        # Precompute the single shared packed bridge projection [G_q; G_p; A].
        packed = np.concatenate([
            np.asarray(weights["selected_prefix_bridge.gate.weight"], dtype=np.float32),
            np.asarray(weights["selected_prefix_bridge.predecessor_gate.weight"], dtype=np.float32),
            np.asarray(weights["selected_prefix_bridge.prev.weight"], dtype=np.float32),
        ], axis=0)
        a = _to_bf16_bits(packed)
        b = _buf(a.nbytes)
        self._staged_h2d(b.ptr, a)
        self.wdev["__packed"] = b.ptr
        self._rope_max = self.max_context_len + self.block_size + 2
        cos, sin = _dflash2_rope_tables(self._rope_max, self.rope_theta, self.head_dim)
        cb = _buf(cos.nbytes); self._staged_h2d(cb.ptr, cos); self.cos_table = cb.ptr
        sb = _buf(sin.nbytes); self._staged_h2d(sb.ptr, sin); self.sin_table = sb.ptr
        bs = self.block_size
        hb = bs * self.hidden_size * 2
        self.block_hidden_a = _buf(hb)
        self.block_hidden_b = _buf(hb)
        self.block_norm = _buf(hb)
        self.attn_out = _buf(hb)
        self.o_proj_out = _buf(hb)
        mb = bs * self.intermediate_size * 2
        self.mlp_gate = _buf(mb)
        self.mlp_up = _buf(mb)
        self.conv_out = _buf(hb)
        self.q_out = _buf(bs * self.q_features * 4)
        self.q_rot = _buf(bs * self.q_features * 4)
        self.k_out = _buf(bs * self.kv_features * 4)
        self.v_out = _buf(bs * self.kv_features * 2)
        self.k_block = _buf(bs * self.kv_features * 4)
        self.attn_rows = self.sliding_window + bs
        self.k_cat = _buf(self.attn_rows * self.kv_features * 4)
        self.v_cat = _buf(self.attn_rows * self.kv_features * 2)
        self.qpos = _buf(bs * 4)
        self.kpos = _buf(self.attn_rows * 4)
        self.br_normalized = _buf(hb)
        self.br_rms = _buf(bs * 4)
        self.br_projected = _buf(bs * self.code_size * 4)
        self.br_codes = _buf(bs * self.code_size * 2)
        self.br_packed = _buf(bs * 3 * self.rank * 4)
        self.br_message = _buf(bs * self.rank * 2)
        self.br_delta = _buf(bs * self.hidden_size * 4)
        self.br_out = _buf(hb)
        self.anchor_buf = _buf(4)
        self.projected_ctx = _buf(self.max_context_len * self.hidden_size * 2)
        self.ctx_k = [_buf(self.max_context_len * self.kv_features * 4) for _ in range(self.num_layers)]
        self.ctx_v = [_buf(self.max_context_len * self.kv_features * 2) for _ in range(self.num_layers)]
        self.ctx_kpos = _buf(self.max_context_len * 4)
        self.ctx_len = 0
        self._numpy = DSpineNumpyDrafter(config, weights)
        self._last_stash: dict = {}
        self._last_final_hidden: np.ndarray | None = None

    def _read_f32(self, buf, count):
        out = np.empty(count, dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), buf, runtime=self.runtime)
        return out

    def _bridge(self, layer: int, anchor_id: int) -> None:
        bs, H, C, R = self.block_size, self.hidden_size, self.code_size, self.rank
        from hipengine.kernels.cpu_reference.dspine import bf16_to_f32
        # Bridge weights are global (selected_prefix_bridge.*) with per-layer
        # read/write slices; gate/codebook/packed are shared across layers.
        if layer + 1 < self.num_layers:
            read_ptr = self._w("selected_prefix_bridge.early_read_weights") + layer * C * H * 2
        else:
            read_ptr = self._w("selected_prefix_bridge.read_weight")
        write_ptr = self._w("selected_prefix_bridge.feature_write_weights") + layer * H * R * 2
        dspine_rms_normalize(self.block_hidden_a.ptr, self.br_normalized.ptr, self.br_rms.ptr, bs, H, self.eps, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.br_normalized.ptr, read_ptr, self.br_projected.ptr, bs, H, C, library=self._lib1, runtime=self.runtime)
        dspine_codes(self.br_projected.ptr, self._w("selected_prefix_bridge.codebook"), self.br_codes.ptr, anchor_id, bs, C, self.eps, float(np.sqrt(C)), runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.br_codes.ptr, self._w("__packed"), self.br_packed.ptr, bs, C, 3 * R, library=self._lib1, runtime=self.runtime)
        dspine_gate_message(self.br_packed.ptr, self._w("selected_prefix_bridge.gate.bias"), self.br_message.ptr, bs, R, runtime=self.runtime)
        dflash_dense_bf16_to_f32(self.br_message.ptr, write_ptr, self.br_delta.ptr, bs, R, H, library=self._lib1, runtime=self.runtime)
        dspine_residual(self.block_hidden_a.ptr, self.br_rms.ptr, self.br_delta.ptr, self.br_out.ptr, bs, H, runtime=self.runtime)
        if layer == self.num_layers - 1:
            self.runtime.device_synchronize()
            raw_u16 = np.empty(bs * H, dtype=np.uint16)
            copy_device_to_host(host_array_ptr(raw_u16), self.block_hidden_a, runtime=self.runtime)
            rms = self._read_f32(self.br_rms, bs)
            packed_full = self._read_f32(self.br_packed, bs * 3 * R).reshape(bs, 3 * R)
            self._last_stash = {"raw": bf16_to_f32(raw_u16).reshape(bs, H), "rms": rms, "packed": packed_full}
        nn = self._w(f"layers.{layer + 1}.input_layernorm.weight") if layer + 1 < self.num_layers else self._w("norm.weight")
        dflash_rmsnorm_bf16(self.br_out.ptr, nn, self.block_norm.ptr, bs, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        self._d2d(self.block_hidden_a.ptr, self.br_out.ptr, bs * H * 2)

    def _forward_layer(self, layer: int, positions: np.ndarray, *, anchor_id: int, debug_callback=None) -> None:
        bs, H, I = self.block_size, self.hidden_size, self.intermediate_size
        self._run_attention(self.block_norm.ptr, positions, layer)
        dflash_add_bf16(self.o_proj_out.ptr, self.block_hidden_a.ptr, self.block_hidden_b.ptr, bs * H, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.block_hidden_b.ptr, self._w(f"layers.{layer}.post_attention_layernorm.weight"), self.block_norm.ptr, bs, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.block_norm.ptr, self._w(f"layers.{layer}.mlp.gate_proj.weight"), self.mlp_gate.ptr, bs, H, I, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.block_norm.ptr, self._w(f"layers.{layer}.mlp.up_proj.weight"), self.mlp_up.ptr, bs, H, I, library=self._lib1, runtime=self.runtime)
        dflash_silu_mul_bf16(self.mlp_gate.ptr, self.mlp_up.ptr, self.mlp_gate.ptr, bs * I, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.mlp_gate.ptr, self._w(f"layers.{layer}.mlp.down_proj.weight"), self.conv_out.ptr, bs, I, H, library=self._lib1, runtime=self.runtime)
        dflash_add_bf16(self.conv_out.ptr, self.block_hidden_b.ptr, self.block_hidden_a.ptr, bs * H, library=self._lib1, runtime=self.runtime)
        self._bridge(layer, anchor_id)
        if debug_callback is not None:
            debug_callback(layer, self.block_hidden_a.ptr)

    def forward(self, noise_bf16: np.ndarray, positions: np.ndarray, *, anchor_id: int, debug_callback=None) -> None:
        bs, H = self.block_size, self.hidden_size
        from hipengine.kernels.cpu_reference.dspine import bf16_to_f32
        self._staged_h2d(self.block_hidden_a.ptr, _to_bf16_bits(noise_bf16.astype(np.float32)))
        dflash_rmsnorm_bf16(self.block_hidden_a.ptr, self._w("layers.0.input_layernorm.weight"), self.block_norm.ptr, bs, H, eps=self.eps, library=self._lib1, runtime=self.runtime)
        self._last_stash = {}
        for layer in range(self.num_layers):
            self._forward_layer(layer, positions, anchor_id=anchor_id, debug_callback=debug_callback)
        self.runtime.device_synchronize()
        u16 = np.empty(bs * H, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(u16), self.block_norm, runtime=self.runtime)
        self._last_final_hidden = bf16_to_f32(u16).reshape(bs, H)

    def refine(self, output_head_weight: np.ndarray, anchor: int) -> np.ndarray:
        drafts, _cand = self._numpy.refine_greedy(self._last_final_hidden, self._last_stash, output_head_weight, anchor)
        return drafts
