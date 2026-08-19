"""GPU DFlash2 drafter forward (D2b).

Runs the 5-layer DFlash2 drafter forward on the GPU using the native kernels
(grouped dynamic conv, head-norm+rotary, sliding-window attention) and the
reused DFlash drafter primitives (dense, rmsnorm, add+rmsnorm, silu_mul,
qkv projection).  The projected context (fc + hidden_norm over the tap concat)
and each layer's rotated context K / context V are appended per accepted row
so the forward cost per cycle is bounded by the block, not the context.

This is the exact-math native path for D2b; it is cross-validated against the
D1 NumPy drafter (`DFlash2NumpyDrafter`) in the D2b RED test and the cycle
driver.
"""

from __future__ import annotations

import numpy as np

from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash2 import (
    build_dflash2,
    dflash2_grouped_conv,
    dflash2_selector,
    dflash2_sliding_attention_f32_bf16,
    dflash2_top16_rows,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter,
    dflash_add_bf16,
    dflash_add_rmsnorm_bf16,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_head_rmsnorm_rotary_f32,
    dflash_key_rmsnorm_rotary_f32,
    dflash_qkv_proj_bf16_mixed,
    dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
)
from hipengine.loading.dflash import DFlashDraftConfig

# Re-exported helpers used by tests / the cycle driver.
__all__ = ["DFlash2NativeDrafter"]


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


def _dflash2_rope_tables(max_positions: int, rope_theta: float, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-repeat Qwen3 rotary tables: (max_positions, head_dim) f32 cos/sin."""
    pos = np.arange(max_positions, dtype=np.float32)
    dims = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = np.float32(1.0) / (np.float32(rope_theta) ** (dims / np.float32(head_dim)))
    freqs = pos[:, None] * inv_freq[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


class DFlash2NativeDrafter:
    """Device-resident DFlash2 drafter forward + selector (single request)."""

    def __init__(
        self,
        config: DFlashDraftConfig,
        weights: dict[str, np.ndarray],
        *,
        runtime: HipRuntime | None = None,
        max_context_len: int = 4096,
        threads: int = 128,
    ) -> None:
        if not config.is_dflash2:
            raise ValueError(f"config is not a DFlash2 drafter: {config.architecture!r}")
        self.config = config
        self.runtime = runtime or get_hip_runtime()
        self.threads = threads
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.intermediate_size = int(config.intermediate_size)
        self.head_dim = int(config.head_dim)
        self.num_q_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.q_features = self.num_q_heads * self.head_dim
        self.kv_features = self.num_kv_heads * self.head_dim
        self.group_size = int(config.conv_group_size)
        self.conv_proj_features = 2 * 2 * (self.hidden_size // self.group_size)  # 1280
        self.block_size = int(config.block_size)
        self.selector_rank = int(config.selector_rank)
        self.selector_top_k = int(config.selector_top_k)
        self.sliding_window = int(config.sliding_windows[0]) if config.sliding_windows else 0
        self.rope_theta = float(config.rope_theta)
        self.eps = float(config.rms_norm_eps)
        self.max_context_len = int(max_context_len)
        self.vocab_size = int(config.vocab_size)
        self._lib2 = build_dflash2(load=True)
        self._lib1 = build_dflash_drafter(load=True)

        # Host staging buffer for H2D uploads.  On this ROCm/driver combo,
        # hipMemcpy H2D from a host buffer that was allocated *after* a
        # device hipMalloc can segfault / corrupt the source.  Allocating the
        # staging buffer before any device allocation and memcpy'ing through
        # it sidesteps that quirk deterministically.
        max_wbytes = max((int(np.ascontiguousarray(a).nbytes) for a in weights.values()), default=0)
        staging_bytes = int(np.ceil(max(max_wbytes, 16 * 1024 * 1024) / 4)) * 4
        self._staging_np = np.empty(staging_bytes // 4, dtype=np.float32)
        self._staging_ptr = int(self._staging_np.ctypes.data)
        self._staging_nbytes = staging_bytes

        self._bufs: list = []

        def _buf(nbytes: int):
            b = malloc(nbytes, runtime=self.runtime)
            self._bufs.append(b)
            return b

        _wdev: dict[str, int] = {}

        def _upload(name: str, arr: np.ndarray):
            b = _buf(arr.nbytes)
            self._staged_h2d(b.ptr, arr)
            _wdev[name] = b.ptr
            return b

        # Upload weights as BF16 (store raw device pointers).
        self.wdev: dict[str, int] = {
            name: _upload(name, _to_bf16_bits(arr)).ptr for name, arr in weights.items()
        }
        # Precomputed rotary tables (position * rotary_dim layout).
        self._rope_max = self.max_context_len + self.block_size + 2
        cos, sin = _dflash2_rope_tables(self._rope_max, self.rope_theta, self.head_dim)
        self.cos_table = _upload("__cos", cos).ptr
        self.sin_table = _upload("__sin", sin).ptr

        # Scratch buffers (block_size rows unless noted).
        hbytes = self.block_size * self.hidden_size * 2
        self.block_hidden_a = _buf(hbytes)  # layer input
        self.block_hidden_b = _buf(hbytes)  # layer output
        self.block_norm = _buf(hbytes)  # rmsnorm output
        self.conv_proj = _buf(self.block_size * self.conv_proj_features * 2)
        self.conv_out = _buf(hbytes)  # conv finish / mlp down output
        self.mlp_finish = _buf(hbytes)  # mlp conv finish output (no in-place alias)
        self.attn_out = _buf(hbytes)  # attention (attn scores) output
        self.o_proj_out = _buf(hbytes)  # o_proj output (separate from attn input)
        mlp_inter_bytes = self.block_size * int(config.intermediate_size) * 2
        self.mlp_gate = _buf(mlp_inter_bytes)  # gate projection (bs, intermediate) bf16
        self.mlp_up = _buf(mlp_inter_bytes)  # up projection (bs, intermediate) bf16
        self.q_out = _buf(self.block_size * self.q_features * 4)
        self.q_rot = _buf(self.block_size * self.q_features * 4)  # rotary output (no in-place pair race)
        self.k_out = _buf(self.block_size * self.kv_features * 4)
        self.v_out = _buf(self.block_size * self.kv_features * 2)
        self.k_block = _buf(self.block_size * self.kv_features * 4)
        self.v_block = _buf(self.block_size * self.kv_features * 2)
        # Contiguous context+block attention buffers (window + block rows).
        self.attn_rows = self.sliding_window + self.block_size
        self.k_cat = _buf(self.attn_rows * self.kv_features * 4)
        self.v_cat = _buf(self.attn_rows * self.kv_features * 2)
        self.qpos = _buf(self.block_size * 4)
        self.kpos = _buf(self.attn_rows * 4)
        self.draft_hidden = _buf((self.block_size - 1) * self.hidden_size * 2)
        self.logits = _buf(self.block_size * self.vocab_size * 4)
        self.topk_ids = _buf(self.block_size * self.selector_top_k * 4)
        self.topk_vals = _buf(self.block_size * self.selector_top_k * 4)
        self.selector_h = _buf(self.block_size * self.selector_rank * 4)
        self.selector_path = _buf(self.block_size * 4)
        self.selector_scores = _buf(self.block_size * self.selector_top_k * 4)
        self.anchor_buf = _buf(4)

        # Projected-context + per-layer context K/V caches (append-only).
        self.projected_ctx = _buf(self.max_context_len * self.hidden_size * 2)
        self.ctx_k = [_buf(self.max_context_len * self.kv_features * 4) for _ in range(self.num_layers)]
        self.ctx_v = [_buf(self.max_context_len * self.kv_features * 2) for _ in range(self.num_layers)]
        self.ctx_kpos = _buf(self.max_context_len * 4)
        self.ctx_len = 0

    def close(self) -> None:
        for buf in self._bufs:
            free(buf, runtime=self.runtime)
        self._bufs.clear()

    def __enter__(self) -> "DFlash2NativeDrafter":
        return self

    def __exit__(self, *exc) -> None:  # noqa: D105
        self.close()

    # -- helpers -----------------------------------------------------------

    def _w(self, name: str) -> int:
        try:
            return self.wdev[name]
        except KeyError as exc:
            raise KeyError(f"missing DFlash2 drafter device weight {name!r}") from exc

    def upload_weight(self, name: str, arr: np.ndarray) -> int:
        """Upload a BF16 weight (e.g. the target ``output_head``) through the
        persistent host staging buffer and return its device pointer.

        IMPORTANT: on this ROCm/driver combo, a host->device memcpy of a
        freshly ``hipMalloc``'d buffer corrupts the first bytes.  Always route
        H2D uploads through ``_staged_h2d``/this method, never raw
        ``copy_host_to_device`` after a fresh malloc.  Re-uploading an existing
        name frees and re-allocates the device buffer."""
        old = self.wdev.pop(name, None)
        if old is not None:
            for buf in list(self._bufs):
                if buf.ptr == old:
                    free(buf, runtime=self.runtime)
                    self._bufs.remove(buf)
                    break
        arr_bf16 = _to_bf16_bits(np.ascontiguousarray(arr, dtype=np.float32))
        b = malloc(arr_bf16.nbytes, runtime=self.runtime)
        self._staged_h2d(b.ptr, arr_bf16)
        self._bufs.append(b)
        self.wdev[name] = b.ptr
        return b.ptr

    def _h2d(self, buf, arr: np.ndarray) -> None:
        self._staged_h2d(buf.ptr, arr)

    def _staged_h2d(self, dst_ptr: int, arr: np.ndarray) -> None:
        """Upload ``arr`` to ``dst_ptr`` through the persistent host staging
        buffer (avoids the post-hipMalloc H2D segfault quirk)."""
        arr_c = np.ascontiguousarray(arr)
        nbytes = arr_c.nbytes
        if nbytes > self._staging_nbytes:
            raise ValueError(f"H2D upload of {nbytes} bytes exceeds host staging buffer")
        staging_view = self._staging_np.view(np.uint8)[:nbytes]
        staging_view[:] = arr_c.view(np.uint8).ravel()
        self.runtime.memcpy(dst_ptr, self._staging_ptr, nbytes, HipMemcpyKind.HOST_TO_DEVICE)

    def _d2h(self, buf, shape: tuple[int, ...], dtype) -> np.ndarray:
        arr = np.empty(shape, dtype=dtype)
        if isinstance(buf, int):
            self.runtime.memcpy(host_array_ptr(arr), buf, arr.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        else:
            copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes, runtime=self.runtime)
        return arr

    def _d2d(self, dst, src, nbytes: int) -> None:
        self.runtime.memcpy_async(dst, src, nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, 0)

    def _dense_bf16(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, out_features: int) -> None:
        dflash_dense_bf16_to_bf16(
            x_ptr, w_ptr, out_ptr, rows, self.hidden_size, out_features,
            library=self._lib1, runtime=self.runtime,
        )

    def _conv(self, hidden_ptr: int, proj_ptr: int, base_ptr: int, out_ptr: int, rows: int, side_offset: int) -> None:
        dflash2_grouped_conv(
            hidden_ptr, proj_ptr, base_ptr, out_ptr,
            rows, self.hidden_size, self.group_size,
            dyn_offset=side_offset, dyn_stride=self.conv_proj_features,
            library=self._lib2, runtime=self.runtime,
        )

    # -- projected-context cache -------------------------------------------

    def _append_ctx_kv(self, proj_src_ptr: int, positions_i32: np.ndarray, rows: int) -> None:
        """Project+rotate+store per-layer context K (f32) and V (bf16) for the
        given projected rows, appending at ``self.ctx_len``."""
        self._h2d(self.ctx_kpos, positions_i32)
        kpos_ptr = self.ctx_kpos.ptr
        for layer in range(self.num_layers):
            k_proj = self._w(f"layers.{layer}.self_attn.k_proj.weight")
            v_proj = self._w(f"layers.{layer}.self_attn.v_proj.weight")
            k_norm = self._w(f"layers.{layer}.self_attn.k_norm.weight")
            k_proj_out = malloc(rows * self.kv_features * 4, runtime=self.runtime)
            v_proj_out = malloc(rows * self.kv_features * 2, runtime=self.runtime)
            k_rot = malloc(rows * self.kv_features * 4, runtime=self.runtime)
            try:
                dflash_dense_bf16_to_f32(
                    proj_src_ptr, k_proj, k_proj_out.ptr, rows, self.hidden_size, self.kv_features,
                    library=self._lib1, runtime=self.runtime,
                )
                dflash_dense_bf16_to_bf16(
                    proj_src_ptr, v_proj, v_proj_out.ptr, rows, self.hidden_size, self.kv_features,
                    library=self._lib1, runtime=self.runtime,
                )
                dflash_key_rmsnorm_rotary_f32(
                    k_proj_out.ptr, k_norm, self.cos_table, self.sin_table,
                    kpos_ptr, k_rot.ptr,
                    rows, self.num_kv_heads, self.head_dim, self.head_dim,
                    self._rope_max, eps=self.eps, library=self._lib1, runtime=self.runtime,
                )
                k_dst = self.ctx_k[layer].ptr + self.ctx_len * self.kv_features * 4
                v_dst = self.ctx_v[layer].ptr + self.ctx_len * self.kv_features * 2
                self._d2d(k_dst, k_rot.ptr, rows * self.kv_features * 4)
                self._d2d(v_dst, v_proj_out.ptr, rows * self.kv_features * 2)
            finally:
                for buf in (k_proj_out, v_proj_out, k_rot):
                    free(buf, runtime=self.runtime)

    def reset_projected_context(self, projected_bf16: np.ndarray) -> None:
        """Seed the projected-context cache with the full prompt (bf16 rows) and
        precompute the per-layer context K/V for those rows."""
        rows = int(projected_bf16.shape[0])
        if rows > self.max_context_len:
            raise ValueError("prompt projected rows exceed native drafter context capacity")
        self._h2d(self.projected_ctx, projected_bf16)
        self.ctx_len = 0
        positions_i32 = np.arange(rows, dtype=np.int32)
        self._append_ctx_kv(self.projected_ctx.ptr, positions_i32, rows)
        self.ctx_len = rows

    def append_projected_rows(self, projected_bf16: np.ndarray, positions: np.ndarray) -> None:
        """Append accepted rows' projected hidden and per-layer rotated K/V."""
        rows = int(projected_bf16.shape[0])
        if rows == 0:
            return
        if self.ctx_len + rows > self.max_context_len:
            raise ValueError("projected-context append exceeds native drafter capacity")
        proj_dst_ptr = self.projected_ctx.ptr + self.ctx_len * self.hidden_size * 2
        self._staged_h2d(proj_dst_ptr, projected_bf16)
        positions_i32 = positions.astype(np.int32)
        self._append_ctx_kv(proj_dst_ptr, positions_i32, rows)
        self.ctx_len += rows

    # -- attention ---------------------------------------------------------

    def _run_attention(self, hidden_ptr: int, positions: np.ndarray, layer: int) -> None:
        """q/k/v projections + head norms + rotary + sliding attention + o_proj
        into ``self.attn_out``."""
        bs = self.block_size
        ctx = self.ctx_len
        q_proj = self._w(f"layers.{layer}.self_attn.q_proj.weight")
        k_proj = self._w(f"layers.{layer}.self_attn.k_proj.weight")
        v_proj = self._w(f"layers.{layer}.self_attn.v_proj.weight")
        o_proj = self._w(f"layers.{layer}.self_attn.o_proj.weight")
        q_norm = self._w(f"layers.{layer}.self_attn.q_norm.weight")
        k_norm = self._w(f"layers.{layer}.self_attn.k_norm.weight")
        dflash_qkv_proj_bf16_mixed(
            hidden_ptr, q_proj, k_proj, v_proj,
            self.q_out.ptr, self.k_out.ptr, self.v_out.ptr,
            bs, self.hidden_size, self.q_features, self.kv_features,
            library=self._lib1, runtime=self.runtime,
        )
        qpos = positions[ctx : ctx + bs].astype(np.int32)
        self._h2d(self.qpos, qpos)
        # Rotate q (block rows) and k (block rows) in one call.  query_out must
        # NOT alias query (the rotary reads paired dims in-place -> race).
        dflash_head_rmsnorm_rotary_f32(
            self.q_out.ptr, self.k_out.ptr, q_norm, k_norm,
            self.cos_table, self.sin_table, self.qpos.ptr, self.qpos.ptr,
            self.q_rot.ptr, self.k_block.ptr,
            1, bs, bs, self.num_q_heads, self.num_kv_heads,
            self.head_dim, self.head_dim, self._rope_max,
            eps=self.eps, library=self._lib1, runtime=self.runtime,
        )
        # Contiguous [trailing ctx K, block K] and [trailing ctx V, block V].
        # Block V is the raw (unnormalized) v_out from the qkv projection.
        window = self.sliding_window
        ctx_start = max(0, ctx - window)
        ctx_rows = ctx - ctx_start
        k_bytes = self.kv_features * 4
        v_bytes = self.kv_features * 2
        kctx_src = self.ctx_k[layer].ptr + ctx_start * k_bytes
        vctx_src = self.ctx_v[layer].ptr + ctx_start * v_bytes
        self._d2d(self.k_cat.ptr, kctx_src, ctx_rows * k_bytes)
        self._d2d(self.k_cat.ptr + ctx_rows * k_bytes, self.k_block.ptr, bs * k_bytes)
        self._d2d(self.v_cat.ptr, vctx_src, ctx_rows * v_bytes)
        self._d2d(self.v_cat.ptr + ctx_rows * v_bytes, self.v_out.ptr, bs * v_bytes)
        kv_len = ctx_rows + bs
        kpos = np.concatenate(
            [np.arange(ctx_start, ctx, dtype=np.int32), positions[ctx : ctx + bs].astype(np.int32)]
        )
        self._h2d(self.kpos, kpos)
        dflash2_sliding_attention_f32_bf16(
            self.q_rot.ptr, self.k_cat.ptr, self.v_cat.ptr, self.qpos.ptr, self.kpos.ptr, self.attn_out.ptr,
            1, bs, kv_len, self.num_q_heads, self.num_kv_heads, self.head_dim,
            sliding_window=window, is_causal=False,
            library=self._lib2, runtime=self.runtime,
        )
        # o_proj maps attention output (q_features) -> hidden; in-place would
        # alias the WMMA input, so write to the dedicated o_proj_out buffer.
        dflash_dense_bf16_to_bf16(
            self.attn_out.ptr, o_proj, self.o_proj_out.ptr,
            bs, self.q_features, self.hidden_size,
            library=self._lib1, runtime=self.runtime,
        )

    # -- forward -----------------------------------------------------------

    def _forward_layer(self, layer: int, positions: np.ndarray, *, debug_callback=None) -> None:
        """Run one drafter layer; ``block_hidden_a`` holds the layer output on
        return (and becomes the next layer's input)."""
        bs = self.block_size
        half_offset = 2 * (self.hidden_size // self.group_size)  # 640
        in_layernorm = self._w(f"layers.{layer}.input_layernorm.weight")
        post_attn = self._w(f"layers.{layer}.post_attention_layernorm.weight")
        conv_base = self._w(f"layers.{layer}.attention_conv.base_kernel")
        conv_kp = self._w(f"layers.{layer}.attention_conv.kernel_projection.weight")
        mlp_base = self._w(f"layers.{layer}.mlp_conv.base_kernel")
        mlp_kp = self._w(f"layers.{layer}.mlp_conv.kernel_projection.weight")
        gate_proj = self._w(f"layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._w(f"layers.{layer}.mlp.up_proj.weight")
        down_proj = self._w(f"layers.{layer}.mlp.down_proj.weight")

        # input_layernorm
        dflash_rmsnorm_bf16(self.block_hidden_a.ptr, in_layernorm, self.block_norm.ptr, bs, self.hidden_size,
                            eps=self.eps, library=self._lib1, runtime=self.runtime)
        # attention conv prepare (input side)
        self._dense_bf16(self.block_norm.ptr, conv_kp, self.conv_proj.ptr, bs, self.conv_proj_features)
        self._conv(self.block_norm.ptr, self.conv_proj.ptr, conv_base, self.conv_out.ptr, bs, 0)
        # attention
        self._run_attention(self.conv_out.ptr, positions, layer)
        # attention conv finish (output side) reads o_proj_out.
        self._conv(self.o_proj_out.ptr, self.conv_proj.ptr, conv_base + 2 * self.hidden_size * 2, self.conv_out.ptr, bs, half_offset)
        # residual: layer_out = layer_input + attn_finish (B); mlp_norm = rmsnorm(layer_out)
        dflash_add_bf16(self.conv_out.ptr, self.block_hidden_a.ptr, self.block_hidden_b.ptr,
                        bs * self.hidden_size, library=self._lib1, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.block_hidden_b.ptr, post_attn, self.block_norm.ptr, bs, self.hidden_size,
                            eps=self.eps, library=self._lib1, runtime=self.runtime)
        # MLP conv prepare (input side)
        self._dense_bf16(self.block_norm.ptr, mlp_kp, self.conv_proj.ptr, bs, self.conv_proj_features)
        self._conv(self.block_norm.ptr, self.conv_proj.ptr, mlp_base, self.conv_out.ptr, bs, 0)
        # MLP: gate/up -> silu_mul -> down
        dflash_dense_bf16_to_bf16(self.conv_out.ptr, gate_proj, self.mlp_gate.ptr,
                                  bs, self.hidden_size, self.intermediate_size, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.conv_out.ptr, up_proj, self.mlp_up.ptr,
                                  bs, self.hidden_size, self.intermediate_size, library=self._lib1, runtime=self.runtime)
        dflash_silu_mul_bf16(self.mlp_gate.ptr, self.mlp_up.ptr, self.mlp_gate.ptr,
                             bs * self.intermediate_size, library=self._lib1, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.mlp_gate.ptr, down_proj, self.conv_out.ptr,
                                  bs, self.intermediate_size, self.hidden_size, library=self._lib1, runtime=self.runtime)
        # MLP conv finish (output side) — write to mlp_finish (not in-place).
        self._conv(self.conv_out.ptr, self.conv_proj.ptr, mlp_base + 2 * self.hidden_size * 2, self.mlp_finish.ptr, bs, half_offset)
        # residual: layer_out (B) + mlp_finish -> next input (A)
        dflash_add_bf16(self.mlp_finish.ptr, self.block_hidden_b.ptr, self.block_hidden_a.ptr,
                        bs * self.hidden_size, library=self._lib1, runtime=self.runtime)
        if debug_callback is not None:
            debug_callback(layer, self.block_hidden_a.ptr)

    def forward(
        self,
        noise_bf16: np.ndarray,
        positions: np.ndarray,
        *,
        debug_callback=None,
    ) -> int:
        """Run the 5-layer drafter forward over the block; returns the device
        pointer to the final-normed draft hidden (block_size, hidden)."""
        bs = self.block_size
        self._h2d(self.block_hidden_a, noise_bf16)
        for layer in range(self.num_layers):
            self._forward_layer(layer, positions, debug_callback=debug_callback)
        # final norm: skip the anchor row (block row 0) -> block_size-1 draft rows.
        norm = self._w("norm.weight")
        dflash_rmsnorm_bf16(
            self.block_hidden_a.ptr + self.hidden_size * 2, norm, self.draft_hidden.ptr,
            bs - 1, self.hidden_size,
            eps=self.eps, library=self._lib1, runtime=self.runtime,
        )
        return self.draft_hidden.ptr

    # -- selector ----------------------------------------------------------

    def select(
        self,
        draft_hidden_ptr: int,
        output_head_bf16_ptr: int | None,
        logits_f32_ptr: int | None,
        anchor_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the top-16 selector over the draft hidden block.

        ``output_head_bf16_ptr`` (rows x hidden BF16) is used to compute draft
        logits when ``logits_f32_ptr`` is None.  Returns ``(path, scores)``.
        """
        rows = self.block_size - 1
        if logits_f32_ptr is None:
            if output_head_bf16_ptr is None:
                raise ValueError("select requires an output head or precomputed logits")
            dflash_dense_bf16_to_f32(
                draft_hidden_ptr, output_head_bf16_ptr, self.logits.ptr,
                rows, self.hidden_size, self.vocab_size,
                library=self._lib1, runtime=self.runtime,
            )
            logits_ptr = self.logits.ptr
        else:
            logits_ptr = logits_f32_ptr
        dflash2_top16_rows(
            logits_ptr, self.topk_ids.ptr, self.topk_vals.ptr,
            rows, self.vocab_size, self.selector_top_k,
            library=self._lib2, runtime=self.runtime,
        )
        anchor_ids_arr = np.asarray(anchor_ids)
        if anchor_ids_arr.ndim == 0:
            anchor = np.asarray([int(anchor_ids_arr)], dtype=np.int32)
        else:
            anchor = np.asarray(anchor_ids_arr.ravel()[0], dtype=np.int32).reshape(1)
        self._h2d(self.anchor_buf, anchor)
        dflash2_selector(
            draft_hidden_ptr, self._w("candidate_selector.hidden_projection.weight"),
            self.topk_ids.ptr, self.topk_vals.ptr, self.anchor_buf.ptr,
            self._w("candidate_selector.predecessor_codebook"),
            self._w("candidate_selector.successor_codebook"),
            self.selector_h.ptr, self.selector_path.ptr, self.selector_scores.ptr,
            rows, self.hidden_size, self.selector_rank, self.selector_top_k,
            self.vocab_size,
            library=self._lib2, runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        path = self._d2h(self.selector_path, (rows,), np.int32)
        scores = self._d2h(self.selector_scores, (rows, self.selector_top_k), np.float32)
        return path, scores
