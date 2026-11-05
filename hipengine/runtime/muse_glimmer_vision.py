"""Torch-free Muse Glimmer image encoder using FP32 gfx11 primitives + rocBLAS.

Mirrors ``hipengine.kernels.cpu_reference.muse_glimmer_vision`` stage for stage
(the f32-dequantized llama.cpp oracle pins both). Weights are dequantized to
F32 once at load; every contraction is an exact-arithmetic sgemm, so the tower
computes the architecture's f32 semantics rather than llama.cpp's RDNA3 MMQ /
flash-attention shortcuts (campaign doc §3.2 records the precision finding).
"""
from __future__ import annotations

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.cpu_reference.muse_glimmer_vision import (
    MuseGlimmerVisionConfig,
    bilinear_resize_pos,
    normalize_pixels,
    patch_columns,
    window_mask,
    window_permutations,
)
from hipengine.kernels.registry import resolve
from hipengine.loading.gguf import GGUFReader


class MuseGlimmerVisionRunner:
    """Resident mmproj weights; request-local scratch bounded by the patch grid."""

    def __init__(self, reader: GGUFReader, config: MuseGlimmerVisionConfig, *, backend: str):
        from hipengine.kernels.hip_gfx1100.vision.muse_glimmer_vision import (
            register_muse_glimmer_vision_kernels,
        )

        register_muse_glimmer_vision_kernels()
        self.reader, self.config, self.backend = reader, config, backend
        self._weights: dict[str, DeviceBuffer] = {}
        self._owned: list[DeviceBuffer] = []
        self.closed = False
        self.blas = None
        self.ops = {
            name: resolve(backend=backend, layer='vision_' + name, quant='fp32',
                          variant='muse_glimmer')
            for name in ('layernorm', 'elementwise', 'bias_add', 'rope2d',
                         'transpose', 'softmax', 'gather', 'shuffle')
        }
        try:
            for tensor in reader.info.tensors:
                array = np.ascontiguousarray(reader.dequantize_tensor(tensor.name), dtype=np.float32)
                if tensor.name == 'v.patch_embd.weight':
                    array = array.reshape(array.shape[0], -1)
                self._weights[tensor.name] = self._upload(array)
            self.blas = Rocblas.load()
        except BaseException:
            self.close()
            raise

    def _upload(self, array: np.ndarray) -> DeviceBuffer:
        value = np.ascontiguousarray(array)
        buffer = malloc(value.nbytes)
        self._owned.append(buffer)
        copy_host_array_to_device(buffer, value, value.nbytes)
        return buffer

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.blas is not None:
            self.blas.close()
        for buffer in reversed(self._owned):
            free(buffer)
        self._owned.clear()
        self._weights.clear()

    def encode(self, pixels: np.ndarray, *, request=None) -> np.ndarray:
        """Encode one RGB uint8 image; returns projected rows [n_out, projection_dim].

        Preprocessing replicates the mtmd muse-glimmer handler for grid-aligned
        inputs; non-aligned inputs are refused until the Lanczos resize path
        lands (named capability, not a silent downgrade).
        """

        from hipengine.generation.deadline import raise_if_generation_deadline_expired
        from hipengine.kernels.cpu_reference.muse_glimmer_vision import grid_size

        if self.closed:
            raise RuntimeError('Muse Glimmer vision runner is closed')
        cfg = self.config
        pixels = np.asarray(pixels)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('Muse Glimmer vision expects an HWC RGB image')
        h, w = pixels.shape[:2]
        patch_hw = cfg.patch_size * cfg.spatial_merge_size
        tw, th = grid_size(w, h, patch_hw, cfg.max_image_tokens)
        if (tw, th) != (w, h):
            raise NotImplementedError(
                f'Muse Glimmer vision bring-up implements grid-aligned images; '
                f'{w}x{h} would resize to {tw}x{th} (Lanczos path pending)'
            )
        if request is not None:
            raise_if_generation_deadline_expired(request)
        return self.forward(normalize_pixels(pixels, cfg), (w // cfg.patch_size, h // cfg.patch_size))

    def forward(self, pixels_chw: np.ndarray, grid: tuple[int, int], *, request=None,
                _debug_inject: dict | None = None) -> np.ndarray:
        from hipengine.generation.deadline import raise_if_generation_deadline_expired

        cfg = self.config
        gw, gh = map(int, grid)
        n = gw * gh
        f = cfg.spatial_merge_size
        m = (gw // f) * (gh // f)
        h, inter, heads, dim = cfg.n_embd, cfg.n_ff, cfg.n_heads, cfg.head_dim
        p = np.ascontiguousarray(pixels_chw, dtype=np.float32)
        if p.shape != (3, gh * cfg.patch_size, gw * cfg.patch_size) or not np.isfinite(p).all():
            raise ValueError('invalid Muse Glimmer vision patch geometry')

        # Host-side input preparation (patch im2col, position resize, permutations).
        cols = patch_columns(p, cfg.patch_size)
        pos_rows = bilinear_resize_pos(
            np.asarray(self.reader.dequantize_tensor('v.position_embd.weight'), dtype=np.float32),
            cfg.pos_grid, gw, gh,
        )
        sp_perm, inv_perm, pos_w, pos_h, slens = window_permutations(gw, gh, cfg.pos_grid)
        mask = window_mask(n, slens)

        buffers: list[DeviceBuffer] = []

        def alloc(shape, dtype=np.float32, value=None) -> int:
            arr = None if value is None else np.ascontiguousarray(value, dtype=dtype)
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            buffer = malloc(nbytes)
            buffers.append(buffer)
            if arr is not None:
                copy_host_array_to_device(buffer, arr, arr.nbytes)
            return buffer.ptr

        def weight(name: str) -> int:
            return self._weights[name].ptr

        def norm(x, prefix, y, rows=n, width=h):
            self.ops['layernorm'](x, weight(prefix + '.weight'), weight(prefix + '.bias'),
                                  y, rows, width, cfg.norm_eps)

        def linear(x, name, y, rows=n, width=h, out=h):
            self.blas.sgemm_rowmajor_nt(x, weight(name), y, rows=rows,
                                        in_features=width, out_features=out)

        try:
            patch = alloc(cols.shape, value=cols)
            learned = alloc(pos_rows.shape, value=pos_rows)
            sp_buf = alloc((n,), dtype=np.int32, value=sp_perm.astype(np.int32))
            inv_buf = alloc((n,), dtype=np.int32, value=inv_perm.astype(np.int32))
            pw_buf = alloc((n,), dtype=np.int32, value=pos_w.astype(np.int32))
            ph_buf = alloc((n,), dtype=np.int32, value=pos_h.astype(np.int32))
            mask_buf = alloc(mask.shape, value=mask)
            x, xn, permuted = [alloc((n, h)) for _ in range(3)]
            q, k, v = [alloc((n, h)) for _ in range(3)]
            q_rot, k_rot = [alloc((n, h)) for _ in range(2)]
            qh, kh, vh = [alloc((n, h)) for _ in range(3)]
            scores = alloc((heads, n, n))
            context, attended, projected = [alloc((n, h)) for _ in range(3)]
            ff = alloc((n, inter))
            shuffled = alloc((m, h * f * f))
            mm_hidden, mm_hidden2 = [alloc((m, 4096)) for _ in range(2)]
            output = alloc((m, cfg.projection_dim))

            linear(patch, 'v.patch_embd.weight', x, width=cols.shape[1])
            self.ops['elementwise'](x, learned, x, n * h, 0)
            self.ops['gather'](x, sp_buf, permuted, n, h)
            x, permuted = permuted, x
            norm(x, 'v.pre_ln', xn)
            x, xn = xn, x
            if _debug_inject and 'pre_ln' in _debug_inject:
                copy_host_array_to_device(DeviceBuffer(x, n * h * 4),
                                          np.ascontiguousarray(_debug_inject['pre_ln'], dtype=np.float32))
            scale = float(dim) ** -0.5
            for il in range(cfg.n_layers):
                if request is not None:
                    raise_if_generation_deadline_expired(request)
                prefix = f'v.blk.{il}'
                norm(x, prefix + '.ln1', xn)
                for label, dest in (('q', q), ('k', k), ('v', v)):
                    linear(xn, prefix + f'.attn_{label}.weight', dest)
                    self.ops['bias_add'](dest, weight(prefix + f'.attn_{label}.bias'), dest, n, h)
                self.ops['rope2d'](q, pw_buf, ph_buf, q_rot, n, heads, dim, cfg.rope_theta)
                self.ops['rope2d'](k, pw_buf, ph_buf, k_rot, n, heads, dim, cfg.rope_theta)
                self.ops['elementwise'](q_rot, 0, q_rot, n * h, 2, scale)
                for src, dst in ((q_rot, qh), (k_rot, kh), (v, vh)):
                    self.ops['transpose'](src, dst, n, heads, dim, 0)
                # scores[h] = Q[h] K[h]^T, row-major [head, query, key].
                self.blas.sgemm_strided_batched(
                    kh, qh, scores, m=n, n=n, k=dim,
                    lda=dim, ldb=dim, ldc=n, stride_a=n * dim, stride_b=n * dim,
                    stride_c=n * n, batch=heads, trans_a=True, trans_b=False,
                )
                is_global = il == cfg.n_layers - 1 or (il + 1) % cfg.sparse_factor == 0
                self.ops['softmax'](scores, 0 if is_global else mask_buf, heads * n, n, n)
                # context[h] = softmax(scores)[h] V[h], row-major [head, query, channel].
                self.blas.sgemm_strided_batched(
                    vh, scores, context, m=dim, n=n, k=n,
                    lda=dim, ldb=n, ldc=dim, stride_a=n * dim, stride_b=n * n,
                    stride_c=n * dim, batch=heads, trans_a=False, trans_b=False,
                )
                self.ops['transpose'](context, attended, n, heads, dim, 1)
                linear(attended, prefix + '.attn_out.weight', projected)
                self.ops['bias_add'](projected, weight(prefix + '.attn_out.bias'), projected, n, h)
                self.ops['elementwise'](x, projected, x, n * h, 0)
                norm(x, prefix + '.ln2', xn)
                linear(xn, prefix + '.ffn_up.weight', ff, out=inter)
                self.ops['bias_add'](ff, weight(prefix + '.ffn_up.bias'), ff, n, inter)
                self.ops['elementwise'](ff, 0, ff, n * inter, 1)
                linear(ff, prefix + '.ffn_down.weight', projected, width=inter)
                self.ops['bias_add'](projected, weight(prefix + '.ffn_down.bias'), projected, n, h)
                self.ops['elementwise'](x, projected, x, n * h, 0)
                if _debug_inject and f'layer_out-{il}' in _debug_inject:
                    copy_host_array_to_device(DeviceBuffer(x, n * h * 4),
                                              np.ascontiguousarray(_debug_inject[f'layer_out-{il}'], dtype=np.float32))
            norm(x, 'v.post_ln', xn)
            x, xn = xn, x
            self.ops['gather'](x, inv_buf, permuted, n, h)
            x, permuted = permuted, x
            self.ops['shuffle'](x, shuffled, gh, gw, h, f)
            linear(shuffled, 'mm.0.weight', mm_hidden, rows=m, width=h * f * f, out=4096)
            self.ops['elementwise'](mm_hidden, 0, mm_hidden, m * 4096, 1)
            linear(mm_hidden, 'mm.1.weight', mm_hidden2, rows=m, width=4096, out=4096)
            self.ops['elementwise'](mm_hidden2, 0, mm_hidden2, m * 4096, 1)
            linear(mm_hidden2, 'mm.2.weight', output, rows=m, width=4096, out=cfg.projection_dim)
            result = np.empty((m, cfg.projection_dim), dtype=np.float32)
            copy_device_to_host(host_array_ptr(result), DeviceBuffer(output, result.nbytes),
                                result.nbytes)
            if not np.isfinite(result).all():
                raise FloatingPointError('Muse Glimmer vision encoder produced non-finite features')
            return result
        finally:
            for buffer in reversed(buffers):
                free(buffer)


__all__ = ['MuseGlimmerVisionRunner']