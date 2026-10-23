"""Torch-free Gemma 4 image encoder using FP32 gfx11 primitives and rocBLAS."""
from __future__ import annotations

import numpy as np

from hipengine.core.memory import (
    malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr,
)
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.registry import resolve
from hipengine.loading.gguf import GGUFReader


class Gemma4VisionRunner:
    """Resident vision weights with request-local, bounded attention scratch.

    The input is a raster patch matrix in CHW patch order and X/Y coordinates.
    The output contains projected, unscaled language-model embedding rows.
    """

    def __init__(self, reader: GGUFReader, config, *, backend: str):
        from hipengine.kernels.hip_gfx1100.vision.gemma4_vision import register_gemma4_vision_kernels

        if config.head_count_kv != config.heads:
            raise NotImplementedError('Gemma 4 HIP vision requires equal query and KV head counts')
        if config.clipped_linears:
            raise NotImplementedError('Gemma 4 HIP vision does not implement clipped linear tensors')
        register_gemma4_vision_kernels()
        self.reader, self.config, self.backend = reader, config, backend
        self._weights = {}
        self._out_scales = {}
        self._owned = []
        self.closed = False
        self.blas = None
        self.ops = {name: resolve(backend=backend, layer='vision_' + name, quant='fp32', variant='gemma4')
                    for name in ('rmsnorm', 'elementwise', 'heads', 'softmax', 'pool')}
        self.positions = np.asarray(reader.dequantize_tensor('v.position_embd.weight'), dtype=np.float32)
        try:
            for tensor in reader.info.tensors:
                name = tensor.name
                if name == 'v.position_embd.weight' or not name.startswith(('v.', 'mm.input_projection.')):
                    continue
                if name.endswith('.out_scale.weight'):
                    self._out_scales[name] = float(np.asarray(reader.dequantize_tensor(name)).reshape(-1)[0])
                    continue
                array = np.ascontiguousarray(reader.dequantize_tensor(name), dtype=np.float32)
                if name == 'v.patch_embd.weight':
                    array = array.reshape(array.shape[0], -1)
                self._weights[name] = self._upload(array)
            self.blas = Rocblas.load()
        except BaseException:
            self.close()
            raise

    def _upload(self, array):
        value = np.ascontiguousarray(array)
        buffer = malloc(value.nbytes)
        self._owned.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(value), value.nbytes)
        return buffer

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.blas is not None:
            self.blas.close()
        for buffer in reversed(self._owned):
            free(buffer)
        self._owned.clear()
        self._weights.clear()

    def forward(self, patches, positions, grid, *, request=None, capture_layers=None):
        """Encode one image, isolating its attention from every other image."""

        from hipengine.generation.deadline import raise_if_generation_deadline_expired

        if self.closed:
            raise RuntimeError('Gemma 4 vision runner is closed')
        c = self.config
        h, inter, heads, dim = c.hidden_size, c.intermediate_size, c.heads, c.head_dim
        gh, gw = map(int, grid)
        n, m = gh * gw, (gh // c.spatial_merge_size) * (gw // c.spatial_merge_size)
        p = np.ascontiguousarray(patches, dtype=np.float32)
        if np.any((p < 0) | (p > 1)):
            raise ValueError('Gemma 4 vision patches must contain pixels in [0, 1]')
        pos = np.ascontiguousarray(positions, dtype=np.int32)
        if (gh <= 0 or gw <= 0 or gh % c.spatial_merge_size or gw % c.spatial_merge_size
                or p.shape != (n, 3*c.patch_size**2) or pos.shape != (n,2)
                or not np.isfinite(p).all()):
            raise ValueError('invalid Gemma 4 vision patch geometry')
        expected_pos = np.stack(np.meshgrid(np.arange(gw), np.arange(gh)), axis=-1).reshape(n,2)
        if not np.array_equal(pos, expected_pos):
            raise ValueError('Gemma 4 vision positions must follow raster X/Y order')
        if pos.max() >= self.positions.shape[1]:
            raise ValueError('Gemma 4 vision grid exceeds the learned position table')
        if request is not None:
            raise_if_generation_deadline_expired(request)

        buffers = []
        def alloc(shape, dtype=np.float32, value=None):
            buffer = malloc(int(np.prod(shape))*np.dtype(dtype).itemsize)
            buffers.append(buffer)
            if value is not None:
                a = np.ascontiguousarray(value, dtype=dtype)
                copy_host_to_device(buffer, host_array_ptr(a), a.nbytes)
            return buffer.ptr

        def weight(name):
            return self._weights[name].ptr

        def norm(x, name, y, rows=n, width=h):
            self.ops['rmsnorm'](x, 0 if name is None else weight(name), y, rows, width, c.norm_epsilon)

        def element(x, b, y, size, op, lo=0., hi=0.):
            self.ops['elementwise'](x,b,y,size,op,lo,hi)

        def linear(x, name, y, rows=n, width=h, out=h):
            self.blas.sgemm_rowmajor_nt(x,weight(name),y,rows=rows,in_features=width,out_features=out)

        try:
            patch = alloc(p.shape,value=2.0*p-1.0)
            xy = alloc(pos.shape,dtype=np.int32,value=pos)
            position_rows = self.positions[0,pos[:,0]] + self.positions[1,pos[:,1]]
            learned = alloc((n,h),value=position_rows)
            x, normalized, projected = [alloc((n,h)) for _ in range(3)]
            q,k,v,qh,kh,vh,context,attended = [alloc((n,h)) for _ in range(8)]
            gate, up = [alloc((n,inter)) for _ in range(2)]
            scores = alloc((heads,n,n))
            pooled, pooled_norm = [alloc((m,h)) for _ in range(2)]
            output = alloc((m,c.output_size))
            linear(patch,'v.patch_embd.weight',x,width=p.shape[1])
            element(x,learned,x,n*h,0)
            for layer in range(c.block_count):
                if request is not None:
                    raise_if_generation_deadline_expired(request)
                prefix = f'v.blk.{layer}.'
                norm(x,prefix+'ln1.weight',normalized)
                for label, dest, head_dest in (('q',q,qh),('k',k,kh),('v',v,vh)):
                    linear(normalized,prefix+f'attn_{label}.weight',dest)
                    norm(dest,None if label=='v' else prefix+f'attn_{label}_norm.weight',
                         dest,rows=n*heads,width=dim)
                    self.ops['heads'](dest,xy,head_dest,n,heads,dim,c.rope_theta,0 if label=='v' else 1)
                # Row-major QK^T, stored as [head,query,key], without a scale.
                self.blas.sgemm_strided_batched(kh,qh,scores,m=n,n=n,k=dim,
                    lda=dim,ldb=dim,ldc=n,stride_a=n*dim,stride_b=n*dim,stride_c=n*n,
                    batch=heads,trans_a=True,trans_b=False)
                self.ops['softmax'](scores,heads*n,n)
                # Row-major softmax(QK^T) V -> [head,query,channel].
                self.blas.sgemm_strided_batched(vh,scores,context,m=dim,n=n,k=n,
                    lda=dim,ldb=n,ldc=dim,stride_a=n*dim,stride_b=n*n,stride_c=n*dim,
                    batch=heads,trans_a=False,trans_b=False)
                self.ops['heads'](context,0,attended,n,heads,dim,c.rope_theta,2)
                linear(attended,prefix+'attn_out.weight',projected)
                norm(projected,prefix+'attn_post_norm.weight',projected)
                element(x,projected,x,n*h,0)
                norm(x,prefix+'ln2.weight',normalized)
                linear(normalized,prefix+'ffn_gate.weight',gate,out=inter)
                linear(normalized,prefix+'ffn_up.weight',up,out=inter)
                element(gate,0,gate,n*inter,1)
                element(gate,up,gate,n*inter,2)
                linear(gate,prefix+'ffn_down.weight',projected,width=inter)
                norm(projected,prefix+'ffn_post_norm.weight',projected)
                element(x,projected,x,n*h,0)
                out_scale = self._out_scales.get(prefix+'out_scale.weight')
                if out_scale is not None:
                    element(x,0,x,n*h,4,out_scale)
                if capture_layers is not None:
                    from hipengine.core.memory import DeviceBuffer
                    host = np.empty((n,h),dtype=np.float32)
                    copy_device_to_host(host_array_ptr(host),DeviceBuffer(x,host.nbytes),host.nbytes)
                    capture_layers.append(host)
            bias = self._weights.get('v.std_bias')
            scale = self._weights.get('v.std_scale')
            self.ops['pool'](x,0 if bias is None else bias.ptr,0 if scale is None else scale.ptr,
                             pooled,gh,gw,h,c.spatial_merge_size)
            norm(pooled,None,pooled_norm,rows=m)
            linear(pooled_norm,'mm.input_projection.weight',output,rows=m,out=c.output_size)
            from hipengine.core.memory import DeviceBuffer
            result = np.empty((m,c.output_size),dtype=np.float32)
            copy_device_to_host(host_array_ptr(result),DeviceBuffer(output,result.nbytes),result.nbytes)
            if not np.isfinite(result).all():
                raise FloatingPointError('Gemma 4 vision encoder produced non-finite features')
            return result
        finally:
            for buffer in reversed(buffers):
                free(buffer)
