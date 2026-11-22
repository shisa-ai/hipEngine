"""Packed projection ownership for the Inkling GGUF decoder."""

from __future__ import annotations

from functools import partial

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.kernels.backends import hip_target_arch_for_backend, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_gemv, inkling2_experts  # noqa: F401
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.loading.inkling2_device_experts import build_expert_route_tiles
from hipengine.quant.gguf import GGMLQuantizationType as Q

# Do not hand HIP a whole file-backed expert bank. Read each window before
# submitting its transfer, and reuse this host staging across physical banks.
_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_SELECTED_ROUTES = 65535


class Inkling2GGUFDeviceProjections(Inkling2DeviceDense):
    """Share dense transfer scratch; upload physical expert banks only once."""

    def __init__(self, weights, *, runtime, blas, backend):
        super().__init__(weights, runtime=runtime, blas=blas)
        load_backend_kernel_package(backend)
        self._tensors = {t.name: t for t in weights.reader.info.tensors}
        self._target_arch = hip_target_arch_for_backend(backend)
        self._launches = {q: resolve(backend=backend, layer="linear", quant=f"gguf_{q.name.lower()}",
                                    variant="gemv_f32_f32_out") for q in (Q.NVFP4, Q.Q8_0, Q.Q5_K)}
        self._bound_launches = {}
        self._selected_launches = {}
        self._tiled_launches = {}
        self._k_library = None
        prefill = KernelKey(backend, "linear", "gguf_q8_0", "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out")
        self._q8_prefill = (self._bind_packed(Q.Q8_0, resolve(backend=prefill.backend, layer=prefill.layer,
                                                           quant=prefill.quant, variant=prefill.variant))
                            if is_registered(prefill) else None)
        for quant in self._launches:
            for variant, launches in (("selected_f32_f32_out", self._selected_launches),
                                      ("selected_tile4_f32_f32_out", self._tiled_launches)):
                key = KernelKey(backend, "moe_linear", f"gguf_{quant.name.lower()}", variant)
                if is_registered(key):
                    launch = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
                    launches[quant] = self._bind_packed(quant, launch)
        self._mix = None
        key = KernelKey(backend, "inkling2_expert_pointwise", "inkling2_mixed", "ordered_mix_exact")
        if (is_registered(key) and all(Q(layer.down.weight.ggml_type) in self._selected_launches
                                       for layer in weights.mapping.layers)):
            self._mix = partial(resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant),
                                target_arch=self._target_arch)
        self._bank_scales = {}
        self._upload_staging = None

    @property
    def supports_device_mix(self):
        return self._mix is not None

    def _bind_packed(self, quant, launch):
        if quant == Q.NVFP4:
            return launch

        def run(*args, **kwargs):
            if self._k_library is None:
                self._k_library = build_gguf_k_gemv(target_arch=self._target_arch)
            return launch(*args, library=self._k_library, runtime=self._runtime, **kwargs)
        return run

    def _packed_launch(self, quant):
        if quant not in self._bound_launches:
            self._bound_launches[quant] = self._bind_packed(quant, self._launches[quant])
        return self._bound_launches[quant]

    def _upload(self, array):
        host = np.ascontiguousarray(array)
        buf = malloc(host.nbytes, runtime=self._runtime)
        try:
            if isinstance(array, np.memmap) and host.nbytes > _UPLOAD_CHUNK_BYTES:
                if self._upload_staging is None:
                    self._upload_staging = np.empty(_UPLOAD_CHUNK_BYTES, dtype=np.uint8)
                raw = host.view(np.uint8).reshape(-1)
                for begin in range(0, raw.size, _UPLOAD_CHUNK_BYTES):
                    payload = raw[begin:begin + _UPLOAD_CHUNK_BYTES]
                    staging = self._upload_staging[:payload.size]
                    staging[:] = payload
                    self._runtime.memcpy(buf.ptr + begin, staging.ctypes.data, staging.nbytes,
                                         HipMemcpyKind.HOST_TO_DEVICE)
            else:
                self._runtime.memcpy(buf.ptr, host.ctypes.data, host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        except BaseException:
            free(buf, runtime=self._runtime)
            raise
        return buf

    def _matrix(self, name, limit, *, preserve_bf16=False):
        key = name, limit, False
        if key not in self._matrices:
            tensor = self._tensors[name]
            if len(tensor.shape) != 2 or (limit is not None and not 0 < limit <= tensor.shape[0]):
                raise ValueError(f"invalid Inkling GGUF matrix or row limit for {name}")
            rows = slice(limit)
            if tensor.ggml_type == Q.Q8_0:
                data, packed = self._weights.raw(tensor, cache=False)[rows], Q.Q8_0
            else:
                data, packed = self._weights.decode(tensor, rows), False
            self._matrices[key] = self._upload(data), (data.shape[0], tensor.shape[1]), packed
        return self._matrices[key]

    def _launch_projection(self, src, dst, rows, ndim, matrix, native_bf16):
        weight, (m, n), packed = matrix
        if packed:
            launch = self._q8_prefill if packed == Q.Q8_0 and rows > 1 and m % 8 == 0 else None
            (launch or self._packed_launch(packed))(src, weight.ptr, dst, rows, n, m)
        else:
            super()._launch_projection(src, dst, rows, ndim, matrix, native_bf16)

    def _bank(self, bank):
        weight = bank.weight
        key = weight.name, None, "experts"
        if key not in self._matrices:
            scales = None if bank.scale is None else self._weights.decode(bank.scale)
            if scales is not None and not np.isfinite(scales).all():
                raise ValueError(f"non-finite Inkling GGUF expert scales for {weight.name}")
            packed = self._upload(self._weights.raw(weight, cache=False))
            self._matrices[key] = packed, weight.shape, weight.ggml_type
            self._bank_scales[key] = scales
        return self._matrices[key][0], self._bank_scales[key]

    def close(self):
        try:
            super().close()
        finally:
            self._upload_staging = None

    @property
    def packed_bank_bytes(self):
        return sum(self._matrices[key][0].nbytes for key in self._bank_scales if key in self._matrices)

    def project_selected(self, visit, projection, expert_ids, hidden):
        """Project one FP32 row per selected route, returning results in route order."""
        layer = self._weights.mapping.layers[visit.weight_layer]
        bank = {"gate_up": layer.gate_up, "down": layer.down}[projection]
        experts, m, n = bank.weight.shape
        ids = np.asarray(expert_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or np.any(ids < 0) or np.any(ids >= experts)
                or hidden.shape != (len(ids), n)):
            raise ValueError("invalid Inkling GGUF selected projection geometry")
        quant = Q(bank.weight.ggml_type)
        selected = self._selected_launches.get(quant) if len(ids) <= _MAX_SELECTED_ROUTES else None
        if selected is not None:
            ids = np.ascontiguousarray(ids, dtype=np.int32)
            with self._lock:
                if self._closed:
                    raise RuntimeError("Inkling GGUF projection residency is closed")
                weight, scales = self._bank(bank)
                return self._project_routes(quant, selected, weight.ptr, scales, ids, hidden, experts, m, n)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        x = np.ascontiguousarray(hidden[order], dtype=np.float32)
        y = np.empty((len(ids), m), np.float32)
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            weight, scales = self._bank(bank)
            launch = self._packed_launch(quant)
            inp = self._scratch("_input", x.nbytes)
            out = self._scratch("_output", y.nbytes)
            stride = bank.weight.nbytes // experts
            with self._transfer.exchange(inp.ptr, out.ptr, x, y, pinned=True):
                for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                    expert = int(ids[start])
                    scale = {} if quant != Q.NVFP4 or scales is None else {"outer_scale": float(scales[expert])}
                    launch(inp.ptr + int(start) * n * x.itemsize, weight.ptr + expert * stride,
                           out.ptr + int(start) * m * y.itemsize, int(stop - start), n, m, **scale)
            if scales is not None and quant != Q.NVFP4:
                y *= scales[ids, None]
        result = np.empty_like(y)
        result[order] = y
        return result

    def _route_tiles(self, quant, ids):
        if quant not in self._tiled_launches or not 16 <= ids.size <= _MAX_SELECTED_ROUTES:
            return None
        tiles = build_expert_route_tiles(ids, 4)
        return tiles if len(tiles) < ids.size else None

    @staticmethod
    def _stage_routes(ids, hidden, scales, extra=0, tiles=None):
        # Activations, route metadata and any mix coefficients share one upload.
        count = hidden.size
        records = ids if tiles is None else tiles.ravel()
        scales_start = count + records.size
        x = np.empty(scales_start + (ids.size if scales is not None else 0) + extra, np.float32)
        x[:count].reshape(hidden.shape)[...] = hidden
        x[count:scales_start].view(np.int32)[:] = records
        if scales is not None:
            x[scales_start:scales_start + ids.size] = scales[ids]
        return x, records.nbytes

    def _project_routes(self, quant, launch, weight, scales, ids, hidden, experts, m, n):
        tiles = self._route_tiles(quant, ids)
        x, record_bytes = self._stage_routes(ids, hidden, scales, tiles=tiles)
        y = np.empty((ids.size, m), np.float32)
        inp = self._scratch("_input", x.nbytes)
        out = self._scratch("_output", y.nbytes)
        ids_ptr = inp.ptr + hidden.size * x.itemsize
        scales_ptr = ids_ptr + record_bytes if scales is not None else 0
        with self._transfer.exchange(inp.ptr, out.ptr, x, y, pinned=True):
            if tiles is None:
                launch(inp.ptr, weight, out.ptr, ids_ptr, scales_ptr, ids.size, experts, n, m)
            else:
                self._tiled_launches[quant](inp.ptr, weight, out.ptr, ids_ptr, scales_ptr, len(tiles), ids.size,
                                            experts, n, m)
        return y

    def project_down_mixed(self, visit, ids, mix, activated, *, transform_output=None):
        """Keep the selected down projection, ordered mix and optional consumer on device."""
        bank = self._weights.mapping.layers[visit.weight_layer].down
        experts, m, n = bank.weight.shape
        if (ids.ndim not in (1, 2) or not ids.size or not 1 <= ids.shape[-1] <= _MAX_SELECTED_ROUTES
                or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= experts)
                or mix.shape != ids.shape or activated.shape != (ids.size, n)):
            raise ValueError("invalid Inkling GGUF mixed down projection geometry")
        k = ids.shape[-1]
        flat_ids = np.ascontiguousarray(ids.ravel(), dtype=np.int32)
        tokens = ids.size // k
        tile = _MAX_SELECTED_ROUTES // k
        result = np.empty((*ids.shape[:-1], m), dtype=np.float32)
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling GGUF projection residency is closed")
            weight, scales = self._bank(bank)
            quant = Q(bank.weight.ggml_type)
            launch = self._selected_launches[quant]
            tiles = self._route_tiles(quant, flat_ids)
            x, record_bytes = self._stage_routes(flat_ids, activated, scales, extra=ids.size, tiles=tiles)
            x[-ids.size:].reshape(ids.shape)[...] = mix
            inp = self._scratch("_input", x.nbytes)
            itemsize = x.itemsize
            out = self._scratch("_output", result.nbytes + min(tile, tokens) * k * m * itemsize)
            route_out = out.ptr + result.nbytes
            ids_ptr = inp.ptr + activated.size * itemsize
            scales_ptr = ids_ptr + record_bytes if scales is not None else 0
            mix_ptr = inp.ptr + x.nbytes - flat_ids.nbytes
            with self._transfer.exchange(inp.ptr, out.ptr, x, result, pinned=True):
                for row in range(0, tokens, tile):
                    count = min(tile, tokens - row)
                    offset = row * k * itemsize
                    if tiles is None:
                        launch(inp.ptr + offset * n, weight.ptr, route_out, ids_ptr + offset,
                               scales_ptr + offset if scales_ptr else 0, count * k, experts, n, m)
                    else:
                        # The tile builder admits only route lists that fit one selected launch.
                        self._tiled_launches[quant](inp.ptr, weight.ptr, route_out, ids_ptr, scales_ptr, len(tiles),
                                                   flat_ids.size, experts, n, m)
                    self._mix(route_out, mix_ptr + offset, out.ptr + row * m * itemsize, k, m, tokens=count)
                if transform_output is not None:
                    transform_output(out.ptr, result.shape)
        return result
