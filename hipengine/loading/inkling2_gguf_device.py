"""Packed projection ownership for the Inkling GGUF decoder."""

from __future__ import annotations

from functools import partial

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.kernels.backends import hip_target_arch_for_backend, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_gemv  # noqa: F401
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.quant.gguf import GGMLQuantizationType as Q


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
        selected_key = KernelKey(backend, "moe_linear", "gguf_nvfp4", "selected_f32_f32_out")
        self._selected_nvfp4 = (resolve(backend=backend, layer=selected_key.layer, quant=selected_key.quant,
                                        variant=selected_key.variant) if is_registered(selected_key) else None)
        self._bank_scales = {}

    def _packed_launch(self, quant):
        if quant not in self._bound_launches:
            launch = self._launches[quant]
            if quant in (Q.Q8_0, Q.Q5_K):
                launch = partial(launch, library=build_gguf_k_gemv(target_arch=self._target_arch), runtime=self._runtime)
            self._bound_launches[quant] = launch
        return self._bound_launches[quant]

    def _upload(self, array):
        host = np.ascontiguousarray(array)
        buf = malloc(host.nbytes, runtime=self._runtime)
        try:
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
                data, packed = self._weights.raw(tensor)[rows], Q.Q8_0
            else:
                data, packed = self._weights.decode(tensor, rows), False
            self._matrices[key] = self._upload(data), (data.shape[0], tensor.shape[1]), packed
        return self._matrices[key]

    def _launch_projection(self, src, dst, rows, ndim, matrix, native_bf16):
        weight, (m, n), packed = matrix
        if packed:
            self._packed_launch(packed)(src, weight.ptr, dst, rows, n, m)
        else:
            super()._launch_projection(src, dst, rows, ndim, matrix, native_bf16)

    def _bank(self, bank):
        weight = bank.weight
        key = weight.name, None, "experts"
        if key not in self._matrices:
            scales = None if bank.scale is None else self._weights.decode(bank.scale)
            if scales is not None and not np.isfinite(scales).all():
                raise ValueError(f"non-finite Inkling GGUF expert scales for {weight.name}")
            self._matrices[key] = self._upload(self._weights.raw(weight)), weight.shape, weight.ggml_type
            self._bank_scales[key] = scales
        return self._matrices[key][0], self._bank_scales[key]

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
        selected = self._selected_nvfp4 if quant == Q.NVFP4 else None
        if selected is not None:
            ids = np.ascontiguousarray(ids, dtype=np.int32)
            with self._lock:
                if self._closed:
                    raise RuntimeError("Inkling GGUF projection residency is closed")
                weight, scales = self._bank(bank)
                return self._project_nvfp4_routes(selected, weight.ptr, scales, ids, hidden, experts, m, n)
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

    def _project_nvfp4_routes(self, launch, weight, scales, ids, hidden, experts, m, n):
        # One transfer for the activations and route metadata, all 4-byte aligned.
        input_values = hidden.size
        routes = len(ids)
        x = np.empty(input_values + routes * (2 if scales is not None else 1), np.float32)
        x[:input_values].reshape(hidden.shape)[...] = hidden
        x[input_values:input_values + routes].view(np.int32)[:] = ids
        if scales is not None:
            x[input_values + routes:] = scales[ids]
        y = np.empty((routes, m), np.float32)
        inp = self._scratch("_input", x.nbytes)
        out = self._scratch("_output", y.nbytes)
        ids_ptr = inp.ptr + input_values * x.itemsize
        scales_ptr = ids_ptr + ids.nbytes if scales is not None else 0
        with self._transfer.exchange(inp.ptr, out.ptr, x, y, pinned=True):
            launch(inp.ptr, weight, out.ptr, ids_ptr, scales_ptr, routes, experts, n, m)
        return y
