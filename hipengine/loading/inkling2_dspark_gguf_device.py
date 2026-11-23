"""Resident packed projections for the Inkling DSpark GGUF drafter."""

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.kernels.backends import hip_target_arch_for_backend, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
from hipengine.kernels.registry import resolve
from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense
from hipengine.quant.gguf import GGMLQuantizationType as Q


class Inkling2DSparkDeviceProjections(Inkling2DeviceDense):
    """Own packed Q8_0 draft matrices and the recurrent projection scratch."""

    def __init__(self, weights, *, runtime, blas, backend):
        super().__init__(weights, runtime=runtime, blas=blas)
        load_backend_kernel_package(backend)
        self._arch = hip_target_arch_for_backend(backend)
        self._q8 = resolve(backend=backend, layer="linear", quant="gguf_q8_0", variant="gemv_f32_f32_out")
        self._library = None

    def _matrix(self, name, limit, *, preserve_bf16=False):
        tensor = self._weights.names[name]
        if len(tensor.shape) != 2 or (limit is not None and not 0 < limit <= tensor.shape[0]):
            raise ValueError(f"invalid Inkling DSpark matrix or row limit for {name}")
        if tensor.ggml_type != Q.Q8_0:
            return super()._matrix(name, limit, preserve_bf16=preserve_bf16)
        key = name, limit, False
        if key not in self._matrices:
            # Fault the file-backed weights in before submitting the device copy.
            data = np.array(self._weights.raw(name)[:limit], copy=True, order="C")
            buf = malloc(data.nbytes, runtime=self._runtime)
            try:
                self._runtime.memcpy(buf.ptr, data.ctypes.data, data.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            except BaseException:
                free(buf, runtime=self._runtime)
                raise
            self._matrices[key] = buf, (len(data), tensor.shape[1]), Q.Q8_0
        return self._matrices[key]

    def _launch_projection(self, src, dst, rows, ndim, matrix, native_bf16):
        weight, (m, n), packed = matrix
        if packed == Q.Q8_0:
            if self._library is None:
                self._library = build_gguf_k_gemv(target_arch=self._arch)
            self._q8(src, weight.ptr, dst, rows, n, m, library=self._library, runtime=self._runtime)
        else:
            super()._launch_projection(src, dst, rows, ndim, matrix, native_bf16)
