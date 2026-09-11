"""Cold-path ownership of reusable K-major Q8 MMQ weight buffers."""
from types import MappingProxyType

from hipengine.core.memory import malloc, free
from hipengine.kernels.registry import resolve
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    q8_mmq_prepacked_weight_nbytes,
)


class Q8MMQWeightSidecars:
    def __init__(self, *, runtime, library):
        self.runtime = runtime
        self.library = library
        self._buffers = {}
        self.closed = False

    @property
    def mapping(self):
        return MappingProxyType({
            key: (buf.ptr, buf.nbytes) for key, buf in self._buffers.items()
        })

    @property
    def nbytes(self):
        return sum(buf.nbytes for buf in self._buffers.values())

    def prepare(self, weight):
        if self.closed:
            raise RuntimeError("Q8 MMQ sidecar owner is closed")
        n,k = map(int,weight.spec.source.shape)
        raw = int(weight.allocation("raw").tensor.ptr)
        key = (raw,k,n)
        if key in self._buffers:
            return
        pack = resolve(
            backend=weight.backend,layer="weight_pack",
            quant=weight.spec.quant_key,variant="mmq_kmajor76")
        size = q8_mmq_prepacked_weight_nbytes(k,n)
        buffer = malloc(size,runtime=self.runtime)
        try:
            pack(raw,buffer.ptr,k,n,library=self.library,runtime=self.runtime)
            # Publish only after preparation completes; future requests may use another stream.
            self.runtime.device_synchronize()
        except BaseException:
            free(buffer,runtime=self.runtime)
            raise
        self._buffers[key] = buffer

    def close(self):
        if self.closed:
            return
        for buffer in reversed(tuple(self._buffers.values())):
            free(buffer,runtime=self.runtime)
        self._buffers.clear()
        self.closed = True
