// Cold-streaming DRAM read control for the Gemma 4 T16 gate/up probe.
//
// This exists so the DRAM read counter used on the expert leaf can be checked
// against a byte count that is known by construction: the kernel reads a
// contiguous buffer exactly once, 16 bytes per thread per grid-stride step, and
// writes nothing that survives. The expected `GL2C_EA_RDREQ_DRAM` traffic is
// therefore the buffer size and nothing else.
//
// It doubles as the same-host, same-session achievable-read measurement, so the
// leaf's measured rate is compared against a rate this part was observed to
// deliver in this run rather than against a figure carried in from elsewhere.
//
// Diagnostic only. It is not registered in the kernel registry and no production
// path reaches it.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef int int4_t __attribute__((ext_vector_type(4)));

// One pass over `n4` int4 elements. The NaN test keeps the loads live without
// adding a write stream: finite inputs never satisfy it, so `sink` is touched
// zero times in practice.
__global__ void hipengine_g4_dram_stream_read_kernel(
    const int4_t* __restrict__ src,
    float* __restrict__ sink,
    int64_t n4) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  float acc = 0.0f;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4;
       i += stride) {
    const int4_t v = src[i];
    acc += static_cast<float>(v.x) + static_cast<float>(v.y) +
        static_cast<float>(v.z) + static_cast<float>(v.w);
  }
  if (acc != acc) {
    sink[0] = acc;
  }
}

}  // namespace

extern "C" int hipengine_g4_dram_stream_read(
    const void* src,
    void* sink,
    int64_t n4,
    int blocks,
    int threads,
    hipStream_t stream) {
  if (src == nullptr || sink == nullptr || n4 <= 0 || blocks <= 0 || threads <= 0) {
    return 1;
  }
  hipengine_g4_dram_stream_read_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const int4_t*>(src),
      reinterpret_cast<float*>(sink),
      n4);
  return 0;
}
