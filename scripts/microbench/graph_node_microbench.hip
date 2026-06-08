// M16.1 diagnostic micro-bench: isolate the ROCm HIP graph-replay per-node cost.
//
// A trivial kernel (one block, writes memory so it cannot be elided) is launched
// back-to-back `count` times from a C loop. Issuing the whole burst from C removes
// all Python/ctypes per-node overhead, so the wall time measures the *pure HIP*
// host-issue + GPU-dispatch cost of N serial launches.
//
// The same C burst is captured into ONE HIP graph (fixed args, fixed stream) and
// replayed in steady state; comparing direct-burst vs graph-replay per-node cost
// answers whether HIP graphs collapse the dispatch residual on W7900/gfx1100.
//
// Not a production kernel: no registry entry, no KVLiveSpans ABI. Diagnostic only.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void gmb_noop_kernel(float* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Real (tiny) memory write so the launch cannot be optimized away.
    out[idx] = out[idx] + 1.0f;
  }
}

// Issue `count` back-to-back launches on `stream`. When called inside a stream
// capture, this records `count` kernel nodes with fixed arguments.
extern "C" int gmb_launch_n(float* out, int n, int count, hipStream_t stream) {
  if (n < 0 || count < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = static_cast<unsigned int>((n + kBlock - 1) / kBlock);
  if (grid_x == 0) {
    grid_x = 1;
  }
  for (int i = 0; i < count; ++i) {
    hipLaunchKernelGGL(
        gmb_noop_kernel,
        dim3(grid_x),
        dim3(kBlock),
        0,
        stream,
        out,
        n);
  }
  return static_cast<int>(hipGetLastError());
}
