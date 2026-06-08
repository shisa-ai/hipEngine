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

// Wide-argument variant: same trivial compute, but the kernel takes 16 arguments
// (8 pointers + 8 int64) so hipLaunchKernelGGL must marshal a realistically large
// kernel-arg buffer per launch — matching the hot W4 GEMV / fused launchers, which
// carry ~10-20 args, rather than the 2-arg trivial kernel above. All pointer args
// alias the same buffer; the ints are touched so they cannot be dead-code-elided.
extern "C" __global__ void gmb_wide_kernel(
    float* __restrict__ p0,
    const float* __restrict__ p1,
    const float* __restrict__ p2,
    const float* __restrict__ p3,
    const float* __restrict__ p4,
    const float* __restrict__ p5,
    const float* __restrict__ p6,
    const float* __restrict__ p7,
    int64_t a0,
    int64_t a1,
    int64_t a2,
    int64_t a3,
    int64_t a4,
    int64_t a5,
    int64_t a6,
    int64_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Touch every arg so none is elided (forces real marshaling of all 16).
    float acc = p0[idx] + p1[idx] + p2[idx] + p3[idx] + p4[idx] + p5[idx] + p6[idx] + p7[idx];
    int64_t bits = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6;
    p0[idx] = acc + static_cast<float>(bits & 1);
  }
}

// grid_blocks <= 0 -> auto (ceil(n/256)); otherwise force that many workgroups
// so the kernel's launch/dispatch cost can be measured against grid size while
// keeping the compute trivial (blocks past ceil(n/256) return immediately).
static inline unsigned int gmb_grid(int n, int grid_blocks) {
  if (grid_blocks > 0) {
    return static_cast<unsigned int>(grid_blocks);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = static_cast<unsigned int>((n + kBlock - 1) / kBlock);
  return grid_x == 0 ? 1u : grid_x;
}

extern "C" int gmb_launch_wide_n(float* buf, int n, int count, int grid_blocks, hipStream_t stream) {
  if (n < 0 || count < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = gmb_grid(n, grid_blocks);
  for (int i = 0; i < count; ++i) {
    hipLaunchKernelGGL(
        gmb_wide_kernel,
        dim3(grid_x),
        dim3(kBlock),
        0,
        stream,
        buf, buf, buf, buf, buf, buf, buf, buf,
        static_cast<int64_t>(i + 1),
        static_cast<int64_t>(2),
        static_cast<int64_t>(3),
        static_cast<int64_t>(4),
        static_cast<int64_t>(5),
        static_cast<int64_t>(6),
        static_cast<int64_t>(7),
        static_cast<int64_t>(n));
  }
  return static_cast<int>(hipGetLastError());
}

// Issue `count` back-to-back launches on `stream`. When called inside a stream
// capture, this records `count` kernel nodes with fixed arguments.
extern "C" int gmb_launch_n(float* out, int n, int count, int grid_blocks, hipStream_t stream) {
  if (n < 0 || count < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kBlock = 256;
  unsigned int grid_x = gmb_grid(n, grid_blocks);
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
