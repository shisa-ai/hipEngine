// Persistent-barrier microbench (task #105, Phase 2 step 1).
//
// Decides whether a persistent kernel with IN-KERNEL grid-wide barriers can
// beat N separate HIP launches for the same memory-bound work -- the gate for
// the whole persistent-forward (3-5x) program.  M12.1/M13.D already showed HIP
// graph replay (hipGraphLaunch per node) ~= direct dispatch; this asks the
// different question: does cooperative_groups::this_grid().sync() cost less
// than a kernel-launch dispatch boundary?
//
// Each "stage" is a memory-bound grid-stride pass over a working buffer with a
// read-modify-write dependency on the previous stage (so stages cannot fuse and
// the barrier ordering is load-bearing, mimicking layer-to-layer dependence).
// Both variants do identical total work; only the stage boundary differs
// (N launches vs N in-kernel barriers).

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

// One memory-bound pass: out[i] = in[i]*a + b, grid-stride over n.
__global__ void hipengine_pbm_stage_kernel(
    const float* __restrict__ in, float* __restrict__ out, int64_t n, float a, float b) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = in[i] * a + b;
  }
}

// Persistent: n_stages memory-bound passes over buf with a grid.sync() between.
// Read-modify-write in place so each stage depends on the previous one.
__global__ void hipengine_pbm_persistent_kernel(
    float* __restrict__ buf, int64_t n, int n_stages, float a, float b) {
  cg::grid_group grid = cg::this_grid();
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int s = 0; s < n_stages; ++s) {
    for (int64_t i = base; i < n; i += stride) {
      buf[i] = buf[i] * a + b;
    }
    grid.sync();
  }
}

// AR-faithful: each stage reads a DISTINCT cache-cold slice of a big buffer
// (slice_n floats, advancing by slice_n per stage, wrapping), modelling
// weight-streaming where every layer reads fresh HBM (no reuse). buf has
// total_n floats; slice_n is one stage's working set.
__global__ void hipengine_pbm_persistent_distinct_kernel(
    float* __restrict__ buf, int64_t total_n, int64_t slice_n, int n_stages, float a, float b) {
  cg::grid_group grid = cg::this_grid();
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t slices = total_n / slice_n;
  for (int s = 0; s < n_stages; ++s) {
    int64_t off = (static_cast<int64_t>(s) % slices) * slice_n;
    for (int64_t i = base; i < slice_n; i += stride) {
      buf[off + i] = buf[off + i] * a + b;
    }
    grid.sync();
  }
}

__global__ void hipengine_pbm_stage_distinct_kernel(
    float* __restrict__ buf, int64_t off, int64_t slice_n, float a, float b) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < slice_n; i += stride) {
    buf[off + i] = buf[off + i] * a + b;
  }
}

static int64_t _ceil_div(int64_t x, int64_t y) { return (x + y - 1) / y; }

// N-launch baseline: n_stages separate launches of the stage kernel (in place
// via aliased in==out), timed with HIP events.  grid_x is the caller-chosen
// grid (e.g. occupancy-filling, matching the persistent grid for fairness).
extern "C" int hipengine_pbm_run_nlaunch(
    float* buf, int64_t n, int n_stages, int block, int grid_x,
    float a, float b, float* out_ms, hipStream_t stream) {
  hipEvent_t t0, t1;
  if (hipEventCreate(&t0) != hipSuccess || hipEventCreate(&t1) != hipSuccess)
    return static_cast<int>(hipErrorInvalidValue);
  hipEventRecord(t0, stream);
  for (int s = 0; s < n_stages; ++s) {
    hipLaunchKernelGGL(hipengine_pbm_stage_kernel,
                       dim3(static_cast<unsigned>(grid_x)), dim3(static_cast<unsigned>(block)),
                       0, stream, buf, buf, n, a, b);
  }
  hipEventRecord(t1, stream);
  hipEventSynchronize(t1);
  float ms = 0.f; hipEventElapsedTime(&ms, t0, t1);
  if (out_ms) *out_ms = ms;
  hipError_t err = hipGetLastError();
  hipEventDestroy(t0); hipEventDestroy(t1);
  return static_cast<int>(err);
}

// Persistent variant: one cooperative launch. grid_x must be <= the max
// occupancy-resident blocks for the kernel (computed by the wrapper below).
extern "C" int hipengine_pbm_run_persistent(
    float* buf, int64_t n, int n_stages, int block, int grid_x,
    float a, float b, float* out_ms, hipStream_t stream) {
  hipEvent_t t0, t1;
  if (hipEventCreate(&t0) != hipSuccess || hipEventCreate(&t1) != hipSuccess)
    return static_cast<int>(hipErrorInvalidValue);
  void* args[] = {(void*)&buf, (void*)&n, (void*)&n_stages, (void*)&a, (void*)&b};
  hipEventRecord(t0, stream);
  hipError_t lerr = hipLaunchCooperativeKernel(
      reinterpret_cast<const void*>(&hipengine_pbm_persistent_kernel),
      dim3(static_cast<unsigned>(grid_x)), dim3(static_cast<unsigned>(block)),
      args, 0, stream);
  hipEventRecord(t1, stream);
  hipEventSynchronize(t1);
  float ms = 0.f; hipEventElapsedTime(&ms, t0, t1);
  if (out_ms) *out_ms = ms;
  if (lerr != hipSuccess) { hipEventDestroy(t0); hipEventDestroy(t1); return static_cast<int>(lerr); }
  hipError_t err = hipGetLastError();
  hipEventDestroy(t0); hipEventDestroy(t1);
  return static_cast<int>(err);
}

// AR-faithful distinct-slice timing: big buffer total_n floats, each stage a
// fresh slice_n slice. N-launch vs persistent, both HBM-bound (no cache reuse).
extern "C" int hipengine_pbm_run_distinct(
    float* buf, int64_t total_n, int64_t slice_n, int n_stages, int block, int grid_x,
    float a, float b, int persistent, float* out_ms, hipStream_t stream) {
  hipEvent_t t0, t1;
  if (hipEventCreate(&t0) != hipSuccess || hipEventCreate(&t1) != hipSuccess)
    return static_cast<int>(hipErrorInvalidValue);
  int64_t slices = total_n / slice_n;
  hipEventRecord(t0, stream);
  hipError_t lerr = hipSuccess;
  if (persistent) {
    void* args[] = {(void*)&buf, (void*)&total_n, (void*)&slice_n, (void*)&n_stages, (void*)&a, (void*)&b};
    lerr = hipLaunchCooperativeKernel(
        reinterpret_cast<const void*>(&hipengine_pbm_persistent_distinct_kernel),
        dim3(static_cast<unsigned>(grid_x)), dim3(static_cast<unsigned>(block)), args, 0, stream);
  } else {
    for (int s = 0; s < n_stages; ++s) {
      int64_t off = (static_cast<int64_t>(s) % slices) * slice_n;
      hipLaunchKernelGGL(hipengine_pbm_stage_distinct_kernel,
                         dim3(static_cast<unsigned>(grid_x)), dim3(static_cast<unsigned>(block)),
                         0, stream, buf, off, slice_n, a, b);
    }
  }
  hipEventRecord(t1, stream);
  hipEventSynchronize(t1);
  float ms = 0.f; hipEventElapsedTime(&ms, t0, t1);
  if (out_ms) *out_ms = ms;
  if (lerr != hipSuccess) { hipEventDestroy(t0); hipEventDestroy(t1); return static_cast<int>(lerr); }
  hipError_t err = hipGetLastError();
  hipEventDestroy(t0); hipEventDestroy(t1);
  return static_cast<int>(err);
}

// Report the max occupancy-resident grid (blocks) for the persistent kernel at
// a given block size, and the device CU count -- the cooperative launch ceiling.
extern "C" int hipengine_pbm_max_grid(int block, int* out_grid, int* out_cus) {
  int dev = 0; hipGetDevice(&dev);
  hipDeviceProp_t prop; hipGetDeviceProperties(&prop, dev);
  int blocks_per_cu = 0;
  hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_cu, reinterpret_cast<const void*>(&hipengine_pbm_persistent_kernel), block, 0);
  if (out_cus) *out_cus = prop.multiProcessorCount;
  if (out_grid) *out_grid = blocks_per_cu * prop.multiProcessorCount;
  return static_cast<int>(err);
}
