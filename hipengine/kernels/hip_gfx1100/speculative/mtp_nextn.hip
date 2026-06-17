// Native MTP NextN draft-head kernels (GGUF path).
//
// Correctness-first, size-agnostic F32 kernels for the Qwen3.6 GGUF MTP NextN
// draft layer.  These mirror the ``cpu_reference`` numpy oracle math exactly so
// the M3 fixture gate (KL<=0.05 / top-1>=90%) can run on a real GPU backend
// instead of the registry's cpu_reference fallback.  M6 swaps the hot inner
// GEMVs for WMMA / K-quant tuned kernels on real shapes; these kernels remain
// the correctness baseline.
//
// eh_proj = enorm(embed) + hnorm(hidden) -> concat -> eh_proj linear.
//   out[row, j] = sum_k e_norm[row,k]*weight[j, k] + h_norm[row,k]*weight[j, k+hidden]
//   where weight is row-major [hidden, 2*hidden] (cpu_reference: fused @ weight.T).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ inline float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) {
      total += shared[i];
    }
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

}  // namespace

// RMSNorm F32, one block per row.  out[row,:] = x[row,:] * rsqrt(mean(x^2)+eps) * weight.
extern "C" __global__ void hipengine_mtp_rmsnorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  extern __shared__ float shared[];
  const float* xr = x + row * hidden;
  float* outr = out + row * hidden;
  float sumsq = 0.0f;
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < hidden;
       i += static_cast<int64_t>(blockDim.x)) {
    const float v = xr[i];
    sumsq += v * v;
  }
  sumsq = block_reduce_sum(sumsq, shared);
  const float inv = rsqrtf(sumsq / static_cast<float>(hidden) + eps);
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < hidden;
       i += static_cast<int64_t>(blockDim.x)) {
    outr[i] = xr[i] * inv * weight[i];
  }
}

extern "C" int hipengine_mtp_rmsnorm_f32(
    const float* x,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kWarpSize = 64;
  int block = 256;
  if (hidden < block) {
    block = static_cast<int>(hidden);
  }
  const int warps = (block + kWarpSize - 1) / kWarpSize;
  const int shbytes = (warps + 1) * sizeof(float);
  hipLaunchKernelGGL(
      hipengine_mtp_rmsnorm_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(block)),
      shbytes,
      stream,
      x, weight, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

// eh_proj F32 GEMV, one block per row, threads stride over output features j.
// out[row, j] = sum_{k=0}^{hidden-1} e_norm[row,k]*weight[j*2hidden + k]
//             + sum_{k=0}^{hidden-1} h_norm[row,k]*weight[j*2hidden + hidden + k]
extern "C" __global__ void hipengine_mtp_eh_proj_f32_kernel(
    const float* __restrict__ e_norm,
    const float* __restrict__ h_norm,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float* er = e_norm + row * hidden;
  const float* hr = h_norm + row * hidden;
  float* outr = out + row * hidden;
  const int64_t two_h = hidden * 2;
  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden;
       j += static_cast<int64_t>(blockDim.x)) {
    const float* w = weight + j * two_h;
    float acc = 0.0f;
    for (int64_t k = 0; k < hidden; ++k) {
      acc += er[k] * w[k];
    }
    for (int64_t k = 0; k < hidden; ++k) {
      acc += hr[k] * w[hidden + k];
    }
    outr[j] = acc;
  }
}

extern "C" int hipengine_mtp_eh_proj_f32(
    const float* e_norm,
    const float* h_norm,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  hipLaunchKernelGGL(
      hipengine_mtp_eh_proj_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(block),
      0,
      stream,
      e_norm, h_norm, weight, out, rows, hidden);
  return static_cast<int>(hipGetLastError());
}
