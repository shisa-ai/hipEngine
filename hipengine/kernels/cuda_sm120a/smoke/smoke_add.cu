#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void hipengine_cuda_sm120a_smoke_add_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

extern "C" int hipengine_cuda_sm120a_smoke_add_f32(
    const float* a,
    const float* b,
    float* out,
    int64_t n,
    cudaStream_t stream) {
  if (n < 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  constexpr int kBlock = 256;
  int64_t grid_x = (n + kBlock - 1) / kBlock;
  if (grid_x > INT32_MAX) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dim3 grid(static_cast<unsigned int>(grid_x));
  dim3 block(kBlock);
  hipengine_cuda_sm120a_smoke_add_f32_kernel<<<grid, block, 0, stream>>>(
      a, b, out, n);
  return static_cast<int>(cudaGetLastError());
}
