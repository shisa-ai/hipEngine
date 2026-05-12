#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void hipengine_smoke_add_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

extern "C" int hipengine_smoke_add_f32(
    const float* a,
    const float* b,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kBlock = 256;
  int64_t grid_x = (n + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(
      hipengine_smoke_add_f32_kernel,
      dim3(static_cast<unsigned int>(grid_x)),
      dim3(kBlock),
      0,
      stream,
      a,
      b,
      out,
      n);
  return static_cast<int>(hipGetLastError());
}
