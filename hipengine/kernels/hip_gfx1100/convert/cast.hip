// Torch-free dtype cast helpers for HIPENGINE runtime glue.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ out, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = float_to_bf16_bits(x[idx]);
  }
}

__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ out, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = bf16_bits_to_float(x[idx]);
  }
}

bool valid_count(int64_t count) {
  return count > 0;
}

}  // namespace

extern "C" int hipengine_f32_to_bf16(const float* x, uint16_t* out, int64_t count, hipStream_t stream) {
  if (!valid_count(count)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(f32_to_bf16_kernel, dim3(blocks), dim3(threads), 0, stream, x, out, count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_bf16_to_f32(const uint16_t* x, float* out, int64_t count, hipStream_t stream) {
  if (!valid_count(count)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(bf16_to_f32_kernel, dim3(blocks), dim3(threads), 0, stream, x, out, count);
  return static_cast<int>(hipGetLastError());
}
