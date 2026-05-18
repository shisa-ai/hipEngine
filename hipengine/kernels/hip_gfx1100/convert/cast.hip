// Torch-free dtype cast helpers for hipEngine runtime glue.

#include <hip/hip_runtime.h>
#include <stdint.h>

using half_t = _Float16;

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

__global__ void f32_to_fp16_kernel(const float* __restrict__ x, half_t* __restrict__ out, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = static_cast<half_t>(x[idx]);
  }
}

__global__ void fp16_to_f32_kernel(const half_t* __restrict__ x, float* __restrict__ out, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = static_cast<float>(x[idx]);
  }
}

__global__ void fp16_to_bf16_kernel(const half_t* __restrict__ x, uint16_t* __restrict__ out, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = float_to_bf16_bits(static_cast<float>(x[idx]));
  }
}

__global__ void fp16_to_bf16_strided_rows_kernel(
    const half_t* __restrict__ x,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t cols,
    int64_t dst_row_stride,
    int64_t dst_col_offset) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = rows * cols;
  if (idx < count) {
    const int64_t row = idx / cols;
    const int64_t col = idx - row * cols;
    out[row * dst_row_stride + dst_col_offset + col] = float_to_bf16_bits(static_cast<float>(x[idx]));
  }
}

bool valid_count(int64_t count) {
  return count > 0;
}

bool valid_strided_rows(int64_t rows, int64_t cols, int64_t dst_row_stride, int64_t dst_col_offset) {
  return rows > 0 && cols > 0 && dst_row_stride >= cols && dst_col_offset >= 0 && dst_col_offset + cols <= dst_row_stride;
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

extern "C" int hipengine_f32_to_fp16(const float* x, half_t* out, int64_t count, hipStream_t stream) {
  if (!valid_count(count)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(f32_to_fp16_kernel, dim3(blocks), dim3(threads), 0, stream, x, out, count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_fp16_to_f32(const half_t* x, float* out, int64_t count, hipStream_t stream) {
  if (!valid_count(count)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(fp16_to_f32_kernel, dim3(blocks), dim3(threads), 0, stream, x, out, count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_fp16_to_bf16(const half_t* x, uint16_t* out, int64_t count, hipStream_t stream) {
  if (!valid_count(count)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(fp16_to_bf16_kernel, dim3(blocks), dim3(threads), 0, stream, x, out, count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_fp16_to_bf16_strided_rows(
    const half_t* x,
    uint16_t* out,
    int64_t rows,
    int64_t cols,
    int64_t dst_row_stride,
    int64_t dst_col_offset,
    hipStream_t stream) {
  if (!valid_strided_rows(rows, cols, dst_row_stride, dst_col_offset)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t count = rows * cols;
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(fp16_to_bf16_strided_rows_kernel, dim3(blocks), dim3(threads), 0, stream, x, out, rows, cols, dst_row_stride, dst_col_offset);
  return static_cast<int>(hipGetLastError());
}
