// Moonshine FP16 LayerNorm with FP32 mean and variance statistics.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

using half_t = _Float16;

__device__ inline float moonshine_block_sum(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  __shared__ float partial[8];
  if (lane == 0) partial[wave] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int index = 0; index < waves; ++index) total += partial[index];
    partial[0] = total;
  }
  __syncthreads();
  return partial[0];
}

__global__ void moonshine_layernorm_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t hidden_size,
    float epsilon) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t offset = row * hidden_size;
  float sum = 0.0f;
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    sum += static_cast<float>(input[offset + index]);
  }
  const float mean = moonshine_block_sum(sum) / static_cast<float>(hidden_size);
  float squared = 0.0f;
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    const float centered = static_cast<float>(input[offset + index]) - mean;
    squared += centered * centered;
  }
  const float variance =
      moonshine_block_sum(squared) / static_cast<float>(hidden_size);
  const float inverse_std = rsqrtf(variance + epsilon);
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    const float centered = static_cast<float>(input[offset + index]) - mean;
    output[offset + index] = static_cast<half_t>(
        centered * inverse_std * static_cast<float>(weight[index]));
  }
}

}  // namespace

extern "C" int hipengine_moonshine_layernorm_fp16(
    const half_t* input,
    const half_t* weight,
    half_t* output,
    int64_t rows,
    int64_t hidden_size,
    float epsilon,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || !(epsilon > 0.0f) ||
      threads <= 0 || threads > 256 || (threads % 32) != 0) {
    return hipErrorInvalidValue;
  }
  moonshine_layernorm_fp16_kernel<<<
      dim3(rows), dim3(threads), 0, stream>>>(
      input, weight, output, rows, hidden_size, epsilon);
  return hipGetLastError();
}
