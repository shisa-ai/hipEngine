// Moonshine FP16 LayerNorm with FP32 mean and variance statistics.
//
// CUDA ``sm_120a`` port of the correctness-qualified HIP reference in
// ``hip_gfx1100/norm/moonshine_layernorm.hip``.  The reduction is the same
// ordered FP32 warp-butterfly + cross-warp shared reduction, so numerical
// behavior matches the retained HIP oracle within the established tolerance.
// LayerNorm is bias-free: the FP16 weight is a per-dimension scale.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;

__device__ inline float moonshine_block_sum(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
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

__global__ void moonshine_residual_layernorm_fp16_kernel(
    const half_t* __restrict__ residual,
    const half_t* __restrict__ update,
    const half_t* __restrict__ weight,
    half_t* __restrict__ residual_output,
    half_t* __restrict__ norm_output,
    int64_t rows,
    int64_t hidden_size,
    float epsilon) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t offset = row * hidden_size;
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    residual_output[offset + index] = static_cast<half_t>(
        static_cast<float>(residual[offset + index]) +
        static_cast<float>(update[offset + index]));
  }
  __syncthreads();
  float sum = 0.0f;
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    sum += static_cast<float>(residual_output[offset + index]);
  }
  const float mean = moonshine_block_sum(sum) / static_cast<float>(hidden_size);
  float squared = 0.0f;
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    const float centered =
        static_cast<float>(residual_output[offset + index]) - mean;
    squared += centered * centered;
  }
  const float variance =
      moonshine_block_sum(squared) / static_cast<float>(hidden_size);
  const float inverse_std = rsqrtf(variance + epsilon);
  for (int64_t index = threadIdx.x; index < hidden_size;
       index += blockDim.x) {
    const float centered =
        static_cast<float>(residual_output[offset + index]) - mean;
    norm_output[offset + index] = static_cast<half_t>(
        centered * inverse_std * static_cast<float>(weight[index]));
  }
}

bool valid_layernorm_contract(
    int64_t rows,
    int64_t hidden_size,
    float epsilon,
    int64_t threads) {
  return rows > 0 && hidden_size > 0 && epsilon > 0.0f && threads > 0 &&
      threads <= 256 && (threads % 32) == 0;
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_layernorm_fp16(
    const half_t* input,
    const half_t* weight,
    half_t* output,
    int64_t rows,
    int64_t hidden_size,
    float epsilon,
    int64_t threads,
    cudaStream_t stream) {
  if (!valid_layernorm_contract(rows, hidden_size, epsilon, threads)) {
    return cudaErrorInvalidValue;
  }
  moonshine_layernorm_fp16_kernel<<<
      dim3(rows), dim3(threads), 0, stream>>>(
      input, weight, output, rows, hidden_size, epsilon);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16(
    const half_t* residual,
    const half_t* update,
    const half_t* weight,
    half_t* residual_output,
    half_t* norm_output,
    int64_t rows,
    int64_t hidden_size,
    float epsilon,
    int64_t threads,
    cudaStream_t stream) {
  if (!valid_layernorm_contract(rows, hidden_size, epsilon, threads)) {
    return cudaErrorInvalidValue;
  }
  moonshine_residual_layernorm_fp16_kernel<<<
      dim3(rows), dim3(threads), 0, stream>>>(
      residual,
      update,
      weight,
      residual_output,
      norm_output,
      rows,
      hidden_size,
      epsilon);
  return cudaGetLastError();
}
